//===- unittests/Support/TimeProfilerTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTMutationListener.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"
#include <tuple>

using namespace clang;
using namespace llvm;

namespace {

// Should be called before testing.
void setupProfiler() {
  timeTraceProfilerInitialize(/*TimeTraceGranularity=*/0, "test",
                              /*TimeTraceVerbose=*/true);
}

// Should be called after `compileFromString()`.
// Returns profiler's JSON dump.
std::string teardownProfiler() {
  SmallVector<char, 1024> SmallVec;
  raw_svector_ostream OS(SmallVec);
  timeTraceProfilerWrite(OS);
  timeTraceProfilerCleanup();
  return OS.str().str();
}

class TestASTConsumer : public ASTConsumer {
public:
  TestASTConsumer(ASTMutationListener *MutationListener)
      : MutationListener(MutationListener) {}

  ASTMutationListener *GetASTMutationListener() override {
    return MutationListener;
  }

private:
  ASTMutationListener *MutationListener;
};

class TestFrontendAction : public ASTFrontendAction {
public:
  TestFrontendAction(ASTMutationListener *MutationListener)
      : MutationListener(MutationListener) {}

private:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<TestASTConsumer>(MutationListener);
  }

  ASTMutationListener *MutationListener;
};

// Returns true if code compiles successfully.
// We only parse AST here. This is enough for constexpr evaluation.
bool compileFromString(StringRef Code, StringRef Standard, StringRef File,
                       llvm::StringMap<std::string> Headers = {},
                       ASTMutationListener *MutationListener = nullptr) {

  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FS->addFile(File, 0, MemoryBuffer::getMemBuffer(Code));
  for (const auto &Header : Headers) {
    FS->addFile(Header.getKey(), 0,
                MemoryBuffer::getMemBuffer(Header.getValue()));
  }

  auto Invocation = std::make_shared<CompilerInvocation>();
  std::vector<const char *> Args = {Standard.data(), File.data()};
  DiagnosticOptions InvocationDiagOpts;
  auto InvocationDiags =
      CompilerInstance::createDiagnostics(*FS, InvocationDiagOpts);
  CompilerInvocation::CreateFromArgs(*Invocation, Args, *InvocationDiags);

  CompilerInstance Compiler(std::move(Invocation));
  Compiler.setVirtualFileSystem(std::move(FS));
  Compiler.createDiagnostics();
  Compiler.createFileManager();

  TestFrontendAction Action(MutationListener);
  return Compiler.ExecuteAction(Action);
}

bool compileFromArgs(ArrayRef<const char *> Args, FrontendAction &Action) {
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = llvm::vfs::getRealFileSystem();
  auto Invocation = std::make_shared<CompilerInvocation>();
  DiagnosticOptions InvocationDiagOpts;
  auto InvocationDiags =
      CompilerInstance::createDiagnostics(*FS, InvocationDiagOpts);
  if (!CompilerInvocation::CreateFromArgs(*Invocation, Args, *InvocationDiags))
    return false;

  CompilerInstance Compiler(std::move(Invocation));
  Compiler.setVirtualFileSystem(std::move(FS));
  Compiler.createDiagnostics();
  Compiler.createFileManager();

  return Compiler.ExecuteAction(Action);
}

struct SpecializationCounts {
  unsigned ClassTemplateSpecializations = 0;
  unsigned FunctionTemplateSpecializations = 0;
  unsigned VarTemplateSpecializations = 0;
};

class SpecializationCountingListener : public ASTMutationListener {
public:
  SpecializationCountingListener(SpecializationCounts &Counts)
      : Counts(Counts) {}

  void AddedCXXTemplateSpecialization(
      const ClassTemplateDecl *TD,
      const ClassTemplateSpecializationDecl *D) override {
    ++Counts.ClassTemplateSpecializations;
  }

  void AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                      const FunctionDecl *D) override {
    ++Counts.FunctionTemplateSpecializations;
  }

  void AddedCXXTemplateSpecialization(
      const VarTemplateDecl *TD,
      const VarTemplateSpecializationDecl *D) override {
    ++Counts.VarTemplateSpecializations;
  }

private:
  SpecializationCounts &Counts;
};

std::string GetMetadata(json::Object *Event) {
  std::string M;
  llvm::raw_string_ostream OS(M);
  if (json::Object *Args = Event->getObject("args")) {
    if (auto Detail = Args->getString("detail"))
      OS << Detail;
    // Use only filename to not include os-specific path separators.
    if (auto File = Args->getString("file"))
      OS << (M.empty() ? "" : ", ") << llvm::sys::path::filename(*File);
    if (auto Line = Args->getInteger("line"))
      OS << ":" << *Line;
  }
  return M;
}

// Returns pretty-printed trace graph.
std::string buildTraceGraph(StringRef Json, StringRef ExpectedGraph) {
  struct EventRecord {
    int64_t TimestampBegin;
    int64_t TimestampEnd;
    std::string Name;
    std::string Metadata;
    bool IsInstant;
  };
  std::vector<EventRecord> Events;

  // Parse `EventRecord`s from JSON dump.
  Expected<json::Value> Root = json::parse(Json);
  if (!Root)
    return "";
  for (json::Value &TraceEventValue :
       *Root->getAsObject()->getArray("traceEvents")) {
    json::Object *TraceEventObj = TraceEventValue.getAsObject();

    int64_t TimestampBegin = TraceEventObj->getInteger("ts").value_or(0);
    int64_t TimestampEnd =
        TimestampBegin + TraceEventObj->getInteger("dur").value_or(0);
    std::string Name = TraceEventObj->getString("name").value_or("").str();
    std::string Metadata = GetMetadata(TraceEventObj);
    bool IsInstant = TraceEventObj->getString("ph") == "i";

    // Source events are asynchronous events and may not perfectly nest the
    // synchronous events. Skip testing them.
    if (Name == "Source")
      continue;

    // This is a "summary" event, like "Total PerformPendingInstantiations",
    // skip it
    if (TimestampBegin == 0)
      continue;

    Events.emplace_back(
        EventRecord{TimestampBegin, TimestampEnd, Name, Metadata, IsInstant});
  }

  // Parse the expected graph: each line is indented one "| " per level.
  struct ExpectedNode {
    unsigned Depth;
    std::string Label;
    int Parent;
    SmallVector<int, 4> Children;
  };
  std::vector<ExpectedNode> Nodes;
  SmallVector<int, 8> LastAtDepth;
  SmallVector<StringRef, 32> Lines;
  ExpectedGraph.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Line : Lines) {
    unsigned Depth = 0;
    while (Line.consume_front("| "))
      ++Depth;
    if (Depth > LastAtDepth.size())
      return "expected graph skips a level";
    LastAtDepth.resize(Depth);
    int Parent = Depth ? LastAtDepth.back() : -1;
    int Index = Nodes.size();
    LastAtDepth.push_back(Index);
    if (Parent >= 0)
      Nodes[Parent].Children.push_back(Index);
    Nodes.push_back({Depth, Line.str(), Parent, {}});
  }

  // Timestamps are truncated to microseconds, so two sibling events that fit
  // in the same microsecond look nested and the graph cannot be rebuilt from
  // the timestamps alone. The order of events in the trace is exact, though:
  // an event is written when it ends, and instant events are written right
  // after their parent. Check the expected graph's end order against it, then
  // that each parent contains its children and that siblings do not overlap,
  // and print the graph with any violation marked.
  StringSet<> InstantLabels;
  auto Label = [](const EventRecord &E) {
    return E.Metadata.empty() ? E.Name : E.Name + " (" + E.Metadata + ")";
  };
  for (const EventRecord &Event : Events)
    if (Event.IsInstant)
      InstantLabels.insert(Label(Event));
  auto IsInstant = [&](int N) {
    return InstantLabels.contains(Nodes[N].Label);
  };

  std::vector<int> EndOrder;
  auto Visit = [&](auto &Self, int N) -> void {
    for (int C : Nodes[N].Children)
      if (!IsInstant(C))
        Self(Self, C);
    EndOrder.push_back(N);
    for (int C : Nodes[N].Children)
      if (IsInstant(C))
        EndOrder.push_back(C);
  };
  for (auto [N, Node] : enumerate(Nodes))
    if (Node.Parent < 0)
      Visit(Visit, N);

  auto Labels = [](auto &&Range, auto Label) {
    return join(map_range(Range, Label), "\n");
  };
  std::string ExpectedOrder =
      Labels(EndOrder, [&](int N) -> StringRef { return Nodes[N].Label; });
  std::string ActualOrder = Labels(Events, Label);
  if (ExpectedOrder != ActualOrder)
    return "\n(events in end order)\n" + ActualOrder + "\n";

  std::vector<const EventRecord *> EventOf(Nodes.size());
  for (auto [I, N] : enumerate(EndOrder))
    EventOf[N] = &Events[I];

  std::vector<std::string> Marks(Nodes.size());
  for (auto [N, Node] : enumerate(Nodes)) {
    if (Node.Parent < 0)
      continue;
    const EventRecord &Child = *EventOf[N];
    const EventRecord &Parent = *EventOf[Node.Parent];
    if (Child.TimestampBegin < Parent.TimestampBegin)
      Marks[N] += " [starts before parent]";
    if (Child.TimestampEnd > Parent.TimestampEnd)
      Marks[N] += " [ends after parent]";
  }
  for (const ExpectedNode &Node : Nodes)
    for (auto [S, T] : zip(Node.Children, drop_begin(Node.Children)))
      if (EventOf[S]->TimestampEnd > EventOf[T]->TimestampBegin)
        Marks[S] += " [overlaps next sibling]";

  std::string Graph = "\n";
  for (auto [N, Node] : enumerate(Nodes)) {
    for (unsigned I = 0; I < Node.Depth; ++I)
      Graph += "| ";
    Graph += Node.Label + Marks[N] + "\n";
  }
  return Graph;
}

} // namespace

TEST(TimeProfilerTest, ConstantEvaluationCxx20) {
  std::string Code = R"(
void print(double value);

namespace slow_namespace {

consteval double slow_func() {
    double d = 0.0;
    for (int i = 0; i < 100; ++i) { // 8th line
        d += i;                     // 9th line
    }
    return d;
}

} // namespace slow_namespace

void slow_test() {
    constexpr auto slow_value = slow_namespace::slow_func(); // 17th line
    print(slow_namespace::slow_func());                      // 18th line
    print(slow_value);
}

int slow_arr[12 + 34 * 56 +                                  // 22nd line
             static_cast<int>(slow_namespace::slow_func())]; // 23rd line

constexpr int slow_init_list[] = {1, 1, 2, 3, 5, 8, 13, 21}; // 25th line
    )";

  setupProfiler();
  ASSERT_TRUE(compileFromString(Code, "-std=c++20", "test.cc"));
  std::string Json = teardownProfiler();
  std::string ExpectedGraph = R"(
ExecuteCompiler
| Frontend (test.cc)
| | ParseDeclarationOrFunctionDefinition (test.cc:2:1)
| | ParseDeclarationOrFunctionDefinition (test.cc:6:1)
| | | ParseFunctionDefinition (slow_func)
| | | | EvaluateAsRValue (<test.cc:8:21>)
| | | | EvaluateForOverflow (<test.cc:8:21, col:25>)
| | | | EvaluateForOverflow (<test.cc:8:30, col:32>)
| | | | EvaluateAsRValue (<test.cc:9:14>)
| | | | EvaluateForOverflow (<test.cc:9:9, col:14>)
| | | | isPotentialConstantExpr (slow_namespace::slow_func)
| | | | BuildCFG
| | | | | EvaluateAsBooleanCondition (<test.cc:8:21, col:25>)
| | | | | | EvaluateAsRValue (<test.cc:8:21, col:25>)
| | | | | EvaluateAsBooleanCondition (<test.cc:8:21, col:25>)
| | | | | | EvaluateAsRValue (<test.cc:8:21, col:25>)
| | ParseDeclarationOrFunctionDefinition (test.cc:16:1)
| | | ParseFunctionDefinition (slow_test)
| | | | EvaluateAsMandatedConstantInitializer (slow_value)
| | | | EvaluateAsMandatedConstantExpr (<test.cc:17:33, col:59>)
| | | | EvaluateAsMandatedConstantExpr (<test.cc:18:11, col:37>)
| | ParseDeclarationOrFunctionDefinition (test.cc:22:1)
| | | EvaluateAsMandatedConstantExpr (<test.cc:23:31, col:57>)
| | | EvaluateAsMandatedConstantRValue (<test.cc:22:14, line:23:58>)
| | ParseDeclarationOrFunctionDefinition (test.cc:25:1)
| | | EvaluateAsMandatedConstantInitializer (slow_init_list)
| | PerformPendingInstantiations
)";
  ASSERT_EQ(ExpectedGraph, buildTraceGraph(Json, ExpectedGraph));
}

TEST(TimeProfilerTest, ClassTemplateInstantiations) {
  std::string Code = R"(
    template<class T>
    struct S
    {
      void foo() {}
      void bar();
    };

    template struct S<double>; // explicit instantiation of S<double>

    void user() {
      S<int> a; // implicit instantiation of S<int>
      S<float>* b;
      b->foo(); // implicit instatiation of S<float> and S<float>::foo()
    }
  )";

  setupProfiler();
  ASSERT_TRUE(compileFromString(Code, "-std=c++20", "test.cc"));
  std::string Json = teardownProfiler();
  std::string ExpectedGraph = R"(
ExecuteCompiler
| Frontend (test.cc)
| | ParseClass (S)
| | CheckConstraintSatisfaction (<test.cc:9:21, col:29>)
| | InstantiateClass (S<double>, test.cc:9)
| | InstantiateFunction (S<double>::foo, test.cc:5)
| | ParseDeclarationOrFunctionDefinition (test.cc:11:5)
| | | ParseFunctionDefinition (user)
| | | | CheckConstraintSatisfaction (<test.cc:12:7, col:12>)
| | | | InstantiateClass (S<int>, test.cc:3)
| | | | CheckConstraintSatisfaction (<test.cc:13:7, col:14>)
| | | | InstantiateClass (S<float>, test.cc:3)
| | | | DeferInstantiation (S<float>::foo)
| | PerformPendingInstantiations
| | | InstantiateFunction (S<float>::foo, test.cc:5)
)";
  ASSERT_EQ(ExpectedGraph, buildTraceGraph(Json, ExpectedGraph));
}

TEST(TimeProfilerTest, TemplateInstantiations) {
  std::string B_H = R"(
    template <typename T>
    T fooC(T t) {
      return T();
    }

    template <typename T>
    constexpr T fooB(T t) {
      return fooC(t);
    }

    #define MacroTemp(x) template <typename T> void foo##x(T) { T(); }
  )";

  std::string A_H = R"(
    #include "b.h"

    MacroTemp(MTA)

    template <typename T>
    void fooA(T t) { fooB(t); fooMTA(t); }
  )";
  std::string Code = R"(
    #include "a.h"
    void user() { fooA(0); }
  )";

  setupProfiler();
  ASSERT_TRUE(compileFromString(Code, "-std=c++20", "test.cc",
                                /*Headers=*/{{"a.h", A_H}, {"b.h", B_H}}));
  std::string Json = teardownProfiler();
  std::string ExpectedGraph = R"(
ExecuteCompiler
| Frontend (test.cc)
| | ParseFunctionDefinition (fooC)
| | ParseFunctionDefinition (fooB)
| | ParseFunctionDefinition (fooMTA)
| | ParseFunctionDefinition (fooA)
| | ParseDeclarationOrFunctionDefinition (test.cc:3:5)
| | | ParseFunctionDefinition (user)
| | | | DeferInstantiation (fooA<int>)
| | PerformPendingInstantiations
| | | InstantiateFunction (fooA<int>, a.h:7)
| | | | InstantiateFunction (fooB<int>, b.h:8)
| | | | | DeferInstantiation (fooC<int>)
| | | | | BuildCFG
| | | | DeferInstantiation (fooMTA<int>)
| | | | InstantiateFunction (fooC<int>, b.h:3)
| | | | | BuildCFG
| | | | InstantiateFunction (fooMTA<int>, a.h:4)
)";
  ASSERT_EQ(ExpectedGraph, buildTraceGraph(Json, ExpectedGraph));
}

static SpecializationCounts
countAddedSpecializationsFromPCH(StringRef SourceFile, StringRef PCHFile,
                                 bool EnableTimeTrace) {
  SpecializationCounts Counts;
  SpecializationCountingListener Listener(Counts);
  TestFrontendAction Action(&Listener);
  std::string SourcePath = SourceFile.str();
  std::string PCHPath = PCHFile.str();

  if (EnableTimeTrace)
    setupProfiler();

  const char *Args[] = {"-std=c++20", "-include-pch", PCHPath.c_str(),
                        "-fsyntax-only", SourcePath.c_str()};
  EXPECT_TRUE(compileFromArgs(Args, Action));

  if (EnableTimeTrace)
    (void)teardownProfiler();

  return Counts;
}

TEST(TimeProfilerTest, TimeTraceDoesNotChangePCHSpecializationCount) {
  StringRef Code = R"(
#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

inline namespace {

// The first declarations give f's body references to many function templates.
#define DECLARE_G(N) template <typename T> T g##N(T v) { return v; }

DECLARE_G(0)
DECLARE_G(1)
DECLARE_G(2)
DECLARE_G(3)
DECLARE_G(4)
DECLARE_G(5)
DECLARE_G(6)
DECLARE_G(7)
DECLARE_G(8)
DECLARE_G(9)
DECLARE_G(10)
DECLARE_G(11)
DECLARE_G(12)
DECLARE_G(13)
DECLARE_G(14)
DECLARE_G(15)
DECLARE_G(16)
DECLARE_G(17)
DECLARE_G(18)
DECLARE_G(19)
DECLARE_G(20)
DECLARE_G(21)
DECLARE_G(22)
DECLARE_G(23)
DECLARE_G(24)
DECLARE_G(25)
DECLARE_G(26)
DECLARE_G(27)
DECLARE_G(28)
DECLARE_G(29)
DECLARE_G(30)
DECLARE_G(31)

#undef DECLARE_G

template <typename T> T f(T v) {
  return g0(v) + g1(v) + g2(v) + g3(v) + g4(v) + g5(v) + g6(v) +
         g7(v) + g8(v) + g9(v) + g10(v) + g11(v) + g12(v) + g13(v) +
         g14(v) + g15(v) + g16(v) + g17(v) + g18(v) + g19(v) + g20(v) +
         g21(v) + g22(v) + g23(v) + g24(v) + g25(v) + g26(v) + g27(v) +
         g28(v) + g29(v) + g30(v) + g31(v);
}

// These later declarations are deserialized while -ftime-trace prints the
// qualified name of a specialization lookup. Loading enough of them grows the
// ASTReader specialization DenseMap and used to invalidate the active lookup.
#define DECLARE_G(N) template <typename T> T g##N();

DECLARE_G(0)
DECLARE_G(1)
DECLARE_G(2)
DECLARE_G(3)
DECLARE_G(4)
DECLARE_G(5)
DECLARE_G(6)
DECLARE_G(7)
DECLARE_G(8)
DECLARE_G(9)
DECLARE_G(10)
DECLARE_G(11)
DECLARE_G(12)
DECLARE_G(13)
DECLARE_G(14)
DECLARE_G(15)
DECLARE_G(16)
DECLARE_G(17)
DECLARE_G(18)
DECLARE_G(19)
DECLARE_G(20)
DECLARE_G(21)
DECLARE_G(22)
DECLARE_G(23)
DECLARE_G(24)
DECLARE_G(25)
DECLARE_G(26)
DECLARE_G(27)
DECLARE_G(28)
DECLARE_G(29)
DECLARE_G(30)
DECLARE_G(31)

#undef DECLARE_G

} // namespace

#else

int x;
void i() { f(x); }

#endif
)";

  int SourceFD;
  SmallString<256> SourceFileName;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ftime-trace-specialization-lookup", "cpp", SourceFD, SourceFileName));
  llvm::FileRemover SourceFileRemover(SourceFileName);
  {
    raw_fd_ostream SourceOS(SourceFD, /*shouldClose=*/true);
    SourceOS << Code;
    SourceOS.flush();
    ASSERT_FALSE(SourceOS.error());
  }

  int PCHFD;
  SmallString<256> PCHFileName;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ftime-trace-specialization-lookup", "pch", PCHFD, PCHFileName));
  llvm::FileRemover PCHFileRemover(PCHFileName);
  {
    raw_fd_ostream PCHOS(PCHFD, /*shouldClose=*/true);
    PCHOS.flush();
    ASSERT_FALSE(PCHOS.error());
  }

  GeneratePCHAction GeneratePCH;
  const char *PCHArgs[] = {"-std=c++20", "-emit-pch", "-o", PCHFileName.c_str(),
                           SourceFileName.c_str()};
  ASSERT_TRUE(compileFromArgs(PCHArgs, GeneratePCH));

  SpecializationCounts WithoutTimeTrace = countAddedSpecializationsFromPCH(
      SourceFileName, PCHFileName, /*EnableTimeTrace=*/false);
  SpecializationCounts WithTimeTrace = countAddedSpecializationsFromPCH(
      SourceFileName, PCHFileName, /*EnableTimeTrace=*/true);

  EXPECT_GT(WithoutTimeTrace.FunctionTemplateSpecializations, 0u);
  EXPECT_EQ(WithoutTimeTrace.ClassTemplateSpecializations,
            WithTimeTrace.ClassTemplateSpecializations);
  EXPECT_EQ(WithoutTimeTrace.FunctionTemplateSpecializations,
            WithTimeTrace.FunctionTemplateSpecializations);
  EXPECT_EQ(WithoutTimeTrace.VarTemplateSpecializations,
            WithTimeTrace.VarTemplateSpecializations);
}

TEST(TimeProfilerTest, ConstantEvaluationC99) {
  std::string Code = R"(
struct {
  short quantval[4]; // 3rd line
} value;
    )";

  setupProfiler();
  ASSERT_TRUE(compileFromString(Code, "-std=c99", "test.c"));
  std::string Json = teardownProfiler();
  std::string ExpectedGraph = R"(
ExecuteCompiler
| Frontend (test.c)
| | ParseDeclarationOrFunctionDefinition (test.c:2:1)
| | | isIntegerConstantExpr (<test.c:3:18>)
| | | EvaluateKnownConstIntCheckOverflow (<test.c:3:18>)
| | PerformPendingInstantiations
)";
  ASSERT_EQ(ExpectedGraph, buildTraceGraph(Json, ExpectedGraph));
}
