//===- unittests/Support/TimeProfilerTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <stack>

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

// Returns true if code compiles successfully.
// We only parse AST here. This is enough for constexpr evaluation.
bool compileFromString(StringRef Code, StringRef Standard, StringRef File,
                       llvm::StringMap<std::string> Headers = {}) {

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS(
      new llvm::vfs::InMemoryFileSystem());
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

  class TestFrontendAction : public ASTFrontendAction {
  private:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) override {
      return std::make_unique<ASTConsumer>();
    }
  } Action;
  return Compiler.ExecuteAction(Action);
}

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
std::string buildTraceGraph(StringRef Json) {
  struct EventRecord {
    int64_t TimestampBegin;
    int64_t TimestampEnd;
    std::string Name;
    std::string Metadata;
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

    // Source events are asynchronous events and may not perfectly nest the
    // synchronous events. Skip testing them.
    if (Name == "Source")
      continue;

    // This is a "summary" event, like "Total PerformPendingInstantiations",
    // skip it
    if (TimestampBegin == 0)
      continue;

    Events.emplace_back(
        EventRecord{TimestampBegin, TimestampEnd, Name, Metadata});
  }

  // There can be nested events that are very fast, for example:
  // {"name":"EvaluateAsBooleanCondition",... ,"ts":2380,"dur":1}
  // {"name":"EvaluateAsRValue",... ,"ts":2380,"dur":1}
  // Therefore we should reverse the events list, so that events that have
  // started earlier are first in the list.
  // Then do a stable sort, we need it for the trace graph.
  std::reverse(Events.begin(), Events.end());
  llvm::stable_sort(Events, [](const auto &lhs, const auto &rhs) {
    return std::make_pair(lhs.TimestampBegin, -lhs.TimestampEnd) <
           std::make_pair(rhs.TimestampBegin, -rhs.TimestampEnd);
  });

  std::stringstream Stream;
  // Write a newline for better testing with multiline string literal.
  Stream << "\n";

  // Keep the current event stack.
  std::stack<const EventRecord *> EventStack;
  for (const auto &Event : Events) {
    // Pop every event in the stack until meeting the parent event.
    while (!EventStack.empty()) {
      bool InsideCurrentEvent =
          Event.TimestampBegin >= EventStack.top()->TimestampBegin &&
          Event.TimestampEnd <= EventStack.top()->TimestampEnd;

      // Presumably due to timer rounding, PerformPendingInstantiations often
      // appear to be within the timer interval of the immediately previous
      // event group. We always know these events occur at level 1, not level 2,
      // in our tests, so pop an event in that case.
      if (InsideCurrentEvent && Event.Name == "PerformPendingInstantiations" &&
          EventStack.size() == 2) {
        InsideCurrentEvent = false;
      }

      if (!InsideCurrentEvent)
        EventStack.pop();
      else
        break;
    }
    EventStack.push(&Event);

    // Write indentaion, name, detail, newline.
    for (size_t i = 1; i < EventStack.size(); ++i) {
      Stream << "| ";
    }
    Stream.write(Event.Name.data(), Event.Name.size());
    if (!Event.Metadata.empty()) {
      Stream << " (";
      Stream.write(Event.Metadata.data(), Event.Metadata.size());
      Stream << ")";
    }
    Stream << "\n";
  }
  return Stream.str();
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
  ASSERT_EQ(R"(
Frontend (test.cc)
| ParseDeclarationOrFunctionDefinition (test.cc:2:1)
| ParseDeclarationOrFunctionDefinition (test.cc:6:1)
| | ParseFunctionDefinition (slow_func)
| | | EvaluateAsRValue (<test.cc:8:21>)
| | | EvaluateForOverflow (<test.cc:8:21, col:25>)
| | | EvaluateForOverflow (<test.cc:8:30, col:32>)
| | | EvaluateAsRValue (<test.cc:9:14>)
| | | EvaluateForOverflow (<test.cc:9:9, col:14>)
| | | isPotentialConstantExpr (slow_namespace::slow_func)
| | | EvaluateAsBooleanCondition (<test.cc:8:21, col:25>)
| | | | EvaluateAsRValue (<test.cc:8:21, col:25>)
| | | EvaluateAsBooleanCondition (<test.cc:8:21, col:25>)
| | | | EvaluateAsRValue (<test.cc:8:21, col:25>)
| ParseDeclarationOrFunctionDefinition (test.cc:16:1)
| | ParseFunctionDefinition (slow_test)
| | | EvaluateAsInitializer (slow_value)
| | | EvaluateAsConstantExpr (<test.cc:17:33, col:59>)
| | | EvaluateAsConstantExpr (<test.cc:18:11, col:37>)
| ParseDeclarationOrFunctionDefinition (test.cc:22:1)
| | EvaluateAsConstantExpr (<test.cc:23:31, col:57>)
| | EvaluateAsRValue (<test.cc:22:14, line:23:58>)
| ParseDeclarationOrFunctionDefinition (test.cc:25:1)
| | EvaluateAsInitializer (slow_init_list)
| PerformPendingInstantiations
)",
            buildTraceGraph(Json));
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
  ASSERT_EQ(R"(
Frontend (test.cc)
| ParseClass (S)
| InstantiateClass (S<double>, test.cc:9)
| InstantiateFunction (S<double>::foo, test.cc:5)
| ParseDeclarationOrFunctionDefinition (test.cc:11:5)
| | ParseFunctionDefinition (user)
| | | InstantiateClass (S<int>, test.cc:3)
| | | InstantiateClass (S<float>, test.cc:3)
| | | DeferInstantiation (S<float>::foo)
| PerformPendingInstantiations
| | InstantiateFunction (S<float>::foo, test.cc:5)
)",
            buildTraceGraph(Json));
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
  ASSERT_EQ(R"(
Frontend (test.cc)
| ParseFunctionDefinition (fooC)
| ParseFunctionDefinition (fooB)
| ParseFunctionDefinition (fooMTA)
| ParseFunctionDefinition (fooA)
| ParseDeclarationOrFunctionDefinition (test.cc:3:5)
| | ParseFunctionDefinition (user)
| | | DeferInstantiation (fooA<int>)
| PerformPendingInstantiations
| | InstantiateFunction (fooA<int>, a.h:7)
| | | InstantiateFunction (fooB<int>, b.h:8)
| | | | DeferInstantiation (fooC<int>)
| | | DeferInstantiation (fooMTA<int>)
| | | InstantiateFunction (fooC<int>, b.h:3)
| | | InstantiateFunction (fooMTA<int>, a.h:4)
)",
            buildTraceGraph(Json));
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
  ASSERT_EQ(R"(
Frontend (test.c)
| ParseDeclarationOrFunctionDefinition (test.c:2:1)
| | isIntegerConstantExpr (<test.c:3:18>)
| | EvaluateKnownConstIntCheckOverflow (<test.c:3:18>)
| PerformPendingInstantiations
)",
            buildTraceGraph(Json));
}
