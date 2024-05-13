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

#include "llvm/Support/JSON.h"
#include "llvm/Support/TimeProfiler.h"

#include "gtest/gtest.h"

using namespace clang;
using namespace llvm;

namespace {

// Should be called before testing.
void setupProfiler() {
  timeTraceProfilerInitialize(/*TimeTraceGranularity=*/0, "test");
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
bool compileFromString(StringRef Code, StringRef Standard, StringRef FileName) {
  CompilerInstance Compiler;
  Compiler.createDiagnostics();

  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      FileName, MemoryBuffer::getMemBuffer(Code).release());
  const char *Args[] = {Standard.data(), FileName.data()};
  CompilerInvocation::CreateFromArgs(*Invocation, Args,
                                     Compiler.getDiagnostics());
  Compiler.setInvocation(std::move(Invocation));

  class TestFrontendAction : public ASTFrontendAction {
  private:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef InFile) override {
      return std::make_unique<ASTConsumer>();
    }
  } Action;
  return Compiler.ExecuteAction(Action);
}

// Returns pretty-printed trace graph.
std::string buildTraceGraph(StringRef Json) {
  struct EventRecord {
    int64_t TimestampBegin;
    int64_t TimestampEnd;
    StringRef Name;
    StringRef Detail;
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
    StringRef Name = TraceEventObj->getString("name").value_or("");
    StringRef Detail = "";
    if (json::Object *Args = TraceEventObj->getObject("args"))
      Detail = Args->getString("detail").value_or("");

    // This is a "summary" event, like "Total PerformPendingInstantiations",
    // skip it
    if (TimestampBegin == 0)
      continue;

    Events.emplace_back(
        EventRecord{TimestampBegin, TimestampEnd, Name, Detail});
  }

  // There can be nested events that are very fast, for example:
  // {"name":"EvaluateAsBooleanCondition",... ,"ts":2380,"dur":1}
  // {"name":"EvaluateAsRValue",... ,"ts":2380,"dur":1}
  // Therefore we should reverse the events list, so that events that have
  // started earlier are first in the list.
  // Then do a stable sort, we need it for the trace graph.
  std::reverse(Events.begin(), Events.end());
  std::stable_sort(
      Events.begin(), Events.end(), [](const auto &lhs, const auto &rhs) {
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
    if (!Event.Detail.empty()) {
      Stream << " (";
      Stream.write(Event.Detail.data(), Event.Detail.size());
      Stream << ")";
    }
    Stream << "\n";
  }
  return Stream.str();
}

} // namespace

TEST(TimeProfilerTest, ConstantEvaluationCxx20) {
  constexpr StringRef Code = R"(
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
  std::string TraceGraph = buildTraceGraph(Json);
  ASSERT_TRUE(TraceGraph == R"(
Frontend
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
)");

  // NOTE: If this test is failing, run this test with
  // `llvm::errs() << TraceGraph;` and change the assert above.
}

TEST(TimeProfilerTest, ConstantEvaluationC99) {
  constexpr StringRef Code = R"(
struct {
  short quantval[4]; // 3rd line
} value;
    )";

  setupProfiler();
  ASSERT_TRUE(compileFromString(Code, "-std=c99", "test.c"));
  std::string Json = teardownProfiler();
  std::string TraceGraph = buildTraceGraph(Json);
  ASSERT_TRUE(TraceGraph == R"(
Frontend
| ParseDeclarationOrFunctionDefinition (test.c:2:1)
| | isIntegerConstantExpr (<test.c:3:18>)
| | EvaluateKnownConstIntCheckOverflow (<test.c:3:18>)
| PerformPendingInstantiations
)");

  // NOTE: If this test is failing, run this test with
  // `llvm::errs() << TraceGraph;` and change the assert above.
}
