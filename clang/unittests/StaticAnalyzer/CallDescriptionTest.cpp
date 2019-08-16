//===- unittests/StaticAnalyzer/CallDescriptionTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Reusables.h"

#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ento {
namespace {

// A wrapper around CallDescriptionMap<bool> that allows verifying that
// all functions have been found. This is needed because CallDescriptionMap
// isn't supposed to support iteration.
class ResultMap {
  size_t Found, Total;
  CallDescriptionMap<bool> Impl;

public:
  ResultMap(std::initializer_list<std::pair<CallDescription, bool>> Data)
      : Found(0),
        Total(std::count_if(Data.begin(), Data.end(),
                            [](const std::pair<CallDescription, bool> &Pair) {
                              return Pair.second == true;
                            })),
        Impl(std::move(Data)) {}

  const bool *lookup(const CallEvent &Call) {
    const bool *Result = Impl.lookup(Call);
    // If it's a function we expected to find, remember that we've found it.
    if (Result && *Result)
      ++Found;
    return Result;
  }

  // Fail the test if we haven't found all the true-calls we were looking for.
  ~ResultMap() { EXPECT_EQ(Found, Total); }
};

// Scan the code body for call expressions and see if we find all calls that
// we were supposed to find ("true" in the provided ResultMap) and that we
// don't find the ones that we weren't supposed to find
// ("false" in the ResultMap).
class CallDescriptionConsumer : public ExprEngineConsumer {
  ResultMap &RM;
  void performTest(const Decl *D) {
    using namespace ast_matchers;

    if (!D->hasBody())
      return;

    const CallExpr *CE = findNode<CallExpr>(D, callExpr());
    const StackFrameContext *SFC =
        Eng.getAnalysisDeclContextManager().getStackFrame(D);
    ProgramStateRef State = Eng.getInitialState(SFC);
    CallEventRef<> Call =
        Eng.getStateManager().getCallEventManager().getCall(CE, State, SFC);

    const bool *LookupResult = RM.lookup(*Call);
    // Check that we've found the function in the map
    // with the correct description.
    EXPECT_TRUE(LookupResult && *LookupResult);

    // ResultMap is responsible for making sure that we've found *all* calls.
  }

public:
  CallDescriptionConsumer(CompilerInstance &C,
                          ResultMap &RM)
      : ExprEngineConsumer(C), RM(RM) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (const auto *D : DG)
      performTest(D);
    return true;
  }
};

class CallDescriptionAction : public ASTFrontendAction {
  ResultMap RM;

public:
  CallDescriptionAction(
      std::initializer_list<std::pair<CallDescription, bool>> Data)
      : RM(Data) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    return std::make_unique<CallDescriptionConsumer>(Compiler, RM);
  }
};

TEST(CallEvent, CallDescription) {
  // Test simple name matching.
  EXPECT_TRUE(tooling::runToolOnCode(
      new CallDescriptionAction({
          {{"bar"}, false}, // false: there's no call to 'bar' in this code.
          {{"foo"}, true},  // true: there's a call to 'foo' in this code.
      }), "void foo(); void bar() { foo(); }"));

  // Test arguments check.
  EXPECT_TRUE(tooling::runToolOnCode(
      new CallDescriptionAction({
          {{"foo", 1}, true},
          {{"foo", 2}, false},
      }), "void foo(int); void foo(int, int); void bar() { foo(1); }"));

  // Test lack of arguments check.
  EXPECT_TRUE(tooling::runToolOnCode(
      new CallDescriptionAction({
          {{"foo", None}, true},
          {{"foo", 2}, false},
      }), "void foo(int); void foo(int, int); void bar() { foo(1); }"));

  // Test qualified names.
  EXPECT_TRUE(tooling::runToolOnCode(
      new CallDescriptionAction({
          {{{"std", "basic_string", "c_str"}}, true},
      }),
      "namespace std { inline namespace __1 {"
      "  template<typename T> class basic_string {"
      "  public:"
      "    T *c_str();"
      "  };"
      "}}"
      "void foo() {"
      "  using namespace std;"
      "  basic_string<char> s;"
      "  s.c_str();"
      "}"));

  // A negative test for qualified names.
  EXPECT_TRUE(tooling::runToolOnCode(
      new CallDescriptionAction({
          {{{"foo", "bar"}}, false},
          {{{"bar", "foo"}}, false},
          {{"foo"}, true},
      }), "void foo(); struct bar { void foo(); }; void test() { foo(); }"));

  // Test CDF_MaybeBuiltin - a flag that allows matching weird builtins.
  EXPECT_TRUE(tooling::runToolOnCode(
      new CallDescriptionAction({
          {{"memset", 3}, false},
          {{CDF_MaybeBuiltin, "memset", 3}, true}
      }),
      "void foo() {"
      "  int x;"
      "  __builtin___memset_chk(&x, 0, sizeof(x),"
      "                         __builtin_object_size(&x, 0));"
      "}"));
}

} // namespace
} // namespace ento
} // namespace clang
