//===- unittests/StaticAnalyzer/CallDescriptionTest.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "Reusables.h"

#include "clang/AST/ExprCXX.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <type_traits>

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
template <typename MatchedExprT>
class CallDescriptionConsumer : public ExprEngineConsumer {
  ResultMap &RM;
  void performTest(const Decl *D) {
    using namespace ast_matchers;
    using T = MatchedExprT;

    if (!D->hasBody())
      return;

    const StackFrameContext *SFC =
        Eng.getAnalysisDeclContextManager().getStackFrame(D);
    const ProgramStateRef State = Eng.getInitialState(SFC);

    // FIXME: Maybe use std::variant and std::visit for these.
    const auto MatcherCreator = []() {
      if (std::is_same<T, CallExpr>::value)
        return callExpr();
      if (std::is_same<T, CXXConstructExpr>::value)
        return cxxConstructExpr();
      if (std::is_same<T, CXXMemberCallExpr>::value)
        return cxxMemberCallExpr();
      if (std::is_same<T, CXXOperatorCallExpr>::value)
        return cxxOperatorCallExpr();
      llvm_unreachable("Only these expressions are supported for now.");
    };

    const Expr *E = findNode<T>(D, MatcherCreator());

    CallEventManager &CEMgr = Eng.getStateManager().getCallEventManager();
    CallEventRef<> Call = [=, &CEMgr]() -> CallEventRef<CallEvent> {
      CFGBlock::ConstCFGElementRef ElemRef = {SFC->getCallSiteBlock(),
                                              SFC->getIndex()};
      if (std::is_base_of<CallExpr, T>::value)
        return CEMgr.getCall(E, State, SFC, ElemRef);
      if (std::is_same<T, CXXConstructExpr>::value)
        return CEMgr.getCXXConstructorCall(cast<CXXConstructExpr>(E),
                                           /*Target=*/nullptr, State, SFC,
                                           ElemRef);
      llvm_unreachable("Only these expressions are supported for now.");
    }();

    // If the call actually matched, check if we really expected it to match.
    const bool *LookupResult = RM.lookup(*Call);
    EXPECT_TRUE(!LookupResult || *LookupResult);

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

template <typename MatchedExprT = CallExpr>
class CallDescriptionAction : public ASTFrontendAction {
  ResultMap RM;

public:
  CallDescriptionAction(
      std::initializer_list<std::pair<CallDescription, bool>> Data)
      : RM(Data) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                 StringRef File) override {
    return std::make_unique<CallDescriptionConsumer<MatchedExprT>>(Compiler,
                                                                   RM);
  }
};

TEST(CallDescription, SimpleNameMatching) {
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"bar"}}, false}, // false: there's no call to 'bar' in this code.
          {{{"foo"}}, true},  // true: there's a call to 'foo' in this code.
      })),
      "void foo(); void bar() { foo(); }"));
}

TEST(CallDescription, RequiredArguments) {
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"foo"}, 1}, true},
          {{{"foo"}, 2}, false},
      })),
      "void foo(int); void foo(int, int); void bar() { foo(1); }"));
}

TEST(CallDescription, LackOfRequiredArguments) {
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"foo"}, std::nullopt}, true},
          {{{"foo"}, 2}, false},
      })),
      "void foo(int); void foo(int, int); void bar() { foo(1); }"));
}

constexpr StringRef MockStdStringHeader = R"code(
  namespace std { inline namespace __1 {
    template<typename T> class basic_string {
      class Allocator {};
    public:
      basic_string();
      explicit basic_string(const char*, const Allocator & = Allocator());
      ~basic_string();
      T *c_str();
    };
  } // namespace __1
  using string = __1::basic_string<char>;
  } // namespace std
)code";

TEST(CallDescription, QualifiedNames) {
  constexpr StringRef AdditionalCode = R"code(
    void foo() {
      using namespace std;
      basic_string<char> s;
      s.c_str();
    })code";
  const std::string Code = (Twine{MockStdStringHeader} + AdditionalCode).str();
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"std", "basic_string", "c_str"}}, true},
      })),
      Code));
}

TEST(CallDescription, MatchConstructor) {
  constexpr StringRef AdditionalCode = R"code(
    void foo() {
      using namespace std;
      basic_string<char> s("hello");
    })code";
  const std::string Code = (Twine{MockStdStringHeader} + AdditionalCode).str();
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(
          new CallDescriptionAction<CXXConstructExpr>({
              {{{"std", "basic_string", "basic_string"}, 2, 2}, true},
          })),
      Code));
}

// FIXME: Test matching destructors: {"std", "basic_string", "~basic_string"}
//        This feature is actually implemented, but the test infra is not yet
//        sophisticated enough for testing this. To do that, we will need to
//        implement a much more advanced dispatching mechanism using the CFG for
//        the implicit destructor events.

TEST(CallDescription, MatchConversionOperator) {
  constexpr StringRef Code = R"code(
    namespace aaa {
    namespace bbb {
    struct Bar {
      operator int();
    };
    } // bbb
    } // aaa
    void foo() {
      aaa::bbb::Bar x;
      int tmp = x;
    })code";
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"aaa", "bbb", "Bar", "operator int"}}, true},
      })),
      Code));
}

TEST(CallDescription, RejectOverQualifiedNames) {
  constexpr auto Code = R"code(
    namespace my {
    namespace std {
      struct container {
        const char *data() const;
      };
    } // namespace std
    } // namespace my

    void foo() {
      using namespace my;
      std::container v;
      v.data();
    })code";

  // FIXME: We should **not** match.
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"std", "container", "data"}}, true},
      })),
      Code));
}

TEST(CallDescription, DontSkipNonInlineNamespaces) {
  constexpr auto Code = R"code(
    namespace my {
    /*not inline*/ namespace v1 {
      void bar();
    } // namespace v1
    } // namespace my
    void foo() {
      my::v1::bar();
    })code";

  {
    SCOPED_TRACE("my v1 bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"my", "v1", "bar"}}, true},
        })),
        Code));
  }
  {
    // FIXME: We should **not** skip non-inline namespaces.
    SCOPED_TRACE("my bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"my", "bar"}}, true},
        })),
        Code));
  }
}

TEST(CallDescription, SkipTopInlineNamespaces) {
  constexpr auto Code = R"code(
    inline namespace my {
    namespace v1 {
      void bar();
    } // namespace v1
    } // namespace my
    void foo() {
      using namespace v1;
      bar();
    })code";

  {
    SCOPED_TRACE("my v1 bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"my", "v1", "bar"}}, true},
        })),
        Code));
  }
  {
    SCOPED_TRACE("v1 bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"v1", "bar"}}, true},
        })),
        Code));
  }
}

TEST(CallDescription, SkipAnonimousNamespaces) {
  constexpr auto Code = R"code(
    namespace {
    namespace std {
    namespace {
    inline namespace {
      struct container {
        const char *data() const { return nullptr; };
      };
    } // namespace inline anonymous
    } // namespace anonymous
    } // namespace std
    } // namespace anonymous

    void foo() {
      std::container v;
      v.data();
    })code";

  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"std", "container", "data"}}, true},
      })),
      Code));
}

TEST(CallDescription, AliasNames) {
  constexpr StringRef AliasNamesCode = R"code(
  namespace std {
    struct container {
      const char *data() const;
    };
    using cont = container;
  } // std
)code";

  constexpr StringRef UseAliasInSpelling = R"code(
    void foo() {
      std::cont v;
      v.data();
    })code";
  constexpr StringRef UseStructNameInSpelling = R"code(
    void foo() {
      std::container v;
      v.data();
    })code";
  const std::string UseAliasInSpellingCode =
      (Twine{AliasNamesCode} + UseAliasInSpelling).str();
  const std::string UseStructNameInSpellingCode =
      (Twine{AliasNamesCode} + UseStructNameInSpelling).str();

  // Test if the code spells the alias, wile we match against the struct name,
  // and again matching against the alias.
  {
    SCOPED_TRACE("Using alias in spelling");
    {
      SCOPED_TRACE("std container data");
      EXPECT_TRUE(tooling::runToolOnCode(
          std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
              {{{"std", "container", "data"}}, true},
          })),
          UseAliasInSpellingCode));
    }
    {
      // FIXME: We should be able to see-through aliases.
      SCOPED_TRACE("std cont data");
      EXPECT_TRUE(tooling::runToolOnCode(
          std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
              {{{"std", "cont", "data"}}, false},
          })),
          UseAliasInSpellingCode));
    }
  }

  // Test if the code spells the struct name, wile we match against the struct
  // name, and again matching against the alias.
  {
    SCOPED_TRACE("Using struct name in spelling");
    {
      SCOPED_TRACE("std container data");
      EXPECT_TRUE(tooling::runToolOnCode(
          std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
              {{{"std", "container", "data"}}, true},
          })),
          UseAliasInSpellingCode));
    }
    {
      // FIXME: We should be able to see-through aliases.
      SCOPED_TRACE("std cont data");
      EXPECT_TRUE(tooling::runToolOnCode(
          std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
              {{{"std", "cont", "data"}}, false},
          })),
          UseAliasInSpellingCode));
    }
  }
}

TEST(CallDescription, AliasSingleNamespace) {
  constexpr StringRef Code = R"code(
    namespace aaa {
    namespace bbb {
    namespace ccc {
      void bar();
    }} // namespace bbb::ccc
    namespace bbb_alias = bbb;
    } // namespace aaa
    void foo() {
      aaa::bbb_alias::ccc::bar();
    })code";
  {
    SCOPED_TRACE("aaa bbb ccc bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"aaa", "bbb", "ccc", "bar"}}, true},
        })),
        Code));
  }
  {
    // FIXME: We should be able to see-through namespace aliases.
    SCOPED_TRACE("aaa bbb_alias ccc bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"aaa", "bbb_alias", "ccc", "bar"}}, false},
        })),
        Code));
  }
}

TEST(CallDescription, AliasMultipleNamespaces) {
  constexpr StringRef Code = R"code(
    namespace aaa {
    namespace bbb {
    namespace ccc {
      void bar();
    }}} // namespace aaa::bbb::ccc
    namespace aaa_bbb_ccc = aaa::bbb::ccc;
    void foo() {
      using namespace aaa_bbb_ccc;
      bar();
    })code";
  {
    SCOPED_TRACE("aaa bbb ccc bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"aaa", "bbb", "ccc", "bar"}}, true},
        })),
        Code));
  }
  {
    // FIXME: We should be able to see-through namespace aliases.
    SCOPED_TRACE("aaa_bbb_ccc bar");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
            {{{"aaa_bbb_ccc", "bar"}}, false},
        })),
        Code));
  }
}

TEST(CallDescription, NegativeMatchQualifiedNames) {
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>({
          {{{"foo", "bar"}}, false},
          {{{"bar", "foo"}}, false},
          {{{"foo"}}, true},
      })),
      "void foo(); struct bar { void foo(); }; void test() { foo(); }"));
}

TEST(CallDescription, MatchBuiltins) {
  // Test CDM::CLibrary - a flag that allows matching weird builtins.
  EXPECT_TRUE(tooling::runToolOnCode(
      std::unique_ptr<FrontendAction>(new CallDescriptionAction<>(
          {{{{"memset"}, 3}, false}, {{CDM::CLibrary, {"memset"}, 3}, true}})),
      "void foo() {"
      "  int x;"
      "  __builtin___memset_chk(&x, 0, sizeof(x),"
      "                         __builtin_object_size(&x, 0));"
      "}"));

  {
    SCOPED_TRACE("multiple similar builtins");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>(
            {{{CDM::CLibrary, {"memcpy"}, 3}, false},
             {{CDM::CLibrary, {"wmemcpy"}, 3}, true}})),
        R"(void foo(wchar_t *x, wchar_t *y) {
            __builtin_wmemcpy(x, y, sizeof(wchar_t));
          })"));
  }
  {
    SCOPED_TRACE("multiple similar builtins reversed order");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(new CallDescriptionAction<>(
            {{{CDM::CLibrary, {"wmemcpy"}, 3}, true},
             {{CDM::CLibrary, {"memcpy"}, 3}, false}})),
        R"(void foo(wchar_t *x, wchar_t *y) {
            __builtin_wmemcpy(x, y, sizeof(wchar_t));
          })"));
  }
  {
    SCOPED_TRACE("lookbehind and lookahead mismatches");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(
            new CallDescriptionAction<>({{{CDM::CLibrary, {"func"}}, false}})),
        R"(
          void funcXXX();
          void XXXfunc();
          void XXXfuncXXX();
          void test() {
            funcXXX();
            XXXfunc();
            XXXfuncXXX();
          })"));
  }
  {
    SCOPED_TRACE("lookbehind and lookahead matches");
    EXPECT_TRUE(tooling::runToolOnCode(
        std::unique_ptr<FrontendAction>(
            new CallDescriptionAction<>({{{CDM::CLibrary, {"func"}}, true}})),
        R"(
          void func();
          void func_XXX();
          void XXX_func();
          void XXX_func_XXX();

          void test() {
            func(); // exact match
            func_XXX();
            XXX_func();
            XXX_func_XXX();
          })"));
  }
}

//===----------------------------------------------------------------------===//
// Testing through a checker interface.
//
// Above, the static analyzer isn't run properly, only the bare minimum to
// create CallEvents. This causes CallEvents through function pointers to not
// refer to the pointee function, but this works fine if we run
// AnalysisASTConsumer.
//===----------------------------------------------------------------------===//

class CallDescChecker
    : public Checker<check::PreCall, check::PreStmt<CallExpr>> {
  CallDescriptionSet Set = {{{"bar"}, 0}};

public:
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    if (Set.contains(Call)) {
      C.getBugReporter().EmitBasicReport(
          Call.getDecl(), this, "CallEvent match", categories::LogicError,
          "CallEvent match",
          PathDiagnosticLocation{Call.getDecl(), C.getSourceManager()});
    }
  }

  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const {
    if (Set.containsAsWritten(*CE)) {
      C.getBugReporter().EmitBasicReport(
          CE->getCalleeDecl(), this, "CallExpr match", categories::LogicError,
          "CallExpr match",
          PathDiagnosticLocation{CE->getCalleeDecl(), C.getSourceManager()});
    }
  }
};

void addCallDescChecker(AnalysisASTConsumer &AnalysisConsumer,
                        AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.CallDescChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<CallDescChecker>("test.CallDescChecker", "Description",
                                         "");
  });
}

TEST(CallDescription, CheckCallExprMatching) {
  // Imprecise matching shouldn't catch the call to bar, because its obscured
  // by a function pointer.
  constexpr StringRef FnPtrCode = R"code(
    void bar();
    void foo() {
      void (*fnptr)() = bar;
      fnptr();
    })code";
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addCallDescChecker>(FnPtrCode.str(), Diags,
                                                   /*OnlyEmitWarnings*/ true));
  EXPECT_EQ("test.CallDescChecker: CallEvent match\n", Diags);

  // This should be caught properly by imprecise matching, as the call is done
  // purely through syntactic means.
  constexpr StringRef Code = R"code(
    void bar();
    void foo() {
      bar();
    })code";
  Diags.clear();
  EXPECT_TRUE(runCheckerOnCode<addCallDescChecker>(Code.str(), Diags,
                                                   /*OnlyEmitWarnings*/ true));
  EXPECT_EQ("test.CallDescChecker: CallEvent match\n"
            "test.CallDescChecker: CallExpr match\n",
            Diags);
}

} // namespace
} // namespace ento
} // namespace clang
