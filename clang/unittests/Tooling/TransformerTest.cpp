//===- unittest/Tooling/TransformerTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Transformer.h"

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {
namespace {
using ast_matchers::anyOf;
using ast_matchers::argumentCountIs;
using ast_matchers::callee;
using ast_matchers::callExpr;
using ast_matchers::cxxMemberCallExpr;
using ast_matchers::cxxMethodDecl;
using ast_matchers::cxxRecordDecl;
using ast_matchers::declRefExpr;
using ast_matchers::expr;
using ast_matchers::functionDecl;
using ast_matchers::hasAnyName;
using ast_matchers::hasArgument;
using ast_matchers::hasDeclaration;
using ast_matchers::hasElse;
using ast_matchers::hasName;
using ast_matchers::hasType;
using ast_matchers::ifStmt;
using ast_matchers::member;
using ast_matchers::memberExpr;
using ast_matchers::namedDecl;
using ast_matchers::on;
using ast_matchers::pointsTo;
using ast_matchers::to;
using ast_matchers::unless;

using llvm::StringRef;

constexpr char KHeaderContents[] = R"cc(
  struct string {
    string(const char*);
    char* c_str();
    int size();
  };
  int strlen(const char*);

  namespace proto {
  struct PCFProto {
    int foo();
  };
  struct ProtoCommandLineFlag : PCFProto {
    PCFProto& GetProto();
  };
  }  // namespace proto
)cc";

static ast_matchers::internal::Matcher<clang::QualType>
isOrPointsTo(const clang::ast_matchers::DeclarationMatcher &TypeMatcher) {
  return anyOf(hasDeclaration(TypeMatcher), pointsTo(TypeMatcher));
}

static std::string format(StringRef Code) {
  const std::vector<Range> Ranges(1, Range(0, Code.size()));
  auto Style = format::getLLVMStyle();
  const auto Replacements = format::reformat(Style, Code, Ranges);
  auto Formatted = applyAllReplacements(Code, Replacements);
  if (!Formatted) {
    ADD_FAILURE() << "Could not format code: "
                  << llvm::toString(Formatted.takeError());
    return std::string();
  }
  return *Formatted;
}

static void compareSnippets(StringRef Expected,
                     const llvm::Optional<std::string> &MaybeActual) {
  ASSERT_TRUE(MaybeActual) << "Rewrite failed. Expecting: " << Expected;
  auto Actual = *MaybeActual;
  std::string HL = "#include \"header.h\"\n";
  auto I = Actual.find(HL);
  if (I != std::string::npos)
    Actual.erase(I, HL.size());
  EXPECT_EQ(format(Expected), format(Actual));
}

// FIXME: consider separating this class into its own file(s).
class ClangRefactoringTestBase : public testing::Test {
protected:
  void appendToHeader(StringRef S) { FileContents[0].second += S; }

  void addFile(StringRef Filename, StringRef Content) {
    FileContents.emplace_back(Filename, Content);
  }

  llvm::Optional<std::string> rewrite(StringRef Input) {
    std::string Code = ("#include \"header.h\"\n" + Input).str();
    auto Factory = newFrontendActionFactory(&MatchFinder);
    if (!runToolOnCodeWithArgs(
            Factory->create(), Code, std::vector<std::string>(), "input.cc",
            "clang-tool", std::make_shared<PCHContainerOperations>(),
            FileContents)) {
      return None;
    }
    auto ChangedCodeOrErr =
        applyAtomicChanges("input.cc", Code, Changes, ApplyChangesSpec());
    if (auto Err = ChangedCodeOrErr.takeError()) {
      llvm::errs() << "Change failed: " << llvm::toString(std::move(Err))
                   << "\n";
      return None;
    }
    return *ChangedCodeOrErr;
  }

  void testRule(RewriteRule Rule, StringRef Input, StringRef Expected) {
    Transformer T(std::move(Rule),
                  [this](const AtomicChange &C) { Changes.push_back(C); });
    T.registerMatchers(&MatchFinder);
    compareSnippets(Expected, rewrite(Input));
  }

  clang::ast_matchers::MatchFinder MatchFinder;
  AtomicChanges Changes;

private:
  FileContentMappings FileContents = {{"header.h", ""}};
};

class TransformerTest : public ClangRefactoringTestBase {
protected:
  TransformerTest() { appendToHeader(KHeaderContents); }
};

// Given string s, change strlen($s.c_str()) to $s.size().
static RewriteRule ruleStrlenSize() {
  StringRef StringExpr = "strexpr";
  auto StringType = namedDecl(hasAnyName("::basic_string", "::string"));
  return buildRule(
             callExpr(
                 callee(functionDecl(hasName("strlen"))),
                 hasArgument(0, cxxMemberCallExpr(
                                    on(expr(hasType(isOrPointsTo(StringType)))
                                           .bind(StringExpr)),
                                    callee(cxxMethodDecl(hasName("c_str")))))))
      // Specify the intended type explicitly, because the matcher "type" of
      // `callExpr()` is `Stmt`, not `Expr`.
      .as<clang::Expr>()
      .replaceWith("REPLACED")
      .because("Use size() method directly on string.");
}

TEST_F(TransformerTest, StrlenSize) {
  std::string Input = "int f(string s) { return strlen(s.c_str()); }";
  std::string Expected = "int f(string s) { return REPLACED; }";
  testRule(ruleStrlenSize(), Input, Expected);
}

// Tests that no change is applied when a match is not expected.
TEST_F(TransformerTest, NoMatch) {
  std::string Input = "int f(string s) { return s.size(); }";
  testRule(ruleStrlenSize(), Input, Input);
}

// Tests that expressions in macro arguments are rewritten (when applicable).
TEST_F(TransformerTest, StrlenSizeMacro) {
  std::string Input = R"cc(
#define ID(e) e
    int f(string s) { return ID(strlen(s.c_str())); })cc";
  std::string Expected = R"cc(
#define ID(e) e
    int f(string s) { return ID(REPLACED); })cc";
  testRule(ruleStrlenSize(), Input, Expected);
}

// Tests replacing an expression.
TEST_F(TransformerTest, Flag) {
  StringRef Flag = "flag";
  RewriteRule Rule =
      buildRule(
          cxxMemberCallExpr(on(expr(hasType(cxxRecordDecl(hasName(
                                        "proto::ProtoCommandLineFlag"))))
                                   .bind(Flag)),
                            unless(callee(cxxMethodDecl(hasName("GetProto"))))))
          .change<clang::Expr>(Flag)
          .replaceWith("EXPR")
          .because("Use GetProto() to access proto fields.");

  std::string Input = R"cc(
    proto::ProtoCommandLineFlag flag;
    int x = flag.foo();
    int y = flag.GetProto().foo();
  )cc";
  std::string Expected = R"cc(
    proto::ProtoCommandLineFlag flag;
    int x = EXPR.foo();
    int y = flag.GetProto().foo();
  )cc";

  testRule(std::move(Rule), Input, Expected);
}

TEST_F(TransformerTest, NodePartNameNamedDecl) {
  StringRef Fun = "fun";
  RewriteRule Rule = buildRule(functionDecl(hasName("bad")).bind(Fun))
                         .change<clang::FunctionDecl>(Fun, NodePart::Name)
                         .replaceWith("good");

  std::string Input = R"cc(
    int bad(int x);
    int bad(int x) { return x * x; }
  )cc";
  std::string Expected = R"cc(
    int good(int x);
    int good(int x) { return x * x; }
  )cc";

  testRule(Rule, Input, Expected);
}

TEST_F(TransformerTest, NodePartNameDeclRef) {
  std::string Input = R"cc(
    template <typename T>
    T bad(T x) {
      return x;
    }
    int neutral(int x) { return bad<int>(x) * x; }
  )cc";
  std::string Expected = R"cc(
    template <typename T>
    T bad(T x) {
      return x;
    }
    int neutral(int x) { return good<int>(x) * x; }
  )cc";

  StringRef Ref = "ref";
  testRule(buildRule(declRefExpr(to(functionDecl(hasName("bad")))).bind(Ref))
               .change<clang::Expr>(Ref, NodePart::Name)
               .replaceWith("good"),
           Input, Expected);
}

TEST_F(TransformerTest, NodePartNameDeclRefFailure) {
  std::string Input = R"cc(
    struct Y {
      int operator*();
    };
    int neutral(int x) {
      Y y;
      int (Y::*ptr)() = &Y::operator*;
      return *y + x;
    }
  )cc";

  StringRef Ref = "ref";
  testRule(buildRule(declRefExpr(to(functionDecl())).bind(Ref))
               .change<clang::Expr>(Ref, NodePart::Name)
               .replaceWith("good"),
           Input, Input);
}

TEST_F(TransformerTest, NodePartMember) {
  StringRef E = "expr";
  RewriteRule Rule = buildRule(memberExpr(member(hasName("bad"))).bind(E))
                         .change<clang::Expr>(E, NodePart::Member)
                         .replaceWith("good");

  std::string Input = R"cc(
    struct S {
      int bad;
    };
    int g() {
      S s;
      return s.bad;
    }
  )cc";
  std::string Expected = R"cc(
    struct S {
      int bad;
    };
    int g() {
      S s;
      return s.good;
    }
  )cc";

  testRule(Rule, Input, Expected);
}

TEST_F(TransformerTest, NodePartMemberQualified) {
  std::string Input = R"cc(
    struct S {
      int bad;
      int good;
    };
    struct T : public S {
      int bad;
    };
    int g() {
      T t;
      return t.S::bad;
    }
  )cc";
  std::string Expected = R"cc(
    struct S {
      int bad;
      int good;
    };
    struct T : public S {
      int bad;
    };
    int g() {
      T t;
      return t.S::good;
    }
  )cc";

  StringRef E = "expr";
  testRule(buildRule(memberExpr().bind(E))
               .change<clang::Expr>(E, NodePart::Member)
               .replaceWith("good"),
           Input, Expected);
}

TEST_F(TransformerTest, NodePartMemberMultiToken) {
  std::string Input = R"cc(
    struct Y {
      int operator*();
      int good();
      template <typename T> void foo(T t);
    };
    int neutral(int x) {
      Y y;
      y.template foo<int>(3);
      return y.operator *();
    }
  )cc";
  std::string Expected = R"cc(
    struct Y {
      int operator*();
      int good();
      template <typename T> void foo(T t);
    };
    int neutral(int x) {
      Y y;
      y.template good<int>(3);
      return y.good();
    }
  )cc";

  StringRef MemExpr = "member";
  testRule(buildRule(memberExpr().bind(MemExpr))
               .change<clang::Expr>(MemExpr, NodePart::Member)
               .replaceWith("good"),
           Input, Expected);
}

//
// Negative tests (where we expect no transformation to occur).
//

TEST_F(TransformerTest, NoTransformationInMacro) {
  std::string Input = R"cc(
#define MACRO(str) strlen((str).c_str())
    int f(string s) { return MACRO(s); })cc";
  testRule(ruleStrlenSize(), Input, Input);
}

// This test handles the corner case where a macro called within another macro
// expands to matching code, but the matched code is an argument to the nested
// macro.  A simple check of isMacroArgExpansion() vs. isMacroBodyExpansion()
// will get this wrong, and transform the code. This test verifies that no such
// transformation occurs.
TEST_F(TransformerTest, NoTransformationInNestedMacro) {
  std::string Input = R"cc(
#define NESTED(e) e
#define MACRO(str) NESTED(strlen((str).c_str()))
    int f(string s) { return MACRO(s); })cc";
  testRule(ruleStrlenSize(), Input, Input);
}
} // namespace
} // namespace tooling
} // namespace clang
