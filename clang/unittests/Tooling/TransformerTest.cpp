//===- unittest/Tooling/TransformerTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Transformer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Refactoring/RangeSelector.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace ast_matchers;

namespace {
using ::testing::IsEmpty;

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
  class Logger {};
  void operator<<(Logger& l, string msg);
  Logger& log(int level);
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
      llvm::errs() << "Running tool failed.\n";
      return None;
    }
    if (ErrorCount != 0) {
      llvm::errs() << "Generating changes failed.\n";
      return None;
    }
    auto ChangedCode =
        applyAtomicChanges("input.cc", Code, Changes, ApplyChangesSpec());
    if (!ChangedCode) {
      llvm::errs() << "Applying changes failed: "
                   << llvm::toString(ChangedCode.takeError()) << "\n";
      return None;
    }
    return *ChangedCode;
  }

  Transformer::ChangeConsumer consumer() {
    return [this](Expected<AtomicChange> C) {
      if (C) {
        Changes.push_back(std::move(*C));
      } else {
        consumeError(C.takeError());
        ++ErrorCount;
      }
    };
  }

  template <typename R>
  void testRule(R Rule, StringRef Input, StringRef Expected) {
    Transformer T(std::move(Rule), consumer());
    T.registerMatchers(&MatchFinder);
    compareSnippets(Expected, rewrite(Input));
  }

  clang::ast_matchers::MatchFinder MatchFinder;
  // Records whether any errors occurred in individual changes.
  int ErrorCount = 0;
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
  auto R = makeRule(
      callExpr(callee(functionDecl(hasName("strlen"))),
               hasArgument(0, cxxMemberCallExpr(
                                  on(expr(hasType(isOrPointsTo(StringType)))
                                         .bind(StringExpr)),
                                  callee(cxxMethodDecl(hasName("c_str")))))),
      change(text("REPLACED")), text("Use size() method directly on string."));
  return R;
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
  RewriteRule Rule = makeRule(
      cxxMemberCallExpr(on(expr(hasType(cxxRecordDecl(
                                    hasName("proto::ProtoCommandLineFlag"))))
                               .bind(Flag)),
                        unless(callee(cxxMethodDecl(hasName("GetProto"))))),
      change(node(Flag), text("EXPR")));

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
  RewriteRule Rule = makeRule(functionDecl(hasName("bad")).bind(Fun),
                              change(name(Fun), text("good")));

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
  testRule(makeRule(declRefExpr(to(functionDecl(hasName("bad")))).bind(Ref),
                    change(name(Ref), text("good"))),
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
  Transformer T(makeRule(declRefExpr(to(functionDecl())).bind(Ref),
                         change(name(Ref), text("good"))),
                consumer());
  T.registerMatchers(&MatchFinder);
  EXPECT_FALSE(rewrite(Input));
}

TEST_F(TransformerTest, NodePartMember) {
  StringRef E = "expr";
  RewriteRule Rule = makeRule(memberExpr(member(hasName("bad"))).bind(E),
                              change(member(E), text("good")));

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
  testRule(makeRule(memberExpr().bind(E), change(member(E), text("good"))),
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
  testRule(makeRule(memberExpr().bind(MemExpr),
                    change(member(MemExpr), text("good"))),
           Input, Expected);
}

TEST_F(TransformerTest, InsertBeforeEdit) {
  std::string Input = R"cc(
    int f() {
      return 7;
    }
  )cc";
  std::string Expected = R"cc(
    int f() {
      int y = 3;
      return 7;
    }
  )cc";

  StringRef Ret = "return";
  testRule(makeRule(returnStmt().bind(Ret),
                    insertBefore(statement(Ret), text("int y = 3;"))),
           Input, Expected);
}

TEST_F(TransformerTest, InsertAfterEdit) {
  std::string Input = R"cc(
    int f() {
      int x = 5;
      return 7;
    }
  )cc";
  std::string Expected = R"cc(
    int f() {
      int x = 5;
      int y = 3;
      return 7;
    }
  )cc";

  StringRef Decl = "decl";
  testRule(makeRule(declStmt().bind(Decl),
                    insertAfter(statement(Decl), text("int y = 3;"))),
           Input, Expected);
}

TEST_F(TransformerTest, RemoveEdit) {
  std::string Input = R"cc(
    int f() {
      int x = 5;
      return 7;
    }
  )cc";
  std::string Expected = R"cc(
    int f() {
      return 7;
    }
  )cc";

  StringRef Decl = "decl";
  testRule(makeRule(declStmt().bind(Decl), remove(statement(Decl))), Input,
           Expected);
}

TEST_F(TransformerTest, MultiChange) {
  std::string Input = R"cc(
    void foo() {
      if (10 > 1.0)
        log(1) << "oh no!";
      else
        log(0) << "ok";
    }
  )cc";
  std::string Expected = R"(
    void foo() {
      if (true) { /* then */ }
      else { /* else */ }
    }
  )";

  StringRef C = "C", T = "T", E = "E";
  testRule(makeRule(ifStmt(hasCondition(expr().bind(C)),
                           hasThen(stmt().bind(T)), hasElse(stmt().bind(E))),
                    {change(node(C), text("true")),
                     change(statement(T), text("{ /* then */ }")),
                     change(statement(E), text("{ /* else */ }"))}),
           Input, Expected);
}

TEST_F(TransformerTest, OrderedRuleUnrelated) {
  StringRef Flag = "flag";
  RewriteRule FlagRule = makeRule(
      cxxMemberCallExpr(on(expr(hasType(cxxRecordDecl(
                                    hasName("proto::ProtoCommandLineFlag"))))
                               .bind(Flag)),
                        unless(callee(cxxMethodDecl(hasName("GetProto"))))),
      change(node(Flag), text("PROTO")));

  std::string Input = R"cc(
    proto::ProtoCommandLineFlag flag;
    int x = flag.foo();
    int y = flag.GetProto().foo();
    int f(string s) { return strlen(s.c_str()); }
  )cc";
  std::string Expected = R"cc(
    proto::ProtoCommandLineFlag flag;
    int x = PROTO.foo();
    int y = flag.GetProto().foo();
    int f(string s) { return REPLACED; }
  )cc";

  testRule(applyFirst({ruleStrlenSize(), FlagRule}), Input, Expected);
}

// Version of ruleStrlenSizeAny that inserts a method with a different name than
// ruleStrlenSize, so we can tell their effect apart.
RewriteRule ruleStrlenSizeDistinct() {
  StringRef S;
  return makeRule(
      callExpr(callee(functionDecl(hasName("strlen"))),
               hasArgument(0, cxxMemberCallExpr(
                                  on(expr().bind(S)),
                                  callee(cxxMethodDecl(hasName("c_str")))))),
      change(text("DISTINCT")));
}

TEST_F(TransformerTest, OrderedRuleRelated) {
  std::string Input = R"cc(
    namespace foo {
    struct mystring {
      char* c_str();
    };
    int f(mystring s) { return strlen(s.c_str()); }
    }  // namespace foo
    int g(string s) { return strlen(s.c_str()); }
  )cc";
  std::string Expected = R"cc(
    namespace foo {
    struct mystring {
      char* c_str();
    };
    int f(mystring s) { return DISTINCT; }
    }  // namespace foo
    int g(string s) { return REPLACED; }
  )cc";

  testRule(applyFirst({ruleStrlenSize(), ruleStrlenSizeDistinct()}), Input,
           Expected);
}

// Change the order of the rules to get a different result.
TEST_F(TransformerTest, OrderedRuleRelatedSwapped) {
  std::string Input = R"cc(
    namespace foo {
    struct mystring {
      char* c_str();
    };
    int f(mystring s) { return strlen(s.c_str()); }
    }  // namespace foo
    int g(string s) { return strlen(s.c_str()); }
  )cc";
  std::string Expected = R"cc(
    namespace foo {
    struct mystring {
      char* c_str();
    };
    int f(mystring s) { return DISTINCT; }
    }  // namespace foo
    int g(string s) { return DISTINCT; }
  )cc";

  testRule(applyFirst({ruleStrlenSizeDistinct(), ruleStrlenSize()}), Input,
           Expected);
}

//
// Negative tests (where we expect no transformation to occur).
//

// Tests for a conflict in edits from a single match for a rule.
TEST_F(TransformerTest, TextGeneratorFailure) {
  std::string Input = "int conflictOneRule() { return 3 + 7; }";
  // Try to change the whole binary-operator expression AND one its operands:
  StringRef O = "O";
  auto AlwaysFail = [](const ast_matchers::MatchFinder::MatchResult &)
      -> llvm::Expected<std::string> {
    return llvm::createStringError(llvm::errc::invalid_argument, "ERROR");
  };
  Transformer T(makeRule(binaryOperator().bind(O), change(node(O), AlwaysFail)),
                consumer());
  T.registerMatchers(&MatchFinder);
  EXPECT_FALSE(rewrite(Input));
  EXPECT_THAT(Changes, IsEmpty());
  EXPECT_EQ(ErrorCount, 1);
}

// Tests for a conflict in edits from a single match for a rule.
TEST_F(TransformerTest, OverlappingEditsInRule) {
  std::string Input = "int conflictOneRule() { return 3 + 7; }";
  // Try to change the whole binary-operator expression AND one its operands:
  StringRef O = "O", L = "L";
  Transformer T(makeRule(binaryOperator(hasLHS(expr().bind(L))).bind(O),
                         {change(node(O), text("DELETE_OP")),
                          change(node(L), text("DELETE_LHS"))}),
                consumer());
  T.registerMatchers(&MatchFinder);
  EXPECT_FALSE(rewrite(Input));
  EXPECT_THAT(Changes, IsEmpty());
  EXPECT_EQ(ErrorCount, 1);
}

// Tests for a conflict in edits across multiple matches (of the same rule).
TEST_F(TransformerTest, OverlappingEditsMultipleMatches) {
  std::string Input = "int conflictOneRule() { return -7; }";
  // Try to change the whole binary-operator expression AND one its operands:
  StringRef E = "E";
  Transformer T(makeRule(expr().bind(E), change(node(E), text("DELETE_EXPR"))),
                consumer());
  T.registerMatchers(&MatchFinder);
  // The rewrite process fails because the changes conflict with each other...
  EXPECT_FALSE(rewrite(Input));
  // ... but two changes were produced.
  EXPECT_EQ(Changes.size(), 2u);
  EXPECT_EQ(ErrorCount, 0);
}

TEST_F(TransformerTest, ErrorOccurredMatchSkipped) {
  // Syntax error in the function body:
  std::string Input = "void errorOccurred() { 3 }";
  Transformer T(makeRule(functionDecl(hasName("errorOccurred")),
                         change(text("DELETED;"))),
                consumer());
  T.registerMatchers(&MatchFinder);
  // The rewrite process itself fails...
  EXPECT_FALSE(rewrite(Input));
  // ... and no changes or errors are produced in the process.
  EXPECT_THAT(Changes, IsEmpty());
  EXPECT_EQ(ErrorCount, 0);
}

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
