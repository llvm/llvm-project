//===- unittest/Tooling/TransformerTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/Transformer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace ast_matchers;
namespace {
using ::clang::transformer::addInclude;
using ::clang::transformer::applyFirst;
using ::clang::transformer::before;
using ::clang::transformer::cat;
using ::clang::transformer::changeTo;
using ::clang::transformer::editList;
using ::clang::transformer::makeRule;
using ::clang::transformer::member;
using ::clang::transformer::name;
using ::clang::transformer::node;
using ::clang::transformer::noEdits;
using ::clang::transformer::remove;
using ::clang::transformer::rewriteDescendants;
using ::clang::transformer::RewriteRule;
using ::clang::transformer::RewriteRuleWith;
using ::clang::transformer::statement;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::ResultOf;
using ::testing::UnorderedElementsAre;

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
    FileContents.emplace_back(std::string(Filename), std::string(Content));
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

  Transformer::ChangeSetConsumer consumer() {
    return [this](Expected<MutableArrayRef<AtomicChange>> C) {
      if (C) {
        Changes.insert(Changes.end(), std::make_move_iterator(C->begin()),
                       std::make_move_iterator(C->end()));
      } else {
        // FIXME: stash this error rather than printing.
        llvm::errs() << "Error generating changes: "
                     << llvm::toString(C.takeError()) << "\n";
        ++ErrorCount;
      }
    };
  }

  auto consumerWithStringMetadata() {
    return [this](Expected<TransformerResult<std::string>> C) {
      if (C) {
        Changes.insert(Changes.end(),
                       std::make_move_iterator(C->Changes.begin()),
                       std::make_move_iterator(C->Changes.end()));
        StringMetadata.push_back(std::move(C->Metadata));
      } else {
        // FIXME: stash this error rather than printing.
        llvm::errs() << "Error generating changes: "
                     << llvm::toString(C.takeError()) << "\n";
        ++ErrorCount;
      }
    };
  }

  void testRule(RewriteRule Rule, StringRef Input, StringRef Expected) {
    Transformers.push_back(
        std::make_unique<Transformer>(std::move(Rule), consumer()));
    Transformers.back()->registerMatchers(&MatchFinder);
    compareSnippets(Expected, rewrite(Input));
  }

  void testRule(RewriteRuleWith<std::string> Rule, StringRef Input,
                StringRef Expected) {
    Transformers.push_back(std::make_unique<Transformer>(
        std::move(Rule), consumerWithStringMetadata()));
    Transformers.back()->registerMatchers(&MatchFinder);
    compareSnippets(Expected, rewrite(Input));
  }

  void testRuleFailure(RewriteRule Rule, StringRef Input) {
    Transformers.push_back(
        std::make_unique<Transformer>(std::move(Rule), consumer()));
    Transformers.back()->registerMatchers(&MatchFinder);
    ASSERT_FALSE(rewrite(Input)) << "Expected failure to rewrite code";
  }

  void testRuleFailure(RewriteRuleWith<std::string> Rule, StringRef Input) {
    Transformers.push_back(std::make_unique<Transformer>(
        std::move(Rule), consumerWithStringMetadata()));
    Transformers.back()->registerMatchers(&MatchFinder);
    ASSERT_FALSE(rewrite(Input)) << "Expected failure to rewrite code";
  }

  // Transformers are referenced by MatchFinder.
  std::vector<std::unique_ptr<Transformer>> Transformers;
  clang::ast_matchers::MatchFinder MatchFinder;
  // Records whether any errors occurred in individual changes.
  int ErrorCount = 0;
  AtomicChanges Changes;
  std::vector<std::string> StringMetadata;

private:
  FileContentMappings FileContents = {{"header.h", ""}};
};

class TransformerTest : public ClangRefactoringTestBase {
protected:
  TransformerTest() { appendToHeader(KHeaderContents); }
};

// Given string s, change strlen($s.c_str()) to REPLACED.
static RewriteRuleWith<std::string> ruleStrlenSize() {
  StringRef StringExpr = "strexpr";
  auto StringType = namedDecl(hasAnyName("::basic_string", "::string"));
  auto R = makeRule(
      callExpr(callee(functionDecl(hasName("strlen"))),
               hasArgument(0, cxxMemberCallExpr(
                                  on(expr(hasType(isOrPointsTo(StringType)))
                                         .bind(StringExpr)),
                                  callee(cxxMethodDecl(hasName("c_str")))))),
      changeTo(cat("REPLACED")), cat("Use size() method directly on string."));
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

// Tests replacing an expression.
TEST_F(TransformerTest, Flag) {
  StringRef Flag = "flag";
  RewriteRule Rule = makeRule(
      cxxMemberCallExpr(on(expr(hasType(cxxRecordDecl(
                                    hasName("proto::ProtoCommandLineFlag"))))
                               .bind(Flag)),
                        unless(callee(cxxMethodDecl(hasName("GetProto"))))),
      changeTo(node(std::string(Flag)), cat("EXPR")));

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

TEST_F(TransformerTest, AddIncludeQuoted) {
  RewriteRule Rule =
      makeRule(callExpr(callee(functionDecl(hasName("f")))),
               {addInclude("clang/OtherLib.h"), changeTo(cat("other()"))});

  std::string Input = R"cc(
    int f(int x);
    int h(int x) { return f(x); }
  )cc";
  std::string Expected = R"cc(#include "clang/OtherLib.h"

    int f(int x);
    int h(int x) { return other(); }
  )cc";

  testRule(Rule, Input, Expected);
}

TEST_F(TransformerTest, AddIncludeAngled) {
  RewriteRule Rule = makeRule(
      callExpr(callee(functionDecl(hasName("f")))),
      {addInclude("clang/OtherLib.h", transformer::IncludeFormat::Angled),
       changeTo(cat("other()"))});

  std::string Input = R"cc(
    int f(int x);
    int h(int x) { return f(x); }
  )cc";
  std::string Expected = R"cc(#include <clang/OtherLib.h>

    int f(int x);
    int h(int x) { return other(); }
  )cc";

  testRule(Rule, Input, Expected);
}

TEST_F(TransformerTest, AddIncludeQuotedForRule) {
  RewriteRule Rule = makeRule(callExpr(callee(functionDecl(hasName("f")))),
                              changeTo(cat("other()")));
  addInclude(Rule, "clang/OtherLib.h");

  std::string Input = R"cc(
    int f(int x);
    int h(int x) { return f(x); }
  )cc";
  std::string Expected = R"cc(#include "clang/OtherLib.h"

    int f(int x);
    int h(int x) { return other(); }
  )cc";

  testRule(Rule, Input, Expected);
}

TEST_F(TransformerTest, AddIncludeAngledForRule) {
  RewriteRule Rule = makeRule(callExpr(callee(functionDecl(hasName("f")))),
                              changeTo(cat("other()")));
  addInclude(Rule, "clang/OtherLib.h", transformer::IncludeFormat::Angled);

  std::string Input = R"cc(
    int f(int x);
    int h(int x) { return f(x); }
  )cc";
  std::string Expected = R"cc(#include <clang/OtherLib.h>

    int f(int x);
    int h(int x) { return other(); }
  )cc";

  testRule(Rule, Input, Expected);
}

TEST_F(TransformerTest, NodePartNameNamedDecl) {
  StringRef Fun = "fun";
  RewriteRule Rule = makeRule(functionDecl(hasName("bad")).bind(Fun),
                              changeTo(name(std::string(Fun)), cat("good")));

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
                    changeTo(name(std::string(Ref)), cat("good"))),
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
                         changeTo(name(std::string(Ref)), cat("good"))),
                consumer());
  T.registerMatchers(&MatchFinder);
  EXPECT_FALSE(rewrite(Input));
}

TEST_F(TransformerTest, NodePartMember) {
  StringRef E = "expr";
  RewriteRule Rule =
      makeRule(memberExpr(clang::ast_matchers::member(hasName("bad"))).bind(E),
               changeTo(member(std::string(E)), cat("good")));

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
  testRule(makeRule(memberExpr().bind(E),
                    changeTo(member(std::string(E)), cat("good"))),
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
                    changeTo(member(std::string(MemExpr)), cat("good"))),
           Input, Expected);
}

TEST_F(TransformerTest, NoEdits) {
  using transformer::noEdits;
  std::string Input = "int f(int x) { return x; }";
  testRule(makeRule(returnStmt().bind("return"), noEdits()), Input, Input);
}

TEST_F(TransformerTest, NoopEdit) {
  using transformer::node;
  using transformer::noopEdit;
  std::string Input = "int f(int x) { return x; }";
  testRule(makeRule(returnStmt().bind("return"), noopEdit(node("return"))),
           Input, Input);
}

TEST_F(TransformerTest, IfBound2Args) {
  using transformer::ifBound;
  std::string Input = "int f(int x) { return x; }";
  std::string Expected = "int f(int x) { CHANGE; }";
  testRule(makeRule(returnStmt().bind("return"),
                    ifBound("return", changeTo(cat("CHANGE;")))),
           Input, Expected);
}

TEST_F(TransformerTest, IfBound3Args) {
  using transformer::ifBound;
  std::string Input = "int f(int x) { return x; }";
  std::string Expected = "int f(int x) { CHANGE; }";
  testRule(makeRule(returnStmt().bind("return"),
                    ifBound("nothing", changeTo(cat("ERROR")),
                            changeTo(cat("CHANGE;")))),
           Input, Expected);
}

TEST_F(TransformerTest, ShrinkTo) {
  using transformer::shrinkTo;
  std::string Input = "int f(int x) { return x; }";
  std::string Expected = "return x;";
  testRule(makeRule(functionDecl(hasDescendant(returnStmt().bind("return")))
                        .bind("function"),
                    shrinkTo(node("function"), node("return"))),
           Input, Expected);
}

// Rewrite various Stmts inside a Decl.
TEST_F(TransformerTest, RewriteDescendantsDeclChangeStmt) {
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  std::string Expected =
      "int f(int x) { int y = 3; { int z = 3 * 3; } return 3; }";
  auto InlineX =
      makeRule(declRefExpr(to(varDecl(hasName("x")))), changeTo(cat("3")));
  testRule(makeRule(functionDecl(hasName("f")).bind("fun"),
                    rewriteDescendants("fun", InlineX)),
           Input, Expected);
}

// Rewrite various TypeLocs inside a Decl.
TEST_F(TransformerTest, RewriteDescendantsDeclChangeTypeLoc) {
  std::string Input = "int f(int *x) { return *x; }";
  std::string Expected = "char f(char *x) { return *x; }";
  auto IntToChar = makeRule(typeLoc(loc(qualType(isInteger(), builtinType()))),
                            changeTo(cat("char")));
  testRule(makeRule(functionDecl(hasName("f")).bind("fun"),
                    rewriteDescendants("fun", IntToChar)),
           Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsStmt) {
  // Add an unrelated definition to the header that also has a variable named
  // "x", to test that the rewrite is limited to the scope we intend.
  appendToHeader(R"cc(int g(int x) { return x; })cc");
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  std::string Expected =
      "int f(int x) { int y = 3; { int z = 3 * 3; } return 3; }";
  auto InlineX =
      makeRule(declRefExpr(to(varDecl(hasName("x")))), changeTo(cat("3")));
  testRule(makeRule(functionDecl(hasName("f"), hasBody(stmt().bind("body"))),
                    rewriteDescendants("body", InlineX)),
           Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsStmtWithAdditionalChange) {
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  std::string Expected =
      "int newName(int x) { int y = 3; { int z = 3 * 3; } return 3; }";
  auto InlineX =
      makeRule(declRefExpr(to(varDecl(hasName("x")))), changeTo(cat("3")));
  testRule(
      makeRule(
          functionDecl(hasName("f"), hasBody(stmt().bind("body"))).bind("f"),
          flatten(changeTo(name("f"), cat("newName")),
                  rewriteDescendants("body", InlineX))),
      Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsTypeLoc) {
  std::string Input = "int f(int *x) { return *x; }";
  std::string Expected = "int f(char *x) { return *x; }";
  auto IntToChar =
      makeRule(typeLoc(loc(qualType(isInteger(), builtinType()))).bind("loc"),
               changeTo(cat("char")));
  testRule(
      makeRule(functionDecl(hasName("f"),
                            hasParameter(0, varDecl(hasTypeLoc(
                                                typeLoc().bind("parmType"))))),
               rewriteDescendants("parmType", IntToChar)),
      Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsReferToParentBinding) {
  std::string Input =
      "int f(int p) { int y = p; { int z = p * p; } return p; }";
  std::string Expected =
      "int f(int p) { int y = 3; { int z = 3 * 3; } return 3; }";
  std::string VarId = "var";
  auto InlineVar = makeRule(declRefExpr(to(varDecl(equalsBoundNode(VarId)))),
                            changeTo(cat("3")));
  testRule(makeRule(functionDecl(hasName("f"),
                                 hasParameter(0, varDecl().bind(VarId)))
                        .bind("fun"),
                    rewriteDescendants("fun", InlineVar)),
           Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsUnboundNode) {
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  auto InlineX =
      makeRule(declRefExpr(to(varDecl(hasName("x")))), changeTo(cat("3")));
  Transformer T(makeRule(functionDecl(hasName("f")),
                         rewriteDescendants("UNBOUND", InlineX)),
                consumer());
  T.registerMatchers(&MatchFinder);
  EXPECT_FALSE(rewrite(Input));
  EXPECT_THAT(Changes, IsEmpty());
  EXPECT_EQ(ErrorCount, 1);
}

TEST_F(TransformerTest, RewriteDescendantsInvalidNodeType) {
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  auto IntToChar =
      makeRule(qualType(isInteger(), builtinType()), changeTo(cat("char")));
  Transformer T(
      makeRule(functionDecl(
                   hasName("f"),
                   hasParameter(0, varDecl(hasType(qualType().bind("type"))))),
               rewriteDescendants("type", IntToChar)),
      consumer());
  T.registerMatchers(&MatchFinder);
  EXPECT_FALSE(rewrite(Input));
  EXPECT_THAT(Changes, IsEmpty());
  EXPECT_EQ(ErrorCount, 1);
}

//
// We include one test per typed overload. We don't test extensively since that
// is already covered by the tests above.
//

TEST_F(TransformerTest, RewriteDescendantsTypedStmt) {
  // Add an unrelated definition to the header that also has a variable named
  // "x", to test that the rewrite is limited to the scope we intend.
  appendToHeader(R"cc(int g(int x) { return x; })cc");
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  std::string Expected =
      "int f(int x) { int y = 3; { int z = 3 * 3; } return 3; }";
  auto InlineX =
      makeRule(declRefExpr(to(varDecl(hasName("x")))), changeTo(cat("3")));
  testRule(makeRule(functionDecl(hasName("f"), hasBody(stmt().bind("body"))),
                    [&InlineX](const MatchFinder::MatchResult &R) {
                      const auto *Node = R.Nodes.getNodeAs<Stmt>("body");
                      assert(Node != nullptr && "body must be bound");
                      return transformer::detail::rewriteDescendants(
                          *Node, InlineX, R);
                    }),
           Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsTypedDecl) {
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  std::string Expected =
      "int f(int x) { int y = 3; { int z = 3 * 3; } return 3; }";
  auto InlineX =
      makeRule(declRefExpr(to(varDecl(hasName("x")))), changeTo(cat("3")));
  testRule(makeRule(functionDecl(hasName("f")).bind("fun"),
                    [&InlineX](const MatchFinder::MatchResult &R) {
                      const auto *Node = R.Nodes.getNodeAs<Decl>("fun");
                      assert(Node != nullptr && "fun must be bound");
                      return transformer::detail::rewriteDescendants(
                          *Node, InlineX, R);
                    }),
           Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsTypedTypeLoc) {
  std::string Input = "int f(int *x) { return *x; }";
  std::string Expected = "int f(char *x) { return *x; }";
  auto IntToChar =
      makeRule(typeLoc(loc(qualType(isInteger(), builtinType()))).bind("loc"),
               changeTo(cat("char")));
  testRule(
      makeRule(
          functionDecl(
              hasName("f"),
              hasParameter(0, varDecl(hasTypeLoc(typeLoc().bind("parmType"))))),
          [&IntToChar](const MatchFinder::MatchResult &R) {
            const auto *Node = R.Nodes.getNodeAs<TypeLoc>("parmType");
            assert(Node != nullptr && "parmType must be bound");
            return transformer::detail::rewriteDescendants(*Node, IntToChar, R);
          }),
      Input, Expected);
}

TEST_F(TransformerTest, RewriteDescendantsTypedDynTyped) {
  // Add an unrelated definition to the header that also has a variable named
  // "x", to test that the rewrite is limited to the scope we intend.
  appendToHeader(R"cc(int g(int x) { return x; })cc");
  std::string Input =
      "int f(int x) { int y = x; { int z = x * x; } return x; }";
  std::string Expected =
      "int f(int x) { int y = 3; { int z = 3 * 3; } return 3; }";
  auto InlineX =
      makeRule(declRefExpr(to(varDecl(hasName("x")))), changeTo(cat("3")));
  testRule(
      makeRule(functionDecl(hasName("f"), hasBody(stmt().bind("body"))),
               [&InlineX](const MatchFinder::MatchResult &R) {
                 auto It = R.Nodes.getMap().find("body");
                 assert(It != R.Nodes.getMap().end() && "body must be bound");
                 return transformer::detail::rewriteDescendants(It->second,
                                                                InlineX, R);
               }),
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
  testRule(
      makeRule(returnStmt().bind(Ret),
               insertBefore(statement(std::string(Ret)), cat("int y = 3;"))),
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
  testRule(
      makeRule(declStmt().bind(Decl),
               insertAfter(statement(std::string(Decl)), cat("int y = 3;"))),
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
  testRule(
      makeRule(declStmt().bind(Decl), remove(statement(std::string(Decl)))),
      Input, Expected);
}

TEST_F(TransformerTest, WithMetadata) {
  auto makeMetadata = [](const MatchFinder::MatchResult &R) -> llvm::Any {
    int N =
        R.Nodes.getNodeAs<IntegerLiteral>("int")->getValue().getLimitedValue();
    return N;
  };

  std::string Input = R"cc(
    int f() {
      int x = 5;
      return 7;
    }
  )cc";

  Transformer T(
      makeRule(
          declStmt(containsDeclaration(0, varDecl(hasInitializer(
                                              integerLiteral().bind("int")))))
              .bind("decl"),
          withMetadata(remove(statement(std::string("decl"))), makeMetadata)),
      consumer());
  T.registerMatchers(&MatchFinder);
  auto Factory = newFrontendActionFactory(&MatchFinder);
  EXPECT_TRUE(runToolOnCodeWithArgs(
      Factory->create(), Input, std::vector<std::string>(), "input.cc",
      "clang-tool", std::make_shared<PCHContainerOperations>(), {}));
  ASSERT_EQ(Changes.size(), 1u);
  const llvm::Any &Metadata = Changes[0].getMetadata();
  ASSERT_TRUE(llvm::any_isa<int>(Metadata));
  EXPECT_THAT(llvm::any_cast<int>(Metadata), 5);
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
  testRule(
      makeRule(ifStmt(hasCondition(expr().bind(C)), hasThen(stmt().bind(T)),
                      hasElse(stmt().bind(E))),
               {changeTo(node(std::string(C)), cat("true")),
                changeTo(statement(std::string(T)), cat("{ /* then */ }")),
                changeTo(statement(std::string(E)), cat("{ /* else */ }"))}),
      Input, Expected);
}

TEST_F(TransformerTest, EditList) {
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
                    editList({changeTo(node(std::string(C)), cat("true")),
                              changeTo(statement(std::string(T)),
                                       cat("{ /* then */ }")),
                              changeTo(statement(std::string(E)),
                                       cat("{ /* else */ }"))})),
           Input, Expected);
}

TEST_F(TransformerTest, Flatten) {
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
  testRule(
      makeRule(
          ifStmt(hasCondition(expr().bind(C)), hasThen(stmt().bind(T)),
                 hasElse(stmt().bind(E))),
          flatten(changeTo(node(std::string(C)), cat("true")),
                  changeTo(statement(std::string(T)), cat("{ /* then */ }")),
                  changeTo(statement(std::string(E)), cat("{ /* else */ }")))),
      Input, Expected);
}

TEST_F(TransformerTest, FlattenWithMixedArgs) {
  using clang::transformer::editList;
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
                    flatten(changeTo(node(std::string(C)), cat("true")),
                            edit(changeTo(statement(std::string(T)),
                                          cat("{ /* then */ }"))),
                            editList({changeTo(statement(std::string(E)),
                                               cat("{ /* else */ }"))}))),
           Input, Expected);
}

TEST_F(TransformerTest, OrderedRuleUnrelated) {
  StringRef Flag = "flag";
  RewriteRuleWith<std::string> FlagRule = makeRule(
      cxxMemberCallExpr(on(expr(hasType(cxxRecordDecl(
                                    hasName("proto::ProtoCommandLineFlag"))))
                               .bind(Flag)),
                        unless(callee(cxxMethodDecl(hasName("GetProto"))))),
      changeTo(node(std::string(Flag)), cat("PROTO")), cat(""));

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

TEST_F(TransformerTest, OrderedRuleRelated) {
  std::string Input = R"cc(
    void f1();
    void f2();
    void call_f1() { f1(); }
    void call_f2() { f2(); }
  )cc";
  std::string Expected = R"cc(
    void f1();
    void f2();
    void call_f1() { REPLACE_F1; }
    void call_f2() { REPLACE_F1_OR_F2; }
  )cc";

  RewriteRule ReplaceF1 =
      makeRule(callExpr(callee(functionDecl(hasName("f1")))),
               changeTo(cat("REPLACE_F1")));
  RewriteRule ReplaceF1OrF2 =
      makeRule(callExpr(callee(functionDecl(hasAnyName("f1", "f2")))),
               changeTo(cat("REPLACE_F1_OR_F2")));
  testRule(applyFirst({ReplaceF1, ReplaceF1OrF2}), Input, Expected);
}

// Change the order of the rules to get a different result. When `ReplaceF1OrF2`
// comes first, it applies for both uses, so `ReplaceF1` never applies.
TEST_F(TransformerTest, OrderedRuleRelatedSwapped) {
  std::string Input = R"cc(
    void f1();
    void f2();
    void call_f1() { f1(); }
    void call_f2() { f2(); }
  )cc";
  std::string Expected = R"cc(
    void f1();
    void f2();
    void call_f1() { REPLACE_F1_OR_F2; }
    void call_f2() { REPLACE_F1_OR_F2; }
  )cc";

  RewriteRule ReplaceF1 =
      makeRule(callExpr(callee(functionDecl(hasName("f1")))),
               changeTo(cat("REPLACE_F1")));
  RewriteRule ReplaceF1OrF2 =
      makeRule(callExpr(callee(functionDecl(hasAnyName("f1", "f2")))),
               changeTo(cat("REPLACE_F1_OR_F2")));
  testRule(applyFirst({ReplaceF1OrF2, ReplaceF1}), Input, Expected);
}

// Verify that a set of rules whose matchers have different base kinds works
// properly, including that `applyFirst` produces multiple matchers.  We test
// two different kinds of rules: Expr and Decl. We place the Decl rule in the
// middle to test that `buildMatchers` works even when the kinds aren't grouped
// together.
TEST_F(TransformerTest, OrderedRuleMultipleKinds) {
  std::string Input = R"cc(
    void f1();
    void f2();
    void call_f1() { f1(); }
    void call_f2() { f2(); }
  )cc";
  std::string Expected = R"cc(
    void f1();
    void DECL_RULE();
    void call_f1() { REPLACE_F1; }
    void call_f2() { REPLACE_F1_OR_F2; }
  )cc";

  RewriteRule ReplaceF1 =
      makeRule(callExpr(callee(functionDecl(hasName("f1")))),
               changeTo(cat("REPLACE_F1")));
  RewriteRule ReplaceF1OrF2 =
      makeRule(callExpr(callee(functionDecl(hasAnyName("f1", "f2")))),
               changeTo(cat("REPLACE_F1_OR_F2")));
  RewriteRule DeclRule = makeRule(functionDecl(hasName("f2")).bind("fun"),
                                  changeTo(name("fun"), cat("DECL_RULE")));

  RewriteRule Rule = applyFirst({ReplaceF1, DeclRule, ReplaceF1OrF2});
  EXPECT_EQ(transformer::detail::buildMatchers(Rule).size(), 2UL);
  testRule(Rule, Input, Expected);
}

// Verifies that a rule with a top-level matcher for an implicit node (like
// `implicitCastExpr`) works correctly -- the implicit nodes are not skipped.
TEST_F(TransformerTest, OrderedRuleImplicitMatched) {
  std::string Input = R"cc(
    void f1();
    int f2();
    void call_f1() { f1(); }
    float call_f2() { return f2(); }
  )cc";
  std::string Expected = R"cc(
    void f1();
    int f2();
    void call_f1() { REPLACE_F1; }
    float call_f2() { return REPLACE_F2; }
  )cc";

  RewriteRule ReplaceF1 =
      makeRule(callExpr(callee(functionDecl(hasName("f1")))),
               changeTo(cat("REPLACE_F1")));
  RewriteRule ReplaceF2 =
      makeRule(implicitCastExpr(hasSourceExpression(callExpr())),
               changeTo(cat("REPLACE_F2")));
  testRule(applyFirst({ReplaceF1, ReplaceF2}), Input, Expected);
}

//
// Negative tests (where we expect no transformation to occur).
//

// Tests for a conflict in edits from a single match for a rule.
TEST_F(TransformerTest, TextGeneratorFailure) {
  std::string Input = "int conflictOneRule() { return 3 + 7; }";
  // Try to change the whole binary-operator expression AND one its operands:
  StringRef O = "O";
  class AlwaysFail : public transformer::MatchComputation<std::string> {
    llvm::Error eval(const ast_matchers::MatchFinder::MatchResult &,
                     std::string *) const override {
      return llvm::createStringError(llvm::errc::invalid_argument, "ERROR");
    }
    std::string toString() const override { return "AlwaysFail"; }
  };
  Transformer T(
      makeRule(binaryOperator().bind(O),
               changeTo(node(std::string(O)), std::make_shared<AlwaysFail>())),
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
                         {changeTo(node(std::string(O)), cat("DELETE_OP")),
                          changeTo(node(std::string(L)), cat("DELETE_LHS"))}),
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
  Transformer T(makeRule(expr().bind(E),
                         changeTo(node(std::string(E)), cat("DELETE_EXPR"))),
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
                         changeTo(cat("DELETED;"))),
                consumer());
  T.registerMatchers(&MatchFinder);
  // The rewrite process itself fails...
  EXPECT_FALSE(rewrite(Input));
  // ... and no changes or errors are produced in the process.
  EXPECT_THAT(Changes, IsEmpty());
  EXPECT_EQ(ErrorCount, 0);
}

TEST_F(TransformerTest, ImplicitNodes_ConstructorDecl) {

  std::string OtherStructPrefix = R"cpp(
struct Other {
)cpp";
  std::string OtherStructSuffix = "};";

  std::string CopyableStructName = "struct Copyable";
  std::string BrokenStructName = "struct explicit Copyable";

  std::string CodeSuffix = R"cpp(
{
    Other m_i;
    Copyable();
};
)cpp";

  std::string CopyCtor = "Other(const Other&) = default;";
  std::string ExplicitCopyCtor = "explicit Other(const Other&) = default;";
  std::string BrokenExplicitCopyCtor =
      "explicit explicit explicit Other(const Other&) = default;";

  std::string RewriteInput = OtherStructPrefix + CopyCtor + OtherStructSuffix +
                             CopyableStructName + CodeSuffix;
  std::string ExpectedRewriteOutput = OtherStructPrefix + ExplicitCopyCtor +
                                      OtherStructSuffix + CopyableStructName +
                                      CodeSuffix;
  std::string BrokenRewriteOutput = OtherStructPrefix + BrokenExplicitCopyCtor +
                                    OtherStructSuffix + BrokenStructName +
                                    CodeSuffix;

  auto MatchedRecord =
      cxxConstructorDecl(isCopyConstructor()).bind("copyConstructor");

  auto RewriteRule =
      changeTo(before(node("copyConstructor")), cat("explicit "));

  testRule(makeRule(traverse(TK_IgnoreUnlessSpelledInSource, MatchedRecord),
                    RewriteRule),
           RewriteInput, ExpectedRewriteOutput);

  testRule(makeRule(traverse(TK_AsIs, MatchedRecord), RewriteRule),
           RewriteInput, BrokenRewriteOutput);
}

TEST_F(TransformerTest, ImplicitNodes_RangeFor) {

  std::string CodePrefix = R"cpp(
struct Container
{
    int* begin() const;
    int* end() const;
    int* cbegin() const;
    int* cend() const;
};

void foo()
{
  const Container c;
)cpp";

  std::string BeginCallBefore = "  c.begin();";
  std::string BeginCallAfter = "  c.cbegin();";

  std::string ForLoop = "for (auto i : c)";
  std::string BrokenForLoop = "for (auto i :.cbegin() c)";

  std::string CodeSuffix = R"cpp(
  {
  }
}
)cpp";

  std::string RewriteInput =
      CodePrefix + BeginCallBefore + ForLoop + CodeSuffix;
  std::string ExpectedRewriteOutput =
      CodePrefix + BeginCallAfter + ForLoop + CodeSuffix;
  std::string BrokenRewriteOutput =
      CodePrefix + BeginCallAfter + BrokenForLoop + CodeSuffix;

  auto MatchedRecord =
      cxxMemberCallExpr(on(expr(hasType(qualType(isConstQualified(),
                                                 hasDeclaration(cxxRecordDecl(
                                                     hasName("Container"))))))
                               .bind("callTarget")),
                        callee(cxxMethodDecl(hasName("begin"))))
          .bind("constBeginCall");

  auto RewriteRule =
      changeTo(node("constBeginCall"), cat(name("callTarget"), ".cbegin()"));

  testRule(makeRule(traverse(TK_IgnoreUnlessSpelledInSource, MatchedRecord),
                    RewriteRule),
           RewriteInput, ExpectedRewriteOutput);

  testRule(makeRule(traverse(TK_AsIs, MatchedRecord), RewriteRule),
           RewriteInput, BrokenRewriteOutput);
}

TEST_F(TransformerTest, ImplicitNodes_ForStmt) {

  std::string CodePrefix = R"cpp(
struct NonTrivial {
    NonTrivial() {}
    NonTrivial(NonTrivial&) {}
    NonTrivial& operator=(NonTrivial const&) { return *this; }

    ~NonTrivial() {}
};

struct ContainsArray {
    NonTrivial arr[2];
    ContainsArray& operator=(ContainsArray const&) = default;
};

void testIt()
{
    ContainsArray ca1;
    ContainsArray ca2;
    ca2 = ca1;
)cpp";

  auto CodeSuffix = "}";

  auto LoopBody = R"cpp(
    {

    }
)cpp";

  auto RawLoop = "for (auto i = 0; i != 5; ++i)";

  auto RangeLoop = "for (auto i : boost::irange(5))";

  // Expect to rewrite the raw loop to the ranged loop.
  // This works in TK_IgnoreUnlessSpelledInSource mode, but TK_AsIs
  // mode also matches the hidden for loop generated in the copy assignment
  // operator of ContainsArray. Transformer then fails to transform the code at
  // all.

  auto RewriteInput =
      CodePrefix + RawLoop + LoopBody + RawLoop + LoopBody + CodeSuffix;

  auto RewriteOutput =
      CodePrefix + RangeLoop + LoopBody + RangeLoop + LoopBody + CodeSuffix;

  auto MatchedLoop = forStmt(
      has(declStmt(hasSingleDecl(
          varDecl(hasInitializer(integerLiteral(equals(0)))).bind("loopVar")))),
      has(binaryOperator(hasOperatorName("!="),
                         hasLHS(ignoringImplicit(declRefExpr(
                             to(varDecl(equalsBoundNode("loopVar")))))),
                         hasRHS(expr().bind("upperBoundExpr")))),
      has(unaryOperator(hasOperatorName("++"),
                        hasUnaryOperand(declRefExpr(
                            to(varDecl(equalsBoundNode("loopVar"))))))
              .bind("incrementOp")));

  auto RewriteRule =
      changeTo(transformer::enclose(node("loopVar"), node("incrementOp")),
               cat("auto ", name("loopVar"), " : boost::irange(",
                   node("upperBoundExpr"), ")"));

  testRule(makeRule(traverse(TK_IgnoreUnlessSpelledInSource, MatchedLoop),
                    RewriteRule),
           RewriteInput, RewriteOutput);

  testRuleFailure(makeRule(traverse(TK_AsIs, MatchedLoop), RewriteRule),
                  RewriteInput);
}

TEST_F(TransformerTest, ImplicitNodes_ForStmt2) {

  std::string CodePrefix = R"cpp(
struct NonTrivial {
    NonTrivial() {}
    NonTrivial(NonTrivial&) {}
    NonTrivial& operator=(NonTrivial const&) { return *this; }

    ~NonTrivial() {}
};

struct ContainsArray {
    NonTrivial arr[2];
    ContainsArray& operator=(ContainsArray const&) = default;
};

void testIt()
{
    ContainsArray ca1;
    ContainsArray ca2;
    ca2 = ca1;
)cpp";

  auto CodeSuffix = "}";

  auto LoopBody = R"cpp(
    {

    }
)cpp";

  auto RawLoop = "for (auto i = 0; i != 5; ++i)";

  auto RangeLoop = "for (auto i : boost::irange(5))";

  // Expect to rewrite the raw loop to the ranged loop.
  // This works in TK_IgnoreUnlessSpelledInSource mode, but TK_AsIs
  // mode also matches the hidden for loop generated in the copy assignment
  // operator of ContainsArray. Transformer then fails to transform the code at
  // all.

  auto RewriteInput =
      CodePrefix + RawLoop + LoopBody + RawLoop + LoopBody + CodeSuffix;

  auto RewriteOutput =
      CodePrefix + RangeLoop + LoopBody + RangeLoop + LoopBody + CodeSuffix;
  auto MatchedLoop = forStmt(
      hasLoopInit(declStmt(hasSingleDecl(
          varDecl(hasInitializer(integerLiteral(equals(0)))).bind("loopVar")))),
      hasCondition(binaryOperator(hasOperatorName("!="),
                                  hasLHS(ignoringImplicit(declRefExpr(to(
                                      varDecl(equalsBoundNode("loopVar")))))),
                                  hasRHS(expr().bind("upperBoundExpr")))),
      hasIncrement(unaryOperator(hasOperatorName("++"),
                                 hasUnaryOperand(declRefExpr(
                                     to(varDecl(equalsBoundNode("loopVar"))))))
                       .bind("incrementOp")));

  auto RewriteRule =
      changeTo(transformer::enclose(node("loopVar"), node("incrementOp")),
               cat("auto ", name("loopVar"), " : boost::irange(",
                   node("upperBoundExpr"), ")"));

  testRule(makeRule(traverse(TK_IgnoreUnlessSpelledInSource, MatchedLoop),
                    RewriteRule),
           RewriteInput, RewriteOutput);

  testRuleFailure(makeRule(traverse(TK_AsIs, MatchedLoop), RewriteRule),
                  RewriteInput);
}

TEST_F(TransformerTest, TemplateInstantiation) {

  std::string NonTemplatesInput = R"cpp(
struct S {
  int m_i;
};
)cpp";
  std::string NonTemplatesExpected = R"cpp(
struct S {
  safe_int m_i;
};
)cpp";

  std::string TemplatesInput = R"cpp(
template<typename T>
struct TemplStruct {
  TemplStruct() {}
  ~TemplStruct() {}

private:
  T m_t;
};

void instantiate()
{
  TemplStruct<int> ti;
}
)cpp";

  auto MatchedField = fieldDecl(hasType(asString("int"))).bind("theField");

  // Changes the 'int' in 'S', but not the 'T' in 'TemplStruct':
  testRule(makeRule(traverse(TK_IgnoreUnlessSpelledInSource, MatchedField),
                    changeTo(cat("safe_int ", name("theField"), ";"))),
           NonTemplatesInput + TemplatesInput,
           NonTemplatesExpected + TemplatesInput);

  // In AsIs mode, template instantiations are modified, which is
  // often not desired:

  std::string IncorrectTemplatesExpected = R"cpp(
template<typename T>
struct TemplStruct {
  TemplStruct() {}
  ~TemplStruct() {}

private:
  safe_int m_t;
};

void instantiate()
{
  TemplStruct<int> ti;
}
)cpp";

  // Changes the 'int' in 'S', and (incorrectly) the 'T' in 'TemplStruct':
  testRule(makeRule(traverse(TK_AsIs, MatchedField),
                    changeTo(cat("safe_int ", name("theField"), ";"))),

           NonTemplatesInput + TemplatesInput,
           NonTemplatesExpected + IncorrectTemplatesExpected);
}

// Transformation of macro source text when the change encompasses the entirety
// of the expanded text.
TEST_F(TransformerTest, SimpleMacro) {
  std::string Input = R"cc(
#define ZERO 0
    int f(string s) { return ZERO; }
  )cc";
  std::string Expected = R"cc(
#define ZERO 0
    int f(string s) { return 999; }
  )cc";

  StringRef zero = "zero";
  RewriteRule R = makeRule(integerLiteral(equals(0)).bind(zero),
                           changeTo(node(std::string(zero)), cat("999")));
  testRule(R, Input, Expected);
}

// Transformation of macro source text when the change encompasses the entirety
// of the expanded text, for the case of function-style macros.
TEST_F(TransformerTest, FunctionMacro) {
  std::string Input = R"cc(
#define MACRO(str) strlen((str).c_str())
    int f(string s) { return MACRO(s); }
  )cc";
  std::string Expected = R"cc(
#define MACRO(str) strlen((str).c_str())
    int f(string s) { return REPLACED; }
  )cc";

  testRule(ruleStrlenSize(), Input, Expected);
}

// Tests that expressions in macro arguments can be rewritten.
TEST_F(TransformerTest, MacroArg) {
  std::string Input = R"cc(
#define PLUS(e) e + 1
    int f(string s) { return PLUS(strlen(s.c_str())); }
  )cc";
  std::string Expected = R"cc(
#define PLUS(e) e + 1
    int f(string s) { return PLUS(REPLACED); }
  )cc";

  testRule(ruleStrlenSize(), Input, Expected);
}

// Tests that expressions in macro arguments can be rewritten, even when the
// macro call occurs inside another macro's definition.
TEST_F(TransformerTest, MacroArgInMacroDef) {
  std::string Input = R"cc(
#define NESTED(e) e
#define MACRO(str) NESTED(strlen((str).c_str()))
    int f(string s) { return MACRO(s); }
  )cc";
  std::string Expected = R"cc(
#define NESTED(e) e
#define MACRO(str) NESTED(strlen((str).c_str()))
    int f(string s) { return REPLACED; }
  )cc";

  testRule(ruleStrlenSize(), Input, Expected);
}

// Tests the corner case of the identity macro, specifically that it is
// discarded in the rewrite rather than preserved (like PLUS is preserved in the
// previous test).  This behavior is of dubious value (and marked with a FIXME
// in the code), but we test it to verify (and demonstrate) how this case is
// handled.
TEST_F(TransformerTest, IdentityMacro) {
  std::string Input = R"cc(
#define ID(e) e
    int f(string s) { return ID(strlen(s.c_str())); }
  )cc";
  std::string Expected = R"cc(
#define ID(e) e
    int f(string s) { return REPLACED; }
  )cc";

  testRule(ruleStrlenSize(), Input, Expected);
}

// Tests that two changes in a single macro expansion do not lead to conflicts
// in applying the changes.
TEST_F(TransformerTest, TwoChangesInOneMacroExpansion) {
  std::string Input = R"cc(
#define PLUS(a,b) (a) + (b)
    int f() { return PLUS(3, 4); }
  )cc";
  std::string Expected = R"cc(
#define PLUS(a,b) (a) + (b)
    int f() { return PLUS(LIT, LIT); }
  )cc";

  testRule(makeRule(integerLiteral(), changeTo(cat("LIT"))), Input, Expected);
}

// Tests case where the rule's match spans both source from the macro and its
// arg, with the begin location (the "anchor") being the arg.
TEST_F(TransformerTest, MatchSpansMacroTextButChangeDoesNot) {
  std::string Input = R"cc(
#define PLUS_ONE(a) a + 1
    int f() { return PLUS_ONE(3); }
  )cc";
  std::string Expected = R"cc(
#define PLUS_ONE(a) a + 1
    int f() { return PLUS_ONE(LIT); }
  )cc";

  StringRef E = "expr";
  testRule(makeRule(binaryOperator(hasLHS(expr().bind(E))),
                    changeTo(node(std::string(E)), cat("LIT"))),
           Input, Expected);
}

// Tests case where the rule's match spans both source from the macro and its
// arg, with the begin location (the "anchor") being inside the macro.
TEST_F(TransformerTest, MatchSpansMacroTextButChangeDoesNotAnchoredInMacro) {
  std::string Input = R"cc(
#define PLUS_ONE(a) 1 + a
    int f() { return PLUS_ONE(3); }
  )cc";
  std::string Expected = R"cc(
#define PLUS_ONE(a) 1 + a
    int f() { return PLUS_ONE(LIT); }
  )cc";

  StringRef E = "expr";
  testRule(makeRule(binaryOperator(hasRHS(expr().bind(E))),
                    changeTo(node(std::string(E)), cat("LIT"))),
           Input, Expected);
}

// No rewrite is applied when the changed text does not encompass the entirety
// of the expanded text. That is, the edit would have to be applied to the
// macro's definition to succeed and editing the expansion point would not
// suffice.
TEST_F(TransformerTest, NoPartialRewriteOMacroExpansion) {
  std::string Input = R"cc(
#define ZERO_PLUS 0 + 3
    int f(string s) { return ZERO_PLUS; })cc";

  StringRef zero = "zero";
  RewriteRule R = makeRule(integerLiteral(equals(0)).bind(zero),
                           changeTo(node(std::string(zero)), cat("0")));
  testRule(R, Input, Input);
}

// This test handles the corner case where a macro expands within another macro
// to matching code, but that code is an argument to the nested macro call.  A
// simple check of isMacroArgExpansion() vs. isMacroBodyExpansion() will get
// this wrong, and transform the code.
TEST_F(TransformerTest, NoPartialRewriteOfMacroExpansionForMacroArgs) {
  std::string Input = R"cc(
#define NESTED(e) e
#define MACRO(str) 1 + NESTED(strlen((str).c_str()))
    int f(string s) { return MACRO(s); }
  )cc";

  testRule(ruleStrlenSize(), Input, Input);
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
// Verifies that `Type` and `QualType` are not allowed as top-level matchers in
// rules.
TEST(TransformerDeathTest, OrderedRuleTypes) {
  RewriteRule QualTypeRule = makeRule(qualType(), changeTo(cat("Q")));
  EXPECT_DEATH(transformer::detail::buildMatchers(QualTypeRule),
               "Matcher must be.*node matcher");

  RewriteRule TypeRule = makeRule(arrayType(), changeTo(cat("T")));
  EXPECT_DEATH(transformer::detail::buildMatchers(TypeRule),
               "Matcher must be.*node matcher");
}
#endif

// Edits are able to span multiple files; in this case, a header and an
// implementation file.
TEST_F(TransformerTest, MultipleFiles) {
  std::string Header = R"cc(void RemoveThisFunction();)cc";
  std::string Source = R"cc(#include "input.h"
                            void RemoveThisFunction();)cc";
  Transformer T(
      makeRule(functionDecl(hasName("RemoveThisFunction")), changeTo(cat(""))),
      consumer());
  T.registerMatchers(&MatchFinder);
  auto Factory = newFrontendActionFactory(&MatchFinder);
  EXPECT_TRUE(runToolOnCodeWithArgs(
      Factory->create(), Source, std::vector<std::string>(), "input.cc",
      "clang-tool", std::make_shared<PCHContainerOperations>(),
      {{"input.h", Header}}));

  llvm::sort(Changes, [](const AtomicChange &L, const AtomicChange &R) {
    return L.getFilePath() < R.getFilePath();
  });

  ASSERT_EQ(Changes[0].getFilePath(), "./input.h");
  EXPECT_THAT(Changes[0].getInsertedHeaders(), IsEmpty());
  EXPECT_THAT(Changes[0].getRemovedHeaders(), IsEmpty());
  llvm::Expected<std::string> UpdatedCode =
      clang::tooling::applyAllReplacements(Header,
                                           Changes[0].getReplacements());
  ASSERT_TRUE(static_cast<bool>(UpdatedCode))
      << "Could not update code: " << llvm::toString(UpdatedCode.takeError());
  EXPECT_EQ(format(*UpdatedCode), "");

  ASSERT_EQ(Changes[1].getFilePath(), "input.cc");
  EXPECT_THAT(Changes[1].getInsertedHeaders(), IsEmpty());
  EXPECT_THAT(Changes[1].getRemovedHeaders(), IsEmpty());
  UpdatedCode = clang::tooling::applyAllReplacements(
      Source, Changes[1].getReplacements());
  ASSERT_TRUE(static_cast<bool>(UpdatedCode))
      << "Could not update code: " << llvm::toString(UpdatedCode.takeError());
  EXPECT_EQ(format(*UpdatedCode), format("#include \"input.h\"\n"));
}

TEST_F(TransformerTest, AddIncludeMultipleFiles) {
  std::string Header = R"cc(void RemoveThisFunction();)cc";
  std::string Source = R"cc(#include "input.h"
                            void Foo() {RemoveThisFunction();})cc";
  Transformer T(
      makeRule(callExpr(callee(
                   functionDecl(hasName("RemoveThisFunction")).bind("fun"))),
               addInclude(node("fun"), "header.h")),
      consumer());
  T.registerMatchers(&MatchFinder);
  auto Factory = newFrontendActionFactory(&MatchFinder);
  EXPECT_TRUE(runToolOnCodeWithArgs(
      Factory->create(), Source, std::vector<std::string>(), "input.cc",
      "clang-tool", std::make_shared<PCHContainerOperations>(),
      {{"input.h", Header}}));

  ASSERT_EQ(Changes.size(), 1U);
  ASSERT_EQ(Changes[0].getFilePath(), "./input.h");
  EXPECT_THAT(Changes[0].getInsertedHeaders(), ElementsAre("header.h"));
  EXPECT_THAT(Changes[0].getRemovedHeaders(), IsEmpty());
  llvm::Expected<std::string> UpdatedCode =
      clang::tooling::applyAllReplacements(Header,
                                           Changes[0].getReplacements());
  ASSERT_TRUE(static_cast<bool>(UpdatedCode))
      << "Could not update code: " << llvm::toString(UpdatedCode.takeError());
  EXPECT_EQ(format(*UpdatedCode), format(Header));
}

// A single change set can span multiple files.
TEST_F(TransformerTest, MultiFileEdit) {
  // NB: The fixture is unused for this test, but kept for the test suite name.
  std::string Header = R"cc(void Func(int id);)cc";
  std::string Source = R"cc(#include "input.h"
                            void Caller() {
                              int id = 0;
                              Func(id);
                            })cc";
  int ErrorCount = 0;
  std::vector<AtomicChanges> ChangeSets;
  clang::ast_matchers::MatchFinder MatchFinder;
  Transformer T(
      makeRule(callExpr(callee(functionDecl(hasName("Func"))),
                        forEachArgumentWithParam(expr().bind("arg"),
                                                 parmVarDecl().bind("param"))),
               {changeTo(node("arg"), cat("ARG")),
                changeTo(node("param"), cat("PARAM"))}),
      [&](Expected<MutableArrayRef<AtomicChange>> Changes) {
        if (Changes)
          ChangeSets.push_back(AtomicChanges(Changes->begin(), Changes->end()));
        else
          ++ErrorCount;
      });
  T.registerMatchers(&MatchFinder);
  auto Factory = newFrontendActionFactory(&MatchFinder);
  EXPECT_TRUE(runToolOnCodeWithArgs(
      Factory->create(), Source, std::vector<std::string>(), "input.cc",
      "clang-tool", std::make_shared<PCHContainerOperations>(),
      {{"input.h", Header}}));

  EXPECT_EQ(ErrorCount, 0);
  EXPECT_THAT(
      ChangeSets,
      UnorderedElementsAre(UnorderedElementsAre(
          ResultOf([](const AtomicChange &C) { return C.getFilePath(); },
                   "input.cc"),
          ResultOf([](const AtomicChange &C) { return C.getFilePath(); },
                   "./input.h"))));
}

TEST_F(TransformerTest, GeneratesMetadata) {
  std::string Input = R"cc(int target = 0;)cc";
  std::string Expected = R"cc(REPLACE)cc";
  RewriteRuleWith<std::string> Rule = makeRule(
      varDecl(hasName("target")), changeTo(cat("REPLACE")), cat("METADATA"));
  testRule(std::move(Rule), Input, Expected);
  EXPECT_EQ(ErrorCount, 0);
  EXPECT_THAT(StringMetadata, UnorderedElementsAre("METADATA"));
}

TEST_F(TransformerTest, GeneratesMetadataWithNoEdits) {
  std::string Input = R"cc(int target = 0;)cc";
  RewriteRuleWith<std::string> Rule = makeRule(
      varDecl(hasName("target")).bind("var"), noEdits(), cat("METADATA"));
  testRule(std::move(Rule), Input, Input);
  EXPECT_EQ(ErrorCount, 0);
  EXPECT_THAT(StringMetadata, UnorderedElementsAre("METADATA"));
}

TEST_F(TransformerTest, PropagateMetadataErrors) {
  class AlwaysFail : public transformer::MatchComputation<std::string> {
    llvm::Error eval(const ast_matchers::MatchFinder::MatchResult &,
                     std::string *) const override {
      return llvm::createStringError(llvm::errc::invalid_argument, "ERROR");
    }
    std::string toString() const override { return "AlwaysFail"; }
  };
  std::string Input = R"cc(int target = 0;)cc";
  RewriteRuleWith<std::string> Rule = makeRule<std::string>(
      varDecl(hasName("target")).bind("var"), changeTo(cat("REPLACE")),
      std::make_shared<AlwaysFail>());
  testRuleFailure(std::move(Rule), Input);
  EXPECT_EQ(ErrorCount, 1);
}

} // namespace
