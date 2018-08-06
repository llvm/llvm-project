// unittests/ASTMatchers/ASTMatchersInternalTest.cpp - AST matcher unit tests //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTMatchersTest.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

#if GTEST_HAS_DEATH_TEST
TEST(HasNameDeathTest, DiesOnEmptyName) {
  ASSERT_DEBUG_DEATH({
    DeclarationMatcher HasEmptyName = recordDecl(hasName(""));
    EXPECT_TRUE(notMatches("class X {};", HasEmptyName));
  }, "");
}

TEST(HasNameDeathTest, DiesOnEmptyPattern) {
  ASSERT_DEBUG_DEATH({
      DeclarationMatcher HasEmptyName = recordDecl(matchesName(""));
      EXPECT_TRUE(notMatches("class X {};", HasEmptyName));
    }, "");
}

TEST(IsDerivedFromDeathTest, DiesOnEmptyBaseName) {
  ASSERT_DEBUG_DEATH({
    DeclarationMatcher IsDerivedFromEmpty = cxxRecordDecl(isDerivedFrom(""));
    EXPECT_TRUE(notMatches("class X {};", IsDerivedFromEmpty));
  }, "");
}
#endif

TEST(ConstructVariadic, MismatchedTypes_Regression) {
  EXPECT_TRUE(
      matches("const int a = 0;",
              internal::DynTypedMatcher::constructVariadic(
                  internal::DynTypedMatcher::VO_AnyOf,
                  ast_type_traits::ASTNodeKind::getFromNodeKind<QualType>(),
                  {isConstQualified(), arrayType()})
                  .convertTo<QualType>()));
}

// For testing AST_MATCHER_P().
AST_MATCHER_P(Decl, just, internal::Matcher<Decl>, AMatcher) {
  // Make sure all special variables are used: node, match_finder,
  // bound_nodes_builder, and the parameter named 'AMatcher'.
  return AMatcher.matches(Node, Finder, Builder);
}

TEST(AstMatcherPMacro, Works) {
  DeclarationMatcher HasClassB = just(has(recordDecl(hasName("B")).bind("b")));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, llvm::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, llvm::make_unique<VerifyIdIsBoundTo<Decl>>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, llvm::make_unique<VerifyIdIsBoundTo<Decl>>("b")));
}

AST_POLYMORPHIC_MATCHER_P(polymorphicHas,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt),
                          internal::Matcher<Decl>, AMatcher) {
  return Finder->matchesChildOf(
      Node, AMatcher, Builder,
      ASTMatchFinder::TK_IgnoreImplicitCastsAndParentheses,
      ASTMatchFinder::BK_First);
}

TEST(AstPolymorphicMatcherPMacro, Works) {
  DeclarationMatcher HasClassB =
      polymorphicHas(recordDecl(hasName("B")).bind("b"));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, llvm::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, llvm::make_unique<VerifyIdIsBoundTo<Decl>>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, llvm::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

  StatementMatcher StatementHasClassB =
      polymorphicHas(recordDecl(hasName("B")));

  EXPECT_TRUE(matches("void x() { class B {}; }", StatementHasClassB));
}

TEST(MatchFinder, CheckProfiling) {
  MatchFinder::MatchFinderOptions Options;
  llvm::StringMap<llvm::TimeRecord> Records;
  Options.CheckProfiling.emplace(Records);
  MatchFinder Finder(std::move(Options));

  struct NamedCallback : public MatchFinder::MatchCallback {
    void run(const MatchFinder::MatchResult &Result) override {}
    StringRef getID() const override { return "MyID"; }
  } Callback;
  Finder.addMatcher(decl(), &Callback);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), "int x;"));

  EXPECT_EQ(1u, Records.size());
  EXPECT_EQ("MyID", Records.begin()->getKey());
}

class VerifyStartOfTranslationUnit : public MatchFinder::MatchCallback {
public:
  VerifyStartOfTranslationUnit() : Called(false) {}
  void run(const MatchFinder::MatchResult &Result) override {
    EXPECT_TRUE(Called);
  }
  void onStartOfTranslationUnit() override { Called = true; }
  bool Called;
};

TEST(MatchFinder, InterceptsStartOfTranslationUnit) {
  MatchFinder Finder;
  VerifyStartOfTranslationUnit VerifyCallback;
  Finder.addMatcher(decl(), &VerifyCallback);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), "int x;"));
  EXPECT_TRUE(VerifyCallback.Called);

  VerifyCallback.Called = false;
  std::unique_ptr<ASTUnit> AST(tooling::buildASTFromCode("int x;"));
  ASSERT_TRUE(AST.get());
  Finder.matchAST(AST->getASTContext());
  EXPECT_TRUE(VerifyCallback.Called);
}

class VerifyEndOfTranslationUnit : public MatchFinder::MatchCallback {
public:
  VerifyEndOfTranslationUnit() : Called(false) {}
  void run(const MatchFinder::MatchResult &Result) override {
    EXPECT_FALSE(Called);
  }
  void onEndOfTranslationUnit() override { Called = true; }
  bool Called;
};

TEST(MatchFinder, InterceptsEndOfTranslationUnit) {
  MatchFinder Finder;
  VerifyEndOfTranslationUnit VerifyCallback;
  Finder.addMatcher(decl(), &VerifyCallback);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), "int x;"));
  EXPECT_TRUE(VerifyCallback.Called);

  VerifyCallback.Called = false;
  std::unique_ptr<ASTUnit> AST(tooling::buildASTFromCode("int x;"));
  ASSERT_TRUE(AST.get());
  Finder.matchAST(AST->getASTContext());
  EXPECT_TRUE(VerifyCallback.Called);
}

TEST(Matcher, matchOverEntireASTContext) {
  std::unique_ptr<ASTUnit> AST =
      clang::tooling::buildASTFromCode("struct { int *foo; };");
  ASSERT_TRUE(AST.get());
  auto PT = selectFirst<PointerType>(
      "x", match(pointerType().bind("x"), AST->getASTContext()));
  EXPECT_NE(nullptr, PT);
}

TEST(IsInlineMatcher, IsInline) {
  EXPECT_TRUE(matches("void g(); inline void f();",
                      functionDecl(isInline(), hasName("f"))));
  EXPECT_TRUE(matches("namespace n { inline namespace m {} }",
                      namespaceDecl(isInline(), hasName("m"))));
}

// FIXME: Figure out how to specify paths so the following tests pass on
// Windows.
#ifndef _WIN32

TEST(Matcher, IsExpansionInMainFileMatcher) {
  EXPECT_TRUE(matches("class X {};",
                      recordDecl(hasName("X"), isExpansionInMainFile())));
  EXPECT_TRUE(notMatches("", recordDecl(isExpansionInMainFile())));
  FileContentMappings M;
  M.push_back(std::make_pair("/other", "class X {};"));
  EXPECT_TRUE(matchesConditionally("#include <other>\n",
                                   recordDecl(isExpansionInMainFile()), false,
                                   "-isystem/", M));
}

TEST(Matcher, IsExpansionInSystemHeader) {
  FileContentMappings M;
  M.push_back(std::make_pair("/other", "class X {};"));
  EXPECT_TRUE(matchesConditionally(
      "#include \"other\"\n", recordDecl(isExpansionInSystemHeader()), true,
      "-isystem/", M));
  EXPECT_TRUE(matchesConditionally("#include \"other\"\n",
                                   recordDecl(isExpansionInSystemHeader()),
                                   false, "-I/", M));
  EXPECT_TRUE(notMatches("class X {};",
                         recordDecl(isExpansionInSystemHeader())));
  EXPECT_TRUE(notMatches("", recordDecl(isExpansionInSystemHeader())));
}

TEST(Matcher, IsExpansionInFileMatching) {
  FileContentMappings M;
  M.push_back(std::make_pair("/foo", "class A {};"));
  M.push_back(std::make_pair("/bar", "class B {};"));
  EXPECT_TRUE(matchesConditionally(
      "#include <foo>\n"
      "#include <bar>\n"
      "class X {};",
      recordDecl(isExpansionInFileMatching("b.*"), hasName("B")), true,
      "-isystem/", M));
  EXPECT_TRUE(matchesConditionally(
      "#include <foo>\n"
      "#include <bar>\n"
      "class X {};",
      recordDecl(isExpansionInFileMatching("f.*"), hasName("X")), false,
      "-isystem/", M));
}

#endif // _WIN32

} // end namespace ast_matchers
} // end namespace clang
