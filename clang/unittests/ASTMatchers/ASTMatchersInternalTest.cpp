// unittests/ASTMatchers/ASTMatchersInternalTest.cpp - AST matcher unit tests //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTMatchersTest.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Host.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {
using internal::DynTypedMatcher;

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

// FIXME: Figure out why back traces aren't being generated on clang builds on
// windows.
#if ENABLE_BACKTRACES && (!defined(_MSC_VER) || !defined(__clang__))
template <typename MatcherT>
static void crashTestNodeDump(MatcherT Matcher,
                              ArrayRef<StringRef> MatchedNodes,
                              StringRef Code) {
  llvm::EnablePrettyStackTrace();
  MatchFinder Finder;

  struct CrashCallback : public MatchFinder::MatchCallback {
    void run(const MatchFinder::MatchResult &Result) override { abort(); }
    llvm::Optional<TraversalKind> getCheckTraversalKind() const override {
      return TK_IgnoreUnlessSpelledInSource;
    }
    StringRef getID() const override { return "CrashTester"; }
  } Callback;
  Finder.addMatcher(std::move(Matcher), &Callback);
  if (MatchedNodes.empty()) {
    ASSERT_DEATH(tooling::runToolOnCode(
                     newFrontendActionFactory(&Finder)->create(), Code),
                 testing::HasSubstr(
                     "ASTMatcher: Processing 'CrashTester'\nNo bound nodes"));
  } else {
    std::vector<testing::PolymorphicMatcher<
        testing::internal::HasSubstrMatcher<std::string>>>
        Matchers;
    Matchers.reserve(MatchedNodes.size());
    for (auto Node : MatchedNodes) {
      Matchers.push_back(testing::HasSubstr(Node.str()));
    }
    auto CrashMatcher = testing::AllOf(
        testing::HasSubstr(
            "ASTMatcher: Processing 'CrashTester'\n--- Bound Nodes Begin ---"),
        testing::HasSubstr("--- Bound Nodes End ---"),
        testing::AllOfArray(Matchers));

    ASSERT_DEATH(tooling::runToolOnCode(
                     newFrontendActionFactory(&Finder)->create(), Code),
                 CrashMatcher);
  }
}
TEST(MatcherCrashDeathTest, CrashOnCallbackDump) {
  crashTestNodeDump(forStmt(), {}, "void foo() { for(;;); }");
  crashTestNodeDump(
      forStmt(hasLoopInit(declStmt(hasSingleDecl(
                                       varDecl(hasType(qualType().bind("QT")),
                                               hasType(type().bind("T")),
                                               hasInitializer(
                                                   integerLiteral().bind("IL")))
                                           .bind("VD")))
                              .bind("DS")))
          .bind("FS"),
      {"FS - { ForStmt : <input.cc:3:5, line:4:5> }",
       "DS - { DeclStmt : <input.cc:3:10, col:19> }",
       "IL - { IntegerLiteral : <input.cc:3:18> }", "QT - { QualType : int }",
       "T - { BuiltinType : int }",
       "VD - { VarDecl I : <input.cc:3:10, col:18> }"},
      R"cpp(
  void foo() {
    for (int I = 0; I < 5; ++I) {
    }
  }
  )cpp");
  crashTestNodeDump(
      cxxRecordDecl(hasMethod(cxxMethodDecl(hasName("operator+")).bind("Op+")))
          .bind("Unnamed"),
      {"Unnamed - { CXXRecordDecl (anonymous) : <input.cc:1:1, col:36> }",
       "Op+ - { CXXMethodDecl (anonymous struct)::operator+ : <input.cc:1:10, "
       "col:29> }"},
      "struct { int operator+(int) const; } Unnamed;");
  crashTestNodeDump(
      cxxRecordDecl(hasMethod(cxxConstructorDecl(isDefaulted()).bind("Ctor")),
                    hasMethod(cxxDestructorDecl(isDefaulted()).bind("Dtor"))),
      {"Ctor - { CXXConstructorDecl Foo::Foo : <input.cc:1:14, col:28> }",
       "Dtor - { CXXDestructorDecl Foo::~Foo : <input.cc:1:31, col:46> }"},
      "struct Foo { Foo() = default; ~Foo() = default; };");
}
#endif // ENABLE_BACKTRACES
#endif

TEST(ConstructVariadic, MismatchedTypes_Regression) {
  EXPECT_TRUE(
      matches("const int a = 0;", internal::DynTypedMatcher::constructVariadic(
                                      internal::DynTypedMatcher::VO_AnyOf,
                                      ASTNodeKind::getFromNodeKind<QualType>(),
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
      HasClassB, std::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, std::make_unique<VerifyIdIsBoundTo<Decl>>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, std::make_unique<VerifyIdIsBoundTo<Decl>>("b")));
}

AST_POLYMORPHIC_MATCHER_P(polymorphicHas,
                          AST_POLYMORPHIC_SUPPORTED_TYPES(Decl, Stmt),
                          internal::Matcher<Decl>, AMatcher) {
  return Finder->matchesChildOf(
      Node, AMatcher, Builder,
      ASTMatchFinder::BK_First);
}

TEST(AstPolymorphicMatcherPMacro, Works) {
  DeclarationMatcher HasClassB =
      polymorphicHas(recordDecl(hasName("B")).bind("b"));

  EXPECT_TRUE(matchAndVerifyResultTrue("class A { class B {}; };",
      HasClassB, std::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class B {}; };",
      HasClassB, std::make_unique<VerifyIdIsBoundTo<Decl>>("a")));

  EXPECT_TRUE(matchAndVerifyResultFalse("class A { class C {}; };",
      HasClassB, std::make_unique<VerifyIdIsBoundTo<Decl>>("b")));

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

TEST(DynTypedMatcherTest, TraversalKindForwardsToImpl) {
  auto M = DynTypedMatcher(decl());
  EXPECT_FALSE(M.getTraversalKind().hasValue());

  M = DynTypedMatcher(traverse(TK_AsIs, decl()));
  EXPECT_THAT(M.getTraversalKind(), llvm::ValueIs(TK_AsIs));
}

TEST(DynTypedMatcherTest, ConstructWithTraversalKindSetsTK) {
  auto M = DynTypedMatcher(decl()).withTraversalKind(TK_AsIs);
  EXPECT_THAT(M.getTraversalKind(), llvm::ValueIs(TK_AsIs));
}

TEST(DynTypedMatcherTest, ConstructWithTraversalKindOverridesNestedTK) {
  auto M = DynTypedMatcher(decl()).withTraversalKind(TK_AsIs).withTraversalKind(
      TK_IgnoreUnlessSpelledInSource);
  EXPECT_THAT(M.getTraversalKind(),
              llvm::ValueIs(TK_IgnoreUnlessSpelledInSource));
}

TEST(IsInlineMatcher, IsInline) {
  EXPECT_TRUE(matches("void g(); inline void f();",
                      functionDecl(isInline(), hasName("f"))));
  EXPECT_TRUE(matches("namespace n { inline namespace m {} }",
                      namespaceDecl(isInline(), hasName("m"))));
  EXPECT_TRUE(matches("inline int Foo = 5;",
                      varDecl(isInline(), hasName("Foo")), {Lang_CXX17}));
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
                                   {"-isystem/"}, M));
}

TEST(Matcher, IsExpansionInSystemHeader) {
  FileContentMappings M;
  M.push_back(std::make_pair("/other", "class X {};"));
  EXPECT_TRUE(matchesConditionally("#include \"other\"\n",
                                   recordDecl(isExpansionInSystemHeader()),
                                   true, {"-isystem/"}, M));
  EXPECT_TRUE(matchesConditionally("#include \"other\"\n",
                                   recordDecl(isExpansionInSystemHeader()),
                                   false, {"-I/"}, M));
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
      {"-isystem/"}, M));
  EXPECT_TRUE(matchesConditionally(
      "#include <foo>\n"
      "#include <bar>\n"
      "class X {};",
      recordDecl(isExpansionInFileMatching("f.*"), hasName("X")), false,
      {"-isystem/"}, M));
}

#endif // _WIN32

} // end namespace ast_matchers
} // end namespace clang
