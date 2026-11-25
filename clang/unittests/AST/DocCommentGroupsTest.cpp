//===- unittests/AST/DocCommentGroupsTest.cpp - Doxygen group tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/RawCommentList.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace {

class DocCommentGroupsTest : public ::testing::Test {
protected:
  void build(StringRef Code, StringRef FileName = "input.cc") {
    AST = tooling::buildASTFromCodeWithArgs(
        Code, {"-fretain-comments", "-std=c++17"}, FileName);
    ASSERT_TRUE(AST);
  }

  const NamedDecl *decl(StringRef Name) {
    const auto *D = selectFirst<NamedDecl>(
        "d", match(namedDecl(hasName(Name)).bind("d"), ctx()));
    EXPECT_TRUE(D) << "no declaration named " << Name.str();
    return D;
  }

  const DocCommentGroup *group(StringRef DeclName) {
    const NamedDecl *D = decl(DeclName);
    return D ? ctx().getDocCommentGroupForDecl(D) : nullptr;
  }

  ArrayRef<const DocCommentGroup *> groupsInMainFile() {
    return ctx().getDocCommentGroupsInFile(
        ctx().getSourceManager().getMainFileID());
  }

  ASTContext &ctx() { return AST->getASTContext(); }

  std::unique_ptr<ASTUnit> AST;
};

TEST_F(DocCommentGroupsTest, DefGroupFromIssueExample) {
  build(R"cpp(
/**
 * \defgroup IntVariables Global integer variables
 * @{
 */

/** an integer variable */
extern int IntegerVariable;

/**@}*/

extern int After;
)cpp");
  const DocCommentGroup *G = group("IntegerVariable");
  ASSERT_TRUE(G);
  EXPECT_EQ(G->getKind(), DocCommentGroup::GK_DefGroup);
  EXPECT_EQ(G->getKindName(), "defgroup");
  EXPECT_EQ(G->getName(), "IntVariables");
  EXPECT_EQ(G->getTitle(), "Global integer variables");
  EXPECT_TRUE(G->isTerminated());
  EXPECT_FALSE(G->getParent());
  EXPECT_FALSE(group("After"));
}

TEST_F(DocCommentGroupsTest, MemberGroup) {
  build(R"cpp(
/// \name Group
/// \{
void func1();

void func2();
/// \}

void func3();
)cpp");
  const DocCommentGroup *G = group("func1");
  ASSERT_TRUE(G);
  EXPECT_EQ(G->getKind(), DocCommentGroup::GK_Member);
  EXPECT_EQ(G->getName(), "Group");
  EXPECT_TRUE(G->getTitle().empty());
  EXPECT_EQ(group("func2"), G);
  EXPECT_FALSE(group("func3"));
}

TEST_F(DocCommentGroupsTest, OtherGroupingCommands) {
  build(R"cpp(
/// \addtogroup Added More things
/// @{
int a;
/// @}

/// \weakgroup Weak
/// @{
int w;
/// @}

/// @{
int u;
/// @}
)cpp");
  const DocCommentGroup *A = group("a");
  ASSERT_TRUE(A);
  EXPECT_EQ(A->getKind(), DocCommentGroup::GK_AddToGroup);
  EXPECT_EQ(A->getName(), "Added");
  EXPECT_EQ(A->getTitle(), "More things");

  const DocCommentGroup *W = group("w");
  ASSERT_TRUE(W);
  EXPECT_EQ(W->getKind(), DocCommentGroup::GK_WeakGroup);
  EXPECT_EQ(W->getName(), "Weak");
  EXPECT_TRUE(W->getTitle().empty());

  const DocCommentGroup *U = group("u");
  ASSERT_TRUE(U);
  EXPECT_EQ(U->getKind(), DocCommentGroup::GK_Unnamed);
  EXPECT_TRUE(U->getName().empty());

  EXPECT_EQ(groupsInMainFile().size(), 3u);
}

TEST_F(DocCommentGroupsTest, IngroupKeepsPendingGroupingCommand) {
  build(R"cpp(
/// \defgroup Outer
/// \ingroup Other
/// @{
int x;
/// @}
)cpp");
  const DocCommentGroup *G = group("x");
  ASSERT_TRUE(G);
  EXPECT_EQ(G->getKind(), DocCommentGroup::GK_DefGroup);
  EXPECT_EQ(G->getName(), "Outer");
}

TEST_F(DocCommentGroupsTest, NestedGroups) {
  build(R"cpp(
/// \defgroup Outer
/// @{

int inOuter1;

/// \name Inner
/// @{
int inInner;
/// @}

int inOuter2;

/// @}
)cpp");
  const DocCommentGroup *Outer = group("inOuter1");
  ASSERT_TRUE(Outer);
  EXPECT_EQ(Outer->getName(), "Outer");
  EXPECT_EQ(group("inOuter2"), Outer);

  const DocCommentGroup *Inner = group("inInner");
  ASSERT_TRUE(Inner);
  EXPECT_EQ(Inner->getName(), "Inner");
  EXPECT_EQ(Inner->getParent(), Outer);

  ArrayRef<const DocCommentGroup *> Groups = groupsInMainFile();
  ASSERT_EQ(Groups.size(), 2u);
  EXPECT_EQ(Groups[0], Outer);
  EXPECT_EQ(Groups[1], Inner);
}

TEST_F(DocCommentGroupsTest, ClassMembers) {
  build(R"cpp(
/// \defgroup Classes
/// @{

/// A class.
class Grouped {
public:
  int notInGroup();

  /// \name Accessors
  /// \{
  int getX();
  int getY();
  /// \}

  int alsoNotInGroup();
};

/// @}
)cpp");
  const DocCommentGroup *Classes = group("Grouped");
  ASSERT_TRUE(Classes);
  EXPECT_EQ(Classes->getName(), "Classes");

  EXPECT_FALSE(group("notInGroup"));
  EXPECT_FALSE(group("alsoNotInGroup"));

  const DocCommentGroup *Accessors = group("getX");
  ASSERT_TRUE(Accessors);
  EXPECT_EQ(Accessors->getName(), "Accessors");
  EXPECT_EQ(Accessors->getParent(), Classes);
  EXPECT_EQ(group("getY"), Accessors);
}

TEST_F(DocCommentGroupsTest, NamespacesAndLinkageSpecsDoNotLimitGroups) {
  build(R"cpp(
/// @{
namespace ns {
int inNamespace;
}
extern "C" {
int inLinkageSpec;
}
/// @}
)cpp");
  const DocCommentGroup *G = group("ns");
  ASSERT_TRUE(G);
  EXPECT_EQ(group("ns::inNamespace"), G);
  EXPECT_EQ(group("inLinkageSpec"), G);
}

TEST_F(DocCommentGroupsTest, LocalsAndParametersAreNotInGroups) {
  build(R"cpp(
/// @{
void f(int param) {
  int local;
}
/// @}
)cpp");
  EXPECT_TRUE(group("f"));
  EXPECT_FALSE(group("param"));
  EXPECT_FALSE(group("local"));
}

TEST_F(DocCommentGroupsTest, ClassTemplateMembers) {
  build(R"cpp(
template <typename T>
struct S {
  /// \name Group
  /// @{
  void m();
  /// @}
};

void use() { S<int>().m(); }
)cpp");
  const auto *Pattern = selectFirst<CXXMethodDecl>(
      "m", match(cxxMethodDecl(hasName("m"), hasParent(cxxRecordDecl(hasParent(
                                                 classTemplateDecl()))))
                     .bind("m"),
                 ctx()));
  ASSERT_TRUE(Pattern);
  const DocCommentGroup *G = ctx().getDocCommentGroupForDecl(Pattern);
  ASSERT_TRUE(G);
  EXPECT_EQ(G->getName(), "Group");

  const auto *Instantiated = selectFirst<CXXMethodDecl>(
      "m", match(cxxMethodDecl(hasName("m"),
                               ofClass(classTemplateSpecializationDecl()))
                     .bind("m"),
                 ctx()));
  ASSERT_TRUE(Instantiated);
  EXPECT_EQ(ctx().getDocCommentGroupForDecl(Instantiated), G);
}

TEST_F(DocCommentGroupsTest, MarkersInCodeBlocksAndEscapesAreIgnored) {
  build(R"cpp(
/// Example:
/// \code
///   @{
/// \endcode
int notInGroup1;

/// An escaped backslash: \\{
int notInGroup2;

// Ordinary comments are ignored: @{
int notInGroup3;
)cpp");
  EXPECT_FALSE(group("notInGroup1"));
  EXPECT_FALSE(group("notInGroup2"));
  EXPECT_FALSE(group("notInGroup3"));
  EXPECT_TRUE(groupsInMainFile().empty());
}

TEST_F(DocCommentGroupsTest, UnmatchedMarkers) {
  build(R"cpp(
/// @}
int afterStrayClose;

/// \name Open
/// @{
int inUnterminated;
)cpp");
  EXPECT_FALSE(group("afterStrayClose"));
  const DocCommentGroup *G = group("inUnterminated");
  ASSERT_TRUE(G);
  EXPECT_FALSE(G->isTerminated());
  EXPECT_EQ(G->getSourceRange().getBegin(), G->getSourceRange().getEnd());
}

TEST_F(DocCommentGroupsTest, MarkerLocations) {
  build("/// @{\nint x;\n/// @}\n");
  const DocCommentGroup *G = group("x");
  ASSERT_TRUE(G);
  const SourceManager &SM = ctx().getSourceManager();
  EXPECT_EQ(SM.getSpellingLineNumber(G->getOpenLoc()), 1u);
  EXPECT_EQ(SM.getSpellingColumnNumber(G->getOpenLoc()), 5u);
  EXPECT_EQ(SM.getSpellingLineNumber(G->getCloseLoc()), 3u);
  EXPECT_EQ(SM.getSpellingColumnNumber(G->getCloseLoc()), 5u);
}

TEST_F(DocCommentGroupsTest, NoComments) {
  build("int x;");
  EXPECT_FALSE(group("x"));
  EXPECT_TRUE(groupsInMainFile().empty());
}

} // namespace
