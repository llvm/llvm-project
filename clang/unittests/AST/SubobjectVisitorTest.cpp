//===- unittests/AST/SubobjectVisitorTest.cpp - Subobject Visitor tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/SubobjectVisitor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

namespace {

// Helper class to record visited subobjects
class RecordingVisitor : public SubobjectVisitor<RecordingVisitor> {
public:
  std::vector<std::string> VisitedBases;
  std::vector<std::string> VisitedFields;

  RecordingVisitor(ASTContext &Ctx) : SubobjectVisitor<RecordingVisitor>(Ctx) {}

  bool visitBaseSpecifierPre(CXXBaseSpecifier *BS) {
    std::string Name = BS->getType()->getAsCXXRecordDecl()->getNameAsString();
    VisitedBases.push_back(Name);
    return true;
  }

  bool visitFieldDeclPre(FieldDecl *FD) {
    std::string Name = FD->getNameAsString();
    VisitedFields.push_back(Name);
    return true;
  }

};

// Helper function to get a CXXRecordDecl by name
const CXXRecordDecl *getCXXRecordDecl(ASTUnit *AST, const std::string &Name) {
  auto Result = match(cxxRecordDecl(hasName(Name)).bind("record"),
                      AST->getASTContext());
  if (Result.empty())
    return nullptr;
  return Result[0].getNodeAs<CXXRecordDecl>("record");
}

TEST(SubobjectVisitorTest, Basic) {
  auto AST = buildASTFromCode(R"cpp(
    struct Base {
      int BaseF;
    };
    struct Inner {
      int InnerF;
    };
    struct S : public Base {
      int MemberF;
      Inner StructF;
    };
  )cpp");
  ASSERT_TRUE(AST.get());

  const CXXRecordDecl *RD = getCXXRecordDecl(AST.get(), "S");
  ASSERT_TRUE(RD);

  RecordingVisitor Visitor(AST->getASTContext());
  Visitor.visit(AST->getASTContext().getTagType(ElaboratedTypeKeyword::None,
                                                 std::nullopt, RD, false));

  EXPECT_EQ(Visitor.VisitedBases.size(), 1u);
  EXPECT_EQ(Visitor.VisitedFields.size(), 4u);
  EXPECT_EQ(Visitor.VisitedFields[0], "BaseF");
  EXPECT_EQ(Visitor.VisitedFields[1], "MemberF");
  EXPECT_EQ(Visitor.VisitedFields[2], "StructF");
  EXPECT_EQ(Visitor.VisitedFields[3], "InnerF");
}

TEST(SubobjectVisitorTest, Atomic) {
  auto AST = buildASTFromCode(R"cpp(
    struct S {
      int SField;
    };
    struct T {
      _Atomic S InnerAtomic;
    };
  )cpp");
  ASSERT_TRUE(AST.get());

  const CXXRecordDecl *RD = getCXXRecordDecl(AST.get(), "T");
  ASSERT_TRUE(RD);

  RecordingVisitor Visitor(AST->getASTContext());
  Visitor.visit(AST->getASTContext().getTagType(ElaboratedTypeKeyword::None,
                                                 std::nullopt, RD, false));

  EXPECT_EQ(Visitor.VisitedBases.size(), 0);
  EXPECT_EQ(Visitor.VisitedFields.size(), 2);
  EXPECT_EQ(Visitor.VisitedFields[0], "InnerAtomic");
  EXPECT_EQ(Visitor.VisitedFields[1], "SField");
}

TEST(SubobjectVisitorTest, FAM) {
  auto AST = buildASTFromCode(R"cpp(
    struct S {
      int SField[];
    };
    struct T {
      S Inner;
    };
  )cpp");
  ASSERT_TRUE(AST.get());

  const CXXRecordDecl *RD = getCXXRecordDecl(AST.get(), "T");
  ASSERT_TRUE(RD);

  RecordingVisitor Visitor(AST->getASTContext());
  Visitor.visit(AST->getASTContext().getTagType(ElaboratedTypeKeyword::None,
                                                 std::nullopt, RD, false));

  EXPECT_EQ(Visitor.VisitedBases.size(), 0);
  EXPECT_EQ(Visitor.VisitedFields.size(), 2);
  EXPECT_EQ(Visitor.VisitedFields[0], "Inner");
  EXPECT_EQ(Visitor.VisitedFields[1], "SField");
}

} // namespace
