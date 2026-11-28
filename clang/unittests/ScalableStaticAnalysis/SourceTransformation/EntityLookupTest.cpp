//===- EntityLookupTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/EntityLookup.h"
#include "FindDecl.h"
#include "TestFixture.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include <optional>

using namespace clang;
using namespace clang::ssaf;

namespace {

constexpr llvm::StringLiteral TestCompilationUnitId = "test-cu";
constexpr llvm::StringLiteral TestLinkUnitId = "test-lu";

NestedBuildNamespace testTUNamespace() {
  return NestedBuildNamespace::makeCompilationUnit(TestCompilationUnitId);
}

NestedBuildNamespace testLUNamespace() {
  return NestedBuildNamespace::makeLinkUnit(TestLinkUnitId);
}

SSAFOptions options(StringRef CompilationUnitId = TestCompilationUnitId,
                    StringRef LinkUnitId = TestLinkUnitId) {
  SSAFOptions Opts;
  Opts.CompilationUnitId = CompilationUnitId.str();
  Opts.LinkUnitId = LinkUnitId.str();
  return Opts;
}

class EntityLookupTest : public TestFixture {
protected:
  ASTContext &parse(StringRef Code) {
    AST = tooling::buildASTFromCode(Code);
    return AST->getASTContext();
  }

  EntityId add(std::optional<EntityName> Name) {
    assert(Name && "the declaration has no entity name");
    return getIdTable(Suite).getId(*Name);
  }

  EntityId addEntityOf(const Decl *D) {
    return add(getQualifiedEntityName(D, testTUNamespace(), testLUNamespace()));
  }

  EntityId addReturnOf(const FunctionDecl *FD) {
    return add(getQualifiedEntityNameForReturn(FD, testTUNamespace(),
                                               testLUNamespace()));
  }

  std::unique_ptr<ASTUnit> AST;
  WPASuite Suite = makeWPASuite();
};

TEST_F(EntityLookupTest, Declaration) {
  ASTContext &Ctx = parse("void f(); void g();");
  const FunctionDecl *F = findFnByName("f", Ctx);
  const FunctionDecl *G = findFnByName("g", Ctx);
  EntityId FId = addEntityOf(F);
  EntityId GId = addEntityOf(G);

  EntityLookup Lookup(Suite, options());
  EXPECT_EQ(Lookup.getEntityId(F), FId);
  EXPECT_EQ(Lookup.getEntityId(G), GId);
}

TEST_F(EntityLookupTest, Parameter) {
  ASTContext &Ctx = parse("void f(int *p, int *q);");
  const FunctionDecl *F = findFnByName("f", Ctx);
  EntityId QId = addEntityOf(F->getParamDecl(1));

  EntityLookup Lookup(Suite, options());
  EXPECT_EQ(Lookup.getEntityId(F->getParamDecl(1)), QId);
  EXPECT_EQ(Lookup.getEntityId(F->getParamDecl(0)), std::nullopt);
}

TEST_F(EntityLookupTest, ReturnValue) {
  ASTContext &Ctx = parse("int *f();");
  const FunctionDecl *F = findFnByName("f", Ctx);
  EntityId FId = addEntityOf(F);
  EntityId ReturnId = addReturnOf(F);
  ASSERT_NE(FId, ReturnId);

  EntityLookup Lookup(Suite, options());
  EXPECT_EQ(Lookup.getEntityId(F), FId);
  EXPECT_EQ(Lookup.getEntityIdForReturn(F), ReturnId);
}

TEST_F(EntityLookupTest, DeclarationMissingFromSuite) {
  ASTContext &Ctx = parse("void f(); void g();");
  const FunctionDecl *F = findFnByName("f", Ctx);
  addEntityOf(F);

  EntityLookup Lookup(Suite, options());
  EXPECT_EQ(Lookup.getEntityId(findFnByName("g", Ctx)), std::nullopt);
  EXPECT_EQ(Lookup.getEntityIdForReturn(F), std::nullopt);
}

TEST_F(EntityLookupTest, InternalLinkageIsQualifiedByCompilationUnit) {
  ASTContext &Ctx = parse("static void f() {}");
  const FunctionDecl *F = findFnByName("f", Ctx);
  EntityId Id = addEntityOf(F);

  EXPECT_EQ(EntityLookup(Suite, options()).getEntityId(F), Id);
  EXPECT_EQ(EntityLookup(Suite, options("other-cu")).getEntityId(F),
            std::nullopt);
  EXPECT_EQ(EntityLookup(Suite, options(TestCompilationUnitId, "other-lu"))
                .getEntityId(F),
            std::nullopt);
}

TEST_F(EntityLookupTest, ExternalLinkageIsQualifiedByLinkUnitOnly) {
  ASTContext &Ctx = parse("void f();");
  const FunctionDecl *F = findFnByName("f", Ctx);
  EntityId Id = addEntityOf(F);

  EXPECT_EQ(EntityLookup(Suite, options("other-cu")).getEntityId(F), Id);
  EXPECT_EQ(EntityLookup(Suite, options(TestCompilationUnitId, "other-lu"))
                .getEntityId(F),
            std::nullopt);
}

} // namespace
