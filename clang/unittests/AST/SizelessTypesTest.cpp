//===- unittests/AST/SizelessTypesTest.cpp --- Sizeless type tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for clang::Type queries related to sizeless types.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;

struct SizelessTypeTester : public ::testing::Test {
  // Declare an incomplete structure type.
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      "struct foo;", {"-target", "aarch64-linux-gnu"});
  ASTContext &Ctx = AST->getASTContext();
  TranslationUnitDecl &TU = *Ctx.getTranslationUnitDecl();
  TypeDecl *Foo = cast<TypeDecl>(TU.lookup(&Ctx.Idents.get("foo")).front());
  const Type *FooTy = Foo->getTypeForDecl();
};

TEST_F(SizelessTypeTester, TestSizelessBuiltin) {
  ASSERT_TRUE(Ctx.SveInt8Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveInt16Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveInt32Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveInt64Ty->isSizelessBuiltinType());

  ASSERT_TRUE(Ctx.SveUint8Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveUint16Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveUint32Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveUint64Ty->isSizelessBuiltinType());

  ASSERT_TRUE(Ctx.SveFloat16Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveFloat32Ty->isSizelessBuiltinType());
  ASSERT_TRUE(Ctx.SveFloat64Ty->isSizelessBuiltinType());

  ASSERT_TRUE(Ctx.SveBFloat16Ty->isSizelessBuiltinType());

  ASSERT_TRUE(Ctx.SveBoolTy->isSizelessBuiltinType());

  ASSERT_FALSE(Ctx.VoidTy->isSizelessBuiltinType());
  ASSERT_FALSE(Ctx.PseudoObjectTy->isSizelessBuiltinType());
  ASSERT_FALSE(FooTy->isSizelessBuiltinType());

  ASSERT_FALSE(Ctx.getPointerType(Ctx.SveBoolTy)->isSizelessBuiltinType());
  ASSERT_FALSE(
      Ctx.getLValueReferenceType(Ctx.SveBoolTy)->isSizelessBuiltinType());
  ASSERT_FALSE(
      Ctx.getRValueReferenceType(Ctx.SveBoolTy)->isSizelessBuiltinType());
}

TEST_F(SizelessTypeTester, TestSizeless) {
  ASSERT_TRUE(Ctx.SveInt8Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveInt16Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveInt32Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveInt64Ty->isSizelessType());

  ASSERT_TRUE(Ctx.SveUint8Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveUint16Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveUint32Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveUint64Ty->isSizelessType());

  ASSERT_TRUE(Ctx.SveFloat16Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveFloat32Ty->isSizelessType());
  ASSERT_TRUE(Ctx.SveFloat64Ty->isSizelessType());

  ASSERT_TRUE(Ctx.SveBFloat16Ty->isSizelessType());

  ASSERT_TRUE(Ctx.SveBoolTy->isSizelessType());

  ASSERT_FALSE(Ctx.VoidTy->isSizelessType());
  ASSERT_FALSE(Ctx.PseudoObjectTy->isSizelessType());
  ASSERT_FALSE(FooTy->isSizelessType());

  ASSERT_FALSE(Ctx.getPointerType(Ctx.SveBoolTy)->isSizelessType());
  ASSERT_FALSE(Ctx.getLValueReferenceType(Ctx.SveBoolTy)->isSizelessType());
  ASSERT_FALSE(Ctx.getRValueReferenceType(Ctx.SveBoolTy)->isSizelessType());
}
