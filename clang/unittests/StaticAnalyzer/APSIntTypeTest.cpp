//===- unittest/StaticAnalyzer/APSIntTest.cpp - getAPSIntType  test --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/RangedConstraintManager.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

namespace {

TEST(getAPSIntTypeTest, APSIntTypeTests) {
  std::unique_ptr<clang::ASTUnit> AST = clang::tooling::buildASTFromCode("");
  clang::ASTContext &Context = AST->getASTContext();
  llvm::BumpPtrAllocator Arena;
  clang::ento::BasicValueFactory BVF{Context, Arena};

  clang::ento::APSIntType Ty = BVF.getAPSIntType(Context.LongAccumTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getLongAccumWidth());
  EXPECT_FALSE(Ty.isUnsigned());

  Ty = BVF.getAPSIntType(Context.UnsignedLongAccumTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getLongAccumWidth());
  EXPECT_TRUE(Ty.isUnsigned());

  Ty = BVF.getAPSIntType(Context.LongFractTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getLongFractWidth());
  EXPECT_FALSE(Ty.isUnsigned());

  Ty = BVF.getAPSIntType(Context.UnsignedLongFractTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getLongFractWidth());
  EXPECT_TRUE(Ty.isUnsigned());

  Ty = BVF.getAPSIntType(Context.SignedCharTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getCharWidth());
  EXPECT_FALSE(Ty.isUnsigned());

  Ty = BVF.getAPSIntType(Context.UnsignedCharTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getCharWidth());
  EXPECT_TRUE(Ty.isUnsigned());

  Ty = BVF.getAPSIntType(Context.LongTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getLongWidth());
  EXPECT_FALSE(Ty.isUnsigned());

  Ty = BVF.getAPSIntType(Context.UnsignedLongTy);
  EXPECT_TRUE(Ty.getBitWidth() == Context.getTargetInfo().getLongWidth());
  EXPECT_TRUE(Ty.isUnsigned());
}

} // end namespace
