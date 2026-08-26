//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/TypeBase.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>
#include <string>

using namespace clang::tooling;
using namespace clang;

static void ConvertUTF8ToWideString(unsigned CharByteWidth, StringRef Source,
                                    SmallString<32> &Target) {
  Target.resize(CharByteWidth * (Source.size() + 1));
  char *ResultPtr = &Target[0];
  const llvm::UTF8 *ErrorPtr;
  bool success =
      llvm::ConvertUTF8toWide(CharByteWidth, Source, ResultPtr, ErrorPtr);
  (void)success;
  assert(success);
  Target.resize(ResultPtr - &Target[0]);
}

TEST(StringLiteral, findZeroCodeUnit) {
  auto AST = tooling::buildASTFromCodeWithArgs("", {});
  ASTContext &Ctx = AST->getASTContext();

  auto getCharArrayType = [&Ctx](unsigned Size) -> QualType {
    return Ctx.getStringLiteralArrayType(Ctx.CharTy.withConst(), Size);
  };
  auto getWCharArrayType = [&Ctx](unsigned Size) -> QualType {
    return Ctx.getStringLiteralArrayType(Ctx.WCharTy.withConst(), Size);
  };

  const auto *S1 =
      StringLiteral::Create(Ctx, "abcdef", StringLiteralKind::Ordinary, false,
                            getCharArrayType(7), {});
  ASSERT_EQ(S1->getLength(), 6u);
  ASSERT_EQ(*S1->findZeroCodeUnit(), 6u);
  ASSERT_EQ(*S1->findZeroCodeUnit(4), 2u);
  ASSERT_FALSE(S1->findZeroCodeUnit(16).has_value());

  const auto *S2 = StringLiteral::Create(Ctx, StringRef("a\0bcd", 6),
                                         StringLiteralKind::Ordinary, false,
                                         getCharArrayType(6), {});
  ASSERT_EQ(S2->getLength(), 6u);
  ASSERT_EQ(*S2->findZeroCodeUnit(), 1u);
  ASSERT_EQ(*S2->findZeroCodeUnit(1), 0u);
  ASSERT_EQ(*S2->findZeroCodeUnit(2), 3u);

  SmallString<32> RawChars;
  ConvertUTF8ToWideString(4, "abcdef", RawChars);
  const auto *S3 = StringLiteral::Create(
      Ctx, RawChars, StringLiteralKind::UTF32, false, getWCharArrayType(7), {});
  ASSERT_EQ(S3->getLength(), 6u);
  ASSERT_EQ(*S3->findZeroCodeUnit(), 6u);
  ASSERT_EQ(*S3->findZeroCodeUnit(2), 4u);

  ConvertUTF8ToWideString(4, StringRef("abc\0ef", 6), RawChars);
  const auto *S4 = StringLiteral::Create(
      Ctx, RawChars, StringLiteralKind::UTF32, false, getWCharArrayType(7), {});
  ASSERT_EQ(S4->getLength(), 6u);
  ASSERT_EQ(S4->findZeroCodeUnit(), 3u);
  ASSERT_EQ(S4->findZeroCodeUnit(3u), 0u);
  ASSERT_EQ(S4->findZeroCodeUnit(4u), 2u);
}
