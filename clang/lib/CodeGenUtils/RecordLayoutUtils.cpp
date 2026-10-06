//===--- RecordLayoutUtils.cpp - Shared record layout queries -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/RecordLayoutUtils.h"

namespace clang::CodeGenUtils {

bool isEmptyFieldForLayout(const ASTContext &Ctx, const FieldDecl *FD) {
  if (FD->isZeroLengthBitField())
    return true;

  if (FD->isUnnamedBitField())
    return false;

  return isEmptyRecordForLayout(Ctx, FD->getType());
}

bool isEmptyRecordForLayout(const ASTContext &Ctx, QualType T) {
  const auto *RD = T->getAsRecordDecl();
  if (!RD)
    return false;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (CXXRD->isDynamicClass())
      return false;

    for (const auto &I : CXXRD->bases())
      if (!isEmptyRecordForLayout(Ctx, I.getType()))
        return false;
  }

  for (const auto *I : RD->fields())
    if (!isEmptyFieldForLayout(Ctx, I))
      return false;

  return true;
}

} // namespace clang::CodeGenUtils
