//===--- CallUtils.cpp - Shared call emission queries ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/CallUtils.h"

namespace clang::CodeGenUtils {

CanQual<FunctionProtoType> getFormalType(const CXXMethodDecl *MD) {
  return MD->getType()
      ->getCanonicalTypeUnqualified()
      .getAs<FunctionProtoType>();
}

llvm::FPClassTest getNoFPClassTestMask(const LangOptions &LangOpts) {
  llvm::FPClassTest Mask = llvm::fcNone;
  if (LangOpts.NoHonorInfs)
    Mask |= llvm::fcInf;
  if (LangOpts.NoHonorNaNs)
    Mask |= llvm::fcNan;
  return Mask;
}

} // namespace clang::CodeGenUtils
