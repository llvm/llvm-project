//===- FrontendOptions.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendOptions.h"
#include "flang/FrontendTool/Utils.h"

using namespace Fortran::frontend;

InputKind FrontendOptions::GetInputKindForExtension(llvm::StringRef extension) {
  if (isFixedFormSuffix(extension) || isFreeFormSuffix(extension)) {
    return Language::Fortran;
  }
  return Language::Unknown;
}
