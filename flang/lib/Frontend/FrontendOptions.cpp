//===- FrontendOptions.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendOptions.h"
#include "llvm/ADT/StringSwitch.h"

using namespace Fortran::frontend;

InputKind FrontendOptions::GetInputKindForExtension(llvm::StringRef extension) {
  return llvm::StringSwitch<InputKind>(extension)
      // TODO: Should match the list in flang/test/lit.cfg.py
      // FIXME: Currently this API allows at most 9 items per case.
      .Cases("f", "F", "f77", "f90", "F90", "f95", "F95", "ff95", "f18", "F18",
          Language::Fortran)
      .Default(Language::Unknown);
}
