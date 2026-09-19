//===--- FlangTidy.h - flang-tidy -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDY_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDY_H

#include "FlangTidyOptions.h"

namespace Fortran::tidy {

int runFlangTidy(const FlangTidyOptions &options);

} // namespace Fortran::tidy

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_FLANGTIDY_H
