//===--- FlangTidyToolMain.cpp - flang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FlangTidyMain.h"

int main(int argc, const char **argv) {
  return Fortran::tidy::flangTidyMain(argc, argv);
}
