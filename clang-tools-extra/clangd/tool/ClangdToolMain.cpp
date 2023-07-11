//===--- ClangdToolMain.cpp - clangd main function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangdMain.h"

int main(int argc, char **argv) {
  return clang::clangd::clangdMain(argc, argv);
}
