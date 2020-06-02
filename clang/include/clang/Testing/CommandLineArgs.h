//===--- CommandLineArgs.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines language options for Clang unittests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TESTING_COMMANDLINEARGS_H
#define LLVM_CLANG_TESTING_COMMANDLINEARGS_H

#include <string>
#include <vector>

namespace clang {

enum TestLanguage {
  Lang_C89,
  Lang_C99,
  Lang_CXX03,
  Lang_CXX11,
  Lang_CXX14,
  Lang_CXX17,
  Lang_CXX20,
  Lang_OpenCL,
  Lang_OBJCXX
};

std::vector<std::string> getCommandLineArgsForTesting(TestLanguage Lang);

} // end namespace clang

#endif
