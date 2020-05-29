//===------ unittest/AST/Language.h - AST unit test support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines language options for AST unittests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_AST_LANGUAGE_H
#define LLVM_CLANG_UNITTESTS_AST_LANGUAGE_H

#include <vector>
#include <string>

namespace clang {

enum TestLanguage {
  Lang_C,
  Lang_C89,
  Lang_CXX,
  Lang_CXX11,
  Lang_CXX14,
  Lang_CXX17,
  Lang_CXX2a,
  Lang_OpenCL,
  Lang_OBJCXX
};

std::vector<std::string> getCommandLineArgsForTesting(TestLanguage Lang);

} // end namespace clang

#endif
