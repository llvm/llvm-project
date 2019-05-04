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

#include "llvm/Support/ErrorHandling.h"
#include <vector>
#include <string>

namespace clang {
namespace ast_matchers {

typedef std::vector<std::string> ArgVector;

enum Language {
    Lang_C,
    Lang_C89,
    Lang_CXX,
    Lang_CXX11,
    Lang_CXX14,
    Lang_CXX2a,
    Lang_OpenCL,
    Lang_OBJCXX
};

ArgVector getBasicRunOptionsForLanguage(Language Lang);

} // end namespace ast_matchers
} // end namespace clang

#endif
