//===------ unittest/AST/Language.cpp - AST unit test support -------------===//
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

#include "Language.h"

namespace clang {
namespace ast_matchers {

ArgVector getBasicRunOptionsForLanguage(Language Lang) {
  ArgVector BasicArgs;
  // Test with basic arguments.
  switch (Lang) {
  case Lang_C:
    BasicArgs = {"-x", "c", "-std=c99"};
    break;
  case Lang_C89:
    BasicArgs = {"-x", "c", "-std=c89"};
    break;
  case Lang_CXX:
    BasicArgs = {"-std=c++98", "-frtti"};
    break;
  case Lang_CXX11:
    BasicArgs = {"-std=c++11", "-frtti"};
    break;
  case Lang_CXX14:
    BasicArgs = {"-std=c++14", "-frtti"};
    break;
  case Lang_CXX2a:
    BasicArgs = {"-std=c++2a", "-frtti"};
    break;
  case Lang_OpenCL:
  case Lang_OBJCXX:
    llvm_unreachable("Not implemented yet!");
  }
  return BasicArgs;
}

} // end namespace ast_matchers
} // end namespace clang
