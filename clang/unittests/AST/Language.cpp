//===------ unittest/AST/Language.cpp - AST unit test support -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  case Lang_OpenCL:
  case Lang_OBJCXX:
    llvm_unreachable("Not implemented yet!");
  }
  return BasicArgs;
}

} // end namespace ast_matchers
} // end namespace clang
