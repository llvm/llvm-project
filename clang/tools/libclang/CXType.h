//===- CXTypes.h - Routines for manipulating CXTypes ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXCursors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CXTYPES_H
#define LLVM_CLANG_CXTYPES_H

#include "clang-c/Index.h"
#include "clang/AST/Type.h"

namespace clang {
  
class ASTUnit;
  
namespace cxtype {
  
CXType MakeCXType(QualType T, CXTranslationUnit TU);
  
}} // end namespace clang::cxtype
#endif
