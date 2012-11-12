//===--- SemaInternal.h - Internal Sema Interfaces --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides common API and #includes for the internal
// implementation of Sema.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMA_INTERNAL_H
#define LLVM_CLANG_SEMA_SEMA_INTERNAL_H

#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/ASTContext.h"

namespace clang {

inline PartialDiagnostic Sema::PDiag(unsigned DiagID) {
  return PartialDiagnostic(DiagID, Context.getDiagAllocator());
}

}

#endif
