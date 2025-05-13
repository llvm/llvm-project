//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENCXXABI_H
#define LLVM_CLANG_LIB_CIR_CIRGENCXXABI_H

#include "CIRGenModule.h"

#include "clang/AST/Mangle.h"

namespace clang::CIRGen {

/// Implements C++ ABI-specific code generation functions.
class CIRGenCXXABI {
protected:
  CIRGenModule &cgm;
  std::unique_ptr<clang::MangleContext> mangleContext;

public:
  // TODO(cir): make this protected when target-specific CIRGenCXXABIs are
  // implemented.
  CIRGenCXXABI(CIRGenModule &cgm)
      : cgm(cgm), mangleContext(cgm.getASTContext().createMangleContext()) {}
  ~CIRGenCXXABI();

public:
  /// Gets the mangle context.
  clang::MangleContext &getMangleContext() { return *mangleContext; }
};

/// Creates and Itanium-family ABI
CIRGenCXXABI *CreateCIRGenItaniumCXXABI(CIRGenModule &cgm);

} // namespace clang::CIRGen

#endif
