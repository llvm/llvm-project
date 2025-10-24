//===----- CXXABIShared.h - Shared C++ ABI Base Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a base class for C++ ABI functionality that can be shared
// between LLVM IR codegen and CIR codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENSHARED_CXXABISHARED_H
#define LLVM_CLANG_CODEGENSHARED_CXXABISHARED_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/ABI.h"

namespace clang {
class ASTContext;

/// Implements C++ ABI functionality that can be shared between LLVM IR codegen
/// and CIR codegen.
class CXXABIShared {
protected:
  ASTContext &Context;
  std::unique_ptr<MangleContext> MangleCtx;

  CXXABIShared(ASTContext &Context)
      : Context(Context), MangleCtx(Context.createMangleContext()) {}

public:
  virtual ~CXXABIShared() = default;

  /// Similar to AddedStructorArgs, but only notes the number of additional
  /// arguments.
  struct AddedStructorArgCounts {
    unsigned Prefix = 0;
    unsigned Suffix = 0;
    AddedStructorArgCounts() = default;
    AddedStructorArgCounts(unsigned P, unsigned S) : Prefix(P), Suffix(S) {}
    static AddedStructorArgCounts prefix(unsigned N) { return {N, 0}; }
    static AddedStructorArgCounts suffix(unsigned N) { return {0, N}; }
  };

  /// Get the AST context.
  ASTContext &getContext() const { return Context; }

  /// Gets the mangle context.
  MangleContext &getMangleContext() { return *MangleCtx; }

  /// Determine whether there's something special about the rules of
  /// the ABI tell us that 'this' is a complete object within the
  /// given function.  Obvious common logic like being defined on a
  /// final class will have been taken care of by the caller.
  virtual bool isThisCompleteObject(GlobalDecl GD) const = 0;

  /// Returns true if the most-derived return value should be returned.
  virtual bool hasMostDerivedReturn(GlobalDecl GD) const { return false; }

  /// Return whether the given global decl needs a VTT parameter.
  virtual bool NeedsVTTParameter(GlobalDecl GD) const { return false; }

  /// Returns true if the given constructor or destructor is one of the
  /// kinds that the ABI says returns 'this' (only applies when called
  /// non-virtually for destructors).
  ///
  /// There currently is no way to indicate if a destructor returns 'this'
  /// when called virtually, and code generation does not support the case.
  /// Returns true if the given constructor or destructor is one of the
  /// kinds that the ABI says returns 'this' (only applies when called
  /// non-virtually for destructors).
  ///
  /// There currently is no way to indicate if a destructor returns 'this'
  /// when called virtually, and code generation does not support the case.
  virtual bool HasThisReturn(GlobalDecl GD) const {
    if (isa<CXXConstructorDecl>(GD.getDecl()) ||
        (isa<CXXDestructorDecl>(GD.getDecl()) &&
         GD.getDtorType() != Dtor_Deleting))
      return constructorsAndDestructorsReturnThis();
    return false;
  }

  /// Returns true if the given destructor type should be emitted as a linkonce
  /// delegating thunk, regardless of whether the dtor is defined in this TU or
  /// not.
  virtual bool useThunkForDtorVariant(const CXXDestructorDecl *Dtor,
                                      CXXDtorType DT) const = 0;

protected:
  virtual bool constructorsAndDestructorsReturnThis() const = 0;
};

} // namespace clang

#endif // LLVM_CLANG_CODEGENSHARED_CXXABISHARED_H
