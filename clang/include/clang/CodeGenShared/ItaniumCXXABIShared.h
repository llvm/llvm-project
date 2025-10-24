//===--- ItaniumCXXABIShared.h - Itanium C++ ABI Shared Base ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a base class for Itanium C++ ABI implementations that can
// be shared between LLVM IR codegen and CIR codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENSHARED_ITANIUMCXXABISHARED_H
#define LLVM_CLANG_CODEGENSHARED_ITANIUMCXXABISHARED_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/ABI.h"
#include <utility>

namespace clang {

template <typename BaseT> class ItaniumCXXABIShared : public BaseT {
protected:
  /// Constructor forwarding to the base ABI class.
  template <typename... Args>
  ItaniumCXXABIShared(Args &&...args) : BaseT(std::forward<Args>(args)...) {}

public:
  /// Determine whether there's something special about the rules of
  /// the ABI tell us that 'this' is a complete object within the
  /// given function.  Obvious common logic like being defined on a
  /// final class will have been taken care of by the caller.
  bool isThisCompleteObject(GlobalDecl GD) const override {
    // The Itanium ABI has separate complete-object vs. base-object
    // variants of both constructors and destructors.
    if (isa<CXXDestructorDecl>(GD.getDecl())) {
      switch (GD.getDtorType()) {
      case Dtor_Complete:
      case Dtor_Deleting:
        return true;

      case Dtor_Base:
        return false;

      case Dtor_Comdat:
        llvm_unreachable("emitting dtor comdat as function?");
      case Dtor_Unified:
        llvm_unreachable("emitting unified dtor as function?");
      }
      llvm_unreachable("bad dtor kind");
    }
    if (isa<CXXConstructorDecl>(GD.getDecl())) {
      switch (GD.getCtorType()) {
      case Ctor_Complete:
        return true;

      case Ctor_Base:
        return false;

      case Ctor_CopyingClosure:
      case Ctor_DefaultClosure:
        llvm_unreachable("closure ctors in Itanium ABI?");

      case Ctor_Comdat:
        llvm_unreachable("emitting ctor comdat as function?");

      case Ctor_Unified:
        llvm_unreachable("emitting unified ctor as function?");
      }
      llvm_unreachable("bad ctor kind");
    }

    // No other kinds.
    return false;
  }

  bool NeedsVTTParameter(GlobalDecl GD) const override {
    const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());

    // We don't have any virtual bases, just return early.
    if (!MD->getParent()->getNumVBases())
      return false;

    // Check if we have a base constructor.
    if (isa<CXXConstructorDecl>(MD) && GD.getCtorType() == Ctor_Base)
      return true;

    // Check if we have a base destructor.
    if (isa<CXXDestructorDecl>(MD) && GD.getDtorType() == Dtor_Base)
      return true;

    return false;
  }

  /// Returns true if the given destructor type should be emitted as a linkonce
  /// delegating thunk, regardless of whether the dtor is defined in this TU or
  /// not.
  ///
  /// Itanium does not emit any destructor variant as an inline thunk.
  /// Delegating may occur as an optimization, but all variants are either
  /// emitted with external linkage or as linkonce if they are inline and used.
  bool useThunkForDtorVariant(const CXXDestructorDecl *Dtor,
                              CXXDtorType DT) const override {
    return false;
  }

  /// Returns true if this ABI initializes vptrs in constructors/destructors.
  /// For Itanium, this is always true.
  bool doStructorsInitializeVPtrs(const CXXRecordDecl *VTableClass) override {
    return true;
  }
};

} // namespace clang

#endif // LLVM_CLANG_CODEGENSHARED_ITANIUMCXXABISHARED_H
