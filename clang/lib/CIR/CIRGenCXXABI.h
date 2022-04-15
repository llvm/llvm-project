//===----- CIRGenCXXABI.h - Interface to C++ ABIs ---------------*- C++ -*-===//
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

#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/Mangle.h"

namespace cir {

class CIRGenFunction;
class CIRGenFunctionInfo;

/// Implements C++ ABI-specific code generation functions.
class CIRGenCXXABI {
protected:
  cir::CIRGenModule &CGM;
  std::unique_ptr<clang::MangleContext> MangleCtx;

  CIRGenCXXABI(CIRGenModule &CGM)
      : CGM{CGM}, MangleCtx(CGM.getASTContext().createMangleContext()) {}

  clang::ASTContext &getContext() const { return CGM.getASTContext(); }

public:
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

  /// Additional implicit arguments to add to the beginning (Prefix) and end
  /// (Suffix) of a constructor / destructor arg list.
  ///
  /// Note that Prefix should actually be inserted *after* the first existing
  /// arg; `this` arguments always come first.
  struct AddedStructorArgs {
    struct Arg {
      mlir::Value Value;
      clang::QualType Type;
    };
    llvm::SmallVector<Arg, 1> Prefix;
    llvm::SmallVector<Arg, 1> Suffix;
    AddedStructorArgs() = default;
    AddedStructorArgs(llvm::SmallVector<Arg, 1> P, llvm::SmallVector<Arg, 1> S)
        : Prefix(std::move(P)), Suffix(std::move(S)) {}
    static AddedStructorArgs prefix(llvm::SmallVector<Arg, 1> Args) {
      return {std::move(Args), {}};
    }
    static AddedStructorArgs suffix(llvm::SmallVector<Arg, 1> Args) {
      return {{}, std::move(Args)};
    }
  };

  AddedStructorArgCounts
  addImplicitConstructorArgs(CIRGenFunction &CGF,
                             const clang::CXXConstructorDecl *D,
                             clang::CXXCtorType Type, bool ForVirtualBase,
                             bool Delegating, CallArgList &Args);

  virtual AddedStructorArgs getImplicitConstructorArgs(
      CIRGenFunction &CGF, const clang::CXXConstructorDecl *D,
      clang::CXXCtorType Type, bool ForVirtualBase, bool Delegating) = 0;

  /// Return whether the given global decl needs a VTT parameter.
  virtual bool NeedsVTTParameter(clang::GlobalDecl GD);

  /// If the C++ ABI requires the given type be returned in a particular way,
  /// this method sets RetAI and returns true.
  virtual bool classifyReturnType(CIRGenFunctionInfo &FI) const = 0;

  /// Gets the mangle context.
  clang::MangleContext &getMangleContext() { return *MangleCtx; }

  /// Specify how one should pass an argument of a record type.
  enum class RecordArgABI {
    /// Pass it using the normal C aggregate rules for the ABI, potentially
    /// introducing extra copies and passing some or all of it in registers.
    Default = 0,

    /// Pass it on the stack using its defined layout. The argument must be
    /// evaluated directly into the correct stack position in the arguments
    /// area, and the call machinery must not move it or introduce extra copies.
    DirectInMemory,

    /// Pass it as a pointer to temporary memory.
    Indirect
  };

  /// Returns how an argument of the given record type should be passed.
  virtual RecordArgABI
  getRecordArgABI(const clang::CXXRecordDecl *RD) const = 0;

  /// Checks if ABI requires to initialize vptrs for given dynamic class.
  virtual bool
  doStructorsInitializeVPtrs(const clang::CXXRecordDecl *VTableClass) = 0;

  /// Returns true if the given constructor or destructor is one of the kinds
  /// that the ABI says returns 'this' (only applies when called non-virtually
  /// for destructors).
  ///
  /// There currently is no way to indicate if a destructor returns 'this' when
  /// called virtually, and CIR generation does not support this case.
  virtual bool HasThisReturn(clang::GlobalDecl GD) const { return false; }

  virtual bool hasMostDerivedReturn(clang::GlobalDecl GD) const {
    return false;
  }

  virtual ~CIRGenCXXABI();
};

/// Creates and Itanium-family ABI
CIRGenCXXABI *CreateCIRGenItaniumCXXABI(CIRGenModule &CGM);

} // namespace cir

#endif
