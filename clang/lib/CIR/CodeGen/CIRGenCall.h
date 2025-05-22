//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CODEGEN_CIRGENCALL_H
#define CLANG_LIB_CODEGEN_CIRGENCALL_H

#include "CIRGenValue.h"
#include "mlir/IR/Operation.h"
#include "clang/AST/GlobalDecl.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::CIRGen {

class CIRGenFunction;

/// Abstract information about a function or function prototype.
class CIRGenCalleeInfo {
  const clang::FunctionProtoType *calleeProtoTy;
  clang::GlobalDecl calleeDecl;

public:
  explicit CIRGenCalleeInfo() : calleeProtoTy(nullptr), calleeDecl() {}
  CIRGenCalleeInfo(const clang::FunctionProtoType *calleeProtoTy,
                   clang::GlobalDecl calleeDecl)
      : calleeProtoTy(calleeProtoTy), calleeDecl(calleeDecl) {}
  CIRGenCalleeInfo(clang::GlobalDecl calleeDecl) : calleeDecl(calleeDecl) {}

  const clang::FunctionProtoType *getCalleeFunctionProtoType() const {
    return calleeProtoTy;
  }
  clang::GlobalDecl getCalleeDecl() const { return calleeDecl; }
};

class CIRGenCallee {
  enum class SpecialKind : uintptr_t {
    Invalid,

    Last = Invalid,
  };

  SpecialKind kindOrFunctionPtr;

  union {
    CIRGenCalleeInfo abstractInfo;
  };

public:
  CIRGenCallee() : kindOrFunctionPtr(SpecialKind::Invalid) {}

  CIRGenCallee(const CIRGenCalleeInfo &abstractInfo, mlir::Operation *funcPtr)
      : kindOrFunctionPtr(SpecialKind(reinterpret_cast<uintptr_t>(funcPtr))),
        abstractInfo(abstractInfo) {
    assert(funcPtr && "configuring callee without function pointer");
  }

  static CIRGenCallee
  forDirect(mlir::Operation *funcPtr,
            const CIRGenCalleeInfo &abstractInfo = CIRGenCalleeInfo()) {
    return CIRGenCallee(abstractInfo, funcPtr);
  }

  bool isOrdinary() const {
    return uintptr_t(kindOrFunctionPtr) > uintptr_t(SpecialKind::Last);
  }

  /// If this is a delayed callee computation of some sort, prepare a concrete
  /// callee
  CIRGenCallee prepareConcreteCallee(CIRGenFunction &cgf) const;

  mlir::Operation *getFunctionPointer() const {
    assert(isOrdinary());
    return reinterpret_cast<mlir::Operation *>(kindOrFunctionPtr);
  }
};

/// Type for representing both the decl and type of parameters to a function.
/// The decl must be either a ParmVarDecl or ImplicitParamDecl.
class FunctionArgList : public llvm::SmallVector<const clang::VarDecl *, 16> {};

struct CallArg {
private:
  union {
    RValue rv;
    LValue lv; // This argument is semantically a load from this l-value
  };
  bool hasLV;

  /// A data-flow flag to make sure getRValue and/or copyInto are not
  /// called twice for duplicated IR emission.
  mutable bool isUsed;

public:
  clang::QualType ty;

  CallArg(RValue rv, clang::QualType ty)
      : rv(rv), hasLV(false), isUsed(false), ty(ty) {}

  bool hasLValue() const { return hasLV; }

  RValue getKnownRValue() const {
    assert(!hasLV && !isUsed);
    return rv;
  }

  bool isAggregate() const { return hasLV || rv.isAggregate(); }
};

class CallArgList : public llvm::SmallVector<CallArg, 8> {
public:
  void add(RValue rvalue, clang::QualType type) { emplace_back(rvalue, type); }

  /// Add all the arguments from another CallArgList to this one. After doing
  /// this, the old CallArgList retains its list of arguments, but must not
  /// be used to emit a call.
  void addFrom(const CallArgList &other) {
    insert(end(), other.begin(), other.end());
    // Classic codegen has handling for these here. We may not need it here for
    // CIR, but if not we should implement equivalent handling in lowering.
    assert(!cir::MissingFeatures::writebacks());
    assert(!cir::MissingFeatures::cleanupsToDeactivate());
    assert(!cir::MissingFeatures::stackBase());
  }
};

/// Contains the address where the return value of a function can be stored, and
/// whether the address is volatile or not.
class ReturnValueSlot {};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CODEGEN_CIRGENCALL_H
