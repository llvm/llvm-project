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
  CIRGenCalleeInfo(clang::GlobalDecl calleeDecl)
      : calleeProtoTy(nullptr), calleeDecl(calleeDecl) {}

  const clang::FunctionProtoType *getCalleeFunctionProtoType() const {
    return calleeProtoTy;
  }
  clang::GlobalDecl getCalleeDecl() const { return calleeDecl; }
};

class CIRGenCallee {
  enum class SpecialKind : uintptr_t {
    Invalid,
    Builtin,
    PseudoDestructor,
    Virtual,

    Last = Virtual
  };

  struct BuiltinInfoStorage {
    const clang::FunctionDecl *decl;
    unsigned id;
  };
  struct PseudoDestructorInfoStorage {
    const clang::CXXPseudoDestructorExpr *expr;
  };
  struct VirtualInfoStorage {
    const clang::CallExpr *ce;
    clang::GlobalDecl md;
    Address addr;
    cir::FuncType fTy;
  };

  SpecialKind kindOrFunctionPtr;

  union {
    CIRGenCalleeInfo abstractInfo;
    BuiltinInfoStorage builtinInfo;
    PseudoDestructorInfoStorage pseudoDestructorInfo;
    VirtualInfoStorage virtualInfo;
  };

  explicit CIRGenCallee(SpecialKind kind) : kindOrFunctionPtr(kind) {}

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

  bool isBuiltin() const { return kindOrFunctionPtr == SpecialKind::Builtin; }

  const clang::FunctionDecl *getBuiltinDecl() const {
    assert(isBuiltin());
    return builtinInfo.decl;
  }
  unsigned getBuiltinID() const {
    assert(isBuiltin());
    return builtinInfo.id;
  }

  static CIRGenCallee forBuiltin(unsigned builtinID,
                                 const clang::FunctionDecl *builtinDecl) {
    CIRGenCallee result(SpecialKind::Builtin);
    result.builtinInfo.decl = builtinDecl;
    result.builtinInfo.id = builtinID;
    return result;
  }

  static CIRGenCallee
  forPseudoDestructor(const clang::CXXPseudoDestructorExpr *expr) {
    CIRGenCallee result(SpecialKind::PseudoDestructor);
    result.pseudoDestructorInfo.expr = expr;
    return result;
  }

  bool isPseudoDestructor() const {
    return kindOrFunctionPtr == SpecialKind::PseudoDestructor;
  }

  const CXXPseudoDestructorExpr *getPseudoDestructorExpr() const {
    assert(isPseudoDestructor());
    return pseudoDestructorInfo.expr;
  }

  bool isOrdinary() const {
    return uintptr_t(kindOrFunctionPtr) > uintptr_t(SpecialKind::Last);
  }

  /// If this is a delayed callee computation of some sort, prepare a concrete
  /// callee
  CIRGenCallee prepareConcreteCallee(CIRGenFunction &cgf) const;

  CIRGenCalleeInfo getAbstractInfo() const {
    if (isVirtual())
      return virtualInfo.md;
    assert(isOrdinary());
    return abstractInfo;
  }

  mlir::Operation *getFunctionPointer() const {
    assert(isOrdinary());
    return reinterpret_cast<mlir::Operation *>(kindOrFunctionPtr);
  }

  bool isVirtual() const { return kindOrFunctionPtr == SpecialKind::Virtual; }

  static CIRGenCallee forVirtual(const clang::CallExpr *ce,
                                 clang::GlobalDecl md, Address addr,
                                 cir::FuncType fTy) {
    CIRGenCallee result(SpecialKind::Virtual);
    result.virtualInfo.ce = ce;
    result.virtualInfo.md = md;
    result.virtualInfo.addr = addr;
    result.virtualInfo.fTy = fTy;
    return result;
  }

  const clang::CallExpr *getVirtualCallExpr() const {
    assert(isVirtual());
    return virtualInfo.ce;
  }

  clang::GlobalDecl getVirtualMethodDecl() const {
    assert(isVirtual());
    return virtualInfo.md;
  }

  Address getThisAddress() const {
    assert(isVirtual());
    return virtualInfo.addr;
  }

  cir::FuncType getVirtualFunctionType() const {
    assert(isVirtual());
    return virtualInfo.fTy;
  }

  void setFunctionPointer(mlir::Operation *functionPtr) {
    assert(isOrdinary());
    kindOrFunctionPtr = SpecialKind(reinterpret_cast<uintptr_t>(functionPtr));
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
  [[maybe_unused]] mutable bool isUsed;

public:
  clang::QualType ty;

  CallArg(RValue rv, clang::QualType ty)
      : rv(rv), hasLV(false), isUsed(false), ty(ty) {}

  CallArg(LValue lv, clang::QualType ty)
      : lv(lv), hasLV(true), isUsed(false), ty(ty) {}

  bool hasLValue() const { return hasLV; }

  LValue getKnownLValue() const {
    assert(hasLV && !isUsed);
    return lv;
  }

  RValue getKnownRValue() const {
    assert(!hasLV && !isUsed);
    return rv;
  }

  bool isAggregate() const { return hasLV || rv.isAggregate(); }
};

class CallArgList : public llvm::SmallVector<CallArg, 8> {
public:
  void add(RValue rvalue, clang::QualType type) { emplace_back(rvalue, type); }

  void addUncopiedAggregate(LValue lvalue, clang::QualType type) {
    emplace_back(lvalue, type);
  }

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
class ReturnValueSlot {
  Address addr = Address::invalid();

public:
  ReturnValueSlot() = default;
  ReturnValueSlot(Address addr) : addr(addr) {}

  Address getValue() const { return addr; }
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CODEGEN_CIRGENCALL_H
