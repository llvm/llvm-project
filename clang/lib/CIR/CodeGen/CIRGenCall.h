//===----- CIRGenCall.h - Encapsulate calling convention details ----------===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CIRGENCALL_H
#define LLVM_CLANG_LIB_CODEGEN_CIRGENCALL_H

#include "CIRGenValue.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Type.h"

#include "llvm/ADT/SmallVector.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/IR/BuiltinOps.h"

namespace cir {
class CIRGenFunction;

/// Abstract information about a function or function prototype.
class CIRGenCalleeInfo {
  const clang::FunctionProtoType *CalleeProtoTy;
  clang::GlobalDecl CalleeDecl;

public:
  explicit CIRGenCalleeInfo() : CalleeProtoTy(nullptr), CalleeDecl() {}
  CIRGenCalleeInfo(const clang::FunctionProtoType *calleeProtoTy,
                   clang::GlobalDecl calleeDecl)
      : CalleeProtoTy(calleeProtoTy), CalleeDecl(calleeDecl) {}
  CIRGenCalleeInfo(const clang::FunctionProtoType *calleeProtoTy)
      : CalleeProtoTy(calleeProtoTy) {}
  CIRGenCalleeInfo(clang::GlobalDecl calleeDecl)
      : CalleeProtoTy(nullptr), CalleeDecl(calleeDecl) {}

  const clang::FunctionProtoType *getCalleeFunctionProtoType() const {
    return CalleeProtoTy;
  }
  const clang::GlobalDecl getCalleeDecl() const { return CalleeDecl; }
};

/// All available information about a concrete callee.
class CIRGenCallee {
  enum class SpecialKind : uintptr_t {
    Invalid,
    Builtin,
    PsuedoDestructor,
    Virtual,

    Last = Virtual
  };

  struct BuiltinInfoStorage {
    const clang::FunctionDecl *Decl;
    unsigned ID;
  };
  struct PseudoDestructorInfoStorage {
    const clang::CXXPseudoDestructorExpr *Expr;
  };
  struct VirtualInfoStorage {
    const clang::CallExpr *CE;
    clang::GlobalDecl MD;
    Address Addr;
    mlir::cir::FuncType FTy;
  };

  SpecialKind KindOrFunctionPointer;

  union {
    CIRGenCalleeInfo AbstractInfo;
    BuiltinInfoStorage BuiltinInfo;
    PseudoDestructorInfoStorage PseudoDestructorInfo;
    VirtualInfoStorage VirtualInfo;
  };

  explicit CIRGenCallee(SpecialKind kind) : KindOrFunctionPointer(kind) {}

public:
  CIRGenCallee() : KindOrFunctionPointer(SpecialKind::Invalid) {}

  // Construct a callee. Call this constructor directly when this isn't a direct
  // call.
  CIRGenCallee(const CIRGenCalleeInfo &abstractInfo,
               mlir::Operation *functionPtr)
      : KindOrFunctionPointer(
            SpecialKind(reinterpret_cast<uintptr_t>(functionPtr))) {
    AbstractInfo = abstractInfo;
    assert(functionPtr && "configuring callee without function pointer");
    // TODO: codegen asserts functionPtr is a pointer
    // TODO: codegen asserts functionPtr is either an opaque pointer type or a
    // pointer to a function
  }

  static CIRGenCallee
  forDirect(mlir::Operation *functionPtr,
            const CIRGenCalleeInfo &abstractInfo = CIRGenCalleeInfo()) {
    return CIRGenCallee(abstractInfo, functionPtr);
  }

  bool isBuiltin() const {
    return KindOrFunctionPointer == SpecialKind::Builtin;
  }

  const clang::FunctionDecl *getBuiltinDecl() const {
    assert(isBuiltin());
    return BuiltinInfo.Decl;
  }
  unsigned getBuiltinID() const {
    assert(isBuiltin());
    return BuiltinInfo.ID;
  }

  static CIRGenCallee forBuiltin(unsigned builtinID,
                                 const clang::FunctionDecl *builtinDecl) {
    CIRGenCallee result(SpecialKind::Builtin);
    result.BuiltinInfo.Decl = builtinDecl;
    result.BuiltinInfo.ID = builtinID;
    return result;
  }

  bool isPsuedoDestructor() const {
    return KindOrFunctionPointer == SpecialKind::PsuedoDestructor;
  }

  bool isOrdinary() const {
    return uintptr_t(KindOrFunctionPointer) > uintptr_t(SpecialKind::Last);
  }

  /// If this is a delayed callee computation of some sort, prepare a concrete
  /// callee
  CIRGenCallee prepareConcreteCallee(CIRGenFunction &CGF) const;

  mlir::Operation *getFunctionPointer() const {
    assert(isOrdinary());
    return reinterpret_cast<mlir::Operation *>(KindOrFunctionPointer);
  }

  CIRGenCalleeInfo getAbstractInfo() const {
    if (isVirtual())
      return VirtualInfo.MD;
    assert(isOrdinary());
    return AbstractInfo;
  }

  bool isVirtual() const {
    return KindOrFunctionPointer == SpecialKind::Virtual;
  }

  static CIRGenCallee forVirtual(const clang::CallExpr *CE,
                                 clang::GlobalDecl MD, Address Addr,
                                 mlir::cir::FuncType FTy) {
    CIRGenCallee result(SpecialKind::Virtual);
    result.VirtualInfo.CE = CE;
    result.VirtualInfo.MD = MD;
    result.VirtualInfo.Addr = Addr;
    result.VirtualInfo.FTy = FTy;
    return result;
  }

  const clang::CallExpr *getVirtualCallExpr() const {
    assert(isVirtual());
    return VirtualInfo.CE;
  }

  clang::GlobalDecl getVirtualMethodDecl() const {
    assert(isVirtual());
    return VirtualInfo.MD;
  }
  Address getThisAddress() const {
    assert(isVirtual());
    return VirtualInfo.Addr;
  }
  mlir::cir::FuncType getVirtualFunctionType() const {
    assert(isVirtual());
    return VirtualInfo.FTy;
  }

  void setFunctionPointer(mlir::Operation *functionPtr) {
    assert(isOrdinary());
    KindOrFunctionPointer =
        SpecialKind(reinterpret_cast<uintptr_t>(functionPtr));
  }
};

struct CallArg {
private:
  union {
    RValue RV;
    LValue LV; /// This argument is semantically a load from this l-value
  };
  bool HasLV;

  /// A data-flow flag to make sure getRValue and/or copyInto are not
  /// called twice for duplicated IR emission.
  mutable bool IsUsed;

public:
  clang::QualType Ty;
  CallArg(RValue rv, clang::QualType ty)
      : RV(rv), HasLV(false), IsUsed(false), Ty(ty) {
    (void)IsUsed;
  }
  CallArg(LValue lv, clang::QualType ty)
      : LV(lv), HasLV(true), IsUsed(false), Ty(ty) {}

  /// \returns an independent RValue. If the CallArg contains an LValue,
  /// a temporary copy is returned.
  RValue getRValue(CIRGenFunction &CGF, mlir::Location loc) const;

  bool hasLValue() const { return HasLV; }

  LValue getKnownLValue() const {
    assert(HasLV && !IsUsed);
    return LV;
  }

  RValue getKnownRValue() const {
    assert(!HasLV && !IsUsed);
    return RV;
  }

  bool isAggregate() const { return HasLV || RV.isAggregate(); }
};

class CallArgList : public llvm::SmallVector<CallArg, 8> {
public:
  CallArgList() {}

  struct Writeback {
    LValue Source;
  };

  void add(RValue rvalue, clang::QualType type) {
    push_back(CallArg(rvalue, type));
  }

  void addUncopiedAggregate(LValue LV, clang::QualType type) {
    push_back(CallArg(LV, type));
  }

  /// Add all the arguments from another CallArgList to this one. After doing
  /// this, the old CallArgList retains its list of arguments, but must not
  /// be used to emit a call.
  void addFrom(const CallArgList &other) {
    insert(end(), other.begin(), other.end());
    // TODO: Writebacks, CleanupsToDeactivate, StackBase???
  }
};

/// Type for representing both the decl and type of parameters to a function.
/// The decl must be either a ParmVarDecl or ImplicitParamDecl.
class FunctionArgList : public llvm::SmallVector<const clang::VarDecl *, 16> {};

/// Contains the address where the return value of a function can be stored, and
/// whether the address is volatile or not.
class ReturnValueSlot {
  Address Addr = Address::invalid();

  // Return value slot flags
  unsigned IsVolatile : 1;
  unsigned IsUnused : 1;
  unsigned IsExternallyDestructed : 1;

public:
  ReturnValueSlot()
      : IsVolatile(false), IsUnused(false), IsExternallyDestructed(false) {}
  ReturnValueSlot(Address Addr, bool IsVolatile, bool IsUnused = false,
                  bool IsExternallyDestructed = false)
      : Addr(Addr), IsVolatile(IsVolatile), IsUnused(IsUnused),
        IsExternallyDestructed(IsExternallyDestructed) {}

  bool isNull() const { return !Addr.isValid(); }
  bool isVolatile() const { return IsVolatile; }
  Address getValue() const { return Addr; }
  bool isUnused() const { return IsUnused; }
  bool isExternallyDestructed() const { return IsExternallyDestructed; }
};

enum class FnInfoOpts {
  None = 0,
  IsInstanceMethod = 1 << 0,
  IsChainCall = 1 << 1,
  IsDelegateCall = 1 << 2,
};

} // namespace cir

#endif
