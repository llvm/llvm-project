//===--- CIRGenCall.cpp - Encapsulate calling convention details ----------===//
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

#include "CIRGenBuilder.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"
#include "TargetInfo.h"

#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

#include "UnimplementedFeatureGuarding.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"

using namespace cir;
using namespace clang;

CIRGenFunctionInfo *CIRGenFunctionInfo::create(
    unsigned cirCC, bool instanceMethod, bool chainCall,
    const FunctionType::ExtInfo &info,
    llvm::ArrayRef<ExtParameterInfo> paramInfos, CanQualType resultType,
    llvm::ArrayRef<CanQualType> argTypes, RequiredArgs required) {
  assert(paramInfos.empty() || paramInfos.size() == argTypes.size());
  assert(!required.allowsOptionalArgs() ||
         required.getNumRequiredArgs() <= argTypes.size());

  void *buffer = operator new(totalSizeToAlloc<ArgInfo, ExtParameterInfo>(
      argTypes.size() + 1, paramInfos.size()));

  CIRGenFunctionInfo *FI = new (buffer) CIRGenFunctionInfo();
  FI->CallingConvention = cirCC;
  FI->EffectiveCallingConvention = cirCC;
  FI->ASTCallingConvention = info.getCC();
  FI->InstanceMethod = instanceMethod;
  FI->ChainCall = chainCall;
  FI->CmseNSCall = info.getCmseNSCall();
  FI->NoReturn = info.getNoReturn();
  FI->ReturnsRetained = info.getProducesResult();
  FI->NoCallerSavedRegs = info.getNoCallerSavedRegs();
  FI->NoCfCheck = info.getNoCfCheck();
  FI->Required = required;
  FI->HasRegParm = info.getHasRegParm();
  FI->RegParm = info.getRegParm();
  FI->ArgStruct = nullptr;
  FI->ArgStructAlign = 0;
  FI->NumArgs = argTypes.size();
  FI->HasExtParameterInfos = !paramInfos.empty();
  FI->getArgsBuffer()[0].type = resultType;
  for (unsigned i = 0; i < argTypes.size(); ++i)
    FI->getArgsBuffer()[i + 1].type = argTypes[i];
  for (unsigned i = 0; i < paramInfos.size(); ++i)
    FI->getExtParameterInfosBuffer()[i] = paramInfos[i];

  return FI;
}

namespace {

/// Encapsulates information about the way function arguments from
/// CIRGenFunctionInfo should be passed to actual CIR function.
class ClangToCIRArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned InallocaArgNo;
  unsigned SRetArgNo;
  unsigned TotalCIRArgs;

  /// Arguments of CIR function corresponding to single Clang argument.
  struct CIRArgs {
    unsigned PaddingArgIndex = 0;
    // Argument is expanded to CIR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex = 0;
    unsigned NumberOfArgs = 0;

    CIRArgs()
        : PaddingArgIndex(InvalidIndex), FirstArgIndex(InvalidIndex),
          NumberOfArgs(0) {}
  };

  SmallVector<CIRArgs, 8> ArgInfo;

public:
  ClangToCIRArgMapping(const ASTContext &Context, const CIRGenFunctionInfo &FI,
                       bool OnlyRequiredArgs = false)
      : InallocaArgNo(InvalidIndex), SRetArgNo(InvalidIndex), TotalCIRArgs(0),
        ArgInfo(OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size()) {
    construct(Context, FI, OnlyRequiredArgs);
  }

  bool hasSRetArg() const { return SRetArgNo != InvalidIndex; }

  bool hasInallocaArg() const { return InallocaArgNo != InvalidIndex; }

  unsigned totalCIRArgs() const { return TotalCIRArgs; }

  bool hasPaddingArg(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return ArgInfo[ArgNo].PaddingArgIndex != InvalidIndex;
  }

  /// Returns index of first CIR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getCIRArgs(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return std::make_pair(ArgInfo[ArgNo].FirstArgIndex,
                          ArgInfo[ArgNo].NumberOfArgs);
  }

private:
  void construct(const ASTContext &Context, const CIRGenFunctionInfo &FI,
                 bool OnlyRequiredArgs);
};

void ClangToCIRArgMapping::construct(const ASTContext &Context,
                                     const CIRGenFunctionInfo &FI,
                                     bool OnlyRequiredArgs) {
  unsigned CIRArgNo = 0;
  bool SwapThisWithSRet = false;
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  assert(RetAI.getKind() != ABIArgInfo::Indirect && "NYI");

  unsigned ArgNo = 0;
  unsigned NumArgs = OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size();
  for (CIRGenFunctionInfo::const_arg_iterator I = FI.arg_begin();
       ArgNo < NumArgs; ++I, ++ArgNo) {
    assert(I != FI.arg_end());
    const ABIArgInfo &AI = I->info;
    // Collect data about CIR arguments corresponding to Clang argument ArgNo.
    auto &CIRArgs = ArgInfo[ArgNo];

    assert(!AI.getPaddingType() && "NYI");

    switch (AI.getKind()) {
    default:
      llvm_unreachable("NYI");
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      // Postpone splitting structs into elements since this makes it way
      // more complicated for analysis to obtain information on the original
      // arguments.
      //
      // TODO(cir): a LLVM lowering prepare pass should break this down into
      // the appropriated pieces.
      assert(!UnimplementedFeature::constructABIArgDirectExtend());
      CIRArgs.NumberOfArgs = 1;
      break;
    }
    }

    if (CIRArgs.NumberOfArgs > 0) {
      CIRArgs.FirstArgIndex = CIRArgNo;
      CIRArgNo += CIRArgs.NumberOfArgs;
    }

    assert(!SwapThisWithSRet && "NYI");
  }
  assert(ArgNo == ArgInfo.size());

  assert(!FI.usesInAlloca() && "NYI");

  TotalCIRArgs = CIRArgNo;
}

} // namespace

static bool hasInAllocaArgs(CIRGenModule &CGM, CallingConv ExplicitCC,
                            ArrayRef<QualType> ArgTypes) {
  assert(ExplicitCC != CC_Swift && ExplicitCC != CC_SwiftAsync && "Swift NYI");
  assert(!CGM.getTarget().getCXXABI().isMicrosoft() && "MSABI NYI");

  return false;
}

mlir::cir::FuncType CIRGenTypes::GetFunctionType(GlobalDecl GD) {
  const CIRGenFunctionInfo &FI = arrangeGlobalDeclaration(GD);
  return GetFunctionType(FI);
}

mlir::cir::FuncType CIRGenTypes::GetFunctionType(const CIRGenFunctionInfo &FI) {
  bool Inserted = FunctionsBeingProcessed.insert(&FI).second;
  (void)Inserted;
  assert(Inserted && "Recursively being processed?");

  mlir::Type resultType = nullptr;
  const ABIArgInfo &retAI = FI.getReturnInfo();
  switch (retAI.getKind()) {
  case ABIArgInfo::Ignore:
    // TODO(CIR): This should probably be the None type from the builtin
    // dialect.
    resultType = nullptr;
    break;

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    resultType = retAI.getCoerceToType();
    break;

  default:
    assert(false && "NYI");
  }

  ClangToCIRArgMapping CIRFunctionArgs(getContext(), FI, true);
  SmallVector<mlir::Type, 8> ArgTypes(CIRFunctionArgs.totalCIRArgs());

  assert(!CIRFunctionArgs.hasSRetArg() && "NYI");
  assert(!CIRFunctionArgs.hasInallocaArg() && "NYI");

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  CIRGenFunctionInfo::const_arg_iterator it = FI.arg_begin(),
                                         ie = it + FI.getNumRequiredArgs();

  for (; it != ie; ++it, ++ArgNo) {
    const auto &ArgInfo = it->info;

    assert(!CIRFunctionArgs.hasPaddingArg(ArgNo) && "NYI");

    unsigned FirstCIRArg, NumCIRArgs;
    std::tie(FirstCIRArg, NumCIRArgs) = CIRFunctionArgs.getCIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    default:
      llvm_unreachable("NYI");
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      mlir::Type argType = ArgInfo.getCoerceToType();
      // TODO: handle the test against llvm::StructType from codegen
      assert(NumCIRArgs == 1);
      ArgTypes[FirstCIRArg] = argType;
      break;
    }
    }
  }

  bool Erased = FunctionsBeingProcessed.erase(&FI);
  (void)Erased;
  assert(Erased && "Not in set?");

  return mlir::cir::FuncType::get(
      ArgTypes, (resultType ? resultType : Builder.getVoidTy()),
      FI.isVariadic());
}

mlir::cir::FuncType CIRGenTypes::GetFunctionTypeForVTable(GlobalDecl GD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();

  if (!isFuncTypeConvertible(FPT)) {
    llvm_unreachable("NYI");
    // return llvm::StructType::get(getLLVMContext());
  }

  return GetFunctionType(GD);
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &CGF) const {
  if (isVirtual()) {
    const CallExpr *CE = getVirtualCallExpr();
    return CGF.CGM.getCXXABI().getVirtualFunctionPointer(
        CGF, getVirtualMethodDecl(), getThisAddress(), getVirtualFunctionType(),
        CE ? CE->getBeginLoc() : SourceLocation());
  }
  return *this;
}

void CIRGenFunction::buildAggregateStore(mlir::Value Val, Address Dest,
                                         bool DestIsVolatile) {
  // In LLVM codegen:
  // Function to store a first-class aggregate into memory. We prefer to
  // store the elements rather than the aggregate to be more friendly to
  // fast-isel.
  // In CIR codegen:
  // Emit the most simple cir.store possible (e.g. a store for a whole
  // struct), which can later be broken down in other CIR levels (or prior
  // to dialect codegen).
  (void)DestIsVolatile;
  builder.create<mlir::cir::StoreOp>(*currSrcLoc, Val, Dest.getPointer());
}

static Address emitAddressAtOffset(CIRGenFunction &CGF, Address addr,
                                   const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    llvm_unreachable("NYI");
  }
  return addr;
}

RValue CIRGenFunction::buildCall(const CIRGenFunctionInfo &CallInfo,
                                 const CIRGenCallee &Callee,
                                 ReturnValueSlot ReturnValue,
                                 const CallArgList &CallArgs,
                                 mlir::cir::CallOp *callOrInvoke,
                                 bool IsMustTail, mlir::Location loc) {
  auto builder = CGM.getBuilder();
  // FIXME: We no longer need the types from CallArgs; lift up and simplify

  assert(Callee.isOrdinary() || Callee.isVirtual());

  // Handle struct-return functions by passing a pointer to the location that we
  // would like to return info.
  QualType RetTy = CallInfo.getReturnType();
  const auto &RetAI = CallInfo.getReturnInfo();

  mlir::cir::FuncType CIRFuncTy = getTypes().GetFunctionType(CallInfo);

  const Decl *TargetDecl = Callee.getAbstractInfo().getCalleeDecl().getDecl();
  // This is not always tied to a FunctionDecl (e.g. builtins that are xformed
  // into calls to other functions)
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl)) {
    // We can only guarantee that a function is called from the correct
    // context/function based on the appropriate target attributes,
    // so only check in the case where we have both always_inline and target
    // since otherwise we could be making a conditional call after a check for
    // the proper cpu features (and it won't cause code generation issues due to
    // function based code generation).
    if (TargetDecl->hasAttr<AlwaysInlineAttr>() &&
        (TargetDecl->hasAttr<TargetAttr>() ||
         (CurFuncDecl && CurFuncDecl->hasAttr<TargetAttr>()))) {
      // FIXME(cir): somehow refactor this function to use SourceLocation?
      SourceLocation Loc;
      checkTargetFeatures(Loc, FD);
    }

    // Some architectures (such as x86-64) have the ABI changed based on
    // attribute-target/features. Give them a chance to diagnose.
    assert(!UnimplementedFeature::checkFunctionCallABI());
  }

  // TODO: add DNEBUG code

  // 1. Set up the arguments

  // If we're using inalloca, insert the allocation after the stack save.
  // FIXME: Do this earlier rather than hacking it in here!
  Address ArgMemory = Address::invalid();
  assert(!CallInfo.getArgStruct() && "NYI");

  ClangToCIRArgMapping CIRFunctionArgs(CGM.getASTContext(), CallInfo);
  SmallVector<mlir::Value, 16> CIRCallArgs(CIRFunctionArgs.totalCIRArgs());

  // If the call returns a temporary with struct return, create a temporary
  // alloca to hold the result, unless one is given to us.
  assert(!RetAI.isIndirect() && !RetAI.isInAlloca() &&
         !RetAI.isCoerceAndExpand() && "NYI");

  // When passing arguments using temporary allocas, we need to add the
  // appropriate lifetime markers. This vector keeps track of all the lifetime
  // markers that need to be ended right after the call.
  assert(!UnimplementedFeature::shouldEmitLifetimeMarkers() && "NYI");

  // Translate all of the arguments as necessary to match the CIR lowering.
  assert(CallInfo.arg_size() == CallArgs.size() &&
         "Mismatch between function signature & arguments.");
  unsigned ArgNo = 0;
  CIRGenFunctionInfo::const_arg_iterator info_it = CallInfo.arg_begin();
  for (CallArgList::const_iterator I = CallArgs.begin(), E = CallArgs.end();
       I != E; ++I, ++info_it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = info_it->info;

    // Insert a padding argument to ensure proper alignment.
    assert(!CIRFunctionArgs.hasPaddingArg(ArgNo) && "Padding args NYI");

    unsigned FirstCIRArg, NumCIRArgs;
    std::tie(FirstCIRArg, NumCIRArgs) = CIRFunctionArgs.getCIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Direct: {
      if (!ArgInfo.getCoerceToType().isa<mlir::cir::StructType>() &&
          ArgInfo.getCoerceToType() == convertType(info_it->type) &&
          ArgInfo.getDirectOffset() == 0) {
        assert(NumCIRArgs == 1);
        mlir::Value V;
        assert(!I->isAggregate() && "Aggregate NYI");
        V = I->getKnownRValue().getScalarVal();

        assert(CallInfo.getExtParameterInfo(ArgNo).getABI() !=
                   ParameterABI::SwiftErrorResult &&
               "swift NYI");

        // We might have to widen integers, but we should never truncate.
        if (ArgInfo.getCoerceToType() != V.getType() &&
            V.getType().isa<mlir::cir::IntType>())
          llvm_unreachable("NYI");

        // If the argument doesn't match, perform a bitcast to coerce it. This
        // can happen due to trivial type mismatches.
        if (FirstCIRArg < CIRFuncTy.getNumInputs() &&
            V.getType() != CIRFuncTy.getInput(FirstCIRArg))
          V = builder.createBitcast(V, CIRFuncTy.getInput(FirstCIRArg));

        CIRCallArgs[FirstCIRArg] = V;
        break;
      }

      // FIXME: Avoid the conversion through memory if possible.
      Address Src = Address::invalid();
      if (!I->isAggregate()) {
        llvm_unreachable("NYI");
      } else {
        Src = I->hasLValue() ? I->getKnownLValue().getAddress()
                             : I->getKnownRValue().getAggregateAddress();
      }

      // If the value is offset in memory, apply the offset now.
      Src = emitAddressAtOffset(*this, Src, ArgInfo);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      auto STy = dyn_cast<mlir::cir::StructType>(ArgInfo.getCoerceToType());
      if (STy && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        auto SrcTy = Src.getElementType();
        // FIXME(cir): get proper location for each argument.
        auto argLoc = loc;

        // If the source type is smaller than the destination type of the
        // coerce-to logic, copy the source value into a temp alloca the size
        // of the destination type to allow loading all of it. The bits past
        // the source value are left undef.
        // FIXME(cir): add data layout info and compare sizes instead of
        // matching the types.
        //
        // uint64_t SrcSize = CGM.getDataLayout().getTypeAllocSize(SrcTy);
        // uint64_t DstSize = CGM.getDataLayout().getTypeAllocSize(STy);
        // if (SrcSize < DstSize) {
        if (SrcTy != STy)
          llvm_unreachable("NYI");
        else {
          // FIXME(cir): this currently only runs when the types are different,
          // but should be when alloc sizes are different, fix this as soon as
          // datalayout gets introduced.
          Src = builder.createElementBitCast(argLoc, Src, STy);
        }

        // assert(NumCIRArgs == STy.getMembers().size());
        // In LLVMGen: Still only pass the struct without any gaps but mark it
        // as such somehow.
        //
        // In CIRGen: Emit a load from the "whole" struct,
        // which shall be broken later by some lowering step into multiple
        // loads.
        assert(NumCIRArgs == 1 && "dont break up arguments here!");
        CIRCallArgs[FirstCIRArg] = builder.createLoad(argLoc, Src);
      } else {
        llvm_unreachable("NYI");
      }

      break;
    }
    default:
      assert(false && "Only Direct support so far");
    }
  }

  const CIRGenCallee &ConcreteCallee = Callee.prepareConcreteCallee(*this);
  auto CalleePtr = ConcreteCallee.getFunctionPointer();

  // If we're using inalloca, set up that argument.
  assert(!ArgMemory.isValid() && "inalloca NYI");

  // 2. Prepare the function pointer.

  // TODO: simplifyVariadicCallee

  // 3. Perform the actual call.

  // TODO: Deactivate any cleanups that we're supposed to do immediately before
  // the call.
  // if (!CallArgs.getCleanupsToDeactivate().empty())
  //   deactivateArgCleanupsBeforeCall(*this, CallArgs);

  // TODO: Update the largest vector width if any arguments have vector types.
  // TODO: Compute the calling convention and attributes.

  // TODO: strictfp
  // TODO: Add call-site nomerge, noinline, always_inline attribute if exists.

  // Apply some call-site-specific attributes.
  // TODO: work this into building the attribute set.

  // Apply always_inline to all calls within flatten functions.
  // FIXME: should this really take priority over __try, below?
  // assert(!CurCodeDecl->hasAttr<FlattenAttr>() &&
  //        !TargetDecl->hasAttr<NoInlineAttr>() && "NYI");

  // Disable inlining inside SEH __try blocks.
  if (isSEHTryScope())
    llvm_unreachable("NYI");

  // Decide whether to use a call or an invoke.
  bool CannotThrow;
  if (currentFunctionUsesSEHTry()) {
    // SEH cares about asynchronous exceptions, so everything can "throw."
    CannotThrow = false;
  } else if (isCleanupPadScope() &&
             EHPersonality::get(*this).isMSVCXXPersonality()) {
    // The MSVC++ personality will implicitly terminate the program if an
    // exception is thrown during a cleanup outside of a try/catch.
    // We don't need to model anything in IR to get this behavior.
    CannotThrow = true;
  } else {
    // FIXME(cir): pass down nounwind attribute
    CannotThrow = true;
  }
  (void)CannotThrow;

  // TODO: UnusedReturnSizePtr
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(CurFuncDecl))
    assert(!FD->hasAttr<StrictFPAttr>() && "NYI");

  // TODO: alignment attributes

  // Emit the actual call op.
  auto callLoc = loc;
  assert(builder.getInsertionBlock() && "expected valid basic block");

  mlir::cir::CallOp theCall;
  if (auto fnOp = dyn_cast<mlir::cir::FuncOp>(CalleePtr)) {
    assert(fnOp && "only direct call supported");
    theCall = builder.create<mlir::cir::CallOp>(callLoc, fnOp, CIRCallArgs);
  } else if (auto loadOp = dyn_cast<mlir::cir::LoadOp>(CalleePtr)) {
    theCall = builder.create<mlir::cir::CallOp>(callLoc, loadOp->getResult(0),
                                                CIRFuncTy, CIRCallArgs);
  } else if (auto getGlobalOp = dyn_cast<mlir::cir::GetGlobalOp>(CalleePtr)) {
    // FIXME(cir): This peephole optimization to avoids indirect calls for
    // builtins. This should be fixed in the builting declaration instead by not
    // emitting an unecessary get_global in the first place.
    auto *globalOp = mlir::SymbolTable::lookupSymbolIn(CGM.getModule(),
                                                       getGlobalOp.getName());
    assert(getGlobalOp && "undefined global function");
    auto callee = llvm::dyn_cast<mlir::cir::FuncOp>(globalOp);
    assert(callee && "operation is not a function");
    theCall = builder.create<mlir::cir::CallOp>(callLoc, callee, CIRCallArgs);
  } else {
    llvm_unreachable("expected call variant to be handled");
  }

  if (callOrInvoke)
    callOrInvoke = &theCall;

  if (const auto *FD = dyn_cast_or_null<FunctionDecl>(CurFuncDecl))
    assert(!FD->getAttr<CFGuardAttr>() && "NYI");

  // TODO: set attributes on callop
  // assert(!theCall.getResults().getType().front().isSignlessInteger() &&
  //        "Vector NYI");
  // TODO: LLVM models indirect calls via a null callee, how should we do this?
  assert(!CGM.getLangOpts().ObjCAutoRefCount && "Not supported");
  assert((!TargetDecl || !TargetDecl->hasAttr<NotTailCalledAttr>()) && "NYI");
  assert(!getDebugInfo() && "No debug info yet");
  assert((!TargetDecl || !TargetDecl->hasAttr<ErrorAttr>()) && "NYI");

  // 4. Finish the call.

  // If the call doesn't return, finish the basic block and clear the insertion
  // point; this allows the rest of CIRGen to discard unreachable code.
  // TODO: figure out how to support doesNotReturn

  assert(!IsMustTail && "NYI");

  // TODO: figure out writebacks? seems like ObjC only __autorelease

  // TODO: cleanup argument memory at the end

  // Extract the return value.
  RValue ret = [&] {
    switch (RetAI.getKind()) {
    case ABIArgInfo::Direct: {
      mlir::Type RetCIRTy = convertType(RetTy);
      if (RetAI.getCoerceToType() == RetCIRTy && RetAI.getDirectOffset() == 0) {
        switch (getEvaluationKind(RetTy)) {
        case TEK_Aggregate: {
          Address DestPtr = ReturnValue.getValue();
          bool DestIsVolatile = ReturnValue.isVolatile();

          if (!DestPtr.isValid()) {
            DestPtr = CreateMemTemp(RetTy, callLoc, getCounterAggTmpAsString());
            DestIsVolatile = false;
          }

          auto Results = theCall.getResults();
          assert(Results.size() <= 1 && "multiple returns NYI");

          SourceLocRAIIObject Loc{*this, callLoc};
          buildAggregateStore(Results[0], DestPtr, DestIsVolatile);
          return RValue::getAggregate(DestPtr);
        }
        case TEK_Scalar: {
          // If the argument doesn't match, perform a bitcast to coerce it. This
          // can happen due to trivial type mismatches.
          auto Results = theCall.getResults();
          assert(Results.size() <= 1 && "multiple returns NYI");
          assert(Results[0].getType() == RetCIRTy && "Bitcast support NYI");
          return RValue::get(Results[0]);
        }
        default:
          llvm_unreachable("NYI");
        }
      } else {
        llvm_unreachable("No other forms implemented yet.");
      }
    }

    case ABIArgInfo::Ignore:
      // If we are ignoring an argument that had a result, make sure to
      // construct the appropriate return value for our caller.
      return GetUndefRValue(RetTy);

    default:
      llvm_unreachable("NYI");
    }

    llvm_unreachable("NYI");
    return RValue{};
  }();

  // TODO: implement assumed_aligned

  // TODO: implement lifetime extensions

  assert(RetTy.isDestructedType() != QualType::DK_nontrivial_c_struct && "NYI");

  return ret;
}

RValue CIRGenFunction::GetUndefRValue(QualType Ty) {
  assert(Ty->isVoidType() && "Only VoidType supported so far.");
  return RValue::get(nullptr);
}

void CIRGenFunction::buildCallArg(CallArgList &args, const Expr *E,
                                  QualType type) {
  // TODO: Add the DisableDebugLocationUpdates helper
  assert(!dyn_cast<ObjCIndirectCopyRestoreExpr>(E) && "NYI");

  assert(type->isReferenceType() == E->isGLValue() &&
         "reference binding to unmaterialized r-value!");

  if (E->isGLValue()) {
    assert(E->getObjectKind() == OK_Ordinary);
    return args.add(buildReferenceBindingToExpr(E), type);
  }

  bool HasAggregateEvalKind = hasAggregateEvaluationKind(type);

  // In the Microsoft C++ ABI, aggregate arguments are destructed by the callee.
  // However, we still have to push an EH-only cleanup in case we unwind before
  // we make it to the call.
  if (type->isRecordType() &&
      type->castAs<RecordType>()->getDecl()->isParamDestroyedInCallee()) {
    llvm_unreachable("Microsoft C++ ABI is NYI");
  }

  if (HasAggregateEvalKind && isa<ImplicitCastExpr>(E) &&
      cast<CastExpr>(E)->getCastKind() == CK_LValueToRValue) {
    LValue L = buildLValue(cast<CastExpr>(E)->getSubExpr());
    assert(L.isSimple());
    args.addUncopiedAggregate(L, type);
    return;
  }

  args.add(buildAnyExprToTemp(E), type);
}

QualType CIRGenFunction::getVarArgType(const Expr *Arg) {
  // System headers on Windows define NULL to 0 instead of 0LL on Win64. MSVC
  // implicitly widens null pointer constants that are arguments to varargs
  // functions to pointer-sized ints.
  if (!getTarget().getTriple().isOSWindows())
    return Arg->getType();

  if (Arg->getType()->isIntegerType() &&
      getContext().getTypeSize(Arg->getType()) <
          getContext().getTargetInfo().getPointerWidth(LangAS::Default) &&
      Arg->isNullPointerConstant(getContext(),
                                 Expr::NPC_ValueDependentIsNotNull)) {
    return getContext().getIntPtrType();
  }

  return Arg->getType();
}

/// Similar to buildAnyExpr(), however, the result will always be accessible
/// even if no aggregate location is provided.
RValue CIRGenFunction::buildAnyExprToTemp(const Expr *E) {
  AggValueSlot AggSlot = AggValueSlot::ignored();

  if (hasAggregateEvaluationKind(E->getType()))
    AggSlot = CreateAggTemp(E->getType(), getLoc(E->getSourceRange()),
                            getCounterAggTmpAsString());

  return buildAnyExpr(E, AggSlot);
}

void CIRGenFunction::buildCallArgs(
    CallArgList &Args, PrototypeWrapper Prototype,
    llvm::iterator_range<CallExpr::const_arg_iterator> ArgRange,
    AbstractCallee AC, unsigned ParamsToSkip, EvaluationOrder Order) {

  llvm::SmallVector<QualType, 16> ArgTypes;

  assert((ParamsToSkip == 0 || Prototype.P) &&
         "Can't skip parameters if type info is not provided");

  // This variable only captures *explicitly* written conventions, not those
  // applied by default via command line flags or target defaults, such as
  // thiscall, appcs, stdcall via -mrtd, etc. Computing that correctly would
  // require knowing if this is a C++ instance method or being able to see
  // unprotyped FunctionTypes.
  CallingConv ExplicitCC = CC_C;

  // First, if a prototype was provided, use those argument types.
  bool IsVariadic = false;
  if (Prototype.P) {
    const auto *MD = Prototype.P.dyn_cast<const ObjCMethodDecl *>();
    assert(!MD && "ObjCMethodDecl NYI");

    const auto *FPT = Prototype.P.get<const FunctionProtoType *>();
    IsVariadic = FPT->isVariadic();
    ExplicitCC = FPT->getExtInfo().getCC();
    ArgTypes.assign(FPT->param_type_begin() + ParamsToSkip,
                    FPT->param_type_end());
  }

  // If we still have any arguments, emit them using the type of the argument.
  for (auto *A : llvm::drop_begin(ArgRange, ArgTypes.size()))
    ArgTypes.push_back(IsVariadic ? getVarArgType(A) : A->getType());
  assert((int)ArgTypes.size() == (ArgRange.end() - ArgRange.begin()));

  // We must evaluate arguments from right to left in the MS C++ ABI, because
  // arguments are destroyed left to right in the callee. As a special case,
  // there are certain language constructs taht require left-to-right
  // evaluation, and in those cases we consider the evaluation order requirement
  // to trump the "destruction order is reverse construction order" guarantee.
  bool LeftToRight = true;
  assert(!CGM.getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee() &&
         "MSABI NYI");
  assert(!hasInAllocaArgs(CGM, ExplicitCC, ArgTypes) && "NYI");

  auto MaybeEmitImplicitObjectSize = [&](unsigned I, const Expr *Arg,
                                         RValue EmittedArg) {
    if (!AC.hasFunctionDecl() || I >= AC.getNumParams())
      return;
    auto *PS = AC.getParamDecl(I)->getAttr<PassObjectSizeAttr>();
    if (PS == nullptr)
      return;

    const auto &Context = getContext();
    auto SizeTy = Context.getSizeType();
    auto T = builder.getUIntNTy(Context.getTypeSize(SizeTy));
    assert(EmittedArg.getScalarVal() && "We emitted nothing for the arg?");
    auto V = evaluateOrEmitBuiltinObjectSize(
        Arg, PS->getType(), T, EmittedArg.getScalarVal(), PS->isDynamic());
    Args.add(RValue::get(V), SizeTy);
    // If we're emitting args in reverse, be sure to do so with
    // pass_object_size, as well.
    if (!LeftToRight)
      std::swap(Args.back(), *(&Args.back() - 1));
  };

  // Evaluate each argument in the appropriate order.
  size_t CallArgsStart = Args.size();
  for (unsigned I = 0, E = ArgTypes.size(); I != E; ++I) {
    unsigned Idx = LeftToRight ? I : E - I - 1;
    CallExpr::const_arg_iterator Arg = ArgRange.begin() + Idx;
    unsigned InitialArgSize = Args.size();
    assert(!isa<ObjCIndirectCopyRestoreExpr>(*Arg) && "NYI");
    assert(!isa_and_nonnull<ObjCMethodDecl>(AC.getDecl()) && "NYI");

    buildCallArg(Args, *Arg, ArgTypes[Idx]);
    // In particular, we depend on it being the last arg in Args, and the
    // objectsize bits depend on there only being one arg if !LeftToRight.
    assert(InitialArgSize + 1 == Args.size() &&
           "The code below depends on only adding one arg per buildCallArg");
    (void)InitialArgSize;
    // Since pointer argument are never emitted as LValue, it is safe to emit
    // non-null argument check for r-value only.
    if (!Args.back().hasLValue()) {
      RValue RVArg = Args.back().getKnownRValue();
      assert(!SanOpts.has(SanitizerKind::NonnullAttribute) && "Sanitizers NYI");
      assert(!SanOpts.has(SanitizerKind::NullabilityArg) && "Sanitizers NYI");
      // @llvm.objectsize should never have side-effects and shouldn't need
      // destruction/cleanups, so we can safely "emit" it after its arg,
      // regardless of right-to-leftness
      MaybeEmitImplicitObjectSize(Idx, *Arg, RVArg);
    }
  }

  if (!LeftToRight) {
    // Un-reverse the arguments we just evaluated so they match up with the CIR
    // function.
    std::reverse(Args.begin() + CallArgsStart, Args.end());
  }
}

/// Returns the canonical formal type of the given C++ method.
static CanQual<FunctionProtoType> GetFormalType(const CXXMethodDecl *MD) {
  return MD->getType()
      ->getCanonicalTypeUnqualified()
      .getAs<FunctionProtoType>();
}

/// TODO(cir): this should be shared with LLVM codegen
static void addExtParameterInfosForCall(
    llvm::SmallVectorImpl<FunctionProtoType::ExtParameterInfo> &paramInfos,
    const FunctionProtoType *proto, unsigned prefixArgs, unsigned totalArgs) {
  assert(proto->hasExtParameterInfos());
  assert(paramInfos.size() <= prefixArgs);
  assert(proto->getNumParams() + prefixArgs <= totalArgs);

  paramInfos.reserve(totalArgs);

  // Add default infos for any prefix args that don't already have infos.
  paramInfos.resize(prefixArgs);

  // Add infos for the prototype.
  for (const auto &ParamInfo : proto->getExtParameterInfos()) {
    paramInfos.push_back(ParamInfo);
    // pass_object_size params have no parameter info.
    if (ParamInfo.hasPassObjectSize())
      paramInfos.emplace_back();
  }

  assert(paramInfos.size() <= totalArgs &&
         "Did we forget to insert pass_object_size args?");
  // Add default infos for the variadic and/or suffix arguments.
  paramInfos.resize(totalArgs);
}

/// Adds the formal parameters in FPT to the given prefix. If any parameter in
/// FPT has pass_object_size_attrs, then we'll add parameters for those, too.
/// TODO(cir): this should be shared with LLVM codegen
static void appendParameterTypes(
    const CIRGenTypes &CGT, SmallVectorImpl<CanQualType> &prefix,
    SmallVectorImpl<FunctionProtoType::ExtParameterInfo> &paramInfos,
    CanQual<FunctionProtoType> FPT) {
  // Fast path: don't touch param info if we don't need to.
  if (!FPT->hasExtParameterInfos()) {
    assert(paramInfos.empty() &&
           "We have paramInfos, but the prototype doesn't?");
    prefix.append(FPT->param_type_begin(), FPT->param_type_end());
    return;
  }

  unsigned PrefixSize = prefix.size();
  // In the vast majority of cases, we'll have precisely FPT->getNumParams()
  // parameters; the only thing that can change this is the presence of
  // pass_object_size. So, we preallocate for the common case.
  prefix.reserve(prefix.size() + FPT->getNumParams());

  auto ExtInfos = FPT->getExtParameterInfos();
  assert(ExtInfos.size() == FPT->getNumParams());
  for (unsigned I = 0, E = FPT->getNumParams(); I != E; ++I) {
    prefix.push_back(FPT->getParamType(I));
    if (ExtInfos[I].hasPassObjectSize())
      prefix.push_back(CGT.getContext().getSizeType());
  }

  addExtParameterInfosForCall(paramInfos, FPT.getTypePtr(), PrefixSize,
                              prefix.size());
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXStructorDeclaration(GlobalDecl GD) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());

  llvm::SmallVector<CanQualType, 16> argTypes;
  SmallVector<FunctionProtoType::ExtParameterInfo, 16> paramInfos;
  argTypes.push_back(DeriveThisType(MD->getParent(), MD));

  bool PassParams = true;

  if (auto *CD = dyn_cast<CXXConstructorDecl>(MD)) {
    // A base class inheriting constructor doesn't get forwarded arguments
    // needed to construct a virtual base (or base class thereof)
    assert(!CD->getInheritedConstructor() && "Inheritance NYI");
  }

  CanQual<FunctionProtoType> FTP = GetFormalType(MD);

  if (PassParams)
    appendParameterTypes(*this, argTypes, paramInfos, FTP);

  assert(paramInfos.empty() && "NYI");

  assert(!MD->isVariadic() && "Variadic fns NYI");
  RequiredArgs required = RequiredArgs::All;
  (void)required;

  FunctionType::ExtInfo extInfo = FTP->getExtInfo();

  assert(!TheCXXABI.HasThisReturn(GD) && "NYI");

  CanQualType resultType = Context.VoidTy;
  (void)resultType;

  return arrangeCIRFunctionInfo(resultType, /*instanceMethod=*/true,
                                /*chainCall=*/false, argTypes, extInfo,
                                paramInfos, required);
}

/// Derives the 'this' type for CIRGen purposes, i.e. ignoring method CVR
/// qualification. Either or both of RD and MD may be null. A null RD indicates
/// that there is no meaningful 'this' type, and a null MD can occur when
/// calling a method pointer.
CanQualType CIRGenTypes::DeriveThisType(const CXXRecordDecl *RD,
                                        const CXXMethodDecl *MD) {
  QualType RecTy;
  if (RD)
    RecTy = getContext().getTagDeclType(RD)->getCanonicalTypeInternal();
  else
    assert(false && "CXXMethodDecl NYI");

  if (MD)
    RecTy = getContext().getAddrSpaceQualType(
        RecTy, MD->getMethodQualifiers().getAddressSpace());
  return getContext().getPointerType(CanQualType::CreateUnsafe(RecTy));
}

/// Arrange the CIR function layout for a value of the given function type, on
/// top of any implicit parameters already stored.
static const CIRGenFunctionInfo &
arrangeCIRFunctionInfo(CIRGenTypes &CGT, bool instanceMethod,
                       SmallVectorImpl<CanQualType> &prefix,
                       CanQual<FunctionProtoType> FTP) {
  SmallVector<FunctionProtoType::ExtParameterInfo, 16> paramInfos;
  RequiredArgs Required = RequiredArgs::forPrototypePlus(FTP, prefix.size());
  // FIXME: Kill copy. -- from codegen
  appendParameterTypes(CGT, prefix, paramInfos, FTP);
  CanQualType resultType = FTP->getReturnType().getUnqualifiedType();

  return CGT.arrangeCIRFunctionInfo(resultType, instanceMethod,
                                    /*chainCall=*/false, prefix,
                                    FTP->getExtInfo(), paramInfos, Required);
}

/// Arrange the argument and result information for a value of the given
/// freestanding function type.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> FTP) {
  SmallVector<CanQualType, 16> argTypes;
  return ::arrangeCIRFunctionInfo(*this, /*instanceMethod=*/false, argTypes,
                                  FTP);
}

/// Arrange the argument and result information for a value of the given
/// unprototyped freestanding function type.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionNoProtoType> FTNP) {
  // When translating an unprototyped function type, always use a
  // variadic type.
  return arrangeCIRFunctionInfo(FTNP->getReturnType().getUnqualifiedType(),
                                /*instanceMethod=*/false,
                                /*chainCall=*/false, std::nullopt,
                                FTNP->getExtInfo(), {}, RequiredArgs(0));
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// ExtraPrefixArgs is the number of ABI-specific args passed after the `this`
/// parameter.
/// ExtraSuffixArgs is the number of ABI-specific args passed at the end of
/// args.
/// PassProtoArgs indicates whether `args` has args for the parameters in the
/// given CXXConstructorDecl.
const CIRGenFunctionInfo &CIRGenTypes::arrangeCXXConstructorCall(
    const CallArgList &Args, const CXXConstructorDecl *D, CXXCtorType CtorKind,
    unsigned ExtraPrefixArgs, unsigned ExtraSuffixArgs, bool PassProtoArgs) {

  // FIXME: Kill copy.
  llvm::SmallVector<CanQualType, 16> ArgTypes;
  for (const auto &Arg : Args)
    ArgTypes.push_back(Context.getCanonicalParamType(Arg.Ty));

  // +1 for implicit this, which should always be args[0]
  unsigned TotalPrefixArgs = 1 + ExtraPrefixArgs;

  CanQual<FunctionProtoType> FPT = GetFormalType(D);
  RequiredArgs Required = PassProtoArgs
                              ? RequiredArgs::forPrototypePlus(
                                    FPT, TotalPrefixArgs + ExtraSuffixArgs)
                              : RequiredArgs::All;

  GlobalDecl GD(D, CtorKind);
  assert(!TheCXXABI.HasThisReturn(GD) && "ThisReturn NYI");
  assert(!TheCXXABI.hasMostDerivedReturn(GD) && "Most derived return NYI");
  CanQualType ResultType = Context.VoidTy;

  FunctionType::ExtInfo Info = FPT->getExtInfo();
  llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16> ParamInfos;
  // If the prototype args are elided, we should onlyy have ABI-specific args,
  // which never have param info.
  assert(!FPT->hasExtParameterInfos() && "NYI");

  return arrangeCIRFunctionInfo(ResultType, /*instanceMethod=*/true,
                                /*chainCall=*/false, ArgTypes, Info, ParamInfos,
                                Required);
}

bool CIRGenTypes::inheritingCtorHasParams(const InheritedConstructor &Inherited,
                                          CXXCtorType Type) {

  // Parameters are unnecessary if we're constructing a base class subobject and
  // the inherited constructor lives in a virtual base.
  return Type == Ctor_Complete ||
         !Inherited.getShadowDecl()->constructsVirtualBase() ||
         !Target.getCXXABI().hasConstructorVariants();
}

bool CIRGenModule::MayDropFunctionReturn(const ASTContext &Context,
                                         QualType ReturnType) {
  // We can't just disard the return value for a record type with a complex
  // destructor or a non-trivially copyable type.
  if (const RecordType *RT =
          ReturnType.getCanonicalType()->getAs<RecordType>()) {
    llvm_unreachable("NYI");
  }

  return ReturnType.isTriviallyCopyableType(Context);
}

static bool isInAllocaArgument(CIRGenCXXABI &ABI, QualType type) {
  const auto *RD = type->getAsCXXRecordDecl();
  return RD &&
         ABI.getRecordArgABI(RD) == CIRGenCXXABI::RecordArgABI::DirectInMemory;
}

void CIRGenFunction::buildDelegateCallArg(CallArgList &args,
                                          const VarDecl *param,
                                          SourceLocation loc) {
  // StartFunction converted the ABI-lowered parameter(s) into a local alloca.
  // We need to turn that into an r-value suitable for buildCall
  Address local = GetAddrOfLocalVar(param);

  QualType type = param->getType();

  if (isInAllocaArgument(CGM.getCXXABI(), type)) {
    llvm_unreachable("NYI");
  }

  // GetAddrOfLocalVar returns a pointer-to-pointer for references, but the
  // argument needs to be the original pointer.
  if (type->isReferenceType()) {
    args.add(
        RValue::get(builder.createLoad(getLoc(param->getSourceRange()), local)),
        type);
  } else if (getLangOpts().ObjCAutoRefCount) {
    llvm_unreachable("NYI");
    // For the most part, we just need to load the alloca, except that aggregate
    // r-values are actually pointers to temporaries.
  } else {
    args.add(convertTempToRValue(local, type, loc), type);
  }

  // Deactivate the cleanup for the callee-destructed param that was pushed.
  if (type->isRecordType() && !CurFuncIsThunk &&
      type->castAs<RecordType>()->getDecl()->isParamDestroyedInCallee() &&
      param->needsDestruction(getContext())) {
    llvm_unreachable("NYI");
  }
}

/// Returns the "extra-canonicalized" return type, which discards qualifiers on
/// the return type. Codegen doesn't care about them, and it makes ABI code a
/// little easier to be able to assume that all parameter and return types are
/// top-level unqualified.
/// FIXME(CIR): This should be a common helper extracted from CodeGen
static CanQualType GetReturnType(QualType RetTy) {
  return RetTy->getCanonicalTypeUnqualified().getUnqualifiedType();
}

/// Arrange a call as unto a free function, except possibly with an additional
/// number of formal parameters considered required.
static const CIRGenFunctionInfo &
arrangeFreeFunctionLikeCall(CIRGenTypes &CGT, CIRGenModule &CGM,
                            const CallArgList &args, const FunctionType *fnType,
                            unsigned numExtraRequiredArgs, bool chainCall) {
  assert(args.size() >= numExtraRequiredArgs);
  assert(!chainCall && "Chain call NYI");

  llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16> paramInfos;

  // In most cases, there are no optional arguments.
  RequiredArgs required = RequiredArgs::All;

  // If we have a variadic prototype, the required arguments are the
  // extra prefix plus the arguments in the prototype.
  if (const FunctionProtoType *proto = dyn_cast<FunctionProtoType>(fnType)) {
    if (proto->isVariadic())
      required = RequiredArgs::forPrototypePlus(proto, numExtraRequiredArgs);

    if (proto->hasExtParameterInfos())
      addExtParameterInfosForCall(paramInfos, proto, numExtraRequiredArgs,
                                  args.size());
  } else if (llvm::isa<FunctionNoProtoType>(fnType)) {
    assert(!UnimplementedFeature::targetCodeGenInfoIsProtoCallVariadic());
    required = RequiredArgs(args.size());
  }

  // FIXME: Kill copy.
  SmallVector<CanQualType, 16> argTypes;
  for (const auto &arg : args)
    argTypes.push_back(CGT.getContext().getCanonicalParamType(arg.Ty));
  return CGT.arrangeCIRFunctionInfo(
      GetReturnType(fnType->getReturnType()), /*instanceMethod=*/false,
      chainCall, argTypes, fnType->getExtInfo(), paramInfos, required);
}

static llvm::SmallVector<CanQualType, 16>
getArgTypesForCall(ASTContext &ctx, const CallArgList &args) {
  llvm::SmallVector<CanQualType, 16> argTypes;
  for (auto &arg : args)
    argTypes.push_back(ctx.getCanonicalParamType(arg.Ty));
  return argTypes;
}

static llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16>
getExtParameterInfosForCall(const FunctionProtoType *proto, unsigned prefixArgs,
                            unsigned totalArgs) {
  llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16> result;
  if (proto->hasExtParameterInfos()) {
    llvm_unreachable("NYI");
  }
  return result;
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// numPrefixArgs is the number of the ABI-specific prefix arguments we have. It
/// does not count `this`.
const CIRGenFunctionInfo &CIRGenTypes::arrangeCXXMethodCall(
    const CallArgList &args, const FunctionProtoType *proto,
    RequiredArgs required, unsigned numPrefixArgs) {
  assert(numPrefixArgs + 1 <= args.size() &&
         "Emitting a call with less args than the required prefix?");
  // Add one to account for `this`. It is a bit awkard here, but we don't count
  // `this` in similar places elsewhere.
  auto paramInfos =
      getExtParameterInfosForCall(proto, numPrefixArgs + 1, args.size());

  // FIXME: Kill copy.
  auto argTypes = getArgTypesForCall(Context, args);

  auto info = proto->getExtInfo();
  return arrangeCIRFunctionInfo(
      GetReturnType(proto->getReturnType()), /*instanceMethod=*/true,
      /*chainCall=*/false, argTypes, info, paramInfos, required);
}

/// Figure out the rules for calling a function with the given formal type using
/// the given arguments. The arguments are necessary because the function might
/// be unprototyped, in which case it's target-dependent in crazy ways.
const CIRGenFunctionInfo &CIRGenTypes::arrangeFreeFunctionCall(
    const CallArgList &args, const FunctionType *fnType, bool ChainCall) {
  assert(!ChainCall && "ChainCall NYI");
  return arrangeFreeFunctionLikeCall(*this, CGM, args, fnType,
                                     ChainCall ? 1 : 0, ChainCall);
}

/// Set calling convention for CUDA/HIP kernel.
static void setCUDAKernelCallingConvention(CanQualType &FTy, CIRGenModule &CGM,
                                           const FunctionDecl *FD) {
  if (FD->hasAttr<CUDAGlobalAttr>()) {
    llvm_unreachable("NYI");
  }
}

/// Arrange the argument and result information for a declaration or definition
/// of the given C++ non-static member function. The member function must be an
/// ordinary function, i.e. not a constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodDeclaration(const CXXMethodDecl *MD) {
  assert(!isa<CXXConstructorDecl>(MD) && "wrong method for constructors!");
  assert(!isa<CXXDestructorDecl>(MD) && "wrong method for destructors!");

  CanQualType FT = GetFormalType(MD).getAs<Type>();
  setCUDAKernelCallingConvention(FT, CGM, MD);
  auto prototype = FT.getAs<FunctionProtoType>();

  if (MD->isInstance()) {
    // The abstarct case is perfectly fine.
    auto *ThisType = TheCXXABI.getThisArgumentTypeForMethod(MD);
    return arrangeCXXMethodType(ThisType, prototype.getTypePtr(), MD);
  }

  return arrangeFreeFunctionType(prototype);
}

/// Arrange the argument and result information for a call to an unknown C++
/// non-static member function of the given abstract type. (A null RD means we
/// don't have any meaningful "this" argument type, so fall back to a generic
/// pointer type). The member fucntion must be an ordinary function, i.e. not a
/// constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodType(const CXXRecordDecl *RD,
                                  const FunctionProtoType *FTP,
                                  const CXXMethodDecl *MD) {
  llvm::SmallVector<CanQualType, 16> argTypes;

  // Add the 'this' pointer.
  argTypes.push_back(DeriveThisType(RD, MD));

  return ::arrangeCIRFunctionInfo(
      *this, true, argTypes,
      FTP->getCanonicalTypeUnqualified().getAs<FunctionProtoType>());
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFunctionDeclaration(const FunctionDecl *FD) {
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
    if (MD->isInstance())
      return arrangeCXXMethodDeclaration(MD);

  auto FTy = FD->getType()->getCanonicalTypeUnqualified();

  assert(isa<FunctionType>(FTy));
  // TODO: setCUDAKernelCallingConvention

  // When declaring a function without a prototype, always use a non-variadic
  // type.
  if (CanQual<FunctionNoProtoType> noProto = FTy.getAs<FunctionNoProtoType>()) {
    return arrangeCIRFunctionInfo(noProto->getReturnType(),
                                  /*instanceMethod=*/false,
                                  /*chainCall=*/false, std::nullopt,
                                  noProto->getExtInfo(), {}, RequiredArgs::All);
  }

  return arrangeFreeFunctionType(FTy.castAs<FunctionProtoType>());
}

RValue CallArg::getRValue(CIRGenFunction &CGF, mlir::Location loc) const {
  if (!HasLV)
    return RV;
  LValue Copy = CGF.makeAddrLValue(CGF.CreateMemTemp(Ty, loc), Ty);
  CGF.buildAggregateCopy(Copy, LV, Ty, AggValueSlot::DoesNotOverlap,
                         LV.isVolatile());
  IsUsed = true;
  return RValue::getAggregate(Copy.getAddress());
}

void CIRGenFunction::buildNonNullArgCheck(RValue RV, QualType ArgType,
                                          SourceLocation ArgLoc,
                                          AbstractCallee AC, unsigned ParmNum) {
  if (!AC.getDecl() || !(SanOpts.has(SanitizerKind::NonnullAttribute) ||
                         SanOpts.has(SanitizerKind::NullabilityArg)))
    return;
  llvm_unreachable("non-null arg check is NYI");
}

/* VarArg handling */

// FIXME(cir): This completely abstracts away the ABI with a generic CIR Op. We
// need to decide how to handle va_arg target-specific codegen.
mlir::Value CIRGenFunction::buildVAArg(VAArgExpr *VE, Address &VAListAddr) {
  assert(!VE->isMicrosoftABI() && "NYI");
  auto loc = CGM.getLoc(VE->getExprLoc());
  auto type = ConvertType(VE->getType());
  auto vaList = buildVAListRef(VE->getSubExpr()).getPointer();
  return builder.create<mlir::cir::VAArgOp>(loc, type, vaList);
}
