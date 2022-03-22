#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"

#include "clang/AST/GlobalDecl.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

using namespace cir;
using namespace clang;

CIRGenFunctionInfo *CIRGenFunctionInfo::create(
    unsigned cirCC, bool instanceMethod, bool chainCall,
    const clang::FunctionType::ExtInfo &info,
    llvm::ArrayRef<ExtParameterInfo> paramInfos, clang::CanQualType resultType,
    llvm::ArrayRef<clang::CanQualType> argTypes, RequiredArgs required) {
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

/// Encapsulates information about hte way function arguments from
/// CIRGenFunctionInfo should be passed to actual CIR function.
class ClangToCIRArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned InallocaArgNo;
  unsigned SRetArgNo;
  unsigned TotalCIRArgs;

  /// Arguments of CIR function corresponding to single Clang argument.
  struct CIRArgs {
    unsigned PaddingArgIndex;
    // Argument is expanded to CIR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex;
    unsigned NumberOfArgs;

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
      assert(false && "NYI");
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      assert(!AI.getCoerceToType().dyn_cast<mlir::cir::StructType>() && "NYI");
      // FIXME: handle sseregparm someday...
      // FIXME: handle structs
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

mlir::FunctionType CIRGenTypes::GetFunctionType(clang::GlobalDecl GD) {
  const CIRGenFunctionInfo &FI = arrangeGlobalDeclaration(GD);
  return GetFunctionType(FI);
}

mlir::FunctionType CIRGenTypes::GetFunctionType(const CIRGenFunctionInfo &FI) {
  bool Inserted = FunctionsBeingProcessed.insert(&FI).second;
  (void)Inserted;
  assert(Inserted && "Recursively being processed?");

  mlir::Type resultType = nullptr;
  const ABIArgInfo &retAI = FI.getReturnInfo();
  switch (retAI.getKind()) {
  case ABIArgInfo::Ignore:
    // TODO: where to get VoidTy?
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
      assert(false && "NYI");
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

  return Builder.getFunctionType(ArgTypes,
                                 resultType ? resultType : mlir::TypeRange());
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &CGF) const {
  assert(!isVirtual() && "Virtual NYI");
  return *this;
}

RValue CIRGenFunction::buildCall(const CIRGenFunctionInfo &CallInfo,
                                 const CIRGenCallee &Callee,
                                 ReturnValueSlot ReturnValue,
                                 const CallArgList &CallArgs,
                                 mlir::func::CallOp &callOrInvoke,
                                 bool IsMustTail, clang::SourceLocation Loc) {
  // FIXME: We no longer need the types from CallArgs; lift up and simplify

  assert(Callee.isOrdinary() || Callee.isVirtual());

  // Handle struct-return functions by passing a pointer to the location that we
  // would like to return info.
  QualType RetTy = CallInfo.getReturnType();
  const auto &RetAI = CallInfo.getReturnInfo();

  const Decl *TargetDecl = Callee.getAbstractInfo().getCalleeDecl().getDecl();

  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl);
  assert(FD && "Only functiondecl supported so far");
  // We can only guarantee that a function is called from the correct
  // context/function based on the appropriate target attributes, so only check
  // in hte case where we have both always_inline and target since otherwise we
  // could be making a conditional call after a check for the proper cpu
  // features (and it won't cause code generation issues due to function based
  // code generation).
  assert(!TargetDecl->hasAttr<AlwaysInlineAttr>() && "NYI");
  assert(!TargetDecl->hasAttr<TargetAttr>() && "NYI");

  // Some architectures (such as x86-64) have the ABI changed based on
  // attribute-target/features. Give them a chance to diagnose.
  // TODO: support this eventually, just assume the trivial result for now
  // !CGM.getTargetCIRGenInfo().checkFunctionCallABI(
  //     CGM, Loc, dyn_cast_or_null<FunctionDecl>(CurCodeDecl), FD, CallArgs);

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
        assert(ArgInfo.getCoerceToType() == V.getType() && "widening NYI");

        mlir::FunctionType CIRFuncTy = getTypes().GetFunctionType(CallInfo);

        // If the argument doesn't match, perform a bitcast to coerce it. This
        // can happen due to trivial type mismatches.
        if (FirstCIRArg < CIRFuncTy.getNumInputs() &&
            V.getType() != CIRFuncTy.getInput(FirstCIRArg))
          assert(false && "Shouldn't have to bitcast anything yet");

        CIRCallArgs[FirstCIRArg] = V;
        break;
      }
      assert(false && "this code path shouldn't be hit yet");
    }
    default:
      assert(false && "Only Direct support so far");
    }
  }

  const CIRGenCallee &ConcreteCallee = Callee.prepareConcreteCallee(*this);
  mlir::FuncOp CalleePtr = ConcreteCallee.getFunctionPointer();

  // If we're using inalloca, set up that argument.
  assert(!ArgMemory.isValid() && "inalloca NYI");

  // TODO: simplifyVariadicCallee

  // 3. Perform the actual call.

  // Deactivate any cleanups that we're supposed to do immediately before the
  // call.
  // TODO: do this

  // TODO: Update the largest vector width if any arguments have vector types.
  // TODO: Compute the calling convention and attributes.
  assert(!FD->hasAttr<StrictFPAttr>() && "NYI");

  // TODO: InNoMergeAttributedStmt
  // assert(!CurCodeDecl->hasAttr<FlattenAttr>() &&
  //        !TargetDecl->hasAttr<NoInlineAttr>() && "NYI");

  // TODO: isSEHTryScope

  // TODO: currentFunctionUsesSEHTry
  // TODO: isCleanupPadScope

  // TODO: UnusedReturnSizePtr

  assert(!FD->hasAttr<StrictFPAttr>() && "NYI");

  // TODO: alignment attributes

  // Emit the actual call op.
  auto callLoc = CGM.getLoc(Loc);
  auto theCall = CGM.getBuilder().create<mlir::func::CallOp>(callLoc, CalleePtr,
                                                             CIRCallArgs);

  if (callOrInvoke)
    callOrInvoke = theCall;

  if (const auto *FD = dyn_cast_or_null<FunctionDecl>(CurFuncDecl)) {
    assert(!FD->getAttr<CFGuardAttr>() && "NYI");
  }

  // TODO: set attributes on callop

  // assert(!theCall.getResults().getType().front().isSignlessInteger() &&
  //        "Vector NYI");

  // TODO: LLVM models indirect calls via a null callee, how should we do this?

  assert(!CGM.getLangOpts().ObjCAutoRefCount && "Not supported");

  assert(!TargetDecl->hasAttr<NotTailCalledAttr>() && "NYI");

  assert(!getDebugInfo() && "No debug info yet");

  assert(!TargetDecl->hasAttr<ErrorAttr>() && "NYI");

  // 4. Finish the call.

  // If the call doesn't return, finish the basic block and clear the insertion
  // point; this allows the rest of CIRGen to discard unreachable code.
  // TODO: figure out how to support doesNotReturn

  assert(!IsMustTail && "NYI");

  // TODO: figure out writebacks? seems like ObjC only __autorelease

  // TODO: cleanup argument memory at the end

  // Extract the return value.
  RValue Ret = [&] {
    switch (RetAI.getKind()) {
    case ABIArgInfo::Direct: {
      mlir::Type RetCIRTy = convertType(RetTy);
      if (RetAI.getCoerceToType() == RetCIRTy && RetAI.getDirectOffset() == 0) {
        switch (getEvaluationKind(RetTy)) {
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

  return Ret;
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

  assert(!E->isGLValue() && "NYI");

  bool HasAggregateEvalKind = hasAggregateEvaluationKind(type);

  // In the Microsoft C++ ABI, aggregate arguments are destructed by the callee.
  // However, we still have to push an EH-only cleanup in case we unwind before
  // we make it to the call.
  assert(!type->isRecordType() && "Record type args NYI");

  assert(!HasAggregateEvalKind && "aggregate args NYI");
  assert(!isa<ImplicitCastExpr>(E) && "Casted args NYI");

  args.add(buildAnyExprToTemp(E), type);
}

/// buildAnyExprToTemp - Similar to buildAnyExpr(), however, the result will
/// always be accessible even if no aggregate location is provided.
RValue CIRGenFunction::buildAnyExprToTemp(const Expr *E) {
  AggValueSlot AggSlot = AggValueSlot::ignored();

  assert(!hasAggregateEvaluationKind(E->getType()) && "aggregate args NYI");
  return buildAnyExpr(E, AggSlot);
}
