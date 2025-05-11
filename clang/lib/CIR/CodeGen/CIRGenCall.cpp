//===--- CIRGenCall.cpp - Encapsulate calling convention details ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function definition used
// to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenFunctionInfo *
CIRGenFunctionInfo::create(CanQualType resultType,
                           llvm::ArrayRef<CanQualType> argTypes) {
  // The first slot allocated for ArgInfo is for the return value.
  void *buffer = operator new(totalSizeToAlloc<ArgInfo>(argTypes.size() + 1));

  CIRGenFunctionInfo *fi = new (buffer) CIRGenFunctionInfo();
  fi->numArgs = argTypes.size();

  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoParamInfo());

  ArgInfo *argsBuffer = fi->getArgsBuffer();
  (argsBuffer++)->type = resultType;
  for (CanQualType ty : argTypes)
    (argsBuffer++)->type = ty;

  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());

  return fi;
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &cgf) const {
  assert(!cir::MissingFeatures::opCallVirtual());
  return *this;
}

static const CIRGenFunctionInfo &
arrangeFreeFunctionLikeCall(CIRGenTypes &cgt, CIRGenModule &cgm,
                            const CallArgList &args,
                            const FunctionType *fnType) {
  if (const auto *proto = dyn_cast<FunctionProtoType>(fnType)) {
    if (proto->isVariadic())
      cgm.errorNYI("call to variadic function");
    if (proto->hasExtParameterInfos())
      cgm.errorNYI("call to functions with extra parameter info");
  } else if (cgm.getTargetCIRGenInfo().isNoProtoCallVariadic(
                 cast<FunctionNoProtoType>(fnType)))
    cgm.errorNYI("call to function without a prototype");

  SmallVector<CanQualType, 16> argTypes;
  for (const CallArg &arg : args)
    argTypes.push_back(cgt.getASTContext().getCanonicalParamType(arg.ty));

  CanQualType retType = fnType->getReturnType()
                            ->getCanonicalTypeUnqualified()
                            .getUnqualifiedType();
  return cgt.arrangeCIRFunctionInfo(retType, argTypes);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionCall(const CallArgList &args,
                                     const FunctionType *fnType) {
  return arrangeFreeFunctionLikeCall(*this, cgm, args, fnType);
}

static cir::CIRCallOpInterface
emitCallLikeOp(CIRGenFunction &cgf, mlir::Location callLoc,
               cir::FuncOp directFuncOp,
               const SmallVectorImpl<mlir::Value> &cirCallArgs) {
  CIRGenBuilderTy &builder = cgf.getBuilder();

  assert(!cir::MissingFeatures::opCallSurroundingTry());
  assert(!cir::MissingFeatures::invokeOp());

  assert(builder.getInsertionBlock() && "expected valid basic block");
  assert(!cir::MissingFeatures::opCallIndirect());

  return builder.createCallOp(callLoc, directFuncOp, cirCallArgs);
}

RValue CIRGenFunction::emitCall(const CIRGenFunctionInfo &funcInfo,
                                const CIRGenCallee &callee,
                                ReturnValueSlot returnValue,
                                const CallArgList &args,
                                cir::CIRCallOpInterface *callOp,
                                mlir::Location loc) {
  QualType retTy = funcInfo.getReturnType();

  SmallVector<mlir::Value, 16> cirCallArgs(args.size());

  assert(!cir::MissingFeatures::emitLifetimeMarkers());

  // Translate all of the arguments as necessary to match the CIR lowering.
  for (auto [argNo, arg, argInfo] :
       llvm::enumerate(args, funcInfo.arguments())) {
    // Insert a padding argument to ensure proper alignment.
    assert(!cir::MissingFeatures::opCallPaddingArgs());

    mlir::Type argType = convertType(argInfo.type);
    if (!mlir::isa<cir::RecordType>(argType)) {
      mlir::Value v;
      if (arg.isAggregate())
        cgm.errorNYI(loc, "emitCall: aggregate call argument");
      v = arg.getKnownRValue().getScalarVal();

      // We might have to widen integers, but we should never truncate.
      if (argType != v.getType() && mlir::isa<cir::IntType>(v.getType()))
        cgm.errorNYI(loc, "emitCall: widening integer call argument");

      // If the argument doesn't match, perform a bitcast to coerce it. This
      // can happen due to trivial type mismatches.
      // TODO(cir): When getFunctionType is added, assert that this isn't
      // needed.
      assert(!cir::MissingFeatures::opCallBitcastArg());
      cirCallArgs[argNo] = v;
    } else {
      assert(!cir::MissingFeatures::opCallAggregateArgs());
      cgm.errorNYI("emitCall: aggregate function call argument");
    }
  }

  const CIRGenCallee &concreteCallee = callee.prepareConcreteCallee(*this);
  mlir::Operation *calleePtr = concreteCallee.getFunctionPointer();

  assert(!cir::MissingFeatures::opCallInAlloca());

  mlir::NamedAttrList attrs;
  StringRef funcName;
  if (auto calleeFuncOp = dyn_cast<cir::FuncOp>(calleePtr))
    funcName = calleeFuncOp.getName();

  assert(!cir::MissingFeatures::opCallCallConv());
  assert(!cir::MissingFeatures::opCallSideEffect());
  assert(!cir::MissingFeatures::opCallAttrs());

  assert(!cir::MissingFeatures::invokeOp());

  auto directFuncOp = dyn_cast<cir::FuncOp>(calleePtr);
  assert(!cir::MissingFeatures::opCallIndirect());
  assert(!cir::MissingFeatures::opCallAttrs());

  cir::CIRCallOpInterface theCall =
      emitCallLikeOp(*this, loc, directFuncOp, cirCallArgs);

  if (callOp)
    *callOp = theCall;

  assert(!cir::MissingFeatures::opCallMustTail());
  assert(!cir::MissingFeatures::opCallReturn());

  mlir::Type retCIRTy = convertType(retTy);
  if (isa<cir::VoidType>(retCIRTy))
    return getUndefRValue(retTy);
  switch (getEvaluationKind(retTy)) {
  case cir::TEK_Scalar: {
    mlir::ResultRange results = theCall->getOpResults();
    assert(results.size() == 1 && "unexpected number of returns");

    // If the argument doesn't match, perform a bitcast to coerce it. This
    // can happen due to trivial type mismatches.
    if (results[0].getType() != retCIRTy)
      cgm.errorNYI(loc, "bitcast on function return value");

    mlir::Region *region = builder.getBlock()->getParent();
    if (region != theCall->getParentRegion())
      cgm.errorNYI(loc, "function calls with cleanup");

    return RValue::get(results[0]);
  }
  case cir::TEK_Complex:
  case cir::TEK_Aggregate:
    cgm.errorNYI(loc, "unsupported evaluation kind of function call result");
    return getUndefRValue(retTy);
  }
  llvm_unreachable("Invalid evaluation kind");
}

void CIRGenFunction::emitCallArg(CallArgList &args, const clang::Expr *e,
                                 clang::QualType argType) {
  assert(argType->isReferenceType() == e->isGLValue() &&
         "reference binding to unmaterialized r-value!");

  if (e->isGLValue()) {
    assert(e->getObjectKind() == OK_Ordinary);
    return args.add(emitReferenceBindingToExpr(e), argType);
  }

  bool hasAggregateEvalKind = hasAggregateEvaluationKind(argType);

  if (hasAggregateEvalKind) {
    assert(!cir::MissingFeatures::opCallAggregateArgs());
    cgm.errorNYI(e->getSourceRange(),
                 "emitCallArg: aggregate function call argument");
  }

  args.add(emitAnyExprToTemp(e), argType);
}

/// Similar to emitAnyExpr(), however, the result will always be accessible
/// even if no aggregate location is provided.
RValue CIRGenFunction::emitAnyExprToTemp(const Expr *e) {
  assert(!cir::MissingFeatures::opCallAggregateArgs());

  if (hasAggregateEvaluationKind(e->getType()))
    cgm.errorNYI(e->getSourceRange(), "emit aggregate value to temp");

  return emitAnyExpr(e);
}

void CIRGenFunction::emitCallArgs(
    CallArgList &args, PrototypeWrapper prototype,
    llvm::iterator_range<clang::CallExpr::const_arg_iterator> argRange,
    AbstractCallee callee, unsigned paramsToSkip) {
  llvm::SmallVector<QualType, 16> argTypes;

  assert(!cir::MissingFeatures::opCallCallConv());

  // First, if a prototype was provided, use those argument types.
  assert(!cir::MissingFeatures::opCallVariadic());
  if (prototype.p) {
    assert(!cir::MissingFeatures::opCallObjCMethod());

    const auto *fpt = cast<const FunctionProtoType *>(prototype.p);
    argTypes.assign(fpt->param_type_begin() + paramsToSkip,
                    fpt->param_type_end());
  }

  // If we still have any arguments, emit them using the type of the argument.
  for (const clang::Expr *a : llvm::drop_begin(argRange, argTypes.size()))
    argTypes.push_back(a->getType());
  assert(argTypes.size() == (size_t)(argRange.end() - argRange.begin()));

  // We must evaluate arguments from right to left in the MS C++ ABI, because
  // arguments are destroyed left to right in the callee. As a special case,
  // there are certain language constructs taht require left-to-right
  // evaluation, and in those cases we consider the evaluation order requirement
  // to trump the "destruction order is reverse construction order" guarantee.
  auto leftToRight = true;
  assert(!cir::MissingFeatures::msabi());

  auto maybeEmitImplicitObjectSize = [&](size_t i, const Expr *arg,
                                         RValue emittedArg) {
    if (callee.hasFunctionDecl() || i >= callee.getNumParams())
      return;
    auto *ps = callee.getParamDecl(i)->getAttr<PassObjectSizeAttr>();
    if (!ps)
      return;

    assert(!cir::MissingFeatures::opCallImplicitObjectSizeArgs());
    cgm.errorNYI("emit implicit object size for call arg");
  };

  // Evaluate each argument in the appropriate order.
  size_t callArgsStart = args.size();
  for (size_t i = 0; i != argTypes.size(); ++i) {
    size_t idx = leftToRight ? i : argTypes.size() - i - 1;
    CallExpr::const_arg_iterator currentArg = argRange.begin() + idx;
    size_t initialArgSize = args.size();

    emitCallArg(args, *currentArg, argTypes[idx]);

    // In particular, we depend on it being the last arg in Args, and the
    // objectsize bits depend on there only being one arg if !LeftToRight.
    assert(initialArgSize + 1 == args.size() &&
           "The code below depends on only adding one arg per emitCallArg");
    (void)initialArgSize;

    // Since pointer argument are never emitted as LValue, it is safe to emit
    // non-null argument check for r-value only.
    if (!args.back().hasLValue()) {
      RValue rvArg = args.back().getKnownRValue();
      assert(!cir::MissingFeatures::sanitizers());
      maybeEmitImplicitObjectSize(idx, *currentArg, rvArg);
    }

    if (!leftToRight)
      std::reverse(args.begin() + callArgsStart, args.end());
  }
}
