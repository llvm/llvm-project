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

CIRGenFunctionInfo *CIRGenFunctionInfo::create(CanQualType resultType) {
  void *buffer = operator new(totalSizeToAlloc<ArgInfo>(1));

  CIRGenFunctionInfo *fi = new (buffer) CIRGenFunctionInfo();
  fi->getArgsBuffer()[0].type = resultType;

  return fi;
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &cgf) const {
  assert(!cir::MissingFeatures::opCallVirtual());
  return *this;
}

static const CIRGenFunctionInfo &
arrangeFreeFunctionLikeCall(CIRGenTypes &cgt, CIRGenModule &cgm,
                            const FunctionType *fnType) {
  if (const auto *proto = dyn_cast<FunctionProtoType>(fnType)) {
    if (proto->isVariadic())
      cgm.errorNYI("call to variadic function");
    if (proto->hasExtParameterInfos())
      cgm.errorNYI("call to functions with extra parameter info");
  } else if (cgm.getTargetCIRGenInfo().isNoProtoCallVariadic(
                 cast<FunctionNoProtoType>(fnType)))
    cgm.errorNYI("call to function without a prototype");

  assert(!cir::MissingFeatures::opCallArgs());

  CanQualType retType = fnType->getReturnType()
                            ->getCanonicalTypeUnqualified()
                            .getUnqualifiedType();
  return cgt.arrangeCIRFunctionInfo(retType);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionCall(const FunctionType *fnType) {
  return arrangeFreeFunctionLikeCall(*this, cgm, fnType);
}

static cir::CIRCallOpInterface emitCallLikeOp(CIRGenFunction &cgf,
                                              mlir::Location callLoc,
                                              cir::FuncOp directFuncOp) {
  CIRGenBuilderTy &builder = cgf.getBuilder();

  assert(!cir::MissingFeatures::opCallSurroundingTry());
  assert(!cir::MissingFeatures::invokeOp());

  assert(builder.getInsertionBlock() && "expected valid basic block");
  assert(!cir::MissingFeatures::opCallIndirect());

  return builder.createCallOp(callLoc, directFuncOp);
}

RValue CIRGenFunction::emitCall(const CIRGenFunctionInfo &funcInfo,
                                const CIRGenCallee &callee,
                                ReturnValueSlot returnValue,
                                cir::CIRCallOpInterface *callOp,
                                mlir::Location loc) {
  QualType retTy = funcInfo.getReturnType();
  const cir::ABIArgInfo &retInfo = funcInfo.getReturnInfo();

  assert(!cir::MissingFeatures::opCallArgs());
  assert(!cir::MissingFeatures::emitLifetimeMarkers());

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

  cir::CIRCallOpInterface theCall = emitCallLikeOp(*this, loc, directFuncOp);

  if (callOp)
    *callOp = theCall;

  assert(!cir::MissingFeatures::opCallMustTail());
  assert(!cir::MissingFeatures::opCallReturn());

  RValue ret;
  switch (retInfo.getKind()) {
  case cir::ABIArgInfo::Direct: {
    mlir::Type retCIRTy = convertType(retTy);
    if (retInfo.getCoerceToType() == retCIRTy &&
        retInfo.getDirectOffset() == 0) {
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
      default:
        cgm.errorNYI(loc,
                     "unsupported evaluation kind of function call result");
      }
    } else
      cgm.errorNYI(loc, "unsupported function call form");

    break;
  }
  case cir::ABIArgInfo::Ignore:
    // If we are ignoring an argument that had a result, make sure to construct
    // the appropriate return value for our caller.
    ret = getUndefRValue(retTy);
    break;
  default:
    cgm.errorNYI(loc, "unsupported return value information");
  }

  return ret;
}
