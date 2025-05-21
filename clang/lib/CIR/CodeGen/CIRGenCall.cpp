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
#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenFunctionInfo *
CIRGenFunctionInfo::create(CanQualType resultType,
                           llvm::ArrayRef<CanQualType> argTypes,
                           RequiredArgs required) {
  // The first slot allocated for arg type slot is for the return value.
  void *buffer = operator new(
      totalSizeToAlloc<CanQualType>(argTypes.size() + 1));

  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoParamInfo());

  CIRGenFunctionInfo *fi = new (buffer) CIRGenFunctionInfo();

  fi->required = required;
  fi->numArgs = argTypes.size();

  fi->getArgTypes()[0] = resultType;
  std::copy(argTypes.begin(), argTypes.end(), fi->argTypesBegin());
  assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());

  return fi;
}

cir::FuncType CIRGenTypes::getFunctionType(const CIRGenFunctionInfo &fi) {
  mlir::Type resultType = convertType(fi.getReturnType());
  SmallVector<mlir::Type, 8> argTypes;
  argTypes.reserve(fi.getNumRequiredArgs());

  for (const CanQualType &argType : fi.requiredArguments())
    argTypes.push_back(convertType(argType));

  return cir::FuncType::get(argTypes,
                            (resultType ? resultType : builder.getVoidTy()),
                            fi.isVariadic());
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &cgf) const {
  assert(!cir::MissingFeatures::opCallVirtual());
  return *this;
}

/// Adds the formal parameters in FPT to the given prefix. If any parameter in
/// FPT has pass_object_size_attrs, then we'll add parameters for those, too.
/// TODO(cir): this should be shared with LLVM codegen
static void appendParameterTypes(const CIRGenTypes &cgt,
                                 SmallVectorImpl<CanQualType> &prefix,
                                 CanQual<FunctionProtoType> fpt) {
  assert(!cir::MissingFeatures::opCallExtParameterInfo());
  // Fast path: don't touch param info if we don't need to.
  if (!fpt->hasExtParameterInfos()) {
    prefix.append(fpt->param_type_begin(), fpt->param_type_end());
    return;
  }

  cgt.getCGModule().errorNYI("appendParameterTypes: hasExtParameterInfos");
}

/// Derives the 'this' type for CIRGen purposes, i.e. ignoring method CVR
/// qualification. Either or both of `rd` and `md` may be null. A null `rd`
/// indicates that there is no meaningful 'this' type, and a null `md` can occur
/// when calling a method pointer.
CanQualType CIRGenTypes::deriveThisType(const CXXRecordDecl *rd,
                                        const CXXMethodDecl *md) {
  QualType recTy;
  if (rd) {
    recTy = getASTContext().getTagDeclType(rd)->getCanonicalTypeInternal();
  } else {
    // This can happen with the MS ABI. It shouldn't need anything more than
    // setting recTy to VoidTy here, but we're flagging it for now because we
    // don't have the full handling implemented.
    cgm.errorNYI("deriveThisType: no record decl");
    recTy = getASTContext().VoidTy;
  }

  if (md)
    recTy = getASTContext().getAddrSpaceQualType(
        recTy, md->getMethodQualifiers().getAddressSpace());
  return getASTContext().getPointerType(CanQualType::CreateUnsafe(recTy));
}

/// Arrange the CIR function layout for a value of the given function type, on
/// top of any implicit parameters already stored.
static const CIRGenFunctionInfo &
arrangeCIRFunctionInfo(CIRGenTypes &cgt, SmallVectorImpl<CanQualType> &prefix,
                       CanQual<FunctionProtoType> ftp) {
  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  RequiredArgs required =
      RequiredArgs::getFromProtoWithExtraSlots(ftp, prefix.size());
  assert(!cir::MissingFeatures::opCallExtParameterInfo());
  appendParameterTypes(cgt, prefix, ftp);
  CanQualType resultType = ftp->getReturnType().getUnqualifiedType();
  return cgt.arrangeCIRFunctionInfo(resultType, prefix, required);
}

static const CIRGenFunctionInfo &
arrangeFreeFunctionLikeCall(CIRGenTypes &cgt, CIRGenModule &cgm,
                            const CallArgList &args,
                            const FunctionType *fnType) {

  RequiredArgs required = RequiredArgs::All;

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

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return cgt.arrangeCIRFunctionInfo(retType, argTypes, required);
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// numPrefixArgs is the number of the ABI-specific prefix arguments we have. It
/// does not count `this`.
const CIRGenFunctionInfo &CIRGenTypes::arrangeCXXMethodCall(
    const CallArgList &args, const FunctionProtoType *proto,
    RequiredArgs required, unsigned numPrefixArgs) {
  assert(!cir::MissingFeatures::opCallExtParameterInfo());
  assert(numPrefixArgs + 1 <= args.size() &&
         "Emitting a call with less args than the required prefix?");

  // FIXME: Kill copy.
  llvm::SmallVector<CanQualType, 16> argTypes;
  for (const CallArg &arg : args)
    argTypes.push_back(astContext.getCanonicalParamType(arg.ty));

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return arrangeCIRFunctionInfo(proto->getReturnType()
                                    ->getCanonicalTypeUnqualified()
                                    .getUnqualifiedType(),
                                argTypes, required);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionCall(const CallArgList &args,
                                     const FunctionType *fnType) {
  return arrangeFreeFunctionLikeCall(*this, cgm, args, fnType);
}

/// Arrange the argument and result information for a declaration or definition
/// of the given C++ non-static member function. The member function must be an
/// ordinary function, i.e. not a constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodDeclaration(const CXXMethodDecl *md) {
  assert(!isa<CXXConstructorDecl>(md) && "wrong method for constructors!");
  assert(!isa<CXXDestructorDecl>(md) && "wrong method for destructors!");

  auto prototype =
      md->getType()->getCanonicalTypeUnqualified().getAs<FunctionProtoType>();
  assert(!cir::MissingFeatures::cudaSupport());

  if (md->isInstance()) {
    // The abstract case is perfectly fine.
    auto *thisType = theCXXABI.getThisArgumentTypeForMethod(md);
    return arrangeCXXMethodType(thisType, prototype.getTypePtr(), md);
  }

  return arrangeFreeFunctionType(prototype);
}

/// Arrange the argument and result information for a call to an unknown C++
/// non-static member function of the given abstract type. (A null RD means we
/// don't have any meaningful "this" argument type, so fall back to a generic
/// pointer type). The member fucntion must be an ordinary function, i.e. not a
/// constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodType(const CXXRecordDecl *rd,
                                  const FunctionProtoType *ftp,
                                  const CXXMethodDecl *md) {
  llvm::SmallVector<CanQualType, 16> argTypes;

  // Add the 'this' pointer.
  argTypes.push_back(deriveThisType(rd, md));

  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return ::arrangeCIRFunctionInfo(
      *this, argTypes,
      ftp->getCanonicalTypeUnqualified().getAs<FunctionProtoType>());
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFunctionDeclaration(const FunctionDecl *fd) {
  if (const auto *md = dyn_cast<CXXMethodDecl>(fd))
    if (md->isInstance())
      return arrangeCXXMethodDeclaration(md);

  CanQualType funcTy = fd->getType()->getCanonicalTypeUnqualified();

  assert(isa<FunctionType>(funcTy));
  // TODO: setCUDAKernelCallingConvention
  assert(!cir::MissingFeatures::cudaSupport());

  // When declaring a function without a prototype, always use a non-variadic
  // type.
  if (CanQual<FunctionNoProtoType> noProto =
          funcTy.getAs<FunctionNoProtoType>()) {
    assert(!cir::MissingFeatures::opCallCIRGenFuncInfoExtParamInfo());
    assert(!cir::MissingFeatures::opCallFnInfoOpts());
    return arrangeCIRFunctionInfo(noProto->getReturnType(), std::nullopt,
                                  RequiredArgs::All);
  }

  return arrangeFreeFunctionType(funcTy.castAs<FunctionProtoType>());
}

static cir::CIRCallOpInterface
emitCallLikeOp(CIRGenFunction &cgf, mlir::Location callLoc,
               cir::FuncType indirectFuncTy, mlir::Value indirectFuncVal,
               cir::FuncOp directFuncOp,
               const SmallVectorImpl<mlir::Value> &cirCallArgs) {
  CIRGenBuilderTy &builder = cgf.getBuilder();

  assert(!cir::MissingFeatures::opCallSurroundingTry());
  assert(!cir::MissingFeatures::invokeOp());

  assert(builder.getInsertionBlock() && "expected valid basic block");

  if (indirectFuncTy) {
    // TODO(cir): Set calling convention for indirect calls.
    assert(!cir::MissingFeatures::opCallCallConv());
    return builder.createIndirectCallOp(callLoc, indirectFuncVal,
                                        indirectFuncTy, cirCallArgs);
  }

  return builder.createCallOp(callLoc, directFuncOp, cirCallArgs);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> fpt) {
  SmallVector<CanQualType, 16> argTypes;
  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return ::arrangeCIRFunctionInfo(*this, argTypes, fpt);
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionNoProtoType> fnpt) {
  CanQualType resultType = fnpt->getReturnType().getUnqualifiedType();
  assert(!cir::MissingFeatures::opCallFnInfoOpts());
  return arrangeCIRFunctionInfo(resultType, {}, RequiredArgs(0));
}

RValue CIRGenFunction::emitCall(const CIRGenFunctionInfo &funcInfo,
                                const CIRGenCallee &callee,
                                ReturnValueSlot returnValue,
                                const CallArgList &args,
                                cir::CIRCallOpInterface *callOp,
                                mlir::Location loc) {
  QualType retTy = funcInfo.getReturnType();
  cir::FuncType cirFuncTy = getTypes().getFunctionType(funcInfo);

  SmallVector<mlir::Value, 16> cirCallArgs(args.size());

  assert(!cir::MissingFeatures::emitLifetimeMarkers());

  // Translate all of the arguments as necessary to match the CIR lowering.
  for (auto [argNo, arg, canQualArgType] :
       llvm::enumerate(args, funcInfo.argTypes())) {

    // Insert a padding argument to ensure proper alignment.
    assert(!cir::MissingFeatures::opCallPaddingArgs());

    mlir::Type argType = convertType(canQualArgType);
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

  cir::FuncType indirectFuncTy;
  mlir::Value indirectFuncVal;
  cir::FuncOp directFuncOp;
  if (auto fnOp = dyn_cast<cir::FuncOp>(calleePtr)) {
    directFuncOp = fnOp;
  } else {
    [[maybe_unused]] mlir::ValueTypeRange<mlir::ResultRange> resultTypes =
        calleePtr->getResultTypes();
    [[maybe_unused]] auto funcPtrTy =
        mlir::dyn_cast<cir::PointerType>(resultTypes.front());
    assert(funcPtrTy && mlir::isa<cir::FuncType>(funcPtrTy.getPointee()) &&
           "expected pointer to function");

    indirectFuncTy = cirFuncTy;
    indirectFuncVal = calleePtr->getResult(0);
  }

  assert(!cir::MissingFeatures::opCallAttrs());

  cir::CIRCallOpInterface theCall = emitCallLikeOp(
      *this, loc, indirectFuncTy, indirectFuncVal, directFuncOp, cirCallArgs);

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
    if (!callee.hasFunctionDecl() || i >= callee.getNumParams())
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
