//===- CIRABIRewriteContext.cpp - CIR ABI rewrite context ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRABIRewriteContext.h"
#include "mlir/IR/Builders.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace cir;
using namespace mlir;
using namespace mlir::abi;

namespace {

bool needsRewrite(const FunctionClassification &fc) {
  if (fc.returnInfo.kind != ArgKind::Direct || fc.returnInfo.coercedType)
    return true;
  for (const ArgClassification &ac : fc.argInfos)
    if (ac.kind != ArgKind::Direct || ac.coercedType)
      return true;
  return false;
}

SmallVector<unsigned> ignoredArgIndices(const FunctionClassification &fc) {
  SmallVector<unsigned> v;
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos))
    if (ac.kind == ArgKind::Ignore)
      v.push_back(idx);
  return v;
}

LogicalResult buildNewArgTypes(ArrayRef<Type> oldArgTypes,
                               const FunctionClassification &fc,
                               SmallVectorImpl<Type> &newArgTypes,
                               function_ref<InFlightDiagnostic()> emitError) {
  newArgTypes.reserve(oldArgTypes.size());
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    Type origTy = oldArgTypes[idx];
    switch (ac.kind) {
    case ArgKind::Direct:
      if (ac.coercedType) {
        emitError() << "Direct with coerced type at arg " << idx
                    << " not yet implemented in CallConvLowering";
        return failure();
      }
      newArgTypes.push_back(origTy);
      break;
    case ArgKind::Ignore:
      break;
    case ArgKind::Expand:
      newArgTypes.push_back(origTy);
      break;
    case ArgKind::Extend:
      emitError() << "Extend at arg " << idx
                  << " not yet implemented in CallConvLowering";
      return failure();
    case ArgKind::Indirect:
      emitError() << "Indirect at arg " << idx
                  << " not yet implemented in CallConvLowering";
      return failure();
    }
  }
  return success();
}

Type computeNewReturnType(Type origRetTy, const ArgClassification &retInfo,
                          MLIRContext *ctx,
                          function_ref<InFlightDiagnostic()> emitError) {
  switch (retInfo.kind) {
  case ArgKind::Direct:
    if (retInfo.coercedType) {
      emitError() << "Direct return with coerced type not yet implemented "
                  << "in CallConvLowering";
      return nullptr;
    }
    return origRetTy;
  case ArgKind::Ignore:
    return cir::VoidType::get(ctx);
  case ArgKind::Expand:
    return origRetTy;
  case ArgKind::Extend:
    emitError() << "Extend return not yet implemented in CallConvLowering";
    return nullptr;
  case ArgKind::Indirect:
    emitError() << "Indirect return (sret) not yet implemented in "
                << "CallConvLowering";
    return nullptr;
  }
  llvm_unreachable("all ArgKind cases handled");
}

} // namespace

LogicalResult CIRABIRewriteContext::rewriteFunctionDefinition(
    FunctionOpInterface funcOp, const FunctionClassification &fc,
    OpBuilder &rewriter) {
  if (!needsRewrite(fc))
    return success();

  ArrayRef<Type> oldArgTypes = funcOp.getArgumentTypes();
  ArrayRef<Type> oldResultTypes = funcOp.getResultTypes();
  MLIRContext *ctx = funcOp->getContext();

  SmallVector<Type> newArgTypes;
  if (failed(buildNewArgTypes(oldArgTypes, fc, newArgTypes,
                              [&]() { return funcOp.emitOpError(); })))
    return failure();

  Type voidTy = cir::VoidType::get(ctx);
  Type origRetTy = oldResultTypes.empty() ? voidTy : oldResultTypes[0];
  Type newRetTy = computeNewReturnType(origRetTy, fc.returnInfo, ctx,
                                       [&]() { return funcOp.emitOpError(); });
  if (!newRetTy)
    return failure();
  SmallVector<Type> newResultTypes = {newRetTy};

  if (!funcOp.isDeclaration()) {
    Region &body = funcOp->getRegion(0);
    if (!body.empty()) {
      Block &entry = body.front();

      SmallVector<unsigned> ignored = ignoredArgIndices(fc);
      for (int i = static_cast<int>(ignored.size()) - 1; i >= 0; --i) {
        unsigned blockIdx = ignored[i];
        if (blockIdx >= entry.getNumArguments())
          continue;
        BlockArgument arg = entry.getArgument(blockIdx);
        if (!arg.use_empty()) {
          rewriter.setInsertionPointToStart(&entry);
          auto ptrTy = cir::PointerType::get(arg.getType());
          auto alloca = cir::AllocaOp::create(
              rewriter, funcOp.getLoc(), ptrTy, arg.getType(),
              rewriter.getStringAttr("ignored"), rewriter.getI64IntegerAttr(1));
          auto load = cir::LoadOp::create(
              rewriter, funcOp.getLoc(), arg.getType(), alloca, UnitAttr(),
              UnitAttr(), IntegerAttr(), cir::SyncScopeKindAttr(),
              cir::MemOrderAttr());
          arg.replaceAllUsesWith(load);
        }
        entry.eraseArgument(blockIdx);
      }
    }

    if (fc.returnInfo.kind == ArgKind::Ignore && !oldResultTypes.empty()) {
      SmallVector<cir::ReturnOp> returns;
      funcOp.walk([&](cir::ReturnOp r) { returns.push_back(r); });
      for (cir::ReturnOp r : returns) {
        if (r.getNumOperands() == 0)
          continue;
        rewriter.setInsertionPoint(r);
        cir::ReturnOp::create(rewriter, r.getLoc());
        r.erase();
      }
    }
  }

  Type newFnTy = funcOp.cloneTypeWith(newArgTypes, newResultTypes);
  funcOp.setFunctionTypeAttr(TypeAttr::get(newFnTy));

  SmallVector<unsigned> ignored = ignoredArgIndices(fc);
  if (!ignored.empty())
    if (auto existing = funcOp->getAttrOfType<ArrayAttr>("arg_attrs")) {
      SmallVector<Attribute> kept;
      kept.reserve(newArgTypes.size());
      for (auto [oldIdx, attr] : llvm::enumerate(existing.getValue()))
        if (oldIdx >= fc.argInfos.size() ||
            fc.argInfos[oldIdx].kind != ArgKind::Ignore)
          kept.push_back(attr);
      funcOp->setAttr("arg_attrs", ArrayAttr::get(ctx, kept));
    }

  return success();
}

LogicalResult CIRABIRewriteContext::rewriteCallSite(
    Operation *callOp, const FunctionClassification &fc, OpBuilder &rewriter) {
  if (!needsRewrite(fc))
    return success();

  auto call = cast<cir::CallOp>(callOp);

  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    switch (ac.kind) {
    case ArgKind::Direct:
      if (ac.coercedType)
        return call.emitOpError()
               << "Direct with coerced type at call-site arg " << idx
               << " not yet implemented in CallConvLowering";
      break;
    case ArgKind::Ignore:
    case ArgKind::Expand:
      break;
    case ArgKind::Extend:
      return call.emitOpError() << "Extend at call-site arg " << idx
                                << " not yet implemented in CallConvLowering";
    case ArgKind::Indirect:
      return call.emitOpError() << "Indirect at call-site arg " << idx
                                << " not yet implemented in CallConvLowering";
    }
  }

  SmallVector<Value> newArgs;
  ValueRange argOperands = call.getArgOperands();
  newArgs.reserve(argOperands.size());
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (idx >= argOperands.size())
      break;
    if (ac.kind == ArgKind::Ignore)
      continue;
    newArgs.push_back(argOperands[idx]);
  }
  for (unsigned i = fc.argInfos.size(); i < argOperands.size(); ++i)
    newArgs.push_back(argOperands[i]);

  bool hasResult = call.getNumResults() > 0;
  Type origRetTy = hasResult ? call.getResult().getType()
                             : cir::VoidType::get(callOp->getContext());
  Type callRetTy = origRetTy;
  if (fc.returnInfo.kind == ArgKind::Ignore && hasResult)
    callRetTy = cir::VoidType::get(callOp->getContext());
  if ((fc.returnInfo.kind == ArgKind::Direct ||
       fc.returnInfo.kind == ArgKind::Extend) &&
      fc.returnInfo.coercedType)
    return call.emitOpError() << "Direct/Extend return with coerced type at "
                              << "call-site not yet implemented in "
                              << "CallConvLowering";

  rewriter.setInsertionPoint(call);
  auto newCall = cir::CallOp::create(rewriter, call.getLoc(),
                                     call.getCalleeAttr(), callRetTy, newArgs);
  for (NamedAttribute attr : call->getAttrs())
    if (!newCall->hasAttr(attr.getName()))
      newCall->setAttr(attr.getName(), attr.getValue());

  if (hasResult && fc.returnInfo.kind == ArgKind::Ignore) {
    if (!call.getResult().use_empty()) {
      rewriter.setInsertionPointAfter(newCall);
      auto ptrTy = cir::PointerType::get(origRetTy);
      auto alloca = cir::AllocaOp::create(
          rewriter, call.getLoc(), ptrTy, origRetTy,
          rewriter.getStringAttr("ignored"), rewriter.getI64IntegerAttr(1));
      auto load = cir::LoadOp::create(
          rewriter, call.getLoc(), origRetTy, alloca, UnitAttr(), UnitAttr(),
          IntegerAttr(), cir::SyncScopeKindAttr(), cir::MemOrderAttr());
      call.getResult().replaceAllUsesWith(load);
    }
  } else if (hasResult) {
    call.getResult().replaceAllUsesWith(newCall.getResult());
  }

  call->erase();
  return success();
}
