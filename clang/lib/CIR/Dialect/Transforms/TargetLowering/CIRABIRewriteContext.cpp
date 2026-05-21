//===- CIRABIRewriteContext.cpp - CIR ABI rewrite context ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRABIRewriteContext.h"
#include "mlir/IR/Builders.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace cir;
using namespace mlir;
using namespace mlir::abi;

// This rewrite context currently supports only the Direct (no coercion) and
// Ignore classifications.  All other ArgKinds emit an errorNYI here rather
// than silently passing through, because the IR they would produce is wrong
// (e.g. Expand should flatten an aggregate into multiple primitives, not
// pass it through as a single value).  Subsequent PRs in the
// CallConvLowering split series add the remaining kinds and the
// signature-shaping behavior that goes with them (sret / byval insert
// extra arguments, struct coercion replaces one argument with several).

namespace {

bool needsRewrite(const FunctionClassification &fc) {
  if ((fc.returnInfo.kind != ArgKind::Direct) || fc.returnInfo.coercedType)
    return true;
  for (const ArgClassification &ac : fc.argInfos)
    if ((ac.kind != ArgKind::Direct) || ac.coercedType)
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

/// Build the new argument-type list for a function whose ABI classification
/// is \p fc.  This currently handles only Direct (no coercion) and Ignore;
/// other kinds emit an error.  Classifications that add arguments (e.g.
/// Indirect-sret would prepend a return-pointer arg) are not yet
/// implemented and will arrive in a subsequent PR.
LogicalResult buildNewArgTypes(ArrayRef<Type> oldArgTypes,
                               const FunctionClassification &fc,
                               SmallVectorImpl<Type> &newArgTypes,
                               function_ref<InFlightDiagnostic()> emitError) {
  assert(newArgTypes.empty() && "expected an empty output vector");
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
      emitError() << "Expand at arg " << idx
                  << " not yet implemented in CallConvLowering";
      return failure();
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

/// Compute the new return type for a function whose return classification
/// is \p retInfo.  As with `buildNewArgTypes`, only Direct (no coercion)
/// and Ignore are implemented here; the remaining kinds emit an error.
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
    emitError() << "Expand return is not allowed (classic codegen rejects "
                << "it in EmitFunctionEpilog)";
    return nullptr;
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

/// Create a typed poison constant to stand in for a value the body of a
/// function (or the result of a call) still references but whose ABI
/// classification is Ignore.  Using poison is honest -- the value is
/// genuinely unused at the ABI boundary -- and avoids a fake alloca+load
/// pattern that would suggest we have a value when we don't.
Value createIgnoredValue(OpBuilder &builder, Location loc, Type ty) {
  return cir::ConstantOp::create(builder, loc, ty, cir::PoisonAttr::get(ty));
}

} // namespace

LogicalResult CIRABIRewriteContext::rewriteFunctionDefinition(
    FunctionOpInterface funcOpInterface, const FunctionClassification &fc,
    OpBuilder &builder) {
  // The pass driver (CallConvLoweringPass) only ever hands us cir.func ops,
  // and the body of this routine is end-to-end CIR (it creates cir.constant,
  // cir.return, etc.).  Cast once at the top so the rest of the function
  // reads in CIR's own vocabulary, and so we can dispatch to the
  // CIRGlobalValueInterface for isDefinition() (FunctionOpInterface alone
  // does not inherit from CIRGlobalValueInterface).
  cir::FuncOp funcOp = cast<cir::FuncOp>(funcOpInterface);

  if (!needsRewrite(fc))
    return success();

  ArrayRef<Type> oldArgTypes = funcOp.getArgumentTypes();
  ArrayRef<Type> oldResultTypes = funcOp.getResultTypes();
  MLIRContext *ctx = funcOp->getContext();

  // CIR follows LLVM IR's single-result rule: a function returns either
  // zero or one value.  Document the invariant so a future multi-result
  // change forces us to revisit the return-handling below.
  assert(oldResultTypes.size() <= 1 &&
         "CIR functions return zero or one value");

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

  if (funcOp.isDefinition()) {
    Region &body = funcOp->getRegion(0);
    if (!body.empty()) {
      Block &entry = body.front();

      // For each Ignored argument: drop the block argument and, if the
      // body still references it, replace those uses with a poison
      // constant.  Ignore classifications mean the value is empty / not
      // passed at the ABI level, so any remaining uses are vacuous;
      // poison says exactly that.
      SmallVector<unsigned> ignored = ignoredArgIndices(fc);
      for (unsigned blockIdx : llvm::reverse(ignored)) {
        if (blockIdx >= entry.getNumArguments())
          continue;
        BlockArgument arg = entry.getArgument(blockIdx);
        if (!arg.use_empty()) {
          builder.setInsertionPointToStart(&entry);
          Value poison =
              createIgnoredValue(builder, funcOp.getLoc(), arg.getType());
          arg.replaceAllUsesWith(poison);
        }
        entry.eraseArgument(blockIdx);
      }
    }

    // When the return is classified Ignore but the original function had
    // a non-void return type, every cir.return becomes a naked return.
    // This relies on the invariant that computeNewReturnType has set
    // newRetTy = void for Ignore above, and that the function type is
    // updated below to match.  Asserting this keeps the dependency
    // explicit.
    if (fc.returnInfo.kind == ArgKind::Ignore && !oldResultTypes.empty()) {
      assert(isa<cir::VoidType>(newRetTy) &&
             "Ignore-return path requires the new return type to be void");
      SmallVector<cir::ReturnOp> returns;
      funcOp.walk([&](cir::ReturnOp r) { returns.push_back(r); });
      for (cir::ReturnOp r : returns) {
        if (r.getNumOperands() == 0)
          continue;
        builder.setInsertionPoint(r);
        cir::ReturnOp::create(builder, r.getLoc());
        r.erase();
      }
    }
  }

  Type newFnTy = funcOp.cloneTypeWith(newArgTypes, newResultTypes);
  funcOp.setFunctionTypeAttr(TypeAttr::get(newFnTy));

  // Keep the arg_attrs array in sync with the new argument count by
  // dropping entries for every Ignored argument.  Without this the
  // attribute array would have stale entries that no longer match any
  // block argument.
  SmallVector<unsigned> ignored = ignoredArgIndices(fc);
  if (!ignored.empty()) {
    if (auto existing = funcOp->getAttrOfType<ArrayAttr>("arg_attrs")) {
      SmallVector<Attribute> kept;
      kept.reserve(newArgTypes.size());
      for (auto [oldIdx, attr] : llvm::enumerate(existing.getValue())) {
        if (oldIdx >= fc.argInfos.size() ||
            fc.argInfos[oldIdx].kind != ArgKind::Ignore)
          kept.push_back(attr);
      }
      funcOp->setAttr("arg_attrs", ArrayAttr::get(ctx, kept));
    }
  }

  return success();
}

LogicalResult CIRABIRewriteContext::rewriteCallSite(
    Operation *callOp, const FunctionClassification &fc, OpBuilder &builder) {
  if (!needsRewrite(fc))
    return success();

  if (isa<cir::TryCallOp>(callOp))
    return callOp->emitOpError()
           << "TryCallOp not yet implemented in CallConvLowering";

  auto call = cast<cir::CallOp>(callOp);
  if (call.isIndirect())
    return call.emitOpError()
           << "indirect call not yet implemented in CallConvLowering";

  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    switch (ac.kind) {
    case ArgKind::Direct:
      if (ac.coercedType)
        return call.emitOpError()
               << "Direct with coerced type at call-site arg " << idx
               << " not yet implemented in CallConvLowering";
      break;
    case ArgKind::Ignore:
      break;
    case ArgKind::Expand:
      return call.emitOpError() << "Expand at call-site arg " << idx
                                << " not yet implemented in CallConvLowering";
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
  if (argOperands.size() > fc.argInfos.size())
    return call.emitOpError()
           << "variadic arguments not yet implemented in CallConvLowering";
  assert(fc.argInfos.size() == argOperands.size() &&
         "call operand count must match classified arg count");
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    newArgs.push_back(argOperands[idx]);
  }

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

  builder.setInsertionPoint(call);
  auto newCall = cir::CallOp::create(builder, call.getLoc(),
                                     call.getCalleeAttr(), callRetTy, newArgs);
  for (NamedAttribute attr : call->getAttrs())
    if (!newCall->hasAttr(attr.getName()))
      newCall->setAttr(attr.getName(), attr.getValue());

  if (hasResult && fc.returnInfo.kind == ArgKind::Ignore) {
    // The new call returns void, but the original call's result may still
    // have uses.  Substitute a poison constant of the original type so
    // those uses remain well-formed without pretending we have a real
    // value at the ABI boundary.
    if (!call.getResult().use_empty()) {
      builder.setInsertionPointAfter(newCall);
      Value poison = createIgnoredValue(builder, call.getLoc(), origRetTy);
      call.getResult().replaceAllUsesWith(poison);
    }
  } else if (hasResult) {
    call.getResult().replaceAllUsesWith(newCall.getResult());
  }

  call->erase();
  return success();
}
