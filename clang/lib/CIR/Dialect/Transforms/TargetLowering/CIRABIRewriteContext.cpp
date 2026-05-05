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
  // Direct without coercion is a true pass-through; any other kind (or a
  // coerced Direct) means the rewriter must touch the IR.  Extend is
  // technically attribute-only at the IR level but still counts because the
  // attribute attachment changes observable behavior.
  if ((fc.returnInfo.kind != ArgKind::Direct) || fc.returnInfo.coercedType)
    return true;
  for (const ArgClassification &ac : fc.argInfos)
    if ((ac.kind != ArgKind::Direct) || ac.coercedType)
      return true;
  return false;
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
      // Direct with a coerced type means the wire signature uses the
      // coerced type; the body still expects origTy and we'll insert a
      // reinterpret/coercion at the entry block.  Direct without a
      // coerced type is a true pass-through.
      newArgTypes.push_back(ac.coercedType ? ac.coercedType : origTy);
      break;
    case ArgKind::Ignore:
      break;
    case ArgKind::Expand:
      emitError() << "Expand at arg " << idx
                  << " not yet implemented in CallConvLowering";
      return failure();
    case ArgKind::Extend:
      // Extend keeps the original (narrow) type in the signature; the
      // sign/zero extension is communicated to LLVM via the llvm.signext /
      // llvm.zeroext arg attribute, attached separately below.  Any
      // coercedType the classifier set on the Extend ArgClassification is
      // informational (typically the register-width type the value gets
      // extended to in registers) but does not change the CIR signature.
      newArgTypes.push_back(origTy);
      break;
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
    // Direct return with a coerced type uses the coerced type on the wire;
    // the rewriter inserts a coercion before each cir.return.
    return retInfo.coercedType ? retInfo.coercedType : origRetTy;
  case ArgKind::Ignore:
    return cir::VoidType::get(ctx);
  case ArgKind::Expand:
    emitError() << "Expand return is not allowed (classic codegen rejects "
                << "it in EmitFunctionEpilog)";
    return nullptr;
  case ArgKind::Extend:
    // Same convention as Extend args: keep the original return type in the
    // signature; the sign/zero extension is communicated via the
    // llvm.signext / llvm.zeroext res attribute attached separately below.
    return origRetTy;
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

/// Build an updated arg_attrs ArrayAttr that drops Ignore'd args and adds
/// llvm.signext / llvm.zeroext on Extend args.  Preserves any existing arg
/// attributes on retained arg slots.
ArrayAttr updateArgAttrs(MLIRContext *ctx, ArrayAttr existingArgAttrs,
                         const FunctionClassification &fc) {
  SmallVector<Attribute> newArgAttrs;
  newArgAttrs.reserve(fc.argInfos.size());
  for (auto [oldIdx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    DictionaryAttr existing = DictionaryAttr::get(ctx);
    if (existingArgAttrs && oldIdx < existingArgAttrs.size())
      existing = cast<DictionaryAttr>(existingArgAttrs[oldIdx]);
    if (ac.kind == ArgKind::Extend) {
      StringRef attrName = ac.signExtend ? "llvm.signext" : "llvm.zeroext";
      NamedAttribute extAttr(StringAttr::get(ctx, attrName),
                             UnitAttr::get(ctx));
      if (existing.empty()) {
        newArgAttrs.push_back(DictionaryAttr::get(ctx, {extAttr}));
      } else {
        SmallVector<NamedAttribute> attrs(existing.begin(), existing.end());
        attrs.push_back(extAttr);
        newArgAttrs.push_back(DictionaryAttr::get(ctx, attrs));
      }
    } else {
      newArgAttrs.push_back(existing);
    }
  }
  return ArrayAttr::get(ctx, newArgAttrs);
}

/// Build an updated res_attrs ArrayAttr (single entry, since CIR funcs have
/// at most one result) that adds llvm.signext / llvm.zeroext on an Extend
/// return.  Preserves any existing res attributes.
ArrayAttr updateResAttrs(MLIRContext *ctx, ArrayAttr existingResAttrs,
                         const ArgClassification &retInfo) {
  if (retInfo.kind != ArgKind::Extend)
    return existingResAttrs;

  SmallVector<NamedAttribute> attrs;
  if (existingResAttrs && !existingResAttrs.empty())
    for (NamedAttribute na : cast<DictionaryAttr>(existingResAttrs[0]))
      attrs.push_back(na);
  StringRef attrName = retInfo.signExtend ? "llvm.signext" : "llvm.zeroext";
  attrs.push_back(
      NamedAttribute(StringAttr::get(ctx, attrName), UnitAttr::get(ctx)));
  return ArrayAttr::get(ctx, {DictionaryAttr::get(ctx, attrs)});
}

/// Coerce \p src to type \p dstTy at the current builder insertion point.
///
/// Three strategies, in order of preference:
///   - If src and dst are the same type, return src unchanged and leave
///     \p createdOps empty.
///   - If both are non-aggregate same-bit-width values that just differ in
///     vector-vs-scalar shape (e.g. !cir.vector<2 x !cir.float> ↔
///     !cir.complex<!cir.float>), use cir.reinterpret_cast which is free at
///     the IR level.
///   - Otherwise go through memory: allocate a slot of the source type
///     (using max(srcAlign, dstAlign) for the alloca alignment), store
///     the source, bitcast the pointer to the destination type, load the
///     destination type back.
///
/// The temporary alloca is placed at the start of the enclosing function's
/// entry block so that it composes correctly with the HoistAllocas pass
/// regardless of pipeline ordering.
///
/// Any operations the helper creates are appended to \p createdOps so the
/// caller can pass them to replaceAllUsesExcept and avoid clobbering the
/// store's value operand when later rewiring the source value.
Value emitCoercion(OpBuilder &rewriter, Location loc, Type dstTy, Value src,
                   FunctionOpInterface funcOp, const DataLayout &dl,
                   SmallPtrSetImpl<Operation *> &createdOps) {
  Type srcTy = src.getType();
  if (srcTy == dstTy)
    return src;

  // Reinterpret path: same total bit width, neither side is a record, and
  // the shapes differ only in vector-vs-non-vector.  Going through memory
  // is wasteful for these — they have the same in-register representation.
  bool isAggregate = isa<cir::RecordType>(srcTy) || isa<cir::RecordType>(dstTy);
  bool vectorMismatch =
      isa<cir::VectorType>(srcTy) != isa<cir::VectorType>(dstTy);
  if (!isAggregate && vectorMismatch &&
      dl.getTypeSizeInBits(srcTy) == dl.getTypeSizeInBits(dstTy)) {
    auto reinterpret =
        cir::ReinterpretCastOp::create(rewriter, loc, dstTy, src);
    createdOps.insert(reinterpret);
    return reinterpret;
  }

  // Memory path: alloca + store + ptr-cast + load.  The alloca goes in the
  // entry block (Andy's review comment #3 on the original PR), with
  // alignment = max(srcAlign, dstAlign) to satisfy both the store and the
  // load (review comment #1).
  uint64_t srcAlign = dl.getTypeABIAlignment(srcTy);
  uint64_t dstAlign = dl.getTypeABIAlignment(dstTy);
  uint64_t allocaAlign = std::max(srcAlign, dstAlign);

  auto srcPtrTy = cir::PointerType::get(srcTy);
  auto dstPtrTy = cir::PointerType::get(dstTy);

  cir::AllocaOp alloca;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block &entry = funcOp->getRegion(0).front();
    rewriter.setInsertionPointToStart(&entry);
    alloca = cir::AllocaOp::create(rewriter, loc, srcPtrTy, srcTy,
                                   rewriter.getStringAttr("coerce"),
                                   rewriter.getI64IntegerAttr(allocaAlign));
  }
  createdOps.insert(alloca);

  auto store = cir::StoreOp::create(rewriter, loc, src, alloca,
                                    /*isVolatile=*/UnitAttr(),
                                    /*alignment=*/IntegerAttr(),
                                    /*sync_scope=*/cir::SyncScopeKindAttr(),
                                    /*mem_order=*/cir::MemOrderAttr());
  createdOps.insert(store);

  auto ptrCast = cir::CastOp::create(rewriter, loc, dstPtrTy,
                                     cir::CastKind::bitcast, alloca);
  createdOps.insert(ptrCast);

  auto load = cir::LoadOp::create(rewriter, loc, dstTy, ptrCast,
                                  /*isDeref=*/UnitAttr(),
                                  /*isVolatile=*/UnitAttr(),
                                  /*alignment=*/IntegerAttr(),
                                  /*sync_scope=*/cir::SyncScopeKindAttr(),
                                  /*mem_order=*/cir::MemOrderAttr());
  createdOps.insert(load);
  return load;
}

/// Convenience overload for callers that don't need the createdOps set
/// (e.g. call-site coercion where we don't replaceAllUsesExcept).
Value emitCoercion(OpBuilder &rewriter, Location loc, Type dstTy, Value src,
                   FunctionOpInterface funcOp, const DataLayout &dl) {
  SmallPtrSet<Operation *, 4> ignored;
  return emitCoercion(rewriter, loc, dstTy, src, funcOp, dl, ignored);
}

/// Insert coercion before each cir.return so the returned value matches the
/// new (coerced) return type.
void insertReturnCoercion(FunctionOpInterface funcOp, Type origRetTy,
                          Type coercedRetTy, OpBuilder &rewriter,
                          const DataLayout &dl) {
  SmallVector<cir::ReturnOp> returns;
  funcOp.walk([&](cir::ReturnOp r) { returns.push_back(r); });
  for (cir::ReturnOp r : returns) {
    if (r.getInput().empty())
      continue;
    Value origVal = r.getInput()[0];
    if (origVal.getType() == coercedRetTy)
      continue;
    rewriter.setInsertionPoint(r);
    Value coerced =
        emitCoercion(rewriter, r.getLoc(), coercedRetTy, origVal, funcOp, dl);
    r->setOperand(0, coerced);
  }
}

/// For each Direct arg with a coerced type, change the block argument's type
/// to the coerced type and insert a coercion at function entry that maps it
/// back to the original type for body uses.
void insertArgCoercion(FunctionOpInterface funcOp,
                       const FunctionClassification &fc, OpBuilder &rewriter,
                       const DataLayout &dl) {
  Region &body = funcOp->getRegion(0);
  if (body.empty())
    return;
  Block &entry = body.front();

  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind != ArgKind::Direct || !ac.coercedType)
      continue;
    if (idx >= entry.getNumArguments())
      continue;

    BlockArgument blockArg = entry.getArgument(idx);
    Type oldArgTy = blockArg.getType();
    Type newArgTy = ac.coercedType;
    if (oldArgTy == newArgTy)
      continue;

    blockArg.setType(newArgTy);

    rewriter.setInsertionPointToStart(&entry);
    SmallPtrSet<Operation *, 4> coercionOps;
    Value adapted = emitCoercion(rewriter, funcOp.getLoc(), oldArgTy, blockArg,
                                 funcOp, dl, coercionOps);

    // Replace blockArg uses with the adapted value, except inside the helper
    // ops we just created.  This is critical: the StoreOp's value operand is
    // blockArg, and if we naively replaceAllUses it gets swapped to adapted
    // (now of the original type != the alloca's pointee type).
    blockArg.replaceAllUsesExcept(adapted, coercionOps);
  }
}

} // namespace

LogicalResult CIRABIRewriteContext::rewriteFunctionDefinition(
    FunctionOpInterface funcOpInterface, const FunctionClassification &fc,
    OpBuilder &builder) {
  // The pass driver (CallConvLoweringPass) only ever hands us cir.func ops.
  // Cast once at the top so the rest of the function reads in CIR's own
  // vocabulary, and so we can dispatch to the CIRGlobalValueInterface for
  // isDefinition() (FunctionOpInterface alone does not inherit from
  // CIRGlobalValueInterface).
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
      // In-body coercion for Direct-with-coerce / Extend args: change
      // block-arg types to the coerced types and insert a
      // cir.reinterpret_cast at the top of the entry block that converts
      // each coerced value back to its original type, then route existing
      // body uses (including in-body cir.call operands) through the cast.
      // Done before the Ignore-drop below so the entry block argument
      // indices used here still refer to the original positions.
      insertArgCoercion(funcOp, fc, builder, dl);

      // Direct return with coerced type: insert a coercion at every
      // cir.return so the returned value matches the (coerced) return
      // type in the new function signature set below.
      if (fc.returnInfo.kind == ArgKind::Direct && fc.returnInfo.coercedType &&
          !oldResultTypes.empty() && fc.returnInfo.coercedType != origRetTy)
        insertReturnCoercion(funcOp, origRetTy, fc.returnInfo.coercedType,
                             builder, dl);

      Block &entry = body.front();

      // For each Ignored argument: drop the block argument and, if the
      // body still references it, replace those uses with a poison
      // constant.  Ignore classifications mean the value is empty / not
      // passed at the ABI level, so any remaining uses are vacuous;
      // poison says exactly that.  Iterate in reverse so that earlier
      // indices stay stable as later ones are erased.
      for (int blockIdx = static_cast<int>(fc.argInfos.size()) - 1;
           blockIdx >= 0; --blockIdx) {
        if (fc.argInfos[blockIdx].kind != ArgKind::Ignore)
          continue;
        if (static_cast<unsigned>(blockIdx) >= entry.getNumArguments())
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

  // Rebuild arg_attrs when any arg is Ignore (dropped from the output array)
  // or Extend (needs llvm.signext / llvm.zeroext layered on).
  bool needsArgAttrUpdate =
      llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend;
      });
  if (needsArgAttrUpdate) {
    auto existing = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
    funcOp->setAttr("arg_attrs", updateArgAttrs(ctx, existing, fc));
  }

  // Rebuild res_attrs: layer llvm.signext / llvm.zeroext onto an Extend
  // return.
  if (fc.returnInfo.kind == ArgKind::Extend) {
    auto existing = funcOp->getAttrOfType<ArrayAttr>("res_attrs");
    funcOp->setAttr("res_attrs", updateResAttrs(ctx, existing, fc.returnInfo));
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

  MLIRContext *ctx = callOp->getContext();
  auto enclosingFunc = call->getParentOfType<FunctionOpInterface>();

  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    switch (ac.kind) {
    case ArgKind::Direct:
    case ArgKind::Ignore:
      break;
    case ArgKind::Expand:
      return call.emitOpError() << "Expand at call-site arg " << idx
                                << " not yet implemented in CallConvLowering";
    case ArgKind::Extend:
      // Direct (with or without coercion), Ignore, Expand, and Extend are
      // all handled below.  Extend is attribute-only at the IR level.
      break;
    case ArgKind::Indirect:
      return call.emitOpError() << "Indirect at call-site arg " << idx
                                << " not yet implemented in CallConvLowering";
    }
  }

  builder.setInsertionPoint(call);

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
    Value arg = argOperands[idx];
    if (ac.kind == ArgKind::Direct && ac.coercedType &&
        arg.getType() != ac.coercedType)
      arg = emitCoercion(builder, call.getLoc(), ac.coercedType, arg,
                         enclosingFunc, dl);
    newArgs.push_back(arg);
  }

  bool hasResult = call.getNumResults() > 0;
  Type origRetTy =
      hasResult ? call.getResult().getType() : cir::VoidType::get(ctx);
  Type callRetTy = origRetTy;
  if (fc.returnInfo.kind == ArgKind::Ignore && hasResult)
    callRetTy = cir::VoidType::get(ctx);
  bool returnNeedsCoercion =
      hasResult && fc.returnInfo.kind == ArgKind::Direct &&
      fc.returnInfo.coercedType && fc.returnInfo.coercedType != origRetTy;
  if (returnNeedsCoercion)
    callRetTy = fc.returnInfo.coercedType;

  builder.setInsertionPoint(call);
  auto newCall = cir::CallOp::create(builder, call.getLoc(),
                                     call.getCalleeAttr(), callRetTy, newArgs);
  for (NamedAttribute attr : call->getAttrs())
    if (!newCall->hasAttr(attr.getName()))
      newCall->setAttr(attr.getName(), attr.getValue());

  // Direct return with coercion: the new call returns the coerced type;
  // emit a coercion back to the original type for the call's existing uses.
  if (returnNeedsCoercion) {
    builder.setInsertionPointAfter(newCall);
    Value coercedBack = emitCoercion(builder, call.getLoc(), origRetTy,
                                     newCall.getResult(), enclosingFunc, dl);
    call.getResult().replaceAllUsesWith(coercedBack);
  }

  // Layer llvm.signext / llvm.zeroext onto the new call's arg_attrs and
  // res_attrs for Extend args/return.  Ignore args also require a rebuild
  // because their slots are dropped from the output array.
  bool needsArgAttrUpdate =
      llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend;
      });
  if (needsArgAttrUpdate) {
    auto existing = call->getAttrOfType<ArrayAttr>("arg_attrs");
    newCall->setAttr("arg_attrs", updateArgAttrs(ctx, existing, fc));
  }
  if (fc.returnInfo.kind == ArgKind::Extend) {
    auto existing = call->getAttrOfType<ArrayAttr>("res_attrs");
    newCall->setAttr("res_attrs", updateResAttrs(ctx, existing, fc.returnInfo));
  }

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
  } else if (hasResult && !returnNeedsCoercion) {
    // returnNeedsCoercion already wired up the coerced result above.
    call.getResult().replaceAllUsesWith(newCall.getResult());
  }

  call->erase();
  return success();
}
