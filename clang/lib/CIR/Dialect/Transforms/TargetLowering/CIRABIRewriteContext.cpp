//===- CIRABIRewriteContext.cpp - CIR ABI rewrite context ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRABIRewriteContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

using namespace cir;
using namespace mlir;
using namespace mlir::abi;

// This rewrite context supports the Direct (with or without coercion),
// Extend, Ignore, and Indirect-return (sret) classifications.  Indirect
// arguments (byval) and Expand still emit an errorNYI here rather than
// silently passing through, because the IR they would produce is wrong
// (e.g. Expand should flatten an aggregate into multiple primitives, not
// pass it through as a single value).  byval and struct coercion are not
// yet handled here; they need the signature-shaping that goes with them
// (byval inserts an extra pointer argument, struct coercion replaces one
// argument with several).

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
/// is \p fc.  Handles Direct (with or without coercion), Extend, and Ignore.
/// Indirect (byval) arguments and Expand emit an error.  The sret return
/// pointer, when present, is prepended by rewriteFunctionDefinition rather
/// than here.
mlir::LogicalResult
buildNewArgTypes(ArrayRef<mlir::Type> oldArgTypes,
                 const FunctionClassification &fc,
                 SmallVectorImpl<mlir::Type> &newArgTypes,
                 function_ref<mlir::InFlightDiagnostic()> emitError) {
  assert(newArgTypes.empty() && "expected an empty output vector");
  newArgTypes.reserve(oldArgTypes.size());
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    mlir::Type origTy = oldArgTypes[idx];
    switch (ac.kind) {
    case ArgKind::Direct:
      // Direct with a coerced type means the wire signature uses the
      // coerced type; the body still expects origTy and we'll insert a
      // coercion at the entry block.  Direct without a coerced type is a
      // true pass-through.
      newArgTypes.push_back(ac.coercedType ? ac.coercedType : origTy);
      break;
    case ArgKind::Ignore:
      break;
    case ArgKind::Expand:
      emitError() << "Expand at arg " << idx
                  << " not yet implemented in CallConvLowering";
      return mlir::failure();
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
      return mlir::failure();
    }
  }
  return mlir::success();
}

/// Compute the new return type for a function whose return classification
/// is \p retInfo.  Direct returns keep (or coerce to) their type, Ignore and
/// Indirect (sret) returns become void, Extend keeps its type; Expand emits
/// an error.
mlir::Type
computeNewReturnType(mlir::Type origRetTy, const ArgClassification &retInfo,
                     mlir::MLIRContext *ctx,
                     function_ref<mlir::InFlightDiagnostic()> emitError) {
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
    // sret: the value is returned through a pointer argument that the ABI
    // synthesizes (rewriteFunctionDefinition prepends it to the argument
    // list); it is not part of the source-level signature, so the wire
    // return type becomes void.
    return cir::VoidType::get(ctx);
  }
  llvm_unreachable("all ArgKind cases handled");
}

/// Create a typed poison constant to stand in for a value the body of a
/// function (or the result of a call) still references but whose ABI
/// classification is Ignore.  Using poison is honest -- the value is
/// genuinely unused at the ABI boundary -- and avoids a fake alloca+load
/// pattern that would suggest we have a value when we don't.
mlir::Value createIgnoredValue(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Type ty) {
  return cir::ConstantOp::create(builder, loc, ty, cir::PoisonAttr::get(ty));
}

/// Build an updated arg_attrs ArrayAttr that drops Ignore'd args and adds
/// llvm.signext / llvm.zeroext on Extend args.  Preserves any existing arg
/// attributes on retained arg slots.
mlir::ArrayAttr updateArgAttrs(mlir::MLIRContext *ctx,
                               mlir::ArrayAttr existingArgAttrs,
                               const FunctionClassification &fc) {
  SmallVector<mlir::Attribute> newArgAttrs;
  newArgAttrs.reserve(fc.argInfos.size());
  for (auto [oldIdx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    mlir::DictionaryAttr existing = mlir::DictionaryAttr::get(ctx);
    if (existingArgAttrs && oldIdx < existingArgAttrs.size())
      existing = mlir::cast<mlir::DictionaryAttr>(existingArgAttrs[oldIdx]);
    if (ac.kind == ArgKind::Extend) {
      StringRef attrName = ac.signExtend ? "llvm.signext" : "llvm.zeroext";
      mlir::NamedAttribute extAttr(mlir::StringAttr::get(ctx, attrName),
                                   mlir::UnitAttr::get(ctx));
      if (existing.empty()) {
        newArgAttrs.push_back(mlir::DictionaryAttr::get(ctx, {extAttr}));
      } else {
        SmallVector<mlir::NamedAttribute> attrs(existing.begin(),
                                                existing.end());
        attrs.push_back(extAttr);
        newArgAttrs.push_back(mlir::DictionaryAttr::get(ctx, attrs));
      }
    } else {
      newArgAttrs.push_back(existing);
    }
  }
  return mlir::ArrayAttr::get(ctx, newArgAttrs);
}

/// Build an updated res_attrs ArrayAttr (single entry, since CIR funcs have
/// at most one result) that adds llvm.signext / llvm.zeroext on an Extend
/// return.  Preserves any existing res attributes.
mlir::ArrayAttr updateResAttrs(mlir::MLIRContext *ctx,
                               mlir::ArrayAttr existingResAttrs,
                               const ArgClassification &retInfo) {
  if (retInfo.kind != ArgKind::Extend)
    return existingResAttrs;

  SmallVector<mlir::NamedAttribute> attrs;
  if (existingResAttrs && !existingResAttrs.empty())
    for (mlir::NamedAttribute na :
         mlir::cast<mlir::DictionaryAttr>(existingResAttrs[0]))
      attrs.push_back(na);
  StringRef attrName = retInfo.signExtend ? "llvm.signext" : "llvm.zeroext";
  attrs.push_back(mlir::NamedAttribute(mlir::StringAttr::get(ctx, attrName),
                                       mlir::UnitAttr::get(ctx)));
  return mlir::ArrayAttr::get(ctx, {mlir::DictionaryAttr::get(ctx, attrs)});
}

/// Coerce \p src to type \p dstTy at the current builder insertion point by
/// going through memory: allocate a slot, store the source, then load the
/// destination type back out.  Lowers uniformly for scalar, vector, and
/// record types.
///
/// The slot is sized to the larger of the two types so that neither the
/// store nor the load ever runs past it: the coerced ABI type can be larger
/// than the original (e.g. a 12-byte aggregate returned as `{i64, i64}`), so
/// loading the destination out of a source-sized slot would over-read.
/// Alignment is max(srcAlign, dstAlign) to satisfy both accesses.  The slot
/// is accessed through a source-typed view for the store and a
/// destination-typed view for the load.
///
/// The temporary alloca is placed at the start of the enclosing function's
/// entry block so that it composes correctly with the HoistAllocas pass
/// regardless of pipeline ordering.
///
/// Any operations the helper creates are appended to \p createdOps so the
/// caller can pass them to replaceAllUsesExcept and avoid clobbering the
/// store's value operand when later rewiring the source value.
mlir::Value emitCoercion(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type dstTy, mlir::Value src,
                         mlir::FunctionOpInterface funcOp,
                         const mlir::DataLayout &dl,
                         SmallPtrSetImpl<mlir::Operation *> &createdOps) {
  mlir::Type srcTy = src.getType();
  assert(srcTy != dstTy &&
         "emitCoercion callers must pre-check that the types differ");

  uint64_t srcAlign = dl.getTypeABIAlignment(srcTy);
  uint64_t dstAlign = dl.getTypeABIAlignment(dstTy);
  uint64_t allocaAlign = std::max(srcAlign, dstAlign);
  mlir::Type slotTy =
      dl.getTypeSize(srcTy) >= dl.getTypeSize(dstTy) ? srcTy : dstTy;

  auto slotPtrTy = cir::PointerType::get(slotTy);
  auto srcPtrTy = cir::PointerType::get(srcTy);
  auto dstPtrTy = cir::PointerType::get(dstTy);

  cir::AllocaOp alloca;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block &entry = funcOp->getRegion(0).front();
    builder.setInsertionPointToStart(&entry);
    alloca = cir::AllocaOp::create(builder, loc, slotPtrTy,
                                   builder.getStringAttr("coerce"),
                                   builder.getI64IntegerAttr(allocaAlign));
  }
  createdOps.insert(alloca);

  // Store through a source-typed view of the slot.
  mlir::Value srcSlot = alloca;
  if (slotTy != srcTy) {
    auto srcCast = cir::CastOp::create(builder, loc, srcPtrTy,
                                       cir::CastKind::bitcast, alloca);
    createdOps.insert(srcCast);
    srcSlot = srcCast;
  }
  auto store = cir::StoreOp::create(builder, loc, src, srcSlot);
  createdOps.insert(store);

  // Load through a destination-typed view of the slot.
  mlir::Value dstSlot = alloca;
  if (slotTy != dstTy) {
    auto dstCast = cir::CastOp::create(builder, loc, dstPtrTy,
                                       cir::CastKind::bitcast, alloca);
    createdOps.insert(dstCast);
    dstSlot = dstCast;
  }
  auto load = cir::LoadOp::create(builder, loc, dstSlot);
  createdOps.insert(load);
  return load;
}

/// Convenience overload for callers that don't need the createdOps set
/// (e.g. call-site coercion where we don't replaceAllUsesExcept).
mlir::Value emitCoercion(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type dstTy, mlir::Value src,
                         mlir::FunctionOpInterface funcOp,
                         const mlir::DataLayout &dl) {
  SmallPtrSet<mlir::Operation *, 4> ignored;
  return emitCoercion(builder, loc, dstTy, src, funcOp, dl, ignored);
}

/// Insert coercion before each cir.return so the returned value matches the
/// new (coerced) return type.
void insertReturnCoercion(mlir::FunctionOpInterface funcOp,
                          mlir::Type origRetTy, mlir::Type coercedRetTy,
                          mlir::OpBuilder &builder,
                          const mlir::DataLayout &dl) {
  SmallVector<cir::ReturnOp> returns;
  funcOp.walk([&](cir::ReturnOp r) { returns.push_back(r); });
  for (cir::ReturnOp r : returns) {
    if (r.getInput().empty())
      continue;
    mlir::Value origVal = r.getInput()[0];
    if (origVal.getType() == coercedRetTy)
      continue;
    builder.setInsertionPoint(r);
    mlir::Value coerced =
        emitCoercion(builder, r.getLoc(), coercedRetTy, origVal, funcOp, dl);
    r->setOperand(0, coerced);
  }
}

/// For each Direct arg with a coerced type, change the block argument's type
/// to the coerced type and insert a coercion at function entry that maps it
/// back to the original type for body uses.
///
/// The entry block arguments mirror the function's ABI signature: argument
/// \p hasSRetArg shifts the classification index by one because a hidden
/// sret pointer occupies block argument 0 when the function returns by
/// reference.  So fc.argInfos[i] corresponds to block argument
/// i + hasSRetArg.
void insertArgCoercion(mlir::FunctionOpInterface funcOp,
                       const FunctionClassification &fc,
                       mlir::OpBuilder &builder, const mlir::DataLayout &dl,
                       bool hasSRetArg) {
  mlir::Region &body = funcOp->getRegion(0);
  if (body.empty())
    return;
  mlir::Block &entry = body.front();

  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind != ArgKind::Direct || !ac.coercedType)
      continue;
    unsigned blockIdx = idx + hasSRetArg;
    if (blockIdx >= entry.getNumArguments())
      continue;

    mlir::BlockArgument blockArg = entry.getArgument(blockIdx);
    mlir::Type oldArgTy = blockArg.getType();
    mlir::Type newArgTy = ac.coercedType;
    if (oldArgTy == newArgTy)
      continue;

    blockArg.setType(newArgTy);

    builder.setInsertionPointToStart(&entry);
    SmallPtrSet<mlir::Operation *, 4> coercionOps;
    mlir::Value adapted = emitCoercion(builder, funcOp.getLoc(), oldArgTy,
                                       blockArg, funcOp, dl, coercionOps);

    // Replace blockArg uses with the adapted value, except inside the helper
    // ops we just created.  This is critical: the StoreOp's value operand is
    // blockArg, and if we naively replaceAllUses it gets swapped to adapted
    // (now of the original type != the alloca's pointee type).
    blockArg.replaceAllUsesExcept(adapted, coercionOps);
  }
}

/// Rewrite each cir.return so the return value flows through the sret
/// pointer (the prepended first block argument) and the function returns
/// void.
///
/// CIRGen emits a local `__retval` alloca and emits `cir.return %loaded`
/// where `%loaded = cir.load __retval`.  The naive lowering -- store the
/// loaded SSA value through the sret pointer -- byte-copies the record,
/// which is wrong for non-trivially-copyable types: e.g. libstdc++'s SSO
/// `std::string` has a `_M_p` pointer that aliases the source's internal
/// `_M_local_buf`, so a byte-copy leaves the destination pointing at the
/// source's (now-dying) stack storage and the destination's destructor
/// later `free()`s a stack pointer.
///
/// Instead, route construction directly into the sret slot: find the
/// `__retval` alloca, replace its uses with the sret pointer, and drop the
/// trailing `cir.load __retval` so the rewritten return has no operand.
/// The CIRGen-emitted constructor / store-into-`__retval` then targets the
/// sret slot uniformly, matching classic CodeGen's "construct directly into
/// `%agg.result`" pattern.
///
/// CIRGen emits one `%v = cir.load %__retval` / `cir.return %v` pair per
/// return statement, and every such load reads the single `__retval`
/// alloca (CIR does not merge returns into a shared epilogue block).  The
/// alloca is therefore rewired to the sret pointer once; each cir.return is
/// then collapsed to a bare return and its now-dead load erased.  This
/// `cir.return (cir.load <alloca>)` shape is an invariant guaranteed by
/// CIRGen, so it is asserted via `cast<>` rather than guarded with a
/// fallback.
void insertSRetStores(mlir::FunctionOpInterface funcOp, mlir::Type origRetTy,
                      mlir::OpBuilder &builder) {
  mlir::Value sretPtr = funcOp.getArguments()[0];

  SmallVector<cir::ReturnOp> returnOps;
  funcOp->walk([&](cir::ReturnOp retOp) { returnOps.push_back(retOp); });

  cir::AllocaOp retAlloca = nullptr;
  for (cir::ReturnOp retOp : returnOps) {
    // Every cir.return in an sret function must carry the loaded return
    // value -- a bare return would mean the sret slot was never written.
    assert(!retOp.getInput().empty() &&
           "cir.return in sret function must have an operand");

    cir::LoadOp retLoad =
        mlir::cast<cir::LoadOp>(retOp.getInput()[0].getDefiningOp());

    // Rewire the shared `__retval` alloca to the sret pointer once.
    // replaceAllUsesWith updates every load of the alloca (including those
    // feeding the other cir.return ops) to read from sretPtr instead, so
    // all returns are covered by this single rewiring.  Only then is the
    // now-unused alloca safe to erase.
    if (!retAlloca) {
      retAlloca = mlir::cast<cir::AllocaOp>(retLoad.getAddr().getDefiningOp());
      retAlloca.getResult().replaceAllUsesWith(sretPtr);
      retAlloca->erase();
    }

    // The sret slot now holds the return value directly; replace the
    // value-carrying return with a void return (no operand).
    builder.setInsertionPoint(retOp);
    cir::ReturnOp::create(builder, retOp.getLoc());
    retOp->erase();
    if (retLoad.use_empty())
      retLoad->erase();
  }
}

/// Build the attribute dictionary for the sret slot (slot 0 of an
/// sret-returning function or call).  Matches classic CodeGen's
/// `sret(T) align A [noalias] writable dead_on_unwind`.  noalias is only
/// valid on the callee's parameter, not at the call site, so it is gated by
/// \p withNoalias.  Key order is irrelevant: DictionaryAttr sorts by name.
SmallVector<mlir::NamedAttribute> buildSretSlotAttrs(mlir::OpBuilder &builder,
                                                     mlir::Type retTy,
                                                     uint64_t align,
                                                     bool withNoalias) {
  SmallVector<mlir::NamedAttribute> attrs;
  // The sret type must be carried explicitly: LLVM's sret attribute requires
  // it, and once the CIR `!cir.ptr<retTy>` lowers to an opaque LLVM `ptr` the
  // pointee type can no longer be recovered from the pointer.
  attrs.push_back(
      builder.getNamedAttr("llvm.sret", mlir::TypeAttr::get(retTy)));
  attrs.push_back(
      builder.getNamedAttr("llvm.align", builder.getI64IntegerAttr(align)));
  if (withNoalias)
    attrs.push_back(
        builder.getNamedAttr("llvm.noalias", builder.getUnitAttr()));
  attrs.push_back(builder.getNamedAttr("llvm.writable", builder.getUnitAttr()));
  attrs.push_back(
      builder.getNamedAttr("llvm.dead_on_unwind", builder.getUnitAttr()));
  return attrs;
}

/// Prepend the sret slot's attrs at position 0 of newCall's arg_attrs.
/// Called after the call has been rewritten with the sret pointer at
/// operand 0, so the operand count now includes the sret slot.  \p argAttrs
/// must already be shaped for the rewritten argument list (Extend slots
/// carry signext/zeroext, Ignore slots dropped); it is shifted to slots
/// 1..N behind the sret slot.
void applySretSlotAttrs(cir::CallOp newCall, mlir::ArrayAttr argAttrs,
                        mlir::Type retTy, uint64_t align,
                        mlir::OpBuilder &builder) {
  mlir::MLIRContext *ctx = newCall->getContext();
  SmallVector<mlir::NamedAttribute> sretAttrs =
      buildSretSlotAttrs(builder, retTy, align, /*withNoalias=*/false);

  SmallVector<mlir::Attribute> newArgAttrs;
  newArgAttrs.reserve(newCall.getArgOperands().size());
  newArgAttrs.push_back(mlir::DictionaryAttr::get(ctx, sretAttrs));
  if (argAttrs)
    llvm::append_range(newArgAttrs, argAttrs);
  assert(newArgAttrs.size() <= newCall.getArgOperands().size() &&
         "arg_attrs wider than the rewritten call's operand list");
  newArgAttrs.resize(newCall.getArgOperands().size(),
                     mlir::DictionaryAttr::get(ctx));
  newCall->setAttr("arg_attrs", mlir::ArrayAttr::get(ctx, newArgAttrs));
}

/// Rewrite an indirect-return (sret) call site: prepend a return-slot
/// pointer as operand 0, make the call return void, and either reuse a
/// dominating single-use store destination as the slot (so construction
/// flows directly into it) or allocate a fresh slot and load the result
/// back out.  \p newArgs is the already-shaped (Ignore-dropped,
/// coercion-applied) non-sret argument list.  The caller guarantees the
/// call has a result and an indirect-return classification.
void rewriteIndirectReturnCall(cir::CallOp call,
                               const FunctionClassification &fc,
                               ArrayRef<mlir::Value> newArgs,
                               mlir::Type origRetTy, mlir::OpBuilder &builder) {
  mlir::MLIRContext *ctx = call->getContext();
  auto ptrTy = cir::PointerType::get(origRetTy);
  builder.setInsertionPoint(call);
  uint64_t sretAlign = fc.returnInfo.indirectAlign.value();

  // CIRGen emits `cir.store %callResult, %dest` when the call's result is
  // bound to a local (e.g. `T s = make();`).  Allocating a fresh sret slot
  // and copying into %dest would byte-copy the record, which is wrong for
  // non-trivially-copyable types (the libstdc++ SSO `_M_p` pointer
  // survives a byte-copy but ends up pointing at the dying temp's local
  // buffer, so the destination's destructor later `free()`s a stack
  // pointer).  When the result has a single store-into-%dest use, use
  // %dest as the sret slot directly so construction flows into it,
  // matching classic CodeGen's "pass %s as sret" pattern.  %dest must
  // dominate the call so the rewritten call (which takes it as operand 0)
  // does not use a value before its definition.
  mlir::Value sretSlot = nullptr;
  cir::StoreOp reuseStore = nullptr;
  if (call.getResult().hasOneUse()) {
    mlir::Operation *user = *call.getResult().getUsers().begin();
    if (auto store = mlir::dyn_cast<cir::StoreOp>(user))
      if (store.getValue() == call.getResult() &&
          store.getAddr().getType() == ptrTy &&
          mlir::DominanceInfo().properlyDominates(store.getAddr(), call)) {
        sretSlot = store.getAddr();
        reuseStore = store;
      }
  }
  if (!sretSlot) {
    auto alloca = cir::AllocaOp::create(
        builder, call.getLoc(), ptrTy,
        /*name=*/builder.getStringAttr("sret"),
        /*alignment=*/builder.getI64IntegerAttr(sretAlign));
    sretSlot = alloca;
  }

  SmallVector<mlir::Value> sretArgs;
  sretArgs.push_back(sretSlot);
  sretArgs.append(newArgs.begin(), newArgs.end());

  mlir::Type sretVoidTy = cir::VoidType::get(ctx);
  auto newCall = cir::CallOp::create(
      builder, call.getLoc(), call.getCalleeAttr(), sretVoidTy, sretArgs);
  for (mlir::NamedAttribute attr : call->getAttrs())
    if (!newCall->hasAttr(attr.getName()))
      newCall->setAttr(attr.getName(), attr.getValue());

  // Shape the per-argument attrs exactly as the non-sret path does
  // (signext / zeroext for Extend, drop Ignore slots) before prepending
  // the sret slot, so sret composes correctly with Extend / Ignore args.
  mlir::ArrayAttr argAttrs = call->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
  bool needsArgAttrUpdate =
      llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend;
      });
  if (needsArgAttrUpdate)
    argAttrs = updateArgAttrs(ctx, argAttrs, fc);
  applySretSlotAttrs(newCall, argAttrs, origRetTy, sretAlign, builder);

  if (reuseStore) {
    // The callee now constructs directly into the destination slot, so the
    // original store-from-result is redundant; dropping it avoids a
    // byte-copy of the record.
    reuseStore->erase();
  } else {
    builder.setInsertionPointAfter(newCall);
    auto load = cir::LoadOp::create(builder, call.getLoc(), origRetTy, sretSlot,
                                    /*isDeref=*/mlir::UnitAttr(),
                                    /*isVolatile=*/mlir::UnitAttr(),
                                    /*alignment=*/mlir::IntegerAttr(),
                                    /*sync_scope=*/cir::SyncScopeKindAttr(),
                                    /*mem_order=*/cir::MemOrderAttr());
    call.getResult().replaceAllUsesWith(load);
  }
  call->erase();
}

} // namespace

mlir::LogicalResult CIRABIRewriteContext::rewriteFunctionDefinition(
    mlir::FunctionOpInterface funcOpInterface, const FunctionClassification &fc,
    mlir::OpBuilder &builder) {
  // The pass driver (CallConvLoweringPass) only ever hands us cir.func ops.
  // Cast once at the top so the rest of the function reads in CIR's own
  // vocabulary, and so we can dispatch to the CIRGlobalValueInterface for
  // isDefinition() (FunctionOpInterface alone does not inherit from
  // CIRGlobalValueInterface).
  cir::FuncOp funcOp = mlir::cast<cir::FuncOp>(funcOpInterface);

  if (!needsRewrite(fc))
    return mlir::success();

  ArrayRef<mlir::Type> oldArgTypes = funcOp.getArgumentTypes();
  ArrayRef<mlir::Type> oldResultTypes = funcOp.getResultTypes();
  mlir::MLIRContext *ctx = funcOp->getContext();

  // CIR follows LLVM IR's single-result rule: a function returns either
  // zero or one value.  Document the invariant so a future multi-result
  // change forces us to revisit the return-handling below.
  assert(oldResultTypes.size() <= 1 &&
         "CIR functions return zero or one value");

  SmallVector<mlir::Type> newArgTypes;
  if (mlir::failed(buildNewArgTypes(oldArgTypes, fc, newArgTypes,
                                    [&]() { return funcOp.emitOpError(); })))
    return mlir::failure();

  mlir::Type voidTy = cir::VoidType::get(ctx);
  mlir::Type origRetTy = oldResultTypes.empty() ? voidTy : oldResultTypes[0];
  mlir::Type newRetTy = computeNewReturnType(
      origRetTy, fc.returnInfo, ctx, [&]() { return funcOp.emitOpError(); });
  if (!newRetTy)
    return mlir::failure();
  SmallVector<mlir::Type> newResultTypes = {newRetTy};

  // sret return: the value is returned through a pointer the ABI inserts as
  // argument 0.  This pointer is not part of the function's source-level
  // signature -- it is synthesized here -- and the wire return type was
  // already set to void by computeNewReturnType.  Every classification index
  // therefore maps to a block argument shifted by one in the body handling
  // below.
  bool hasSRet =
      fc.returnInfo.kind == ArgKind::Indirect && !oldResultTypes.empty();
  if (hasSRet)
    newArgTypes.insert(newArgTypes.begin(), cir::PointerType::get(origRetTy));

  if (funcOp.isDefinition()) {
    mlir::Region &body = funcOp->getRegion(0);
    if (!body.empty()) {
      // Prepend the sret pointer block argument and route every cir.return
      // through it before any index-based argument handling below (which
      // then accounts for the +1 offset).
      if (hasSRet) {
        body.front().insertArgument(0u, cir::PointerType::get(origRetTy),
                                    funcOp.getLoc());
        insertSRetStores(funcOp, origRetTy, builder);
      }

      // In-body coercion for Direct-with-coerce / Extend args: change
      // block-arg types to the coerced types and insert a memory roundtrip
      // at the top of the entry block that converts each coerced value back
      // to its original type, then route existing body uses (including
      // in-body cir.call operands) through the recovered value.  Done before
      // the Ignore-drop below so the entry block argument indices used here
      // still refer to the original positions.
      insertArgCoercion(funcOp, fc, builder, dl, hasSRet);

      // Direct return with coerced type: insert a coercion at every
      // cir.return so the returned value matches the (coerced) return
      // type in the new function signature set below.
      if (fc.returnInfo.kind == ArgKind::Direct && fc.returnInfo.coercedType &&
          !oldResultTypes.empty() && fc.returnInfo.coercedType != origRetTy)
        insertReturnCoercion(funcOp, origRetTy, fc.returnInfo.coercedType,
                             builder, dl);

      mlir::Block &entry = body.front();

      // For each Ignored argument: drop the block argument and, if the
      // body still references it, replace those uses with a poison
      // constant.  Ignore classifications mean the value is empty / not
      // passed at the ABI level, so any remaining uses are vacuous;
      // poison says exactly that.  Iterate in reverse so that earlier
      // indices stay stable as later ones are erased.
      for (int argInfoIdx = static_cast<int>(fc.argInfos.size()) - 1;
           argInfoIdx >= 0; --argInfoIdx) {
        if (fc.argInfos[argInfoIdx].kind != ArgKind::Ignore)
          continue;
        unsigned blockIdx = static_cast<unsigned>(argInfoIdx) + hasSRet;
        if (blockIdx >= entry.getNumArguments())
          continue;
        mlir::BlockArgument arg = entry.getArgument(blockIdx);
        if (!arg.use_empty()) {
          builder.setInsertionPointToStart(&entry);
          mlir::Value poison =
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
      assert(mlir::isa<cir::VoidType>(newRetTy) &&
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

  mlir::Type newFnTy = funcOp.cloneTypeWith(newArgTypes, newResultTypes);
  funcOp.setFunctionTypeAttr(mlir::TypeAttr::get(newFnTy));

  // Rebuild arg_attrs when the function has an sret slot (slot 0 needs the
  // sret attribute set) or any arg is Ignore (dropped from the output array)
  // or Extend (needs llvm.signext / llvm.zeroext layered on).
  bool needsArgAttrUpdate =
      hasSRet || llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend;
      });
  if (needsArgAttrUpdate) {
    auto existing = funcOp->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
    mlir::ArrayAttr updated = updateArgAttrs(ctx, existing, fc);
    if (hasSRet) {
      // Prepend the sret slot's attribute dict (slot 0); the per-argument
      // dicts shift to slots 1..N.  noalias is valid only on the callee's
      // parameter, so it is added only for definitions.
      SmallVector<mlir::NamedAttribute> sretAttrs = buildSretSlotAttrs(
          builder, origRetTy, fc.returnInfo.indirectAlign.value(),
          /*withNoalias=*/funcOp.isDefinition());
      SmallVector<mlir::Attribute> withSret;
      withSret.push_back(mlir::DictionaryAttr::get(ctx, sretAttrs));
      llvm::append_range(withSret, updated);
      funcOp->setAttr("arg_attrs", mlir::ArrayAttr::get(ctx, withSret));
    } else {
      funcOp->setAttr("arg_attrs", updated);
    }
  }

  // Rebuild res_attrs: layer llvm.signext / llvm.zeroext onto an Extend
  // return.
  if (fc.returnInfo.kind == ArgKind::Extend) {
    auto existing = funcOp->getAttrOfType<mlir::ArrayAttr>("res_attrs");
    funcOp->setAttr("res_attrs", updateResAttrs(ctx, existing, fc.returnInfo));
  }

  return mlir::success();
}

mlir::LogicalResult
CIRABIRewriteContext::rewriteCallSite(mlir::Operation *callOp,
                                      const FunctionClassification &fc,
                                      mlir::OpBuilder &builder) {
  if (!needsRewrite(fc))
    return mlir::success();

  if (mlir::isa<cir::TryCallOp>(callOp))
    return callOp->emitOpError()
           << "TryCallOp not yet implemented in CallConvLowering";

  auto call = mlir::cast<cir::CallOp>(callOp);
  if (call.isIndirect())
    return call.emitOpError()
           << "indirect call not yet implemented in CallConvLowering";

  mlir::MLIRContext *ctx = callOp->getContext();
  auto enclosingFunc = call->getParentOfType<mlir::FunctionOpInterface>();

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

  SmallVector<mlir::Value> newArgs;
  mlir::ValueRange argOperands = call.getArgOperands();
  newArgs.reserve(argOperands.size());
  if (argOperands.size() > fc.argInfos.size())
    return call.emitOpError()
           << "variadic arguments not yet implemented in CallConvLowering";
  assert(fc.argInfos.size() == argOperands.size() &&
         "call operand count must match classified arg count");
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    mlir::Value arg = argOperands[idx];
    if (ac.kind == ArgKind::Direct && ac.coercedType &&
        arg.getType() != ac.coercedType)
      arg = emitCoercion(builder, call.getLoc(), ac.coercedType, arg,
                         enclosingFunc, dl);
    newArgs.push_back(arg);
  }

  bool hasResult = call.getNumResults() > 0;
  mlir::Type origRetTy =
      hasResult ? call.getResult().getType() : cir::VoidType::get(ctx);

  // An indirect (sret) return has a different call shape than the coerce /
  // extend / ignore return handling further down (the value is returned
  // through a prepended pointer slot, not as a result), so dispatch to a
  // dedicated helper for it; everything below handles the by-value returns.
  if (fc.returnInfo.kind == ArgKind::Indirect && hasResult) {
    rewriteIndirectReturnCall(call, fc, newArgs, origRetTy, builder);
    return mlir::success();
  }

  mlir::Type callRetTy = origRetTy;
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
  for (mlir::NamedAttribute attr : call->getAttrs())
    if (!newCall->hasAttr(attr.getName()))
      newCall->setAttr(attr.getName(), attr.getValue());

  // Direct return with coercion: the new call returns the coerced type;
  // emit a coercion back to the original type for the call's existing uses.
  if (returnNeedsCoercion) {
    builder.setInsertionPointAfter(newCall);
    mlir::Value coercedBack =
        emitCoercion(builder, call.getLoc(), origRetTy, newCall.getResult(),
                     enclosingFunc, dl);
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
    auto existing = call->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
    newCall->setAttr("arg_attrs", updateArgAttrs(ctx, existing, fc));
  }
  if (fc.returnInfo.kind == ArgKind::Extend) {
    auto existing = call->getAttrOfType<mlir::ArrayAttr>("res_attrs");
    newCall->setAttr("res_attrs", updateResAttrs(ctx, existing, fc.returnInfo));
  }

  if (hasResult && fc.returnInfo.kind == ArgKind::Ignore) {
    // The new call returns void, but the original call's result may still
    // have uses.  Substitute a poison constant of the original type so
    // those uses remain well-formed without pretending we have a real
    // value at the ABI boundary.
    if (!call.getResult().use_empty()) {
      builder.setInsertionPointAfter(newCall);
      mlir::Value poison =
          createIgnoredValue(builder, call.getLoc(), origRetTy);
      call.getResult().replaceAllUsesWith(poison);
    }
  } else if (hasResult && !returnNeedsCoercion) {
    // returnNeedsCoercion already wired up the coerced result above.
    call.getResult().replaceAllUsesWith(newCall.getResult());
  }

  call->erase();
  return mlir::success();
}
