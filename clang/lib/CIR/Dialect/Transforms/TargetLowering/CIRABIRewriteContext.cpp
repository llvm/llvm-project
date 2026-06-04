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
// Extend, Ignore, Indirect-return (sret), Indirect-argument (byval and
// byref), and Expand (struct flattening) classifications.
//
// For byval (ArgClassification::byVal == true) the callee gets
// llvm.byval + llvm.noalias + llvm.noundef; for byref (byVal == false)
// the callee gets llvm.byref without the ownership attrs.  Both pass
// through an alloca+store at the call site.
//
// For Expand, the single struct argument is replaced by N scalar arguments
// (one per field).  At the callee, the N block arguments are reassembled
// into the struct via an alloca+get_member+store+load sequence.  At the
// call site, the struct operand is decomposed into its fields using
// cir.extract_member.

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
/// is \p fc.  Handles Direct (with or without coercion), Extend, Ignore,
/// Indirect (byval and byref), and Expand (struct flattening) arguments.
/// The sret return pointer, when present, is prepended by
/// rewriteFunctionDefinition rather than here.
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
      // coercion at the entry block.  Direct without a coerced type is a
      // true pass-through.
      newArgTypes.push_back(ac.coercedType ? ac.coercedType : origTy);
      break;
    case ArgKind::Ignore:
      break;
    case ArgKind::Expand: {
      // Flatten the struct into one wire argument per field.  The
      // reassembly in the callee body and the decomposition at the call
      // site are handled by insertArgCoercion and rewriteCallSite.
      auto recTy = cast<cir::RecordType>(origTy);
      assert(recTy.isStruct() &&
             "Expand classification requires a struct type, not a union");
      assert(!recTy.getMembers().empty() &&
             "Expand classification requires at least one struct field");
      for (Type memberTy : recTy.getMembers())
        newArgTypes.push_back(memberTy);
      break;
    }
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
      // byval and byref both use a pointer wire type.  The attribute
      // distinction (llvm.byval vs llvm.byref) is applied in updateArgAttrs;
      // the call-site rewrite guards against byref separately because passing
      // a byref pointer from a CIR value requires the original alloca address,
      // which the rewriter does not yet track.
      newArgTypes.push_back(cir::PointerType::get(origTy));
      break;
    }
  }
  return success();
}

/// Compute the new return type for a function whose return classification
/// is \p retInfo.  Direct returns keep (or coerce to) their type, Ignore and
/// Indirect (sret) returns become void, Extend keeps its type; Expand emits
/// an error.
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
    // sret: the value is returned through a hidden pointer argument that
    // rewriteFunctionDefinition prepends to the argument list, so the wire
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
Value createIgnoredValue(OpBuilder &builder, Location loc, Type ty) {
  return cir::ConstantOp::create(builder, loc, ty, cir::PoisonAttr::get(ty));
}

/// Build an updated arg_attrs ArrayAttr that drops Ignore'd args, adds
/// llvm.signext / llvm.zeroext on Extend args, and adds llvm.byval /
/// llvm.align on Indirect args.  Preserves any existing arg attributes on
/// retained arg slots.  \p origArgTypes provides the pre-rewrite type for
/// each arg slot (needed to compute the llvm.byval pointee type).
ArrayAttr updateArgAttrs(MLIRContext *ctx, ArrayRef<Type> origArgTypes,
                         ArrayAttr existingArgAttrs,
                         const FunctionClassification &fc) {
  SmallVector<Attribute> newArgAttrs;
  newArgAttrs.reserve(fc.argInfos.size());
  for (auto [oldIdx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    DictionaryAttr existing = DictionaryAttr::get(ctx);
    if (existingArgAttrs && oldIdx < existingArgAttrs.size())
      existing = cast<DictionaryAttr>(existingArgAttrs[oldIdx]);
    if (ac.kind == ArgKind::Expand) {
      // Push one empty attribute dict per expanded field; the flattened
      // scalar arguments carry no special ABI attributes.
      auto recTy = cast<cir::RecordType>(origArgTypes[oldIdx]);
      for (unsigned i = 0; i < recTy.getNumElements(); ++i)
        newArgAttrs.push_back(DictionaryAttr::get(ctx));
    } else if (ac.kind == ArgKind::Extend) {
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
    } else if (ac.kind == ArgKind::Indirect) {
      // byval: caller-allocated copy; callee receives pointer to copy.
      // byref: callee receives pointer to the caller's original storage.
      // Both use llvm.align(A).  The ownership flag differs: llvm.byval(T)
      // vs llvm.byref(T).  The pointee type T is the pre-rewrite arg type.
      //
      // For byval, two additional attributes match classic CodeGen:
      //   llvm.noundef — the copy is always fully defined (the caller's
      //     original must be defined or UB has already occurred, and the
      //     copy inherits that property).
      //   llvm.noalias — the copy is a fresh caller-allocated alloca that
      //     no other pointer in the function can alias.  Classic CodeGen
      //     emits this when -fpass-by-value-is-noalias is set; here we
      //     emit it unconditionally because our call-site rewrite always
      //     produces a fresh alloca+store.
      Type pointeeTy = origArgTypes[oldIdx];
      StringRef ownershipAttr = ac.byVal ? "llvm.byval" : "llvm.byref";
      SmallVector<NamedAttribute> attrs(existing.begin(), existing.end());
      attrs.push_back(
          NamedAttribute(StringAttr::get(ctx, "llvm.align"),
                         IntegerAttr::get(IntegerType::get(ctx, 64),
                                          ac.indirectAlign.value())));
      attrs.push_back(NamedAttribute(StringAttr::get(ctx, ownershipAttr),
                                     TypeAttr::get(pointeeTy)));
      if (ac.byVal) {
        attrs.push_back(NamedAttribute(StringAttr::get(ctx, "llvm.noalias"),
                                       UnitAttr::get(ctx)));
        attrs.push_back(NamedAttribute(StringAttr::get(ctx, "llvm.noundef"),
                                       UnitAttr::get(ctx)));
      }
      newArgAttrs.push_back(DictionaryAttr::get(ctx, attrs));
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
Value emitCoercion(OpBuilder &rewriter, Location loc, Type dstTy, Value src,
                   FunctionOpInterface funcOp, const DataLayout &dl,
                   SmallPtrSetImpl<Operation *> &createdOps) {
  Type srcTy = src.getType();
  assert(srcTy != dstTy &&
         "emitCoercion callers must pre-check that the types differ");

  uint64_t srcAlign = dl.getTypeABIAlignment(srcTy);
  uint64_t dstAlign = dl.getTypeABIAlignment(dstTy);
  uint64_t allocaAlign = std::max(srcAlign, dstAlign);
  Type slotTy = dl.getTypeSize(srcTy) >= dl.getTypeSize(dstTy) ? srcTy : dstTy;

  auto slotPtrTy = cir::PointerType::get(slotTy);
  auto srcPtrTy = cir::PointerType::get(srcTy);
  auto dstPtrTy = cir::PointerType::get(dstTy);

  cir::AllocaOp alloca;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block &entry = funcOp->getRegion(0).front();
    rewriter.setInsertionPointToStart(&entry);
    alloca = cir::AllocaOp::create(rewriter, loc, slotPtrTy, slotTy,
                                   rewriter.getStringAttr("coerce"),
                                   rewriter.getI64IntegerAttr(allocaAlign));
  }
  createdOps.insert(alloca);

  // Store through a source-typed view of the slot.
  Value srcSlot = alloca;
  if (slotTy != srcTy) {
    auto srcCast = cir::CastOp::create(rewriter, loc, srcPtrTy,
                                       cir::CastKind::bitcast, alloca);
    createdOps.insert(srcCast);
    srcSlot = srcCast;
  }
  auto store = cir::StoreOp::create(rewriter, loc, src, srcSlot);
  createdOps.insert(store);

  // Load through a destination-typed view of the slot.
  Value dstSlot = alloca;
  if (slotTy != dstTy) {
    auto dstCast = cir::CastOp::create(rewriter, loc, dstPtrTy,
                                       cir::CastKind::bitcast, alloca);
    createdOps.insert(dstCast);
    dstSlot = dstCast;
  }
  auto load = cir::LoadOp::create(rewriter, loc, dstSlot);
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
/// back to the original type for body uses.  For each Indirect (byval/byref)
/// arg, change the block argument's type to a pointer and insert a load at
/// entry so the body sees the original value type.  For each Expand arg,
/// replace the single struct block argument with N scalar block arguments (one
/// per field) and insert an alloca+get_member+store+load sequence at entry to
/// reassemble the struct for body uses.
///
/// \p sretOffset is 1 when the function has an sret return (a hidden return
/// pointer is prepended as block argument 0).  Expand arguments expand the
/// block argument count, so a running index tracks the current block argument
/// position rather than computing \p classIdx + \p sretOffset directly.
void insertArgCoercion(FunctionOpInterface funcOp,
                       const FunctionClassification &fc, OpBuilder &rewriter,
                       const DataLayout &dl, unsigned sretOffset) {
  Region &body = funcOp->getRegion(0);
  if (body.empty())
    return;
  Block &entry = body.front();

  // Running block argument index.  Each non-Expand classification occupies
  // one block argument slot; each Expand classification occupies N slots
  // (one per struct field), so the running index must be incremented by N
  // rather than 1 after processing an Expand arg.
  unsigned blockArgIdx = sretOffset;

  for (const ArgClassification &ac : fc.argInfos) {
    if (blockArgIdx >= entry.getNumArguments())
      break;

    if (ac.kind == ArgKind::Expand) {
      // The block arg at blockArgIdx currently has the original struct type.
      // Replace it with N scalar args (one per field) and insert an entry
      // sequence that reassembles them back into the struct.
      BlockArgument origArg = entry.getArgument(blockArgIdx);
      auto recTy = cast<cir::RecordType>(origArg.getType());
      assert(recTy.isStruct() &&
             "Expand classification requires a struct type, not a union");
      unsigned numFields = recTy.getNumElements();
      assert(numFields > 0 &&
             "Expand classification requires at least one struct field");
      Location loc = funcOp.getLoc();

      // Change slot 0 to field 0's type; insert slots 1..N-1 after it.
      origArg.setType(recTy.getElementType(0));
      for (unsigned f = 1; f < numFields; ++f)
        entry.insertArgument(blockArgIdx + f, recTy.getElementType(f), loc);

      // setInsertionPointToStart places each Expand arg's prolog at the
      // top of the entry block.  When multiple Expand args are present
      // the second call pushes its prolog before the first one, inverting
      // the emission order relative to the classification order.  The SSA
      // subgraphs are fully independent — each alloca is written through
      // its own field block args — so the inverted ordering is safe.
      rewriter.setInsertionPointToStart(&entry);
      auto ptrTy = cir::PointerType::get(recTy);
      uint64_t align = dl.getTypeABIAlignment(recTy);
      auto slot = cir::AllocaOp::create(rewriter, loc, ptrTy, recTy,
                                        rewriter.getStringAttr("expand"),
                                        rewriter.getI64IntegerAttr(align));
      SmallPtrSet<Operation *, 8> expandOps = {slot};
      for (unsigned f = 0; f < numFields; ++f) {
        Type fieldPtrTy = cir::PointerType::get(recTy.getElementType(f));
        auto fieldPtr = cir::GetMemberOp::create(rewriter, loc, fieldPtrTy,
                                                 slot, /*name=*/"",
                                                 /*index=*/f);
        expandOps.insert(fieldPtr);
        auto storeOp = cir::StoreOp::create(
            rewriter, loc, entry.getArgument(blockArgIdx + f), fieldPtr);
        expandOps.insert(storeOp);
      }
      auto loaded = cir::LoadOp::create(rewriter, loc, recTy, slot.getResult());
      expandOps.insert(loaded);

      // Replace all original body uses of the struct block arg with the
      // reassembled value.  The store for field 0 uses origArg and is
      // in expandOps, so it is preserved.
      origArg.replaceAllUsesExcept(loaded, expandOps);

      blockArgIdx += numFields;
      continue;
    }

    BlockArgument blockArg = entry.getArgument(blockArgIdx);

    if (ac.kind == ArgKind::Direct && ac.coercedType) {
      Type oldArgTy = blockArg.getType();
      Type newArgTy = ac.coercedType;
      if (oldArgTy == newArgTy) {
        blockArgIdx++;
        continue;
      }
      blockArg.setType(newArgTy);

      rewriter.setInsertionPointToStart(&entry);
      SmallPtrSet<Operation *, 4> coercionOps;
      Value adapted = emitCoercion(rewriter, funcOp.getLoc(), oldArgTy,
                                   blockArg, funcOp, dl, coercionOps);

      // Replace blockArg uses with the adapted value, except inside the
      // helper ops we just created.  This is critical: the StoreOp's
      // value operand is blockArg, and if we naively replaceAllUses it
      // gets swapped to adapted (now of the original type != the alloca's
      // pointee type).
      blockArg.replaceAllUsesExcept(adapted, coercionOps);
    } else if (ac.kind == ArgKind::Indirect) {
      // byval and byref: the wire type is !cir.ptr<T>.  Change the block arg
      // to the pointer type and insert a load so the body sees the original T.
      // The body transformation is the same for both; the distinction between
      // byval (llvm.byval) and byref (llvm.byref) is in the arg attributes
      // applied by updateArgAttrs.
      Type origTy = blockArg.getType();
      auto ptrTy = cir::PointerType::get(origTy);
      blockArg.setType(ptrTy);

      rewriter.setInsertionPointToStart(&entry);
      auto loadOp =
          cir::LoadOp::create(rewriter, funcOp.getLoc(), origTy, blockArg);
      SmallPtrSet<Operation *, 1> loadOps = {loadOp};
      blockArg.replaceAllUsesExcept(loadOp.getResult(), loadOps);
    }
    // Ignore, Extend, and Direct-without-coerce need no block-level changes.

    blockArgIdx++;
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
void insertSRetStores(FunctionOpInterface funcOp, Type origRetTy,
                      OpBuilder &rewriter) {
  Value sretPtr = funcOp.getArguments()[0];

  SmallVector<cir::ReturnOp> returnOps;
  funcOp->walk([&](cir::ReturnOp retOp) { returnOps.push_back(retOp); });

  cir::AllocaOp retAlloca = nullptr;
  for (cir::ReturnOp retOp : returnOps) {
    if (retOp.getInput().empty())
      continue;

    cir::LoadOp retLoad =
        cast<cir::LoadOp>(retOp.getInput()[0].getDefiningOp());

    // Rewire the shared `__retval` alloca to the sret pointer once; all
    // other returns' loads point at the same alloca and are updated by the
    // replaceAllUsesWith below.
    if (!retAlloca) {
      retAlloca = cast<cir::AllocaOp>(retLoad.getAddr().getDefiningOp());
      retAlloca.getResult().replaceAllUsesWith(sretPtr);
      retAlloca->erase();
    }

    rewriter.setInsertionPoint(retOp);
    cir::ReturnOp::create(rewriter, retOp.getLoc());
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
SmallVector<NamedAttribute> buildSretSlotAttrs(OpBuilder &rewriter, Type retTy,
                                               uint64_t align,
                                               bool withNoalias) {
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("llvm.sret", TypeAttr::get(retTy)));
  attrs.push_back(
      rewriter.getNamedAttr("llvm.align", rewriter.getI64IntegerAttr(align)));
  if (withNoalias)
    attrs.push_back(
        rewriter.getNamedAttr("llvm.noalias", rewriter.getUnitAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("llvm.writable", rewriter.getUnitAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("llvm.dead_on_unwind", rewriter.getUnitAttr()));
  return attrs;
}

/// Prepend the sret slot's attrs at position 0 of newCall's arg_attrs.
/// Called after the call has been rewritten with the sret pointer at
/// operand 0, so the operand count now includes the sret slot.  \p argAttrs
/// must already be shaped for the rewritten argument list (Extend slots
/// carry signext/zeroext, Ignore slots dropped); it is shifted to slots
/// 1..N behind the sret slot.
void applySretSlotAttrs(cir::CallOp newCall, ArrayAttr argAttrs, Type retTy,
                        uint64_t align, OpBuilder &rewriter) {
  MLIRContext *ctx = newCall->getContext();
  SmallVector<NamedAttribute> sretAttrs =
      buildSretSlotAttrs(rewriter, retTy, align, /*withNoalias=*/false);

  SmallVector<Attribute> newArgAttrs;
  newArgAttrs.reserve(newCall.getArgOperands().size());
  newArgAttrs.push_back(DictionaryAttr::get(ctx, sretAttrs));
  if (argAttrs)
    for (Attribute a : argAttrs)
      newArgAttrs.push_back(a);
  while (newArgAttrs.size() < newCall.getArgOperands().size())
    newArgAttrs.push_back(DictionaryAttr::get(ctx));
  newCall->setAttr("arg_attrs", ArrayAttr::get(ctx, newArgAttrs));
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

  // sret return: the value is returned through a hidden pointer prepended as
  // argument 0.  The wire return type was already set to void by
  // computeNewReturnType.  Every classification index therefore maps to a
  // block argument shifted by this offset in the body handling below.
  bool hasSRet =
      fc.returnInfo.kind == ArgKind::Indirect && !oldResultTypes.empty();
  if (hasSRet)
    newArgTypes.insert(newArgTypes.begin(), cir::PointerType::get(origRetTy));
  unsigned sretOffset = hasSRet ? 1 : 0;

  if (funcOp.isDefinition()) {
    Region &body = funcOp->getRegion(0);
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
      insertArgCoercion(funcOp, fc, builder, dl, sretOffset);

      // Direct return with coerced type: insert a coercion at every
      // cir.return so the returned value matches the (coerced) return
      // type in the new function signature set below.
      if (fc.returnInfo.kind == ArgKind::Direct && fc.returnInfo.coercedType &&
          !oldResultTypes.empty() && fc.returnInfo.coercedType != origRetTy)
        insertReturnCoercion(funcOp, origRetTy, fc.returnInfo.coercedType,
                             builder, dl);

      Block &entry = body.front();

      // Build a mapping from classification index to the block argument
      // start index it occupies after insertArgCoercion has run.  Each
      // non-Expand classification occupies one block argument slot; each
      // Expand classification occupies N slots (its number of fields).
      // This mapping is needed because the Ignore-drop loop below erases
      // by block argument index, not by classification index.
      SmallVector<unsigned> classToBlockArg(fc.argInfos.size());
      {
        unsigned runningIdx = sretOffset;
        for (unsigned i = 0; i < fc.argInfos.size(); ++i) {
          classToBlockArg[i] = runningIdx;
          if (fc.argInfos[i].kind == ArgKind::Expand) {
            auto recTy = cast<cir::RecordType>(oldArgTypes[i]);
            runningIdx += recTy.getNumElements();
          } else {
            runningIdx += 1;
          }
        }
      }

      // For each Ignored argument: drop the block argument and, if the
      // body still references it, replace those uses with a poison
      // constant.  Ignore classifications mean the value is empty / not
      // passed at the ABI level, so any remaining uses are vacuous;
      // poison says exactly that.  Iterate in reverse classification
      // order so that erasing a later block argument does not shift the
      // block argument indices for earlier classifications.
      for (int classIdx = static_cast<int>(fc.argInfos.size()) - 1;
           classIdx >= 0; --classIdx) {
        if (fc.argInfos[classIdx].kind != ArgKind::Ignore)
          continue;
        unsigned realIdx = classToBlockArg[classIdx];
        if (realIdx >= entry.getNumArguments())
          continue;
        BlockArgument arg = entry.getArgument(realIdx);
        if (!arg.use_empty()) {
          builder.setInsertionPointToStart(&entry);
          Value poison =
              createIgnoredValue(builder, funcOp.getLoc(), arg.getType());
          arg.replaceAllUsesWith(poison);
        }
        entry.eraseArgument(realIdx);
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

  // Rebuild arg_attrs when the function has an sret slot (slot 0 needs the
  // sret attribute set) or any arg is Ignore (dropped from the output array),
  // Extend (needs llvm.signext / llvm.zeroext), Indirect (needs
  // llvm.byval / llvm.align), or Expand (changes the argument count).
  bool needsArgAttrUpdate =
      hasSRet || llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend ||
               ac.kind == ArgKind::Indirect || ac.kind == ArgKind::Expand;
      });
  if (needsArgAttrUpdate) {
    auto existing = funcOp->getAttrOfType<ArrayAttr>("arg_attrs");
    ArrayAttr updated = updateArgAttrs(ctx, oldArgTypes, existing, fc);
    if (hasSRet) {
      // Prepend the sret slot's attribute dict (slot 0); the per-argument
      // dicts shift to slots 1..N.  noalias is valid only on the callee's
      // parameter, so it is added only for definitions.
      SmallVector<NamedAttribute> sretAttrs = buildSretSlotAttrs(
          builder, origRetTy, fc.returnInfo.indirectAlign.value(),
          /*withNoalias=*/funcOp.isDefinition());
      SmallVector<Attribute> withSret;
      withSret.push_back(DictionaryAttr::get(ctx, sretAttrs));
      for (Attribute a : updated)
        withSret.push_back(a);
      funcOp->setAttr("arg_attrs", ArrayAttr::get(ctx, withSret));
    } else {
      funcOp->setAttr("arg_attrs", updated);
    }
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

  builder.setInsertionPoint(call);

  SmallVector<Value> newArgs;
  ValueRange argOperands = call.getArgOperands();
  newArgs.reserve(argOperands.size());

  // Capture original arg types before building newArgs (byval slots change
  // the wire argument from T to !cir.ptr<T>, so we save the pre-rewrite
  // types here for use in updateArgAttrs).
  SmallVector<Type> origCallArgTypes;
  origCallArgTypes.reserve(argOperands.size());
  for (Value v : argOperands)
    origCallArgTypes.push_back(v.getType());
  if (argOperands.size() > fc.argInfos.size())
    return call.emitOpError()
           << "variadic arguments not yet implemented in CallConvLowering";
  assert(fc.argInfos.size() == argOperands.size() &&
         "call operand count must match classified arg count");
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    Value arg = argOperands[idx];
    if (ac.kind == ArgKind::Expand) {
      // Decompose the struct value into its constituent scalar fields and
      // pass each as a separate argument.  cir.extract_member extracts the
      // field value directly without a memory round-trip.
      auto recTy = cast<cir::RecordType>(arg.getType());
      assert(recTy.isStruct() &&
             "Expand classification requires a struct type, not a union");
      for (unsigned f = 0; f < recTy.getNumElements(); ++f) {
        Value field =
            cir::ExtractMemberOp::create(builder, call.getLoc(), arg, f);
        newArgs.push_back(field);
      }
    } else if (ac.kind == ArgKind::Direct && ac.coercedType &&
               arg.getType() != ac.coercedType) {
      arg = emitCoercion(builder, call.getLoc(), ac.coercedType, arg,
                         enclosingFunc, dl);
      newArgs.push_back(arg);
    } else if (ac.kind == ArgKind::Indirect) {
      // byval and byref: allocate a stack slot, copy the value in, and pass
      // the pointer.  The alloca+store pattern is identical for both; the
      // attribute distinction (llvm.byval vs llvm.byref) is applied by
      // updateArgAttrs.  byref does not receive llvm.noalias or llvm.noundef
      // because it does not assert exclusive ownership of the storage.
      Type argTy = arg.getType();
      auto ptrTy = cir::PointerType::get(argTy);
      uint64_t align = ac.indirectAlign.value();
      StringRef slotName = ac.byVal ? "byval" : "byref";
      auto slot = cir::AllocaOp::create(builder, call.getLoc(), ptrTy, argTy,
                                        builder.getStringAttr(slotName),
                                        builder.getI64IntegerAttr(align));
      cir::StoreOp::create(builder, call.getLoc(), arg, slot);
      arg = slot;
      newArgs.push_back(arg);
    } else {
      newArgs.push_back(arg);
    }
  }

  bool hasResult = call.getNumResults() > 0;
  Type origRetTy =
      hasResult ? call.getResult().getType() : cir::VoidType::get(ctx);

  // sret return: prepend a return-slot pointer to the call, make the call
  // return void, and load the result back out of the slot.  Handled as an
  // early return because the coerce / extend / ignore return handling below
  // does not apply to an indirect return.
  if (fc.returnInfo.kind == ArgKind::Indirect && hasResult) {
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
    Value sretSlot = nullptr;
    cir::StoreOp reuseStore = nullptr;
    if (call.getResult().hasOneUse()) {
      Operation *user = *call.getResult().getUsers().begin();
      if (auto store = dyn_cast<cir::StoreOp>(user))
        if (store.getValue() == call.getResult() &&
            store.getAddr().getType() == ptrTy &&
            mlir::DominanceInfo().properlyDominates(store.getAddr(), call)) {
          sretSlot = store.getAddr();
          reuseStore = store;
        }
    }
    if (!sretSlot) {
      auto alloca = cir::AllocaOp::create(
          builder, call.getLoc(), ptrTy, origRetTy,
          /*name=*/builder.getStringAttr("sret"),
          /*alignment=*/builder.getI64IntegerAttr(sretAlign));
      sretSlot = alloca;
    }

    SmallVector<Value> sretArgs;
    sretArgs.push_back(sretSlot);
    sretArgs.append(newArgs.begin(), newArgs.end());

    Type sretVoidTy = cir::VoidType::get(ctx);
    auto newCall = cir::CallOp::create(
        builder, call.getLoc(), call.getCalleeAttr(), sretVoidTy, sretArgs);
    for (NamedAttribute attr : call->getAttrs())
      if (!newCall->hasAttr(attr.getName()))
        newCall->setAttr(attr.getName(), attr.getValue());

    // Shape the per-argument attrs exactly as the non-sret path does
    // (signext / zeroext for Extend, drop Ignore slots, byval / align for
    // Indirect, flatten for Expand) before prepending the sret slot.
    ArrayAttr argAttrs = call->getAttrOfType<ArrayAttr>("arg_attrs");
    bool needsArgAttrUpdate =
        llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
          return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend ||
                 ac.kind == ArgKind::Indirect || ac.kind == ArgKind::Expand;
        });
    if (needsArgAttrUpdate)
      argAttrs = updateArgAttrs(ctx, origCallArgTypes, argAttrs, fc);
    applySretSlotAttrs(newCall, argAttrs, origRetTy, sretAlign, builder);

    if (reuseStore) {
      // The callee now constructs directly into the destination slot, so the
      // original store-from-result is redundant; dropping it avoids a
      // byte-copy of the record.
      reuseStore->erase();
    } else {
      builder.setInsertionPointAfter(newCall);
      auto load =
          cir::LoadOp::create(builder, call.getLoc(), origRetTy, sretSlot,
                              /*isDeref=*/mlir::UnitAttr(),
                              /*isVolatile=*/mlir::UnitAttr(),
                              /*alignment=*/mlir::IntegerAttr(),
                              /*sync_scope=*/cir::SyncScopeKindAttr(),
                              /*mem_order=*/cir::MemOrderAttr());
      call.getResult().replaceAllUsesWith(load);
    }
    call->erase();
    return success();
  }

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
  // res_attrs for Extend args/return.  Ignore args require a rebuild because
  // their slots are dropped; Indirect args need llvm.byval / llvm.align;
  // Expand args change the argument count.
  bool needsArgAttrUpdate =
      llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend ||
               ac.kind == ArgKind::Indirect || ac.kind == ArgKind::Expand;
      });
  if (needsArgAttrUpdate) {
    auto existing = call->getAttrOfType<ArrayAttr>("arg_attrs");
    newCall->setAttr("arg_attrs",
                     updateArgAttrs(ctx, origCallArgTypes, existing, fc));
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
