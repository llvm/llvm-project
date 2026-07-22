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
// through an alloca+store at the call site.  At the callee, byval loads
// the incoming pointer (a local copy), while byref rewires the CIRGen
// param-slot alloca to the incoming pointer so the body mutates the
// caller's storage in place.
//
// For Expand, the single struct argument is replaced by N scalar arguments
// (one per field).  At the callee, the N field block arguments are stored
// directly into the parameter's own alloca (the CIRGen spill slot).  At the
// call site, the struct operand is decomposed into its fields by reading
// each member from the source alloca (get_member + load) when the operand is
// a load of an alloca, or via cir.extract_member otherwise.
//
// For Direct + canFlatten (where the coerced type is a multi-field struct),
// the coerced struct is similarly flattened into N individual wire arguments.
// The callee reassembles the N scalar block args into the coerced struct,
// then coerces to the original argument type if the two types differ.  The
// call site coerces the original type to the coerced struct, then extracts
// each field as a separate call argument.

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

/// Return the coerced RecordType for a Direct classification that should be
/// flattened into individual scalar arguments, or a null type if the
/// classification does not call for flattening.
///
/// Flattening applies when all four conditions hold:
///   1. The classification is Direct with a non-null coercedType.
///   2. canFlatten is set.
///   3. The coercedType is a struct (not a union).
///   4. The struct has more than one field (single-field structs are already
///      scalar; flattening them produces no benefit and classic CodeGen skips
///      them for the same reason).
cir::RecordType getFlattenedCoercedType(const ArgClassification &ac) {
  if (ac.kind != ArgKind::Direct || !ac.coercedType || !ac.canFlatten)
    return {};
  auto recTy = dyn_cast<cir::RecordType>(ac.coercedType);
  if (!recTy || !recTy.isStruct() || recTy.getNumElements() <= 1)
    return {};
  return recTy;
}

/// Build the new argument-type list for a function whose ABI classification
/// is \p fc.  Handles Direct (with or without coercion), Extend, Ignore,
/// Indirect (byval and byref), and Expand (struct flattening) arguments.
/// The sret return pointer, when present, is prepended by
/// rewriteFunctionDefinition rather than here.
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
      // Direct with canFlatten and a struct coerced type: push one wire type
      // per field of the coerced struct rather than the struct itself.
      // Single-field coerced structs fall through to the non-flatten path —
      // the struct is already scalar-sized and flattening adds no value.
      if (cir::RecordType flatTy = getFlattenedCoercedType(ac)) {
        llvm::append_range(newArgTypes, flatTy.getMembers());
      } else {
        // Direct with a coerced type: the wire signature uses the coerced
        // type; the body still expects origTy and insertArgCoercion recovers
        // it via a memory round-trip.  Direct without coercion is a
        // pass-through.
        newArgTypes.push_back(ac.coercedType ? ac.coercedType : origTy);
      }
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
      llvm::append_range(newArgTypes, recTy.getMembers());
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

/// Build an updated arg_attrs ArrayAttr that drops Ignore'd args, adds
/// llvm.signext / llvm.zeroext on Extend args, and adds llvm.byval /
/// llvm.align on Indirect args.  Preserves any existing arg attributes on
/// retained arg slots.  \p origArgTypes provides the pre-rewrite type for
/// each arg slot (needed to compute the llvm.byval pointee type).
mlir::ArrayAttr updateArgAttrs(mlir::MLIRContext *ctx,
                               ArrayRef<mlir::Type> origArgTypes,
                               mlir::ArrayAttr existingArgAttrs,
                               const FunctionClassification &fc) {
  mlir::Builder builder(ctx);
  SmallVector<mlir::Attribute> newArgAttrs;
  newArgAttrs.reserve(fc.argInfos.size());
  for (auto [oldIdx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    mlir::DictionaryAttr existing = builder.getDictionaryAttr({});
    if (existingArgAttrs && oldIdx < existingArgAttrs.size())
      existing = mlir::cast<mlir::DictionaryAttr>(existingArgAttrs[oldIdx]);
    if (cir::RecordType flatTy = getFlattenedCoercedType(ac)) {
      // Direct + canFlatten: one empty attribute dict per flattened field; the
      // flattened scalar arguments carry no special ABI attributes.
      newArgAttrs.append(flatTy.getNumElements(),
                         builder.getDictionaryAttr({}));
    } else if (ac.kind == ArgKind::Expand) {
      // Push one empty attribute dict per expanded field; the flattened
      // scalar arguments carry no special ABI attributes.
      auto recTy = cast<cir::RecordType>(origArgTypes[oldIdx]);
      newArgAttrs.append(recTy.getNumElements(), builder.getDictionaryAttr({}));
    } else if (ac.kind == ArgKind::Extend) {
      StringRef attrName = ac.signExtend ? "llvm.signext" : "llvm.zeroext";
      SmallVector<mlir::NamedAttribute> attrs(existing.begin(), existing.end());
      attrs.push_back(builder.getNamedAttr(attrName, builder.getUnitAttr()));
      newArgAttrs.push_back(builder.getDictionaryAttr(attrs));
    } else if (ac.kind == ArgKind::Indirect) {
      // byval: caller-allocated copy; callee receives pointer to copy.
      // byref: callee receives pointer to the caller's original storage.
      // Both use llvm.align(A).  The ownership flag differs: llvm.byval(T)
      // vs llvm.byref(T).  Both are typed attributes carrying the pointee
      // type T (the pre-rewrite arg type); T is recorded explicitly because
      // it cannot be recovered from the opaque LLVM pointer after lowering.
      //
      // For byval, two additional attributes match classic CodeGen:
      //   llvm.noundef -- the copy is always fully defined (the caller's
      //     original must be defined or UB has already occurred, and the
      //     copy inherits that property).
      //   llvm.noalias -- the copy is a fresh caller-allocated alloca that
      //     no other pointer in the function can alias.  Classic CodeGen
      //     emits this when -fpass-by-value-is-noalias is set; here we
      //     emit it unconditionally because our call-site rewrite always
      //     produces a fresh alloca+store.
      mlir::Type pointeeTy = origArgTypes[oldIdx];
      StringRef ownershipAttr = ac.byVal ? "llvm.byval" : "llvm.byref";
      SmallVector<mlir::NamedAttribute> attrs(existing.begin(), existing.end());
      attrs.push_back(builder.getNamedAttr(
          "llvm.align", builder.getI64IntegerAttr(ac.indirectAlign.value())));
      attrs.push_back(
          builder.getNamedAttr(ownershipAttr, mlir::TypeAttr::get(pointeeTy)));
      if (ac.byVal) {
        attrs.push_back(
            builder.getNamedAttr("llvm.noalias", builder.getUnitAttr()));
        attrs.push_back(
            builder.getNamedAttr("llvm.noundef", builder.getUnitAttr()));
      }
      newArgAttrs.push_back(builder.getDictionaryAttr(attrs));
    } else {
      newArgAttrs.push_back(existing);
    }
  }
  return builder.getArrayAttr(newArgAttrs);
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

/// Coerce \p src into a temporary memory slot typed for \p dstTy at the
/// current builder insertion point, and return the destination-typed pointer
/// to that slot without loading the value back out.  This is the shared
/// memory half of emitCoercion: callers that want the whole coerced value use
/// emitCoercion (below); callers that want to read individual members of a
/// coerced struct (the call-site struct flattening) take the returned pointer
/// and emit their own cir.get_member + cir.load per field.  Lowers uniformly
/// for scalar, vector, and record types.
///
/// The slot is sized to the larger of the two types so that neither the store
/// nor a later load ever runs past it: the coerced ABI type can be larger
/// than the original (e.g. a 12-byte aggregate passed as `{i64, i64}`), so
/// accessing the destination through a source-sized slot would over-read.
/// Alignment is max(srcAlign, dstAlign) to satisfy both accesses.  The slot
/// is written through a source-typed view and returned as a destination-typed
/// view.
///
/// The temporary alloca is placed at the start of the enclosing function's
/// entry block so that it composes correctly with the HoistAllocas pass
/// regardless of pipeline ordering.
///
/// Any operations the helper creates are appended to \p createdOps so the
/// caller can pass them to replaceAllUsesExcept and avoid clobbering the
/// store's value operand when later rewiring the source value.
mlir::Value
emitCoercionToMemory(mlir::OpBuilder &builder, mlir::Location loc,
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

  // Return a destination-typed view of the slot.
  if (slotTy != dstTy) {
    auto dstCast = cir::CastOp::create(builder, loc, dstPtrTy,
                                       cir::CastKind::bitcast, alloca);
    createdOps.insert(dstCast);
    return dstCast;
  }
  return alloca;
}

/// Coerce \p src to type \p dstTy by going through memory and load the whole
/// coerced value back out.  Builds on emitCoercionToMemory, adding the final
/// load of the destination-typed view.
mlir::Value emitCoercion(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type dstTy, mlir::Value src,
                         mlir::FunctionOpInterface funcOp,
                         const mlir::DataLayout &dl,
                         SmallPtrSetImpl<mlir::Operation *> &createdOps) {
  mlir::Value dstSlot =
      emitCoercionToMemory(builder, loc, dstTy, src, funcOp, dl, createdOps);
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

/// Decompose a struct value into one scalar call argument per field of \p
/// recTy, appending the field values to \p newArgs.  When \p structVal is a
/// plain (non-volatile, non-atomic) load straight from an alloca, read each
/// field with cir.get_member + cir.load from that alloca, emitted at the
/// original load's position so they observe the same memory state, and record
/// the now-dead whole-struct load in \p replacedWholeLoads for later erasure.
/// Otherwise (a call result, compound literal, or qualified load) extract each
/// field from the value with cir.extract_member.  Loading the members from the
/// alloca rather than extracting from a whole-struct value keeps the result in
/// a form SROA can promote (it does not reason about extractvalue).  Shared by
/// the Expand and Direct+canFlatten argument paths.
static void
emitStructFieldArgs(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value structVal, cir::RecordType recTy,
                    SmallVectorImpl<mlir::Value> &newArgs,
                    SmallVectorImpl<cir::LoadOp> &replacedWholeLoads) {
  cir::LoadOp wholeLoad = structVal.getDefiningOp<cir::LoadOp>();
  cir::AllocaOp srcAlloca;
  if (wholeLoad && !wholeLoad.getIsVolatile() && !wholeLoad.getMemOrder())
    srcAlloca = wholeLoad.getAddr().getDefiningOp<cir::AllocaOp>();

  if (srcAlloca) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(wholeLoad);
    for (auto [f, fieldTy] : llvm::enumerate(recTy.getMembers())) {
      mlir::Type fieldPtrTy = cir::PointerType::get(fieldTy);
      mlir::Value fieldPtr = cir::GetMemberOp::create(
          builder, loc, fieldPtrTy, srcAlloca, /*name=*/"", /*index=*/f);
      newArgs.push_back(cir::LoadOp::create(builder, loc, fieldPtr));
    }
    replacedWholeLoads.push_back(wholeLoad);
  } else {
    for (unsigned f = 0; f < recTy.getNumElements(); ++f)
      newArgs.push_back(
          cir::ExtractMemberOp::create(builder, loc, structVal, f));
  }
}

/// For each Direct arg with a coerced type, change the block argument's type
/// to the coerced type and insert a coercion at function entry that maps it
/// back to the original type for body uses.  For each Indirect byval arg,
/// change the block argument's type to a pointer and insert a load at entry
/// so the body sees a local copy of the original value type.  For each
/// Indirect byref arg, change the block argument to a pointer and rewire the
/// CIRGen param-slot alloca to that pointer (no entry load / byte-copy) so
/// the body operates on the caller's storage in place.  For each Expand arg,
/// replace the single struct block argument with N scalar block arguments (one
/// per field) and store each field directly into the parameter's own alloca
/// (the CIRGen spill slot), erasing the original whole-struct store.
///
/// \p hasSRetArg is true when the function has an sret return (a hidden return
/// pointer is prepended as block argument 0).  Expand arguments expand the
/// block argument count, so a running index tracks the current block argument
/// position rather than computing the classification index + \p hasSRetArg
/// directly.
void insertArgCoercion(mlir::FunctionOpInterface funcOp,
                       const FunctionClassification &fc,
                       mlir::OpBuilder &builder, const mlir::DataLayout &dl,
                       bool hasSRetArg) {
  mlir::Region &body = funcOp->getRegion(0);
  if (body.empty())
    return;
  mlir::Block &entry = body.front();

  // Running block argument index.  Each non-Expand classification occupies
  // one block argument slot; each Expand classification occupies N slots
  // (one per struct field), so the running index must be incremented by N
  // rather than 1 after processing an Expand arg.
  unsigned blockArgIdx = hasSRetArg ? 1 : 0;

  for (const ArgClassification &ac : fc.argInfos) {
    assert(blockArgIdx < entry.getNumArguments() &&
           "classification count must not exceed entry block arguments");

    if (ac.kind == ArgKind::Expand) {
      // The block arg at blockArgIdx currently has the original struct type.
      // Replace it with N scalar args (one per field) and store each field
      // directly into the parameter's own alloca.
      mlir::BlockArgument origArg = entry.getArgument(blockArgIdx);
      auto recTy = cast<cir::RecordType>(origArg.getType());
      assert(recTy.isStruct() &&
             "Expand classification requires a struct type, not a union");
      unsigned numFields = recTy.getNumElements();
      assert(numFields > 0 &&
             "Expand classification requires at least one struct field");
      mlir::Location loc = funcOp.getLoc();

      // CIRGen spills every by-value struct parameter into its local alloca
      // with a single store before any other use, so the struct block arg's
      // only use is that spill.  Capture it and the destination alloca so the
      // expanded fields can be stored straight into that alloca, preserving
      // the alloca's variable name and `init` flag and avoiding a
      // reassemble-then-reload roundtrip.  DCE may have run earlier and
      // removed the spill (leaving the block arg unused); tolerate that by
      // only flattening the signature and emitting no field stores.
      cir::StoreOp paramStore;
      cir::AllocaOp destAlloca;
      if (!origArg.use_empty()) {
        assert(origArg.hasOneUse() &&
               "Expand arg must have exactly one use (the CIRGen param spill)");
        paramStore = cast<cir::StoreOp>(*origArg.user_begin());
        assert(paramStore.getValue() == origArg &&
               "Expand arg's use must be the value operand of its store");
        destAlloca = cast<cir::AllocaOp>(paramStore.getAddr().getDefiningOp());
      }

      // Erase the original whole-struct spill before retyping the block
      // argument, so the store is never left feeding a type-mismatched value.
      // The field stores take its place, just before the following operation
      // (the spill always precedes the entry block's terminator).
      mlir::Operation *fieldStoreInsertPt = nullptr;
      if (paramStore) {
        fieldStoreInsertPt = paramStore->getNextNode();
        assert(fieldStoreInsertPt &&
               "param spill must be followed by a block terminator");
        paramStore->erase();
      }

      // Split the single struct block arg into N scalar field block args (slot
      // 0 reuses the original; slots 1..N-1 are inserted after it).  The
      // reshape needs no insertion point.  The field stores are gated on the
      // same destAlloca condition: when the spill survived we set the insert
      // point to its old slot (which sits after the CIRGen allocas) and store
      // each field there; when DCE removed the spill the parameter is dead, so
      // we only reshape the signature and emit no stores.
      if (destAlloca)
        builder.setInsertionPoint(fieldStoreInsertPt);
      for (auto [f, fieldTy] : llvm::enumerate(recTy.getMembers())) {
        if (f == 0)
          origArg.setType(fieldTy);
        else
          entry.insertArgument(blockArgIdx + f, fieldTy, loc);
        if (!destAlloca)
          continue;
        mlir::Type fieldPtrTy = cir::PointerType::get(fieldTy);
        auto fieldPtr = cir::GetMemberOp::create(builder, loc, fieldPtrTy,
                                                 destAlloca, /*name=*/"",
                                                 /*index=*/f);
        cir::StoreOp::create(builder, loc, entry.getArgument(blockArgIdx + f),
                             fieldPtr);
      }

      blockArgIdx += numFields;
      continue;
    }

    mlir::BlockArgument blockArg = entry.getArgument(blockArgIdx);

    if (cir::RecordType flatTy = getFlattenedCoercedType(ac)) {
      // Direct + canFlatten: the coerced type is a struct whose fields become
      // individual wire arguments.  The reconstruction mirrors the Expand path
      // — replace the single block arg with N scalar block args, store them
      // into an alloca of the coerced struct type, reload — but then applies
      // an additional coercion from the coerced struct type to the original
      // argument type if the two differ in layout.
      unsigned numFields = flatTy.getNumElements();
      assert(numFields >= 2 && "getFlattenedCoercedType guarantees >1 fields");
      Type origTy = blockArg.getType();
      Location loc = funcOp.getLoc();

      // Change slot 0 to field 0's type; insert slots 1..N-1 after it.
      blockArg.setType(flatTy.getElementType(0));
      for (unsigned f = 1; f < numFields; ++f)
        entry.insertArgument(blockArgIdx + f, flatTy.getElementType(f), loc);

      // setInsertionPointToStart: see comment in the Expand arm above.
      builder.setInsertionPointToStart(&entry);
      auto flatPtrTy = cir::PointerType::get(flatTy);
      uint64_t flatAlign = dl.getTypeABIAlignment(flatTy);
      auto flatSlot = cir::AllocaOp::create(
          builder, loc, flatPtrTy, builder.getStringAttr("coerce"),
          builder.getI64IntegerAttr(flatAlign));
      SmallPtrSet<Operation *, 8> flattenOps = {flatSlot};
      for (auto [f, fieldTy] : llvm::enumerate(flatTy.getMembers())) {
        Type fieldPtrTy = cir::PointerType::get(fieldTy);
        auto fieldPtr = cir::GetMemberOp::create(builder, loc, fieldPtrTy,
                                                 flatSlot, /*name=*/"",
                                                 /*index=*/f);
        flattenOps.insert(fieldPtr);
        auto storeOp = cir::StoreOp::create(
            builder, loc, entry.getArgument(blockArgIdx + f), fieldPtr);
        flattenOps.insert(storeOp);
      }
      auto flatLoaded =
          cir::LoadOp::create(builder, loc, flatTy, flatSlot.getResult());
      flattenOps.insert(flatLoaded);

      // If the coerced struct type differs from the original argument type,
      // insert a memory round-trip to recover the original type for body uses.
      Value finalVal = flatLoaded;
      if (origTy != flatTy) {
        SmallPtrSet<Operation *, 4> coercionOps;
        finalVal = emitCoercion(builder, loc, origTy, flatLoaded, funcOp, dl,
                                coercionOps);
        flattenOps.insert(coercionOps.begin(), coercionOps.end());
      }

      // Replace all original body uses of the struct block arg (now field 0)
      // with the recovered original-type value.
      blockArg.replaceAllUsesExcept(finalVal, flattenOps);

      blockArgIdx += numFields;
      continue;
    }

    if (ac.kind == ArgKind::Direct && ac.coercedType) {
      mlir::Type oldArgTy = blockArg.getType();
      mlir::Type newArgTy = ac.coercedType;
      if (oldArgTy == newArgTy) {
        ++blockArgIdx;
        continue;
      }
      blockArg.setType(newArgTy);

      builder.setInsertionPointToStart(&entry);
      SmallPtrSet<mlir::Operation *, 4> coercionOps;
      mlir::Value adapted = emitCoercion(builder, funcOp.getLoc(), oldArgTy,
                                         blockArg, funcOp, dl, coercionOps);

      // Replace blockArg uses with the adapted value, except inside the
      // helper ops we just created.  This is critical: the StoreOp's value
      // operand is blockArg, and if we naively replaceAllUses it gets swapped
      // to adapted (now of the original type != the alloca's pointee type).
      blockArg.replaceAllUsesExcept(adapted, coercionOps);
    } else if (ac.kind == ArgKind::Indirect) {
      // byval and byref share a !cir.ptr<T> wire type; the llvm.byval vs
      // llvm.byref distinction is in the attrs applied by updateArgAttrs.
      // Body lowering differs: byval copies into the callee (load at entry),
      // while byref must operate on the caller's storage in place.
      auto ptrTy = cir::PointerType::get(blockArg.getType());

      if (!ac.byVal) {
        // byref: CIRGen spills every by-value parameter into a local alloca
        // with a single store before any other use, and CallConvLowering runs
        // on that CIRGen output before any alloca-promoting/splitting pass, so
        // the block argument still has exactly that one use here.  Rewire the
        // alloca to the incoming pointer and drop the store so the body
        // operates on the caller's storage in place.  A byte-copy would be
        // wrong for non-trivially-copyable aggregates (e.g. libstdc++ SSO
        // std::string, where it would leave `_M_p` aliasing the source's
        // `_M_local_buf`).  DCE may have removed a dead spill; tolerate that by
        // only retyping the block argument.
        cir::StoreOp paramStore;
        cir::AllocaOp destAlloca;
        if (!blockArg.use_empty()) {
          assert(blockArg.hasOneUse() &&
                 "byref arg must have exactly one use (the CIRGen param "
                 "spill)");
          paramStore = cast<cir::StoreOp>(*blockArg.user_begin());
          assert(paramStore.getValue() == blockArg &&
                 "byref arg's use must be the value operand of its store");
          destAlloca =
              cast<cir::AllocaOp>(paramStore.getAddr().getDefiningOp());
        }

        if (paramStore)
          paramStore->erase();

        // Update the block argument to point to its original type.
        blockArg.setType(ptrTy);

        if (destAlloca) {
          destAlloca.getResult().replaceAllUsesWith(blockArg);
          destAlloca->erase();
        }
      } else {
        // byval: load the incoming pointer so the body sees a T value (and
        // any CIRGen param-slot store becomes a local copy of that value).
        blockArg.setType(ptrTy);

        builder.setInsertionPointToStart(&entry);
        auto loadOp = cir::LoadOp::create(builder, funcOp.getLoc(), blockArg);
        SmallPtrSet<mlir::Operation *, 1> loadOps = {loadOp};
        blockArg.replaceAllUsesExcept(loadOp.getResult(), loadOps);
      }
    }
    // Ignore, Extend, and Direct-without-coerce need no block-level changes.

    ++blockArgIdx;
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
                               mlir::Type origRetTy,
                               ArrayRef<mlir::Type> origCallArgTypes,
                               mlir::OpBuilder &builder) {
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
  // (signext / zeroext for Extend, drop Ignore slots, byval / align for
  // Indirect, flatten for Expand and Direct+canFlatten) before prepending the
  // sret slot, so sret composes correctly with Extend / Ignore / Indirect /
  // Expand / Direct+canFlatten args.
  mlir::ArrayAttr argAttrs = call->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
  bool needsArgAttrUpdate =
      llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend ||
               ac.kind == ArgKind::Indirect || ac.kind == ArgKind::Expand ||
               getFlattenedCoercedType(ac);
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
    auto load = cir::LoadOp::create(builder, call.getLoc(), origRetTy, sretSlot,
                                    /*isDeref=*/mlir::UnitAttr(),
                                    /*isVolatile=*/mlir::UnitAttr(),
                                    /*is_nontemporal=*/mlir::UnitAttr(),
                                    /*alignment=*/mlir::IntegerAttr(),
                                    /*sync_scope=*/cir::SyncScopeKindAttr(),
                                    /*mem_order=*/cir::MemOrderAttr(),
                                    /*invariant=*/mlir::UnitAttr());
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

      // Drop each Ignored argument's block argument, replacing any remaining
      // body uses with a poison constant (an Ignore arg is not passed at the
      // ABI level, so any use is vacuous; poison says exactly that).  Walk
      // forward with a running block-argument index that mirrors
      // insertArgCoercion: an Expand arg or a Direct+canFlatten arg occupies N
      // slots, every other kept kind one.  On erase, do not advance the index
      // -- the next block argument shifts into the vacated slot.
      unsigned blockArgIdx = hasSRet ? 1 : 0;
      for (auto [i, ac] : llvm::enumerate(fc.argInfos)) {
        if (blockArgIdx >= entry.getNumArguments())
          break;
        if (ac.kind == ArgKind::Ignore) {
          mlir::BlockArgument arg = entry.getArgument(blockArgIdx);
          if (!arg.use_empty()) {
            builder.setInsertionPointToStart(&entry);
            mlir::Value poison =
                createIgnoredValue(builder, funcOp.getLoc(), arg.getType());
            arg.replaceAllUsesWith(poison);
          }
          entry.eraseArgument(blockArgIdx);
          continue;
        }
        if (cir::RecordType flatTy = getFlattenedCoercedType(ac))
          blockArgIdx += flatTy.getNumElements();
        else if (ac.kind == ArgKind::Expand)
          blockArgIdx += cast<cir::RecordType>(oldArgTypes[i]).getNumElements();
        else
          ++blockArgIdx;
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
  // sret attribute set) or any arg is Ignore (dropped from the output array),
  // Extend (needs llvm.signext / llvm.zeroext), Indirect (needs
  // llvm.byval / llvm.align), Expand or Direct+canFlatten (both change the
  // argument count).
  bool needsArgAttrUpdate =
      hasSRet || llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend ||
               ac.kind == ArgKind::Indirect || ac.kind == ArgKind::Expand ||
               getFlattenedCoercedType(ac);
      });
  if (needsArgAttrUpdate) {
    auto existing = funcOp->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
    mlir::ArrayAttr updated = updateArgAttrs(ctx, oldArgTypes, existing, fc);
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

  builder.setInsertionPoint(call);

  SmallVector<mlir::Value> newArgs;
  mlir::ValueRange argOperands = call.getArgOperands();
  newArgs.reserve(argOperands.size());

  // Whole-struct loads replaced by direct member loads for Expand operands.
  // They can only be erased once the original call (their remaining user) is
  // gone, so collect them and erase the dead ones at the end.
  SmallVector<cir::LoadOp> replacedWholeLoads;

  // Capture original arg types before building newArgs (byval slots change
  // the wire argument from T to !cir.ptr<T>, so we save the pre-rewrite
  // types here for use in updateArgAttrs).
  SmallVector<mlir::Type> origCallArgTypes;
  llvm::append_range(origCallArgTypes, argOperands.getTypes());
  if (argOperands.size() > fc.argInfos.size())
    return call.emitOpError()
           << "variadic arguments not yet implemented in CallConvLowering";
  assert(fc.argInfos.size() == argOperands.size() &&
         "call operand count must match classified arg count");
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind == ArgKind::Ignore)
      continue;
    mlir::Value arg = argOperands[idx];
    if (cir::RecordType flatTy = getFlattenedCoercedType(ac)) {
      // Direct + canFlatten: pass one scalar call argument per field of the
      // ABI-coerced struct.  When the original and coerced types differ in
      // layout, coerce through a memory slot and read each field with
      // cir.get_member + cir.load from that slot.  When the types already
      // match, decompose the struct value directly (reading from its source
      // alloca when possible).
      if (arg.getType() != flatTy) {
        SmallPtrSet<mlir::Operation *, 4> coercionOps;
        mlir::Value coercedPtr =
            emitCoercionToMemory(builder, call.getLoc(), flatTy, arg,
                                 enclosingFunc, dl, coercionOps);
        for (auto [f, fieldTy] : llvm::enumerate(flatTy.getMembers())) {
          mlir::Type fieldPtrTy = cir::PointerType::get(fieldTy);
          auto fieldPtr =
              cir::GetMemberOp::create(builder, call.getLoc(), fieldPtrTy,
                                       coercedPtr, /*name=*/"", /*index=*/f);
          newArgs.push_back(cir::LoadOp::create(builder, call.getLoc(), fieldTy,
                                                fieldPtr.getResult()));
        }
      } else {
        emitStructFieldArgs(builder, call.getLoc(), arg, flatTy, newArgs,
                            replacedWholeLoads);
      }
    } else if (ac.kind == ArgKind::Expand) {
      // Decompose the struct value into its constituent scalar fields and
      // pass each as a separate argument.
      auto recTy = cast<cir::RecordType>(arg.getType());
      assert(recTy.isStruct() &&
             "Expand classification requires a struct type, not a union");
      emitStructFieldArgs(builder, call.getLoc(), arg, recTy, newArgs,
                          replacedWholeLoads);
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
      mlir::Type argTy = arg.getType();
      auto ptrTy = cir::PointerType::get(argTy);
      uint64_t align = ac.indirectAlign.value();
      StringRef slotName = ac.byVal ? "byval" : "byref";
      auto slot = cir::AllocaOp::create(builder, call.getLoc(), ptrTy,
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
  mlir::Type origRetTy =
      hasResult ? call.getResult().getType() : cir::VoidType::get(ctx);

  // An indirect (sret) return has a different call shape than the coerce /
  // extend / ignore return handling further down (the value is returned
  // through a prepended pointer slot, not as a result), so dispatch to a
  // dedicated helper for it; everything below handles the by-value returns.
  if (fc.returnInfo.kind == ArgKind::Indirect && hasResult) {
    rewriteIndirectReturnCall(call, fc, newArgs, origRetTy, origCallArgTypes,
                              builder);
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
  // res_attrs for Extend args/return.  Ignore args require a rebuild because
  // their slots are dropped; Indirect args need llvm.byval / llvm.align;
  // Expand and Direct+canFlatten args change the argument count.
  bool needsArgAttrUpdate =
      llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend ||
               ac.kind == ArgKind::Indirect || ac.kind == ArgKind::Expand ||
               getFlattenedCoercedType(ac);
      });
  if (needsArgAttrUpdate) {
    auto existing = call->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
    newCall->setAttr("arg_attrs",
                     updateArgAttrs(ctx, origCallArgTypes, existing, fc));
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

  // Now that the original call is gone, drop any whole-struct loads whose
  // members we read directly from the source alloca, if nothing else uses
  // them.  A single load can feed several Expand operands (e.g. after CSE
  // merges identical loads), so dedupe before erasing to avoid touching a
  // freed op twice.
  SmallPtrSet<mlir::Operation *, 4> erased;
  for (cir::LoadOp wholeLoad : replacedWholeLoads)
    if (erased.insert(wholeLoad).second && wholeLoad.use_empty())
      wholeLoad->erase();

  return mlir::success();
}
