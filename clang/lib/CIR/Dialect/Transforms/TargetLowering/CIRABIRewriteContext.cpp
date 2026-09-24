//===- CIRABIRewriteContext.cpp - CIR ABI rewrite context ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRABIRewriteContext.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <utility>

using namespace cir;
using namespace mlir;
using namespace mlir::abi;

// This rewrite context supports the Direct (with or without coercion),
// Extend, Ignore, Indirect-return (sret), Indirect-argument (byval and
// non-byval), and Expand (struct flattening) classifications.
//
// An Indirect argument is byval or not, following its classification's
// byVal flag.  byval is a by-value parameter the ABI passes in memory rather
// than registers, usually for its size.  Non-byval is a by-value parameter
// whose type cannot be copied freely, because it has a non-trivial copy
// constructor, move constructor, or destructor, so the callee works on the
// caller's own object rather than a copy.
//
// At the call site byval copies into a fresh alloca while a non-byval
// argument forwards the caller's storage.  At the callee, byval loads the
// incoming pointer (a local copy), while non-byval rewires the CIRGen
// param-slot alloca to the incoming pointer so the body mutates the caller's
// storage in place.
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
/// Indirect (byval and non-byval), and Expand (struct flattening) arguments.
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
/// llvm.signext / llvm.zeroext on Extend args, and adds the pointer
/// attributes for Indirect args.  Preserves any existing arg attributes on
/// retained arg slots.  \p origArgTypes provides the pre-rewrite type for
/// each arg slot.
mlir::ArrayAttr updateArgAttrs(mlir::MLIRContext *ctx,
                               ArrayRef<mlir::Type> origArgTypes,
                               mlir::ArrayAttr existingArgAttrs,
                               const FunctionClassification &fc,
                               const mlir::DataLayout &dl) {
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
      StringRef attrName = ac.signExtend
                               ? mlir::LLVM::LLVMDialect::getSExtAttrName()
                               : mlir::LLVM::LLVMDialect::getZExtAttrName();
      mlir::NamedAttrList attrs(existing);
      attrs.set(attrName, builder.getUnitAttr());
      newArgAttrs.push_back(attrs.getDictionary(ctx));
    } else if (ac.kind == ArgKind::Indirect) {
      // byval hands the callee its own copy.  Without byval it gets a pointer
      // to the caller's own object.  Both state llvm.align and llvm.noundef,
      // which constrains the pointer operand, not the pointee's contents.
      //
      // llvm.byval(T) records the pre-rewrite arg type because the opaque
      // LLVM pointer cannot carry it.  llvm.nofreeobj says the object cannot
      // be freed while the callee runs, which holds because the caller owns it
      // across the call.
      mlir::Type pointeeTy = origArgTypes[oldIdx];
      mlir::NamedAttrList attrs(existing);
      attrs.set(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                builder.getI64IntegerAttr(ac.indirectAlign.value()));
      attrs.set(mlir::LLVM::LLVMDialect::getNoUndefAttrName(),
                builder.getUnitAttr());
      if (ac.byVal) {
        // Classic adds llvm.noalias under -fpass-by-value-is-noalias, which
        // CIR does not plumb through.
        assert(!cir::MissingFeatures::noaliasOnByvalAttr());
        attrs.set(mlir::LLVM::LLVMDialect::getByValAttrName(),
                  mlir::TypeAttr::get(pointeeTy));
      } else {
        // Classic adds llvm.dead_on_return when the object's lifetime ends in
        // the callee, which needs the destructor's triviality from
        // cir.record_layout's has_trivial_dtor.
        assert(!cir::MissingFeatures::deadOnReturnAttr());
        attrs.set(mlir::LLVM::LLVMDialect::getNoFreeObjAttrName(),
                  builder.getUnitAttr());
        attrs.set(mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
                  builder.getI64IntegerAttr(
                      dl.getTypeSize(pointeeTy).getFixedValue()));
      }
      newArgAttrs.push_back(attrs.getDictionary(ctx));
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

/// The number of bytes a coercion memory slot needs to hold a value of type
/// \p ty without truncating it. For most types this is the ordinary storage
/// size. For a _BitInt it is deliberately the value's own literal byte
/// footprint (ceil(width/8)) rather than the wider, ABI-alignment-padded
/// footprint a _BitInt gets as a record member (see
/// cir::IntType::getStorageTypeWidth): this coercion is about how many bytes
/// the *value* needs to round-trip, not how a record would lay it out, and
/// those are genuinely different questions for a _BitInt (e.g. _BitInt(33)
/// only needs 5 bytes here, even though it occupies 8 padded bytes as a
/// record member).
static uint64_t coercionByteSize(mlir::Type ty, const mlir::DataLayout &dl) {
  if (auto intTy = mlir::dyn_cast<cir::IntType>(ty))
    return llvm::divideCeil(intTy.getWidth(), 8);
  return dl.getTypeSize(ty);
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
/// The temporary alloca is placed at the start of \p slotBlock, which must
/// dominate every use of the coerced value and must be a block that ends up
/// inside the enclosing function's entry block after any later outlining.
///
/// \p offset is where the coerced value sits within the larger of the two
/// types, non-zero when the ABI passes a value in a register read from
/// partway into its storage.  The coerced type must be the smaller of the
/// two, so one that can exceed the value it coerces, such as a multi-field
/// register tuple, must not be given an offset.
///
/// Any operations the helper creates are appended to \p createdOps so the
/// caller can pass them to replaceAllUsesExcept and avoid clobbering the
/// store's value operand when later rewiring the source value.
mlir::Value emitCoercionToMemory(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Type dstTy, mlir::Value src,
                                 mlir::Block *slotBlock,
                                 const mlir::DataLayout &dl,
                                 SmallPtrSetImpl<mlir::Operation *> &createdOps,
                                 unsigned offset) {
  mlir::Type srcTy = src.getType();
  assert(srcTy != dstTy &&
         "emitCoercion callers must pre-check that the types differ");

  uint64_t srcAlign = dl.getTypeABIAlignment(srcTy);
  uint64_t dstAlign = dl.getTypeABIAlignment(dstTy);
  uint64_t allocaAlign = std::max(srcAlign, dstAlign);
  mlir::Type slotTy = coercionByteSize(srcTy, dl) >= coercionByteSize(dstTy, dl)
                          ? srcTy
                          : dstTy;

  // Sizes are compared two ways on purpose: relative size in
  // coercionByteSize terms, which is how slotTy was picked, and capacity in
  // getTypeSize terms, which is what the alloca is given.
  [[maybe_unused]] mlir::Type coercedTy = ((slotTy == srcTy) ? dstTy : srcTy);
  assert((offset == 0 ||
          coercionByteSize(coercedTy, dl) < coercionByteSize(slotTy, dl)) &&
         "a direct offset must land on the coerced side, the smaller one");
  assert((offset == 0 ||
          offset + dl.getTypeSize(coercedTy) <= dl.getTypeSize(slotTy)) &&
         "coerce slot too small for offset access");
  assert((offset == 0 || offset % dl.getTypeABIAlignment(coercedTy) == 0) &&
         "a direct offset must be aligned for the coerced access");

  auto slotPtrTy = cir::PointerType::get(slotTy);
  auto srcPtrTy = cir::PointerType::get(srcTy);
  auto dstPtrTy = cir::PointerType::get(dstTy);

  cir::AllocaOp alloca;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(slotBlock);
    alloca = cir::AllocaOp::create(builder, loc, slotPtrTy,
                                   builder.getStringAttr("coerce"),
                                   builder.getI64IntegerAttr(allocaAlign));
  }
  createdOps.insert(alloca);

  // The alloca already has slotTy, so asking for that type returns it
  // unchanged.  Any other type is reached by a bitcast, preceded by a byte
  // stride when the coerced value lives at an offset.
  auto slotView = [&](mlir::Type wantTy,
                      cir::PointerType wantPtrTy) -> mlir::Value {
    if (wantTy == slotTy)
      return alloca;
    mlir::Value base = alloca;
    if (offset != 0) {
      auto u8Ty =
          cir::IntType::get(builder.getContext(), 8, /*isSigned=*/false);
      auto u8PtrTy = cir::PointerType::get(u8Ty);
      auto u8Base = cir::CastOp::create(builder, loc, u8PtrTy,
                                        cir::CastKind::bitcast, alloca);
      createdOps.insert(u8Base);
      auto strideTy =
          cir::IntType::get(builder.getContext(), 64, /*isSigned=*/true);
      auto strideVal = cir::ConstantOp::create(
          builder, loc, cir::IntAttr::get(strideTy, offset));
      createdOps.insert(strideVal);
      base = cir::PtrStrideOp::create(builder, loc, u8PtrTy, u8Base, strideVal);
      createdOps.insert(base.getDefiningOp());
    }
    auto cast = cir::CastOp::create(builder, loc, wantPtrTy,
                                    cir::CastKind::bitcast, base);
    createdOps.insert(cast);
    return cast;
  };

  // Store through a source-typed view of the slot.
  mlir::Value srcSlot = slotView(srcTy, srcPtrTy);
  auto store = cir::StoreOp::create(builder, loc, src, srcSlot);
  createdOps.insert(store);

  // Return a destination-typed view of the slot.
  return slotView(dstTy, dstPtrTy);
}

/// Coerce \p src to type \p dstTy by going through memory and load the whole
/// coerced value back out.  Builds on emitCoercionToMemory, adding the final
/// load of the destination-typed view.
mlir::Value emitCoercion(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type dstTy, mlir::Value src,
                         mlir::Block *slotBlock, const mlir::DataLayout &dl,
                         SmallPtrSetImpl<mlir::Operation *> &createdOps,
                         unsigned offset) {
  mlir::Value dstSlot = emitCoercionToMemory(builder, loc, dstTy, src,
                                             slotBlock, dl, createdOps, offset);
  auto load = cir::LoadOp::create(builder, loc, dstSlot);
  createdOps.insert(load);
  return load;
}

/// Convenience overload for callers that don't need the createdOps set
/// (e.g. call-site coercion where we don't replaceAllUsesExcept).
mlir::Value emitCoercion(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type dstTy, mlir::Value src,
                         mlir::Block *slotBlock, const mlir::DataLayout &dl,
                         unsigned offset) {
  SmallPtrSet<mlir::Operation *, 4> ignored;
  return emitCoercion(builder, loc, dstTy, src, slotBlock, dl, ignored, offset);
}

/// The block a coercion slot's alloca belongs at the start of.
///
/// Normally the enclosing function's entry block, where HoistAllocas expects
/// allocas to be.  A body carrying a call is not always inside a function
/// when this pass runs, though, because LoweringPrepare runs after it: a
/// namespace-scope `T g = makeT();` is still in its cir.global ctor region,
/// and an OpenACC recipe's init and destroy bodies are in regions the module
/// owns.  Those take the outermost region below the module, which dominates
/// the whole body and travels with it when the body is outlined.
mlir::Block *coercionSlotBlock(mlir::Operation *op) {
  if (auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>())
    return &funcOp->getRegion(0).front();
  mlir::Region *region = op->getParentRegion();
  while (mlir::Region *outer = region->getParentRegion()) {
    if (mlir::isa<mlir::ModuleOp>(outer->getParentOp()))
      break;
    region = outer;
  }
  assert(!region->empty() && "coercion slot needs a block to hold the alloca");
  return &region->front();
}

/// Insert coercion before each cir.return so the returned value matches the
/// new (coerced) return type.
void insertReturnCoercion(mlir::FunctionOpInterface funcOp,
                          mlir::Type origRetTy, mlir::Type coercedRetTy,
                          mlir::OpBuilder &builder, const mlir::DataLayout &dl,
                          unsigned offset) {
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
        emitCoercion(builder, r.getLoc(), coercedRetTy, origVal,
                     &funcOp->getRegion(0).front(), dl, offset);
    r->setOperand(0, coerced);
  }
}

/// \p val's defining load, if it is simple, meaning neither volatile nor
/// atomic.  Null otherwise: a non-simple load's access has to survive as
/// written, and a call result or any other first-class value has no defining
/// load at all.
static cir::LoadOp maybeGetSimpleLoad(mlir::Value val) {
  cir::LoadOp load = val.getDefiningOp<cir::LoadOp>();
  if (!load || load.getIsVolatile() || load.getMemOrder())
    return {};
  return load;
}

/// \p recordVal's defining load, if it is simple and its address resolves to
/// an alloca.  Null otherwise.
static cir::LoadOp getWholeRecordLoad(mlir::Value recordVal) {
  cir::LoadOp load = maybeGetSimpleLoad(recordVal);
  if (!load || !cir::getUnderlyingAlloca(load.getAddr()))
    return {};
  return load;
}

/// Whether a non-byval indirect argument may name \p addr, given the callee is
/// told the argument is \p minAlign aligned.  A slot allocated here qualifies,
/// reached through storage-preserving casts, and so does the enclosing
/// function's own non-byval parameter: its slot stands until
/// finalizeParameterSlots, and states the alignment the parameter promises
/// rather than the one CIRGen chose for a local copy.
///
/// The slot must already state that alignment.  Raising it here would not
/// survive one that stands in for a parameter, since finalizeParameterSlots
/// replaces it with the incoming pointer, which would discard the raise and
/// leave the callee over-promised.
static bool forwardableNonByvalStorage(mlir::Value addr, uint64_t minAlign) {
  cir::AllocaOp slot = cir::getUnderlyingAlloca(addr);
  return slot && slot.getAlignment() >= minAlign;
}

/// Decompose a struct value into one scalar call argument per field of \p
/// recTy, appending the field values to \p newArgs.  When \p structVal is a
/// simple load from an alloca, read each field with cir.get_member +
/// cir.load from the address the load used, emitted at the original load's
/// position so they observe the same memory state, and record the now-dead
/// whole-struct load in \p deadRecordLoads for later erasure.  Otherwise (a
/// call result, compound literal, or a volatile or atomic load) extract each
/// field from the value with cir.extract_member.  Loading the members from
/// memory rather than extracting from a whole-struct value keeps the result in
/// a form SROA can promote (it does not reason about extractvalue).  Shared by
/// the Expand and Direct+canFlatten argument paths.
static void emitStructFieldArgs(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value structVal, cir::RecordType recTy,
                                SmallVectorImpl<mlir::Value> &newArgs,
                                SmallVectorImpl<cir::LoadOp> &deadRecordLoads) {
  cir::LoadOp srcLoad = getWholeRecordLoad(structVal);

  if (srcLoad) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(srcLoad);
    cir::PointerType baseTy = srcLoad.getAddr().getType();
    for (auto [f, fieldTy] : llvm::enumerate(recTy.getMembers())) {
      mlir::Type fieldPtrTy =
          cir::PointerType::get(fieldTy, baseTy.getAddrSpace());
      mlir::Value fieldPtr = cir::GetMemberOp::create(
          builder, loc, fieldPtrTy, srcLoad.getAddr(), /*name=*/"",
          /*index=*/f);
      newArgs.push_back(cir::LoadOp::create(builder, loc, fieldPtr));
    }
    deadRecordLoads.push_back(srcLoad);
  } else {
    for (unsigned f = 0; f < recTy.getNumElements(); ++f)
      newArgs.push_back(
          cir::ExtractMemberOp::create(builder, loc, structVal, f));
  }
}

/// Erase the loads that a rewritten call left unused.  The old call must
/// already be erased, since until then it still counts as a user.  One load can
/// feed two operands of the same call, as in f(s, s), so \p loads can hold the
/// same load twice.  A load that another op still reads is left alone.
static void eraseDeadRecordLoads(ArrayRef<cir::LoadOp> loads) {
  llvm::SmallSetVector<mlir::Operation *, 4> uniqueLoads(llvm::from_range,
                                                         loads);
  for (mlir::Operation *load : uniqueLoads)
    if (load->use_empty())
      load->erase();
}

/// The store that spills non-byval indirect parameter \p blockArg, and the
/// slot it spills into.  prepareNonByvalParameters has already established
/// that the spill is the block argument's only use and that it stores into
/// an alloca, which is what the assertions below rest on.  Both results are
/// null when the parameter has no spill, which is so only when nothing uses
/// it at all.
static std::pair<cir::StoreOp, cir::AllocaOp>
findParamSpill(mlir::BlockArgument blockArg) {
  if (blockArg.use_empty())
    return {};
  assert(blockArg.hasOneUse() &&
         "non-byval arg must have exactly one use (the CIRGen param spill)");
  auto store = cast<cir::StoreOp>(*blockArg.user_begin());
  assert(store.getValue() == blockArg &&
         "non-byval arg's use must be the value operand of its store");
  return {store, cast<cir::AllocaOp>(store.getAddr().getDefiningOp())};
}

/// For each Direct arg with a coerced type, change the block argument's type
/// to the coerced type and insert a coercion at function entry that maps it
/// back to the original type for body uses.  For each Indirect byval arg,
/// change the block argument's type to a pointer and insert a load at entry
/// so the body sees a local copy of the original value type.  For each
/// Indirect non-byval arg, change the block argument to a pointer and queue
/// the param-slot alloca to be replaced by it (no entry load /
/// byte-copy) so the body operates on the caller's storage in place.  For each
/// Expand arg, replace the single struct block argument with N scalar block
/// arguments (one per field) and store each field directly into the parameter's
/// own alloca (the CIRGen spill slot), erasing the original whole-struct store.
///
/// \p hasSRetArg is true when the function has an sret return (a hidden return
/// pointer is prepended as block argument 0).  Expand arguments expand the
/// block argument count, so a running index tracks the current block argument
/// position rather than computing the classification index + \p hasSRetArg
/// directly.
void insertArgCoercion(
    mlir::FunctionOpInterface funcOp, const FunctionClassification &fc,
    mlir::OpBuilder &builder, const mlir::DataLayout &dl, bool hasSRetArg,
    SmallVectorImpl<std::pair<cir::AllocaOp, mlir::BlockArgument>>
        &pendingParamSlots) {
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
        assert(!ac.directOffset &&
               "each field is read from slot offset 0 here, so a flattened "
               "coercion cannot honor a direct offset");
        finalVal = emitCoercion(builder, loc, origTy, flatLoaded, &entry, dl,
                                coercionOps, /*offset=*/0);
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
      mlir::Value adapted =
          emitCoercion(builder, funcOp.getLoc(), oldArgTy, blockArg, &entry, dl,
                       coercionOps, ac.directOffset);

      // Replace blockArg uses with the adapted value, except inside the
      // helper ops we just created.  This is critical: the StoreOp's value
      // operand is blockArg, and if we naively replaceAllUses it gets swapped
      // to adapted (now of the original type != the alloca's pointee type).
      blockArg.replaceAllUsesExcept(adapted, coercionOps);
    } else if (ac.kind == ArgKind::Indirect) {
      // byval and non-byval both lower to !cir.ptr<T>, and which it is shows
      // up only in the attrs updateArgAttrs applies.  Body lowering differs:
      // byval copies into the callee (load at entry), while non-byval must
      // operate on the caller's storage in place.
      auto ptrTy = cir::PointerType::get(blockArg.getType());

      if (!ac.byVal) {
        // Without byval, drop the spill store and let the slot's uses read the
        // incoming pointer, so the body operates on the caller's storage in
        // place.  A byte-copy would be wrong for non-trivially-copyable
        // aggregates (e.g. libstdc++ SSO std::string, where it would leave
        // `_M_p` aliasing the source's `_M_local_buf`).
        auto [paramStore, destAlloca] = findParamSpill(blockArg);

        if (paramStore)
          paramStore->erase();

        // Update the block argument to point to its original type.
        blockArg.setType(ptrTy);

        // Pointing the slot's uses at the incoming pointer waits until every
        // call site has been rewritten.  A call that hands this parameter
        // straight on recognizes it by the slot its operand was loaded from,
        // and collapsing the slot here would leave that call reading a block
        // argument with no defining operation to inspect.  A dead spill DCE
        // already removed leaves nothing to collapse.
        if (destAlloca)
          pendingParamSlots.emplace_back(destAlloca, blockArg);
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

/// Copy the call attributes from \p source to the rebuilt \p target. The
/// callee is already established by CallOp::create and may have been rewritten,
/// so it is not copied. Existing discardable attributes on \p target take
/// precedence over attributes from \p source.
static void copyCallAttributes(cir::CallOp source, cir::CallOp target) {
  source->getName().walkInherentAttrs(source, [&](llvm::StringRef name,
                                                  mlir::Attribute &attr) {
    if (name != cir::CIRDialect::getCalleeAttrName())
      target->setInherentAttr(mlir::StringAttr::get(source->getContext(), name),
                              attr);
  });
  for (mlir::NamedAttribute attr : source->getDiscardableAttrs())
    if (!target->hasDiscardableAttr(attr.getName()))
      target->setDiscardableAttr(attr.getName(), attr.getValue());
}

/// For an indirect call, prepend the callee function pointer as operand 0 so
/// CallOp::create rebuilds it as an indirect call, bitcasting it to a function
/// pointer whose signature matches the rewritten operands and return type.
/// No-op for direct calls.
static void prependIndirectCallee(cir::CallOp call,
                                  SmallVectorImpl<mlir::Value> &args,
                                  mlir::Type retTy, mlir::OpBuilder &builder) {
  if (!call.isIndirect())
    return;
  mlir::Value calleePtr = call.getIndirectCall();
  SmallVector<mlir::Type> paramTypes;
  paramTypes.reserve(args.size());
  llvm::transform(args, std::back_inserter(paramTypes),
                  [](mlir::Value v) { return v.getType(); });
  // Lowering builds an indirect call's LLVM function type from the callee
  // pointer's pointee and takes the call's result from that type, so the
  // pointee's return type has to track the rewrite: an sret return would
  // leave a result the call no longer produces, and a coerced return one of
  // the wrong type.  The ellipsis has to survive for the same reason: the
  // rebuilt pointee is what makes the lowered call variadic, and only a
  // variadic call gets the vector-register count that the x86_64 SysV ABI
  // passes in AL and that the callee's va_arg reads back.
  auto calleeFnTy = cast<cir::FuncType>(
      cast<cir::PointerType>(calleePtr.getType()).getPointee());
  auto newPtrTy = cir::PointerType::get(
      cir::FuncType::get(paramTypes, retTy, calleeFnTy.isVarArg()));
  if (calleePtr.getType() != newPtrTy)
    calleePtr = cir::CastOp::create(builder, call.getLoc(), newPtrTy,
                                    cir::CastKind::bitcast, calleePtr);
  args.insert(args.begin(), calleePtr);
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
                               mlir::OpBuilder &builder,
                               const mlir::DataLayout &dl) {
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
  prependIndirectCallee(call, sretArgs, sretVoidTy, builder);
  auto newCall = cir::CallOp::create(
      builder, call.getLoc(), call.getCalleeAttr(), sretVoidTy, sretArgs);
  copyCallAttributes(call, newCall);
  newCall->removeAttr("res_attrs");

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
    argAttrs = updateArgAttrs(ctx, origCallArgTypes, argAttrs, fc, dl);
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

/// Whether \p ty, a type the classifier named for one register of a
/// coercion, is carried in a vector register.
bool isSSERegisterClass(mlir::Type ty) {
  return mlir::isa<cir::VectorType, cir::FPTypeInterface>(ty);
}

} // namespace

/// Bring \p funcOp's non-byval indirect parameter \p argNo into the shape the
/// rest of the rewrite assumes.  \p claimedSlots carries the slots \p funcOp's
/// earlier non-byval indirect parameters took.  The shape itself, and the
/// cases reported rather than repaired, are documented on
/// CIRABIRewriteContext::prepareNonByvalParameters in the header.
static mlir::LogicalResult prepareOneNonByvalParameter(
    cir::FuncOp funcOp, unsigned argNo, const ArgClassification &ac,
    mlir::BlockArgument blockArg, mlir::DominanceInfo &dom,
    SmallPtrSetImpl<mlir::Operation *> &claimedSlots) {
  // The spill is the store that writes the parameter itself.  Any other use
  // consumes the record value.  At -O1 and above such a use comes from
  // cir-simplify: CIRGen marks a const-qualified parameter's slot const, so
  // the load of it folds to the stored parameter.
  cir::StoreOp spill;
  cir::StoreOp extraSpill;
  mlir::Operation *otherUse = nullptr;
  SmallVector<mlir::OpOperand *> callArgs;
  for (mlir::OpOperand &use : blockArg.getUses()) {
    auto store = dyn_cast<cir::StoreOp>(use.getOwner());
    if (store && store.getValue() == blockArg) {
      // Which of two stores is the spill decides which slot stands in for the
      // parameter, and the use list is in no particular order, so there is
      // nothing to prefer between them.
      if (spill)
        extraSpill = store;
      else
        spill = store;
      continue;
    }
    // Only an argument is served by a load of the slot.  Any other consumer
    // belongs to a rewrite that reads the parameter its own way: a returned
    // record, for one, is rewritten through the sret slot, which assumes the
    // returned load names the return slot and not this one.
    auto call = dyn_cast<cir::CIRCallOpInterface>(use.getOwner());
    if (call && llvm::is_contained(call.getArgOperands(), blockArg))
      callArgs.push_back(&use);
    else if (!otherUse)
      otherUse = use.getOwner();
  }

  if (extraSpill)
    return extraSpill->emitOpError()
           << "non-byval parameter " << argNo
           << " spilled more than once is not yet implemented in "
              "CallConvLowering";

  if (otherUse)
    return otherUse->emitOpError()
           << "non-byval parameter " << argNo
           << " consumed other than as a call argument is not yet implemented "
              "in CallConvLowering";

  if (!spill) {
    // Every other kind of use was reported above, so the parameter has no
    // uses at all and needs neither a slot nor a read.
    if (callArgs.empty())
      return mlir::success();

    // Without a spill the parameter becomes the incoming pointer directly,
    // which is the storage an argument taken from it has to name.  Giving it
    // the spill it lacks lets the checks and the read below apply unchanged,
    // and neither survives the pass: insertArgCoercion erases the store and
    // finalizeParameterSlots replaces the slot.
    mlir::OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(blockArg.getOwner());
    auto synthesized = cir::AllocaOp::create(
        builder, funcOp.getLoc(), cir::PointerType::get(blockArg.getType()),
        builder.getStringAttr("nonbyval.param"),
        builder.getI64IntegerAttr(ac.indirectAlign.value()));
    spill =
        cir::StoreOp::create(builder, funcOp.getLoc(), blockArg, synthesized);
  }

  // The incoming pointer replaces the slot itself, in the default address
  // space.  Retargeting a cast of it would leave the allocation's other views
  // reading storage nothing writes.  Storage in another address space, or
  // storage that is not a local alloca at all, is not something the incoming
  // pointer can be substituted for.
  cir::AllocaOp slot = spill.getAddr().getDefiningOp<cir::AllocaOp>();
  if (!slot ||
      spill.getAddr().getType() != cir::PointerType::get(blockArg.getType()))
    return spill->emitOpError()
           << "non-byval parameter " << argNo
           << " spilled to storage that cannot take the incoming pointer is "
              "not yet implemented in CallConvLowering";

  // One slot stands in for one non-byval parameter, since the incoming
  // pointer replaces it.  Two of them spilled to the same slot are each
  // spilled once, so the check above cannot see the collision and only
  // comparing the slots can.
  if (!claimedSlots.insert(slot).second)
    return spill->emitOpError()
           << "non-byval parameter " << argNo
           << " sharing its spill slot with another parameter is not yet "
              "implemented in CallConvLowering";

  // CIRGen picked the slot's alignment for a local copy of the record, but the
  // slot is about to stand in for the parameter, and a call forwarding it may
  // only promise what the incoming pointer does.
  slot.setAlignment(ac.indirectAlign.value());

  if (callArgs.empty())
    return mlir::success();

  // A reader the spill does not dominate could read the slot before anything
  // wrote it, and a load placed at the spill would not dominate it either.
  for (mlir::OpOperand *callArg : callArgs)
    if (!dom.properlyDominates(spill, callArg->getOwner()))
      return callArg->getOwner()->emitOpError()
             << "non-byval parameter " << argNo
             << " read before its spill is not yet implemented in "
                "CallConvLowering";

  // Read the record back out of the slot instead, so every reader sees the
  // shape CIRGen emits without cir-simplify: a load naming the storage the
  // argument would have to name anyway.  Reading at the spill, not at the
  // reader, keeps the value the parameter's own whatever the body later
  // stores into the slot, and the load promises only the alignment the
  // classification gives the incoming pointer.
  //
  // A call that forwards the argument consumes the load once it recognizes
  // the slot.  One that wants a copy keeps it, reading the incoming pointer
  // once finalizeParameterSlots replaces the slot.
  mlir::OpBuilder builder(spill);
  builder.setInsertionPointAfter(spill);
  auto reload = cir::LoadOp::create(builder, spill.getLoc(), slot);
  reload.setAlignment(ac.indirectAlign.value());
  for (mlir::OpOperand *callArg : callArgs)
    callArg->set(reload.getResult());
  return mlir::success();
}

mlir::LogicalResult CIRABIRewriteContext::prepareNonByvalParameters(
    cir::FuncOp funcOp, const FunctionClassification &fc) {
  if (!funcOp.isDefinition())
    return mlir::success();
  mlir::Region &body = funcOp->getRegion(0);
  if (body.empty())
    return mlir::success();
  mlir::Block &entry = body.front();
  mlir::DominanceInfo dom;
  SmallPtrSet<mlir::Operation *, 4> claimedSlots;

  // No signature has been rewritten yet, so no sret pointer has been prepended
  // and no Expand argument has been split into its fields.  Every
  // classification therefore still maps to the entry block argument at its own
  // index.
  for (auto [idx, ac] : llvm::enumerate(fc.argInfos)) {
    if (ac.kind != ArgKind::Indirect || ac.byVal)
      continue;
    assert(idx < entry.getNumArguments() &&
           "classification count must not exceed entry block arguments");
    if (failed(prepareOneNonByvalParameter(
            funcOp, idx, ac, entry.getArgument(idx), dom, claimedSlots)))
      return mlir::failure();
  }
  return mlir::success();
}

void CIRABIRewriteContext::finalizeParameterSlots() {
  for (auto [slot, incoming] : pendingParamSlots) {
    slot.getResult().replaceAllUsesWith(incoming);
    slot->erase();
  }
  pendingParamSlots.clear();
}

mlir::LogicalResult CIRABIRewriteContext::rewriteFunctionDefinition(
    mlir::FunctionOpInterface funcOpInterface, const FunctionClassification &fc,
    mlir::OpBuilder &builder) {
  // The pass driver (CallConvLoweringPass) only ever hands us cir.func ops.
  // Cast once at the top so the rest of the function reads in CIR's own
  // vocabulary, and so we can dispatch to the CIRGlobalValueInterface for
  // isDefinition() (FunctionOpInterface alone does not inherit from
  // CIRGlobalValueInterface).
  cir::FuncOp funcOp = mlir::cast<cir::FuncOp>(funcOpInterface);

  if (!fc.needsRewrite())
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
      insertArgCoercion(funcOp, fc, builder, dl, hasSRet, pendingParamSlots);

      // Direct return with coerced type: insert a coercion at every
      // cir.return so the returned value matches the (coerced) return
      // type in the new function signature set below.
      if (fc.returnInfo.kind == ArgKind::Direct && fc.returnInfo.coercedType &&
          !oldResultTypes.empty() && fc.returnInfo.coercedType != origRetTy)
        insertReturnCoercion(funcOp, origRetTy, fc.returnInfo.coercedType,
                             builder, dl, fc.returnInfo.directOffset);

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
  // Extend (needs llvm.signext / llvm.zeroext), Indirect (gains the pointer
  // attributes updateArgAttrs applies), Expand or Direct+canFlatten (both
  // change the argument count).
  bool needsArgAttrUpdate =
      hasSRet || llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
        return ac.kind == ArgKind::Ignore || ac.kind == ArgKind::Extend ||
               ac.kind == ArgKind::Indirect || ac.kind == ArgKind::Expand ||
               getFlattenedCoercedType(ac);
      });
  if (needsArgAttrUpdate) {
    auto existing = funcOp->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
    mlir::ArrayAttr updated =
        updateArgAttrs(ctx, oldArgTypes, existing, fc, dl);
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

  if (mlir::isa<cir::VoidType>(newRetTy)) {
    funcOp->removeAttr("res_attrs");
  } else if (fc.returnInfo.kind == ArgKind::Extend) {
    // Layer llvm.signext / llvm.zeroext onto an Extend return.
    auto existing = funcOp->getAttrOfType<mlir::ArrayAttr>("res_attrs");
    funcOp->setAttr("res_attrs", updateResAttrs(ctx, existing, fc.returnInfo));
  }

  return mlir::success();
}

mlir::LogicalResult
CIRABIRewriteContext::rewriteCallSite(mlir::Operation *callOp,
                                      const FunctionClassification &fc,
                                      mlir::OpBuilder &builder) {
  // The classification covers exactly the callee's declared parameters, and
  // the rewrite below pairs it with the call's operands one for one.  Both
  // directions of a mismatch have to be reported before the pass-through early
  // return, or a call whose declared parameters happen to be pass-through is
  // left as written with its surplus operands never classified.
  //
  // A surplus operand went through an ellipsis.  A shortfall means the callee
  // was declared no_proto, which turns off the verifier's argument-count check
  // altogether.
  unsigned numOperands =
      mlir::cast<cir::CIRCallOpInterface>(callOp).getNumArgOperands();
  if (numOperands > fc.argInfos.size())
    return callOp->emitOpError()
           << "variadic arguments not yet implemented in CallConvLowering";
  if (numOperands < fc.argInfos.size())
    return callOp->emitOpError()
           << "call passes fewer arguments than the callee declares, which is "
              "not yet implemented in CallConvLowering";

  if (!fc.needsRewrite())
    return mlir::success();

  if (mlir::isa<cir::TryCallOp>(callOp))
    return callOp->emitOpError()
           << "TryCallOp not yet implemented in CallConvLowering";

  auto call = mlir::cast<cir::CallOp>(callOp);
  mlir::MLIRContext *ctx = callOp->getContext();
  mlir::Block *slotBlock = coercionSlotBlock(call);

  builder.setInsertionPoint(call);

  SmallVector<mlir::Value> newArgs;
  mlir::ValueRange argOperands = call.getArgOperands();
  newArgs.reserve(argOperands.size());

  // Loads that the new call leaves unused: Expand and Direct+canFlatten read
  // the fields out of the source alloca, and a non-byval argument passes the
  // address the load read from.  The old call still uses them, so erase them
  // only after it is gone.
  SmallVector<cir::LoadOp> deadRecordLoads;

  // Capture original arg types before building newArgs (byval slots change
  // the wire argument from T to !cir.ptr<T>, so we save the pre-rewrite
  // types here for use in updateArgAttrs).
  SmallVector<mlir::Type> origCallArgTypes;
  llvm::append_range(origCallArgTypes, argOperands.getTypes());
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
        assert(!ac.directOffset &&
               "each field is read from slot offset 0 here, so a flattened "
               "coercion cannot honor a direct offset");
        mlir::Value coercedPtr =
            emitCoercionToMemory(builder, call.getLoc(), flatTy, arg, slotBlock,
                                 dl, coercionOps, /*offset=*/0);
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
                            deadRecordLoads);
      }
    } else if (ac.kind == ArgKind::Expand) {
      // Decompose the struct value into its constituent scalar fields and
      // pass each as a separate argument.
      auto recTy = cast<cir::RecordType>(arg.getType());
      assert(recTy.isStruct() &&
             "Expand classification requires a struct type, not a union");
      emitStructFieldArgs(builder, call.getLoc(), arg, recTy, newArgs,
                          deadRecordLoads);
    } else if (ac.kind == ArgKind::Direct && ac.coercedType &&
               arg.getType() != ac.coercedType) {
      arg = emitCoercion(builder, call.getLoc(), ac.coercedType, arg, slotBlock,
                         dl, ac.directOffset);
      newArgs.push_back(arg);
    } else if (ac.kind == ArgKind::Indirect) {
      // byval hands the callee its own copy.  Without byval the argument must
      // name the caller's storage instead, so that the object the callee
      // operates on is the one the caller destroys.  That means forwarding
      // the address the operand was loaded from rather than the loaded value,
      // so a store to that storage after the load is visible to the callee.
      if (!ac.byVal) {
        // The rewritten parameter is a pointer to the argument type in the
        // default address space, so an operand read through an address-space
        // cast cannot be handed on as it stands.  cir.load already pins the
        // pointee type, so only the address space can differ.
        cir::LoadOp srcLoad = maybeGetSimpleLoad(arg);
        if (!srcLoad ||
            srcLoad.getAddr().getType() !=
                cir::PointerType::get(arg.getType()) ||
            !forwardableNonByvalStorage(srcLoad.getAddr(),
                                        ac.indirectAlign.value()))
          return call->emitOpError()
                 << "non-byval indirect argument that does not name the "
                    "caller's storage is not yet implemented in "
                    "CallConvLowering";
        newArgs.push_back(srcLoad.getAddr());
        deadRecordLoads.push_back(srcLoad);
        continue;
      }
      auto ptrTy = cir::PointerType::get(arg.getType());
      auto slot = cir::AllocaOp::create(
          builder, call.getLoc(), ptrTy, builder.getStringAttr("byval"),
          builder.getI64IntegerAttr(ac.indirectAlign.value()));
      cir::StoreOp::create(builder, call.getLoc(), arg, slot);
      newArgs.push_back(slot);
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
                              builder, dl);
    eraseDeadRecordLoads(deadRecordLoads);
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
  prependIndirectCallee(call, newArgs, callRetTy, builder);
  auto newCall = cir::CallOp::create(builder, call.getLoc(),
                                     call.getCalleeAttr(), callRetTy, newArgs);
  copyCallAttributes(call, newCall);

  // Direct return with coercion: the new call returns the coerced type;
  // emit a coercion back to the original type for the call's existing uses.
  if (returnNeedsCoercion) {
    builder.setInsertionPointAfter(newCall);
    mlir::Value coercedBack =
        emitCoercion(builder, call.getLoc(), origRetTy, newCall.getResult(),
                     slotBlock, dl, fc.returnInfo.directOffset);
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
                     updateArgAttrs(ctx, origCallArgTypes, existing, fc, dl));
  }
  if (fc.returnInfo.kind == ArgKind::Extend) {
    auto existing = call->getAttrOfType<mlir::ArrayAttr>("res_attrs");
    newCall->setAttr("res_attrs", updateResAttrs(ctx, existing, fc.returnInfo));
  } else if (hasResult && mlir::isa<cir::VoidType>(callRetTy)) {
    newCall->removeAttr("res_attrs");
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
  eraseDeadRecordLoads(deadRecordLoads);

  return mlir::success();
}

void CIRABIRewriteContext::rewriteFunctionAddress(cir::GetGlobalOp addrOp,
                                                  cir::FuncOp funcOp,
                                                  mlir::OpBuilder &builder) {
  auto oldPtrTy = mlir::cast<cir::PointerType>(addrOp.getAddr().getType());
  cir::FuncType newFuncTy = funcOp.getFunctionType();
  // An extension rides on an argument attribute and leaves the signature
  // alone, so such a callee still matches the written type.
  if (newFuncTy == oldPtrTy.getPointee())
    return;

  // The verifier requires the retype even when nothing reads the address.
  addrOp.getAddr().setType(cir::PointerType::get(newFuncTy));
  if (addrOp.getAddr().use_empty())
    return;

  // A later indirect call through the written type stays correct, since it
  // reclassifies from that type and coerces to the signature funcOp was
  // rewritten to.  Ellipsis arguments are the exception the indirect-call
  // path reports rather than lowers.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(addrOp);
  auto bitcast = cir::CastOp::create(builder, addrOp.getLoc(), oldPtrTy,
                                     cir::CastKind::bitcast, addrOp.getAddr());
  addrOp.getAddr().replaceAllUsesExcept(bitcast.getResult(), bitcast);
}

namespace {

/// What one `va_arg` expansion needs from the op it rewrites.  The x86-64
/// cursor fields are reached through `vaFields` by index: 0 gp_offset,
/// 1 fp_offset, 2 overflow_arg_area, 3 reg_save_area.
struct VAArgFetch {
  VAArgFetch(mlir::Location loc, mlir::Value valist,
             llvm::ArrayRef<mlir::Type> vaFields, mlir::Type resultTy,
             const ArgClassification &ac, const mlir::DataLayout &dl,
             mlir::ModuleOp module)
      : loc(loc), valist(valist), vaFields(vaFields), resultTy(resultTy),
        ac(ac), dl(dl), module(module) {
    assert(vaFields.size() == 4 &&
           "the x86-64 va_list is a four-field cursor, checked by the caller");
  }

  mlir::Location loc;
  mlir::Value valist;
  llvm::ArrayRef<mlir::Type> vaFields;
  mlir::Type resultTy;
  const ArgClassification &ac;
  const mlir::DataLayout &dl;
  mlir::ModuleOp module;
};

/// The cursor fields a register fetch reads, and the predicate saying every
/// class it needs still has room.  An offset is null when the fetch needs no
/// register of that class.
struct RegisterCursor {
  mlir::Value gpOffsetP;
  mlir::Value fpOffsetP;
  mlir::Value gpOffset;
  mlir::Value fpOffset;
  mlir::Value inRegs;
};

mlir::LogicalResult reportVAArgNYI(cir::VAArgOp op, llvm::StringRef what) {
  op->emitOpError() << "va_arg of " << what
                    << " not yet implemented in CallConvLowering";
  return mlir::failure();
}

/// Sets \p isRegPair when a two-register argument is a coerced pair, one
/// register per member, rather than a single wide scalar, and records in
/// \p pairIsSse which element is SSE class.  Fails on a coercion that is not
/// two eightbytes.
mlir::LogicalResult classifyRegisterPair(cir::VAArgOp op,
                                         const ArgClassification &ac,
                                         std::array<bool, 2> &pairIsSse,
                                         bool &isRegPair) {
  auto pairTy = mlir::dyn_cast<cir::RecordType>(ac.coercedType);
  if (!pairTy)
    return mlir::success();

  if (pairTy.getNumElements() != 2)
    return reportVAArgNYI(op, "a register coercion that is not two eightbytes");

  assert(!ac.directOffset &&
         "a pair already spans both eightbytes, so it cannot also start "
         "partway into the value");
  for (auto [i, memberTy] : llvm::enumerate(pairTy.getMembers()))
    pairIsSse[i] = isSSERegisterClass(memberTy);
  isRegPair = true;
  return mlir::success();
}

/// The alignment an argument is placed at.  A record can require more than
/// the types of its members imply, from an `aligned` attribute on the record
/// or on one of its fields, and neither raises the alignment of any type.
/// Only the record layout knows, so the member-derived value alone can be too
/// small.
uint64_t argumentAreaAlign(mlir::Type ty, mlir::ModuleOp modOp,
                           const mlir::DataLayout &dl) {
  uint64_t align = dl.getTypeABIAlignment(ty);
  if (auto recTy = mlir::dyn_cast<cir::RecordType>(ty))
    if (auto layout = cir::tryGetRecordLayout(modOp, recTy.getName()))
      align = std::max<uint64_t>(align, layout.getRecordAlign());
  return align;
}

mlir::Value roundPointerUpToAlignment(CIRBaseBuilderTy &b, mlir::Location loc,
                                      mlir::Value bytePtr, uint64_t align,
                                      const mlir::DataLayout &dl) {
  assert(mlir::cast<cir::PointerType>(bytePtr.getType()).getPointee() ==
             b.getUIntNTy(8) &&
         "the bump strides in bytes, so the pointee must be u8");
  assert(llvm::isPowerOf2_64(align) &&
         "mask rounding needs a power-of-two alignment");
  mlir::Value bumped =
      b.createPtrStride(loc, bytePtr, b.getSignedInt(loc, align - 1, 32));
  std::optional<uint64_t> indexWidth =
      dl.getTypeIndexBitwidth(bytePtr.getType());
  assert(indexWidth && "a pointer in the argument area has an index width");
  mlir::Value mask = b.getSignedInt(loc, -static_cast<int64_t>(align),
                                    static_cast<unsigned>(*indexWidth));
  return cir::PtrMaskOp::create(b, loc, bytePtr.getType(), bumped, mask);
}

/// Reads the argument's address out of the overflow area, which also advances
/// the cursor past the argument.
mlir::Value buildOverflowAddrAndAdvance(CIRBaseBuilderTy &b,
                                        const VAArgFetch &f) {
  cir::IntType byteTy = b.getUIntNTy(8);
  mlir::Value overflowP = b.createGetMember(
      f.loc, b.getPointerTo(f.vaFields[2]), f.valist, "overflow_arg_area", 2);
  mlir::Value overflow = b.createLoad(f.loc, overflowP);
  mlir::Value bytePtr = b.createPtrBitcast(overflow, byteTy);

  uint64_t tyAlign = argumentAreaAlign(f.resultTy, f.module, f.dl);
  if (tyAlign > 8)
    bytePtr = roundPointerUpToAlignment(b, f.loc, bytePtr, tyAlign, f.dl);

  uint64_t tySize = f.dl.getTypeSize(f.resultTy).getFixedValue();
  uint64_t stride = (tySize + 7) & ~UINT64_C(7);
  mlir::Value strideVal = b.getSignedInt(f.loc, stride, 32);
  mlir::Value next = b.createPtrStride(f.loc, bytePtr, strideVal);
  b.createStore(f.loc, next, overflowP);
  return bytePtr;
}

/// Loads the cursor offsets and builds the predicate that sends the fetch to
/// the register-save area.  Both offsets count from the start of that area, so
/// the integer limit is the six GP registers at 48 and the vector limit is
/// those plus the eight SSE registers at 176.
RegisterCursor buildRegisterGate(CIRBaseBuilderTy &b, const VAArgFetch &f,
                                 unsigned neededInt, unsigned neededSse) {
  RegisterCursor cursor;
  if (neededInt) {
    cursor.gpOffsetP = b.createGetMember(f.loc, b.getPointerTo(f.vaFields[0]),
                                         f.valist, "gp_offset", 0);
    cursor.gpOffset = b.createLoad(f.loc, cursor.gpOffsetP);
    mlir::Value limit =
        b.getConstantInt(f.loc, cursor.gpOffset.getType(), 48 - neededInt * 8);
    cursor.inRegs =
        b.createCompare(f.loc, cir::CmpOpKind::le, cursor.gpOffset, limit);
  }
  if (neededSse) {
    cursor.fpOffsetP = b.createGetMember(f.loc, b.getPointerTo(f.vaFields[1]),
                                         f.valist, "fp_offset", 1);
    cursor.fpOffset = b.createLoad(f.loc, cursor.fpOffsetP);
    mlir::Value limit = b.getConstantInt(f.loc, cursor.fpOffset.getType(),
                                         176 - neededSse * 16);
    mlir::Value fitsInFp =
        b.createCompare(f.loc, cir::CmpOpKind::le, cursor.fpOffset, limit);
    cursor.inRegs = cursor.inRegs
                        ? b.createLogicalAnd(f.loc, cursor.inRegs, fitsInFp)
                        : fitsInFp;
  }
  return cursor;
}

/// Copies each half of a non-contiguous pair out of the register-save area
/// into \p regPairTemp, laid out as the coerced pair.
void reassembleRegisterPair(CIRBaseBuilderTy &b, mlir::Location loc,
                            const ArgClassification &ac,
                            const RegisterCursor &cursor,
                            const std::array<bool, 2> &pairIsSse,
                            mlir::Value regSaveArea, mlir::Value regPairTemp) {
  auto pairTy = mlir::cast<cir::RecordType>(ac.coercedType);
  // Both halves of an all-SSE pair sit in 16-byte slots, which classic
  // CodeGen tells the load about.  It leaves a mixed pair to the element's
  // own alignment, so match that rather than claiming the slot there.
  bool bothSse = pairIsSse[0] && pairIsSse[1];
  // Track how many of each class came before, since a slot is reached from
  // its class's own cursor.
  unsigned seenOfClass[2] = {0, 0};
  for (unsigned i = 0; i < 2; ++i) {
    bool isSse = pairIsSse[i];
    mlir::Value base = isSse ? cursor.fpOffset : cursor.gpOffset;
    unsigned regSize = isSse ? 16 : 8;
    unsigned prior = seenOfClass[isSse];
    ++seenOfClass[isSse];
    mlir::Value off = base;
    if (prior) {
      off = b.createAdd(loc, base,
                        b.getConstantInt(loc, base.getType(), prior * regSize));
    }
    mlir::Value src = b.createPtrStride(loc, regSaveArea, off);
    mlir::Type elemTy = pairTy.getElementType(i);
    mlir::Value elemPtr = b.createPtrBitcast(src, elemTy);
    mlir::Value val = bothSse ? b.createAlignedLoad(loc, elemPtr, 16)
                              : b.createLoad(loc, elemPtr);
    b.createStore(
        loc, val,
        b.createGetMember(loc, b.getPointerTo(elemTy), regPairTemp, "", i));
  }
}

/// The bytes carried by the registers of this fetch's one class.
uint64_t registerSlotSize(unsigned neededInt, unsigned neededSse) {
  assert(!(neededInt && neededSse) &&
         "a fetch needing both classes is a pair, reassembled elsewhere");
  return neededSse ? neededSse * 16 : neededInt * 8;
}

/// Whether the register cannot be read in place, either because it carries
/// less than the whole result or because its slot is under-aligned for it.
bool needsTempCopy(const VAArgFetch &f, unsigned neededInt,
                   unsigned neededSse) {
  uint64_t tySize = f.dl.getTypeSize(f.resultTy).getFixedValue();
  if (f.ac.coercedType &&
      (f.ac.directOffset || registerSlotSize(neededInt, neededSse) < tySize))
    return true;
  // A slot is only as aligned as its class, 8 for a GP register and 16 for an
  // SSE one, so a fetch the ABI places more strictly is copied through a temp.
  uint64_t slotAlign = neededSse ? 16 : 8;
  return argumentAreaAlign(f.resultTy, f.module, f.dl) > slotAlign;
}

/// Copies what the registers carry into \p temp, which is the size of the
/// whole result, and returns the address to read the result from.
mlir::Value copyRegisterToTemp(CIRBaseBuilderTy &b, mlir::Location loc,
                               const VAArgFetch &f, mlir::Value regAddr,
                               mlir::Value temp, unsigned neededInt,
                               unsigned neededSse) {
  cir::IntType byteTy = b.getUIntNTy(8);
  uint64_t tySize = f.dl.getTypeSize(f.resultTy).getFixedValue();

  if (f.ac.coercedType &&
      (f.ac.directOffset || registerSlotSize(neededInt, neededSse) < tySize)) {
    // The registers carry less than the whole result, either because the
    // eightbytes below directOffset hold no field or because the result is
    // wider than the registers carrying it.  Copy only what they carry, so
    // that reading the temp cannot run on into the neighboring slot.
    mlir::Value val = b.createAlignedLoad(
        loc, b.createPtrBitcast(regAddr, f.ac.coercedType), 8);
    mlir::Value dst = temp;
    if (f.ac.directOffset) {
      dst = b.createPtrStride(loc, b.createPtrBitcast(temp, byteTy),
                              b.getSignedInt(loc, f.ac.directOffset, 32));
    }
    b.createStore(loc, val, b.createPtrBitcast(dst, f.ac.coercedType));
    return b.createPtrBitcast(temp, byteTy);
  }

  mlir::Value val =
      b.createAlignedLoad(loc, b.createPtrBitcast(regAddr, f.resultTy), 8);
  b.createStore(loc, val, temp);
  return b.createPtrBitcast(temp, byteTy);
}

void advanceRegisterCursors(CIRBaseBuilderTy &b, mlir::Location loc,
                            const RegisterCursor &cursor, unsigned neededInt,
                            unsigned neededSse) {
  if (neededInt) {
    b.createStore(loc,
                  b.createAdd(loc, cursor.gpOffset,
                              b.getConstantInt(loc, cursor.gpOffset.getType(),
                                               neededInt * 8)),
                  cursor.gpOffsetP);
  }
  if (neededSse) {
    b.createStore(loc,
                  b.createAdd(loc, cursor.fpOffset,
                              b.getConstantInt(loc, cursor.fpOffset.getType(),
                                               neededSse * 16)),
                  cursor.fpOffsetP);
  }
}

} // namespace

mlir::LogicalResult
CIRABIRewriteContext::rewriteVAArg(mlir::Operation *vaArgOp,
                                   const ArgClassification &ac,
                                   mlir::OpBuilder &opBuilder) {
  auto op = mlir::cast<cir::VAArgOp>(vaArgOp);
  CIRBaseBuilderTy builder(opBuilder);
  mlir::Location loc = op.getLoc();
  mlir::Type resultTy = op.getType();
  mlir::Value valist = op.getArgList();

  // An ignored type travels in no register and no stack slot, so the fetch
  // reads nothing and must leave the cursor unchanged.  The value holds no
  // bytes, so poison stands in for it.
  if (ac.kind == ArgKind::Ignore) {
    builder.setInsertionPoint(op);
    op.getResult().replaceAllUsesWith(
        createIgnoredValue(builder, loc, resultTy));
    op->erase();
    return mlir::success();
  }

  if (ac.kind == ArgKind::Indirect && !ac.byVal)
    return reportVAArgNYI(op, "a non-trivially-copyable type");

  // neededInt counts 8-byte integer slots and neededSse counts 16-byte vector
  // slots.  Zero of both means the type travels in memory and is read
  // straight from the overflow area.
  unsigned neededInt = ac.neededIntRegs;
  unsigned neededSse = ac.neededSseRegs;

  // Which coerced-pair element (0 = low eightbyte, 1 = high) is SSE rather
  // than INTEGER class.  Only meaningful when isRegPair is set.
  std::array<bool, 2> pairIsSse = {false, false};
  bool isRegPair = false;
  if (ac.kind == ArgKind::Direct && neededInt + neededSse == 2 &&
      ac.coercedType) {
    if (classifyRegisterPair(op, ac, pairIsSse, isRegPair).failed())
      return mlir::failure();
  }

  auto vaListRecTy = mlir::dyn_cast<cir::RecordType>(
      mlir::cast<cir::PointerType>(valist.getType()).getPointee());
  if (!vaListRecTy || vaListRecTy.getNumElements() != 4) {
    return reportVAArgNYI(op,
                          "a va_list that is not the four-field gp_offset / "
                          "fp_offset / overflow_arg_area / reg_save_area "
                          "cursor");
  }

  const VAArgFetch fetch{loc, valist, vaListRecTy.getMembers(), resultTy, ac,
                         dl,  module};
  cir::IntType byteTy = builder.getUIntNTy(8);

  builder.setInsertionPoint(op);

  mlir::Value addr;
  if (neededInt == 0 && neededSse == 0) {
    addr = buildOverflowAddrAndAdvance(builder, fetch);
  } else {
    RegisterCursor cursor =
        buildRegisterGate(builder, fetch, neededInt, neededSse);

    // A two-eightbyte pair that is purely INTEGER class is contiguous in the
    // register-save area, since GP slots are 8-byte packed, so the address of
    // its low eightbyte is already the address of the whole value.  A pure
    // SSE pair or a mixed pair is not contiguous, since SSE slots are 16-byte
    // spaced and a mixed pair's halves live in disjoint areas, so each half is
    // copied into a temp laid out as the coerced pair.
    bool pairNeedsReassembly = isRegPair && neededSse != 0;

    // Both temps are allocated here, since a ternary arm yields their address.
    mlir::Value regPairTemp;
    if (pairNeedsReassembly) {
      // The temp is written one member at a time through the coerced pair, so
      // it has to meet that type's alignment, and read back as the result, so
      // it has to meet the result's alignment too.
      regPairTemp = builder.createAlloca(
          loc, builder.getPointerTo(ac.coercedType), "vaarg.reg",
          clang::CharUnits::fromQuantity(
              std::max(dl.getTypeABIAlignment(ac.coercedType),
                       argumentAreaAlign(resultTy, module, dl))));
    }

    mlir::Value regTemp;
    bool copyThroughTemp =
        !pairNeedsReassembly && needsTempCopy(fetch, neededInt, neededSse);
    if (copyThroughTemp) {
      regTemp =
          builder.createAlloca(loc, builder.getPointerTo(resultTy), "vaarg.reg",
                               clang::CharUnits::fromQuantity(
                                   argumentAreaAlign(resultTy, module, dl)));
    }

    addr = cir::TernaryOp::create(
               builder, loc, cursor.inRegs,
               /*trueBuilder=*/
               [&](mlir::OpBuilder &ob, mlir::Location l) {
                 CIRBaseBuilderTy b(ob);
                 mlir::Value regSaveArea = b.createLoad(
                     l, b.createGetMember(l, b.getPointerTo(fetch.vaFields[3]),
                                          valist, "reg_save_area", 3));
                 regSaveArea = b.createPtrBitcast(regSaveArea, byteTy);

                 mlir::Value regAddr;
                 if (pairNeedsReassembly) {
                   reassembleRegisterPair(b, l, ac, cursor, pairIsSse,
                                          regSaveArea, regPairTemp);
                   regAddr = b.createPtrBitcast(regPairTemp, byteTy);
                 } else {
                   mlir::Value off =
                       neededSse ? cursor.fpOffset : cursor.gpOffset;
                   regAddr = b.createPtrStride(l, regSaveArea, off);
                   if (copyThroughTemp) {
                     regAddr = copyRegisterToTemp(b, l, fetch, regAddr, regTemp,
                                                  neededInt, neededSse);
                   }
                 }

                 advanceRegisterCursors(b, l, cursor, neededInt, neededSse);
                 cir::YieldOp::create(b, l, regAddr);
               },
               /*falseBuilder=*/
               [&](mlir::OpBuilder &ob, mlir::Location l) {
                 CIRBaseBuilderTy b(ob);
                 mlir::Value memAddr = buildOverflowAddrAndAdvance(b, fetch);
                 cir::YieldOp::create(b, l, memAddr);
               })
               .getResult();
  }

  // Every path above places the result at least this well aligned.
  mlir::Value result =
      builder.createAlignedLoad(loc, builder.createPtrBitcast(addr, resultTy),
                                argumentAreaAlign(resultTy, module, dl));
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return mlir::success();
}
