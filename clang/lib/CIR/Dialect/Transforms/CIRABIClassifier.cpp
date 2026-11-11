//===- CIRABIClassifier.cpp - ABI classification for CIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the classification shared by the targets that classify
// through the LLVM ABI Lowering Library, and the two test drivers that stand
// in for a target.
//
//===----------------------------------------------------------------------===//

#include "CIRABIClassifier.h"
#include "TargetLowering/CIRABIRewriteContext.h"

#include "mlir/ABI/Targets/Test/TestTarget.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

//===----------------------------------------------------------------------===//
// LLVM ABI Lowering Library classifier bridge
//
// Maps CIR types to llvm::abi::Type, runs the target's classifier from the
// LLVM ABI Lowering Library, and converts the result back into the
// dialect-agnostic mlir::abi::FunctionClassification that CIRABIRewriteContext
// consumes.
//
// isSupportedType says which CIR types the bridge handles, and a signature
// naming any other fails the pass instead of being misclassified.
//===----------------------------------------------------------------------===//

/// Whether a struct's declared argument-passing kind (from the module's
/// record-layout metadata) allows it to be passed in registers.  A record with
/// no layout entry (e.g. an anonymous struct) has no C++ non-trivial reason to
/// be forced to memory, so it defaults to can-pass-in-registers.
static bool recordCanPassInRegs(ModuleOp modOp, cir::RecordType recTy) {
  auto layout = cir::tryGetRecordLayout(modOp, recTy.getName());
  if (!layout)
    return true;
  return layout.getArgPassingKind() == cir::ArgPassingKind::CanPassInRegs;
}

static bool recordIsTransparentUnion(ModuleOp modOp, cir::RecordType recTy) {
  auto layout = cir::tryGetRecordLayout(modOp, recTy.getName());
  return layout && layout.getTransparentUnion();
}

/// Read from the layout, which also covers a last field that has one.
static bool recordHasFlexibleArrayMember(ModuleOp modOp,
                                         cir::RecordType recTy) {
  auto layout = cir::tryGetRecordLayout(modOp, recTy.getName());
  return layout && layout.getFlexibleArrayMember();
}

/// Whether the classifier could give this type the SSEUP class, looking
/// through arrays and records at the types they hold.
static bool mayReachSseUp(mlir::Type ty, const DataLayout &dl) {
  // The classifier sizes a vector with its width rounded up to a power of two.
  if (isa<cir::VectorType>(ty))
    return llvm::PowerOf2Ceil(dl.getTypeSizeInBits(ty).getFixedValue()) >= 128;
  if (auto fpTy = dyn_cast<cir::FPTypeInterface>(ty))
    return &fpTy.getFloatSemantics() == &llvm::APFloat::IEEEquad();
  if (auto arrTy = dyn_cast<cir::ArrayType>(ty))
    return mayReachSseUp(arrTy.getElementType(), dl);
  if (auto recTy = dyn_cast<cir::RecordType>(ty))
    return recTy.isComplete() &&
           llvm::any_of(recTy.getMembers(),
                        [&](mlir::Type m) { return mayReachSseUp(m, dl); });
  // The rest classify integer, SSE or memory.  A complex is SSE at float and
  // double width and memory above that, so it needs no walk of its own.
  assert((isa<cir::IntType, cir::BoolType, cir::PointerType, cir::VPtrType,
              cir::VoidType, cir::ComplexType, cir::BitFieldType>(ty)) &&
         "unhandled type in the SSEUP walk");
  return false;
}

/// Whether the record's size has no SSEUP coerce.  That coerce is a vector as
/// wide as the record, so only 128, 256 and 512 bits have one.  A record past
/// 512 bits classifies memory before any coerce is asked for.
static bool sseUpCoerceSizeUnsupported(uint64_t recordBits,
                                       llvm::ArrayRef<mlir::Type> members,
                                       const DataLayout &dl) {
  if (recordBits == 128 || recordBits == 256 || recordBits == 512)
    return false;
  if (recordBits < 128 || recordBits > 512)
    return false;
  return llvm::any_of(members,
                      [&](mlir::Type m) { return mayReachSseUp(m, dl); });
}

/// Whether a member is an empty record, looking through arrays, since an array
/// of empty records supplies no bytes either.
static bool memberIsEmptyRecord(mlir::Type ty) {
  while (auto arrTy = dyn_cast<cir::ArrayType>(ty))
    ty = arrTy.getElementType();
  auto recTy = dyn_cast<cir::RecordType>(ty);
  return recTy && recTy.isEmptyForABI();
}

/// A record's declared alignment, which the ABI uses for the byval and sret
/// alignment of an indirect argument.  DataLayout derives alignment from the
/// members, so it cannot see `__attribute__((aligned(N)))`.  The declared value
/// comes from the module's record-layout metadata instead.  CIRGen emits an
/// entry for every record it names, so the computed fallback only serves
/// hand-written CIR.
static llvm::Align recordDeclaredAlign(ModuleOp modOp, cir::RecordType recTy,
                                       const DataLayout &dl) {
  auto layout = cir::tryGetRecordLayout(modOp, recTy.getName());
  if (!layout)
    return llvm::Align(dl.getTypeABIAlignment(recTy));
  return llvm::Align(layout.getRecordAlign());
}

/// A record's unadjusted alignment, the alignment its members give it before
/// an alignment attribute on the record raises it.  Like the declared
/// alignment, it comes from the module's record-layout metadata, and the
/// computed fallback only serves hand-written CIR.
static llvm::Align recordUnadjustedAlign(ModuleOp modOp, cir::RecordType recTy,
                                         const DataLayout &dl) {
  auto layout = cir::tryGetRecordLayout(modOp, recTy.getName());
  if (!layout)
    return llvm::Align(dl.getTypeABIAlignment(recTy));
  return llvm::Align(layout.getUnadjustedAlign());
}

/// The CIR types the bridge handles.  Scalars: an integer up to 128
/// bits (including `__int128`), a `_BitInt` of any width, pointer, vtable
/// pointer, bool, void, or any floating-point type.  Aggregates: a complete
/// struct or union whose members are all themselves supported, or an array
/// of a supported element type.  Also a `_Complex`, or a fixed-width vector,
/// of a supported element type.  Everything else is reported NYI at the
/// reject() choke point in classifySignature.
static bool isSupportedType(mlir::Type ty, const DataLayout &dl) {
  // A pointer is only handled in the default address space (null) or an
  // already-lowered target address space.  A LangAddressSpaceAttr must be
  // lowered before this pass, so reject it rather than silently dropping it.
  if (auto ptrTy = dyn_cast<cir::PointerType>(ty))
    return !ptrTy.getAddrSpace() ||
           mlir::isa<cir::TargetAddressSpaceAttr>(ptrTy.getAddrSpace());
  // cir::VPtrType carries no address-space parameter yet, so its target
  // address space cannot be checked here even though it is not always the
  // default one.
  if (isa<cir::VPtrType>(ty)) {
    assert(!cir::MissingFeatures::addressSpace());
    return true;
  }
  if (isa<cir::VoidType, cir::BoolType>(ty))
    return true;
  // Every CIR floating-point type carries the semantics the classifier
  // switches on, so all of them are handled.
  if (isa<cir::FPTypeInterface>(ty))
    return true;
  if (auto intTy = dyn_cast<cir::IntType>(ty)) {
    // Integers up to 64 bits, __int128, and _BitInt of any width are
    // handled: the classifier extends a width below 32, widens 33 through 63
    // to i64, coerces 65 through 127 to a {i64, i64} pair, and passes 32, 64,
    // and 128 in the natural type.  A _BitInt wider than 128 bits classifies
    // Indirect, the same path an oversized aggregate already takes.
    // Non-_BitInt intermediate widths (65..127) do not arise from C and stay
    // rejected.
    if (intTy.getIsBitInt())
      return true;
    return intTy.getWidth() <= 64 || intTy.getWidth() == 128;
  }
  if (auto complexTy = dyn_cast<cir::ComplexType>(ty))
    return isSupportedType(complexTy.getElementType(), dl);
  if (auto vecTy = dyn_cast<cir::VectorType>(ty)) {
    // A scalable vector has no fixed size, and no target that classifies
    // through the bridge passes one yet.
    if (vecTy.isScalable())
      return false;
    // mapCIRType maps a bool as a one-bit integer, so a bool vector is sized
    // one bit per element, as clang packs it.  Vectors of a _BitInt whose
    // width is not a whole number of bytes are not handled yet.
    mlir::Type elemTy = vecTy.getElementType();
    if (auto elemInt = dyn_cast<cir::IntType>(elemTy)) {
      if (elemInt.getWidth() % 8)
        return false;
    } else if (!isa<cir::FPTypeInterface, cir::BoolType>(elemTy)) {
      return false;
    }
    return isSupportedType(elemTy, dl);
  }
  if (auto arrTy = dyn_cast<cir::ArrayType>(ty))
    return isSupportedType(arrTy.getElementType(), dl);
  // A bit-field is classified from the type it was declared with and the bits
  // it holds, and is loaded through its access unit, so both types have to be
  // ones the bridge handles.  A zero-width bit-field has no unit of its own.
  if (auto bfTy = dyn_cast<cir::BitFieldType>(ty)) {
    if (mlir::Type storageTy = bfTy.getStorageType())
      if (!isSupportedType(storageTy, dl))
        return false;
    return llvm::all_of(bfTy.getFields(), [&](cir::BitFieldDeclAttr decl) {
      return isSupportedType(decl.getDeclaredType(), dl);
    });
  }
  if (auto recTy = dyn_cast<cir::RecordType>(ty)) {
    // An incomplete record has no layout to classify.
    if (!recTy.isComplete())
      return false;
    // The members are checked first so that everything below reasons only
    // about types the bridge handles.
    if (!llvm::all_of(recTy.getMembers(),
                      [&](mlir::Type m) { return isSupportedType(m, dl); }))
      return false;
    if (recTy.isUnion()) {
      llvm::ArrayRef<mlir::Type> members = recTy.getMembers();
      uint64_t recordBits = dl.getTypeSizeInBits(recTy).getFixedValue();
      // Refuse a union that could reach SSEUP at a size no coerce exists for.
      // The classifier asserts on those rather than returning a coerce.
      if (sseUpCoerceSizeUnsupported(recordBits, members, dl))
        return false;
      // A member-less union is all padding, which classifies Ignore up to two
      // eightbytes.  Past that SysV says MEMORY regardless of content, and
      // there is no member here to build the Indirect coercion from.
      if (members.empty() && recordBits > 128)
        return false;
      // A `_BitInt` access unit past an eightbyte has byte-array storage in
      // CIR and integer storage in classic CodeGen, so the narrowing walk
      // finds an i8 in the second eightbyte where classic keeps an i64.
      if (llvm::any_of(members, [&](mlir::Type m) {
            auto bfTy = dyn_cast<cir::BitFieldType>(m);
            if (!bfTy || dl.getTypeSizeInBits(bfTy).getFixedValue() <= 64)
              return false;
            return llvm::any_of(bfTy.getFields(), [](cir::BitFieldDeclAttr d) {
              auto intTy = dyn_cast<cir::IntType>(d.getDeclaredType());
              return intTy && intTy.getIsBitInt();
            });
          }))
        return false;
    } else {
      // A struct's `empty` member that occupies bytes is later read as an
      // unnamed bit-field.  An empty-for-ABI record occupies its own padding,
      // and reading that as a unit would hand the classifier storage classic
      // CodeGen ignores.  A zero-sized one is dropped before classification,
      // so a misaligned one never reaches the rule that sends its record to
      // memory.  A union needs neither rule: mapCIRType skips such a variant
      // instead of reading it as a unit, and every variant sits at offset
      // zero.
      for (auto [idx, memberTy, kind] :
           llvm::enumerate(recTy.getMembers(), recTy.getMemberKinds())) {
        if (kind != cir::RecordMemberKind::Empty)
          continue;
        if (dl.getTypeSizeInBits(memberTy).getFixedValue()) {
          if (memberIsEmptyRecord(memberTy))
            return false;
        } else if (recTy.getElementOffset(dl, idx) %
                   dl.getTypeABIAlignment(memberTy)) {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

/// Convert an llvm::abi::Type coercion type back to a CIR type.
static mlir::Type abiTypeToCIR(const llvm::abi::Type *ty, MLIRContext *ctx) {
  if (!ty)
    return nullptr;
  return llvm::TypeSwitch<const llvm::abi::Type *, mlir::Type>(ty)
      .Case(
          [&](const llvm::abi::VoidType *) { return cir::VoidType::get(ctx); })
      .Case([&](const llvm::abi::IntegerType *intTy) -> mlir::Type {
        // mapCIRType maps a bool as a one-bit integer.
        if (intTy->isBool())
          return cir::BoolType::get(ctx);
        return cir::IntType::get(ctx, intTy->getSizeInBits().getFixedValue(),
                                 intTy->isSigned(), intTy->isBitInt());
      })
      .Case([&](const llvm::abi::FloatType *fltTy) {
        return cir::getFloatingPointType(*fltTy->getSemantics(), ctx);
      })
      .Case([&](const llvm::abi::PointerType *ptrTy) {
        auto voidTy = cir::VoidType::get(ctx);
        if (unsigned as = ptrTy->getAddrSpace())
          return cir::PointerType::get(
              voidTy, cir::TargetAddressSpaceAttr::get(ctx, as));
        return cir::PointerType::get(voidTy);
      })
      .Case([&](const llvm::abi::VectorType *vecTy) -> mlir::Type {
        mlir::Type elemCIR = abiTypeToCIR(vecTy->getElementType(), ctx);
        if (!elemCIR)
          return nullptr;
        return cir::VectorType::getFixed(
            elemCIR, vecTy->getNumElements().getFixedValue());
      })
      .Case([&](const llvm::abi::ArrayType *arrTy) -> mlir::Type {
        mlir::Type elemCIR = abiTypeToCIR(arrTy->getElementType(), ctx);
        if (!elemCIR)
          return nullptr;
        return cir::ArrayType::get(elemCIR, arrTy->getNumElements());
      })
      .Case([&](const llvm::abi::RecordType *recTy) -> mlir::Type {
        SmallVector<mlir::Type> fieldTypes;
        fieldTypes.reserve(recTy->getFields().size());
        for (const auto &field : recTy->getFields()) {
          mlir::Type fieldCIR = abiTypeToCIR(field.FieldType, ctx);
          if (!fieldCIR)
            return nullptr;
          fieldTypes.push_back(fieldCIR);
        }
        // Coercion types are plain register tuples, not the source record.
        return cir::StructType::get(
            ctx, fieldTypes, /*packed=*/false,
            /*is_class=*/false, cir::RecordType::getAllDataKinds(fieldTypes));
      })
      .Default([](const llvm::abi::Type *) -> mlir::Type { return nullptr; });
}

/// Returns \p ty with a LongDoubleType vector element replaced by its
/// underlying format, which is the element abiTypeToCIR builds for the same
/// float.
static mlir::Type unwrapLongDoubleVectorElement(mlir::Type ty) {
  if (auto vecTy = dyn_cast<cir::VectorType>(ty)) {
    if (auto longDoubleTy =
            dyn_cast<cir::LongDoubleType>(vecTy.getElementType()))
      return cir::VectorType::get(longDoubleTy.getUnderlying(),
                                  vecTy.getSize());
  }
  return ty;
}

/// Map a CIR type to an llvm::abi::Type.  classifySignature pre-filters
/// the signature, so only the scalar and struct/array types handled here can
/// reach this function.
static const llvm::abi::Type *mapCIRType(mlir::Type type,
                                         mlir::abi::ABITypeMapper &typeMapper,
                                         const DataLayout &dl, ModuleOp modOp) {
  llvm::abi::TypeBuilder &tb = typeMapper.getTypeBuilder();
  return llvm::TypeSwitch<mlir::Type, const llvm::abi::Type *>(type)
      .Case([&](cir::IntType intTy) {
        return tb.getIntegerType(intTy.getWidth(),
                                 llvm::Align(dl.getTypeABIAlignment(type)),
                                 intTy.isSigned(), intTy.getIsBitInt());
      })
      .Case([&](cir::PointerType ptrTy) {
        unsigned addrSpace = 0;
        if (auto targetAsAttr =
                dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                    ptrTy.getAddrSpace()))
          addrSpace = targetAsAttr.getValue();
        // Any address-space attribute is a qualifier.
        bool hasAddressSpaceQualifier = ptrTy.getAddrSpace() != nullptr;
        return tb.getPointerType(dl.getTypeSizeInBits(type),
                                 llvm::Align(dl.getTypeABIAlignment(type)),
                                 addrSpace, hasAddressSpaceQualifier);
      })
      .Case([&](cir::VPtrType) {
        // cir::VPtrType carries no address-space parameter yet, so this
        // always maps into the default one until that gap closes.
        assert(!cir::MissingFeatures::addressSpace());
        return tb.getPointerType(dl.getTypeSizeInBits(type),
                                 llvm::Align(dl.getTypeABIAlignment(type)));
      })
      .Case([&](cir::BoolType) {
        // A bool is a one-bit integer to the classifier, as QualTypeMapper
        // maps it, so a bool vector is sized one bit per element.
        return tb.getIntegerType(1, llvm::Align(dl.getTypeABIAlignment(type)),
                                 /*Signed=*/false);
      })
      .Case([&](cir::VoidType) { return tb.getVoidType(); })
      .Case([&](cir::FPTypeInterface fpTy) {
        // LongDoubleType reports its underlying format's semantics, so the
        // classifier sees x87 or IEEE quad rather than the wrapper.
        return tb.getFloatType(fpTy.getFloatSemantics(),
                               llvm::Align(dl.getTypeABIAlignment(type)));
      })
      .Case([&](cir::ComplexType complexTy) {
        return tb.getComplexType(
            mapCIRType(complexTy.getElementType(), typeMapper, dl, modOp),
            llvm::Align(dl.getTypeABIAlignment(type)));
      })
      .Case([&](cir::VectorType vecTy) {
        // isSupportedType rejects a scalable vector, so the element count is
        // always fixed here.
        return tb.getVectorType(
            mapCIRType(vecTy.getElementType(), typeMapper, dl, modOp),
            vecTy.getSize(), llvm::Align(dl.getTypeABIAlignment(type)));
      })
      .Case([&](cir::ArrayType arrTy) {
        const llvm::abi::Type *elemAbi =
            mapCIRType(arrTy.getElementType(), typeMapper, dl, modOp);
        return tb.getArrayType(elemAbi, arrTy.getSize(),
                               dl.getTypeSizeInBits(type).getFixedValue());
      })
      .Case([&](cir::RecordType recTy) -> const llvm::abi::Type * {
        llvm::abi::RecordFlags flags = llvm::abi::RecordFlags::None;
        if (recordCanPassInRegs(modOp, recTy))
          flags = flags | llvm::abi::RecordFlags::CanPassInRegisters;
        if (recordIsTransparentUnion(modOp, recTy))
          flags = flags | llvm::abi::RecordFlags::IsTransparent;
        if (recordHasFlexibleArrayMember(modOp, recTy))
          flags = flags | llvm::abi::RecordFlags::HasFlexibleArrayMember;
        llvm::TypeSize sizeBits = llvm::TypeSize::getFixed(
            dl.getTypeSizeInBits(type).getFixedValue());
        llvm::Align align = recordDeclaredAlign(modOp, recTy, dl);
        llvm::Align unadjustedAlign = recordUnadjustedAlign(modOp, recTy, dl);

        // Mapped with no fields, an empty record reaches Ignore on its own.
        // The size still matters, since a target may pass a large empty
        // record in memory whatever its content.
        if (recTy.isEmptyForABI())
          return tb.getRecordType(
              /*Fields=*/{}, sizeBits, align, unadjustedAlign,
              llvm::abi::StructPacking::Default,
              /*BaseClasses=*/{}, /*VirtualBaseClasses=*/{}, flags);

        SmallVector<llvm::abi::FieldInfo> fields;
        fields.reserve(recTy.getMembers().size());

        // The classifier reads one entry per source bit-field, so a unit is
        // spread back out into the fields it holds.  The targets classified
        // here are little-endian, so the unit's first byte holds the first of
        // them, and the rest follow at the widths ahead of them.
        //
        // A coercion for a register is built from the type at the first byte
        // the register holds, where the unit's first field sits, so that
        // field's entry carries the unit rather than its own declared type.
        // The bits holding user data are a separate question, and the answer
        // is the extent of the type each field was declared with, which the
        // classifier reads through to decide whether a narrower coercion
        // would drop something.  One entry cannot answer both, so a unit
        // narrower than its first field's declaration takes an extra entry for
        // that declaration, holding no bits of its own.  A zero-width
        // bit-field has no unit and so contributes only the declaration.
        //
        // A width of zero is how the classifier recognizes the entries that
        // take no part in assigning register classes, so the width goes
        // through as declared.
        auto addAccessUnit = [&](cir::BitFieldType bfTy,
                                 uint64_t unitOffsetBits) {
          mlir::Type unitTy = bfTy.getStorageType();
          uint64_t unitBits =
              unitTy ? dl.getTypeSizeInBits(unitTy).getFixedValue() : 0;
          for (auto [idx, decl] : llvm::enumerate(bfTy.getFields())) {
            const uint64_t offsetBits =
                unitOffsetBits + bfTy.getFieldBitOffset(idx);
            const bool holdsUnit = idx == 0 && unitTy;
            if (holdsUnit)
              fields.push_back(llvm::abi::FieldInfo(
                  mapCIRType(unitTy, typeMapper, dl, modOp), offsetBits,
                  /*IsBitField=*/true, decl.getWidth(),
                  /*IsUnnamedBitField=*/decl.getIsUnnamed()));
            mlir::Type declaredTy = decl.getDeclaredType();
            if (!holdsUnit ||
                dl.getTypeSizeInBits(declaredTy).getFixedValue() > unitBits)
              fields.push_back(llvm::abi::FieldInfo(
                  mapCIRType(declaredTy, typeMapper, dl, modOp), offsetBits,
                  /*IsBitField=*/true, holdsUnit ? 0 : decl.getWidth(),
                  /*IsUnnamedBitField=*/holdsUnit || decl.getIsUnnamed()));
          }
        };

        // The size passed here spans the tail padding, so the classifier
        // covers the whole union rather than just the member it reduces the
        // union to.
        if (recTy.isUnion()) {
          // Classify only the variants that supply data, judged by each
          // variant's own storage rather than by its mark.  One supplying
          // none is skipped so it does not appear as a spurious argument.
          // Whether the record is empty at all is settled from the marks
          // before this.  Every variant starts at offset zero, so a
          // bit-field variant needs no offset of its own.
          assert(!llvm::is_contained(recTy.getMemberKinds(),
                                     cir::RecordMemberKind::Pad) &&
                 "a union member cannot be marked pad");
          for (mlir::Type variantTy : recTy.getMembers()) {
            if (auto bfTy = dyn_cast<cir::BitFieldType>(variantTy)) {
              // A variant that is an unnamed bit-field holds data all the
              // same, so it is flagged rather than skipped, as in the struct
              // path below.
              addAccessUnit(bfTy, /*unitOffsetBits=*/0);
              continue;
            }
            if (dl.getTypeSizeInBits(variantTy).getFixedValue() == 0 ||
                memberIsEmptyRecord(variantTy))
              continue;
            fields.push_back(llvm::abi::FieldInfo(
                mapCIRType(variantTy, typeMapper, dl, modOp)));
          }
          return tb.getUnionType(fields, sizeBits, align, unadjustedAlign,
                                 llvm::abi::StructPacking::Default, flags);
        }

        // Padding is dropped, so the classifier sees its bytes as holding
        // nothing, while the record's full size set above still spans them.  An
        // unnamed bit-field is flagged rather than dropped, since classic
        // CodeGen ignores it when assigning register classes but counts it
        // when choosing the coerce type, and a member's presence alone cannot
        // say both.
        for (auto [idx, fieldTy, kind] :
             llvm::enumerate(recTy.getMembers(), recTy.getMemberKinds())) {
          if (kind == cir::RecordMemberKind::Pad)
            continue;
          if (auto bfTy = dyn_cast<cir::BitFieldType>(fieldTy)) {
            addAccessUnit(bfTy, recTy.getElementOffset(dl, idx) * 8);
            continue;
          }
          // An `empty` member that is not a bit-field still stands for bytes
          // no field of the source reads, so it is flagged the same way.
          bool isUnnamed = kind == cir::RecordMemberKind::Empty;
          uint64_t widthBits = dl.getTypeSizeInBits(fieldTy).getFixedValue();
          if (isUnnamed && widthBits == 0)
            continue;
          assert((!isUnnamed || !memberIsEmptyRecord(fieldTy)) &&
                 "an empty-for-ABI member must not reach the classifier as an "
                 "unnamed bit-field");
          fields.push_back(llvm::abi::FieldInfo(
              mapCIRType(fieldTy, typeMapper, dl, modOp),
              recTy.getElementOffset(dl, idx) * 8,
              /*IsBitField=*/isUnnamed, isUnnamed ? widthBits : 0,
              /*IsUnnamedBitField=*/isUnnamed));
        }

        return tb.getRecordType(fields, sizeBits, align, unadjustedAlign,
                                llvm::abi::StructPacking::Default,
                                /*BaseClasses=*/{}, /*VirtualBaseClasses=*/{},
                                flags);
      })
      .Default([](mlir::Type) -> const llvm::abi::Type * {
        llvm_unreachable(
            "mapCIRType: type not pre-filtered by classifySignature");
      });
}

#ifndef NDEBUG
/// Whether \p coercedAbiTy holds the same value as the scalar \p origTy: an
/// integer of the same width, a float with the same semantics, or a pointer.
/// Used only in an assert in convertDirectArgInfo.
static bool coerceKeepsScalarValue(mlir::Type origTy,
                                   const llvm::abi::Type *coercedAbiTy) {
  if (auto intTy = dyn_cast<cir::IntType>(origTy)) {
    const auto *abiInt = dyn_cast<llvm::abi::IntegerType>(coercedAbiTy);
    return abiInt &&
           abiInt->getSizeInBits().getFixedValue() == intTy.getWidth();
  }
  if (isa<cir::BoolType>(origTy))
    return isa<llvm::abi::IntegerType>(coercedAbiTy);
  if (auto fpTy = dyn_cast<cir::FPTypeInterface>(origTy)) {
    const auto *abiFloat = dyn_cast<llvm::abi::FloatType>(coercedAbiTy);
    return abiFloat && abiFloat->getSemantics() == &fpTy.getFloatSemantics();
  }
  if (isa<cir::PointerType, cir::VPtrType>(origTy))
    return isa<llvm::abi::PointerType>(coercedAbiTy);
  return false;
}
#endif

/// Convert a Direct llvm::abi::ArgInfo.  A null coercedType in the result
/// means the value is passed unchanged in its own type.  Otherwise the
/// rewriter converts the value to coercedType, and when that is a struct and
/// canFlatten is set, passes each field of the struct as a separate argument.
static mlir::FailureOr<ArgClassification>
convertDirectArgInfo(const llvm::abi::ArgInfo &info, MLIRContext *ctx,
                     mlir::Type origTy) {
  assert(origTy && "a Direct value must have a type");
  unsigned offset = info.getDirectOffset();
  llvm::MaybeAlign align = info.getDirectAlign();
  auto passUnchanged = [&] {
    return ArgClassification::getDirect(/*coerced=*/nullptr, /*offset=*/0,
                                        align);
  };
  auto passCoerced = [&](mlir::Type coerced, unsigned coercedOffset) {
    ArgClassification classified =
        ArgClassification::getDirect(coerced, coercedOffset, align);
    classified.canFlatten = info.getCanBeFlattened();
    return classified;
  };

  const llvm::abi::Type *coercedAbiTy = info.getCoerceToType();
  if (!coercedAbiTy) {
    // A record the classifier allows to be flattened is coerced to its own
    // type, so that its fields are passed as separate arguments.  A union is
    // flattened as its LLVM type, the storage member and the tail padding.
    auto recTy = dyn_cast<cir::RecordType>(origTy);
    if (recTy && (recTy.isStruct() || recTy.isUnion()) &&
        info.getCanBeFlattened())
      return passCoerced(recTy, /*offset=*/0);
    return passUnchanged();
  }

  // A nonzero offset means the value's data starts that many bytes into its
  // storage.  The rewriter has no implementation of an offset for a coerced
  // struct, either when it passes the struct's fields as separate arguments
  // or when it fetches the struct in va_arg, so this is reported as NYI.
  bool coercedIsStruct = isa<llvm::abi::RecordType>(coercedAbiTy);
  if (offset && coercedIsStruct)
    return mlir::failure();

  // A scalar at byte 0 that is not split into a struct is coerced when the
  // classifier names a wider integer or a pointer in another address space.
  // Otherwise the classifier names a type of the same kind and width, and the
  // scalar passes unchanged.  The coerce type is not compared with the
  // scalar's own in CIR, since converting it to CIR loses a long double's
  // wrapper and a pointer's pointee.
  bool isScalar =
      !isa<cir::VectorType, cir::RecordType, cir::ArrayType, cir::ComplexType>(
          origTy);
  if (isScalar && !offset && !coercedIsStruct) {
    auto origPtr = dyn_cast<cir::PointerType>(origTy);
    const auto *coercedAbiPtr = dyn_cast<llvm::abi::PointerType>(coercedAbiTy);
    if (origPtr && coercedAbiPtr) {
      // Compare numbers, since an explicit address_space(0) keeps a
      // target_address_space(0) attribute that a plain pointer lacks.
      auto origAS = dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
          origPtr.getAddrSpace());
      if (coercedAbiPtr->getAddrSpace() == (origAS ? origAS.getValue() : 0))
        return passUnchanged();
      auto coercedPtr = cast<cir::PointerType>(abiTypeToCIR(coercedAbiTy, ctx));
      return passCoerced(cir::PointerType::get(origPtr.getPointee(),
                                               coercedPtr.getAddrSpace()),
                         /*offset=*/0);
    }
    auto origInt = dyn_cast<cir::IntType>(origTy);
    const auto *coercedAbiInt = dyn_cast<llvm::abi::IntegerType>(coercedAbiTy);
    bool widensInt =
        origInt && coercedAbiInt &&
        coercedAbiInt->getSizeInBits().getFixedValue() > origInt.getWidth();
    if (!widensInt) {
      // FIXME: A coerce type that changes the scalar's kind or width should
      // be handled like a wider integer.  The scalar is coerced to
      // abiTypeToCIR(coercedAbiTy), which the rewriter emits as a store of
      // the scalar and a load of the coerce type from the same memory.
      assert(coerceKeepsScalarValue(origTy, coercedAbiTy) &&
             "the classifier changed a scalar's kind or width");
      return passUnchanged();
    }
  }

  mlir::Type coerced = abiTypeToCIR(coercedAbiTy, ctx);
  if (!coerced)
    return mlir::failure();
  if (coerced == unwrapLongDoubleVectorElement(origTy)) {
    // ArgClassification has no way to represent a coercion to the value's own
    // type at an offset, since a coerced type equal to the value's own type
    // is treated as no coercion.  This is reported as NYI.
    if (offset)
      return mlir::failure();
    // A vector the classifier passes as itself needs no rewrite.
    if (isa<cir::VectorType>(origTy))
      return passUnchanged();
  }
  // An aggregate or _Complex is unpacked into the registers the coerced type
  // describes.  A vector or scalar is reinterpreted as the coerced type.
  return passCoerced(coerced, offset);
}

/// Convert an llvm::abi::ArgInfo into the ArgClassification consumed by
/// CIRABIRewriteContext.  Returns failure when the classifier picks a
/// coercion this bridge cannot represent, which the caller reports as NYI.
static mlir::FailureOr<ArgClassification>
convertABIArgInfo(const llvm::abi::ArgInfo &info, MLIRContext *ctx,
                  mlir::Type origTy) {
  if (info.isDirect())
    return convertDirectArgInfo(info, ctx, origTy);
  // An extended value is always read from byte 0 of its own storage, so
  // there is no offset to honor here.
  if (info.isExtend()) {
    assert(origTy && "an extended value must have a type");
    if (isa<cir::BoolType>(origTy))
      return ArgClassification::getExtend(nullptr, info.isSignExt());
    assert((isa<cir::IntType, cir::RecordType>(origTy)) &&
           "Extend is only expected for integers, bool, and transparent "
           "unions");
    mlir::Type extendedTy = abiTypeToCIR(info.getCoerceToType(), ctx);
    return ArgClassification::getExtend(extendedTy, info.isSignExt());
  }
  if (info.isIndirect())
    return ArgClassification::getIndirect(info.getIndirectAlign(),
                                          info.getIndirectByVal(),
                                          info.getIndirectAddrSpace());
  if (info.isIndirectAliased())
    return ArgClassification::getIndirectAliased(info.getIndirectAlign(),
                                                 info.getIndirectAddrSpace());
  // AArch64 pure scalable aggregates use CoerceAndExpand, but they aren't
  // supported yet.
  assert(!info.isCoerceAndExpand() && "CoerceAndExpand is not handled yet");
  assert(info.isIgnore() && "Unexpected classification");
  return ArgClassification::getIgnore();
}

/// Where \p fnTy's declared parameters end and its ellipsis arguments begin.
static llvm::abi::RequiredArgs requiredArgs(cir::FuncType fnTy) {
  if (!fnTy.isVarArg())
    return llvm::abi::RequiredArgs::All;
  return llvm::abi::RequiredArgs(fnTy.getNumInputs());
}

/// Classify a signature (return type + argument types) using the LLVM ABI
/// library.  Shared by the cir.func path, the variadic-call path and the
/// indirect-call path (the latter classifies from the callee function
/// pointer's pointee FuncType).  \p required marks where the declared
/// parameters in \p inputs end.  The classifier treats every argument past that
/// point as passed through an ellipsis.  \p callConv is the calling convention
/// the signature is classified under.  Returns failure and emits an NYI
/// error via \p emitError if the signature uses a type the bridge does not
/// handle yet.
mlir::FailureOr<FunctionClassification>
CIRABIClassifierCommon::classifySignature(
    mlir::Type retCIR, mlir::TypeRange inputs, llvm::abi::RequiredArgs required,
    const llvm::abi::TargetInfo &targetInfo, llvm::CallingConv::ID callConv,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  assert(retCIR && "signature return type must be non-null");
  assert((!required.allowsOptionalArgs() ||
          required.getNumRequiredArgs() <= inputs.size()) &&
         "declared parameters cannot outnumber the classified arguments");
  MLIRContext *ctx = modOp->getContext();
  bool voidRet = isa<cir::VoidType>(retCIR);

  auto reject = [&](mlir::Type t) -> bool {
    if (isSupportedType(t, dl))
      return false;
    emitError() << "calling-convention lowering not yet implemented for type "
                << t;
    return true;
  };
  if (!voidRet && reject(retCIR))
    return mlir::failure();
  for (mlir::Type a : inputs)
    if (reject(a))
      return mlir::failure();

  const llvm::abi::Type *retAbi =
      voidRet ? typeMapper.getTypeBuilder().getVoidType()
              : mapCIRType(retCIR, typeMapper, dl, modOp);
  SmallVector<const llvm::abi::Type *> argAbi;
  for (mlir::Type a : inputs)
    argAbi.push_back(mapCIRType(a, typeMapper, dl, modOp));

  std::unique_ptr<llvm::abi::FunctionInfo> fi =
      llvm::abi::FunctionInfo::create(callConv, retAbi, argAbi, required);
  targetInfo.computeInfo(*fi);

  // convertABIArgInfo returns failure when the classifier picks a coercion this
  // bridge cannot represent.
  auto nyiCoercion = [&](mlir::Type t) {
    emitError() << "calling-convention lowering not yet "
                   "implemented for the ABI coercion of type "
                << t;
  };

  FunctionClassification fc;
  fc.returnsVoid = voidRet;
  mlir::Type origRet = voidRet ? mlir::Type() : retCIR;
  mlir::FailureOr<ArgClassification> retAc =
      convertABIArgInfo(fi->getReturnInfo(), ctx, origRet);
  if (failed(retAc)) {
    nyiCoercion(retCIR);
    return mlir::failure();
  }
  fc.returnInfo = *retAc;
  for (unsigned i = 0, e = fi->arg_size(); i < e; ++i) {
    mlir::Type origArg = inputs[i];
    const llvm::abi::ArgInfo &argInfo = fi->getArgInfo(i).Info;
    mlir::FailureOr<ArgClassification> ac =
        convertABIArgInfo(argInfo, ctx, origArg);
    if (failed(ac)) {
      nyiCoercion(origArg);
      return mlir::failure();
    }
    // The classifier passes a transparent union as its first member, and
    // names no coerce type when it passes that member as its own type.  The
    // union is then left unchanged or coerced to itself above.
    auto recTy = dyn_cast<cir::RecordType>(origArg);
    if (ac->kind == ArgKind::Direct && recTy &&
        (!ac->coercedType || ac->coercedType == recTy) &&
        recordIsTransparentUnion(modOp, recTy))
      ac->coercedType = recTy.getMembers().front();
    ac->neededIntRegs = argInfo.getNeededIntRegs();
    ac->neededSseRegs = argInfo.getNeededSseRegs();
    fc.argInfos.push_back(*ac);
  }
  return fc;
}

/// FIXME: Only the AMDGPU kernel convention is mapped. Every other convention
/// is classified as C.
llvm::CallingConv::ID cir::convertCallingConv(cir::CallingConv cc) {
  if (cc == cir::CallingConv::AMDGPUKernel)
    return llvm::CallingConv::AMDGPU_KERNEL;
  return llvm::CallingConv::C;
}

/// Classify a cir.func using the LLVM ABI library.  Returns failure and
/// emits an NYI error if the signature uses a type the bridge does not handle
/// yet.
mlir::FailureOr<FunctionClassification>
CIRABIClassifierCommon::classifyFunction(cir::FuncOp func) {
  cir::FuncType fnTy = func.getFunctionType();
  return classifySignature(fnTy.getReturnType(), fnTy.getInputs(),
                           requiredArgs(fnTy), getTargetInfo(func),
                           convertCallingConv(func.getCallingConv()),
                           [&]() { return func.emitOpError(); });
}

/// Classify the single type fetched by a `cir.va_arg` as an unnamed argument
/// (RequiredArgs(0)) with a full register budget.  Returns failure and
/// emits an NYI error if the type is not one this bridge handles.
mlir::FailureOr<ArgClassification> CIRABIClassifierCommon::classifyVarArgType(
    mlir::Type ty, const llvm::abi::TargetInfo &targetInfo,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = modOp->getContext();
  mlir::FailureOr<FunctionClassification> fc = classifySignature(
      cir::VoidType::get(ctx), mlir::TypeRange(ty), llvm::abi::RequiredArgs(0),
      targetInfo, llvm::CallingConv::C, emitError);
  if (failed(fc))
    return mlir::failure();
  return fc->argInfos[0];
}

/// Classify a call that passes arguments through an ellipsis.  The callee's
/// own classification covers only its declared parameters, but an ellipsis
/// argument competes for the same argument registers as a declared one, so
/// what the ABI does with it depends on the whole argument list: the same
/// small struct is passed in registers early in the list and in memory once
/// the integer registers are gone.  Classifying from the call's operands
/// rather than the callee's signature is what makes that accounting right.
mlir::FailureOr<FunctionClassification>
CIRABIClassifierCommon::classifyVariadicCall(
    cir::CIRCallOpInterface call, cir::FuncType calleeTy,
    const llvm::abi::TargetInfo &targetInfo, llvm::CallingConv::ID callConv) {
  assert(calleeTy.isVarArg() &&
         "only a variadic callee can take more operands than it declares");
  Operation *op = call.getOperation();
  return classifySignature(calleeTy.getReturnType(),
                           call.getArgOperands().getTypes(),
                           requiredArgs(calleeTy), targetInfo, callConv,
                           [&]() { return op->emitOpError(); });
}

mlir::FailureOr<FunctionClassification>
CIRABIClassifierCommon::classifyVariadicCall(cir::CIRCallOpInterface call,
                                             cir::FuncOp callee) {
  return classifyVariadicCall(call, callee.getFunctionType(),
                              getTargetInfo(callee),
                              convertCallingConv(callee.getCallingConv()));
}

mlir::FailureOr<FunctionClassification>
CIRABIClassifierCommon::classifyIndirectCall(cir::CIRCallOpInterface call) {
  cir::FuncType funcTy = getIndirectCalleeType(call);
  const llvm::abi::TargetInfo &targetInfo =
      getTargetInfo(call->getParentOfType<cir::FuncOp>());
  // An indirect callee is an ordinary function pointer, so it uses the C
  // convention.
  if (call.getNumArgOperands() > funcTy.getNumInputs())
    return classifyVariadicCall(call, funcTy, targetInfo, llvm::CallingConv::C);
  return classifySignature(
      funcTy.getReturnType(), funcTy.getInputs(), requiredArgs(funcTy),
      targetInfo, llvm::CallingConv::C, [&]() { return call->emitOpError(); });
}

/// Record on \p fc whether \p returnType is CIR's void.  classifySignature
/// answers this itself, but the other two drivers cannot: the test target is
/// dialect-neutral and has no notion of CIR's void, and the
/// classification-attr schema carries no return type at all.  Both route
/// through here so a classification always reaches needsRewrite paired with
/// the return type it was built from.
static mlir::FailureOr<FunctionClassification>
withReturnVoidness(mlir::FailureOr<FunctionClassification> fc,
                   mlir::Type returnType) {
  if (succeeded(fc))
    fc->returnsVoid = mlir::isa<cir::VoidType>(returnType);
  return fc;
}

/// Classify \p func using whichever driver mode is configured.  Returns
/// failure and emits an error on the function if classification fails
/// (e.g. injection-driver mode but the function is missing the attribute,
/// or the attribute is malformed).
static mlir::FailureOr<FunctionClassification>
classifyFunction(cir::FuncOp func, const DataLayout &dl,
                 cir::CallConvTarget target, StringRef classificationAttrName) {
  ArrayRef<Type> argTypes = func.getFunctionType().getInputs();
  Type returnType = func.getFunctionType().getReturnType();

  if (!classificationAttrName.empty()) {
    auto attr = func->getAttrOfType<DictionaryAttr>(classificationAttrName);
    if (!attr) {
      func.emitOpError()
          << "missing classification attribute '" << classificationAttrName
          << "' (CallConvLowering driver mode 'classification-attr')";
      return mlir::failure();
    }
    return withReturnVoidness(mlir::abi::test::parseClassificationAttr(
                                  attr, [&]() { return func.emitOpError(); }),
                              returnType);
  }

  // The x86_64 and AMDGPU targets classify through CIRABIClassifierCommon, so
  // only the test target reaches here.
  assert(target == cir::CallConvTarget::Test &&
         "classifyFunction only handles the test target");
  return withReturnVoidness(mlir::abi::test::classify(argTypes, returnType, dl),
                            returnType);
}

namespace {

/// Classifies with the MLIR test ABI target: predictable rules that
/// approximate x86_64 SysV, used by tests that need a target-independent
/// driver.
class TestTargetClassifier final : public CIRABIClassifier {
public:
  explicit TestTargetClassifier(const mlir::DataLayout &dl) : dl(dl) {}

  mlir::FailureOr<FunctionClassification>
  classifyFunction(cir::FuncOp func) final {
    return ::classifyFunction(func, dl, cir::CallConvTarget::Test,
                              /*classificationAttrName=*/"");
  }

  /// The test target classifies only declared parameters.
  mlir::FailureOr<FunctionClassification>
  classifyVariadicCall(cir::CIRCallOpInterface call, cir::FuncOp) final {
    call->emitOpError()
        << "variadic arguments not yet implemented in CallConvLowering";
    return mlir::failure();
  }

  mlir::FailureOr<FunctionClassification>
  classifyIndirectCall(cir::CIRCallOpInterface call) final {
    cir::FuncType funcTy = getIndirectCalleeType(call);
    if (call.getNumArgOperands() > funcTy.getNumInputs()) {
      call->emitOpError()
          << "variadic arguments not yet implemented in CallConvLowering";
      return mlir::failure();
    }
    return withReturnVoidness(mlir::abi::test::classify(funcTy.getInputs(),
                                                        funcTy.getReturnType(),
                                                        dl),
                              funcTy.getReturnType());
  }

  mlir::FailureOr<ArgClassification> classifyVAArg(cir::VAArgOp vaArg) final {
    vaArg->emitOpError()
        << "not yet implemented in CallConvLowering for the test target";
    return mlir::failure();
  }

private:
  const mlir::DataLayout &dl;
};

/// Reads a pre-built classification from a function attribute, so a test can
/// inject any classification (including shapes no target produces) without
/// depending on a real ABI target.
class InjectedClassifier final : public CIRABIClassifier {
public:
  InjectedClassifier(const mlir::DataLayout &dl, llvm::StringRef attrName)
      : dl(dl), attrName(attrName) {}

  mlir::FailureOr<FunctionClassification>
  classifyFunction(cir::FuncOp func) final {
    return ::classifyFunction(func, dl, cir::CallConvTarget::None, attrName);
  }

  /// The injected classification covers a function's declared parameters, so
  /// it cannot describe an operand passed through an ellipsis.
  mlir::FailureOr<FunctionClassification>
  classifyVariadicCall(cir::CIRCallOpInterface call, cir::FuncOp) final {
    call->emitOpError()
        << "variadic arguments not yet implemented in CallConvLowering";
    return mlir::failure();
  }

  /// Nor can it describe a callee resolved at run time, which names no
  /// function to read the attribute from.
  mlir::FailureOr<FunctionClassification>
  classifyIndirectCall(cir::CIRCallOpInterface call) final {
    call->emitOpError() << "indirect call cannot be classified in the "
                           "'classification-attr' driver mode";
    return mlir::failure();
  }

  mlir::FailureOr<ArgClassification> classifyVAArg(cir::VAArgOp vaArg) final {
    vaArg->emitOpError() << "cannot be classified in the "
                            "'classification-attr' driver mode";
    return mlir::failure();
  }

private:
  const mlir::DataLayout &dl;
  std::string attrName;
};

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createTestTargetABIClassifier(const mlir::DataLayout &dl) {
  return std::make_unique<TestTargetClassifier>(dl);
}

std::unique_ptr<CIRABIClassifier>
cir::createInjectedABIClassifier(const mlir::DataLayout &dl,
                                 llvm::StringRef attrName) {
  return std::make_unique<InjectedClassifier>(dl, attrName);
}
