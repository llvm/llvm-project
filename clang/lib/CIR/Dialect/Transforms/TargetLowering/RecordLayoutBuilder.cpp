//=== RecordLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/CGRecordLayoutBuilder.cpp. The
// queries are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRLowerContext.h"
#include "CIRRecordLayout.h"
#include "mlir/IR/Types.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace mlir;
using namespace mlir::cir;

namespace {

//===-----------------------------------------------------------------------==//
// EmptySubobjectMap Implementation
//===----------------------------------------------------------------------===//

/// Keeps track of which empty subobjects exist at different offsets while
/// laying out a C++ class.
class EmptySubobjectMap {
  const CIRLowerContext &Context;
  uint64_t CharWidth;

  /// The class whose empty entries we're keeping track of.
  const StructType Class;

  /// The highest offset known to contain an empty base subobject.
  clang::CharUnits MaxEmptyClassOffset;

  /// Compute the size of the largest base or member subobject that is empty.
  void ComputeEmptySubobjectSizes();

public:
  /// This holds the size of the largest empty subobject (either a base
  /// or a member). Will be zero if the record being built doesn't contain
  /// any empty classes.
  clang::CharUnits SizeOfLargestEmptySubobject;

  EmptySubobjectMap(const CIRLowerContext &Context, const StructType Class)
      : Context(Context), CharWidth(Context.getCharWidth()), Class(Class) {
    ComputeEmptySubobjectSizes();
  }

  /// Return whether a field can be placed at the given offset.
  bool canPlaceFieldAtOffset(const Type Ty, clang::CharUnits Offset);
};

void EmptySubobjectMap::ComputeEmptySubobjectSizes() {
  // Check the bases.
  assert(!::cir::MissingFeatures::getCXXRecordBases());

  // Check the fields.
  for (const auto FT : Class.getMembers()) {
    assert(!::cir::MissingFeatures::qualifiedTypes());
    const auto RT = dyn_cast<StructType>(FT);

    // We only care about record types.
    if (!RT)
      continue;

    // TODO(cir): Handle nested record types.
    llvm_unreachable("NYI");
  }
}

bool EmptySubobjectMap::canPlaceFieldAtOffset(const Type Ty,
                                              clang::CharUnits Offset) {
  llvm_unreachable("NYI");
}

//===-----------------------------------------------------------------------==//
// ItaniumRecordLayoutBuilder Implementation
//===----------------------------------------------------------------------===//

class ItaniumRecordLayoutBuilder {
protected:
  // FIXME(cir):  Remove this and make the appropriate fields public.
  friend class mlir::cir::CIRLowerContext;

  const CIRLowerContext &Context;

  EmptySubobjectMap *EmptySubobjects;

  /// Size - The current size of the record layout.
  uint64_t Size;

  /// Alignment - The current alignment of the record layout.
  clang::CharUnits Alignment;

  /// PreferredAlignment - The preferred alignment of the record layout.
  clang::CharUnits PreferredAlignment;

  /// The alignment if attribute packed is not used.
  clang::CharUnits UnpackedAlignment;

  /// \brief The maximum of the alignments of top-level members.
  clang::CharUnits UnadjustedAlignment;

  SmallVector<uint64_t, 16> FieldOffsets;

  /// Whether the external AST source has provided a layout for this
  /// record.
  unsigned UseExternalLayout : 1;

  /// Whether we need to infer alignment, even when we have an
  /// externally-provided layout.
  unsigned InferAlignment : 1;

  /// Packed - Whether the record is packed or not.
  unsigned Packed : 1;

  unsigned IsUnion : 1;

  unsigned IsMac68kAlign : 1;

  unsigned IsNaturalAlign : 1;

  unsigned IsMsStruct : 1;

  /// UnfilledBitsInLastUnit - If the last field laid out was a bitfield,
  /// this contains the number of bits in the last unit that can be used for
  /// an adjacent bitfield if necessary.  The unit in question is usually
  /// a byte, but larger units are used if IsMsStruct.
  unsigned char UnfilledBitsInLastUnit;

  /// LastBitfieldStorageUnitSize - If IsMsStruct, represents the size of the
  /// storage unit of the previous field if it was a bitfield.
  unsigned char LastBitfieldStorageUnitSize;

  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack.
  clang::CharUnits MaxFieldAlignment;

  /// DataSize - The data size of the record being laid out.
  uint64_t DataSize;

  clang::CharUnits NonVirtualSize;
  clang::CharUnits NonVirtualAlignment;
  clang::CharUnits PreferredNVAlignment;

  /// If we've laid out a field but not included its tail padding in Size yet,
  /// this is the size up to the end of that field.
  clang::CharUnits PaddedFieldSize;

  /// The primary base class (if one exists) of the class we're laying out.
  const StructType PrimaryBase;

  /// Whether the primary base of the class we're laying out is virtual.
  bool PrimaryBaseIsVirtual;

  /// Whether the class provides its own vtable/vftbl pointer, as opposed to
  /// inheriting one from a primary base class.
  bool HasOwnVFPtr;

  /// the flag of field offset changing due to packed attribute.
  bool HasPackedField;

  /// An auxiliary field used for AIX. When there are OverlappingEmptyFields
  /// existing in the aggregate, the flag shows if the following first non-empty
  /// or empty-but-non-overlapping field has been handled, if any.
  bool HandledFirstNonOverlappingEmptyField;

public:
  ItaniumRecordLayoutBuilder(const CIRLowerContext &Context,
                             EmptySubobjectMap *EmptySubobjects)
      : Context(Context), EmptySubobjects(EmptySubobjects), Size(0),
        Alignment(clang::CharUnits::One()),
        PreferredAlignment(clang::CharUnits::One()),
        UnpackedAlignment(clang::CharUnits::One()),
        UnadjustedAlignment(clang::CharUnits::One()), UseExternalLayout(false),
        InferAlignment(false), Packed(false), IsUnion(false),
        IsMac68kAlign(false),
        IsNaturalAlign(!Context.getTargetInfo().getTriple().isOSAIX()),
        IsMsStruct(false), UnfilledBitsInLastUnit(0),
        LastBitfieldStorageUnitSize(0),
        MaxFieldAlignment(clang::CharUnits::Zero()), DataSize(0),
        NonVirtualSize(clang::CharUnits::Zero()),
        NonVirtualAlignment(clang::CharUnits::One()),
        PreferredNVAlignment(clang::CharUnits::One()),
        PaddedFieldSize(clang::CharUnits::Zero()), PrimaryBaseIsVirtual(false),
        HasOwnVFPtr(false), HasPackedField(false),
        HandledFirstNonOverlappingEmptyField(false) {}

  void layout(const StructType D);

  void layoutFields(const StructType D);
  void layoutField(const Type Ty, bool InsertExtraPadding);

  void UpdateAlignment(clang::CharUnits NewAlignment,
                       clang::CharUnits UnpackedNewAlignment,
                       clang::CharUnits PreferredAlignment);

  void checkFieldPadding(uint64_t Offset, uint64_t UnpaddedOffset,
                         uint64_t UnpackedOffset, unsigned UnpackedAlign,
                         bool isPacked, const Type Ty);

  clang::CharUnits getSize() const {
    assert(Size % Context.getCharWidth() == 0);
    return Context.toCharUnitsFromBits(Size);
  }
  uint64_t getSizeInBits() const { return Size; }

  void setSize(clang::CharUnits NewSize) { Size = Context.toBits(NewSize); }
  void setSize(uint64_t NewSize) { Size = NewSize; }

  clang::CharUnits getDataSize() const {
    assert(DataSize % Context.getCharWidth() == 0);
    return Context.toCharUnitsFromBits(DataSize);
  }

  /// Initialize record layout for the given record decl.
  void initializeLayout(const Type Ty);

  uint64_t getDataSizeInBits() const { return DataSize; }

  void setDataSize(clang::CharUnits NewSize) {
    DataSize = Context.toBits(NewSize);
  }
  void setDataSize(uint64_t NewSize) { DataSize = NewSize; }
};

void ItaniumRecordLayoutBuilder::layout(const StructType RT) {
  initializeLayout(RT);

  // Lay out the vtable and the non-virtual bases.
  assert(!::cir::MissingFeatures::isCXXRecordDecl() &&
         !::cir::MissingFeatures::CXXRecordIsDynamicClass());

  layoutFields(RT);

  // FIXME(cir): Handle virtual-related layouts.
  assert(!::cir::MissingFeatures::getCXXRecordBases());

  assert(!::cir::MissingFeatures::itaniumRecordLayoutBuilderFinishLayout());
}

void ItaniumRecordLayoutBuilder::initializeLayout(const mlir::Type Ty) {
  if (const auto RT = dyn_cast<StructType>(Ty)) {
    IsUnion = RT.isUnion();
    assert(!::cir::MissingFeatures::recordDeclIsMSStruct());
  }

  assert(!::cir::MissingFeatures::recordDeclIsPacked());

  // Honor the default struct packing maximum alignment flag.
  if (unsigned DefaultMaxFieldAlignment = Context.getLangOpts().PackStruct) {
    llvm_unreachable("NYI");
  }

  // mac68k alignment supersedes maximum field alignment and attribute aligned,
  // and forces all structures to have 2-byte alignment. The IBM docs on it
  // allude to additional (more complicated) semantics, especially with regard
  // to bit-fields, but gcc appears not to follow that.
  if (::cir::MissingFeatures::declHasAlignMac68kAttr()) {
    llvm_unreachable("NYI");
  } else {
    if (::cir::MissingFeatures::declHasAlignNaturalAttr())
      llvm_unreachable("NYI");

    if (::cir::MissingFeatures::declHasMaxFieldAlignmentAttr())
      llvm_unreachable("NYI");

    if (::cir::MissingFeatures::declGetMaxAlignment())
      llvm_unreachable("NYI");
  }

  HandledFirstNonOverlappingEmptyField =
      !Context.getTargetInfo().defaultsToAIXPowerAlignment() || IsNaturalAlign;

  // If there is an external AST source, ask it for the various offsets.
  if (const auto RT = dyn_cast<StructType>(Ty)) {
    if (::cir::MissingFeatures::astContextGetExternalSource()) {
      llvm_unreachable("NYI");
    }
  }
}

void ItaniumRecordLayoutBuilder::layoutField(const Type D,
                                             bool InsertExtraPadding) {
  // auto FieldClass = D.dyn_cast<StructType>();
  assert(!::cir::MissingFeatures::fieldDeclIsPotentiallyOverlapping() &&
         !::cir::MissingFeatures::CXXRecordDeclIsEmptyCXX11());
  bool IsOverlappingEmptyField = false; // FIXME(cir): Needs more features.

  clang::CharUnits FieldOffset = (IsUnion || IsOverlappingEmptyField)
                                     ? clang::CharUnits::Zero()
                                     : getDataSize();

  const bool DefaultsToAIXPowerAlignment =
      Context.getTargetInfo().defaultsToAIXPowerAlignment();
  bool FoundFirstNonOverlappingEmptyFieldForAIX = false;
  if (DefaultsToAIXPowerAlignment && !HandledFirstNonOverlappingEmptyField) {
    llvm_unreachable("NYI");
  }

  assert(!::cir::MissingFeatures::fieldDeclIsBitfield());

  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastUnit;
  // Reset the unfilled bits.
  UnfilledBitsInLastUnit = 0;
  LastBitfieldStorageUnitSize = 0;

  llvm::Triple Target = Context.getTargetInfo().getTriple();

  clang::AlignRequirementKind AlignRequirement =
      clang::AlignRequirementKind::None;
  clang::CharUnits FieldSize;
  clang::CharUnits FieldAlign;
  // The amount of this class's dsize occupied by the field.
  // This is equal to FieldSize unless we're permitted to pack
  // into the field's tail padding.
  clang::CharUnits EffectiveFieldSize;

  auto setDeclInfo = [&](bool IsIncompleteArrayType) {
    auto TI = Context.getTypeInfoInChars(D);
    FieldAlign = TI.Align;
    // Flexible array members don't have any size, but they have to be
    // aligned appropriately for their element type.
    EffectiveFieldSize = FieldSize =
        IsIncompleteArrayType ? clang::CharUnits::Zero() : TI.Width;
    AlignRequirement = TI.AlignRequirement;
  };

  if (isa<ArrayType>(D) && cast<ArrayType>(D).getSize() == 0) {
    llvm_unreachable("NYI");
  } else {
    setDeclInfo(false /* IsIncompleteArrayType */);

    if (::cir::MissingFeatures::fieldDeclIsPotentiallyOverlapping())
      llvm_unreachable("NYI");

    if (IsMsStruct)
      llvm_unreachable("NYI");
  }

  assert(!::cir::MissingFeatures::recordDeclIsPacked() &&
         !::cir::MissingFeatures::CXXRecordDeclIsPOD());
  bool FieldPacked = false; // FIXME(cir): Needs more features.

  // When used as part of a typedef, or together with a 'packed' attribute, the
  // 'aligned' attribute can be used to decrease alignment. In that case, it
  // overrides any computed alignment we have, and there is no need to upgrade
  // the alignment.
  auto alignedAttrCanDecreaseAIXAlignment = [AlignRequirement, FieldPacked] {
    // Enum alignment sources can be safely ignored here, because this only
    // helps decide whether we need the AIX alignment upgrade, which only
    // applies to floating-point types.
    return AlignRequirement == clang::AlignRequirementKind::RequiredByTypedef ||
           (AlignRequirement == clang::AlignRequirementKind::RequiredByRecord &&
            FieldPacked);
  };

  // The AIX `power` alignment rules apply the natural alignment of the
  // "first member" if it is of a floating-point data type (or is an aggregate
  // whose recursively "first" member or element is such a type). The alignment
  // associated with these types for subsequent members use an alignment value
  // where the floating-point data type is considered to have 4-byte alignment.
  //
  // For the purposes of the foregoing: vtable pointers, non-empty base classes,
  // and zero-width bit-fields count as prior members; members of empty class
  // types marked `no_unique_address` are not considered to be prior members.
  clang::CharUnits PreferredAlign = FieldAlign;
  if (DefaultsToAIXPowerAlignment && !alignedAttrCanDecreaseAIXAlignment() &&
      (FoundFirstNonOverlappingEmptyFieldForAIX || IsNaturalAlign)) {
    llvm_unreachable("NYI");
  }

  // The align if the field is not packed. This is to check if the attribute
  // was unnecessary (-Wpacked).
  clang::CharUnits UnpackedFieldAlign = FieldAlign;
  clang::CharUnits PackedFieldAlign = clang::CharUnits::One();
  clang::CharUnits UnpackedFieldOffset = FieldOffset;
  // clang::CharUnits OriginalFieldAlign = UnpackedFieldAlign;

  assert(!::cir::MissingFeatures::fieldDeclGetMaxFieldAlignment());
  clang::CharUnits MaxAlignmentInChars = clang::CharUnits::Zero();
  PackedFieldAlign = std::max(PackedFieldAlign, MaxAlignmentInChars);
  PreferredAlign = std::max(PreferredAlign, MaxAlignmentInChars);
  UnpackedFieldAlign = std::max(UnpackedFieldAlign, MaxAlignmentInChars);

  // The maximum field alignment overrides the aligned attribute.
  if (!MaxFieldAlignment.isZero()) {
    llvm_unreachable("NYI");
  }

  if (!FieldPacked)
    FieldAlign = UnpackedFieldAlign;
  if (DefaultsToAIXPowerAlignment)
    llvm_unreachable("NYI");
  if (FieldPacked) {
    llvm_unreachable("NYI");
  }

  clang::CharUnits AlignTo =
      !DefaultsToAIXPowerAlignment ? FieldAlign : PreferredAlign;
  // Round up the current record size to the field's alignment boundary.
  FieldOffset = FieldOffset.alignTo(AlignTo);
  UnpackedFieldOffset = UnpackedFieldOffset.alignTo(UnpackedFieldAlign);

  if (UseExternalLayout) {
    llvm_unreachable("NYI");
  } else {
    if (!IsUnion && EmptySubobjects) {
      // Check if we can place the field at this offset.
      while (/*!EmptySubobjects->CanPlaceFieldAtOffset(D, FieldOffset)*/
             false) {
        llvm_unreachable("NYI");
      }
    }
  }

  // Place this field at the current location.
  FieldOffsets.push_back(Context.toBits(FieldOffset));

  if (!UseExternalLayout)
    checkFieldPadding(Context.toBits(FieldOffset), UnpaddedFieldOffset,
                      Context.toBits(UnpackedFieldOffset),
                      Context.toBits(UnpackedFieldAlign), FieldPacked, D);

  if (InsertExtraPadding) {
    llvm_unreachable("NYI");
  }

  // Reserve space for this field.
  if (!IsOverlappingEmptyField) {
    // uint64_t EffectiveFieldSizeInBits = Context.toBits(EffectiveFieldSize);
    if (IsUnion)
      llvm_unreachable("NYI");
    else
      setDataSize(FieldOffset + EffectiveFieldSize);

    PaddedFieldSize = std::max(PaddedFieldSize, FieldOffset + FieldSize);
    setSize(std::max(getSizeInBits(), getDataSizeInBits()));
  } else {
    llvm_unreachable("NYI");
  }

  // Remember max struct/class ABI-specified alignment.
  UnadjustedAlignment = std::max(UnadjustedAlignment, FieldAlign);
  UpdateAlignment(FieldAlign, UnpackedFieldAlign, PreferredAlign);

  // For checking the alignment of inner fields against
  // the alignment of its parent record.
  // FIXME(cir): We need to track the parent record of the current type being
  // laid out. A regular mlir::Type has not way of doing this. In fact, we will
  // likely need an external abstraction, as I don't think this is possible with
  // just the field type.
  assert(!::cir::MissingFeatures::fieldDeclAbstraction());

  if (Packed && !FieldPacked && PackedFieldAlign < FieldAlign)
    llvm_unreachable("NYI");
}

void ItaniumRecordLayoutBuilder::layoutFields(const StructType D) {
  // Layout each field, for now, just sequentially, respecting alignment.  In
  // the future, this will need to be tweakable by targets.
  assert(!::cir::MissingFeatures::recordDeclMayInsertExtraPadding() &&
         !Context.getLangOpts().SanitizeAddressFieldPadding);
  bool InsertExtraPadding = false;
  assert(!::cir::MissingFeatures::recordDeclHasFlexibleArrayMember());
  bool HasFlexibleArrayMember = false;
  for (const auto FT : D.getMembers()) {
    layoutField(FT, InsertExtraPadding && (FT != D.getMembers().back() ||
                                           !HasFlexibleArrayMember));
  }
}

void ItaniumRecordLayoutBuilder::UpdateAlignment(
    clang::CharUnits NewAlignment, clang::CharUnits UnpackedNewAlignment,
    clang::CharUnits PreferredNewAlignment) {
  // The alignment is not modified when using 'mac68k' alignment or when
  // we have an externally-supplied layout that also provides overall alignment.
  if (IsMac68kAlign || (UseExternalLayout && !InferAlignment))
    return;

  if (NewAlignment > Alignment) {
    assert(llvm::isPowerOf2_64(NewAlignment.getQuantity()) &&
           "Alignment not a power of 2");
    Alignment = NewAlignment;
  }

  if (UnpackedNewAlignment > UnpackedAlignment) {
    assert(llvm::isPowerOf2_64(UnpackedNewAlignment.getQuantity()) &&
           "Alignment not a power of 2");
    UnpackedAlignment = UnpackedNewAlignment;
  }

  if (PreferredNewAlignment > PreferredAlignment) {
    assert(llvm::isPowerOf2_64(PreferredNewAlignment.getQuantity()) &&
           "Alignment not a power of 2");
    PreferredAlignment = PreferredNewAlignment;
  }
}

void ItaniumRecordLayoutBuilder::checkFieldPadding(
    uint64_t Offset, uint64_t UnpaddedOffset, uint64_t UnpackedOffset,
    unsigned UnpackedAlign, bool isPacked, const Type Ty) {
  // We let objc ivars without warning, objc interfaces generally are not used
  // for padding tricks.
  if (::cir::MissingFeatures::objCIvarDecls())
    llvm_unreachable("NYI");

  // FIXME(cir): Should the following be skiped in CIR?
  // Don't warn about structs created without a SourceLocation.  This can
  // be done by clients of the AST, such as codegen.

  unsigned CharBitNum = Context.getTargetInfo().getCharWidth();

  // Warn if padding was introduced to the struct/class.
  if (!IsUnion && Offset > UnpaddedOffset) {
    unsigned PadSize = Offset - UnpaddedOffset;
    // bool InBits = true;
    if (PadSize % CharBitNum == 0) {
      PadSize = PadSize / CharBitNum;
      // InBits = false;
    }
    assert(::cir::MissingFeatures::bitFieldPaddingDiagnostics());
  }
  if (isPacked && Offset != UnpackedOffset) {
    HasPackedField = true;
  }
}

//===-----------------------------------------------------------------------==//
// Misc. Helper Functions
//===----------------------------------------------------------------------===//

bool isMsLayout(const CIRLowerContext &Context) {
  return Context.getTargetInfo().getCXXABI().isMicrosoft();
}

/// Does the target C++ ABI require us to skip over the tail-padding
/// of the given class (considering it as a base class) when allocating
/// objects?
static bool mustSkipTailPadding(clang::TargetCXXABI ABI, const StructType RD) {
  assert(!::cir::MissingFeatures::recordDeclIsCXXDecl());
  switch (ABI.getTailPaddingUseRules()) {
  case clang::TargetCXXABI::AlwaysUseTailPadding:
    return false;

  case clang::TargetCXXABI::UseTailPaddingUnlessPOD03:
    // http://itanium-cxx-abi.github.io/cxx-abi/abi.html#POD :
    //   In general, a type is considered a POD for the purposes of
    //   layout if it is a POD type (in the sense of ISO C++
    //   [basic.types]). However, a POD-struct or POD-union (in the
    //   sense of ISO C++ [class]) with a bitfield member whose
    //   declared width is wider than the declared type of the
    //   bitfield is not a POD for the purpose of layout.  Similarly,
    //   an array type is not a POD for the purpose of layout if the
    //   element type of the array is not a POD for the purpose of
    //   layout.
    //
    //   Where references to the ISO C++ are made in this paragraph,
    //   the Technical Corrigendum 1 version of the standard is
    //   intended.
    // FIXME(cir): This always returns true since we can't check if a CIR record
    // is a POD type.
    assert(!::cir::MissingFeatures::CXXRecordDeclIsPOD());
    return true;

  case clang::TargetCXXABI::UseTailPaddingUnlessPOD11:
    // This is equivalent to RD->getTypeForDecl().isCXX11PODType(),
    // but with a lot of abstraction penalty stripped off.  This does
    // assume that these properties are set correctly even in C++98
    // mode; fortunately, that is true because we want to assign
    // consistently semantics to the type-traits intrinsics (or at
    // least as many of them as possible).
    llvm_unreachable("NYI");
  }

  llvm_unreachable("bad tail-padding use kind");
}

} // namespace

/// Get or compute information about the layout of the specified record
/// (struct/union/class), which indicates its size and field position
/// information.
const CIRRecordLayout &CIRLowerContext::getCIRRecordLayout(const Type D) const {
  assert(isa<StructType>(D) && "Not a record type");
  auto RT = dyn_cast<StructType>(D);

  assert(RT.isComplete() && "Cannot get layout of forward declarations!");

  // FIXME(cir): Use a more MLIR-based approach by using it's buitin data layout
  // features, such as interfaces, cacheing, and the DLTI dialect.

  const CIRRecordLayout *NewEntry = nullptr;

  if (isMsLayout(*this)) {
    llvm_unreachable("NYI");
  } else {
    // FIXME(cir): Add if-else separating C and C++ records.
    assert(!::cir::MissingFeatures::isCXXRecordDecl());
    EmptySubobjectMap EmptySubobjects(*this, RT);
    ItaniumRecordLayoutBuilder Builder(*this, &EmptySubobjects);
    Builder.layout(RT);

    // In certain situations, we are allowed to lay out objects in the
    // tail-padding of base classes.  This is ABI-dependent.
    // FIXME: this should be stored in the record layout.
    bool skipTailPadding = mustSkipTailPadding(getTargetInfo().getCXXABI(), RT);

    // FIXME: This should be done in FinalizeLayout.
    clang::CharUnits DataSize =
        skipTailPadding ? Builder.getSize() : Builder.getDataSize();
    clang::CharUnits NonVirtualSize =
        skipTailPadding ? DataSize : Builder.NonVirtualSize;
    assert(!::cir::MissingFeatures::CXXRecordIsDynamicClass());
    // FIXME(cir): Whose responsible for freeing the allocation below?
    NewEntry = new CIRRecordLayout(
        *this, Builder.getSize(), Builder.Alignment, Builder.PreferredAlignment,
        Builder.UnadjustedAlignment,
        /*RequiredAlignment : used by MS-ABI)*/
        Builder.Alignment, Builder.HasOwnVFPtr, /*RD->isDynamicClass()=*/false,
        clang::CharUnits::fromQuantity(-1), DataSize, Builder.FieldOffsets,
        NonVirtualSize, Builder.NonVirtualAlignment,
        Builder.PreferredNVAlignment,
        EmptySubobjects.SizeOfLargestEmptySubobject, Builder.PrimaryBase,
        Builder.PrimaryBaseIsVirtual, nullptr, false, false);
  }

  // TODO(cir): Add option to dump the layouts.
  assert(!::cir::MissingFeatures::cacheRecordLayouts());

  return *NewEntry;
}
