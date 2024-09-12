#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace cir;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(mlir::cir::StructType ST, const CIRDataLayout &DL)
    : StructSize(llvm::TypeSize::getFixed(0)) {
  assert(!ST.isIncomplete() && "Cannot get layout of opaque structs");
  IsPadded = false;
  NumElements = ST.getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    mlir::Type Ty = ST.getMembers()[i];
    if (i == 0 && ::cir::MissingFeatures::typeIsScalableType())
      llvm_unreachable("Scalable types are not yet supported in CIR");

    assert(!::cir::MissingFeatures::recordDeclIsPacked() &&
           "Cannot identify packed structs");
    const llvm::Align TyAlign = DL.getABITypeAlign(Ty);

    // Add padding if necessary to align the data element properly.
    // Currently the only structure with scalable size will be the homogeneous
    // scalable vector types. Homogeneous scalable vector types have members of
    // the same data type so no alignment issue will happen. The condition here
    // assumes so and needs to be adjusted if this assumption changes (e.g. we
    // support structures with arbitrary scalable data type, or structure that
    // contains both fixed size and scalable size data type members).
    if (!StructSize.isScalable() && !isAligned(TyAlign, StructSize)) {
      IsPadded = true;
      StructSize = llvm::TypeSize::getFixed(alignTo(StructSize, TyAlign));
    }

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    getMemberOffsets()[i] = StructSize;
    // Consume space for this data item
    StructSize += DL.getTypeAllocSize(Ty);
  }

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!StructSize.isScalable() && !isAligned(StructAlignment, StructSize)) {
    IsPadded = true;
    StructSize = llvm::TypeSize::getFixed(alignTo(StructSize, StructAlignment));
  }
}

/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t FixedOffset) const {
  assert(!StructSize.isScalable() &&
         "Cannot get element at offset for structure containing scalable "
         "vector types");
  llvm::TypeSize Offset = llvm::TypeSize::getFixed(FixedOffset);
  llvm::ArrayRef<llvm::TypeSize> MemberOffsets = getMemberOffsets();

  const auto *SI =
      std::upper_bound(MemberOffsets.begin(), MemberOffsets.end(), Offset,
                       [](llvm::TypeSize LHS, llvm::TypeSize RHS) -> bool {
                         return llvm::TypeSize::isKnownLT(LHS, RHS);
                       });
  assert(SI != MemberOffsets.begin() && "Offset not in structure type!");
  --SI;
  assert(llvm::TypeSize::isKnownLE(*SI, Offset) && "upper_bound didn't work");
  assert((SI == MemberOffsets.begin() ||
          llvm::TypeSize::isKnownLE(*(SI - 1), Offset)) &&
         (SI + 1 == MemberOffsets.end() ||
          llvm::TypeSize::isKnownGT(*(SI + 1), Offset)) &&
         "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI - MemberOffsets.begin();
}

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

namespace {

class StructLayoutMap {
  using LayoutInfoTy = llvm::DenseMap<mlir::cir::StructType, StructLayout *>;
  LayoutInfoTy LayoutInfo;

public:
  ~StructLayoutMap() {
    // Remove any layouts.
    for (const auto &I : LayoutInfo) {
      StructLayout *Value = I.second;
      Value->~StructLayout();
      free(Value);
    }
  }

  StructLayout *&operator[](mlir::cir::StructType STy) {
    return LayoutInfo[STy];
  }
};

} // namespace

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} { reset(); }

void CIRDataLayout::reset() {
  clear();

  LayoutMap = nullptr;
  bigEndian = false;
  // ManglingMode = MM_None;
  // NonIntegralAddressSpaces.clear();
  StructAlignment =
      llvm::LayoutAlignElem::get(llvm::Align(1), llvm::Align(8), 0);

  // NOTE(cir): Alignment setter functions are skipped as these should already
  // be set in MLIR's data layout.
}

void CIRDataLayout::clear() {
  delete static_cast<StructLayoutMap *>(LayoutMap);
  LayoutMap = nullptr;
}

const StructLayout *
CIRDataLayout::getStructLayout(mlir::cir::StructType Ty) const {
  if (!LayoutMap)
    LayoutMap = new StructLayoutMap();

  StructLayoutMap *STM = static_cast<StructLayoutMap *>(LayoutMap);
  StructLayout *&SL = (*STM)[Ty];
  if (SL)
    return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we
  // malloc it, then use placement new.
  StructLayout *L = (StructLayout *)llvm::safe_malloc(
      StructLayout::totalSizeToAlloc<llvm::TypeSize>(Ty.getNumElements()));

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) StructLayout(Ty, *this);

  return L;
}

/*!
  \param abiOrPref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abiOrPref == true) or preferred alignment (\a abiOrPref
  == false) for the requested type \a Ty.
 */
llvm::Align CIRDataLayout::getAlignment(mlir::Type Ty, bool abiOrPref) const {

  if (llvm::isa<mlir::cir::StructType>(Ty)) {
    // Packed structure types always have an ABI alignment of one.
    if (::cir::MissingFeatures::recordDeclIsPacked() && abiOrPref)
      llvm_unreachable("NYI");

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout =
        getStructLayout(llvm::cast<mlir::cir::StructType>(Ty));
    const llvm::Align Align =
        abiOrPref ? StructAlignment.ABIAlign : StructAlignment.PrefAlign;
    return std::max(Align, Layout->getAlignment());
  }

  // FIXME(cir): This does not account for differnt address spaces, and relies
  // on CIR's data layout to give the proper alignment.
  assert(!::cir::MissingFeatures::addressSpace());

  // Fetch type alignment from MLIR's data layout.
  unsigned align = abiOrPref ? layout.getTypeABIAlignment(Ty)
                             : layout.getTypePreferredAlignment(Ty);
  return llvm::Align(align);
}

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
llvm::TypeSize CIRDataLayout::getTypeSizeInBits(mlir::Type Ty) const {
  assert(!::cir::MissingFeatures::typeIsSized() &&
         "Cannot getTypeInfo() on a type that is unsized!");

  if (auto structTy = llvm::dyn_cast<mlir::cir::StructType>(Ty)) {

    // FIXME(cir): CIR struct's data layout implementation doesn't do a good job
    // of handling unions particularities. We should have a separate union type.
    if (structTy.isUnion()) {
      auto largestMember = structTy.getLargestMember(layout);
      return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(largestMember));
    }

    // FIXME(cir): We should be able to query the size of a struct directly to
    // its data layout implementation instead of requiring a separate
    // StructLayout object.
    // Get the layout annotation... which is lazily created on demand.
    return getStructLayout(structTy)->getSizeInBits();
  }

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!::cir::MissingFeatures::addressSpace());

  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(Ty));
}
