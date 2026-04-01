#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace cir;

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

CIRDataLayout::CIRDataLayout(aiir::ModuleOp modOp) : layout(modOp) {
  reset(modOp.getDataLayoutSpec());
}

void CIRDataLayout::reset(aiir::DataLayoutSpecInterface spec) {
  bigEndian = false;
  if (spec) {
    aiir::StringAttr key = aiir::StringAttr::get(
        spec.getContext(), aiir::DLTIDialect::kDataLayoutEndiannessKey);
    if (aiir::DataLayoutEntryInterface entry = spec.getSpecForIdentifier(key))
      if (auto str = llvm::dyn_cast<aiir::StringAttr>(entry.getValue()))
        bigEndian = str == aiir::DLTIDialect::kDataLayoutEndiannessBig;

    aiir::StringAttr addrSpKey = aiir::StringAttr::get(
        spec.getContext(), aiir::DLTIDialect::kDataLayoutProgramMemorySpaceKey);
    if (aiir::DataLayoutEntryInterface entry =
            spec.getSpecForIdentifier(addrSpKey))
      if (auto val = llvm::dyn_cast<aiir::IntegerAttr>(entry.getValue()))
        programAddrSpace = val.getInt();
  }
}

llvm::Align CIRDataLayout::getAlignment(aiir::Type ty, bool useABIAlign) const {
  // FIXME(cir): This does not account for differnt address spaces, and relies
  // on CIR's data layout to give the proper alignment.
  assert(!cir::MissingFeatures::addressSpace());

  // Fetch type alignment from AIIR's data layout.
  unsigned align = useABIAlign ? layout.getTypeABIAlignment(ty)
                               : layout.getTypePreferredAlignment(ty);
  return llvm::Align(align);
}

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
llvm::TypeSize CIRDataLayout::getTypeSizeInBits(aiir::Type ty) const {
  assert(cir::isSized(ty) && "Cannot getTypeInfo() on a type that is unsized!");

  if (auto recordTy = llvm::dyn_cast<cir::RecordType>(ty)) {
    // FIXME(cir): CIR record's data layout implementation doesn't do a good job
    // of handling unions particularities. We should have a separate union type.
    return recordTy.getTypeSizeInBits(layout, {});
  }

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!cir::MissingFeatures::addressSpace());

  // This is calling aiir::DataLayout::getTypeSizeInBits().
  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(ty));
}
