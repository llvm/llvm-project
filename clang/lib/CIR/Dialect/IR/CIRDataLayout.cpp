#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace cir;

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout(modOp) {
  reset(modOp.getDataLayoutSpec());
}

void CIRDataLayout::reset(mlir::DataLayoutSpecInterface spec) {
  bigEndian = false;
  if (spec) {
    mlir::StringAttr key = mlir::StringAttr::get(
        spec.getContext(), mlir::DLTIDialect::kDataLayoutEndiannessKey);
    if (mlir::DataLayoutEntryInterface entry = spec.getSpecForIdentifier(key))
      if (auto str = llvm::dyn_cast<mlir::StringAttr>(entry.getValue()))
        bigEndian = str == mlir::DLTIDialect::kDataLayoutEndiannessBig;
  }
}

llvm::Align CIRDataLayout::getAlignment(mlir::Type ty, bool useABIAlign) const {
  if (auto recTy = llvm::dyn_cast<cir::RecordType>(ty)) {
    // Packed record types always have an ABI alignment of one.
    if (recTy && recTy.getPacked() && useABIAlign)
      return llvm::Align(1);

    // Get the layout annotation... which is lazily created on demand.
    llvm_unreachable("getAlignment()) for record type is not implemented");
  }

  // FIXME(cir): This does not account for differnt address spaces, and relies
  // on CIR's data layout to give the proper alignment.
  assert(!cir::MissingFeatures::addressSpace());

  // Fetch type alignment from MLIR's data layout.
  unsigned align = useABIAlign ? layout.getTypeABIAlignment(ty)
                               : layout.getTypePreferredAlignment(ty);
  return llvm::Align(align);
}

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
llvm::TypeSize CIRDataLayout::getTypeSizeInBits(mlir::Type ty) const {
  assert(cir::isSized(ty) && "Cannot getTypeInfo() on a type that is unsized!");

  if (auto recordTy = llvm::dyn_cast<cir::RecordType>(ty)) {
    // FIXME(cir): CIR record's data layout implementation doesn't do a good job
    // of handling unions particularities. We should have a separate union type.
    return recordTy.getTypeSizeInBits(layout, {});
  }

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!cir::MissingFeatures::addressSpace());

  // This is calling mlir::DataLayout::getTypeSizeInBits().
  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(ty));
}
