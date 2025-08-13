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

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
llvm::TypeSize CIRDataLayout::getTypeSizeInBits(mlir::Type ty) const {
  assert(!cir::MissingFeatures::dataLayoutTypeIsSized());

  if (auto recordTy = llvm::dyn_cast<cir::RecordType>(ty)) {
    // FIXME(cir): CIR record's data layout implementation doesn't do a good job
    // of handling unions particularities. We should have a separate union type.
    return recordTy.getTypeSizeInBits(layout, {});
  }

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!cir::MissingFeatures::addressSpace());

  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(ty));
}
