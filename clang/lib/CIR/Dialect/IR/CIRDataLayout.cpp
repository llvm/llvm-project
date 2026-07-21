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

    mlir::StringAttr addrSpKey = mlir::StringAttr::get(
        spec.getContext(), mlir::DLTIDialect::kDataLayoutProgramMemorySpaceKey);
    if (mlir::DataLayoutEntryInterface entry =
            spec.getSpecForIdentifier(addrSpKey))
      if (auto val = llvm::dyn_cast<mlir::IntegerAttr>(entry.getValue()))
        programAddrSpace = val.getInt();
  }
}

llvm::Align CIRDataLayout::getAlignment(mlir::Type ty, bool useABIAlign) const {
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

  if (auto structTy = llvm::dyn_cast<cir::StructType>(ty))
    return structTy.getTypeSizeInBits(layout, {});
  if (auto unionTy = llvm::dyn_cast<cir::UnionType>(ty))
    return unionTy.getTypeSizeInBits(layout, {});

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!cir::MissingFeatures::addressSpace());

  // This is calling mlir::DataLayout::getTypeSizeInBits().
  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(ty));
}
