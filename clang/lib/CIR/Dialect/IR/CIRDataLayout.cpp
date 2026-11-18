#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cir;

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp)
    : module(modOp), layout(modOp) {
  reset(modOp.getDataLayoutSpec());
}

void CIRDataLayout::reset(mlir::DataLayoutSpecInterface spec) {
  bigEndian = false;
  bitIntMaxAlignBits = cir::IntType::getBitIntMaxAlignBits({});
  if (spec) {
    bitIntMaxAlignBits = cir::IntType::getBitIntMaxAlignBits(
        spec.getSpecForType(cir::IntType::getTypeID()));

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
        programAddrSpace = val.getUInt();
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

uint64_t CIRDataLayout::getTypeLanguageAlignment(mlir::Type ty) const {
  auto maxMemberAlignment = [&](llvm::ArrayRef<mlir::Type> members) {
    uint64_t align = 1;
    for (mlir::Type member : members)
      align = std::max(align, getTypeLanguageAlignment(member));
    return align;
  };
  return llvm::TypeSwitch<mlir::Type, uint64_t>(ty)
      .Case([&](cir::IntType intTy) -> uint64_t {
        if (intTy.isBitInt())
          return intTy.getBitIntAlignRequirement(bitIntMaxAlignBits);
        return layout.getTypeABIAlignment(intTy);
      })
      .Case([&](cir::RecordType recTy) -> uint64_t {
        if (cir::RecordLayoutAttr recLayout =
                cir::tryGetRecordLayout(module, recTy.getName()))
          return recLayout.getRecordAlign();
        // A record without a layout entry, such as one in hand-written CIR.
        assert(recTy.isComplete() && "Cannot get layout of incomplete records");
        if (recTy.getPacked())
          return 1;
        return maxMemberAlignment(recTy.getMembers());
      })
      .Case<cir::ArrayType, cir::ComplexType, cir::MatrixType>(
          [&](auto elementsTy) -> uint64_t {
            return getTypeLanguageAlignment(elementsTy.getElementType());
          })
      .Case([&](cir::BitFieldType bitFieldTy) -> uint64_t {
        if (mlir::Type storage = bitFieldTy.getStorageType())
          return getTypeLanguageAlignment(storage);
        return 1;
      })
      .Default([&](mlir::Type ty) { return layout.getTypeABIAlignment(ty); });
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
  return layout.getTypeSizeInBits(ty);
}
