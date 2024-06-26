#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "llvm/ADT/StringRef.h"

namespace cir {

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} {
  auto dlSpec = mlir::dyn_cast<mlir::DataLayoutSpecAttr>(
      modOp->getAttr(mlir::DLTIDialect::kDataLayoutAttrName));
  assert(dlSpec && "expected dl_spec in the module");
  auto entries = dlSpec.getEntries();

  for (auto entry : entries) {
    auto entryKey = entry.getKey();
    auto strKey = mlir::dyn_cast<mlir::StringAttr>(entryKey);
    if (!strKey)
      continue;
    auto entryName = strKey.strref();
    if (entryName == mlir::DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = mlir::dyn_cast<mlir::StringAttr>(entry.getValue());
      assert(value && "expected string attribute");
      auto endian = value.getValue();
      if (endian == mlir::DLTIDialect::kDataLayoutEndiannessBig)
        bigEndian = true;
      else if (endian == mlir::DLTIDialect::kDataLayoutEndiannessLittle)
        bigEndian = false;
      else
        llvm_unreachable("unknown endianess");
    }
  }
}

void CIRDataLayout::reset(llvm::StringRef Desc) { clear(); }

void CIRDataLayout::clear() {}

} // namespace cir
