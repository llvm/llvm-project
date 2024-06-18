//===- CIRDialect.cpp - MLIR CIR ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CIR DataLayout class and its functions.
//
//===----------------------------------------------------------------------===//a
//

#include "clang/CIR/Dialect/IR/CIRDataLayout.h"

namespace cir {
CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} {
  auto dlSpec = modOp->getAttr(mlir::DLTIDialect::kDataLayoutAttrName)
                    .dyn_cast<mlir::DataLayoutSpecAttr>();
  assert(dlSpec && "expected dl_spec in the module");
  auto entries = dlSpec.getEntries();

  for (auto entry : entries) {
    auto entryKey = entry.getKey();
    auto strKey = entryKey.dyn_cast<mlir::StringAttr>();
    if (!strKey)
      continue;
    auto entryName = strKey.strref();
    if (entryName == mlir::DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = entry.getValue().dyn_cast<mlir::StringAttr>();
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

} // namespace cir
