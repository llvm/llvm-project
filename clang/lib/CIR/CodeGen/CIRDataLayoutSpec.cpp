//===--- CIRDataLayoutSpec.cpp - DLTI data layout for CIR modules ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file attaches the DLTI data-layout spec, including the CIR-native
// pointer entry, to a CIR module.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/CIRDataLayoutSpec.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DataLayout.h"

void cir::setMLIRDataLayout(mlir::ModuleOp mod, const llvm::DataLayout &dl) {
  mlir::MLIRContext *mlirContext = mod.getContext();
  mlir::DataLayoutSpecInterface dlSpec =
      mlir::translateDataLayout(dl, mlirContext);

  // Discover all address spaces that have explicit pointer specifications in
  // the data layout. Address space 0 is always included as the default.
  llvm::SmallSetVector<unsigned, 8> addrSpaces;
  addrSpaces.insert(0);

  for (mlir::DataLayoutEntryInterface entry : dlSpec.getEntries()) {
    if (!entry.isTypeEntry())
      continue;
    if (auto llvmPtrTy = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(
            mlir::cast<mlir::Type>(entry.getKey()))) {
      addrSpaces.insert(llvmPtrTy.getAddressSpace());
    }
  }

  constexpr unsigned kBitsInByte = 8;
  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries(
      dlSpec.getEntries().begin(), dlSpec.getEntries().end());

  for (unsigned as : addrSpaces) {
    unsigned ptrSizeBits = dl.getPointerSizeInBits(as);
    unsigned ptrAbiBits = dl.getPointerABIAlignment(as).value() * kBitsInByte;
    unsigned ptrPrefBits = dl.getPointerPrefAlignment(as).value() * kBitsInByte;
    unsigned ptrIndexBits = dl.getIndexSizeInBits(as);

    cir::PointerType ptrKey;
    if (as == 0) {
      ptrKey = cir::PointerType::get(cir::VoidType::get(mlirContext));
    } else {
      auto asAttr = cir::TargetAddressSpaceAttr::get(mlirContext, as);
      ptrKey = cir::PointerType::get(cir::VoidType::get(mlirContext), asAttr);
    }

    auto ptrSpec = cir::PtrSpecAttr::get(mlirContext, ptrSizeBits, ptrAbiBits,
                                         ptrPrefBits, ptrIndexBits);
    entries.push_back(mlir::DataLayoutEntryAttr::get(ptrKey, ptrSpec));
  }

  mod->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
               mlir::DataLayoutSpecAttr::get(mlirContext, entries));
}
