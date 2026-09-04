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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/IR/DataLayout.h"

void cir::setMLIRDataLayout(mlir::ModuleOp mod, const llvm::DataLayout &dl) {
  mlir::MLIRContext *mlirContext = mod.getContext();
  mlir::DataLayoutSpecInterface dlSpec =
      mlir::translateDataLayout(dl, mlirContext);

  // Append the !cir.ptr-keyed #cir.ptr_spec entry.
  // TODO(cir): only the default address space is recorded.
  assert(!cir::MissingFeatures::dataLayoutPtrHandlingBasedOnLangAS());
  constexpr unsigned kBitsInByte = 8;
  unsigned ptrSizeBits = dl.getPointerSizeInBits(/*AS=*/0);
  unsigned ptrAbiBits =
      dl.getPointerABIAlignment(/*AS=*/0).value() * kBitsInByte;
  unsigned ptrPrefBits =
      dl.getPointerPrefAlignment(/*AS=*/0).value() * kBitsInByte;
  unsigned ptrIndexBits = dl.getIndexSizeInBits(/*AS=*/0);
  auto ptrKey = cir::PointerType::get(cir::VoidType::get(mlirContext));
  auto ptrSpec = cir::PtrSpecAttr::get(mlirContext, ptrSizeBits, ptrAbiBits,
                                       ptrPrefBits, ptrIndexBits);
  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries(
      dlSpec.getEntries().begin(), dlSpec.getEntries().end());
  entries.push_back(mlir::DataLayoutEntryAttr::get(ptrKey, ptrSpec));

  mod->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
               mlir::DataLayoutSpecAttr::get(mlirContext, entries));
}
