//===-- CIRDataLayoutSpec.h - DLTI data layout for CIR modules --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a helper that attaches the DLTI data-layout spec,
// including the CIR-native pointer and _BitInt entries, to a CIR module.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_CIRDATALAYOUTSPEC_H
#define CLANG_CIR_CIRDATALAYOUTSPEC_H

namespace llvm {
class DataLayout;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace cir {

/// Translate \p dl into a DLTI data-layout spec and attach it to \p mod.
/// On top of the plain mlir::translateDataLayout entries this adds:
///
/// - a #cir.ptr_spec entry keyed on !cir.ptr, which CIR pointer types read
///   for their size and alignment. Without it pointer widths default to 64
///   bits.
/// - a #cir.bitint_spec entry keyed on !cir.int<u, 1, bitint>.
///   \p bitIntMaxAlign is TargetInfo::getBitIntMaxAlign(), in bits. _BitInt
///   ABI alignment clamps to that cap.
void setMLIRDataLayout(mlir::ModuleOp mod, const llvm::DataLayout &dl,
                       unsigned bitIntMaxAlign);

} // namespace cir

#endif // CLANG_CIR_CIRDATALAYOUTSPEC_H
