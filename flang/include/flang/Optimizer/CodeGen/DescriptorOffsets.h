//===-- DescriptorOffsets.h -- offsets of descriptors fields ---*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DESCRIPTOR_OFFSETS_H
#define OPTIMIZER_DESCRIPTOR_OFFSETS_H

#include "flang/Optimizer/CodeGen/DescriptorModel.h"

namespace fir {

/// Calculate offset of any field in the descriptor.
template <int Field>
static std::uint64_t getDescComponentOffset(const mlir::DataLayout &dl,
                                            mlir::MLIRContext *context,
                                            mlir::Type fieldType) {
  static_assert(Field > 0 && Field < 8);
  mlir::Type previousFieldType = getDescFieldTypeModel<Field - 1>()(context);
  std::uint64_t previousOffset =
      getDescComponentOffset<Field - 1>(dl, context, previousFieldType);
  std::uint64_t offset = previousOffset + dl.getTypeSize(previousFieldType);
  std::uint64_t fieldAlignment = dl.getTypeABIAlignment(fieldType);
  return llvm::alignTo(offset, fieldAlignment);
}

template <>
std::uint64_t getDescComponentOffset<0>(const mlir::DataLayout &dl,
                                        mlir::MLIRContext *context,
                                        mlir::Type fieldType) {
  return 0;
}

} // namespace fir

#endif // OPTIMIZER_DESCRIPTOR_OFFSETS_H
