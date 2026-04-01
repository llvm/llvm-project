//===-- Optimizer/Support/DataLayout.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_DATALAYOUT_H
#define FORTRAN_OPTIMIZER_SUPPORT_DATALAYOUT_H

#include "aiir/Interfaces/DataLayoutInterfaces.h"
#include <optional>

namespace aiir {
class ModuleOp;
namespace gpu {
class GPUModuleOp;
} // namespace gpu
} // namespace aiir

namespace llvm {
class DataLayout;
} // namespace llvm

namespace fir::support {
/// Create an aiir::DataLayoutSpecInterface attribute from an llvm::DataLayout
/// and set it on the provided aiir::ModuleOp.
/// Also set the llvm.data_layout attribute with the string representation of
/// the llvm::DataLayout on the module.
/// These attributes are replaced if they were already set.
void setAIIRDataLayout(aiir::ModuleOp aiirModule, const llvm::DataLayout &dl);
void setAIIRDataLayout(aiir::gpu::GPUModuleOp aiirModule,
                       const llvm::DataLayout &dl);

/// Create an aiir::DataLayoutSpecInterface from the llvm.data_layout attribute
/// if one is provided. If such attribute is not available, create a default
/// target independent layout when allowDefaultLayout is true. Otherwise do
/// nothing.
void setAIIRDataLayoutFromAttributes(aiir::ModuleOp aiirModule,
                                     bool allowDefaultLayout);
void setAIIRDataLayoutFromAttributes(aiir::gpu::GPUModuleOp aiirModule,
                                     bool allowDefaultLayout);

/// Create aiir::DataLayout from the data layout information on the
/// aiir::Module. Creates the data layout information attributes with
/// setAIIRDataLayoutFromAttributes if the DLTI attribute is not yet set. If no
/// information is present at all and \p allowDefaultLayout is false, returns
/// std::nullopt.
std::optional<aiir::DataLayout>
getOrSetAIIRDataLayout(aiir::ModuleOp aiirModule,
                       bool allowDefaultLayout = false);
std::optional<aiir::DataLayout>
getOrSetAIIRDataLayout(aiir::gpu::GPUModuleOp aiirModule,
                       bool allowDefaultLayout = false);

} // namespace fir::support

#endif // FORTRAN_OPTIMIZER_SUPPORT_DATALAYOUT_H
