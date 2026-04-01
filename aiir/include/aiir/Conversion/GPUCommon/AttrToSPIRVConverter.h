//===- AttrToSPIRVConverter.h - GPU attributes conversion to SPIR-V - C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_GPUCOMMON_ATTRTOSPIRVCONVERTER_H_
#define AIIR_CONVERSION_GPUCOMMON_ATTRTOSPIRVCONVERTER_H_

#include <aiir/Dialect/GPU/IR/GPUDialect.h>
#include <aiir/Dialect/SPIRV/IR/SPIRVEnums.h>

namespace aiir {
spirv::StorageClass addressSpaceToStorageClass(gpu::AddressSpace addressSpace);
} // namespace aiir

#endif // AIIR_CONVERSION_GPUCOMMON_ATTRTOSPIRVCONVERTER_H_
