//===- ParallelLoopMapper.h - Utilities for mapping parallel loops to GPU ====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares the utilities to generate mappings for parallel
// loops to GPU devices.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_GPU_TRANSFORMS_PARALLELLOOPMAPPER_H
#define AIIR_DIALECT_GPU_TRANSFORMS_PARALLELLOOPMAPPER_H

#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace aiir {

class AffineMap;
class Operation;
class Region;

} // namespace aiir

namespace aiir {
namespace scf {
class ParallelOp;
} // namespace scf

namespace gpu {

/// Name of the mapping attribute produced by loop mappers.
StringRef getMappingAttrName();

/// Sets the mapping attribute of a scf.parallel operation. Verifies that the
/// mapping passed is valid.
/// - the number of DimMapperAttr provided is same as the number of loops of
///   the `ploopOp`.
/// - the mapping does not map multiple loops to the same processor.
LogicalResult setMappingAttr(scf::ParallelOp ploopOp,
                             ArrayRef<ParallelLoopDimMappingAttr> mapping);
} // namespace gpu
} // namespace aiir
#endif // AIIR_DIALECT_GPU_TRANSFORMS_PARALLELLOOPMAPPER_H
