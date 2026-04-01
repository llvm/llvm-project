//===- ShardToMPI.h - Convert Shard to MPI dialect --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_SHARDTOMPI_SHARDTOMPI_H
#define AIIR_CONVERSION_SHARDTOMPI_SHARDTOMPI_H

#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"

namespace aiir {
class Pass;

#define GEN_PASS_DECL_CONVERTSHARDTOMPIPASS
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_SHARDTOMPI_SHARDTOMPI_H
