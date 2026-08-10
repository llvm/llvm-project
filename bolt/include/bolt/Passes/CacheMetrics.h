//===- bolt/Passes/CacheMetrics.h - Instruction cache metrics ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to show metrics of cache lines.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CACHEMETRICS_H
#define BOLT_PASSES_CACHEMETRICS_H

#include "bolt/Core/BinaryContext.h"
#include <vector>

namespace llvm {

class raw_ostream;

namespace bolt {
namespace CacheMetrics {

/// Calculate and print various metrics related to instruction cache performance
void printAll(raw_ostream &OS, const BinaryFunctionListType &BinaryFunctions);

} // namespace CacheMetrics
} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_CACHEMETRICS_H
