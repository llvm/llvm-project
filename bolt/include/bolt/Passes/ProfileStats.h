//===- bolt/Passes/ProfileStats.h - profile quality metrics ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to print profile stats to quantify profile quality.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_PROFILESTATS_H
#define BOLT_PASSES_PROFILESTATS_H

#include <vector>

namespace llvm {

class raw_ostream;

namespace bolt {
class BinaryContext;
namespace ProfileStats {

/// Calculate and print various metrics related to profile quality
void printAll(raw_ostream &OS, BinaryContext &BC);

} // namespace ProfileStats
} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_PROFILESTATS_H
