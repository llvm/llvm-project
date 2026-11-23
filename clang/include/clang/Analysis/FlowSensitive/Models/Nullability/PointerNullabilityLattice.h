//===-- PointerNullabilityLattice.h - Nullability lattice -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYLATTICE_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYLATTICE_H

#include "clang/Analysis/FlowSensitive/NoopLattice.h"

namespace clang::dataflow::nullability {

// The analysis does not need any lattice state of its own yet. Pointer
// nullability is tracked on `PointerValue`s in the `Environment`. This becomes
// a real lattice once it needs non-flow-sensitive state and to cache const
// member call results.
using PointerNullabilityLattice = NoopLattice;

} // namespace clang::dataflow::nullability

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYLATTICE_H
