//=== FormulaSerialization.h - Formula De/Serialization support -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_FORMULA_SERIALIZATION_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_FORMULA_SERIALIZATION_H

#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <string>

namespace clang::dataflow {

/// Prints `F` to `OS` in a compact format, optimized for easy parsing
/// (deserialization) rather than human use.
void serializeFormula(const Formula &F, llvm::raw_ostream &OS);

/// Parses `Str` to build a serialized Formula.
/// @returns error on parse failure or if parsing does not fully consume `Str`.
/// @param A used to construct the formula components.
/// @param AtomMap maps serialized Atom identifiers (unsigned ints) to Atoms.
///        This map is provided by the caller to enable consistency across
///        multiple formulas in a single file.
llvm::Expected<const Formula *>
parseFormula(llvm::StringRef Str, Arena &A,
             llvm::DenseMap<unsigned, Atom> &AtomMap);

} // namespace clang::dataflow
#endif
