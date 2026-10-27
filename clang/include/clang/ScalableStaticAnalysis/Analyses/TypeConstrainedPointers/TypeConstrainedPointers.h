//===- TypeConstrainedPointers.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares data structures for the Type Constrained Pointers analysis, which
// identifies pointer entities that must retain their pointer type throughout
// the program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_TYPECONSTRAINEDPOINTERS_TYPECONSTRAINEDPOINTERS_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_TYPECONSTRAINEDPOINTERS_TYPECONSTRAINEDPOINTERS_H

#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "llvm/ADT/StringRef.h"
#include <set>

namespace clang::ssaf {

/// Per-contributor set of pointer entities that must retain their pointer type.
///
/// From `operator new` / `operator delete` overloads:
///  -# The return entity of `operator new` overloads.
///  -# The second parameter of `operator new(size_t, void*)` representing the
///     pointer to the memory area at which to initialize the object.
///  -# The first parameter of `operator delete` overloads representing the
///     pointer to the memory block to deallocate (or a null pointer).
///  -# The second parameter of `operator delete(void*, void*)` representing
///     the placement pointer matching the corresponding placement `new`.
///
/// From the `main` function:
///  -# Pointer-typed parameters of `main`.
struct TypeConstrainedPointersEntitySummary final : public EntitySummary {
  static constexpr llvm::StringLiteral Name = "TypeConstrainedPointers";

  static SummaryName summaryName() { return SummaryName(Name.str()); }

  SummaryName getSummaryName() const override { return summaryName(); }

  bool friend operator==(const TypeConstrainedPointersEntitySummary &This,
                         const TypeConstrainedPointersEntitySummary &Other) {
    return This.Entities == Other.Entities;
  }

  bool operator==(const std::set<EntityId> &OtherEntities) const {
    return Entities == OtherEntities;
  }

  bool empty() const { return Entities.empty(); }

  std::set<EntityId> Entities;
};

/// Whole-program set of pointer entities that must retain their pointer type.
struct TypeConstrainedPointersAnalysisResult final : AnalysisResult {
  static constexpr llvm::StringLiteral Name =
      "TypeConstrainedPointersAnalysisResult";

  static AnalysisName analysisName() { return AnalysisName(Name.str()); }

  std::set<EntityId> Entities;

  bool contains(const EntityId &Id) const {
    return Entities.find(Id) != Entities.end();
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_TYPECONSTRAINEDPOINTERS_TYPECONSTRAINEDPOINTERS_H
