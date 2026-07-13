//===- OperatorNewDeletePointers.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares data structures for analysis that identifies pointer entities in
// operator new/delete overloads that must have a 'void*' type
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_OPERATORNEWDELETE_OPERATORNEWDELETEPOINTERS_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_OPERATORNEWDELETE_OPERATORNEWDELETEPOINTERS_H

#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "llvm/ADT/StringRef.h"
#include <set>

namespace clang::ssaf {

/// \brief Collects specific pointer entities related to operator new and delete
/// overloads within a contributor.
///
/// OperatorNewDeletePointersEntitySummary collects the following entities:
///  -# The returned entities of `operator new` overloads.
///  -# The parameter (optionally the second) of `operator new` overloads
///     representing the pointer to a memory area at which to initialize the
///     object.
///  -# The first parameter of `operator delete` overloads representing the
///  pointer
///     to a memory block to deallocate (or a null pointer).
///  -# The parameter (optionally the second) of `operator delete` overloads
///     representing the pointer used as the placement parameter in the matching
///     placement `new`.
struct OperatorNewDeletePointersEntitySummary final : public EntitySummary {
  static constexpr llvm::StringLiteral Name = "OperatorNewDeletePointers";

  static SummaryName summaryName() { return SummaryName(Name.str()); }

  SummaryName getSummaryName() const override { return summaryName(); }

  bool friend operator==(const OperatorNewDeletePointersEntitySummary &This,
                         const OperatorNewDeletePointersEntitySummary &Other) {
    return This.Entities == Other.Entities;
  }

  bool operator==(const std::set<EntityId> &OtherEntities) const {
    return Entities == OtherEntities;
  }

  bool empty() const { return Entities.empty(); }

  std::set<EntityId> Entities;
};

/// \brief Whole-program result of the operator new/delete pointer
/// analysis.
struct OperatorNewDeletePointersAnalysisResult final : AnalysisResult {
  static constexpr llvm::StringLiteral Name =
      "OperatorNewDeletePointersAnalysisResult";

  static AnalysisName analysisName() { return AnalysisName(Name.str()); }

  std::set<EntityId> Entities;

  bool contains(const EntityId &Id) const {
    return Entities.find(Id) != Entities.end();
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_OPERATORNEWDELETE_OPERATORNEWDELETEPOINTERS_H
