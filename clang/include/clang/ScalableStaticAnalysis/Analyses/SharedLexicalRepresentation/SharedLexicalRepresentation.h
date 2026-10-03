//===- SharedLexicalRepresentation.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_SHAREDLEXICALREPRESENTATION_SHAREDLEXICALREPRESENTATION_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_SHAREDLEXICALREPRESENTATION_SHAREDLEXICALREPRESENTATION_H

#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <tuple>
#include <vector>

namespace clang::ssaf {

/// A canonical (file, line, column) triple for one declaration site.
struct SourceLocationRecord {
  std::string FilePath;
  unsigned Line = 0;
  unsigned Column = 0;

  bool operator==(const SourceLocationRecord &Other) const {
    return std::tie(FilePath, Line, Column) ==
           std::tie(Other.FilePath, Other.Line, Other.Column);
  }

  bool operator<(const SourceLocationRecord &Other) const {
    return std::tie(FilePath, Line, Column) <
           std::tie(Other.FilePath, Other.Line, Other.Column);
  }
};

/// An EntitySourceLocationsSummary contains one SourceLocationRecord per
/// declaration site contributed by an entity.
class EntitySourceLocationsSummary final : public EntitySummary {
  std::vector<SourceLocationRecord> DeclLocations;

  friend EntitySourceLocationsSummary
      buildEntitySourceLocationsSummary(std::vector<SourceLocationRecord>);
  friend llvm::ArrayRef<SourceLocationRecord>
  getDeclLocations(const EntitySourceLocationsSummary &);

  explicit EntitySourceLocationsSummary(
      std::vector<SourceLocationRecord> DeclLocations)
      : DeclLocations(std::move(DeclLocations)) {}

public:
  static constexpr llvm::StringLiteral Name = "EntitySourceLocations";

  SummaryName getSummaryName() const override { return summaryName(); }

  bool operator==(const EntitySourceLocationsSummary &Other) const {
    return DeclLocations == Other.DeclLocations;
  }

  bool empty() const { return DeclLocations.empty(); }

  static SummaryName summaryName() { return SummaryName{Name.str()}; }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_SHAREDLEXICALREPRESENTATION_SHAREDLEXICALREPRESENTATION_H
