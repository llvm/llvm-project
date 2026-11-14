//===- CppBoundedBuffers.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The cpp-bounded-buffers transformation rewrites buffers -- raw pointers and
// arrays -- reachable from unsafe buffer usage into bounded types
// (bounded_ptr<T>, bounded_array<T, N>). Reachable declarators that are not
// rewritten are reported instead, so no reachable buffer is silently left raw.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONS_CPPBOUNDEDBUFFERS_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONS_CPPBOUNDEDBUFFERS_H

#include "clang/AST/Type.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace clang {
class ASTContext;
} // namespace clang

namespace clang::ssaf {

/// The bounded type a raw declarator is rewritten to.
enum class BoundedType { Ptr, Array };

/// Why a reachable declarator was reported instead of rewritten.
enum class ReportReason {
  ArrayNotEndInBracket,
  DeclarationGroup,
  EmissionFailed,
  IncompleteArray,
  MacroExpansion,
  MultiDimensionalArray,
  MultiLevelPointer,
  NoInnerTypeLoc,
  NotPointerTypeEndWithStar,
  NotTransformed,
  PointerToArray,
  ReferenceToPointer,
  TrailingReturnType,
  UnexpectedLeadingQualifier,
  UnexpectedTrailingQualifier,
  UnnamableType,
};

/// Returns the report message for \p Reason.
llvm::StringRef messageFor(ReportReason Reason);

/// The outcome of classifying a declared type against the reachable pointer
/// levels of its entity: a bounded-type rewrite, or a report reason.
struct ClassifyResult {
  // Meaningful only when Skip is nullopt.
  BoundedType NewType = BoundedType::Ptr;
  // Pointee/element spelling; meaningful only when Skip is nullopt.
  std::string InnerSpelling;
  std::optional<ReportReason> Skip = ReportReason::NotTransformed;
};

/// Classifies the declared type \p T of a reachable entity. \p ReachableLevels
/// holds the entity's reachable pointer levels (1-based, outermost is level 1).
ClassifyResult
classifyDeclType(QualType T, const llvm::SmallSet<unsigned, 4> &ReachableLevels,
                 const ASTContext &Ctx);

class CppBoundedBuffers final : public Transformation {
public:
  using Transformation::Transformation;

  void HandleTranslationUnit(clang::ASTContext &Ctx) override;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_SOURCETRANSFORMATION_TRANSFORMATIONS_CPPBOUNDEDBUFFERS_H
