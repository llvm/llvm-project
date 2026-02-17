//===- EntityLinker.h - Class for linking entities --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the EntityLinker class that combines multiple TU summaries
//  into a unified LU summary by deduplicating entities and patching summaries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H

#include "clang/Analysis/Scalable/EntityLinker/LUSummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>
#include <vector>

namespace clang::ssaf {

class EntityLinkage;
class EntityName;
class EntitySummaryEncoding;
class TUSummaryEncoding;

class EntityLinker {
  LUSummaryEncoding Output;

public:
  /// Constructs an EntityLinker for a link unit.
  ///
  /// \param LUNamespace The namespace identifying this link unit.
  EntityLinker(NestedBuildNamespace LUNamespace)
      : Output(std::move(LUNamespace)) {}

  /// Links a translation unit summary into the link unit summary.
  ///
  /// Processes entity names, resolves namespace conflicts based on linkage,
  /// deduplicates entities, and patches entity ID references in the summary
  /// data. The provided TU summary is consumed by this operation.
  ///
  /// \param Summary The TU summary to link. Ownership is transferred.
  /// \returns Error if linking fails (e.g., duplicate internal entities,
  ///          missing linkage information), success otherwise.
  llvm::Error link(std::unique_ptr<TUSummaryEncoding> Summary);

  /// Returns the accumulated link unit summary.
  ///
  /// \returns A const reference to the linked output containing all
  ///          deduplicated and patched entity summaries.
  const LUSummaryEncoding &getOutput() const { return Output; }

private:
  llvm::Expected<EntityId> resolve(const EntityName &OldName,
                                   const EntityLinkage &Linkage);

  llvm::Error
  merge(std::map<SummaryName,
                 std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
            &InputData,
        std::map<SummaryName,
                 std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
            &OutputData,
        const EntityId OldId, const EntityId NewId,
        const EntityLinkage &Linkage,
        std::vector<EntitySummaryEncoding *> &PatchTargets);

  void patch(std::vector<EntitySummaryEncoding *> &PatchTargets,
             const std::map<EntityId, EntityId> &EntityResolutionTable);
};

} // end namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H
