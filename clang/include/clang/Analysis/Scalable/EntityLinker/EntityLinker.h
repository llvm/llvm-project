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
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "llvm/Support/Error.h"
#include <map>
#include <memory>
#include <vector>

namespace clang::ssaf {

class EntitySummaryEncoding;
class TUSummaryEncoding;

class EntityLinker {
  LUSummaryEncoding Output;

public:
  EntityLinker(NestedBuildNamespace LUNamespace)
      : Output(std::move(LUNamespace)) {}

  llvm::Error link(std::unique_ptr<TUSummaryEncoding> Summary);

  const LUSummaryEncoding &getOutput() const { return Output; }

private:
  llvm::Expected<EntityId> resolve(const EntityName &OldName,
                                   const EntityId OldId,
                                   const EntityLinkage &EL);

  llvm::Error
  merge(std::map<SummaryName,
                 std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
            &InputData,
        std::map<SummaryName,
                 std::map<EntityId, std::unique_ptr<EntitySummaryEncoding>>>
            &OutputData,
        const EntityId OldId, const EntityId NewId, const EntityLinkage &EL,
        std::vector<EntitySummaryEncoding *> &PatchTargets);

  void patch(std::vector<EntitySummaryEncoding *> &PatchTargets,
             const std::map<EntityId, EntityId> &EntityResolutionTable);
};

} // end namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_ENTITYLINKER_ENTITYLINKER_H
