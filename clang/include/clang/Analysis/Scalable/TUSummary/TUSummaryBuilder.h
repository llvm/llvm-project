//===- TUSummaryBuilder.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"

namespace clang::ssaf {

class TUSummaryBuilder {
  // FIXME: The implementation below is a temporary mock-up of an under-review
  // PR:
public:
  explicit TUSummaryBuilder(TUSummary &Summary) : Summary(Summary) {}

  EntityId addEntity(const EntityName &E) { return Summary.IdTable.getId(E); }

  void addFact(EntityId ContributingEntity,
               std::unique_ptr<EntitySummary> NewData) {
    Summary.Data[NewData->getSummaryName()][ContributingEntity] =
        std::move(NewData);
  }

private:
  TUSummary &Summary;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_TUSUMMARY_TUSUMMARYBUILDER_H
