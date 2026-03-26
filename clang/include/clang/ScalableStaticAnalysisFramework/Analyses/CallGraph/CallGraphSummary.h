//===- CallGraphSummary.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_CALLGRAPH_CALLGRAPHSUMMARY_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_CALLGRAPH_CALLGRAPHSUMMARY_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/EntitySummary.h"
#include <set>

namespace clang::ssaf {

/// Summary of direct call-graph edges for a single function entity.
///
/// Represents a function definition, and information about its callees.
struct CallGraphSummary final : public EntitySummary {
  struct Location {
    std::string File;
    unsigned Line;
    unsigned Column;
  };

  SummaryName getSummaryName() const override {
    return SummaryName("CallGraph");
  }

  /// Represents the location of the function.
  Location Definition = {};

  /// The set of direct callees of this function.
  std::set<EntityId> DirectCallees;

  /// A human-readable name of the function.
  /// This is not guaranteed to be accurate or unique.
  std::string PrettyName;

  /// Whether this function contains calls that could not be resolved to a
  /// direct callee.
  /// E.g. virtual method calls, or calls through function pointers.
  bool HasIndirectCalls = false;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_CALLGRAPH_CALLGRAPHSUMMARY_H
