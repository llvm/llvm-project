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
///
/// \bug Indirect calls (e.g. function pointers) are not represented.
/// \bug ObjCMessageExprs are not represented.
/// \bug Primary template functions are not represented.
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

  /// The set of virtual callees of this function.
  std::set<EntityId> VirtualCallees;

  /// A human-readable name of the function.
  /// This is not guaranteed to be accurate or unique.
  std::string PrettyName;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_CALLGRAPH_CALLGRAPHSUMMARY_H
