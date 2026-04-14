//===- AnalysisResults.h - Shared analysis result types ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXAMPLE_PLUGIN_ANALYSIS_RESULTS_H
#define EXAMPLE_PLUGIN_ANALYSIS_RESULTS_H

#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include <string>
#include <utility>
#include <vector>

namespace example_plugin {

struct TagsAnalysisResult final : clang::ssaf::AnalysisResult {
  static clang::ssaf::AnalysisName analysisName() {
    return clang::ssaf::AnalysisName("TagsAnalysisResult");
  }

  std::vector<std::string> Tags;
};

struct PairsAnalysisResult final : clang::ssaf::AnalysisResult {
  static clang::ssaf::AnalysisName analysisName() {
    return clang::ssaf::AnalysisName("PairsAnalysisResult");
  }

  std::vector<std::pair<clang::ssaf::EntityId, int>> PairCounts;
};

} // namespace example_plugin

#endif // EXAMPLE_PLUGIN_ANALYSIS_RESULTS_H
