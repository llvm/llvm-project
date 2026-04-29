//===- SourcePassAnalysis.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SourcePassAnalysis applies whole-program analysis (WPA) results to ASTs.
// A SourcePassAnalysis is an ASTConsumer that depends on a WPA AnalysisResult.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SOURCEPASSANALYSIS_SOURCEPASSANALYSIS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SOURCEPASSANALYSIS_SOURCEPASSANALYSIS_H

#include "clang/AST/ASTConsumer.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include <memory>
#include <utility>

namespace clang::ssaf {

class SourcePassAnalysisBase : public ASTConsumer {
public:
  SourcePassAnalysisBase(std::unique_ptr<WPASuite> WPAResult)
      : WPAResult(std::move(WPAResult)) {}

protected:
  std::unique_ptr<WPASuite> WPAResult;
};

// FIXME: Expectations on SourcePassAnalysis result types are TBD.  For a source
// pass that prints WPA results with respect to the AST, the result type can be
// void; for source rewriting tools, it may be serializable CodeReplacements;
// for diagnostic tools, it may be SARIF.

/// A SourcePassAnalysis applies the result of a whole-program analysis (WPA) to
/// ASTs. It depends on a single AnalysisResult.
/// If one finds the need that a SourcePassAnalysis depends on multiple
/// AnalysisResults, one should create a new WPA that depends on those analyses.
template <typename ResultT, typename DepResultT>
class SourcePassAnalysis : public SourcePassAnalysisBase {
  static_assert((std::is_base_of_v<AnalysisResult, DepResultT>),
                "Every DepResultT must derive from AnalysisResult");

public:
  using SourcePassAnalysisBase::SourcePassAnalysisBase;
  static AnalysisName analysisName();

protected:
  llvm::Expected<DepResultT> getDependentWPAResult() {
    return WPAResult->get<DepResultT>();
  }

  EntityIdTable &getEntityIdTable() {
    // FIXME: Need to cast away `const` to lookup via getId(); probably also
    // provide a lookupId() method:
    return const_cast<EntityIdTable &>(WPAResult->getIdTable());
  }

  NestedBuildNamespace getLUNamespace() { return WPAResult->getLUNamespace(); }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SOURCEPASSANALYSIS_SOURCEPASSANALYSIS_H
