//===- SourcePassAnalysis.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

// FIXME: Expectations on SourcePassAnalysis results are TBD.  For a source pass
// that associates WPA results to AST, the result type is simply void; for
// source rewriting tools, it may be serializable CodeReplacements; for
// diagnostic tools, ti maybe SARIF.

///  A SourcePassAnalysis applies WholeProgramAnalysis results to ASTs.
///  Therefore, it is an `ASTConsumer` that depends on a set of
///  `clang::ssaf::AnalysisResult`s.
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
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_SOURCEPASSANALYSIS_SOURCEPASSANALYSIS_H
