//===--- AlteraTidyModule.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "IdDependentBackwardBranchCheck.h"
#include "KernelNameRestrictionCheck.h"
#include "SingleWorkItemBarrierCheck.h"
#include "StructPackAlignCheck.h"
#include "UnrollLoopsCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy {
namespace altera {

class AlteraModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<IdDependentBackwardBranchCheck>(
        "altera-id-dependent-backward-branch");
    CheckFactories.registerCheck<KernelNameRestrictionCheck>(
        "altera-kernel-name-restriction");
    CheckFactories.registerCheck<SingleWorkItemBarrierCheck>(
        "altera-single-work-item-barrier");
    CheckFactories.registerCheck<StructPackAlignCheck>(
        "altera-struct-pack-align");
    CheckFactories.registerCheck<UnrollLoopsCheck>("altera-unroll-loops");
  }
};

} // namespace altera

// Register the AlteraTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<altera::AlteraModule>
    X("altera-module", "Adds Altera FPGA OpenCL lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the AlteraModule.
volatile int AlteraModuleAnchorSource = 0;

} // namespace clang::tidy
