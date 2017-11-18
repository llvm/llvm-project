//===--- BugproneTidyModule.cpp - clang-tidy ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "CopyConstructorInitCheck.h"
#include "IntegerDivisionCheck.h"
#include "SuspiciousMemsetUsageCheck.h"
#include "UndefinedMemoryManipulationCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

class BugproneModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<CopyConstructorInitCheck>(
        "bugprone-copy-constructor-init");
    CheckFactories.registerCheck<IntegerDivisionCheck>(
        "bugprone-integer-division");
    CheckFactories.registerCheck<SuspiciousMemsetUsageCheck>(
        "bugprone-suspicious-memset-usage");
    CheckFactories.registerCheck<UndefinedMemoryManipulationCheck>(
        "bugprone-undefined-memory-manipulation");
  }
};

} // namespace bugprone

// Register the BugproneTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<bugprone::BugproneModule>
    X("bugprone-module", "Adds checks for bugprone code constructs.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the BugproneModule.
volatile int BugproneModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
