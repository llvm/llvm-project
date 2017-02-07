//===------- SafetyTidyModule.cpp - clang-tidy ----------------------------===//
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
#include "NoAssemblerCheck.h"

namespace clang {
namespace tidy {
namespace safety {

class SafetyModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<NoAssemblerCheck>(
        "safety-no-assembler");
  }
};

// Register the SafetyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<SafetyModule>
    X("safety-module", "Adds safety-critical checks.");

} // namespace safety

// This anchor is used to force the linker to link in the generated object file
// and thus register the SafetyModule.
volatile int SafetyModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
