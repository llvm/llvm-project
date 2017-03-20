//===------- HICPPTidyModule.cpp - clang-tidy -----------------------------===//
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
namespace hicpp {

class HICPPModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<NoAssemblerCheck>(
        "hicpp-no-assembler");
  }
};

// Register the HICPPModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<HICPPModule>
    X("hicpp-module", "Adds High-Integrity C++ checks.");

} // namespace hicpp

// This anchor is used to force the linker to link in the generated object file
// and thus register the HICPPModule.
volatile int HICPPModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
