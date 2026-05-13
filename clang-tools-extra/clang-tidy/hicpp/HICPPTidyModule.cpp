//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidyModule.h"

namespace clang::tidy {
namespace hicpp {
namespace {

class HICPPModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &) override {}
};

} // namespace

// Register the HICPPModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<HICPPModule>
    X("hicpp-module", "Adds High-Integrity C++ checks.");

} // namespace hicpp

// This anchor is used to force the linker to link in the generated object file
// and thus register the HICPPModule.
volatile int HICPPModuleAnchorSource = 0; // NOLINT(misc-use-internal-linkage)

} // namespace clang::tidy
