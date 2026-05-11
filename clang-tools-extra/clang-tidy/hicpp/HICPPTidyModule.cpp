//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../bugprone/UndelegatedConstructorCheck.h"
#include "../bugprone/UnhandledCodePathsCheck.h"
#include "../cppcoreguidelines/NoMallocCheck.h"
#include "../cppcoreguidelines/ProTypeVarargCheck.h"
#include "../cppcoreguidelines/SpecialMemberFunctionsCheck.h"
#include "../misc/StaticAssertCheck.h"
#include "../modernize/UseAutoCheck.h"
#include "../modernize/UseEmplaceCheck.h"
#include "../modernize/UseEqualsDefaultCheck.h"
#include "../modernize/UseEqualsDeleteCheck.h"
#include "../modernize/UseNoexceptCheck.h"
#include "../modernize/UseNullptrCheck.h"
#include "../modernize/UseOverrideCheck.h"
#include "../portability/NoAssemblerCheck.h"
#include "../readability/UppercaseLiteralSuffixCheck.h"

namespace clang::tidy {
namespace hicpp {
namespace {

class HICPPModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<bugprone::UnhandledCodePathsCheck>(
        "hicpp-multiway-paths-covered");
    CheckFactories.registerCheck<portability::NoAssemblerCheck>(
        "hicpp-no-assembler");
    CheckFactories.registerCheck<cppcoreguidelines::NoMallocCheck>(
        "hicpp-no-malloc");
    CheckFactories
        .registerCheck<cppcoreguidelines::SpecialMemberFunctionsCheck>(
            "hicpp-special-member-functions");
    CheckFactories.registerCheck<misc::StaticAssertCheck>(
        "hicpp-static-assert");
    CheckFactories.registerCheck<modernize::UseAutoCheck>("hicpp-use-auto");
    CheckFactories.registerCheck<bugprone::UndelegatedConstructorCheck>(
        "hicpp-undelegated-constructor");
    CheckFactories.registerCheck<modernize::UseEmplaceCheck>(
        "hicpp-use-emplace");
    CheckFactories.registerCheck<modernize::UseEqualsDefaultCheck>(
        "hicpp-use-equals-default");
    CheckFactories.registerCheck<modernize::UseEqualsDeleteCheck>(
        "hicpp-use-equals-delete");
    CheckFactories.registerCheck<modernize::UseNoexceptCheck>(
        "hicpp-use-noexcept");
    CheckFactories.registerCheck<modernize::UseNullptrCheck>(
        "hicpp-use-nullptr");
    CheckFactories.registerCheck<modernize::UseOverrideCheck>(
        "hicpp-use-override");
    CheckFactories.registerCheck<readability::UppercaseLiteralSuffixCheck>(
        "hicpp-uppercase-literal-suffix");
    CheckFactories.registerCheck<cppcoreguidelines::ProTypeVarargCheck>(
        "hicpp-vararg");
  }
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
