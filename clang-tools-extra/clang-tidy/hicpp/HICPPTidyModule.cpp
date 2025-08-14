//===------- HICPPTidyModule.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../bugprone/UndelegatedConstructorCheck.h"
#include "../bugprone/UseAfterMoveCheck.h"
#include "../cppcoreguidelines/AvoidGotoCheck.h"
#include "../cppcoreguidelines/NoMallocCheck.h"
#include "../cppcoreguidelines/ProBoundsArrayToPointerDecayCheck.h"
#include "../cppcoreguidelines/ProTypeMemberInitCheck.h"
#include "../cppcoreguidelines/ProTypeVarargCheck.h"
#include "../cppcoreguidelines/SpecialMemberFunctionsCheck.h"
#include "../google/ExplicitConstructorCheck.h"
#include "../misc/NewDeleteOverloadsCheck.h"
#include "../misc/StaticAssertCheck.h"
#include "../modernize/AvoidCArraysCheck.h"
#include "../modernize/DeprecatedHeadersCheck.h"
#include "../modernize/UseAutoCheck.h"
#include "../modernize/UseEmplaceCheck.h"
#include "../modernize/UseEqualsDefaultCheck.h"
#include "../modernize/UseEqualsDeleteCheck.h"
#include "../modernize/UseNoexceptCheck.h"
#include "../modernize/UseNullptrCheck.h"
#include "../modernize/UseOverrideCheck.h"
#include "../performance/MoveConstArgCheck.h"
#include "../performance/NoexceptMoveConstructorCheck.h"
#include "../readability/BracesAroundStatementsCheck.h"
#include "../readability/FunctionSizeCheck.h"
#include "../readability/NamedParameterCheck.h"
#include "../readability/UppercaseLiteralSuffixCheck.h"
#include "ExceptionBaseclassCheck.h"
#include "IgnoredRemoveResultCheck.h"
#include "MultiwayPathsCoveredCheck.h"
#include "NoAssemblerCheck.h"
#include "SignedBitwiseCheck.h"

namespace clang::tidy {
namespace hicpp {

class HICPPModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<modernize::AvoidCArraysCheck>(
        "hicpp-avoid-c-arrays");
    CheckFactories.registerCheck<cppcoreguidelines::AvoidGotoCheck>(
        "hicpp-avoid-goto");
    CheckFactories.registerCheck<readability::BracesAroundStatementsCheck>(
        "hicpp-braces-around-statements");
    CheckFactories.registerCheck<modernize::DeprecatedHeadersCheck>(
        "hicpp-deprecated-headers");
    CheckFactories.registerCheck<ExceptionBaseclassCheck>(
        "hicpp-exception-baseclass");
    CheckFactories.registerCheck<IgnoredRemoveResultCheck>(
        "hicpp-ignored-remove-result");
    CheckFactories.registerCheck<MultiwayPathsCoveredCheck>(
        "hicpp-multiway-paths-covered");
    CheckFactories.registerCheck<SignedBitwiseCheck>("hicpp-signed-bitwise");
    CheckFactories.registerCheck<google::ExplicitConstructorCheck>(
        "hicpp-explicit-conversions");
    CheckFactories.registerCheck<readability::FunctionSizeCheck>(
        "hicpp-function-size");
    CheckFactories.registerCheck<readability::NamedParameterCheck>(
        "hicpp-named-parameter");
    CheckFactories.registerCheck<bugprone::UseAfterMoveCheck>(
        "hicpp-invalid-access-moved");
    CheckFactories.registerCheck<cppcoreguidelines::ProTypeMemberInitCheck>(
        "hicpp-member-init");
    CheckFactories.registerCheck<performance::MoveConstArgCheck>(
        "hicpp-move-const-arg");
    CheckFactories.registerCheck<misc::NewDeleteOverloadsCheck>(
        "hicpp-new-delete-operators");
    CheckFactories.registerCheck<performance::NoexceptMoveConstructorCheck>(
        "hicpp-noexcept-move");
    CheckFactories
        .registerCheck<cppcoreguidelines::ProBoundsArrayToPointerDecayCheck>(
            "hicpp-no-array-decay");
    CheckFactories.registerCheck<NoAssemblerCheck>("hicpp-no-assembler");
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

// Register the HICPPModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<HICPPModule>
    X("hicpp-module", "Adds High-Integrity C++ checks.");

} // namespace hicpp

// This anchor is used to force the linker to link in the generated object file
// and thus register the HICPPModule.
volatile int HICPPModuleAnchorSource = 0; // NOLINT(misc-use-internal-linkage)

} // namespace clang::tidy
