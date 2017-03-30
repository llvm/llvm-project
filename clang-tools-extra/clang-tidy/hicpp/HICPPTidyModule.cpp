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
#include "../cppcoreguidelines/ProTypeMemberInitCheck.h"
#include "../cppcoreguidelines/SpecialMemberFunctionsCheck.h"
#include "../google/DefaultArgumentsCheck.h"
#include "../google/ExplicitConstructorCheck.h"
#include "../misc/NewDeleteOverloadsCheck.h"
#include "../misc/NoexceptMoveConstructorCheck.h"
#include "../misc/UndelegatedConstructor.h"
#include "../misc/UseAfterMoveCheck.h"
#include "../modernize/UseEqualsDefaultCheck.h"
#include "../modernize/UseEqualsDeleteCheck.h"
#include "../modernize/UseOverrideCheck.h"
#include "../readability/FunctionSizeCheck.h"
#include "../readability/IdentifierNamingCheck.h"
#include "NoAssemblerCheck.h"

namespace clang {
namespace tidy {
namespace hicpp {

class HICPPModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<google::ExplicitConstructorCheck>(
        "hicpp-explicit-conversions");
    CheckFactories.registerCheck<readability::FunctionSizeCheck>(
        "hicpp-function-size");
    CheckFactories.registerCheck<readability::IdentifierNamingCheck>(
        "hicpp-named-parameter");
    CheckFactories.registerCheck<misc::UseAfterMoveCheck>(
        "hicpp-invalid-access-moved");
    CheckFactories.registerCheck<cppcoreguidelines::ProTypeMemberInitCheck>(
        "hicpp-member-init");
    CheckFactories.registerCheck<misc::NewDeleteOverloadsCheck>(
        "hicpp-new-delete-operators");
    CheckFactories.registerCheck<misc::NoexceptMoveConstructorCheck>(
        "hicpp-noexcept-move");
    CheckFactories.registerCheck<NoAssemblerCheck>("hicpp-no-assembler");
    CheckFactories
        .registerCheck<cppcoreguidelines::SpecialMemberFunctionsCheck>(
            "hicpp-special-member-functions");
    CheckFactories.registerCheck<misc::UndelegatedConstructorCheck>(
        "hicpp-undelegated-constructor");
    CheckFactories.registerCheck<modernize::UseEqualsDefaultCheck>(
        "hicpp-use-equals-default");
    CheckFactories.registerCheck<modernize::UseEqualsDeleteCheck>(
        "hicpp-use-equals-delete");
    CheckFactories.registerCheck<modernize::UseOverrideCheck>(
        "hicpp-use-override");
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
