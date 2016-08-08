//===--- CppCoreGuidelinesModule.cpp - clang-tidy -------------------------===//
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
#include "../misc/UnconventionalAssignOperatorCheck.h"
#include "InterfacesGlobalInitCheck.h"
#include "ProBoundsArrayToPointerDecayCheck.h"
#include "ProBoundsConstantArrayIndexCheck.h"
#include "ProBoundsPointerArithmeticCheck.h"
#include "ProTypeConstCastCheck.h"
#include "ProTypeCstyleCastCheck.h"
#include "ProTypeMemberInitCheck.h"
#include "ProTypeReinterpretCastCheck.h"
#include "ProTypeStaticCastDowncastCheck.h"
#include "ProTypeUnionAccessCheck.h"
#include "ProTypeVarargCheck.h"
#include "SpecialMemberFunctionsCheck.h"
#include "SlicingCheck.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// A module containing checks of the C++ Core Guidelines
class CppCoreGuidelinesModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<InterfacesGlobalInitCheck>(
        "cppcoreguidelines-interfaces-global-init");
    CheckFactories.registerCheck<ProBoundsArrayToPointerDecayCheck>(
        "cppcoreguidelines-pro-bounds-array-to-pointer-decay");
    CheckFactories.registerCheck<ProBoundsConstantArrayIndexCheck>(
        "cppcoreguidelines-pro-bounds-constant-array-index");
    CheckFactories.registerCheck<ProBoundsPointerArithmeticCheck>(
        "cppcoreguidelines-pro-bounds-pointer-arithmetic");
    CheckFactories.registerCheck<ProTypeConstCastCheck>(
        "cppcoreguidelines-pro-type-const-cast");
    CheckFactories.registerCheck<ProTypeCstyleCastCheck>(
        "cppcoreguidelines-pro-type-cstyle-cast");
    CheckFactories.registerCheck<ProTypeMemberInitCheck>(
        "cppcoreguidelines-pro-type-member-init");
    CheckFactories.registerCheck<ProTypeReinterpretCastCheck>(
        "cppcoreguidelines-pro-type-reinterpret-cast");
    CheckFactories.registerCheck<ProTypeStaticCastDowncastCheck>(
        "cppcoreguidelines-pro-type-static-cast-downcast");
    CheckFactories.registerCheck<ProTypeUnionAccessCheck>(
        "cppcoreguidelines-pro-type-union-access");
    CheckFactories.registerCheck<ProTypeVarargCheck>(
        "cppcoreguidelines-pro-type-vararg");
    CheckFactories.registerCheck<SpecialMemberFunctionsCheck>(
        "cppcoreguidelines-special-member-functions");
    CheckFactories.registerCheck<SlicingCheck>(
        "cppcoreguidelines-slicing");
    CheckFactories.registerCheck<misc::UnconventionalAssignOperatorCheck>(
        "cppcoreguidelines-c-copy-assignment-signature");
  }
};

// Register the LLVMTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<CppCoreGuidelinesModule>
    X("cppcoreguidelines-module", "Adds checks for the C++ Core Guidelines.");

} // namespace cppcoreguidelines

// This anchor is used to force the linker to link in the generated object file
// and thus register the CppCoreGuidelinesModule.
volatile int CppCoreGuidelinesModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
