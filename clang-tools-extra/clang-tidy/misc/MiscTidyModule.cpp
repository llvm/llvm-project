//===--- MiscTidyModule.cpp - clang-tidy ----------------------------------===//
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
#include "ArgumentCommentCheck.h"
#include "AssertSideEffectCheck.h"
#include "AssignOperatorSignatureCheck.h"
#include "BoolPointerImplicitConversionCheck.h"
#include "InaccurateEraseCheck.h"
#include "InefficientAlgorithmCheck.h"
#include "MacroParenthesesCheck.h"
#include "MacroRepeatedSideEffectsCheck.h"
#include "MoveConstantArgumentCheck.h"
#include "MoveConstructorInitCheck.h"
#include "NewDeleteOverloadsCheck.h"
#include "NoexceptMoveConstructorCheck.h"
#include "NonCopyableObjects.h"
#include "SizeofContainerCheck.h"
#include "StaticAssertCheck.h"
#include "StringIntegerAssignmentCheck.h"
#include "SwappedArgumentsCheck.h"
#include "ThrowByValueCatchByReferenceCheck.h"
#include "UndelegatedConstructor.h"
#include "UniqueptrResetReleaseCheck.h"
#include "UnusedAliasDeclsCheck.h"
#include "UnusedParametersCheck.h"
#include "UnusedRAIICheck.h"

namespace clang {
namespace tidy {
namespace misc {

class MiscModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ArgumentCommentCheck>("misc-argument-comment");
    CheckFactories.registerCheck<AssertSideEffectCheck>(
        "misc-assert-side-effect");
    CheckFactories.registerCheck<AssignOperatorSignatureCheck>(
        "misc-assign-operator-signature");
    CheckFactories.registerCheck<BoolPointerImplicitConversionCheck>(
        "misc-bool-pointer-implicit-conversion");
    CheckFactories.registerCheck<InaccurateEraseCheck>(
        "misc-inaccurate-erase");
    CheckFactories.registerCheck<InefficientAlgorithmCheck>(
        "misc-inefficient-algorithm");
    CheckFactories.registerCheck<MacroParenthesesCheck>(
        "misc-macro-parentheses");
    CheckFactories.registerCheck<MacroRepeatedSideEffectsCheck>(
        "misc-macro-repeated-side-effects");
    CheckFactories.registerCheck<MoveConstantArgumentCheck>(
        "misc-move-const-arg");
    CheckFactories.registerCheck<MoveConstructorInitCheck>(
        "misc-move-constructor-init");
    CheckFactories.registerCheck<NewDeleteOverloadsCheck>(
        "misc-new-delete-overloads");
    CheckFactories.registerCheck<NoexceptMoveConstructorCheck>(
        "misc-noexcept-move-constructor");
    CheckFactories.registerCheck<NonCopyableObjectsCheck>(
        "misc-non-copyable-objects");
    CheckFactories.registerCheck<SizeofContainerCheck>("misc-sizeof-container");
    CheckFactories.registerCheck<StaticAssertCheck>(
        "misc-static-assert");
    CheckFactories.registerCheck<StringIntegerAssignmentCheck>(
        "misc-string-integer-assignment");
    CheckFactories.registerCheck<SwappedArgumentsCheck>(
        "misc-swapped-arguments");
    CheckFactories.registerCheck<ThrowByValueCatchByReferenceCheck>(
        "misc-throw-by-value-catch-by-reference");
    CheckFactories.registerCheck<UndelegatedConstructorCheck>(
        "misc-undelegated-constructor");
    CheckFactories.registerCheck<UniqueptrResetReleaseCheck>(
        "misc-uniqueptr-reset-release");
    CheckFactories.registerCheck<UnusedAliasDeclsCheck>(
        "misc-unused-alias-decls");
    CheckFactories.registerCheck<UnusedParametersCheck>(
        "misc-unused-parameters");
    CheckFactories.registerCheck<UnusedRAIICheck>("misc-unused-raii");
  }
};

} // namespace misc

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<misc::MiscModule>
X("misc-module", "Adds miscellaneous lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the MiscModule.
volatile int MiscModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
