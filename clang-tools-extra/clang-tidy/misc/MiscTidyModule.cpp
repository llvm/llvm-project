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
#include "BoolPointerImplicitConversionCheck.h"
#include "DanglingHandleCheck.h"
#include "DefinitionsInHeadersCheck.h"
#include "FoldInitTypeCheck.h"
#include "ForwardDeclarationNamespaceCheck.h"
#include "ForwardingReferenceOverloadCheck.h"
#include "InaccurateEraseCheck.h"
#include "IncorrectRoundings.h"
#include "InefficientAlgorithmCheck.h"
#include "LambdaFunctionNameCheck.h"
#include "MacroParenthesesCheck.h"
#include "MacroRepeatedSideEffectsCheck.h"
#include "MisplacedConstCheck.h"
#include "MisplacedWideningCastCheck.h"
#include "MoveConstantArgumentCheck.h"
#include "MoveConstructorInitCheck.h"
#include "MoveForwardingReferenceCheck.h"
#include "MultipleStatementMacroCheck.h"
#include "NewDeleteOverloadsCheck.h"
#include "NoexceptMoveConstructorCheck.h"
#include "NonCopyableObjects.h"
#include "RedundantExpressionCheck.h"
#include "SizeofContainerCheck.h"
#include "SizeofExpressionCheck.h"
#include "StaticAssertCheck.h"
#include "StringCompareCheck.h"
#include "StringIntegerAssignmentCheck.h"
#include "StringLiteralWithEmbeddedNulCheck.h"
#include "SuspiciousEnumUsageCheck.h"
#include "SuspiciousMissingCommaCheck.h"
#include "SuspiciousSemicolonCheck.h"
#include "SuspiciousStringCompareCheck.h"
#include "SwappedArgumentsCheck.h"
#include "ThrowByValueCatchByReferenceCheck.h"
#include "UnconventionalAssignOperatorCheck.h"
#include "UndelegatedConstructor.h"
#include "UniqueptrResetReleaseCheck.h"
#include "UnusedAliasDeclsCheck.h"
#include "UnusedParametersCheck.h"
#include "UnusedRAIICheck.h"
#include "UnusedUsingDeclsCheck.h"
#include "UseAfterMoveCheck.h"
#include "VirtualNearMissCheck.h"

namespace clang {
namespace tidy {
namespace misc {

class MiscModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ArgumentCommentCheck>("misc-argument-comment");
    CheckFactories.registerCheck<AssertSideEffectCheck>(
        "misc-assert-side-effect");
    CheckFactories.registerCheck<ForwardingReferenceOverloadCheck>(
        "misc-forwarding-reference-overload");
    CheckFactories.registerCheck<LambdaFunctionNameCheck>(
        "misc-lambda-function-name");
    CheckFactories.registerCheck<MisplacedConstCheck>("misc-misplaced-const");
    CheckFactories.registerCheck<UnconventionalAssignOperatorCheck>(
        "misc-unconventional-assign-operator");
    CheckFactories.registerCheck<BoolPointerImplicitConversionCheck>(
        "misc-bool-pointer-implicit-conversion");
    CheckFactories.registerCheck<DanglingHandleCheck>("misc-dangling-handle");
    CheckFactories.registerCheck<DefinitionsInHeadersCheck>(
        "misc-definitions-in-headers");
    CheckFactories.registerCheck<FoldInitTypeCheck>("misc-fold-init-type");
    CheckFactories.registerCheck<ForwardDeclarationNamespaceCheck>(
        "misc-forward-declaration-namespace");
    CheckFactories.registerCheck<InaccurateEraseCheck>("misc-inaccurate-erase");
    CheckFactories.registerCheck<IncorrectRoundings>(
        "misc-incorrect-roundings");
    CheckFactories.registerCheck<InefficientAlgorithmCheck>(
        "misc-inefficient-algorithm");
    CheckFactories.registerCheck<MacroParenthesesCheck>(
        "misc-macro-parentheses");
    CheckFactories.registerCheck<MacroRepeatedSideEffectsCheck>(
        "misc-macro-repeated-side-effects");
    CheckFactories.registerCheck<MisplacedWideningCastCheck>(
        "misc-misplaced-widening-cast");
    CheckFactories.registerCheck<MoveConstantArgumentCheck>(
        "misc-move-const-arg");
    CheckFactories.registerCheck<MoveConstructorInitCheck>(
        "misc-move-constructor-init");
    CheckFactories.registerCheck<MoveForwardingReferenceCheck>(
        "misc-move-forwarding-reference");
    CheckFactories.registerCheck<MultipleStatementMacroCheck>(
        "misc-multiple-statement-macro");
    CheckFactories.registerCheck<NewDeleteOverloadsCheck>(
        "misc-new-delete-overloads");
    CheckFactories.registerCheck<NoexceptMoveConstructorCheck>(
        "misc-noexcept-move-constructor");
    CheckFactories.registerCheck<NonCopyableObjectsCheck>(
        "misc-non-copyable-objects");
    CheckFactories.registerCheck<RedundantExpressionCheck>(
        "misc-redundant-expression");
    CheckFactories.registerCheck<SizeofContainerCheck>("misc-sizeof-container");
    CheckFactories.registerCheck<SizeofExpressionCheck>(
        "misc-sizeof-expression");
    CheckFactories.registerCheck<StaticAssertCheck>("misc-static-assert");
    CheckFactories.registerCheck<StringCompareCheck>("misc-string-compare");
    CheckFactories.registerCheck<StringIntegerAssignmentCheck>(
        "misc-string-integer-assignment");
    CheckFactories.registerCheck<StringLiteralWithEmbeddedNulCheck>(
        "misc-string-literal-with-embedded-nul");
    CheckFactories.registerCheck<SuspiciousEnumUsageCheck>(
        "misc-suspicious-enum-usage");
    CheckFactories.registerCheck<SuspiciousMissingCommaCheck>(
        "misc-suspicious-missing-comma");
    CheckFactories.registerCheck<SuspiciousSemicolonCheck>(
        "misc-suspicious-semicolon");
    CheckFactories.registerCheck<SuspiciousStringCompareCheck>(
        "misc-suspicious-string-compare");
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
    CheckFactories.registerCheck<UnusedUsingDeclsCheck>(
        "misc-unused-using-decls");
    CheckFactories.registerCheck<UseAfterMoveCheck>("misc-use-after-move");
    CheckFactories.registerCheck<VirtualNearMissCheck>(
        "misc-virtual-near-miss");
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
