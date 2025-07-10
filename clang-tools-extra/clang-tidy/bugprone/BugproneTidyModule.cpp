//===--- BugproneTidyModule.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "ArgumentCommentCheck.h"
#include "AssertSideEffectCheck.h"
#include "AssignmentInIfConditionCheck.h"
#include "BadSignalToKillThreadCheck.h"
#include "BitwisePointerCastCheck.h"
#include "BoolPointerImplicitConversionCheck.h"
#include "BranchCloneCheck.h"
#include "CapturingThisInMemberVariableCheck.h"
#include "CastingThroughVoidCheck.h"
#include "ChainedComparisonCheck.h"
#include "ComparePointerToMemberVirtualFunctionCheck.h"
#include "CopyConstructorInitCheck.h"
#include "CrtpConstructorAccessibilityCheck.h"
#include "DanglingHandleCheck.h"
#include "DynamicStaticInitializersCheck.h"
#include "EasilySwappableParametersCheck.h"
#include "EmptyCatchCheck.h"
#include "ExceptionEscapeCheck.h"
#include "FoldInitTypeCheck.h"
#include "ForwardDeclarationNamespaceCheck.h"
#include "ForwardingReferenceOverloadCheck.h"
#include "ImplicitWideningOfMultiplicationResultCheck.h"
#include "InaccurateEraseCheck.h"
#include "IncDecInConditionsCheck.h"
#include "IncorrectEnableIfCheck.h"
#include "IncorrectEnableSharedFromThisCheck.h"
#include "IncorrectRoundingsCheck.h"
#include "InfiniteLoopCheck.h"
#include "IntegerDivisionCheck.h"
#include "LambdaFunctionNameCheck.h"
#include "MacroParenthesesCheck.h"
#include "MacroRepeatedSideEffectsCheck.h"
#include "MisleadingSetterOfReferenceCheck.h"
#include "MisplacedOperatorInStrlenInAllocCheck.h"
#include "MisplacedPointerArithmeticInAllocCheck.h"
#include "MisplacedWideningCastCheck.h"
#include "MoveForwardingReferenceCheck.h"
#include "MultiLevelImplicitPointerConversionCheck.h"
#include "MultipleNewInOneExpressionCheck.h"
#include "MultipleStatementMacroCheck.h"
#include "NarrowingConversionsCheck.h"
#include "NoEscapeCheck.h"
#include "NonZeroEnumToBoolConversionCheck.h"
#include "NondeterministicPointerIterationOrderCheck.h"
#include "NotNullTerminatedResultCheck.h"
#include "OptionalValueConversionCheck.h"
#include "ParentVirtualCallCheck.h"
#include "PointerArithmeticOnPolymorphicObjectCheck.h"
#include "PosixReturnCheck.h"
#include "RedundantBranchConditionCheck.h"
#include "ReservedIdentifierCheck.h"
#include "ReturnConstRefFromParameterCheck.h"
#include "SharedPtrArrayMismatchCheck.h"
#include "SignalHandlerCheck.h"
#include "SignedCharMisuseCheck.h"
#include "SizeofContainerCheck.h"
#include "SizeofExpressionCheck.h"
#include "SpuriouslyWakeUpFunctionsCheck.h"
#include "StandaloneEmptyCheck.h"
#include "StringConstructorCheck.h"
#include "StringIntegerAssignmentCheck.h"
#include "StringLiteralWithEmbeddedNulCheck.h"
#include "StringviewNullptrCheck.h"
#include "SuspiciousEnumUsageCheck.h"
#include "SuspiciousIncludeCheck.h"
#include "SuspiciousMemoryComparisonCheck.h"
#include "SuspiciousMemsetUsageCheck.h"
#include "SuspiciousMissingCommaCheck.h"
#include "SuspiciousReallocUsageCheck.h"
#include "SuspiciousSemicolonCheck.h"
#include "SuspiciousStringCompareCheck.h"
#include "SuspiciousStringviewDataUsageCheck.h"
#include "SwappedArgumentsCheck.h"
#include "SwitchMissingDefaultCaseCheck.h"
#include "TaggedUnionMemberCountCheck.h"
#include "TerminatingContinueCheck.h"
#include "ThrowKeywordMissingCheck.h"
#include "TooSmallLoopVariableCheck.h"
#include "UncheckedOptionalAccessCheck.h"
#include "UndefinedMemoryManipulationCheck.h"
#include "UndelegatedConstructorCheck.h"
#include "UnhandledExceptionAtNewCheck.h"
#include "UnhandledSelfAssignmentCheck.h"
#include "UnintendedCharOstreamOutputCheck.h"
#include "UniquePtrArrayMismatchCheck.h"
#include "UnsafeFunctionsCheck.h"
#include "UnusedLocalNonTrivialVariableCheck.h"
#include "UnusedRaiiCheck.h"
#include "UnusedReturnValueCheck.h"
#include "UseAfterMoveCheck.h"
#include "VirtualNearMissCheck.h"

namespace clang::tidy {
namespace bugprone {

class BugproneModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ArgumentCommentCheck>(
        "bugprone-argument-comment");
    CheckFactories.registerCheck<AssertSideEffectCheck>(
        "bugprone-assert-side-effect");
    CheckFactories.registerCheck<AssignmentInIfConditionCheck>(
        "bugprone-assignment-in-if-condition");
    CheckFactories.registerCheck<BadSignalToKillThreadCheck>(
        "bugprone-bad-signal-to-kill-thread");
    CheckFactories.registerCheck<BitwisePointerCastCheck>(
        "bugprone-bitwise-pointer-cast");
    CheckFactories.registerCheck<BoolPointerImplicitConversionCheck>(
        "bugprone-bool-pointer-implicit-conversion");
    CheckFactories.registerCheck<BranchCloneCheck>("bugprone-branch-clone");
    CheckFactories.registerCheck<CapturingThisInMemberVariableCheck>(
        "bugprone-capturing-this-in-member-variable");
    CheckFactories.registerCheck<CastingThroughVoidCheck>(
        "bugprone-casting-through-void");
    CheckFactories.registerCheck<ChainedComparisonCheck>(
        "bugprone-chained-comparison");
    CheckFactories.registerCheck<ComparePointerToMemberVirtualFunctionCheck>(
        "bugprone-compare-pointer-to-member-virtual-function");
    CheckFactories.registerCheck<CopyConstructorInitCheck>(
        "bugprone-copy-constructor-init");
    CheckFactories.registerCheck<DanglingHandleCheck>(
        "bugprone-dangling-handle");
    CheckFactories.registerCheck<DynamicStaticInitializersCheck>(
        "bugprone-dynamic-static-initializers");
    CheckFactories.registerCheck<EasilySwappableParametersCheck>(
        "bugprone-easily-swappable-parameters");
    CheckFactories.registerCheck<EmptyCatchCheck>("bugprone-empty-catch");
    CheckFactories.registerCheck<ExceptionEscapeCheck>(
        "bugprone-exception-escape");
    CheckFactories.registerCheck<FoldInitTypeCheck>("bugprone-fold-init-type");
    CheckFactories.registerCheck<ForwardDeclarationNamespaceCheck>(
        "bugprone-forward-declaration-namespace");
    CheckFactories.registerCheck<ForwardingReferenceOverloadCheck>(
        "bugprone-forwarding-reference-overload");
    CheckFactories.registerCheck<ImplicitWideningOfMultiplicationResultCheck>(
        "bugprone-implicit-widening-of-multiplication-result");
    CheckFactories.registerCheck<InaccurateEraseCheck>(
        "bugprone-inaccurate-erase");
    CheckFactories.registerCheck<IncorrectEnableIfCheck>(
        "bugprone-incorrect-enable-if");
    CheckFactories.registerCheck<IncorrectEnableSharedFromThisCheck>(
        "bugprone-incorrect-enable-shared-from-this");
    CheckFactories.registerCheck<UnintendedCharOstreamOutputCheck>(
        "bugprone-unintended-char-ostream-output");
    CheckFactories.registerCheck<ReturnConstRefFromParameterCheck>(
        "bugprone-return-const-ref-from-parameter");
    CheckFactories.registerCheck<SwitchMissingDefaultCaseCheck>(
        "bugprone-switch-missing-default-case");
    CheckFactories.registerCheck<IncDecInConditionsCheck>(
        "bugprone-inc-dec-in-conditions");
    CheckFactories.registerCheck<IncorrectRoundingsCheck>(
        "bugprone-incorrect-roundings");
    CheckFactories.registerCheck<InfiniteLoopCheck>("bugprone-infinite-loop");
    CheckFactories.registerCheck<IntegerDivisionCheck>(
        "bugprone-integer-division");
    CheckFactories.registerCheck<LambdaFunctionNameCheck>(
        "bugprone-lambda-function-name");
    CheckFactories.registerCheck<MacroParenthesesCheck>(
        "bugprone-macro-parentheses");
    CheckFactories.registerCheck<MacroRepeatedSideEffectsCheck>(
        "bugprone-macro-repeated-side-effects");
    CheckFactories.registerCheck<MisleadingSetterOfReferenceCheck>(
        "bugprone-misleading-setter-of-reference");
    CheckFactories.registerCheck<MisplacedOperatorInStrlenInAllocCheck>(
        "bugprone-misplaced-operator-in-strlen-in-alloc");
    CheckFactories.registerCheck<MisplacedPointerArithmeticInAllocCheck>(
        "bugprone-misplaced-pointer-arithmetic-in-alloc");
    CheckFactories.registerCheck<MisplacedWideningCastCheck>(
        "bugprone-misplaced-widening-cast");
    CheckFactories.registerCheck<MoveForwardingReferenceCheck>(
        "bugprone-move-forwarding-reference");
    CheckFactories.registerCheck<MultiLevelImplicitPointerConversionCheck>(
        "bugprone-multi-level-implicit-pointer-conversion");
    CheckFactories.registerCheck<MultipleNewInOneExpressionCheck>(
        "bugprone-multiple-new-in-one-expression");
    CheckFactories.registerCheck<MultipleStatementMacroCheck>(
        "bugprone-multiple-statement-macro");
    CheckFactories.registerCheck<NondeterministicPointerIterationOrderCheck>(
        "bugprone-nondeterministic-pointer-iteration-order");
    CheckFactories.registerCheck<OptionalValueConversionCheck>(
        "bugprone-optional-value-conversion");
    CheckFactories.registerCheck<PointerArithmeticOnPolymorphicObjectCheck>(
        "bugprone-pointer-arithmetic-on-polymorphic-object");
    CheckFactories.registerCheck<RedundantBranchConditionCheck>(
        "bugprone-redundant-branch-condition");
    CheckFactories.registerCheck<NarrowingConversionsCheck>(
        "bugprone-narrowing-conversions");
    CheckFactories.registerCheck<NoEscapeCheck>("bugprone-no-escape");
    CheckFactories.registerCheck<NonZeroEnumToBoolConversionCheck>(
        "bugprone-non-zero-enum-to-bool-conversion");
    CheckFactories.registerCheck<NotNullTerminatedResultCheck>(
        "bugprone-not-null-terminated-result");
    CheckFactories.registerCheck<ParentVirtualCallCheck>(
        "bugprone-parent-virtual-call");
    CheckFactories.registerCheck<PosixReturnCheck>("bugprone-posix-return");
    CheckFactories.registerCheck<ReservedIdentifierCheck>(
        "bugprone-reserved-identifier");
    CheckFactories.registerCheck<SharedPtrArrayMismatchCheck>(
        "bugprone-shared-ptr-array-mismatch");
    CheckFactories.registerCheck<SignalHandlerCheck>("bugprone-signal-handler");
    CheckFactories.registerCheck<SignedCharMisuseCheck>(
        "bugprone-signed-char-misuse");
    CheckFactories.registerCheck<SizeofContainerCheck>(
        "bugprone-sizeof-container");
    CheckFactories.registerCheck<SizeofExpressionCheck>(
        "bugprone-sizeof-expression");
    CheckFactories.registerCheck<SpuriouslyWakeUpFunctionsCheck>(
        "bugprone-spuriously-wake-up-functions");
    CheckFactories.registerCheck<StandaloneEmptyCheck>(
        "bugprone-standalone-empty");
    CheckFactories.registerCheck<StringConstructorCheck>(
        "bugprone-string-constructor");
    CheckFactories.registerCheck<StringIntegerAssignmentCheck>(
        "bugprone-string-integer-assignment");
    CheckFactories.registerCheck<StringLiteralWithEmbeddedNulCheck>(
        "bugprone-string-literal-with-embedded-nul");
    CheckFactories.registerCheck<StringviewNullptrCheck>(
        "bugprone-stringview-nullptr");
    CheckFactories.registerCheck<SuspiciousEnumUsageCheck>(
        "bugprone-suspicious-enum-usage");
    CheckFactories.registerCheck<SuspiciousIncludeCheck>(
        "bugprone-suspicious-include");
    CheckFactories.registerCheck<SuspiciousMemoryComparisonCheck>(
        "bugprone-suspicious-memory-comparison");
    CheckFactories.registerCheck<SuspiciousMemsetUsageCheck>(
        "bugprone-suspicious-memset-usage");
    CheckFactories.registerCheck<SuspiciousMissingCommaCheck>(
        "bugprone-suspicious-missing-comma");
    CheckFactories.registerCheck<SuspiciousReallocUsageCheck>(
        "bugprone-suspicious-realloc-usage");
    CheckFactories.registerCheck<SuspiciousSemicolonCheck>(
        "bugprone-suspicious-semicolon");
    CheckFactories.registerCheck<SuspiciousStringCompareCheck>(
        "bugprone-suspicious-string-compare");
    CheckFactories.registerCheck<SuspiciousStringviewDataUsageCheck>(
        "bugprone-suspicious-stringview-data-usage");
    CheckFactories.registerCheck<SwappedArgumentsCheck>(
        "bugprone-swapped-arguments");
    CheckFactories.registerCheck<TaggedUnionMemberCountCheck>(
        "bugprone-tagged-union-member-count");
    CheckFactories.registerCheck<TerminatingContinueCheck>(
        "bugprone-terminating-continue");
    CheckFactories.registerCheck<ThrowKeywordMissingCheck>(
        "bugprone-throw-keyword-missing");
    CheckFactories.registerCheck<TooSmallLoopVariableCheck>(
        "bugprone-too-small-loop-variable");
    CheckFactories.registerCheck<UncheckedOptionalAccessCheck>(
        "bugprone-unchecked-optional-access");
    CheckFactories.registerCheck<UndefinedMemoryManipulationCheck>(
        "bugprone-undefined-memory-manipulation");
    CheckFactories.registerCheck<UndelegatedConstructorCheck>(
        "bugprone-undelegated-constructor");
    CheckFactories.registerCheck<UnhandledSelfAssignmentCheck>(
        "bugprone-unhandled-self-assignment");
    CheckFactories.registerCheck<UnhandledExceptionAtNewCheck>(
        "bugprone-unhandled-exception-at-new");
    CheckFactories.registerCheck<UniquePtrArrayMismatchCheck>(
        "bugprone-unique-ptr-array-mismatch");
    CheckFactories.registerCheck<CrtpConstructorAccessibilityCheck>(
        "bugprone-crtp-constructor-accessibility");
    CheckFactories.registerCheck<UnsafeFunctionsCheck>(
        "bugprone-unsafe-functions");
    CheckFactories.registerCheck<UnusedLocalNonTrivialVariableCheck>(
        "bugprone-unused-local-non-trivial-variable");
    CheckFactories.registerCheck<UnusedRaiiCheck>("bugprone-unused-raii");
    CheckFactories.registerCheck<UnusedReturnValueCheck>(
        "bugprone-unused-return-value");
    CheckFactories.registerCheck<UseAfterMoveCheck>("bugprone-use-after-move");
    CheckFactories.registerCheck<VirtualNearMissCheck>(
        "bugprone-virtual-near-miss");
  }
};

} // namespace bugprone

// Register the BugproneTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<bugprone::BugproneModule>
    X("bugprone-module", "Adds checks for bugprone code constructs.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the BugproneModule.
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int BugproneModuleAnchorSource = 0;

} // namespace clang::tidy
