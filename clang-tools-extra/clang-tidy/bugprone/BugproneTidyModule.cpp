//===--- BugproneTidyModule.cpp - clang-tidy ------------------------------===//
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
#include "../cppcoreguidelines/NarrowingConversionsCheck.h"
#include "ArgumentCommentCheck.h"
#include "AssertSideEffectCheck.h"
#include "BoolPointerImplicitConversionCheck.h"
#include "CopyConstructorInitCheck.h"
#include "DanglingHandleCheck.h"
#include "ExceptionEscapeCheck.h"
#include "FoldInitTypeCheck.h"
#include "ForwardDeclarationNamespaceCheck.h"
#include "ForwardingReferenceOverloadCheck.h"
#include "InaccurateEraseCheck.h"
#include "IncorrectRoundingsCheck.h"
#include "IntegerDivisionCheck.h"
#include "LambdaFunctionNameCheck.h"
#include "MacroParenthesesCheck.h"
#include "MacroRepeatedSideEffectsCheck.h"
#include "MisplacedOperatorInStrlenInAllocCheck.h"
#include "MisplacedWideningCastCheck.h"
#include "MoveForwardingReferenceCheck.h"
#include "MultipleStatementMacroCheck.h"
#include "ParentVirtualCallCheck.h"
#include "SizeofContainerCheck.h"
#include "SizeofExpressionCheck.h"
#include "StringConstructorCheck.h"
#include "StringIntegerAssignmentCheck.h"
#include "StringLiteralWithEmbeddedNulCheck.h"
#include "SuspiciousEnumUsageCheck.h"
#include "SuspiciousMemsetUsageCheck.h"
#include "SuspiciousMissingCommaCheck.h"
#include "SuspiciousSemicolonCheck.h"
#include "SuspiciousStringCompareCheck.h"
#include "SwappedArgumentsCheck.h"
#include "TerminatingContinueCheck.h"
#include "ThrowKeywordMissingCheck.h"
#include "UndefinedMemoryManipulationCheck.h"
#include "UndelegatedConstructorCheck.h"
#include "UnusedRaiiCheck.h"
#include "UnusedReturnValueCheck.h"
#include "UseAfterMoveCheck.h"
#include "VirtualNearMissCheck.h"

namespace clang {
namespace tidy {
namespace bugprone {

class BugproneModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ArgumentCommentCheck>(
        "bugprone-argument-comment");
    CheckFactories.registerCheck<AssertSideEffectCheck>(
        "bugprone-assert-side-effect");
    CheckFactories.registerCheck<BoolPointerImplicitConversionCheck>(
        "bugprone-bool-pointer-implicit-conversion");
    CheckFactories.registerCheck<CopyConstructorInitCheck>(
        "bugprone-copy-constructor-init");
    CheckFactories.registerCheck<DanglingHandleCheck>(
        "bugprone-dangling-handle");
    CheckFactories.registerCheck<ExceptionEscapeCheck>(
        "bugprone-exception-escape");
    CheckFactories.registerCheck<FoldInitTypeCheck>(
        "bugprone-fold-init-type");
    CheckFactories.registerCheck<ForwardDeclarationNamespaceCheck>(
        "bugprone-forward-declaration-namespace");
    CheckFactories.registerCheck<ForwardingReferenceOverloadCheck>(
        "bugprone-forwarding-reference-overload");
    CheckFactories.registerCheck<InaccurateEraseCheck>(
        "bugprone-inaccurate-erase");
    CheckFactories.registerCheck<IncorrectRoundingsCheck>(
        "bugprone-incorrect-roundings");
    CheckFactories.registerCheck<IntegerDivisionCheck>(
        "bugprone-integer-division");
    CheckFactories.registerCheck<LambdaFunctionNameCheck>(
        "bugprone-lambda-function-name");
    CheckFactories.registerCheck<MacroParenthesesCheck>(
        "bugprone-macro-parentheses");
    CheckFactories.registerCheck<MacroRepeatedSideEffectsCheck>(
        "bugprone-macro-repeated-side-effects");
    CheckFactories.registerCheck<MisplacedOperatorInStrlenInAllocCheck>(
        "bugprone-misplaced-operator-in-strlen-in-alloc");
    CheckFactories.registerCheck<MisplacedWideningCastCheck>(
        "bugprone-misplaced-widening-cast");
    CheckFactories.registerCheck<MoveForwardingReferenceCheck>(
        "bugprone-move-forwarding-reference");
    CheckFactories.registerCheck<MultipleStatementMacroCheck>(
        "bugprone-multiple-statement-macro");
    CheckFactories.registerCheck<cppcoreguidelines::NarrowingConversionsCheck>(
        "bugprone-narrowing-conversions");
    CheckFactories.registerCheck<ParentVirtualCallCheck>(
        "bugprone-parent-virtual-call");
    CheckFactories.registerCheck<SizeofContainerCheck>(
        "bugprone-sizeof-container");
    CheckFactories.registerCheck<SizeofExpressionCheck>(
        "bugprone-sizeof-expression");
    CheckFactories.registerCheck<StringConstructorCheck>(
        "bugprone-string-constructor");
    CheckFactories.registerCheck<StringIntegerAssignmentCheck>(
        "bugprone-string-integer-assignment");
    CheckFactories.registerCheck<StringLiteralWithEmbeddedNulCheck>(
        "bugprone-string-literal-with-embedded-nul");
    CheckFactories.registerCheck<SuspiciousEnumUsageCheck>(
        "bugprone-suspicious-enum-usage");
    CheckFactories.registerCheck<SuspiciousMemsetUsageCheck>(
        "bugprone-suspicious-memset-usage");
    CheckFactories.registerCheck<SuspiciousMissingCommaCheck>(
        "bugprone-suspicious-missing-comma");
    CheckFactories.registerCheck<SuspiciousSemicolonCheck>(
        "bugprone-suspicious-semicolon");
    CheckFactories.registerCheck<SuspiciousStringCompareCheck>(
        "bugprone-suspicious-string-compare");
    CheckFactories.registerCheck<SwappedArgumentsCheck>(
        "bugprone-swapped-arguments");
    CheckFactories.registerCheck<TerminatingContinueCheck>(
        "bugprone-terminating-continue");
    CheckFactories.registerCheck<ThrowKeywordMissingCheck>(
        "bugprone-throw-keyword-missing");
    CheckFactories.registerCheck<UndefinedMemoryManipulationCheck>(
        "bugprone-undefined-memory-manipulation");
    CheckFactories.registerCheck<UndelegatedConstructorCheck>(
        "bugprone-undelegated-constructor");
    CheckFactories.registerCheck<UnusedRaiiCheck>(
        "bugprone-unused-raii");
    CheckFactories.registerCheck<UnusedReturnValueCheck>(
        "bugprone-unused-return-value");
    CheckFactories.registerCheck<UseAfterMoveCheck>(
        "bugprone-use-after-move");
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
volatile int BugproneModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
