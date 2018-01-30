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
#include "DefinitionsInHeadersCheck.h"
#include "ForwardingReferenceOverloadCheck.h"
#include "LambdaFunctionNameCheck.h"
#include "MacroParenthesesCheck.h"
#include "MacroRepeatedSideEffectsCheck.h"
#include "MisplacedConstCheck.h"
#include "MisplacedWideningCastCheck.h"
#include "NewDeleteOverloadsCheck.h"
#include "NonCopyableObjects.h"
#include "RedundantExpressionCheck.h"
#include "SizeofContainerCheck.h"
#include "SizeofExpressionCheck.h"
#include "StaticAssertCheck.h"
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

namespace clang {
namespace tidy {
namespace misc {

class MiscModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ForwardingReferenceOverloadCheck>(
        "misc-forwarding-reference-overload");
    CheckFactories.registerCheck<LambdaFunctionNameCheck>(
        "misc-lambda-function-name");
    CheckFactories.registerCheck<MisplacedConstCheck>("misc-misplaced-const");
    CheckFactories.registerCheck<UnconventionalAssignOperatorCheck>(
        "misc-unconventional-assign-operator");
    CheckFactories.registerCheck<DefinitionsInHeadersCheck>(
        "misc-definitions-in-headers");
    CheckFactories.registerCheck<MacroParenthesesCheck>(
        "misc-macro-parentheses");
    CheckFactories.registerCheck<MacroRepeatedSideEffectsCheck>(
        "misc-macro-repeated-side-effects");
    CheckFactories.registerCheck<MisplacedWideningCastCheck>(
        "misc-misplaced-widening-cast");
    CheckFactories.registerCheck<NewDeleteOverloadsCheck>(
        "misc-new-delete-overloads");
    CheckFactories.registerCheck<NonCopyableObjectsCheck>(
        "misc-non-copyable-objects");
    CheckFactories.registerCheck<RedundantExpressionCheck>(
        "misc-redundant-expression");
    CheckFactories.registerCheck<SizeofContainerCheck>("misc-sizeof-container");
    CheckFactories.registerCheck<SizeofExpressionCheck>(
        "misc-sizeof-expression");
    CheckFactories.registerCheck<StaticAssertCheck>("misc-static-assert");
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
