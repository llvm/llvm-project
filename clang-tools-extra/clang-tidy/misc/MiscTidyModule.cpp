//===--- MiscTidyModule.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "ConfusableIdentifierCheck.h"
#include "ConstCorrectnessCheck.h"
#include "CoroutineHostileRAIICheck.h"
#include "DefinitionsInHeadersCheck.h"
#include "HeaderIncludeCycleCheck.h"
#include "IncludeCleanerCheck.h"
#include "MisleadingBidirectional.h"
#include "MisleadingIdentifier.h"
#include "MisplacedConstCheck.h"
#include "NewDeleteOverloadsCheck.h"
#include "NoRecursionCheck.h"
#include "NonCopyableObjects.h"
#include "NonPrivateMemberVariablesInClassesCheck.h"
#include "RedundantExpressionCheck.h"
#include "StaticAssertCheck.h"
#include "ThrowByValueCatchByReferenceCheck.h"
#include "UnconventionalAssignOperatorCheck.h"
#include "UniqueptrResetReleaseCheck.h"
#include "UnusedAliasDeclsCheck.h"
#include "UnusedParametersCheck.h"
#include "UnusedUsingDeclsCheck.h"
#include "UseAnonymousNamespaceCheck.h"

namespace clang::tidy {
namespace misc {

class MiscModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ConfusableIdentifierCheck>(
        "misc-confusable-identifiers");
    CheckFactories.registerCheck<ConstCorrectnessCheck>(
        "misc-const-correctness");
    CheckFactories.registerCheck<CoroutineHostileRAIICheck>(
        "misc-coroutine-hostile-raii");
    CheckFactories.registerCheck<DefinitionsInHeadersCheck>(
        "misc-definitions-in-headers");
    CheckFactories.registerCheck<HeaderIncludeCycleCheck>(
        "misc-header-include-cycle");
    CheckFactories.registerCheck<IncludeCleanerCheck>("misc-include-cleaner");
    CheckFactories.registerCheck<MisleadingBidirectionalCheck>(
        "misc-misleading-bidirectional");
    CheckFactories.registerCheck<MisleadingIdentifierCheck>(
        "misc-misleading-identifier");
    CheckFactories.registerCheck<MisplacedConstCheck>("misc-misplaced-const");
    CheckFactories.registerCheck<NewDeleteOverloadsCheck>(
        "misc-new-delete-overloads");
    CheckFactories.registerCheck<NoRecursionCheck>("misc-no-recursion");
    CheckFactories.registerCheck<NonCopyableObjectsCheck>(
        "misc-non-copyable-objects");
    CheckFactories.registerCheck<NonPrivateMemberVariablesInClassesCheck>(
        "misc-non-private-member-variables-in-classes");
    CheckFactories.registerCheck<RedundantExpressionCheck>(
        "misc-redundant-expression");
    CheckFactories.registerCheck<StaticAssertCheck>("misc-static-assert");
    CheckFactories.registerCheck<ThrowByValueCatchByReferenceCheck>(
        "misc-throw-by-value-catch-by-reference");
    CheckFactories.registerCheck<UnconventionalAssignOperatorCheck>(
        "misc-unconventional-assign-operator");
    CheckFactories.registerCheck<UniqueptrResetReleaseCheck>(
        "misc-uniqueptr-reset-release");
    CheckFactories.registerCheck<UnusedAliasDeclsCheck>(
        "misc-unused-alias-decls");
    CheckFactories.registerCheck<UnusedParametersCheck>(
        "misc-unused-parameters");
    CheckFactories.registerCheck<UnusedUsingDeclsCheck>(
        "misc-unused-using-decls");
    CheckFactories.registerCheck<UseAnonymousNamespaceCheck>(
        "misc-use-anonymous-namespace");
  }
};

} // namespace misc

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<misc::MiscModule>
    X("misc-module", "Adds miscellaneous lint checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the MiscModule.
volatile int MiscModuleAnchorSource = 0;

} // namespace clang::tidy
