//===--- ReadabilityTidyModule.cpp - clang-tidy ---------------------------===//
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
#include "AvoidConstParamsInDecls.h"
#include "BracesAroundStatementsCheck.h"
#include "ContainerSizeEmptyCheck.h"
#include "DeleteNullPointerCheck.h"
#include "DeletedDefaultCheck.h"
#include "ElseAfterReturnCheck.h"
#include "FunctionSizeCheck.h"
#include "IdentifierNamingCheck.h"
#include "ImplicitBoolCastCheck.h"
#include "InconsistentDeclarationParameterNameCheck.h"
#include "MisleadingIndentationCheck.h"
#include "MisplacedArrayIndexCheck.h"
#include "NamedParameterCheck.h"
#include "NonConstParameterCheck.h"
#include "RedundantControlFlowCheck.h"
#include "RedundantDeclarationCheck.h"
#include "RedundantFunctionPtrDereferenceCheck.h"
#include "RedundantMemberInitCheck.h"
#include "RedundantSmartptrGetCheck.h"
#include "RedundantStringCStrCheck.h"
#include "RedundantStringInitCheck.h"
#include "SimplifyBooleanExprCheck.h"
#include "StaticDefinitionInAnonymousNamespaceCheck.h"
#include "UniqueptrDeleteReleaseCheck.h"

namespace clang {
namespace tidy {
namespace readability {

class ReadabilityModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<AvoidConstParamsInDecls>(
        "readability-avoid-const-params-in-decls");
    CheckFactories.registerCheck<BracesAroundStatementsCheck>(
        "readability-braces-around-statements");
    CheckFactories.registerCheck<ContainerSizeEmptyCheck>(
        "readability-container-size-empty");
    CheckFactories.registerCheck<DeleteNullPointerCheck>(
        "readability-delete-null-pointer");
    CheckFactories.registerCheck<DeletedDefaultCheck>(
        "readability-deleted-default");
    CheckFactories.registerCheck<ElseAfterReturnCheck>(
        "readability-else-after-return");
    CheckFactories.registerCheck<FunctionSizeCheck>(
        "readability-function-size");
    CheckFactories.registerCheck<IdentifierNamingCheck>(
        "readability-identifier-naming");
    CheckFactories.registerCheck<ImplicitBoolCastCheck>(
        "readability-implicit-bool-cast");
    CheckFactories.registerCheck<InconsistentDeclarationParameterNameCheck>(
        "readability-inconsistent-declaration-parameter-name");
    CheckFactories.registerCheck<MisleadingIndentationCheck>(
        "readability-misleading-indentation");
    CheckFactories.registerCheck<MisplacedArrayIndexCheck>(
        "readability-misplaced-array-index");
    CheckFactories.registerCheck<RedundantFunctionPtrDereferenceCheck>(
        "readability-redundant-function-ptr-dereference");
    CheckFactories.registerCheck<RedundantMemberInitCheck>(
        "readability-redundant-member-init");
    CheckFactories.registerCheck<StaticDefinitionInAnonymousNamespaceCheck>(
        "readability-static-definition-in-anonymous-namespace");
    CheckFactories.registerCheck<readability::NamedParameterCheck>(
        "readability-named-parameter");
    CheckFactories.registerCheck<NonConstParameterCheck>(
        "readability-non-const-parameter");
    CheckFactories.registerCheck<RedundantControlFlowCheck>(
        "readability-redundant-control-flow");
    CheckFactories.registerCheck<RedundantDeclarationCheck>(
        "readability-redundant-declaration");
    CheckFactories.registerCheck<RedundantSmartptrGetCheck>(
        "readability-redundant-smartptr-get");
    CheckFactories.registerCheck<RedundantStringCStrCheck>(
        "readability-redundant-string-cstr");
    CheckFactories.registerCheck<RedundantStringInitCheck>(
        "readability-redundant-string-init");
    CheckFactories.registerCheck<SimplifyBooleanExprCheck>(
        "readability-simplify-boolean-expr");
    CheckFactories.registerCheck<UniqueptrDeleteReleaseCheck>(
        "readability-uniqueptr-delete-release");
  }
};

// Register the ReadabilityModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<ReadabilityModule>
    X("readability-module", "Adds readability-related checks.");

} // namespace readability

// This anchor is used to force the linker to link in the generated object file
// and thus register the ReadabilityModule.
volatile int ReadabilityModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
