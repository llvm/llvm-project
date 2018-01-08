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
#include "ArgumentCommentCheck.h"
#include "AssertSideEffectCheck.h"
#include "BoolPointerImplicitConversionCheck.h"
#include "CopyConstructorInitCheck.h"
#include "DanglingHandleCheck.h"
#include "FoldInitTypeCheck.h"
#include "ForwardDeclarationNamespaceCheck.h"
#include "InaccurateEraseCheck.h"
#include "IntegerDivisionCheck.h"
#include "MisplacedOperatorInStrlenInAllocCheck.h"
#include "MoveForwardingReferenceCheck.h"
#include "MultipleStatementMacroCheck.h"
#include "StringConstructorCheck.h"
#include "SuspiciousMemsetUsageCheck.h"
#include "UndefinedMemoryManipulationCheck.h"
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
    CheckFactories.registerCheck<FoldInitTypeCheck>(
        "bugprone-fold-init-type");
    CheckFactories.registerCheck<ForwardDeclarationNamespaceCheck>(
        "bugprone-forward-declaration-namespace");
    CheckFactories.registerCheck<InaccurateEraseCheck>(
        "bugprone-inaccurate-erase");
    CheckFactories.registerCheck<IntegerDivisionCheck>(
        "bugprone-integer-division");
    CheckFactories.registerCheck<MisplacedOperatorInStrlenInAllocCheck>(
        "bugprone-misplaced-operator-in-strlen-in-alloc");
    CheckFactories.registerCheck<MoveForwardingReferenceCheck>(
        "bugprone-move-forwarding-reference");
    CheckFactories.registerCheck<MultipleStatementMacroCheck>(
        "bugprone-multiple-statement-macro");
    CheckFactories.registerCheck<StringConstructorCheck>(
        "bugprone-string-constructor");
    CheckFactories.registerCheck<SuspiciousMemsetUsageCheck>(
        "bugprone-suspicious-memset-usage");
    CheckFactories.registerCheck<UndefinedMemoryManipulationCheck>(
        "bugprone-undefined-memory-manipulation");
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
