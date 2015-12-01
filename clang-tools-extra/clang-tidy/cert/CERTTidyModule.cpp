//===--- CERTTidyModule.cpp - clang-tidy ----------------------------------===//
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
#include "../google/UnnamedNamespaceInHeaderCheck.h"
#include "../misc/MoveConstructorInitCheck.h"
#include "../misc/NewDeleteOverloadsCheck.h"
#include "../misc/NonCopyableObjects.h"
#include "../misc/StaticAssertCheck.h"
#include "../misc/ThrowByValueCatchByReferenceCheck.h"
#include "SetLongJmpCheck.h"
#include "StaticObjectExceptionCheck.h"
#include "ThrownExceptionTypeCheck.h"
#include "VariadicFunctionDefCheck.h"

namespace clang {
namespace tidy {
namespace CERT {

class CERTModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    // C++ checkers
    // DCL
    CheckFactories.registerCheck<VariadicFunctionDefCheck>(
        "cert-dcl50-cpp");
    CheckFactories.registerCheck<misc::NewDeleteOverloadsCheck>(
        "cert-dcl54-cpp");
    CheckFactories.registerCheck<google::build::UnnamedNamespaceInHeaderCheck>(
        "cert-dcl59-cpp");
    // OOP
    CheckFactories.registerCheck<MoveConstructorInitCheck>(
        "cert-oop11-cpp");
    // ERR
    CheckFactories.registerCheck<SetLongJmpCheck>(
        "cert-err52-cpp");
    CheckFactories.registerCheck<StaticObjectExceptionCheck>(
        "cert-err58-cpp");
    CheckFactories.registerCheck<ThrownExceptionTypeCheck>(
        "cert-err60-cpp");
    CheckFactories.registerCheck<ThrowByValueCatchByReferenceCheck>(
        "cert-err61-cpp");

    // C checkers
    // DCL
    CheckFactories.registerCheck<StaticAssertCheck>(
        "cert-dcl03-c");

    // FIO
    CheckFactories.registerCheck<NonCopyableObjectsCheck>(
        "cert-fio38-c");
  }
};

} // namespace misc

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<CERT::CERTModule>
X("cert-module",
  "Adds lint checks corresponding to CERT secure coding guidelines.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the CERTModule.
volatile int CERTModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
