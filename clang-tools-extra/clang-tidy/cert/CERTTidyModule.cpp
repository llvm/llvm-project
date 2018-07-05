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
#include "../misc/NewDeleteOverloadsCheck.h"
#include "../misc/NonCopyableObjects.h"
#include "../misc/StaticAssertCheck.h"
#include "../misc/ThrowByValueCatchByReferenceCheck.h"
#include "../performance/MoveConstructorInitCheck.h"
#include "CommandProcessorCheck.h"
#include "DontModifyStdNamespaceCheck.h"
#include "FloatLoopCounter.h"
#include "LimitedRandomnessCheck.h"
#include "PostfixOperatorCheck.h"
#include "ProperlySeededRandomGeneratorCheck.h"
#include "SetLongJmpCheck.h"
#include "StaticObjectExceptionCheck.h"
#include "StrToNumCheck.h"
#include "ThrownExceptionTypeCheck.h"
#include "VariadicFunctionDefCheck.h"

namespace clang {
namespace tidy {
namespace cert {

class CERTModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    // C++ checkers
    // DCL
    CheckFactories.registerCheck<PostfixOperatorCheck>(
        "cert-dcl21-cpp");
    CheckFactories.registerCheck<VariadicFunctionDefCheck>("cert-dcl50-cpp");
    CheckFactories.registerCheck<misc::NewDeleteOverloadsCheck>(
        "cert-dcl54-cpp");
    CheckFactories.registerCheck<DontModifyStdNamespaceCheck>(
        "cert-dcl58-cpp");
    CheckFactories.registerCheck<google::build::UnnamedNamespaceInHeaderCheck>(
        "cert-dcl59-cpp");
    // OOP
    CheckFactories.registerCheck<performance::MoveConstructorInitCheck>(
        "cert-oop11-cpp");
    // ERR
    CheckFactories.registerCheck<misc::ThrowByValueCatchByReferenceCheck>(
        "cert-err09-cpp");
    CheckFactories.registerCheck<SetLongJmpCheck>("cert-err52-cpp");
    CheckFactories.registerCheck<StaticObjectExceptionCheck>("cert-err58-cpp");
    CheckFactories.registerCheck<ThrownExceptionTypeCheck>("cert-err60-cpp");
    CheckFactories.registerCheck<misc::ThrowByValueCatchByReferenceCheck>(
        "cert-err61-cpp");
    // MSC
    CheckFactories.registerCheck<LimitedRandomnessCheck>("cert-msc50-cpp");
    CheckFactories.registerCheck<ProperlySeededRandomGeneratorCheck>(
        "cert-msc51-cpp");

    // C checkers
    // DCL
    CheckFactories.registerCheck<misc::StaticAssertCheck>("cert-dcl03-c");
    // ENV
    CheckFactories.registerCheck<CommandProcessorCheck>("cert-env33-c");
    // FLP
    CheckFactories.registerCheck<FloatLoopCounter>("cert-flp30-c");
    // FIO
    CheckFactories.registerCheck<misc::NonCopyableObjectsCheck>("cert-fio38-c");
    // ERR
    CheckFactories.registerCheck<StrToNumCheck>("cert-err34-c");
    // MSC
    CheckFactories.registerCheck<LimitedRandomnessCheck>("cert-msc30-c");
    CheckFactories.registerCheck<ProperlySeededRandomGeneratorCheck>(
        "cert-msc32-c");
  }
};

} // namespace cert

// Register the MiscTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<cert::CERTModule>
    X("cert-module",
      "Adds lint checks corresponding to CERT secure coding guidelines.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the CERTModule.
volatile int CERTModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
