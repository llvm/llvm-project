//===------- BoostTidyModule.cpp - clang-tidy -----------------------------===//
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
#include "UseToStringCheck.h"
using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace boost {

class BoostModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<UseToStringCheck>("boost-use-to-string");
  }
};

// Register the BoostModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<BoostModule> X("boost-module",
                                                   "Add boost checks.");

} // namespace boost

// This anchor is used to force the linker to link in the generated object file
// and thus register the BoostModule.
volatile int BoostModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
