//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "MtUnsafeCheck.h"
#include "ThreadCanceltypeAsynchronousCheck.h"

namespace clang::tidy {
namespace concurrency {

class ConcurrencyModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<concurrency::MtUnsafeCheck>(
        "concurrency-mt-unsafe");
    CheckFactories.registerCheck<ThreadCanceltypeAsynchronousCheck>(
        "concurrency-thread-canceltype-asynchronous");
  }
};

} // namespace concurrency

// Register the ConcurrencyTidyModule using this statically initialized
// variable.
static ClangTidyModuleRegistry::Add<concurrency::ConcurrencyModule>
    X("concurrency-module", "Adds concurrency checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the ConcurrencyModule.
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int ConcurrencyModuleAnchorSource = 0;

} // namespace clang::tidy
