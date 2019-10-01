//===--- MPITidyModule.cpp - clang-tidy -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "BufferDerefCheck.h"
#include "TypeMismatchCheck.h"

namespace clang {
namespace tidy {
namespace mpi {

class MPIModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<BufferDerefCheck>("mpi-buffer-deref");
    CheckFactories.registerCheck<TypeMismatchCheck>("mpi-type-mismatch");
  }
};

} // namespace mpi

// Register the MPITidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<mpi::MPIModule>
    X("mpi-module", "Adds MPI clang-tidy checks.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the MPIModule.
volatile int MPIModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
