//===- bolt/Passes/CreateClonesAtOrigin.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CreateClonesAtOrigin pass that creates clones of
// functions at their original addresses. Unlike patching (which redirects
// execution from original to optimized code), cloning keeps the original code
// executable and updates its references to relocated functions.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/CreateClonesAtOrigin.h"
#include "bolt/Utils/CommandLineOpts.h"

namespace opts {
extern llvm::cl::opt<bool> NoScan;
extern llvm::cl::opt<bool> UseOldText;
extern llvm::cl::opt<unsigned> Verbosity;
} // namespace opts

namespace llvm {
namespace bolt {

Error CreateClonesAtOrigin::runOnFunctions(BinaryContext &BC) {
  if (!opts::CloneAtOrigin)
    return Error::success();

  if (opts::UseOldText) {
    BC.outs() << "BOLT-INFO: skipping clones at origin with --use-old-text\n";
    return Error::success();
  }

  if (opts::Verbosity >= 1)
    BC.outs() << "BOLT-INFO: creating clones at origin\n";

  unsigned NumClonesCreated = 0;

  // Iterate over input functions.
  for (BinaryFunction &Function :
       llvm::make_second_range(BC.getBinaryFunctions())) {

    // Only clone functions that will be emitted (relocated).
    if (!BC.shouldEmit(Function))
      continue;

    // Skip if already has a clone at origin.
    if (Function.hasCloneAtOrigin())
      continue;

    // Mark the function as having a clone at origin.
    Function.setHasCloneAtOrigin();

    // Run scanExternalRefs() to update the clone's references to relocated
    // functions.
    const bool Success = opts::NoScan || Function.scanExternalRefs();
    if (!Success) {
      BC.errs()
          << "BOLT-ERROR: internal error creating refs for emitted function\n";
      exit(1);
    }

    ++NumClonesCreated;

    if (opts::Verbosity >= 2)
      BC.outs() << "BOLT-INFO: created clone at origin for " << Function
                << '\n';
  }

  if (NumClonesCreated > 0)
    BC.outs() << "BOLT-INFO: created " << NumClonesCreated
              << " clones at origin\n";

  return Error::success();
}

} // namespace bolt
} // namespace llvm
