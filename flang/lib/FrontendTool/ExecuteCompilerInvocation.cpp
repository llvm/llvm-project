//===--- ExecuteCompilerInvocation.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds ExecuteCompilerInvocation(). It is split into its own file to
// minimize the impact of pulling in essentially everything else in Flang.
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CommandLine.h"

namespace Fortran::frontend {
bool ExecuteCompilerInvocation(CompilerInstance *flang) {
  // Honor -help.
  if (flang->GetFrontendOpts().showHelp_) {
    clang::driver::getDriverOptTable().PrintHelp(llvm::outs(),
        "flang-new -fc1 [options] file...", "LLVM 'Flang' Compiler",
        /*Include=*/clang::driver::options::FC1Option,
        /*Exclude=*/0, /*ShowAllAliases=*/false);
    return true;
  }

  // Honor -version.
  if (flang->GetFrontendOpts().showVersion_) {
    llvm::cl::PrintVersionMessage();
    return true;
  }

  return true;
}

} // namespace Fortran::frontend
