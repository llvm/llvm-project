//===--- Parasol.cpp - Parasol ToolChain Implementations ------------*- C++ -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#include "Parasol.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

ParasolToolChain::ParasolToolChain(const Driver &D, const llvm::Triple &Triple,
                               const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  // ProgramPaths are found via 'PATH' environment variable.
}

bool ParasolToolChain::isPICDefault() const { return true; }

bool ParasolToolChain::isPIEDefault(const ArgList &Args) const { return false; }

bool ParasolToolChain::isPICDefaultForced() const { return true; }

bool ParasolToolChain::SupportsProfiling() const { return false; }

bool ParasolToolChain::hasBlocksRuntime() const { return false; }
