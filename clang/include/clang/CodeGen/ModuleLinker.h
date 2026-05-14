//===--- ModuleLinker.h - Shared bitcode link helpers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_MODULELINKER_H
#define LLVM_CLANG_CODEGEN_MODULELINKER_H

#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace clang {
class CompilerInstance;

/// Info about a module to link into the module currently being generated.
/// Shared between the classic clang CodeGen path and the ClangIR path.
struct LinkModule {
  std::unique_ptr<llvm::Module> Module;
  bool PropagateAttrs;
  bool Internalize;
  unsigned LinkFlags;
};

/// Load every bitcode file listed in CodeGenOpts.LinkBitcodeFiles into
/// \p LinkModules. Returns true on error (diagnostic already reported).
/// Appends to \p LinkModules; does not clear it.
bool loadLinkModules(CompilerInstance &CI, llvm::LLVMContext &Ctx,
                     llvm::SmallVectorImpl<LinkModule> &LinkModules);

} // namespace clang

#endif
