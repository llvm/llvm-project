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
class Function;
class LLVMContext;
class Module;
} // namespace llvm

namespace clang {
class CodeGenOptions;
class CompilerInstance;
class LangOptions;
class TargetOptions;

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

namespace CodeGen {
/// Adds attributes to \p F according to our \p CodeGenOpts and \p LangOpts, as
/// though we had emitted it ourselves. We remove any attributes on F that
/// conflict with the attributes we add here.
///
/// This is useful for adding attrs to bitcode modules that you want to link
/// with but don't control, such as CUDA's libdevice.  When linking with such
/// a bitcode library, you might want to set e.g. its functions'
/// denormal_fp_math attribute to match the attr of the functions you're
/// codegen'ing.  Otherwise, LLVM will interpret the bitcode module's lack of
/// denormal-fp-math attrs as tantamount to denormal-fp-math=ieee, and then LLVM
/// will propagate denormal-fp-math=ieee up to every transitive caller of a
/// function in the bitcode library!
///
/// With the exception of fast-math attrs, this will only make the attributes
/// on the function more conservative.  But it's unsafe to call this on a
/// function which relies on particular fast-math attributes for correctness.
/// It's up to you to ensure that this is safe.
void mergeDefaultFunctionDefinitionAttributes(llvm::Function &F,
                                              const CodeGenOptions &CodeGenOpts,
                                              const LangOptions &LangOpts,
                                              const TargetOptions &TargetOpts,
                                              bool WillInternalize);
} // namespace CodeGen

} // namespace clang

#endif
