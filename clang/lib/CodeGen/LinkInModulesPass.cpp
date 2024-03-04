//===-- LinkInModulesPass.cpp - Module Linking pass --------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// LinkInModulesPass implementation.
///
//===----------------------------------------------------------------------===//

#include "LinkInModulesPass.h"
#include "BackendConsumer.h"

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

using namespace llvm;

LinkInModulesPass::LinkInModulesPass(clang::BackendConsumer *BC,
                                     bool ShouldLinkFiles)
    : BC(BC), ShouldLinkFiles(ShouldLinkFiles) {}

PreservedAnalyses LinkInModulesPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!BC)
    return PreservedAnalyses::all();

  // Re-load bitcode modules from files
  if (BC->ReloadModules(&M))
    report_fatal_error("Bitcode module re-loading failed, aborted!");

  if (BC->LinkInModules(&M, ShouldLinkFiles))
    report_fatal_error("Bitcode module re-linking failed, aborted!");

  return PreservedAnalyses::all();
}
