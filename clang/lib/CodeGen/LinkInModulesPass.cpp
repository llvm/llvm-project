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

using namespace llvm;

LinkInModulesPass::LinkInModulesPass(clang::BackendConsumer *BC,
                                     bool ShouldLinkFiles)
    : BC(BC), ShouldLinkFiles(ShouldLinkFiles) {}

PreservedAnalyses LinkInModulesPass::run(Module &M, ModuleAnalysisManager &AM) {

  if (BC && BC->LinkInModules(&M, ShouldLinkFiles))
    report_fatal_error("Bitcode module linking failed, compilation aborted!");

  return PreservedAnalyses::all();
}
