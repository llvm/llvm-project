//===- bolt/Rewrite/MetadataManager.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/MetadataManager.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-metadata"

using namespace llvm;
using namespace bolt;

void MetadataManager::registerRewriter(
    std::unique_ptr<MetadataRewriter> Rewriter) {
  Rewriters.emplace_back(std::move(Rewriter));
}

void MetadataManager::runInitializersPreCFG() {
  for (auto &Rewriter : Rewriters) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: invoking " << Rewriter->getName()
                      << " before CFG construction\n");
    if (Error E = Rewriter->preCFGInitializer()) {
      errs() << "BOLT-ERROR: while running " << Rewriter->getName()
             << " in pre-CFG state: " << toString(std::move(E)) << '\n';
      exit(1);
    }
  }
}

void MetadataManager::runInitializersPostCFG() {
  for (auto &Rewriter : Rewriters) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: invoking " << Rewriter->getName()
                      << " after CFG construction\n");
    if (Error E = Rewriter->postCFGInitializer()) {
      errs() << "BOLT-ERROR: while running " << Rewriter->getName()
             << " in CFG state: " << toString(std::move(E)) << '\n';
      exit(1);
    }
  }
}

void MetadataManager::runFinalizersPreEmit() {
  for (auto &Rewriter : Rewriters) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: invoking " << Rewriter->getName()
                      << " before emitting binary context\n");
    if (Error E = Rewriter->preEmitFinalizer()) {
      errs() << "BOLT-ERROR: while running " << Rewriter->getName()
             << " before emit: " << toString(std::move(E)) << '\n';
      exit(1);
    }
  }
}

void MetadataManager::runFinalizersAfterEmit() {
  for (auto &Rewriter : Rewriters) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: invoking " << Rewriter->getName()
                      << " after emit\n");
    if (Error E = Rewriter->postEmitFinalizer()) {
      errs() << "BOLT-ERROR: while running " << Rewriter->getName()
             << " after emit: " << toString(std::move(E)) << '\n';
      exit(1);
    }
  }
}
