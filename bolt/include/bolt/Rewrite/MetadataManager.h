//===- bolt/Rewrite/MetadataManager.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_METADATA_MANAGER_H
#define BOLT_REWRITE_METADATA_MANAGER_H

#include "bolt/Rewrite/MetadataRewriter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace bolt {

class BinaryContext;

/// This class manages a collection of metadata handlers/rewriters.
/// It is responsible for registering new rewriters and invoking them at
/// certain stages of the binary processing pipeline.
class MetadataManager {
  using RewritersListType = SmallVector<std::unique_ptr<MetadataRewriter>, 1>;
  RewritersListType Rewriters;

public:
  /// Register a new \p Rewriter.
  void registerRewriter(std::unique_ptr<MetadataRewriter> Rewriter);

  /// Execute initialization of rewriters while functions are disassembled, but
  /// CFG is not yet built.
  void runInitializersPreCFG();

  /// Execute metadata initializers after CFG was constructed for functions.
  void runInitializersPostCFG();

  /// Run finalization step of rewriters before the binary is emitted.
  void runFinalizersPreEmit();

  /// Run finalization step of rewriters after code has been emitted.
  void runFinalizersAfterEmit();
};

} // namespace bolt
} // namespace llvm

#endif // BOLT_REWRITE_METADATA_MANAGER_H
