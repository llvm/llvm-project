//===- ExtractAPI/FrontendActions.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ExtractAPIAction frontend action.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_FRONTEND_ACTIONS_H
#define LLVM_CLANG_EXTRACTAPI_FRONTEND_ACTIONS_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {

/// ExtractAPIAction sets up the output file and creates the ExtractAPIVisitor.
class ExtractAPIAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

private:
  /// The synthesized input buffer that contains all the provided input header
  /// files.
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

public:
  /// Prepare to execute the action on the given CompilerInstance.
  ///
  /// This is called before executing the action on any inputs. This generates a
  /// single header that includes all of CI's inputs and replaces CI's input
  /// list with it before actually executing the action.
  bool PrepareToExecuteAction(CompilerInstance &CI) override;

  static std::unique_ptr<llvm::raw_pwrite_stream>
  CreateOutputFile(CompilerInstance &CI, StringRef InFile);
  static StringRef getInputBufferName() { return "<extract-api-includes>"; }
};

} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_FRONTEND_ACTIONS_H
