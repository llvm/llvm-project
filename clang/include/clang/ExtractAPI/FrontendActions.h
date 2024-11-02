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

#include "clang/ExtractAPI/API.h"
#include "clang/Frontend/FrontendAction.h"

namespace clang {

/// ExtractAPIAction sets up the output file and creates the ExtractAPIVisitor.
class ExtractAPIAction : public ASTFrontendAction {
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override;

private:
  /// A representation of the APIs this action extracts.
  std::unique_ptr<extractapi::APISet> API;

  /// A stream to the output file of this action.
  std::unique_ptr<raw_pwrite_stream> OS;

  /// The product this action is extracting API information for.
  std::string ProductName;

  /// The synthesized input buffer that contains all the provided input header
  /// files.
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

  /// The input file originally provided on the command line.
  ///
  /// This captures the spelling used to include the file and whether the
  /// include is quoted or not.
  SmallVector<std::pair<SmallString<32>, bool>> KnownInputFiles;

  /// Prepare to execute the action on the given CompilerInstance.
  ///
  /// This is called before executing the action on any inputs. This generates a
  /// single header that includes all of CI's inputs and replaces CI's input
  /// list with it before actually executing the action.
  bool PrepareToExecuteAction(CompilerInstance &CI) override;

  /// Called after executing the action on the synthesized input buffer.
  ///
  /// Note: Now that we have gathered all the API definitions to surface we can
  /// emit them in this callback.
  void EndSourceFileAction() override;

  static std::unique_ptr<llvm::raw_pwrite_stream>
  CreateOutputFile(CompilerInstance &CI, StringRef InFile);

  static StringRef getInputBufferName() { return "<extract-api-includes>"; }
};

} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_FRONTEND_ACTIONS_H
