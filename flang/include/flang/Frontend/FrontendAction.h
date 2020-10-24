//===- FrontendAction.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the flang::FrontendAction interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_FRONTEND_FRONTENDACTION_H
#define LLVM_FLANG_FRONTEND_FRONTENDACTION_H

#include "flang/Frontend/FrontendOptions.h"
#include "llvm/Support/Error.h"

namespace Fortran::frontend {
class CompilerInstance;

/// Abstract base class for actions which can be performed by the frontend.
class FrontendAction {
  FrontendInputFile currentInput_;
  CompilerInstance *instance_;

protected:
  /// @name Implementation Action Interface
  /// @{

  /// Callback to run the program action, using the initialized
  /// compiler instance.
  virtual void ExecuteAction() = 0;

  /// Callback at the end of processing a single input, to determine
  /// if the output files should be erased or not.
  ///
  /// By default it returns true if a compiler error occurred.
  virtual bool ShouldEraseOutputFiles();

  /// @}

public:
  FrontendAction() : instance_(nullptr) {}
  virtual ~FrontendAction() = default;

  /// @name Compiler Instance Access
  /// @{

  CompilerInstance &GetCompilerInstance() const {
    assert(instance_ && "Compiler instance not registered!");
    return *instance_;
  }

  void SetCompilerInstance(CompilerInstance *value) { instance_ = value; }

  /// @}
  /// @name Current File Information
  /// @{

  const FrontendInputFile &GetCurrentInput() const { return currentInput_; }

  llvm::StringRef GetCurrentFile() const {
    assert(!currentInput_.IsEmpty() && "No current file!");
    return currentInput_.GetFile();
  }

  llvm::StringRef GetCurrentFileOrBufferName() const {
    assert(!currentInput_.IsEmpty() && "No current file!");
    return currentInput_.IsFile()
        ? currentInput_.GetFile()
        : currentInput_.GetBuffer()->getBufferIdentifier();
  }
  void SetCurrentInput(const FrontendInputFile &currentInput);

  /// @}
  /// @name Public Action Interface
  /// @}

  /// Prepare the action for processing the input file \p input.
  ///
  /// This is run after the options and frontend have been initialized,
  /// but prior to executing any per-file processing.
  /// \param ci - The compiler instance this action is being run from. The
  /// action may store and use this object.
  /// \param input - The input filename and kind.
  /// \return True on success; on failure the compilation of this file should
  bool BeginSourceFile(CompilerInstance &ci, const FrontendInputFile &input);

  /// Run the action.
  llvm::Error Execute();

  /// Perform any per-file post processing, deallocate per-file
  /// objects, and run statistics and output file cleanup code.
  void EndSourceFile();
};

} // namespace Fortran::frontend

#endif // LLVM_FLANG_FRONTEND_FRONTENDACTION_H
