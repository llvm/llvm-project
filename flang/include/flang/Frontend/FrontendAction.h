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
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FRONTEND_FRONTENDACTION_H
#define FORTRAN_FRONTEND_FRONTENDACTION_H

#include "flang/Frontend/FrontendOptions.h"
#include "llvm/Support/Error.h"

namespace Fortran::frontend {
class CompilerInstance;

/// Abstract base class for actions which can be performed by the frontend.
class FrontendAction {
  FrontendInputFile currentInput;
  CompilerInstance *instance;

protected:
  /// @name Implementation Action Interface
  /// @{

  /// Callback to run the program action, using the initialized
  /// compiler instance.
  virtual void executeAction() = 0;

  /// Callback at the end of processing a single input, to determine
  /// if the output files should be erased or not.
  ///
  /// By default it returns true if a compiler error occurred.
  virtual bool shouldEraseOutputFiles();

  /// Callback at the start of processing a single input.
  ///
  /// \return True on success; on failure ExecutionAction() and
  /// EndSourceFileAction() will not be called.
  virtual bool beginSourceFileAction() { return true; }

  /// @}

public:
  FrontendAction() : instance(nullptr) {}
  virtual ~FrontendAction() = default;

  /// @name Compiler Instance Access
  /// @{

  CompilerInstance &getInstance() const {
    assert(instance && "Compiler instance not registered!");
    return *instance;
  }

  void setInstance(CompilerInstance *value) { instance = value; }

  /// @}
  /// @name Current File Information
  /// @{

  const FrontendInputFile &getCurrentInput() const { return currentInput; }

  llvm::StringRef getCurrentFile() const {
    assert(!currentInput.isEmpty() && "No current file!");
    return currentInput.getFile();
  }

  llvm::StringRef getCurrentFileOrBufferName() const {
    assert(!currentInput.isEmpty() && "No current file!");
    return currentInput.isFile()
               ? currentInput.getFile()
               : currentInput.getBuffer()->getBufferIdentifier();
  }
  void setCurrentInput(const FrontendInputFile &currentIntput);

  /// @}
  /// @name Public Action Interface
  /// @{

  /// Prepare the action for processing the input file \p input.
  ///
  /// This is run after the options and frontend have been initialized,
  /// but prior to executing any per-file processing.
  /// \param ci - The compiler instance this action is being run from. The
  /// action may store and use this object.
  /// \param input - The input filename and kind.
  /// \return True on success; on failure the compilation of this file should
  bool beginSourceFile(CompilerInstance &ci, const FrontendInputFile &input);

  /// Run the action.
  llvm::Error execute();

  /// Perform any per-file post processing, deallocate per-file
  /// objects, and run statistics and output file cleanup code.
  void endSourceFile();

  /// @}
protected:
  // Prescan the current input file. Return False if fatal errors are reported,
  // True otherwise.
  bool runPrescan();
  // Parse the current input file. Return False if fatal errors are reported,
  // True otherwise.
  bool runParse();
  // Run semantic checks for the current input file. Return False if fatal
  // errors are reported, True otherwise.
  bool runSemanticChecks();
  // Generate run-time type information for derived types. This may lead to new
  // semantic errors. Return False if fatal errors are reported, True
  // otherwise.
  bool generateRtTypeTables();

  // Report fatal semantic errors. Return True if present, false otherwise.
  bool reportFatalSemanticErrors();

  // Report fatal scanning errors. Return True if present, false otherwise.
  inline bool reportFatalScanningErrors() {
    return reportFatalErrors("Could not scan %0");
  }

  // Report fatal parsing errors. Return True if present, false otherwise
  inline bool reportFatalParsingErrors() {
    return reportFatalErrors("Could not parse %0");
  }

private:
  template <unsigned N>
  bool reportFatalErrors(const char (&message)[N]);
};

} // namespace Fortran::frontend

#endif // FORTRAN_FRONTEND_FRONTENDACTION_H
