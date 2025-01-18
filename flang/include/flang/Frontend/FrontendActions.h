//===- FrontendActions.h -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FRONTEND_FRONTENDACTIONS_H
#define FORTRAN_FRONTEND_FRONTENDACTIONS_H

#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Frontend/FrontendAction.h"
#include "flang/Parser/parsing.h"
#include "flang/Semantics/semantics.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include <memory>

namespace Fortran::frontend {

// TODO: This is a copy from f18.cpp. It doesn't really belong here and should
// be moved to a more suitable place in future.
struct MeasurementVisitor {
  template <typename A>
  bool Pre(const A &) {
    return true;
  }
  template <typename A>
  void Post(const A &) {
    ++objects;
    bytes += sizeof(A);
  }
  size_t objects{0}, bytes{0};
};

//===----------------------------------------------------------------------===//
// Custom Consumer Actions
//===----------------------------------------------------------------------===//

class InputOutputTestAction : public FrontendAction {
  void executeAction() override;
};

class InitOnlyAction : public FrontendAction {
  void executeAction() override;
};

//===----------------------------------------------------------------------===//
// Prescan Actions
//===----------------------------------------------------------------------===//
class PrescanAction : public FrontendAction {
  void executeAction() override = 0;
  bool beginSourceFileAction() override;
};

class PrintPreprocessedAction : public PrescanAction {
  void executeAction() override;
};

class DebugDumpProvenanceAction : public PrescanAction {
  void executeAction() override;
};

class DebugDumpParsingLogAction : public PrescanAction {
  void executeAction() override;
};

class DebugMeasureParseTreeAction : public PrescanAction {
  void executeAction() override;
};

//===----------------------------------------------------------------------===//
// PrescanAndParse Actions
//===----------------------------------------------------------------------===//
class PrescanAndParseAction : public FrontendAction {
  void executeAction() override = 0;
  bool beginSourceFileAction() override;
};

class DebugUnparseNoSemaAction : public PrescanAndParseAction {
  void executeAction() override;
};

class DebugDumpParseTreeNoSemaAction : public PrescanAndParseAction {
  void executeAction() override;
};

//===----------------------------------------------------------------------===//
// PrescanAndSema Actions
//
// These actions will parse the input, run the semantic checks and execute
// their actions provided that no parsing or semantic errors were found.
//===----------------------------------------------------------------------===//
class PrescanAndSemaAction : public FrontendAction {

  void executeAction() override = 0;
  bool beginSourceFileAction() override;
};

class DebugUnparseWithSymbolsAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class DebugUnparseWithModulesAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class DebugUnparseAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class DebugDumpSymbolsAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class DebugDumpParseTreeAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class DebugDumpPFTAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class DebugPreFIRTreeAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class GetDefinitionAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class GetSymbolsSourcesAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class ParseSyntaxOnlyAction : public PrescanAndSemaAction {
  void executeAction() override;
};

class PluginParseTreeAction : public PrescanAndSemaAction {
  void executeAction() override = 0;

public:
  Fortran::parser::Parsing &getParsing();
  /// Creates an output file. This is just a wrapper for calling
  /// CreateDefaultOutputFile from CompilerInstance. Use it to make sure that
  /// your plugin respects driver's `-o` flag.
  /// \param extension  The extension to use for the output file (ignored when
  ///                   the user decides to print to stdout via `-o -`)
  /// \return           Null on error, ostream for the output file otherwise
  std::unique_ptr<llvm::raw_pwrite_stream>
  createOutputFile(llvm::StringRef extension);
};

//===----------------------------------------------------------------------===//
// PrescanAndSemaDebug Actions
//
// These actions will parse the input, run the semantic checks and execute
// their actions _regardless of_ whether any semantic errors have been found.
// This can be useful when adding new languge feature and when you wish to
// investigate compiler output (e.g. the parse tree) despite any semantic
// errors.
//
// NOTE: Use with care and for development only!
//===----------------------------------------------------------------------===//
class PrescanAndSemaDebugAction : public FrontendAction {

  void executeAction() override = 0;
  bool beginSourceFileAction() override;
};

class DebugDumpAllAction : public PrescanAndSemaDebugAction {
  void executeAction() override;
};

//===----------------------------------------------------------------------===//
// CodeGen Actions
//===----------------------------------------------------------------------===//
/// Represents the type of "backend" action to perform by the corresponding
/// CodeGenAction. Note that from Flang's perspective, both LLVM and MLIR are
/// "backends" that are used for generating LLVM IR/BC, assembly files or
/// machine code. This enum captures "what" exactly one of these backends is to
/// do. The names are similar to what is used in Clang - this allows us to
/// maintain some level of consistency/similarity between the drivers.
enum class BackendActionTy {
  Backend_EmitAssembly, ///< Emit native assembly files
  Backend_EmitObj,      ///< Emit native object files
  Backend_EmitBC,       ///< Emit LLVM bitcode files
  Backend_EmitLL,       ///< Emit human-readable LLVM assembly
  Backend_EmitFIR,      ///< Emit FIR files, possibly lowering via HLFIR
  Backend_EmitHLFIR,    ///< Emit HLFIR files before any passes run
};

/// Abstract base class for actions that generate code (MLIR, LLVM IR, assembly
/// and machine code). Every action that inherits from this class will at
/// least run the prescanning, parsing, semantic checks and lower the parse
/// tree to an MLIR module.
class CodeGenAction : public FrontendAction {

  void executeAction() override;
  /// Runs prescan, parsing, sema and lowers to MLIR.
  bool beginSourceFileAction() override;
  /// Runs the optimization (aka middle-end) pipeline on the LLVM module
  /// associated with this action.
  void runOptimizationPipeline(llvm::raw_pwrite_stream &os);

protected:
  CodeGenAction(BackendActionTy act) : action{act} {};
  /// @name MLIR
  /// {
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  mlir::OwningOpRef<mlir::ModuleOp> mlirModule;
  /// }

  /// @name LLVM IR
  std::unique_ptr<llvm::LLVMContext> llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule;

  /// Embeds offload objects specified with -fembed-offload-object
  void embedOffloadObjects();

  /// Links in BC libraries spefified with -mlink-builtin-bitcode
  void linkBuiltinBCLibs();

  /// Runs pass pipeline to lower HLFIR into FIR
  void lowerHLFIRToFIR();

  /// Generates an LLVM IR module from CodeGenAction::mlirModule and saves it
  /// in CodeGenAction::llvmModule.
  void generateLLVMIR();

  BackendActionTy action;

  /// }
public:
  ~CodeGenAction() override;
};

class EmitFIRAction : public CodeGenAction {
public:
  EmitFIRAction() : CodeGenAction(BackendActionTy::Backend_EmitFIR) {}
};

class EmitHLFIRAction : public CodeGenAction {
public:
  EmitHLFIRAction() : CodeGenAction(BackendActionTy::Backend_EmitHLFIR) {}
};

class EmitLLVMAction : public CodeGenAction {
public:
  EmitLLVMAction() : CodeGenAction(BackendActionTy::Backend_EmitLL) {}
};

class EmitLLVMBitcodeAction : public CodeGenAction {
public:
  EmitLLVMBitcodeAction() : CodeGenAction(BackendActionTy::Backend_EmitBC) {}
};

class EmitObjAction : public CodeGenAction {
public:
  EmitObjAction() : CodeGenAction(BackendActionTy::Backend_EmitObj) {}
};

class EmitAssemblyAction : public CodeGenAction {
public:
  EmitAssemblyAction() : CodeGenAction(BackendActionTy::Backend_EmitAssembly) {}
};

} // namespace Fortran::frontend

#endif // FORTRAN_FRONTEND_FRONTENDACTIONS_H
