//===--- Interpreter.h - Incremental Compilation and Execution---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the component which performs incremental code
// compilation and execution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_INTERPRETER_H
#define LLVM_CLANG_INTERPRETER_INTERPRETER_H

#include "clang/AST/GlobalDecl.h"
#include "clang/Interpreter/PartialTranslationUnit.h"
#include "clang/Interpreter/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <vector>

namespace llvm {
namespace orc {
class LLJIT;
class LLJITBuilder;
class ThreadSafeContext;
} // namespace orc
} // namespace llvm

namespace clang {

class CompilerInstance;
class CodeGenerator;
class CXXRecordDecl;
class Decl;
class IncrementalExecutor;
class IncrementalParser;

/// Create a pre-configured \c CompilerInstance for incremental processing.
class IncrementalCompilerBuilder {
public:
  IncrementalCompilerBuilder() {}

  void SetCompilerArgs(const std::vector<const char *> &Args) {
    UserArgs = Args;
  }

  void SetTargetTriple(std::string TT) { TargetTriple = TT; }

  // General C++
  llvm::Expected<std::unique_ptr<CompilerInstance>> CreateCpp();

  // Offload options
  void SetOffloadArch(llvm::StringRef Arch) { OffloadArch = Arch; };

  // CUDA specific
  void SetCudaSDK(llvm::StringRef path) { CudaSDKPath = path; };

  llvm::Expected<std::unique_ptr<CompilerInstance>> CreateCudaHost();
  llvm::Expected<std::unique_ptr<CompilerInstance>> CreateCudaDevice();

private:
  static llvm::Expected<std::unique_ptr<CompilerInstance>>
  create(std::string TT, std::vector<const char *> &ClangArgv);

  llvm::Expected<std::unique_ptr<CompilerInstance>> createCuda(bool device);

  std::vector<const char *> UserArgs;
  std::optional<std::string> TargetTriple;

  llvm::StringRef OffloadArch;
  llvm::StringRef CudaSDKPath;
};

class IncrementalAction;
class InProcessPrintingASTConsumer;

/// Provides top-level interfaces for incremental compilation and execution.
class Interpreter {
  friend class Value;
  friend InProcessPrintingASTConsumer;

  std::unique_ptr<llvm::orc::ThreadSafeContext> TSCtx;
  /// Long-lived, incremental parsing action.
  std::unique_ptr<IncrementalAction> Act;
  std::unique_ptr<IncrementalParser> IncrParser;
  std::unique_ptr<IncrementalExecutor> IncrExecutor;

  // An optional parser for CUDA offloading
  std::unique_ptr<IncrementalParser> DeviceParser;

  /// List containing information about each incrementally parsed piece of code.
  std::list<PartialTranslationUnit> PTUs;

  unsigned InitPTUSize = 0;

  // This member holds the last result of the value printing. It's a class
  // member because we might want to access it after more inputs. If no value
  // printing happens, it's in an invalid state.
  Value LastValue;

  /// When CodeGen is created the first llvm::Module gets cached in many places
  /// and we must keep it alive.
  std::unique_ptr<llvm::Module> CachedInCodeGenModule;

  /// Compiler instance performing the incremental compilation.
  std::unique_ptr<CompilerInstance> CI;

protected:
  // Derived classes can use an extended interface of the Interpreter.
  Interpreter(std::unique_ptr<CompilerInstance> Instance, llvm::Error &Err,
              std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder = nullptr,
              std::unique_ptr<clang::ASTConsumer> Consumer = nullptr);

  // Create the internal IncrementalExecutor, or re-create it after calling
  // ResetExecutor().
  llvm::Error CreateExecutor();

  // Delete the internal IncrementalExecutor. This causes a hard shutdown of the
  // JIT engine. In particular, it doesn't run cleanup or destructors.
  void ResetExecutor();

public:
  virtual ~Interpreter();
  static llvm::Expected<std::unique_ptr<Interpreter>>
  create(std::unique_ptr<CompilerInstance> CI);
  static llvm::Expected<std::unique_ptr<Interpreter>>
  createWithCUDA(std::unique_ptr<CompilerInstance> CI,
                 std::unique_ptr<CompilerInstance> DCI);
  const ASTContext &getASTContext() const;
  ASTContext &getASTContext();
  const CompilerInstance *getCompilerInstance() const;
  CompilerInstance *getCompilerInstance();
  llvm::Expected<llvm::orc::LLJIT &> getExecutionEngine();

  llvm::Expected<PartialTranslationUnit &> Parse(llvm::StringRef Code);
  llvm::Error Execute(PartialTranslationUnit &T);
  llvm::Error ParseAndExecute(llvm::StringRef Code, Value *V = nullptr);

  /// Undo N previous incremental inputs.
  llvm::Error Undo(unsigned N = 1);

  /// Link a dynamic library
  llvm::Error LoadDynamicLibrary(const char *name);

  /// \returns the \c ExecutorAddr of a \c GlobalDecl. This interface uses
  /// the CodeGenModule's internal mangling cache to avoid recomputing the
  /// mangled name.
  llvm::Expected<llvm::orc::ExecutorAddr> getSymbolAddress(GlobalDecl GD) const;

  /// \returns the \c ExecutorAddr of a given name as written in the IR.
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddress(llvm::StringRef IRName) const;

  /// \returns the \c ExecutorAddr of a given name as written in the object
  /// file.
  llvm::Expected<llvm::orc::ExecutorAddr>
  getSymbolAddressFromLinkerName(llvm::StringRef LinkerName) const;

  const llvm::SmallVectorImpl<Expr *> &getValuePrintingInfo() const {
    return ValuePrintingInfo;
  }

  Expr *SynthesizeExpr(Expr *E);

private:
  size_t getEffectivePTUSize() const;
  void markUserCodeStart();
  llvm::Expected<Expr *> ExtractValueFromExpr(Expr *E);
  llvm::Expected<llvm::orc::ExecutorAddr> CompileDtorCall(CXXRecordDecl *CXXRD);

  CodeGenerator *getCodeGen() const;
  std::unique_ptr<llvm::Module> GenModule();
  PartialTranslationUnit &RegisterPTU(TranslationUnitDecl *TU);

  // A cache for the compiled destructors used to for de-allocation of managed
  // clang::Values.
  llvm::DenseMap<CXXRecordDecl *, llvm::orc::ExecutorAddr> Dtors;

  llvm::SmallVector<Expr *, 4> ValuePrintingInfo;

  std::unique_ptr<llvm::orc::LLJITBuilder> JITBuilder;
};
} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_INTERPRETER_H
