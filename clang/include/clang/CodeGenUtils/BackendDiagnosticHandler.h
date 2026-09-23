//===--- BackendDiagnosticHandler.h - Shared Backend Diagnostics -C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares BackendDiagnosticConsumer, which turns LLVM backend
// diagnostics (llvm::DiagnosticInfo) into clang::DiagnosticsEngine
// diagnostics. It is shared between classic CodeGen (BackendConsumer) and
// CIR (CIRGenConsumer) so that both LLVM-emitting pipelines report backend
// diagnostics (optimization remarks, inline-asm errors, unsupported
// features, etc.) through the same clang diagnostics machinery instead of
// falling back to LLVM's default stderr-printing handler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_BACKENDDIAGNOSTICHANDLER_H
#define LLVM_CLANG_CODEGENUTILS_BACKENDDIAGNOSTICHANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DiagnosticHandler.h"
#include <memory>
#include <optional>
#include <vector>

namespace llvm {
class DiagnosticInfo;
class DiagnosticInfoDontCall;
class DiagnosticInfoInlineAsm;
class DiagnosticInfoMisExpect;
class DiagnosticInfoOptimizationBase;
class DiagnosticInfoOptimizationFailure;
class DiagnosticInfoResourceLimit;
class DiagnosticInfoSrcMgr;
class DiagnosticInfoStackSize;
class DiagnosticInfoUnsupported;
class DiagnosticInfoUnsupportedTargetIntrinsic;
class DiagnosticInfoWithLocationBase;
class Function;
class Module;
class OptimizationRemarkAnalysisAliasing;
class OptimizationRemarkAnalysisFPCommute;
} // namespace llvm

namespace clang {
class CodeGenOptions;
class DiagnosticsEngine;
class SourceManager;

/// Owns the state needed to translate LLVM backend diagnostics into clang
/// diagnostics, and implements the translation itself. This is not itself an
/// llvm::DiagnosticHandler so that it can be installed/updated independently
/// of the llvm::LLVMContext's handler lifetime; use
/// BackendDiagnosticConsumer::createDiagnosticHandler() to get an
/// llvm::DiagnosticHandler that forwards into it.
class BackendDiagnosticConsumer {
public:
  BackendDiagnosticConsumer(DiagnosticsEngine &Diags,
                            const CodeGenOptions &CodeGenOpts)
      : Diags(Diags), CodeGenOpts(CodeGenOpts) {}

  /// The SourceManager used to translate backend-reported file:line:col
  /// locations back into clang SourceLocations. May be null (e.g. when
  /// compiling a raw LLVM IR input file with no clang AST), in which case
  /// diagnostics are still reported, just without clang-level source
  /// locations.
  void setSourceManager(SourceManager *SM) { this->SM = SM; }

  /// Set the module currently being linked in, used to name DK_Linker
  /// diagnostics.
  void setCurLinkModule(llvm::Module *M) { CurLinkModule = M; }

  /// Record the source location of the function with the given mangled
  /// name, used to approximate a location for backend diagnostics (e.g.
  /// stack-size warnings) that don't carry debug-info locations of their
  /// own.
  void addFunctionSourceLocation(StringRef MangledName, FullSourceLoc Loc);

  std::optional<FullSourceLoc>
  getFunctionSourceLocation(const llvm::Function &F) const;

  /// Get the best possible source location to represent a diagnostic that
  /// may have associated debug info.
  FullSourceLoc
  getBestLocationFromDebugLoc(const llvm::DiagnosticInfoWithLocationBase &D,
                              bool &BadDebugInfo, StringRef &Filename,
                              unsigned &Line, unsigned &Column) const;

  /// Create an llvm::DiagnosticHandler that forwards diagnostics to this
  /// consumer. The returned handler must not outlive this consumer.
  std::unique_ptr<llvm::DiagnosticHandler> createDiagnosticHandler();

  /// This is invoked when the backend needs to report something to the
  /// user.
  void handleDiagnostics(const llvm::DiagnosticInfo &DI);

private:
  bool InlineAsmDiagHandler(const llvm::DiagnosticInfoInlineAsm &D);
  void SrcMgrDiagHandler(const llvm::DiagnosticInfoSrcMgr &D);
  bool StackSizeDiagHandler(const llvm::DiagnosticInfoStackSize &D);
  bool ResourceLimitDiagHandler(const llvm::DiagnosticInfoResourceLimit &D);
  void UnsupportedDiagHandler(const llvm::DiagnosticInfoUnsupported &D);
  void UnsupportedTargetIntrinsicDiagHandler(
      const llvm::DiagnosticInfoUnsupportedTargetIntrinsic &D);
  void EmitOptimizationMessage(const llvm::DiagnosticInfoOptimizationBase &D,
                               unsigned DiagID);
  void OptimizationRemarkHandler(const llvm::DiagnosticInfoOptimizationBase &D);
  void
  OptimizationRemarkHandler(const llvm::OptimizationRemarkAnalysisFPCommute &D);
  void
  OptimizationRemarkHandler(const llvm::OptimizationRemarkAnalysisAliasing &D);
  void
  OptimizationFailureHandler(const llvm::DiagnosticInfoOptimizationFailure &D);
  void DontCallDiagHandler(const llvm::DiagnosticInfoDontCall &D);
  void MisExpectDiagHandler(const llvm::DiagnosticInfoMisExpect &D);

  DiagnosticsEngine &Diags;
  const CodeGenOptions &CodeGenOpts;
  SourceManager *SM = nullptr;
  llvm::Module *CurLinkModule = nullptr;

  // A map from mangled names to their function's source location, used for
  // backend diagnostics as the clang AST may be unavailable. We actually use
  // the mangled name's hash as the key because mangled names can be very
  // long and take up lots of space. Using a hash can cause name collision,
  // but that is rare and the consequences are pointing to a wrong source
  // location which is not severe. This is a vector instead of an actual map
  // because we optimize for time building this map rather than time
  // retrieving an entry, as backend diagnostics are uncommon.
  std::vector<std::pair<llvm::hash_code, FullSourceLoc>> ManglingFullSourceLocs;
};

} // namespace clang

#endif // LLVM_CLANG_CODEGENUTILS_BACKENDDIAGNOSTICHANDLER_H
