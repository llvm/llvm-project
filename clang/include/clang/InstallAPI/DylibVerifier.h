//===- InstallAPI/DylibVerifier.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_DYLIBVERIFIER_H
#define LLVM_CLANG_INSTALLAPI_DYLIBVERIFIER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/InstallAPI/MachO.h"

namespace clang {
namespace installapi {
struct FrontendAttrs;

/// A list of InstallAPI verification modes.
enum class VerificationMode {
  Invalid,
  ErrorsOnly,
  ErrorsAndWarnings,
  Pedantic,
};

/// Service responsible to tracking state of verification across the
/// lifetime of InstallAPI.
/// As declarations are collected during AST traversal, they are
/// compared as symbols against what is available in the binary dylib.
class DylibVerifier {
private:
  struct SymbolContext;

public:
  enum class Result { NoVerify, Ignore, Valid, Invalid };
  struct VerifierContext {
    // Current target being verified against the AST.
    llvm::MachO::Target Target;

    // Target specific API from binary.
    RecordsSlice *DylibSlice;

    // Query state of verification after AST has been traversed.
    Result FrontendState;

    // First error for AST traversal, which is tied to the target triple.
    bool DiscoveredFirstError;

    // Determines what kind of banner to print a violation for.
    bool PrintArch = false;

    // Engine for reporting violations.
    DiagnosticsEngine *Diag = nullptr;

    // Handle diagnostics reporting for target level violations.
    void emitDiag(llvm::function_ref<void()> Report);

    VerifierContext() = default;
    VerifierContext(DiagnosticsEngine *Diag) : Diag(Diag) {}
  };

  DylibVerifier() = default;

  DylibVerifier(llvm::MachO::Records &&Dylib, DiagnosticsEngine *Diag,
                VerificationMode Mode, bool Demangle)
      : Dylib(std::move(Dylib)), Mode(Mode), Demangle(Demangle),
        Exports(std::make_unique<SymbolSet>()), Ctx(VerifierContext{Diag}) {}

  Result verify(GlobalRecord *R, const FrontendAttrs *FA);
  Result verify(ObjCInterfaceRecord *R, const FrontendAttrs *FA);
  Result verify(ObjCIVarRecord *R, const FrontendAttrs *FA,
                const StringRef SuperClass);

  /// Initialize target for verification.
  void setTarget(const Target &T);

  /// Release ownership over exports.
  std::unique_ptr<SymbolSet> getExports() { return std::move(Exports); }

  /// Get result of verification.
  Result getState() const { return Ctx.FrontendState; }

  /// Set different source managers to the same diagnostics engine.
  void setSourceManager(SourceManager &SourceMgr) const {
    if (!Ctx.Diag)
      return;
    Ctx.Diag->setSourceManager(&SourceMgr);
  }

private:
  /// Determine whether to compare declaration to symbol in binary.
  bool canVerify();

  /// Shared implementation for verifying exported symbols.
  Result verifyImpl(Record *R, SymbolContext &SymCtx);

  /// Check if declaration is marked as obsolete, they are
  // expected to result in a symbol mismatch.
  bool shouldIgnoreObsolete(const Record *R, SymbolContext &SymCtx,
                            const Record *DR);

  /// Compare the visibility declarations to the linkage of symbol found in
  /// dylib.
  Result compareVisibility(const Record *R, SymbolContext &SymCtx,
                           const Record *DR);

  /// An ObjCInterfaceRecord can represent up to three symbols. When verifying,
  // account for this granularity.
  bool compareObjCInterfaceSymbols(const Record *R, SymbolContext &SymCtx,
                                   const ObjCInterfaceRecord *DR);

  /// Validate availability annotations against dylib.
  Result compareAvailability(const Record *R, SymbolContext &SymCtx,
                             const Record *DR);

  /// Compare and validate matching symbol flags.
  bool compareSymbolFlags(const Record *R, SymbolContext &SymCtx,
                          const Record *DR);

  /// Update result state on each call to `verify`.
  void updateState(Result State);

  /// Add verified exported symbol.
  void addSymbol(const Record *R, SymbolContext &SymCtx,
                 TargetList &&Targets = {});

  /// Find matching dylib slice for target triple that is being parsed.
  void assignSlice(const Target &T);

  // Symbols in dylib.
  llvm::MachO::Records Dylib;

  // Controls what class of violations to report.
  VerificationMode Mode = VerificationMode::Invalid;

  // Attempt to demangle when reporting violations.
  bool Demangle = false;

  // Valid symbols in final text file.
  std::unique_ptr<SymbolSet> Exports = std::make_unique<SymbolSet>();

  // Track current state of verification while traversing AST.
  VerifierContext Ctx;
};

} // namespace installapi
} // namespace clang
#endif // LLVM_CLANG_INSTALLAPI_DYLIBVERIFIER_H
