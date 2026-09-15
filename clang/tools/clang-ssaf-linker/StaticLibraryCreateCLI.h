//===- StaticLibraryCreateCLI.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Declares the CLI action class for `clang-ssaf-linker static-library
//  create`: validates inputs and output, streams TU summaries into a
//  StaticLibrary one at a time (reading and inserting each in turn),
//  and serializes the result.
//
//  The class is intentionally independent of the tool's cl::opt globals.
//  Every input it needs is passed via a Config struct at run() time, so
//  the class can be reused or unit-tested outside the driver.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_STATICLIBRARYCREATECLI_H
#define LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_STATICLIBRARYCREATECLI_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/StaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Tool/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace clang::ssaf {

/// Runs the `static-library create` action for `clang-ssaf-linker`.
class StaticLibraryCreateCLI {
public:
  /// Everything the action needs at runtime, threaded in from the CLI
  /// driver. StringRef/ArrayRef fields alias the driver's cl::opt storage
  /// and must remain valid for the duration of run().
  struct Config {
    llvm::ArrayRef<std::string> InputPaths;
    llvm::StringRef OutputPath;
    llvm::StringRef Namespace;
    llvm::StringRef TargetTriple;
    bool Verbose = false;
    bool Time = false;
  };

  /// Orchestrator: validate → bundle → write. Non-recoverable errors call
  /// fail() from Tool/Utils.h and terminate the process.
  void run(llvm::TimerGroup &TG, const Config &Cfg);

private:
  /// Validates the output path, input paths, resolves the namespace name,
  /// and validates Cfg.TargetTriple if it is set.
  void validate(llvm::TimerGroup &TG);

  /// Reads each validated input file and inserts it into a StaticLibrary
  /// one at a time.
  ///
  /// \returns The assembled StaticLibrary. Terminates the process via
  ///          fail() on any read error, triple mismatch, or duplicate
  ///          TUNamespace across inputs.
  StaticLibrary bundle(llvm::TimerGroup &TG);

  /// Serializes the StaticLibrary to the validated output path.
  void write(llvm::TimerGroup &TG, const StaticLibrary &Result);

  /// Prints one indented note to stderr when Cfg.Verbose is set.
  template <typename... Ts>
  void info(unsigned Level, const char *Fmt, Ts &&...Args) const {
    if (Cfg.Verbose) {
      llvm::WithColor::note()
          << std::string(Level * IndentationWidth, ' ') << "- "
          << llvm::formatv(Fmt, std::forward<Ts>(Args)...) << "\n";
    }
  }

  static constexpr unsigned IndentationWidth = 2;

  // Configuration set by run() before dispatching to phase methods.
  Config Cfg;

  // State populated during validate() and consumed by later phases.
  FormatFile OutputFile;
  std::vector<FormatFile> InputFiles;
  std::string NamespaceName;

  // Set by validate() only when Cfg.TargetTriple was passed and valid.
  // bundle() uses it to construct the StaticLibrary up front; when unset,
  // bundle() populates it from the first input, and every subsequent
  // input must match.
  std::optional<llvm::Triple> ResolvedTriple;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_STATICLIBRARYCREATECLI_H
