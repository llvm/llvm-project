//===- LinkCLI.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Declares the CLI action class for the link action of `clang-ssaf-linker`.
//  Links TU summaries, static libraries, and members of multi-arch static
//  libraries into one LU summary.
//
//  The class is intentionally independent of the tool's cl::opt globals.
//  Every input it needs is passed to run(), so the class can be reused or
//  unit-tested outside the driver.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_LINKCLI_H
#define LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_LINKCLI_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntityLinker.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormat.h"
#include "clang/ScalableStaticAnalysis/Tool/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Timer.h"
#include "llvm/TargetParser/Triple.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace clang::ssaf {

/// Runs the default linking action for `clang-ssaf-linker`.
class LinkCLI {
public:
  /// Orchestrates validation, linking, and serialization of the LU summary.
  /// Non-recoverable errors call fail() from Tool/Utils.h and terminate the
  /// process.
  void run(llvm::TimerGroup &TG, llvm::ArrayRef<std::string> InputPaths,
           llvm::StringRef OutputPath, llvm::StringRef TargetTriple,
           bool Verbose, bool Time);

private:
  /// Validates the output path and every input path, derives the link unit
  /// name, and validates TargetTriple if it is set.
  void validate(unsigned Level, llvm::Timer &TValidate);

  /// Reads the inputs and folds each into one link unit, in command line
  /// order.
  ///
  /// \returns The accumulated LU summary.
  LUSummaryEncoding link(unsigned Level, llvm::Timer &TRead,
                         llvm::Timer &TLink);

  /// Reads the artifact from \p Input.
  ///
  /// \param Index The input's position, reported as the note's [i/N] counter.
  ArtifactEncoding readInput(const FormatFile &Input, size_t Index,
                             unsigned Level, llvm::Timer &TRead);

  /// Determines the link unit's target triple.
  ///
  /// An explicit --target-triple wins. Otherwise the triple is inferred from
  /// \p First: its own for a TU summary or a static library, and its sole
  /// member's for a single-member multi-arch static library. Any other shape
  /// cannot be inferred from and requires --target-triple.
  ///
  /// \param SourceFile The path \p First was read from, named in diagnostics.
  llvm::Triple resolveTargetTriple(const ArtifactEncoding &First,
                                   llvm::StringRef SourceFile, unsigned Level);

  /// Folds one input into \p EL, reporting whatever EntityLinker rejects --
  /// including an input that does not belong to the resolved target -- with the
  /// input's path as context.
  ///
  /// \param SourceFile The input's path, named in diagnostics and notes.
  /// \param Index The input's position, reported as the note's [i/N] counter.
  void linkInput(EntityLinker &EL, ArtifactEncoding Encoding,
                 llvm::StringRef SourceFile, size_t Index, unsigned Level,
                 llvm::Timer &TLink);

  /// Serializes the LU summary to the validated output path.
  void write(const LUSummaryEncoding &Output, unsigned Level,
             llvm::Timer &TWrite);

  // Arguments captured by run() before dispatching to linking methods.
  // InputPaths, OutputPath, and TargetTriple are non-owning: they alias the
  // driver's cl::opt storage, which outlives the call.
  llvm::ArrayRef<std::string> InputPaths;
  llvm::StringRef OutputPath;
  llvm::StringRef TargetTriple;
  bool Verbose = false;
  bool Time = false;

  // State populated during validate() and consumed by later phases.
  FormatFile OutputFile;
  std::vector<FormatFile> InputFiles;
  std::string LinkUnitName;

  // The triple from --target-triple, parsed and validated by validate(), or
  // nullopt when the flag is not supplied.
  std::optional<llvm::Triple> ExplicitTriple;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_LINKCLI_H
