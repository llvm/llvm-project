//===- MultiArchCreateCLI.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Declares the CLI action class for `clang-ssaf-linker multi-arch create`.
//  Bundles StaticLibrary/MultiArchStaticLibrary inputs into one
//  MultiArchStaticLibrary, or LUSummaryEncoding/MultiArchSharedLibrary inputs
//  into one MultiArchSharedLibrary.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_MULTIARCHCREATECLI_H
#define LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_MULTIARCHCREATECLI_H

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchSharedLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchStaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/StaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormat.h"
#include "clang/ScalableStaticAnalysis/Tool/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Timer.h"
#include <memory>
#include <string>
#include <vector>

namespace clang::ssaf {

/// Runs the `multi-arch create` action for `clang-ssaf-linker`.
class MultiArchCreateCLI {
public:
  /// Orchestrates validation, construction, and serialization of the MultiArch
  /// artifact.
  void run(llvm::TimerGroup &TG, llvm::ArrayRef<std::string> InputPaths,
           llvm::StringRef OutputPath, bool Verbose, bool Time);

private:
  /// Validates the output path and every input path.
  void validate(unsigned Level, llvm::Timer &TValidate);

  /// Constructs the artifact from the inputs.
  ArtifactEncoding create(unsigned Level, llvm::Timer &TRead,
                          llvm::Timer &TBundle);

  /// Reads Artifact from file at \p Index.
  ArtifactEncoding readInput(size_t Index, unsigned Level, llvm::Timer &TRead);

  /// Returns the namespace of the static-family input.
  static const BuildNamespace &staticFamilyNamespace(const ArtifactEncoding &E);

  /// Returns the namespace of the shared-family input.
  static const NestedBuildNamespace &
  sharedFamilyNamespace(const ArtifactEncoding &E);

  /// Constructs a MultiArchStaticLibrary from the inputs.
  ArtifactEncoding createStaticLibrary(ArtifactEncoding First, unsigned Level,
                                       llvm::Timer &TRead,
                                       llvm::Timer &TBundle);

  /// Constructs a MultiArchSharedLibrary from the inputs.
  ArtifactEncoding createSharedLibrary(ArtifactEncoding First, unsigned Level,
                                       llvm::Timer &TRead,
                                       llvm::Timer &TBundle);

  /// Adds one input to the \p Bundle.
  void addStaticInput(MultiArchStaticLibrary &Bundle, ArtifactEncoding Encoding,
                      size_t Index, unsigned Level, llvm::Timer &TBundle);
  void addSharedInput(MultiArchSharedLibrary &Bundle, ArtifactEncoding Encoding,
                      size_t Index, unsigned Level, llvm::Timer &TBundle);

  /// Inserts one member, failing on duplicate target triple.
  void addStaticMember(MultiArchStaticLibrary &Bundle,
                       std::unique_ptr<StaticLibrary> Member,
                       llvm::StringRef SourceFile);
  void addSharedMember(MultiArchSharedLibrary &Bundle,
                       std::unique_ptr<LUSummaryEncoding> Member,
                       llvm::StringRef SourceFile);

  /// Serializes the artifact to the validated output path.
  void write(const ArtifactEncoding &Bundle, unsigned Level,
             llvm::Timer &TWrite);

  // Arguments captured by run() before dispatching to bundling methods.
  // InputPaths and OutputPath are non-owning: they alias the driver's cl::opt
  // storage, which outlives the call.
  llvm::ArrayRef<std::string> InputPaths;
  llvm::StringRef OutputPath;
  bool Verbose = false;
  bool Time = false;

  // State populated during validate() and consumed by later phases.
  FormatFile OutputFile;
  std::vector<FormatFile> InputFiles;

  /// Maps each inserted member to the input file that contributed it, so a
  /// duplicate can name both contributors. Keyed on the member's address, not
  /// its triple spelling: the member sets are keyed by Triple enum components,
  /// which fold alias spellings ("arm64" / "aarch64") and OS versions, so two
  /// colliding slices can carry triple strings that differ. The values alias
  /// InputFiles[I].Path, which outlives the run.
  llvm::DenseMap<const void *, llvm::StringRef> SourceByMember;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_TOOLS_CLANG_SSAF_LINKER_MULTIARCHCREATECLI_H
