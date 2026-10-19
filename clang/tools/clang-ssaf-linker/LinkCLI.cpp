//===- LinkCLI.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements the default (no subcommand) linking action. Inputs are read one
//  at a time and folded into the link unit as they are read, so the first
//  input that cannot be accepted is the one reported.
//
//  The target triple is fixed from the first input (or from --target-triple)
//  before the linker is constructed. Validating every later input against it
//  happens here rather than in EntityLinker: choosing which inputs belong to
//  a target is a command line concern, and EntityLinker treats a mismatch as
//  a fatal precondition violation.
//
//===----------------------------------------------------------------------===//

#include "LinkCLI.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchSharedLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/Support/FormatProviders.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Path.h"
#include <cassert>
#include <memory>
#include <utility>
#include <variant>

using namespace llvm;
using namespace clang::ssaf;

namespace path = llvm::sys::path;

namespace {

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

constexpr const char *ReadingArtifact = "Reading artifact '{0}'";

constexpr const char *LinkingArtifact = "Linking artifact '{0}'";

constexpr const char *NoInputs =
    "no input artifacts: at least one input is required";

constexpr const char *NoMembersToInferFrom =
    "cannot infer target triple from '{0}': multi-arch static library has no "
    "members; pass --target-triple";

constexpr const char *AmbiguousMembersToInferFrom =
    "cannot infer target triple from '{0}': multi-arch static library has {1} "
    "members; pass --target-triple to select one";

constexpr const char *UnsupportedSharedInput =
    "'{0}' is a {1}: linking against shared libraries is not yet supported";

constexpr const char *LinkUnitSummaryName = "link unit summary";
constexpr const char *MultiArchSharedLibraryName = "multi-arch shared library";

//===----------------------------------------------------------------------===//
// ArtifactEncoding Helpers
//===----------------------------------------------------------------------===//

/// Returns the human readable kind of an artifact the linker cannot consume.
///
/// Only the shared-library family reaches this: every linkable alternative is
/// handled before it is called. The static_assert makes a new alternative a
/// compile error here rather than an unhandled case at runtime.
llvm::StringRef unsupportedInputKindName(const ArtifactEncoding &E) {
  static_assert(std::variant_size_v<ArtifactEncoding> == 5,
                "unsupportedInputKindName must cover every ArtifactEncoding "
                "alternative the linker cannot consume");

  if (std::holds_alternative<LUSummaryEncoding>(E)) {
    return LinkUnitSummaryName;
  }

  assert(
      std::holds_alternative<MultiArchSharedLibrary>(E) &&
      "linkable ArtifactEncoding alternatives must be handled by the caller");
  return MultiArchSharedLibraryName;
}

} // namespace

namespace clang::ssaf {

void LinkCLI::run(llvm::TimerGroup &TG, llvm::ArrayRef<std::string> InputPaths,
                  llvm::StringRef OutputPath, llvm::StringRef TargetTriple,
                  bool Verbose, bool Time) {
  this->InputPaths = InputPaths;
  this->OutputPath = OutputPath;
  this->TargetTriple = TargetTriple;
  this->Verbose = Verbose;
  this->Time = Time;

  llvm::Timer TValidate("validate", "Validate Input", TG);
  llvm::Timer TRead("read", "Read Artifacts", TG);
  llvm::Timer TLink("link", "Link Artifacts", TG);
  llvm::Timer TWrite("write", "Write Link Unit Summary", TG);

  // Nesting depth for indenting verbose notes.
  const unsigned Level = 0;

  info(Verbose, Level, "Linking started.");

  validate(Level + 1, TValidate);

  LUSummaryEncoding Output = link(Level + 1, TRead, TLink);

  write(Output, Level + 1, TWrite);

  info(Verbose, Level, "Linking finished.");

  // A second run() should start from a clean slate.
  InputFiles.clear();
  ExplicitTriple.reset();
}

void LinkCLI::validate(unsigned Level, llvm::Timer &TValidate) {
  info(Verbose, Level, "Validating input.");

  llvm::TimeRegion _(Time ? &TValidate : nullptr);

  OutputFile = FormatFile::fromOutputPath(OutputPath);
  LinkUnitName = path::stem(OutputFile.Path).str();
  info(Verbose, Level + 1, "Validated output path '{0}'.", OutputFile.Path);

  if (InputPaths.empty()) {
    fail(NoInputs);
  }
  for (const auto &InputPath : InputPaths) {
    InputFiles.push_back(FormatFile::fromInputPath(InputPath));
  }
  info(Verbose, Level + 1, "Validated {0} input artifact path(s).",
       InputFiles.size());

  if (!TargetTriple.empty()) {
    ExplicitTriple = parseTargetTripleOrFail("--target-triple", TargetTriple);
  }
}

LUSummaryEncoding LinkCLI::link(unsigned Level, llvm::Timer &TRead,
                                llvm::Timer &TLink) {
  info(Verbose, Level, "Creating link unit.");

  const unsigned InputLevel = Level + 1;
  info(Verbose, InputLevel, "Linking artifacts.");

  // The target triple comes from the first input, so it has to be read before
  // the linker can be constructed.
  constexpr size_t FirstIndex = 0;
  ArtifactEncoding First =
      readInput(InputFiles[FirstIndex], FirstIndex, InputLevel + 1, TRead);

  llvm::Triple LinkUnitTriple =
      resolveTargetTriple(First, InputFiles[FirstIndex].Path, InputLevel + 1);

  NestedBuildNamespace LUNamespace(
      BuildNamespace(BuildNamespaceKind::LinkUnit, LinkUnitName));
  EntityLinker EL(LinkUnitTriple, LUNamespace);

  linkInput(EL, std::move(First), InputFiles[FirstIndex].Path, FirstIndex,
            InputLevel + 1, TLink);
  for (size_t Index : llvm::seq<size_t>(FirstIndex + 1, InputFiles.size())) {
    linkInput(EL, readInput(InputFiles[Index], Index, InputLevel + 1, TRead),
              InputFiles[Index].Path, Index, InputLevel + 1, TLink);
  }

  info(Verbose, InputLevel, "Linked {0} translation unit(s).",
       EL.getLinkedTUCount());
  info(Verbose, InputLevel, "Target namespace: '{0}'.", LUNamespace);

  return std::move(EL).takeOutput();
}

ArtifactEncoding LinkCLI::readInput(const FormatFile &Input, size_t Index,
                                    unsigned Level, llvm::Timer &TRead) {
  info(Verbose, Level, "[{0}/{1}] Reading '{2}'.", Index + 1, InputFiles.size(),
       Input.Path);

  llvm::TimeRegion _(Time ? &TRead : nullptr);

  auto ExpectedEncoding = Input.Format->readArtifactEncoding(Input.Path);
  if (!ExpectedEncoding) {
    fail(ErrorBuilder::wrap(ExpectedEncoding.takeError())
             .context(ReadingArtifact, Input.Path)
             .build());
  }
  return std::move(*ExpectedEncoding);
}

llvm::Triple LinkCLI::resolveTargetTriple(const ArtifactEncoding &First,
                                          llvm::StringRef SourceFile,
                                          unsigned Level) {
  if (ExplicitTriple) {
    info(Verbose, Level, "Target triple: '{0}' (from --target-triple).",
         *ExplicitTriple);
    return *ExplicitTriple;
  }

  auto Inferred = [&]() -> llvm::Triple {
    if (const auto *TU = std::get_if<TUSummaryEncoding>(&First)) {
      return TU->getTargetTriple();
    }

    if (const auto *SL = std::get_if<StaticLibrary>(&First)) {
      return SL->TargetTriple;
    }

    if (const auto *MASL = std::get_if<MultiArchStaticLibrary>(&First)) {
      // A single member names the target unambiguously; anything else needs the
      // architecture to be chosen on the command line.
      if (MASL->Members.empty()) {
        fail(NoMembersToInferFrom, SourceFile);
      }
      if (MASL->Members.size() > 1) {
        fail(AmbiguousMembersToInferFrom, SourceFile, MASL->Members.size());
      }
      return (*MASL->Members.begin())->TargetTriple;
    }

    fail(UnsupportedSharedInput, SourceFile, unsupportedInputKindName(First));
  }();

  info(Verbose, Level, "Target triple: '{0}' (inferred from '{1}').", Inferred,
       SourceFile);

  return Inferred;
}

void LinkCLI::linkInput(EntityLinker &EL, ArtifactEncoding Encoding,
                        llvm::StringRef SourceFile, size_t Index,
                        unsigned Level, llvm::Timer &TLink) {
  auto failOnError = [&](llvm::Error Err) {
    if (Err) {
      fail(ErrorBuilder::wrap(std::move(Err))
               .context(LinkingArtifact, SourceFile)
               .build());
    }
  };

  if (auto *TU = std::get_if<TUSummaryEncoding>(&Encoding)) {
    info(Verbose, Level, "[{0}/{1}] Linking '{2}'.", Index + 1,
         InputFiles.size(), SourceFile);
    llvm::TimeRegion _(Time ? &TLink : nullptr);

    failOnError(EL.link(std::make_unique<TUSummaryEncoding>(std::move(*TU))));
    return;
  }

  if (auto *SL = std::get_if<StaticLibrary>(&Encoding)) {
    info(Verbose, Level,
         "[{0}/{1}] Linking '{2}' (static library, {3} member(s)).", Index + 1,
         InputFiles.size(), SourceFile, SL->Members.size());
    llvm::TimeRegion _(Time ? &TLink : nullptr);

    failOnError(EL.link(std::make_unique<StaticLibrary>(std::move(*SL))));
    return;
  }

  if (auto *MASL = std::get_if<MultiArchStaticLibrary>(&Encoding)) {
    info(Verbose, Level,
         "[{0}/{1}] Linking '{2}' (multi-arch static library, {3} member(s)).",
         Index + 1, InputFiles.size(), SourceFile, MASL->Members.size());
    llvm::TimeRegion _(Time ? &TLink : nullptr);

    failOnError(
        EL.link(std::make_unique<MultiArchStaticLibrary>(std::move(*MASL))));
    return;
  }

  fail(UnsupportedSharedInput, SourceFile, unsupportedInputKindName(Encoding));
}

void LinkCLI::write(const LUSummaryEncoding &Output, unsigned Level,
                    llvm::Timer &TWrite) {
  info(Verbose, Level, "Writing link unit summary to '{0}'.", OutputFile.Path);

  llvm::TimeRegion _(Time ? &TWrite : nullptr);

  if (auto Err =
          OutputFile.Format->writeLUSummaryEncoding(Output, OutputFile.Path)) {
    fail(std::move(Err));
  }
}

} // namespace clang::ssaf
