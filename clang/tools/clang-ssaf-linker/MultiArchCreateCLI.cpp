//===- MultiArchCreateCLI.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements the `multi-arch create` CLI action. The run() function picks the
//  family from the first input and hands off to either createStaticLibrary() or
//  createSharedLibrary().
//
//===----------------------------------------------------------------------===//

#include "MultiArchCreateCLI.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/Support/FormatProviders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Timer.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <variant>

using namespace llvm;
using namespace clang::ssaf;

namespace {

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

constexpr const char *ReadingArtifact = "Reading artifact '{0}'";

constexpr const char *NoInputs =
    "no input artifacts: at least one input is required";

constexpr const char *InvalidInputKind =
    "'{0}' is a raw TU summary, not a valid input to multi-arch create: run "
    "static-library create or an entity-linking step first";

constexpr const char *MixedFamily =
    "input '{0}' is a {1} artifact, but a preceding input established this "
    "bundle as {2}";

constexpr const char *NamespaceMismatch =
    "namespace {0} from '{1}' does not match expected namespace {2}";

constexpr const char *NoCandidateMembers =
    "no candidate members could be derived from the given inputs: at least "
    "one member is required";

constexpr const char *DuplicateTriple =
    "duplicate architecture slice '{0}' contributed by both '{1}' and '{2}'";

constexpr const char *StaticFamilyName = "static-library";
constexpr const char *SharedFamilyName = "shared-library";

//===----------------------------------------------------------------------===//
// ArtifactEncoding Helpers
//===----------------------------------------------------------------------===//

bool isStaticFamily(const ArtifactEncoding &E) {
  return std::holds_alternative<StaticLibrary>(E) ||
         std::holds_alternative<MultiArchStaticLibrary>(E);
}

bool isSharedFamily(const ArtifactEncoding &E) {
  return std::holds_alternative<LUSummaryEncoding>(E) ||
         std::holds_alternative<MultiArchSharedLibrary>(E);
}

bool isTUSummaryEncoding(const ArtifactEncoding &E) {
  return std::holds_alternative<TUSummaryEncoding>(E);
}

} // namespace

namespace clang::ssaf {

void MultiArchCreateCLI::run(llvm::TimerGroup &TG,
                             llvm::ArrayRef<std::string> InputPaths,
                             llvm::StringRef OutputPath, bool Verbose,
                             bool Time) {
  this->InputPaths = InputPaths;
  this->OutputPath = OutputPath;
  this->Verbose = Verbose;
  this->Time = Time;

  llvm::Timer TValidate("validate", "Validate Input", TG);
  llvm::Timer TRead("read", "Read Artifacts", TG);
  llvm::Timer TBundle("bundle", "Bundle Input", TG);
  llvm::Timer TWrite("write", "Write Multi-Arch Bundle", TG);

  // Nesting depth for indenting verbose notes.
  const unsigned Level = 0;

  info(Verbose, Level, "Bundling started.");

  validate(Level + 1, TValidate);

  ArtifactEncoding Result = create(Level + 1, TRead, TBundle);

  write(Result, Level + 1, TWrite);

  info(Verbose, Level, "Bundling finished.");

  // Second run() should start from a clean slate.
  InputFiles.clear();
  SourceByMember.clear();
}

void MultiArchCreateCLI::validate(unsigned Level, llvm::Timer &TValidate) {
  info(Verbose, Level, "Validating input.");

  llvm::TimeRegion _(Time ? &TValidate : nullptr);

  OutputFile = FormatFile::fromOutputPath(OutputPath);
  info(Verbose, Level + 1, "Validated output path '{0}'.", OutputFile.Path);

  if (InputPaths.empty()) {
    fail(NoInputs);
  }

  for (const auto &InputPath : InputPaths) {
    InputFiles.push_back(FormatFile::fromInputPath(InputPath));
  }

  info(Verbose, Level + 1, "Validated {0} input artifact path(s).",
       InputFiles.size());
}

ArtifactEncoding MultiArchCreateCLI::create(unsigned Level, llvm::Timer &TRead,
                                            llvm::Timer &TBundle) {
  info(Verbose, Level, "Creating bundle.");

  const unsigned MemberLevel = Level + 1;
  info(Verbose, MemberLevel, "Bundling members.");

  size_t Index = 0;
  ArtifactEncoding First = readInput(Index, MemberLevel + 1, TRead);

  if (isStaticFamily(First)) {
    return createStaticLibrary(std::move(First), MemberLevel, TRead, TBundle);
  }

  if (isSharedFamily(First)) {
    return createSharedLibrary(std::move(First), MemberLevel, TRead, TBundle);
  }

  fail(InvalidInputKind, InputFiles[Index].Path);
}

ArtifactEncoding MultiArchCreateCLI::readInput(size_t Index, unsigned Level,
                                               llvm::Timer &TRead) {
  const FormatFile &InputFile = InputFiles[Index];
  info(Verbose, Level, "[{0}/{1}] Reading '{2}'.", Index + 1, InputFiles.size(),
       InputFile.Path);

  llvm::TimeRegion _(Time ? &TRead : nullptr);
  auto ExpectedEncoding =
      InputFile.Format->readArtifactEncoding(InputFile.Path);
  if (!ExpectedEncoding) {
    fail(ErrorBuilder::wrap(ExpectedEncoding.takeError())
             .context(ReadingArtifact, InputFile.Path)
             .build());
  }
  return std::move(*ExpectedEncoding);
}

const BuildNamespace &
MultiArchCreateCLI::staticFamilyNamespace(const ArtifactEncoding &E) {
  assert(isStaticFamily(E) && "not a static-family artifact");
  if (const auto *SL = std::get_if<StaticLibrary>(&E)) {
    return SL->Namespace;
  }
  return std::get<MultiArchStaticLibrary>(E).Namespace;
}

const NestedBuildNamespace &
MultiArchCreateCLI::sharedFamilyNamespace(const ArtifactEncoding &E) {
  assert(isSharedFamily(E) && "not a shared-family artifact");
  if (const auto *LU = std::get_if<LUSummaryEncoding>(&E)) {
    return LU->LUNamespace;
  }
  return std::get<MultiArchSharedLibrary>(E).Namespace;
}

ArtifactEncoding MultiArchCreateCLI::createStaticLibrary(ArtifactEncoding First,
                                                         unsigned Level,
                                                         llvm::Timer &TRead,
                                                         llvm::Timer &TBundle) {
  MultiArchStaticLibrary Bundle(staticFamilyNamespace(First).withKind(
      BuildNamespaceKind::MultiArchStaticLibrary));

  size_t Index = 0;

  addStaticInput(Bundle, std::move(First), Index, Level + 1, TBundle);
  for (Index = 1; Index < InputFiles.size(); ++Index) {
    addStaticInput(Bundle, readInput(Index, Level + 1, TRead), Index, Level + 1,
                   TBundle);
  }

  if (Bundle.Members.empty()) {
    fail(NoCandidateMembers);
  }

  info(Verbose, Level, "Bundled {0} member(s).", Bundle.Members.size());
  info(Verbose, Level, "Target namespace: '{0}'.", Bundle.Namespace);

  return ArtifactEncoding(std::move(Bundle));
}

ArtifactEncoding MultiArchCreateCLI::createSharedLibrary(ArtifactEncoding First,
                                                         unsigned Level,
                                                         llvm::Timer &TRead,
                                                         llvm::Timer &TBundle) {
  MultiArchSharedLibrary Bundle(sharedFamilyNamespace(First));

  size_t Index = 0;

  addSharedInput(Bundle, std::move(First), Index, Level + 1, TBundle);
  for (Index = 1; Index < InputFiles.size(); ++Index) {
    addSharedInput(Bundle, readInput(Index, Level + 1, TRead), Index, Level + 1,
                   TBundle);
  }

  if (Bundle.Members.empty()) {
    fail(NoCandidateMembers);
  }

  info(Verbose, Level, "Bundled {0} member(s).", Bundle.Members.size());
  info(Verbose, Level, "Target namespace: '{0}'.", Bundle.Namespace);

  return ArtifactEncoding(std::move(Bundle));
}

void MultiArchCreateCLI::addStaticInput(MultiArchStaticLibrary &Bundle,
                                        ArtifactEncoding Encoding, size_t Index,
                                        unsigned Level, llvm::Timer &TBundle) {
  llvm::StringRef SourceFile = InputFiles[Index].Path;
  info(Verbose, Level, "[{0}/{1}] Bundling '{2}'.", Index + 1,
       InputFiles.size(), SourceFile);
  llvm::TimeRegion _(Time ? &TBundle : nullptr);

  if (auto *SL = std::get_if<StaticLibrary>(&Encoding)) {
    BuildNamespace Expected =
        Bundle.Namespace.withKind(BuildNamespaceKind::StaticLibrary);
    if (SL->Namespace != Expected) {
      fail(NamespaceMismatch, SL->Namespace, SourceFile, Expected);
    }
    addStaticMember(Bundle, std::make_unique<StaticLibrary>(std::move(*SL)),
                    SourceFile);
    return;
  }

  if (auto *MASL = std::get_if<MultiArchStaticLibrary>(&Encoding)) {
    if (MASL->Namespace != Bundle.Namespace) {
      fail(NamespaceMismatch, MASL->Namespace, SourceFile, Bundle.Namespace);
    }

    while (!MASL->Members.empty()) {
      auto Node = MASL->Members.extract(MASL->Members.begin());
      addStaticMember(Bundle, std::move(Node.value()), SourceFile);
    }
    return;
  }

  if (isTUSummaryEncoding(Encoding)) {
    fail(InvalidInputKind, SourceFile);
  }

  fail(MixedFamily, SourceFile, SharedFamilyName, StaticFamilyName);
}

void MultiArchCreateCLI::addSharedInput(MultiArchSharedLibrary &Bundle,
                                        ArtifactEncoding Encoding, size_t Index,
                                        unsigned Level, llvm::Timer &TBundle) {
  llvm::StringRef SourceFile = InputFiles[Index].Path;
  info(Verbose, Level, "[{0}/{1}] Bundling '{2}'.", Index + 1,
       InputFiles.size(), SourceFile);
  llvm::TimeRegion _(Time ? &TBundle : nullptr);

  if (auto *LU = std::get_if<LUSummaryEncoding>(&Encoding)) {
    if (LU->LUNamespace != Bundle.Namespace) {
      fail(NamespaceMismatch, LU->LUNamespace, SourceFile, Bundle.Namespace);
    }
    addSharedMember(Bundle, std::make_unique<LUSummaryEncoding>(std::move(*LU)),
                    SourceFile);
    return;
  }

  if (auto *MASharedL = std::get_if<MultiArchSharedLibrary>(&Encoding)) {
    if (MASharedL->Namespace != Bundle.Namespace) {
      fail(NamespaceMismatch, MASharedL->Namespace, SourceFile,
           Bundle.Namespace);
    }
    while (!MASharedL->Members.empty()) {
      auto Node = MASharedL->Members.extract(MASharedL->Members.begin());
      addSharedMember(Bundle, std::move(Node.value()), SourceFile);
    }
    return;
  }

  if (isTUSummaryEncoding(Encoding)) {
    fail(InvalidInputKind, SourceFile);
  }

  fail(MixedFamily, SourceFile, StaticFamilyName, SharedFamilyName);
}

void MultiArchCreateCLI::addStaticMember(MultiArchStaticLibrary &Bundle,
                                         std::unique_ptr<StaticLibrary> Member,
                                         llvm::StringRef SourceFile) {
  auto [It, Inserted] = Bundle.Members.insert(std::move(Member));
  if (!Inserted) {
    fail(DuplicateTriple, llvm::Triple::normalize((*It)->TargetTriple.str()),
         SourceByMember.lookup(It->get()), SourceFile);
  }
  SourceByMember[It->get()] = SourceFile;
}

void MultiArchCreateCLI::addSharedMember(
    MultiArchSharedLibrary &Bundle, std::unique_ptr<LUSummaryEncoding> Member,
    llvm::StringRef SourceFile) {
  auto [It, Inserted] = Bundle.Members.insert(std::move(Member));
  if (!Inserted) {
    fail(DuplicateTriple, llvm::Triple::normalize((*It)->TargetTriple.str()),
         SourceByMember.lookup(It->get()), SourceFile);
  }
  SourceByMember[It->get()] = SourceFile;
}

void MultiArchCreateCLI::write(const ArtifactEncoding &Bundle, unsigned Level,
                               llvm::Timer &TWrite) {
  info(Verbose, Level, "Writing bundle to '{0}'.", OutputFile.Path);

  llvm::TimeRegion _(Time ? &TWrite : nullptr);

  if (auto Err =
          OutputFile.Format->writeArtifactEncoding(Bundle, OutputFile.Path)) {
    fail(std::move(Err));
  }
}

} // namespace clang::ssaf
