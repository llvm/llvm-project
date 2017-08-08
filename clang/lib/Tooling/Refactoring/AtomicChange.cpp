//===--- AtomicChange.cpp - AtomicChange implementation -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/ReplacementsYaml.h"
#include "llvm/Support/YAMLTraits.h"
#include <string>

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::tooling::AtomicChange)

namespace {
/// \brief Helper to (de)serialize an AtomicChange since we don't have direct
/// access to its data members.
/// Data members of a normalized AtomicChange can be directly mapped from/to
/// YAML string.
struct NormalizedAtomicChange {
  NormalizedAtomicChange() = default;

  NormalizedAtomicChange(const llvm::yaml::IO &) {}

  // This converts AtomicChange's internal implementation of the replacements
  // set to a vector of replacements.
  NormalizedAtomicChange(const llvm::yaml::IO &,
                         const clang::tooling::AtomicChange &E)
      : Key(E.getKey()), FilePath(E.getFilePath()), Error(E.getError()),
        InsertedHeaders(E.getInsertedHeaders()),
        RemovedHeaders(E.getRemovedHeaders()),
        Replaces(E.getReplacements().begin(), E.getReplacements().end()) {}

  // This is not expected to be called but needed for template instantiation.
  clang::tooling::AtomicChange denormalize(const llvm::yaml::IO &) {
    llvm_unreachable("Do not convert YAML to AtomicChange directly with '>>'. "
                     "Use AtomicChange::convertFromYAML instead.");
  }
  std::string Key;
  std::string FilePath;
  std::string Error;
  std::vector<std::string> InsertedHeaders;
  std::vector<std::string> RemovedHeaders;
  std::vector<clang::tooling::Replacement> Replaces;
};
} // anonymous namespace

namespace llvm {
namespace yaml {

/// \brief Specialized MappingTraits to describe how an AtomicChange is
/// (de)serialized.
template <> struct MappingTraits<NormalizedAtomicChange> {
  static void mapping(IO &Io, NormalizedAtomicChange &Doc) {
    Io.mapRequired("Key", Doc.Key);
    Io.mapRequired("FilePath", Doc.FilePath);
    Io.mapRequired("Error", Doc.Error);
    Io.mapRequired("InsertedHeaders", Doc.InsertedHeaders);
    Io.mapRequired("RemovedHeaders", Doc.RemovedHeaders);
    Io.mapRequired("Replacements", Doc.Replaces);
  }
};

/// \brief Specialized MappingTraits to describe how an AtomicChange is
/// (de)serialized.
template <> struct MappingTraits<clang::tooling::AtomicChange> {
  static void mapping(IO &Io, clang::tooling::AtomicChange &Doc) {
    MappingNormalization<NormalizedAtomicChange, clang::tooling::AtomicChange>
        Keys(Io, Doc);
    Io.mapRequired("Key", Keys->Key);
    Io.mapRequired("FilePath", Keys->FilePath);
    Io.mapRequired("Error", Keys->Error);
    Io.mapRequired("InsertedHeaders", Keys->InsertedHeaders);
    Io.mapRequired("RemovedHeaders", Keys->RemovedHeaders);
    Io.mapRequired("Replacements", Keys->Replaces);
  }
};

} // end namespace yaml
} // end namespace llvm

namespace clang {
namespace tooling {

AtomicChange::AtomicChange(const SourceManager &SM,
                           SourceLocation KeyPosition) {
  const FullSourceLoc FullKeyPosition(KeyPosition, SM);
  std::pair<FileID, unsigned> FileIDAndOffset =
      FullKeyPosition.getSpellingLoc().getDecomposedLoc();
  const FileEntry *FE = SM.getFileEntryForID(FileIDAndOffset.first);
  assert(FE && "Cannot create AtomicChange with invalid location.");
  FilePath = FE->getName();
  Key = FilePath + ":" + std::to_string(FileIDAndOffset.second);
}

AtomicChange::AtomicChange(std::string Key, std::string FilePath,
                           std::string Error,
                           std::vector<std::string> InsertedHeaders,
                           std::vector<std::string> RemovedHeaders,
                           clang::tooling::Replacements Replaces)
    : Key(std::move(Key)), FilePath(std::move(FilePath)),
      Error(std::move(Error)), InsertedHeaders(std::move(InsertedHeaders)),
      RemovedHeaders(std::move(RemovedHeaders)), Replaces(std::move(Replaces)) {
}

std::string AtomicChange::toYAMLString() {
  std::string YamlContent;
  llvm::raw_string_ostream YamlContentStream(YamlContent);

  llvm::yaml::Output YAML(YamlContentStream);
  YAML << *this;
  YamlContentStream.flush();
  return YamlContent;
}

AtomicChange AtomicChange::convertFromYAML(llvm::StringRef YAMLContent) {
  NormalizedAtomicChange NE;
  llvm::yaml::Input YAML(YAMLContent);
  YAML >> NE;
  AtomicChange E(NE.Key, NE.FilePath, NE.Error, NE.InsertedHeaders,
                 NE.RemovedHeaders, tooling::Replacements());
  for (const auto &R : NE.Replaces) {
    llvm::Error Err = E.Replaces.add(R);
    if (Err)
      llvm_unreachable(
          "Failed to add replacement when Converting YAML to AtomicChange.");
    llvm::consumeError(std::move(Err));
  }
  return E;
}

llvm::Error AtomicChange::replace(const SourceManager &SM,
                                  const CharSourceRange &Range,
                                  llvm::StringRef ReplacementText) {
  return Replaces.add(Replacement(SM, Range, ReplacementText));
}

llvm::Error AtomicChange::replace(const SourceManager &SM, SourceLocation Loc,
                                  unsigned Length, llvm::StringRef Text) {
  return Replaces.add(Replacement(SM, Loc, Length, Text));
}

llvm::Error AtomicChange::insert(const SourceManager &SM, SourceLocation Loc,
                                 llvm::StringRef Text, bool InsertAfter) {
  if (Text.empty())
    return llvm::Error::success();
  Replacement R(SM, Loc, 0, Text);
  llvm::Error Err = Replaces.add(R);
  if (Err) {
    return llvm::handleErrors(
        std::move(Err), [&](const ReplacementError &RE) -> llvm::Error {
          if (RE.get() != replacement_error::insert_conflict)
            return llvm::make_error<ReplacementError>(RE);
          unsigned NewOffset = Replaces.getShiftedCodePosition(R.getOffset());
          if (!InsertAfter)
            NewOffset -=
                RE.getExistingReplacement()->getReplacementText().size();
          Replacement NewR(R.getFilePath(), NewOffset, 0, Text);
          Replaces = Replaces.merge(Replacements(NewR));
          return llvm::Error::success();
        });
  }
  return llvm::Error::success();
}

void AtomicChange::addHeader(llvm::StringRef Header) {
  InsertedHeaders.push_back(Header);
}

void AtomicChange::removeHeader(llvm::StringRef Header) {
  RemovedHeaders.push_back(Header);
}

} // end namespace tooling
} // end namespace clang
