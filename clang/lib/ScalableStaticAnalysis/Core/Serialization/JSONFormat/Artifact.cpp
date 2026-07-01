//===- Artifact.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic artifact read/write entry points that dispatch on the
// self-describing "type" field of the JSON document.
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

namespace clang::ssaf {

llvm::Expected<Artifact> JSONFormat::readArtifact(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "Artifact", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject, "Artifact",
                                "object")
        .context(ErrorMessages::ReadingFromFile, "Artifact", Path)
        .build();
  }

  auto ExpectedType = readSummaryType(*RootObjectPtr);
  if (!ExpectedType) {
    return ErrorBuilder::wrap(ExpectedType.takeError())
        .context(ErrorMessages::ReadingFromFile, "Artifact", Path)
        .build();
  }

  // Dispatch by the self-describing type field. The helpers operate on
  // the already-parsed root object so the file is read and parsed only
  // once per readArtifact call.
  if (*ExpectedType == JSONTypeValueTUSummary) {
    auto ExpectedTU = readTUSummaryFromObject(*RootObjectPtr);
    if (!ExpectedTU) {
      return ErrorBuilder::wrap(ExpectedTU.takeError())
          .context(ErrorMessages::ReadingFromFile, "Artifact", Path)
          .build();
    }
    return Artifact{std::move(*ExpectedTU)};
  }

  if (*ExpectedType == JSONTypeValueLUSummary) {
    auto ExpectedLU = readLUSummaryFromObject(*RootObjectPtr);
    if (!ExpectedLU) {
      return ErrorBuilder::wrap(ExpectedLU.takeError())
          .context(ErrorMessages::ReadingFromFile, "Artifact", Path)
          .build();
    }
    return Artifact{std::move(*ExpectedLU)};
  }

  if (*ExpectedType == JSONTypeValueWPASuite) {
    auto ExpectedWPA = readWPASuiteFromObject(*RootObjectPtr);
    if (!ExpectedWPA) {
      return ErrorBuilder::wrap(ExpectedWPA.takeError())
          .context(ErrorMessages::ReadingFromFile, "Artifact", Path)
          .build();
    }
    return Artifact{std::move(*ExpectedWPA)};
  }

  return ErrorBuilder::create(std::errc::invalid_argument,
                              ErrorMessages::UnknownArtifactType, *ExpectedType,
                              JSONTypeKey, JSONTypeValueTUSummary,
                              JSONTypeValueLUSummary, JSONTypeValueWPASuite)
      .context(ErrorMessages::ReadingFromFile, "Artifact", Path)
      .build();
}

llvm::Error JSONFormat::writeArtifact(const Artifact &A, llvm::StringRef Path) {
  return std::visit(
      [&](const auto &S) -> llvm::Error {
        using T = std::decay_t<decltype(S)>;
        if constexpr (std::is_same_v<T, TUSummary>) {
          return writeTUSummary(S, Path);
        } else if constexpr (std::is_same_v<T, LUSummary>) {
          return writeLUSummary(S, Path);
        } else {
          static_assert(std::is_same_v<T, WPASuite>,
                        "Artifact visitor must cover all variant alternatives");
          return writeWPASuite(S, Path);
        }
      },
      A);
}

llvm::Expected<ArtifactEncoding>
JSONFormat::readArtifactEncoding(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject,
                                "ArtifactEncoding", "object")
        .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
        .build();
  }

  auto ExpectedType = readSummaryType(*RootObjectPtr);
  if (!ExpectedType) {
    return ErrorBuilder::wrap(ExpectedType.takeError())
        .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
        .build();
  }

  // Dispatch by the self-describing type field. The helpers operate on
  // the already-parsed root object so the file is read and parsed only
  // once per readArtifactEncoding call.
  if (*ExpectedType == JSONTypeValueTUSummary) {
    auto ExpectedTU = readTUSummaryEncodingFromObject(*RootObjectPtr);
    if (!ExpectedTU) {
      return ErrorBuilder::wrap(ExpectedTU.takeError())
          .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
          .build();
    }
    return ArtifactEncoding{std::move(*ExpectedTU)};
  }

  if (*ExpectedType == JSONTypeValueLUSummary) {
    auto ExpectedLU = readLUSummaryEncodingFromObject(*RootObjectPtr);
    if (!ExpectedLU) {
      return ErrorBuilder::wrap(ExpectedLU.takeError())
          .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
          .build();
    }
    return ArtifactEncoding{std::move(*ExpectedLU)};
  }

  if (*ExpectedType == JSONTypeValueStaticLibrary) {
    auto ExpectedStaticLibrary = readStaticLibraryFromObject(*RootObjectPtr);
    if (!ExpectedStaticLibrary) {
      return ErrorBuilder::wrap(ExpectedStaticLibrary.takeError())
          .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
          .build();
    }
    return ArtifactEncoding{std::move(*ExpectedStaticLibrary)};
  }

  if (*ExpectedType == JSONTypeValueMultiArchStaticLibrary) {
    auto ExpectedM = readMultiArchStaticLibraryFromObject(*RootObjectPtr);
    if (!ExpectedM) {
      return ErrorBuilder::wrap(ExpectedM.takeError())
          .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
          .build();
    }
    return ArtifactEncoding{std::move(*ExpectedM)};
  }

  if (*ExpectedType == JSONTypeValueMultiArchSharedLibrary) {
    auto ExpectedM = readMultiArchSharedLibraryFromObject(*RootObjectPtr);
    if (!ExpectedM) {
      return ErrorBuilder::wrap(ExpectedM.takeError())
          .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
          .build();
    }
    return ArtifactEncoding{std::move(*ExpectedM)};
  }

  return ErrorBuilder::create(
             std::errc::invalid_argument,
             ErrorMessages::UnknownArtifactEncodingType, *ExpectedType,
             JSONTypeKey, JSONTypeValueTUSummary, JSONTypeValueLUSummary,
             JSONTypeValueStaticLibrary, JSONTypeValueMultiArchStaticLibrary,
             JSONTypeValueMultiArchSharedLibrary)
      .context(ErrorMessages::ReadingFromFile, "ArtifactEncoding", Path)
      .build();
}

llvm::Error JSONFormat::writeArtifactEncoding(const ArtifactEncoding &E,
                                              llvm::StringRef Path) {
  return std::visit(
      [&](const auto &Enc) -> llvm::Error {
        using T = std::decay_t<decltype(Enc)>;
        if constexpr (std::is_same_v<T, TUSummaryEncoding>) {
          return writeTUSummaryEncoding(Enc, Path);
        } else if constexpr (std::is_same_v<T, LUSummaryEncoding>) {
          return writeLUSummaryEncoding(Enc, Path);
        } else if constexpr (std::is_same_v<T, StaticLibrary>) {
          return writeStaticLibrary(Enc, Path);
        } else if constexpr (std::is_same_v<T, MultiArchStaticLibrary>) {
          return writeMultiArchStaticLibrary(Enc, Path);
        } else {
          static_assert(
              std::is_same_v<T, MultiArchSharedLibrary>,
              "ArtifactEncoding visitor must cover all variant alternatives");
          return writeMultiArchSharedLibrary(Enc, Path);
        }
      },
      E);
}

} // namespace clang::ssaf
