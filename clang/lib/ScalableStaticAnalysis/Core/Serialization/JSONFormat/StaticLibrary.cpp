//===- StaticLibrary.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/StaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "llvm/TargetParser/Triple.h"

namespace clang::ssaf {

//----------------------------------------------------------------------------
// StaticLibrary
//----------------------------------------------------------------------------

llvm::Expected<StaticLibrary>
JSONFormat::readStaticLibrary(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "StaticLibrary", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject,
                                "StaticLibrary", "object")
        .context(ErrorMessages::ReadingFromFile, "StaticLibrary", Path)
        .build();
  }

  if (auto Err = checkSummaryType(*RootObjectPtr, JSONTypeValueStaticLibrary)) {
    return ErrorBuilder::wrap(std::move(Err))
        .context(ErrorMessages::ReadingFromFile, "StaticLibrary", Path)
        .build();
  }

  auto ExpectedStaticLibrary = readStaticLibraryFromObject(*RootObjectPtr);
  if (!ExpectedStaticLibrary) {
    return ErrorBuilder::wrap(ExpectedStaticLibrary.takeError())
        .context(ErrorMessages::ReadingFromFile, "StaticLibrary", Path)
        .build();
  }

  return std::move(*ExpectedStaticLibrary);
}

llvm::Expected<StaticLibrary>
JSONFormat::readStaticLibraryFromObject(const Object &RootObject) {
  auto OptTargetTriple = RootObject.getString("target_triple");
  if (!OptTargetTriple) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "TargetTriple", "target_triple", "string")
        .build();
  }

  if (auto Err = validateNormalizedTargetTriple(*OptTargetTriple)) {
    return ErrorBuilder::wrap(std::move(Err))
        .context(ErrorMessages::ReadingFromField, "TargetTriple",
                 "target_triple")
        .build();
  }

  llvm::Triple T(*OptTargetTriple);

  const Object *NamespaceObject = RootObject.getObject("namespace");
  if (!NamespaceObject) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "BuildNamespace", "namespace", "object")
        .build();
  }

  auto ExpectedNamespace = buildNamespaceFromJSON(*NamespaceObject);
  if (!ExpectedNamespace) {
    return ErrorBuilder::wrap(ExpectedNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "BuildNamespace", "namespace")
        .build();
  }

  if (getKind(*ExpectedNamespace) != BuildNamespaceKind::StaticLibrary) {
    return ErrorBuilder::create(
               std::errc::invalid_argument,
               ErrorMessages::MismatchedSummaryType,
               buildNamespaceKindToJSON(BuildNamespaceKind::StaticLibrary),
               "namespace.kind",
               buildNamespaceKindToJSON(getKind(*ExpectedNamespace)))
        .build();
  }

  StaticLibrary S(std::move(T), std::move(*ExpectedNamespace));

  const Array *MembersArray = RootObject.getArray("members");
  if (!MembersArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "StaticLibrary members", "members", "array")
        .build();
  }

  auto &Members = getMembers(S);
  const auto &StaticLibraryTriple = getTargetTriple(S);

  for (const auto &[Index, MemberValue] : llvm::enumerate(*MembersArray)) {
    const Object *MemberObject = MemberValue.getAsObject();
    if (!MemberObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "StaticLibrary member", Index, "object")
          .build();
    }

    if (auto Err = checkSummaryType(*MemberObject, JSONTypeValueTUSummary)) {
      return ErrorBuilder::wrap(std::move(Err))
          .context(ErrorMessages::ReadingFromIndex, "StaticLibrary member",
                   Index)
          .build();
    }

    auto ExpectedMember = readTUSummaryEncodingFromObject(*MemberObject);
    if (!ExpectedMember) {
      return ErrorBuilder::wrap(ExpectedMember.takeError())
          .context(ErrorMessages::ReadingFromIndex, "StaticLibrary member",
                   Index)
          .build();
    }

    if (ExpectedMember->getTargetTriple() != StaticLibraryTriple) {
      return ErrorBuilder::create(
                 std::errc::invalid_argument,
                 ErrorMessages::MismatchedSummaryType,
                 llvm::Triple::normalize(StaticLibraryTriple.str()),
                 "target_triple",
                 llvm::Triple::normalize(
                     ExpectedMember->getTargetTriple().str()))
          .context(ErrorMessages::ReadingFromIndex, "StaticLibrary member",
                   Index)
          .build();
    }

    auto MemberNamespace = getTUNamespace(*ExpectedMember);
    auto Owned =
        std::make_unique<TUSummaryEncoding>(std::move(*ExpectedMember));
    auto [It, Inserted] = Members.insert(std::move(Owned));
    if (!Inserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "StaticLibrary member", Index,
                                  MemberNamespace)
          .build();
    }
  }

  return std::move(S);
}

llvm::Error JSONFormat::writeStaticLibrary(const StaticLibrary &S,
                                           llvm::StringRef Path) {
  if (auto Error = writeJSON(staticLibraryToJSON(S), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "StaticLibrary", Path)
        .build();
  }

  return llvm::Error::success();
}

Object JSONFormat::staticLibraryToJSON(const StaticLibrary &S) const {
  Object RootObject;

  RootObject[JSONTypeKey] = JSONTypeValueStaticLibrary;

  RootObject["target_triple"] =
      llvm::Triple::normalize(getTargetTriple(S).str());

  RootObject["namespace"] = buildNamespaceToJSON(getNamespace(S));

  Array MembersArray;
  MembersArray.reserve(getMembers(S).size());
  for (const auto &Member : getMembers(S)) {
    MembersArray.push_back(tuSummaryEncodingToJSON(*Member));
  }
  RootObject["members"] = std::move(MembersArray);

  return RootObject;
}

} // namespace clang::ssaf
