//===- MultiArchStaticLibrary.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchStaticLibrary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/StaticLibrary.h"
#include "llvm/TargetParser/Triple.h"

namespace clang::ssaf {

//----------------------------------------------------------------------------
// MultiArchStaticLibrary
//----------------------------------------------------------------------------

llvm::Expected<MultiArchStaticLibrary>
JSONFormat::readMultiArchStaticLibrary(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "MultiArchStaticLibrary", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject,
                                "MultiArchStaticLibrary", "object")
        .context(ErrorMessages::ReadingFromFile, "MultiArchStaticLibrary", Path)
        .build();
  }

  if (auto Err = checkSummaryType(*RootObjectPtr,
                                  JSONTypeValueMultiArchStaticLibrary)) {
    return ErrorBuilder::wrap(std::move(Err))
        .context(ErrorMessages::ReadingFromFile, "MultiArchStaticLibrary", Path)
        .build();
  }

  auto ExpectedM = readMultiArchStaticLibraryFromObject(*RootObjectPtr);
  if (!ExpectedM) {
    return ErrorBuilder::wrap(ExpectedM.takeError())
        .context(ErrorMessages::ReadingFromFile, "MultiArchStaticLibrary", Path)
        .build();
  }

  return std::move(*ExpectedM);
}

llvm::Expected<MultiArchStaticLibrary>
JSONFormat::readMultiArchStaticLibraryFromObject(const Object &RootObject) {
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

  if (getKind(*ExpectedNamespace) !=
      BuildNamespaceKind::MultiArchStaticLibrary) {
    return ErrorBuilder::create(
               std::errc::invalid_argument,
               ErrorMessages::MismatchedSummaryType,
               buildNamespaceKindToJSON(
                   BuildNamespaceKind::MultiArchStaticLibrary),
               "namespace.kind",
               buildNamespaceKindToJSON(getKind(*ExpectedNamespace)))
        .build();
  }

  const Array *MembersArray = RootObject.getArray("members");
  if (!MembersArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "MultiArchStaticLibrary members", "members",
                                "array")
        .build();
  }

  MultiArchStaticLibrary M(std::move(*ExpectedNamespace));
  auto &Members = getMembers(M);
  const auto &ExpectedName = getName(getNamespace(M));

  for (const auto &[Index, MemberValue] : llvm::enumerate(*MembersArray)) {
    const Object *MemberObject = MemberValue.getAsObject();
    if (!MemberObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "MultiArchStaticLibrary member", Index,
                                  "object")
          .build();
    }

    if (auto Err =
            checkSummaryType(*MemberObject, JSONTypeValueStaticLibrary)) {
      return ErrorBuilder::wrap(std::move(Err))
          .context(ErrorMessages::ReadingFromIndex,
                   "MultiArchStaticLibrary member", Index)
          .build();
    }

    auto ExpectedMember = readStaticLibraryFromObject(*MemberObject);
    if (!ExpectedMember) {
      return ErrorBuilder::wrap(ExpectedMember.takeError())
          .context(ErrorMessages::ReadingFromIndex,
                   "MultiArchStaticLibrary member", Index)
          .build();
    }

    const auto &MemberName = getName(getNamespace(*ExpectedMember));
    if (MemberName != ExpectedName) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::MismatchedSummaryType,
                                  ExpectedName, "namespace.name", MemberName)
          .context(ErrorMessages::ReadingFromIndex,
                   "MultiArchStaticLibrary member", Index)
          .build();
    }

    auto [It, Inserted] = Members.insert(
        std::make_unique<StaticLibrary>(std::move(*ExpectedMember)));
    if (!Inserted) {
      auto MemberTriple = llvm::Triple::normalize(getTargetTriple(**It).str());
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "MultiArchStaticLibrary member", Index,
                                  MemberTriple)
          .build();
    }
  }

  return std::move(M);
}

llvm::Error
JSONFormat::writeMultiArchStaticLibrary(const MultiArchStaticLibrary &M,
                                        llvm::StringRef Path) {
  Object RootObject;

  RootObject[JSONTypeKey] = JSONTypeValueMultiArchStaticLibrary;

  RootObject["namespace"] = buildNamespaceToJSON(getNamespace(M));

  Array MembersArray;
  MembersArray.reserve(getMembers(M).size());
  for (const auto &Member : getMembers(M)) {
    MembersArray.push_back(staticLibraryToJSON(*Member));
  }
  RootObject["members"] = std::move(MembersArray);

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "MultiArchStaticLibrary", Path)
        .build();
  }

  return llvm::Error::success();
}

} // namespace clang::ssaf
