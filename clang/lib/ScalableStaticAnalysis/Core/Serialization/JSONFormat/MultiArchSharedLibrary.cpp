//===- MultiArchSharedLibrary.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/MultiArchSharedLibrary.h"
#include "llvm/TargetParser/Triple.h"

namespace clang::ssaf {

//----------------------------------------------------------------------------
// MultiArchSharedLibrary
//----------------------------------------------------------------------------

llvm::Expected<MultiArchSharedLibrary>
JSONFormat::readMultiArchSharedLibrary(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "MultiArchSharedLibrary", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject,
                                "MultiArchSharedLibrary", "object")
        .context(ErrorMessages::ReadingFromFile, "MultiArchSharedLibrary", Path)
        .build();
  }

  if (auto Err = checkSummaryType(*RootObjectPtr,
                                  JSONTypeValueMultiArchSharedLibrary)) {
    return ErrorBuilder::wrap(std::move(Err))
        .context(ErrorMessages::ReadingFromFile, "MultiArchSharedLibrary", Path)
        .build();
  }

  auto ExpectedM = readMultiArchSharedLibraryFromObject(*RootObjectPtr);
  if (!ExpectedM) {
    return ErrorBuilder::wrap(ExpectedM.takeError())
        .context(ErrorMessages::ReadingFromFile, "MultiArchSharedLibrary", Path)
        .build();
  }

  return std::move(*ExpectedM);
}

llvm::Expected<MultiArchSharedLibrary>
JSONFormat::readMultiArchSharedLibraryFromObject(const Object &RootObject) {
  const Array *NamespaceArray = RootObject.getArray("namespace");
  if (!NamespaceArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "NestedBuildNamespace", "namespace", "array")
        .build();
  }

  auto ExpectedNamespace = nestedBuildNamespaceFromJSON(*NamespaceArray);
  if (!ExpectedNamespace) {
    return ErrorBuilder::wrap(ExpectedNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "NestedBuildNamespace",
                 "namespace")
        .build();
  }

  const Array *MembersArray = RootObject.getArray("members");
  if (!MembersArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "MultiArchSharedLibrary members", "members",
                                "array")
        .build();
  }

  MultiArchSharedLibrary M(std::move(*ExpectedNamespace));
  auto &Members = getMembers(M);
  const auto &WrapperNamespace = getNamespace(M);

  for (const auto &[Index, MemberValue] : llvm::enumerate(*MembersArray)) {
    const Object *MemberObject = MemberValue.getAsObject();
    if (!MemberObject) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "MultiArchSharedLibrary member", Index,
                                  "object")
          .build();
    }

    if (auto Err = checkSummaryType(*MemberObject, JSONTypeValueLUSummary)) {
      return ErrorBuilder::wrap(std::move(Err))
          .context(ErrorMessages::ReadingFromIndex,
                   "MultiArchSharedLibrary member", Index)
          .build();
    }

    auto ExpectedMember = readLUSummaryEncodingFromObject(*MemberObject);
    if (!ExpectedMember) {
      return ErrorBuilder::wrap(ExpectedMember.takeError())
          .context(ErrorMessages::ReadingFromIndex,
                   "MultiArchSharedLibrary member", Index)
          .build();
    }

    // Every per-arch slice must resolve to the same shared-library identity,
    // so the member's LUNamespace must equal the wrapper's Namespace.
    if (WrapperNamespace != getLUNamespace(*ExpectedMember)) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::MismatchedNestedNamespace,
                                  "lu_namespace")
          .context(ErrorMessages::ReadingFromIndex,
                   "MultiArchSharedLibrary member", Index)
          .build();
    }

    auto [It, Inserted] = Members.insert(
        std::make_unique<LUSummaryEncoding>(std::move(*ExpectedMember)));
    if (!Inserted) {
      auto MemberTriple = llvm::Triple::normalize(getTargetTriple(**It).str());
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "MultiArchSharedLibrary member", Index,
                                  MemberTriple)
          .build();
    }
  }

  return std::move(M);
}

llvm::Error
JSONFormat::writeMultiArchSharedLibrary(const MultiArchSharedLibrary &M,
                                        llvm::StringRef Path) {
  Object RootObject;

  RootObject[JSONTypeKey] = JSONTypeValueMultiArchSharedLibrary;

  RootObject["namespace"] = nestedBuildNamespaceToJSON(getNamespace(M));

  Array MembersArray;
  MembersArray.reserve(getMembers(M).size());
  for (const auto &Member : getMembers(M)) {
    MembersArray.push_back(luSummaryEncodingToJSON(*Member));
  }
  RootObject["members"] = std::move(MembersArray);

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "MultiArchSharedLibrary", Path)
        .build();
  }

  return llvm::Error::success();
}

} // namespace clang::ssaf
