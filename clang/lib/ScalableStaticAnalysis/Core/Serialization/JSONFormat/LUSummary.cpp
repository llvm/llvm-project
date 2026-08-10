//===- LUSummary.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "llvm/TargetParser/Triple.h"

#include <set>

namespace clang::ssaf {

//----------------------------------------------------------------------------
// LUSummary
//----------------------------------------------------------------------------

llvm::Expected<LUSummary> JSONFormat::readLUSummary(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject, "LUSummary",
                                "object")
        .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
        .build();
  }

  if (auto Err = checkSummaryType(*RootObjectPtr, JSONTypeValueLUSummary)) {
    return ErrorBuilder::wrap(std::move(Err))
        .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
        .build();
  }

  auto ExpectedSummary = readLUSummaryFromObject(*RootObjectPtr);
  if (!ExpectedSummary) {
    return ErrorBuilder::wrap(ExpectedSummary.takeError())
        .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
        .build();
  }

  return std::move(*ExpectedSummary);
}

llvm::Expected<LUSummary>
JSONFormat::readLUSummaryFromObject(const Object &RootObject) {
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

  const Array *LUNamespaceArray = RootObject.getArray("lu_namespace");
  if (!LUNamespaceArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "NestedBuildNamespace", "lu_namespace", "array")
        .build();
  }

  auto ExpectedLUNamespace = nestedBuildNamespaceFromJSON(*LUNamespaceArray);
  if (!ExpectedLUNamespace) {
    return ErrorBuilder::wrap(ExpectedLUNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "NestedBuildNamespace",
                 "lu_namespace")
        .build();
  }

  LUSummary Summary(std::move(T), std::move(*ExpectedLUNamespace));

  {
    const Array *IdTableArray = RootObject.getArray("id_table");
    if (!IdTableArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "IdTable", "id_table", "array")
          .build();
    }

    auto ExpectedIdTable = luEntityIdTableFromJSON(*IdTableArray);
    if (!ExpectedIdTable) {
      return ErrorBuilder::wrap(ExpectedIdTable.takeError())
          .context(ErrorMessages::ReadingFromField, "IdTable", "id_table")
          .build();
    }

    getIdTable(Summary) = std::move(*ExpectedIdTable);
  }

  {
    const Array *LinkageTableArray = RootObject.getArray("linkage_table");
    if (!LinkageTableArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "LinkageTable", "linkage_table", "array")
          .build();
    }

    auto ExpectedIdRange =
        llvm::make_second_range(getEntities(getIdTable(Summary)));
    std::set<EntityId> ExpectedIds(ExpectedIdRange.begin(),
                                   ExpectedIdRange.end());

    // Move ExpectedIds in since linkageTableFromJSON consumes it to verify
    // that the linkage table contains exactly the ids present in the IdTable.
    auto ExpectedLinkageTable =
        linkageTableFromJSON(*LinkageTableArray, std::move(ExpectedIds));
    if (!ExpectedLinkageTable) {
      return ErrorBuilder::wrap(ExpectedLinkageTable.takeError())
          .context(ErrorMessages::ReadingFromField, "LinkageTable",
                   "linkage_table")
          .build();
    }

    getLinkageTable(Summary) = std::move(*ExpectedLinkageTable);
  }

  {
    const Array *SummaryDataArray = RootObject.getArray("data");
    if (!SummaryDataArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "SummaryData entries", "data", "array")
          .build();
    }

    auto ExpectedSummaryDataMap =
        summaryDataMapFromJSON(*SummaryDataArray, getIdTable(Summary));
    if (!ExpectedSummaryDataMap) {
      return ErrorBuilder::wrap(ExpectedSummaryDataMap.takeError())
          .context(ErrorMessages::ReadingFromField, "SummaryData entries",
                   "data")
          .build();
    }

    getData(Summary) = std::move(*ExpectedSummaryDataMap);
  }

  return std::move(Summary);
}

llvm::Error JSONFormat::writeLUSummary(const LUSummary &S,
                                       llvm::StringRef Path) {
  Object RootObject;

  RootObject[JSONTypeKey] = JSONTypeValueLUSummary;

  RootObject["target_triple"] =
      llvm::Triple::normalize(getTargetTriple(S).str());

  RootObject["lu_namespace"] = nestedBuildNamespaceToJSON(getLUNamespace(S));

  RootObject["id_table"] = luEntityIdTableToJSON(getIdTable(S));

  RootObject["linkage_table"] = linkageTableToJSON(getLinkageTable(S));

  auto ExpectedDataObject = summaryDataMapToJSON(getData(S));
  if (!ExpectedDataObject) {
    return ErrorBuilder::wrap(ExpectedDataObject.takeError())
        .context(ErrorMessages::WritingToFile, "LUSummary", Path)
        .build();
  }

  RootObject["data"] = std::move(*ExpectedDataObject);

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "LUSummary", Path)
        .build();
  }

  return llvm::Error::success();
}

} // namespace clang::ssaf
