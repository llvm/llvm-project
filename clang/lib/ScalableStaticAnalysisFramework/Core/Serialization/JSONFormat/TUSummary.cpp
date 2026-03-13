//===- TUSummary.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysisFramework/Core/TUSummary/TUSummary.h"

#include <set>

namespace clang::ssaf {

//----------------------------------------------------------------------------
// TUSummary
//----------------------------------------------------------------------------

llvm::Expected<TUSummary> JSONFormat::readTUSummary(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject, "TUSummary",
                                "object")
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  const Object &RootObject = *RootObjectPtr;

  const Object *TUNamespaceObject = RootObject.getObject("tu_namespace");
  if (!TUNamespaceObject) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "BuildNamespace", "tu_namespace", "object")
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  auto ExpectedTUNamespace = buildNamespaceFromJSON(*TUNamespaceObject);
  if (!ExpectedTUNamespace) {
    return ErrorBuilder::wrap(ExpectedTUNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "BuildNamespace",
                 "tu_namespace")
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  TUSummary Summary(std::move(*ExpectedTUNamespace));

  {
    const Array *IdTableArray = RootObject.getArray("id_table");
    if (!IdTableArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "IdTable", "id_table", "array")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    auto ExpectedIdTable = entityIdTableFromJSON(*IdTableArray);
    if (!ExpectedIdTable) {
      return ErrorBuilder::wrap(ExpectedIdTable.takeError())
          .context(ErrorMessages::ReadingFromField, "IdTable", "id_table")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
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
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
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
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
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
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    auto ExpectedSummaryDataMap =
        summaryDataMapFromJSON(*SummaryDataArray, getIdTable(Summary));
    if (!ExpectedSummaryDataMap) {
      return ErrorBuilder::wrap(ExpectedSummaryDataMap.takeError())
          .context(ErrorMessages::ReadingFromField, "SummaryData entries",
                   "data")
          .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
          .build();
    }

    getData(Summary) = std::move(*ExpectedSummaryDataMap);
  }

  return std::move(Summary);
}

llvm::Error JSONFormat::writeTUSummary(const TUSummary &S,
                                       llvm::StringRef Path) {
  Object RootObject;

  RootObject["tu_namespace"] = buildNamespaceToJSON(getTUNamespace(S));

  RootObject["id_table"] = entityIdTableToJSON(getIdTable(S));

  RootObject["linkage_table"] = linkageTableToJSON(getLinkageTable(S));

  auto ExpectedDataObject = summaryDataMapToJSON(getData(S));
  if (!ExpectedDataObject) {
    return ErrorBuilder::wrap(ExpectedDataObject.takeError())
        .context(ErrorMessages::WritingToFile, "TUSummary", Path)
        .build();
  }

  RootObject["data"] = std::move(*ExpectedDataObject);

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "TUSummary", Path)
        .build();
  }

  return llvm::Error::success();
}

} // namespace clang::ssaf
