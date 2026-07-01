//===- TUSummary.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "llvm/TargetParser/Triple.h"

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

  if (auto Err = checkSummaryType(*RootObjectPtr, JSONTypeValueTUSummary)) {
    return ErrorBuilder::wrap(std::move(Err))
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  auto ExpectedSummary = readTUSummaryFromObject(*RootObjectPtr);
  if (!ExpectedSummary) {
    return ErrorBuilder::wrap(ExpectedSummary.takeError())
        .context(ErrorMessages::ReadingFromFile, "TUSummary", Path)
        .build();
  }

  return std::move(*ExpectedSummary);
}

llvm::Expected<TUSummary>
JSONFormat::readTUSummaryFromObject(const Object &RootObject) {
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

  const Object *TUNamespaceObject = RootObject.getObject("tu_namespace");
  if (!TUNamespaceObject) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "BuildNamespace", "tu_namespace", "object")
        .build();
  }

  auto ExpectedTUNamespace = buildNamespaceFromJSON(*TUNamespaceObject);
  if (!ExpectedTUNamespace) {
    return ErrorBuilder::wrap(ExpectedTUNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "BuildNamespace",
                 "tu_namespace")
        .build();
  }

  TUSummary Summary(std::move(T), std::move(*ExpectedTUNamespace));

  {
    const Array *IdTableArray = RootObject.getArray("id_table");
    if (!IdTableArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "IdTable", "id_table", "array")
          .build();
    }

    auto ExpectedIdTable = tuEntityIdTableFromJSON(*IdTableArray);
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

llvm::Error JSONFormat::writeTUSummary(const TUSummary &S,
                                       llvm::StringRef Path) {
  Object RootObject;

  RootObject[JSONTypeKey] = JSONTypeValueTUSummary;

  RootObject["target_triple"] =
      llvm::Triple::normalize(getTargetTriple(S).str());

  RootObject["tu_namespace"] = buildNamespaceToJSON(getTUNamespace(S));

  RootObject["id_table"] = tuEntityIdTableToJSON(getIdTable(S));

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
