//===- LUSummaryEncoding.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "llvm/TargetParser/Triple.h"

#include <set>

namespace clang::ssaf {

//----------------------------------------------------------------------------
// LUSummaryEncoding
//----------------------------------------------------------------------------

llvm::Expected<LUSummaryEncoding>
JSONFormat::readLUSummaryEncoding(llvm::StringRef Path) {
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

  auto ExpectedEncoding = readLUSummaryEncodingFromObject(*RootObjectPtr);
  if (!ExpectedEncoding) {
    return ErrorBuilder::wrap(ExpectedEncoding.takeError())
        .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
        .build();
  }

  return std::move(*ExpectedEncoding);
}

llvm::Expected<LUSummaryEncoding>
JSONFormat::readLUSummaryEncodingFromObject(const Object &RootObject) {
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

  LUSummaryEncoding Encoding(std::move(T), std::move(*ExpectedLUNamespace));

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

    getIdTable(Encoding) = std::move(*ExpectedIdTable);
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
        llvm::make_second_range(getEntities(getIdTable(Encoding)));
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

    getLinkageTable(Encoding) = std::move(*ExpectedLinkageTable);
  }

  {
    const Array *SummaryDataArray = RootObject.getArray("data");
    if (!SummaryDataArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "SummaryData entries", "data", "array")
          .build();
    }

    auto ExpectedEncodingSummaryDataMap =
        encodingSummaryDataMapFromJSON(*SummaryDataArray);
    if (!ExpectedEncodingSummaryDataMap) {
      return ErrorBuilder::wrap(ExpectedEncodingSummaryDataMap.takeError())
          .context(ErrorMessages::ReadingFromField, "SummaryData entries",
                   "data")
          .build();
    }

    getData(Encoding) = std::move(*ExpectedEncodingSummaryDataMap);
  }

  return std::move(Encoding);
}

llvm::Error
JSONFormat::writeLUSummaryEncoding(const LUSummaryEncoding &SummaryEncoding,
                                   llvm::StringRef Path) {
  if (auto Error = writeJSON(luSummaryEncodingToJSON(SummaryEncoding), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "LUSummary", Path)
        .build();
  }

  return llvm::Error::success();
}

Object JSONFormat::luSummaryEncodingToJSON(
    const LUSummaryEncoding &SummaryEncoding) const {
  Object RootObject;

  RootObject[JSONTypeKey] = JSONTypeValueLUSummary;

  RootObject["target_triple"] =
      llvm::Triple::normalize(getTargetTriple(SummaryEncoding).str());

  RootObject["lu_namespace"] =
      nestedBuildNamespaceToJSON(getLUNamespace(SummaryEncoding));

  RootObject["id_table"] = luEntityIdTableToJSON(getIdTable(SummaryEncoding));

  RootObject["linkage_table"] =
      linkageTableToJSON(getLinkageTable(SummaryEncoding));

  RootObject["data"] = encodingSummaryDataMapToJSON(getData(SummaryEncoding));

  return RootObject;
}

} // namespace clang::ssaf
