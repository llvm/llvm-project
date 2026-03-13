//===- LUSummaryEncoding.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummaryEncoding.h"

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

  const Object &RootObject = *RootObjectPtr;

  const Array *LUNamespaceArray = RootObject.getArray("lu_namespace");
  if (!LUNamespaceArray) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "NestedBuildNamespace", "lu_namespace", "array")
        .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
        .build();
  }

  auto ExpectedLUNamespace = nestedBuildNamespaceFromJSON(*LUNamespaceArray);
  if (!ExpectedLUNamespace) {
    return ErrorBuilder::wrap(ExpectedLUNamespace.takeError())
        .context(ErrorMessages::ReadingFromField, "NestedBuildNamespace",
                 "lu_namespace")
        .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
        .build();
  }

  LUSummaryEncoding Encoding(std::move(*ExpectedLUNamespace));

  {
    const Array *IdTableArray = RootObject.getArray("id_table");
    if (!IdTableArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "IdTable", "id_table", "array")
          .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
          .build();
    }

    auto ExpectedIdTable = entityIdTableFromJSON(*IdTableArray);
    if (!ExpectedIdTable) {
      return ErrorBuilder::wrap(ExpectedIdTable.takeError())
          .context(ErrorMessages::ReadingFromField, "IdTable", "id_table")
          .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
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
          .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
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
          .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
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
          .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
          .build();
    }

    auto ExpectedEncodingSummaryDataMap =
        encodingSummaryDataMapFromJSON(*SummaryDataArray);
    if (!ExpectedEncodingSummaryDataMap) {
      return ErrorBuilder::wrap(ExpectedEncodingSummaryDataMap.takeError())
          .context(ErrorMessages::ReadingFromField, "SummaryData entries",
                   "data")
          .context(ErrorMessages::ReadingFromFile, "LUSummary", Path)
          .build();
    }

    getData(Encoding) = std::move(*ExpectedEncodingSummaryDataMap);
  }

  return std::move(Encoding);
}

llvm::Error
JSONFormat::writeLUSummaryEncoding(const LUSummaryEncoding &SummaryEncoding,
                                   llvm::StringRef Path) {
  Object RootObject;

  RootObject["lu_namespace"] =
      nestedBuildNamespaceToJSON(getLUNamespace(SummaryEncoding));

  RootObject["id_table"] = entityIdTableToJSON(getIdTable(SummaryEncoding));

  RootObject["linkage_table"] =
      linkageTableToJSON(getLinkageTable(SummaryEncoding));

  RootObject["data"] = encodingSummaryDataMapToJSON(getData(SummaryEncoding));

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "LUSummary", Path)
        .build();
  }

  return llvm::Error::success();
}

} // namespace clang::ssaf
