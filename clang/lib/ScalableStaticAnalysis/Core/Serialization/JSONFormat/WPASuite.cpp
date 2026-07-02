//===- WPASuite.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatImpl.h"

#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"

namespace clang::ssaf {

//----------------------------------------------------------------------------
// AnalysisResultMapEntry
//----------------------------------------------------------------------------

llvm::Expected<std::pair<AnalysisName, std::unique_ptr<AnalysisResult>>>
JSONFormat::analysisResultMapEntryFromJSON(const Object &Entry) const {
  auto OptName = Entry.getString("analysis_name");
  if (!OptName) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "AnalysisName", "analysis_name", "string")
        .build();
  }

  AnalysisName Name = analysisNameFromJSON(*OptName);

  auto ExpectedCodec = AnalysisResultRegistry::instantiate(Name);
  if (!ExpectedCodec) {
    return ExpectedCodec.takeError();
  }

  const Object *ResultObj = Entry.getObject("result");
  if (!ResultObj) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObjectAtField,
                                "AnalysisResult", "result", "object")
        .build();
  }

  auto ExpectedResult =
      (*ExpectedCodec)->deserialize(*ResultObj, &entityIdFromJSONObject);
  if (!ExpectedResult) {
    return ExpectedResult.takeError();
  }

  return std::make_pair(std::move(Name), std::move(*ExpectedResult));
}

llvm::Expected<Object> JSONFormat::analysisResultMapEntryToJSON(
    const AnalysisName &Name,
    const std::unique_ptr<AnalysisResult> &Result) const {
  auto ExpectedCodec = AnalysisResultRegistry::instantiate(Name);
  if (!ExpectedCodec) {
    return ExpectedCodec.takeError();
  }

  Object Entry;
  Entry["analysis_name"] = analysisNameToJSON(Name);
  Entry["result"] = (*ExpectedCodec)->serialize(*Result, &entityIdToJSONObject);
  return Entry;
}

//----------------------------------------------------------------------------
// AnalysisResultMap
//----------------------------------------------------------------------------

llvm::Expected<std::map<AnalysisName, std::unique_ptr<AnalysisResult>>>
JSONFormat::analysisResultMapFromJSON(const Array &ResultsArray) const {
  std::map<AnalysisName, std::unique_ptr<AnalysisResult>> Results;

  auto AsObject = [](const Value &V) { return V.getAsObject(); };
  auto ObjectsRange = llvm::map_range(ResultsArray, AsObject);

  for (auto [I, Entry] : enumerate(ObjectsRange)) {
    if (!Entry) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtIndex,
                                  "WPA result entry", I, "object")
          .build();
    }

    auto ExpectedPair = analysisResultMapEntryFromJSON(*Entry);
    if (!ExpectedPair) {
      return ErrorBuilder::wrap(ExpectedPair.takeError())
          .context(ErrorMessages::ReadingFromIndex, "WPA result entry", I)
          .build();
    }

    auto [Name, Result] = std::move(*ExpectedPair);
    bool Inserted = Results.try_emplace(Name, std::move(Result)).second;
    if (!Inserted) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedInsertionOnDuplication,
                                  "WPA result", I, Name)
          .build();
    }
  }
  return std::move(Results);
}

llvm::Expected<Array> JSONFormat::analysisResultMapToJSON(
    const std::map<AnalysisName, std::unique_ptr<AnalysisResult>> &Data) const {
  Array Results;
  for (const auto &[Name, Result] : Data) {
    auto ExpectedEntry = analysisResultMapEntryToJSON(Name, Result);
    if (!ExpectedEntry) {
      return ExpectedEntry.takeError();
    }
    Results.push_back(std::move(*ExpectedEntry));
  }
  return Results;
}

//----------------------------------------------------------------------------
// WPASuite
//----------------------------------------------------------------------------

llvm::Expected<WPASuite> JSONFormat::readWPASuite(llvm::StringRef Path) {
  auto ExpectedJSON = readJSON(Path);
  if (!ExpectedJSON) {
    return ErrorBuilder::wrap(ExpectedJSON.takeError())
        .context(ErrorMessages::ReadingFromFile, "WPASuite", Path)
        .build();
  }

  Object *RootObjectPtr = ExpectedJSON->getAsObject();
  if (!RootObjectPtr) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                ErrorMessages::FailedToReadObject, "WPASuite",
                                "object")
        .context(ErrorMessages::ReadingFromFile, "WPASuite", Path)
        .build();
  }

  if (auto Err = checkSummaryType(*RootObjectPtr, JSONTypeValueWPASuite)) {
    return ErrorBuilder::wrap(std::move(Err))
        .context(ErrorMessages::ReadingFromFile, "WPASuite", Path)
        .build();
  }

  auto ExpectedSuite = readWPASuiteFromObject(*RootObjectPtr);
  if (!ExpectedSuite) {
    return ErrorBuilder::wrap(ExpectedSuite.takeError())
        .context(ErrorMessages::ReadingFromFile, "WPASuite", Path)
        .build();
  }

  return std::move(*ExpectedSuite);
}

llvm::Expected<WPASuite>
JSONFormat::readWPASuiteFromObject(const Object &RootObject) {
  WPASuite Suite = makeWPASuite();

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

    getIdTable(Suite) = std::move(*ExpectedIdTable);
  }

  {
    const Array *ResultsArray = RootObject.getArray("results");
    if (!ResultsArray) {
      return ErrorBuilder::create(std::errc::invalid_argument,
                                  ErrorMessages::FailedToReadObjectAtField,
                                  "WPA results", "results", "array")
          .build();
    }

    auto ExpectedResultsMap = analysisResultMapFromJSON(*ResultsArray);
    if (!ExpectedResultsMap) {
      return ErrorBuilder::wrap(ExpectedResultsMap.takeError())
          .context(ErrorMessages::ReadingFromField, "WPA results", "results")
          .build();
    }

    getData(Suite) = std::move(*ExpectedResultsMap);
  }

  return std::move(Suite);
}

llvm::Error JSONFormat::writeWPASuite(const WPASuite &Suite,
                                      llvm::StringRef Path) {
  Object RootObject;

  RootObject[JSONTypeKey] = JSONTypeValueWPASuite;

  RootObject["id_table"] = luEntityIdTableToJSON(getIdTable(Suite));

  auto ExpectedResults = analysisResultMapToJSON(getData(Suite));
  if (!ExpectedResults) {
    return ErrorBuilder::wrap(ExpectedResults.takeError())
        .context(ErrorMessages::WritingToFile, "WPASuite", Path)
        .build();
  }

  RootObject["results"] = std::move(*ExpectedResults);

  if (auto Error = writeJSON(std::move(RootObject), Path)) {
    return ErrorBuilder::wrap(std::move(Error))
        .context(ErrorMessages::WritingToFile, "WPASuite", Path)
        .build();
  }

  return llvm::Error::success();
}

} // namespace clang::ssaf
