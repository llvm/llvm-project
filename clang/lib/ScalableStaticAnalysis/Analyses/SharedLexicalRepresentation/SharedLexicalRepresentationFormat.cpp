//===- SharedLexicalRepresentationFormat.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysis/Analyses/SharedLexicalRepresentation/SharedLexicalRepresentation.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"

#include <memory>
#include <utility>
#include <vector>

using namespace clang;
using namespace ssaf;
using Array = llvm::json::Array;
using Object = llvm::json::Object;
using Value = llvm::json::Value;

namespace clang::ssaf {

EntitySourceLocationsSummary
buildEntitySourceLocationsSummary(std::vector<SourceLocationRecord> Locs) {
  return EntitySourceLocationsSummary(std::move(Locs));
}

llvm::ArrayRef<SourceLocationRecord>
getDeclLocations(const EntitySourceLocationsSummary &S) {
  return S.DeclLocations;
}

} // namespace clang::ssaf

static constexpr llvm::StringLiteral DeclLocationsKey = "decl_locations";
static constexpr llvm::StringLiteral FilePathKey = "file_path";
static constexpr llvm::StringLiteral LineKey = "line";
static constexpr llvm::StringLiteral ColumnKey = "column";

static Object sourceLocationRecordToJSON(const SourceLocationRecord &R) {
  return Object{{FilePathKey.data(), R.FilePath},
                {LineKey.data(), static_cast<int64_t>(R.Line)},
                {ColumnKey.data(), static_cast<int64_t>(R.Column)}};
}

static llvm::Expected<SourceLocationRecord>
sourceLocationRecordFromJSON(const Value &V) {
  const Object *Obj = V.getAsObject();
  if (!Obj)
    return makeSawButExpectedError(V, "an object {%s, %s, %s}",
                                   FilePathKey.data(), LineKey.data(),
                                   ColumnKey.data());

  auto FilePath = Obj->getString(FilePathKey);
  if (!FilePath)
    return makeSawButExpectedError(*Obj, "an object with a string field %s",
                                   FilePathKey.data());

  auto Line = Obj->getInteger(LineKey);
  if (!Line)
    return makeSawButExpectedError(*Obj, "an object with an integer field %s",
                                   LineKey.data());

  auto Column = Obj->getInteger(ColumnKey);
  if (!Column)
    return makeSawButExpectedError(*Obj, "an object with an integer field %s",
                                   ColumnKey.data());

  SourceLocationRecord R;
  R.FilePath = FilePath->str();
  R.Line = static_cast<unsigned>(*Line);
  R.Column = static_cast<unsigned>(*Column);
  return R;
}

static Object serialize(const EntitySummary &ES, JSONFormat::EntityIdToJSONFn) {
  const auto &S = static_cast<const EntitySourceLocationsSummary &>(ES);
  Array Locs;
  for (const auto &R : getDeclLocations(S))
    Locs.push_back(sourceLocationRecordToJSON(R));
  return Object{{DeclLocationsKey.data(), std::move(Locs)}};
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserialize(const Object &Data, EntityIdTable &,
            JSONFormat::EntityIdFromJSONFn) {
  const Array *Locs = Data.getArray(DeclLocationsKey);
  if (!Locs)
    return makeSawButExpectedError(Object(Data),
                                   "an Object with an array field %s",
                                   DeclLocationsKey.data());

  std::vector<SourceLocationRecord> Records;
  Records.reserve(Locs->size());
  for (const Value &V : *Locs) {
    auto R = sourceLocationRecordFromJSON(V);
    if (!R)
      return R.takeError();
    Records.push_back(std::move(*R));
  }
  return std::make_unique<EntitySourceLocationsSummary>(
      buildEntitySourceLocationsSummary(std::move(Records)));
}

namespace {
struct SharedLexicalRepresentationJSONFormatInfo final
    : JSONFormat::FormatInfo {
  SharedLexicalRepresentationJSONFormatInfo()
      : JSONFormat::FormatInfo(EntitySourceLocationsSummary::summaryName(),
                               serialize, deserialize) {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    SharedLexicalRepresentationJSONFormatInfo>
    RegisterSharedLexicalRepresentationJSONFormatInfo(
        EntitySourceLocationsSummary::Name,
        "JSON Format info for EntitySourceLocationsSummary");

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int SharedLexicalRepresentationJSONFormatAnchorSource = 0;
} // namespace clang::ssaf
