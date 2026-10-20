//===- NullabilitySafetyFormat.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysis/Analyses/NullabilitySafety/NullabilitySafety.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowFormat.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

using namespace clang;
using namespace ssaf;
using Object = llvm::json::Object;

static constexpr llvm::StringLiteral NonnullKey = "NonnullEvidence";
static constexpr llvm::StringLiteral NullableKey = "NullableEvidence";
static constexpr llvm::StringLiteral AllReturnsNonnullKey = "AllReturnsNonnull";
static constexpr llvm::StringLiteral MaybeNullKey = "MaybeNullEvidence";
static constexpr llvm::StringLiteral UnknownKey = "UnknownEvidence";
static constexpr llvm::StringLiteral ConditionalKey = "ConditionalEvidence";

static Object serialize(const EntitySummary &S,
                        JSONFormat::EntityIdToJSONFn Fn) {
  const auto &NS = static_cast<const NullabilitySafetyEntitySummary &>(S);
  auto toJSON = [&](const EntityPointerLevelSet &EPLs) {
    return entityPointerLevelSetToJSON(
        llvm::make_range(EPLs.begin(), EPLs.end()), Fn);
  };
  return Object{
      {NonnullKey.data(), toJSON(NS.getNonnullEvidence())},
      {NullableKey.data(), toJSON(NS.getNullableEvidence())},
      {AllReturnsNonnullKey.data(), toJSON(NS.getAllReturnsNonnull())},
      {MaybeNullKey.data(), toJSON(NS.getMaybeNullEvidence())},
      {UnknownKey.data(), toJSON(NS.getUnknownEvidence())},
      {ConditionalKey.data(),
       edgeSetToJSON(llvm::make_range(NS.getConditionalEvidence().begin(),
                                      NS.getConditionalEvidence().end()),
                     Fn)}};
}

static llvm::Expected<EntityPointerLevelSet>
readSet(const Object &Data, llvm::StringLiteral Key,
        JSONFormat::EntityIdFromJSONFn Fn) {
  const llvm::json::Array *Arr = Data.getArray(Key.data());
  if (!Arr)
    return makeSawButExpectedError(Object(Data), "an Object with a key %s",
                                   Key.data());
  return entityPointerLevelSetFromJSON(*Arr, Fn);
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserialize(const Object &Data, EntityIdTable &,
            JSONFormat::EntityIdFromJSONFn Fn) {
  llvm::Expected<EntityPointerLevelSet> Nonnull = readSet(Data, NonnullKey, Fn);
  if (!Nonnull)
    return Nonnull.takeError();
  llvm::Expected<EntityPointerLevelSet> Nullable =
      readSet(Data, NullableKey, Fn);
  if (!Nullable)
    return Nullable.takeError();
  llvm::Expected<EntityPointerLevelSet> AllReturnsNonnull =
      readSet(Data, AllReturnsNonnullKey, Fn);
  if (!AllReturnsNonnull)
    return AllReturnsNonnull.takeError();
  llvm::Expected<EntityPointerLevelSet> MaybeNull =
      readSet(Data, MaybeNullKey, Fn);
  if (!MaybeNull)
    return MaybeNull.takeError();
  llvm::Expected<EntityPointerLevelSet> Unknown = readSet(Data, UnknownKey, Fn);
  if (!Unknown)
    return Unknown.takeError();
  const llvm::json::Array *ConditionalArr =
      Data.getArray(ConditionalKey.data());
  if (!ConditionalArr)
    return makeSawButExpectedError(Object(Data), "an Object with a key %s",
                                   ConditionalKey.data());
  llvm::Expected<EdgeSet> Conditional = edgeSetFromJSON(*ConditionalArr, Fn);
  if (!Conditional)
    return Conditional.takeError();
  return std::make_unique<NullabilitySafetyEntitySummary>(
      std::move(*Nonnull), std::move(*Nullable), std::move(*AllReturnsNonnull),
      std::move(*MaybeNull), std::move(*Unknown), std::move(*Conditional));
}

namespace {
struct NullabilitySafetyJSONFormatInfo final : JSONFormat::FormatInfo {
  NullabilitySafetyJSONFormatInfo()
      : JSONFormat::FormatInfo(NullabilitySafetyEntitySummary::summaryName(),
                               serialize, deserialize) {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<
    NullabilitySafetyJSONFormatInfo>
    RegisterNullabilitySafetyJSONFormatInfo(
        NullabilitySafetyEntitySummary::Name,
        "JSON Format info for NullabilitySafetyEntitySummary");

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int NullabilitySafetyJSONFormatAnchorSource = 0;
} // namespace clang::ssaf
