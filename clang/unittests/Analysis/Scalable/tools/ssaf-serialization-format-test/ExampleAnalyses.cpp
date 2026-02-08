//===- ExampleAnalyses.cpp - Example analysis data for clang-ssaf-dump ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines example analysis data structures and their serialization/
// deserialization functions for the JSONFormat.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Registry.h"
#include <map>
#include <set>

using namespace clang::ssaf;

//===----------------------------------------------------------------------===//
// CallGraphAnalysis - Tracks caller-to-callees relationships
//===----------------------------------------------------------------------===//

namespace {
/// Example analysis that tracks function call relationships.
///
/// This analysis builds a call graph where each function (represented by an
/// EntityId) is mapped to the set of functions it directly calls. This is
/// useful for understanding control flow and dependencies between functions.
///
/// Example structure:
///   CallGraph[functionA] = {functionB, functionC}
///   CallGraph[functionB] = {functionD}
///
/// This indicates that functionA calls functionB and functionC, and
/// functionB calls functionD.
struct CallGraphAnalysis : EntitySummary {
  CallGraphAnalysis() : EntitySummary(SummaryName("CallGraph")) {}

  /// Maps each caller function (EntityId) to the set of functions it calls.
  /// Key: Caller function EntityId
  /// Value: Set of callee function EntityIds
  std::map<EntityId, std::set<EntityId>> CallGraph;
};
} // namespace

static llvm::json::Object
serializeCallGraph(const EntitySummary &Data,
                   const JSONFormat::EntityIdConverter &Converter) {
  const auto &CG = static_cast<const CallGraphAnalysis &>(Data);
  llvm::json::Object Result;

  // Serialize the call graph as an array of objects
  llvm::json::Array CallGraphArray;
  for (const auto &[Caller, Callees] : CG.CallGraph) {
    llvm::json::Object Entry;
    Entry["caller"] = Converter.toJSON(Caller);

    llvm::json::Array CalleesArray;
    for (const auto &Callee : Callees) {
      CalleesArray.push_back(Converter.toJSON(Callee));
    }
    Entry["callees"] = std::move(CalleesArray);

    CallGraphArray.push_back(std::move(Entry));
  }

  Result["call_graph"] = std::move(CallGraphArray);
  return Result;
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserializeCallGraph(const llvm::json::Object &JSONObj, EntityIdTable &Table,
                     const JSONFormat::EntityIdConverter &Converter) {
  auto Result = std::make_unique<CallGraphAnalysis>();

  const llvm::json::Array *CallGraphArray = JSONObj.getArray("call_graph");
  if (!CallGraphArray) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "CallGraph: missing or invalid 'call_graph' field");
  }

  for (size_t Index = 0; Index < CallGraphArray->size(); ++Index) {
    const llvm::json::Object *EntryObj = (*CallGraphArray)[Index].getAsObject();
    if (!EntryObj) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CallGraph: call_graph entry at index %zu is not an object", Index);
    }

    // Parse caller
    const llvm::json::Value *CallerValue = EntryObj->get("caller");
    if (!CallerValue) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CallGraph: entry at index %zu missing 'caller' field", Index);
    }
    auto CallerInt = CallerValue->getAsUINT64();
    if (!CallerInt) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CallGraph: 'caller' at index %zu is not a valid uint64", Index);
    }
    EntityId Caller = Converter.fromJSON(*CallerInt);

    // Parse callees array
    const llvm::json::Array *CalleesArray = EntryObj->getArray("callees");
    if (!CalleesArray) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CallGraph: entry at index %zu missing or invalid 'callees' field",
          Index);
    }

    std::set<EntityId> Callees;
    for (size_t CalleeIndex = 0; CalleeIndex < CalleesArray->size();
         ++CalleeIndex) {
      auto CalleeInt = (*CalleesArray)[CalleeIndex].getAsUINT64();
      if (!CalleeInt) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "CallGraph: callee at index %zu,%zu is not a valid uint64", Index,
            CalleeIndex);
      }
      Callees.insert(Converter.fromJSON(*CalleeInt));
    }

    Result->CallGraph[Caller] = std::move(Callees);
  }

  // Return by value to enable NRVO (Named Return Value Optimization)
  return Result;
}

namespace {
using FormatInfo = JSONFormat::FormatInfo;
struct CallGraphFormatInfo : FormatInfo {
  CallGraphFormatInfo()
      : FormatInfo{
            SummaryName("CallGraph"),
            serializeCallGraph,
            deserializeCallGraph,
        } {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<CallGraphFormatInfo>
    RegisterCallGraph("CallGraphAnalysis",
                      "Format info for CallGraph analysis data");

//===----------------------------------------------------------------------===//
// DefUseAnalysis - Tracks definition-use chains
//===----------------------------------------------------------------------===//

namespace {
/// Example analysis that tracks definition-use chains for variables.
///
/// This analysis builds def-use chains showing how variable definitions flow
/// to their use sites. The structure is a three-level nested map:
///
/// Level 1: Variable EntityId
///   Level 2: Definition EntityId (where the variable is defined)
///     Level 3: Set of Use EntityIds (where that definition is used)
///
/// Example structure:
///   DefUseChains[varX][def1] = {use1, use2}
///   DefUseChains[varX][def2] = {use3}
///   DefUseChains[varY][def3] = {use4, use5, use6}
///
/// This indicates that:
/// - Variable varX has two definitions (def1, def2)
/// - def1 is used at use1 and use2
/// - def2 is used at use3
/// - Variable varY has one definition (def3) used at use4, use5, use6
struct DefUseAnalysis : EntitySummary {
  DefUseAnalysis() : EntitySummary(SummaryName("DefUse")) {}

  /// Maps variables to their definition-use chains.
  /// Key: Variable EntityId
  /// Value: Map from definition EntityId to set of use EntityIds
  std::map<EntityId, std::map<EntityId, std::set<EntityId>>> DefUseChains;
};
} // namespace

static llvm::json::Object
serializeDefUse(const EntitySummary &Data,
                const JSONFormat::EntityIdConverter &Converter) {
  const auto &DU = static_cast<const DefUseAnalysis &>(Data);
  llvm::json::Object Result;

  // Serialize def-use chains as an array of objects
  llvm::json::Array ChainsArray;
  for (const auto &[Variable, DefUseMap] : DU.DefUseChains) {
    llvm::json::Object VarEntry;
    VarEntry["variable"] = Converter.toJSON(Variable);

    llvm::json::Array DefsArray;
    for (const auto &[Definition, Uses] : DefUseMap) {
      llvm::json::Object DefEntry;
      DefEntry["definition"] = Converter.toJSON(Definition);

      llvm::json::Array UsesArray;
      for (const auto &Use : Uses) {
        UsesArray.push_back(Converter.toJSON(Use));
      }
      DefEntry["uses"] = std::move(UsesArray);

      DefsArray.push_back(std::move(DefEntry));
    }
    VarEntry["definitions"] = std::move(DefsArray);

    ChainsArray.push_back(std::move(VarEntry));
  }

  Result["def_use_chains"] = std::move(ChainsArray);
  return Result;
}

static llvm::Expected<std::unique_ptr<EntitySummary>>
deserializeDefUse(const llvm::json::Object &JSONObj, EntityIdTable &Table,
                  const JSONFormat::EntityIdConverter &Converter) {
  auto Result = std::make_unique<DefUseAnalysis>();

  const llvm::json::Array *ChainsArray = JSONObj.getArray("def_use_chains");
  if (!ChainsArray) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "DefUse: missing or invalid 'def_use_chains' field");
  }

  for (size_t VarIndex = 0; VarIndex < ChainsArray->size(); ++VarIndex) {
    const llvm::json::Object *VarObj = (*ChainsArray)[VarIndex].getAsObject();
    if (!VarObj) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "DefUse: def_use_chains entry at index %zu is not an object",
          VarIndex);
    }

    // Parse variable
    const llvm::json::Value *VarValue = VarObj->get("variable");
    if (!VarValue) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "DefUse: entry at index %zu missing 'variable' field", VarIndex);
    }
    auto VarInt = VarValue->getAsUINT64();
    if (!VarInt) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "DefUse: 'variable' at index %zu is not a valid uint64", VarIndex);
    }
    EntityId Variable = Converter.fromJSON(*VarInt);

    // Parse definitions array
    const llvm::json::Array *DefsArray = VarObj->getArray("definitions");
    if (!DefsArray) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "DefUse: entry at index %zu missing or invalid 'definitions' field",
          VarIndex);
    }

    std::map<EntityId, std::set<EntityId>> DefUseMap;
    for (size_t DefIndex = 0; DefIndex < DefsArray->size(); ++DefIndex) {
      const llvm::json::Object *DefObj = (*DefsArray)[DefIndex].getAsObject();
      if (!DefObj) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "DefUse: definition at index %zu,%zu is not an object", VarIndex,
            DefIndex);
      }

      // Parse definition
      const llvm::json::Value *DefValue = DefObj->get("definition");
      if (!DefValue) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "DefUse: definition at index %zu,%zu missing 'definition' field",
            VarIndex, DefIndex);
      }
      auto DefInt = DefValue->getAsUINT64();
      if (!DefInt) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "DefUse: 'definition' at index %zu,%zu is not a valid uint64",
            VarIndex, DefIndex);
      }
      EntityId Definition = Converter.fromJSON(*DefInt);

      // Parse uses array
      const llvm::json::Array *UsesArray = DefObj->getArray("uses");
      if (!UsesArray) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "DefUse: definition at index %zu,%zu missing or invalid 'uses' "
            "field",
            VarIndex, DefIndex);
      }

      std::set<EntityId> Uses;
      for (size_t UseIndex = 0; UseIndex < UsesArray->size(); ++UseIndex) {
        auto UseInt = (*UsesArray)[UseIndex].getAsUINT64();
        if (!UseInt) {
          return llvm::createStringError(
              std::errc::invalid_argument,
              "DefUse: use at index %zu,%zu,%zu is not a valid uint64",
              VarIndex, DefIndex, UseIndex);
        }
        Uses.insert(Converter.fromJSON(*UseInt));
      }

      DefUseMap[Definition] = std::move(Uses);
    }

    Result->DefUseChains[Variable] = std::move(DefUseMap);
  }

  return Result;
}

namespace {
struct DefUseFormatInfo : FormatInfo {
  DefUseFormatInfo()
      : FormatInfo{
            SummaryName("DefUse"),
            serializeDefUse,
            deserializeDefUse,
        } {}
};
} // namespace

static llvm::Registry<JSONFormat::FormatInfo>::Add<DefUseFormatInfo>
    RegisterDefUse("DefUseAnalysis", "Format info for DefUse analysis data");
