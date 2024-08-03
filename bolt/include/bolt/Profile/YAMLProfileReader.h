//===- bolt/Profile/YAMLProfileReader.h - YAML profile reader ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_YAML_PROFILE_READER_H
#define BOLT_PROFILE_YAML_PROFILE_READER_H

#include "bolt/Profile/ProfileReaderBase.h"
#include "bolt/Profile/ProfileYAMLMapping.h"
#include <unordered_set>

namespace llvm {
namespace bolt {

class YAMLProfileReader : public ProfileReaderBase {
public:
  explicit YAMLProfileReader(StringRef Filename)
      : ProfileReaderBase(Filename) {}

  StringRef getReaderName() const override { return "YAML profile reader"; }

  bool isTrustedSource() const override { return false; }

  Error readProfilePreCFG(BinaryContext &BC) override {
    return Error::success();
  }

  Error readProfile(BinaryContext &BC) override;

  Error preprocessProfile(BinaryContext &BC) override;

  bool hasLocalsWithFileName() const override;

  bool mayHaveProfileData(const BinaryFunction &BF) override;

  /// Check if the file contains YAML.
  static bool isYAML(StringRef Filename);

  using ProfileLookupMap =
      DenseMap<uint32_t, yaml::bolt::BinaryFunctionProfile *>;

  /// A class for matching binary functions in functions in the YAML profile.
  /// First, a call graph is constructed for both profiled and binary functions.
  /// Then functions are hashed based on the names of their callee/caller
  /// functions. Finally, functions are matched based on these neighbor hashes.
  class CallGraphMatcher {
  public:
    /// Constructs the call graphs for binary and profiled functions and
    /// computes neighbor hashes for binary functions.
    CallGraphMatcher(BinaryContext &BC, yaml::bolt::BinaryProfile &YamlBP,
                     ProfileLookupMap &IdToYAMLBF);

    /// Returns the YamlBFs adjacent to the parameter YamlBF in the call graph.
    std::optional<std::set<yaml::bolt::BinaryFunctionProfile *>>
    getAdjacentYamlBFs(yaml::bolt::BinaryFunctionProfile &YamlBF) {
      auto It = YamlBFAdjacencyMap.find(&YamlBF);
      return It == YamlBFAdjacencyMap.end() ? std::nullopt
                                            : std::make_optional(It->second);
    }

    /// Returns the binary functions with the parameter neighbor hash.
    std::optional<std::vector<BinaryFunction *>>
    getBFsWithNeighborHash(uint64_t NeighborHash) {
      auto It = NeighborHashToBFs.find(NeighborHash);
      return It == NeighborHashToBFs.end() ? std::nullopt
                                           : std::make_optional(It->second);
    }

  private:
    /// Adds edges to the binary function call graph given the callsites of the
    /// parameter function.
    void constructBFCG(BinaryContext &BC, yaml::bolt::BinaryProfile &YamlBP);

    /// Using the constructed binary function call graph, computes and creates
    /// mappings from "neighbor hash" (composed of the function names of callee
    /// and caller functions of a function) to binary functions.
    void computeBFNeighborHashes(BinaryContext &BC);

    /// Constructs the call graph for profile functions.
    void constructYAMLFCG(yaml::bolt::BinaryProfile &YamlBP,
                          ProfileLookupMap &IdToYAMLBF);

    /// Adjacency map for binary functions in the call graph.
    DenseMap<BinaryFunction *, std::set<BinaryFunction *>> BFAdjacencyMap;

    /// Maps neighbor hashes to binary functions.
    DenseMap<uint64_t, std::vector<BinaryFunction *>> NeighborHashToBFs;

    /// Adjacency map for profile functions in the call graph.
    DenseMap<yaml::bolt::BinaryFunctionProfile *,
             std::set<yaml::bolt::BinaryFunctionProfile *>>
        YamlBFAdjacencyMap;
  };

private:
  /// Adjustments for basic samples profiles (without LBR).
  bool NormalizeByInsnCount{false};
  bool NormalizeByCalls{false};

  /// Binary profile in YAML format.
  yaml::bolt::BinaryProfile YamlBP;

  /// Map a function ID from a YAML profile to a BinaryFunction object.
  std::vector<BinaryFunction *> YamlProfileToFunction;

  using FunctionSet = std::unordered_set<const BinaryFunction *>;
  /// To keep track of functions that have a matched profile before the profile
  /// is attributed.
  FunctionSet ProfiledFunctions;

  /// Maps profiled function id to function, for function matching with calls as
  /// anchors.
  ProfileLookupMap IdToYamLBF;

  /// For LTO symbol resolution.
  /// Map a common LTO prefix to a list of YAML profiles matching the prefix.
  StringMap<std::vector<yaml::bolt::BinaryFunctionProfile *>> LTOCommonNameMap;

  /// Map a common LTO prefix to a set of binary functions.
  StringMap<std::unordered_set<BinaryFunction *>> LTOCommonNameFunctionMap;

  /// Function names in profile.
  StringSet<> ProfileFunctionNames;

  /// BinaryFunction pointers indexed by YamlBP functions.
  std::vector<BinaryFunction *> ProfileBFs;

  /// Populate \p Function profile with the one supplied in YAML format.
  bool parseFunctionProfile(BinaryFunction &Function,
                            const yaml::bolt::BinaryFunctionProfile &YamlBF);

  /// Checks if a function profile matches a binary function.
  bool profileMatches(const yaml::bolt::BinaryFunctionProfile &Profile,
                      const BinaryFunction &BF);

  /// Infer function profile from stale data (collected on older binaries).
  bool inferStaleProfile(BinaryFunction &Function,
                         const yaml::bolt::BinaryFunctionProfile &YamlBF);

  /// Initialize maps for profile matching.
  void buildNameMaps(BinaryContext &BC);

  /// Matches functions using exact name.
  size_t matchWithExactName();

  /// Matches function using LTO comomon name.
  size_t matchWithLTOCommonName();

  /// Matches functions using exact hash.
  size_t matchWithHash(BinaryContext &BC);

  /// Matches functions using the call graph.
  size_t matchWithCallGraph(BinaryContext &BC);

  /// Matches functions with similarly named profiled functions.
  size_t matchWithNameSimilarity(BinaryContext &BC);

  /// Update matched YAML -> BinaryFunction pair.
  void matchProfileToFunction(yaml::bolt::BinaryFunctionProfile &YamlBF,
                              BinaryFunction &BF) {
    if (YamlBF.Id >= YamlProfileToFunction.size())
      YamlProfileToFunction.resize(YamlBF.Id + 1);
    YamlProfileToFunction[YamlBF.Id] = &BF;
    YamlBF.Used = true;

    assert(!ProfiledFunctions.count(&BF) &&
           "function already has an assigned profile");
    ProfiledFunctions.emplace(&BF);
  }

  /// Check if the profile uses an event with a given \p Name.
  bool usesEvent(StringRef Name) const;
};

} // namespace bolt
} // namespace llvm

#endif
