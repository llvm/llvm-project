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
class MCDecodedPseudoProbeInlineTree;

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

  // A class for matching inline tree nodes between profile and binary.
  // Provides the mapping from profile inline tree node id to a
  // corresponding binary MCDecodedPseudoProbeInlineTree node.
  //
  // The whole mapping process is the following:
  //
  //     (profile)                             (binary)
  //      | blocks                                ^
  //      v                                       |
  // BinaryBasicBlockProfile                BasicBlock
  //      | probes                                ^
  //      v                                       | address
  // PseudoProbeInfo                        MCDecodedPseudoProbe
  //      | InlineTreeIndex                       ^
  //      v                                       | probe id
  // [ profile node id (uint32_t)    ->     MCDecodedPseudoProbeInlineTree *]
  //                     InlineTreeNodeMapTy
  using InlineTreeNodeMapTy =
      std::vector<const MCDecodedPseudoProbeInlineTree *>;

  // Match up \p YamlInlineTree with binary inline tree rooted at \p Root.
  // Return the number of matched nodes.
  //
  // This function populates the \p Map indexed by \p YamlInlineTree from
  // profile inline tree node id to a corresponding binary
  // MCDecodedPseudoProbeInlineTree node on a successful match or nullptr.
  size_t matchInlineTreesImpl(
      BinaryFunction &BF, yaml::bolt::BinaryFunctionProfile &YamlBF,
      const MCDecodedPseudoProbeInlineTree &Root, uint32_t RootIdx,
      ArrayRef<yaml::bolt::InlineTreeNode> YamlInlineTree,
      MutableArrayRef<const MCDecodedPseudoProbeInlineTree *> Map, float Scale);

  /// Support mapping the profile to the same binary function more than once.
  /// The index is used to check and prevent recursive (backtracking) matches.
  /// The value is a node map and scale factor, used to downscale outlined
  /// profile when inlining it into the function.
  using RootIdxToMapTy =
      std::map<uint32_t, std::pair<InlineTreeNodeMapTy, float>>;
  /// Probe matching specification: map from \p BinaryFunctionProfile to the
  /// partial match specification \p RootIdxToMapTy.
  using ProbeMatchSpec =
      std::unordered_map<const yaml::bolt::BinaryFunctionProfile *,
                         RootIdxToMapTy>;

private:
  /// Adjustments for basic samples profiles (without LBR).
  bool NormalizeByInsnCount{false};
  bool NormalizeByCalls{false};

  /// Binary profile in YAML format.
  yaml::bolt::BinaryProfile YamlBP;

  /// Map a function ID from a YAML profile to a BinaryFunction object.
  DenseMap<uint32_t, BinaryFunction *> YamlProfileToFunction;

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

  /// For pseudo probe function matching.
  /// Set of profile GUIDs.
  std::unordered_set<uint64_t> YamlGUIDs;
  /// Map binary function GUID to binary function.
  std::unordered_multimap<uint64_t, BinaryFunction *> GUIDToBF;

  /// Function names in profile.
  StringSet<> ProfileFunctionNames;

  /// BinaryFunction pointers indexed by YamlBP functions.
  std::vector<BinaryFunction *> ProfileBFs;

  // Mapping from a binary function to its partial match specification
  // (YAML profile and its inline tree mapping to binary).
  DenseMap<BinaryFunction *, ProbeMatchSpec> BFToProbeMatchSpecs;

  /// Pseudo probe matching stats.
  struct {
    // Inline tree root mismatch causes.
    uint64_t MismatchingRootGUID{0};
    uint64_t MismatchingRootHash{0};
    // Inline tree internal node mismatch causes.
    uint64_t MismatchingNodeHash{0};
    uint64_t MissingProfileNode{0};
    // Incomplete profile causes.
    uint64_t MissingCallProbe{0};
    uint64_t MissingCallee{0};
    uint64_t MissingInlineTree{0};
    uint64_t TotalCallSites{0};
    uint64_t MissingCallCount{0};
    uint64_t TotalCallCount{0};
    // Total matched stats.
    uint64_t MatchedRoots{0};
    uint64_t MatchedNodes{0};
    uint64_t AttemptedNodes{0};
    uint64_t AttemptedRoots{0};
  } ProbeMatchingStats;

  /// Populate \p Function profile with the one supplied in YAML format.
  bool parseFunctionProfile(BinaryFunction &Function,
                            const yaml::bolt::BinaryFunctionProfile &YamlBF);

  /// Checks if a function profile matches a binary function.
  bool profileMatches(const yaml::bolt::BinaryFunctionProfile &Profile,
                      const BinaryFunction &BF);

  /// Infer function profile from stale data (collected on older binaries).
  bool inferStaleProfile(BinaryFunction &Function,
                         const yaml::bolt::BinaryFunctionProfile &YamlBF,
                         const ProbeMatchSpec &ProbeMatchSpecs);

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

  /// Matches functions using the call graph.
  /// Populates BF->partial probe match spec map.
  size_t matchWithPseudoProbes(BinaryContext &BC);

  /// Matches functions with similarly named profiled functions.
  size_t matchWithNameSimilarity(BinaryContext &BC);

  /// Update matched YAML -> BinaryFunction pair.
  void matchProfileToFunction(yaml::bolt::BinaryFunctionProfile &YamlBF,
                              BinaryFunction &BF) {
    YamlProfileToFunction[YamlBF.Id] = &BF;
    YamlBF.Used = true;

    assert(!ProfiledFunctions.count(&BF) &&
           "function already has an assigned profile");
    ProfiledFunctions.emplace(&BF);
  }

  /// Return a top-level binary inline tree node for a given \p BF
  const MCDecodedPseudoProbeInlineTree *
  lookupTopLevelNode(const BinaryFunction &BF);

  /// Match up \p BF binary inline trees starting at root or \p Node if set, and
  /// \p YamlBF profile starting at \p NodeIdx and record the mapping in
  /// BFToProbeMatchSpecs.
  void matchInlineTrees(BinaryFunction &BF,
                        const MCDecodedPseudoProbeInlineTree *Node,
                        yaml::bolt::BinaryFunctionProfile &YamlBF,
                        uint32_t NodeIdx, float Scale);

  /// Check if the profile uses an event with a given \p Name.
  bool usesEvent(StringRef Name) const;
};

} // namespace bolt
} // namespace llvm

#endif
