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

/// An object wrapping several components of a basic block hash. The combined
/// (blended) hash is represented and stored as one uint64_t, while individual
/// components are of smaller size (e.g., uint16_t or uint8_t).
struct BlendedBlockHash {
private:
  using ValueOffset = Bitfield::Element<uint16_t, 0, 16>;
  using ValueOpcode = Bitfield::Element<uint16_t, 16, 16>;
  using ValueInstr = Bitfield::Element<uint16_t, 32, 16>;
  using ValuePred = Bitfield::Element<uint8_t, 48, 8>;
  using ValueSucc = Bitfield::Element<uint8_t, 56, 8>;

public:
  explicit BlendedBlockHash() {}

  explicit BlendedBlockHash(uint64_t Hash) {
    Offset = Bitfield::get<ValueOffset>(Hash);
    OpcodeHash = Bitfield::get<ValueOpcode>(Hash);
    InstrHash = Bitfield::get<ValueInstr>(Hash);
    PredHash = Bitfield::get<ValuePred>(Hash);
    SuccHash = Bitfield::get<ValueSucc>(Hash);
  }

  /// Combine the blended hash into uint64_t.
  uint64_t combine() const {
    uint64_t Hash = 0;
    Bitfield::set<ValueOffset>(Hash, Offset);
    Bitfield::set<ValueOpcode>(Hash, OpcodeHash);
    Bitfield::set<ValueInstr>(Hash, InstrHash);
    Bitfield::set<ValuePred>(Hash, PredHash);
    Bitfield::set<ValueSucc>(Hash, SuccHash);
    return Hash;
  }

  /// Compute a distance between two given blended hashes. The smaller the
  /// distance, the more similar two blocks are. For identical basic blocks,
  /// the distance is zero.
  uint64_t distance(const BlendedBlockHash &BBH) const {
    assert(OpcodeHash == BBH.OpcodeHash &&
           "incorrect blended hash distance computation");
    uint64_t Dist = 0;
    // Account for NeighborHash
    Dist += SuccHash == BBH.SuccHash ? 0 : 1;
    Dist += PredHash == BBH.PredHash ? 0 : 1;
    Dist <<= 16;
    // Account for InstrHash
    Dist += InstrHash == BBH.InstrHash ? 0 : 1;
    Dist <<= 16;
    // Account for Offset
    Dist += (Offset >= BBH.Offset ? Offset - BBH.Offset : BBH.Offset - Offset);
    return Dist;
  }

  /// The offset of the basic block from the function start.
  uint16_t Offset{0};
  /// (Loose) Hash of the basic block instructions, excluding operands.
  uint16_t OpcodeHash{0};
  /// (Strong) Hash of the basic block instructions, including opcodes and
  /// operands.
  uint16_t InstrHash{0};
  /// (Loose) Hashes of the predecessors of the basic block.
  uint8_t PredHash{0};
  /// (Loose) Hashes of the successors of the basic block.
  uint8_t SuccHash{0};
};

struct CallGraphMatcher {};

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
