//===- bolt/Profile/YAMLProfileWriter.h - Write profile in YAML -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_YAML_PROFILE_WRITER_H
#define BOLT_PROFILE_YAML_PROFILE_WRITER_H

#include "bolt/Profile/ProfileYAMLMapping.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

namespace llvm {
namespace bolt {
class BoltAddressTranslation;
class RewriteInstance;

class YAMLProfileWriter {
  YAMLProfileWriter() = delete;

  std::string Filename;

  std::unique_ptr<raw_fd_ostream> OS;

public:
  explicit YAMLProfileWriter(const std::string &Filename)
      : Filename(Filename) {}

  /// Save execution profile for that instance.
  std::error_code writeProfile(const RewriteInstance &RI);

  using InlineTreeMapTy =
      DenseMap<const MCDecodedPseudoProbeInlineTree *, uint32_t>;
  struct InlineTreeDesc {
    template <typename T> using GUIDMapTy = std::unordered_map<uint64_t, T>;
    using GUIDNodeMap = GUIDMapTy<const MCDecodedPseudoProbeInlineTree *>;
    using GUIDNumMap = GUIDMapTy<uint32_t>;
    GUIDNodeMap TopLevelGUIDToInlineTree;
    GUIDNumMap GUIDIdxMap;
    GUIDNumMap HashIdxMap;
  };

  static std::tuple<std::vector<yaml::bolt::InlineTreeNode>, InlineTreeMapTy>
  convertBFInlineTree(const MCPseudoProbeDecoder &Decoder,
                      const InlineTreeDesc &InlineTree, uint64_t GUID);

  static std::tuple<yaml::bolt::ProfilePseudoProbeDesc, InlineTreeDesc>
  convertPseudoProbeDesc(const MCPseudoProbeDecoder &PseudoProbeDecoder);

  static yaml::bolt::BinaryFunctionProfile
  convert(const BinaryFunction &BF, bool UseDFS,
          const InlineTreeDesc &InlineTree,
          const BoltAddressTranslation *BAT = nullptr);

  /// Set CallSiteInfo destination fields from \p Symbol and return a target
  /// BinaryFunction for that symbol.
  static const BinaryFunction *
  setCSIDestination(const BinaryContext &BC, yaml::bolt::CallSiteInfo &CSI,
                    const MCSymbol *Symbol, const BoltAddressTranslation *BAT,
                    uint32_t Offset = 0);

private:
  struct InlineTreeNode {
    const MCDecodedPseudoProbeInlineTree *InlineTree;
    uint64_t GUID;
    uint64_t Hash;
    uint32_t ParentId;
    uint32_t InlineSite;
  };
  static std::vector<InlineTreeNode>
  collectInlineTree(const MCPseudoProbeDecoder &Decoder,
                    const MCDecodedPseudoProbeInlineTree &Root);

  // 0 - block probe, 1 - indirect call, 2 - direct call
  using ProbeList = std::array<SmallVector<uint64_t, 0>, 3>;
  using NodeIdToProbes = DenseMap<uint32_t, ProbeList>;
  static std::vector<yaml::bolt::PseudoProbeInfo>
  convertNodeProbes(NodeIdToProbes &NodeProbes);

public:
  template <typename T>
  static std::vector<yaml::bolt::PseudoProbeInfo>
  writeBlockProbes(T Probes, const InlineTreeMapTy &InlineTreeNodeId) {
    NodeIdToProbes NodeProbes;
    for (const MCDecodedPseudoProbe &Probe : Probes) {
      auto It = InlineTreeNodeId.find(Probe.getInlineTreeNode());
      if (It == InlineTreeNodeId.end())
        continue;
      NodeProbes[It->second][Probe.getType()].emplace_back(Probe.getIndex());
    }
    return convertNodeProbes(NodeProbes);
  }
};
} // namespace bolt
} // namespace llvm

#endif
