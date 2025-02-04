//===- bolt/Profile/ProfileYAMLMapping.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement mapping between binary function profile and YAML representation.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_PROFILEYAMLMAPPING_H
#define BOLT_PROFILE_PROFILEYAMLMAPPING_H

#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/YAMLTraits.h"
#include <vector>

using llvm::bolt::BinaryFunction;

namespace llvm {
namespace yaml {

namespace bolt {
struct CallSiteInfo {
  llvm::yaml::Hex32 Offset{0};
  uint32_t DestId{0};
  uint32_t EntryDiscriminator{0}; /// multiple entry discriminator
  uint64_t Count{0};
  uint64_t Mispreds{0};

  bool operator==(const CallSiteInfo &Other) const {
    return Offset == Other.Offset && DestId == Other.DestId &&
           EntryDiscriminator == Other.EntryDiscriminator;
  }

  bool operator!=(const CallSiteInfo &Other) const { return !(*this == Other); }

  bool operator<(const CallSiteInfo &Other) const {
    if (Offset < Other.Offset)
      return true;
    if (Offset > Other.Offset)
      return false;

    if (DestId < Other.DestId)
      return true;
    if (DestId > Other.DestId)
      return false;

    if (EntryDiscriminator < Other.EntryDiscriminator)
      return true;

    return false;
  }
};
} // end namespace bolt

template <> struct MappingTraits<bolt::CallSiteInfo> {
  static void mapping(IO &YamlIO, bolt::CallSiteInfo &CSI) {
    YamlIO.mapRequired("off", CSI.Offset);
    YamlIO.mapRequired("fid", CSI.DestId);
    YamlIO.mapOptional("disc", CSI.EntryDiscriminator, (uint32_t)0);
    YamlIO.mapRequired("cnt", CSI.Count);
    YamlIO.mapOptional("mis", CSI.Mispreds, (uint64_t)0);
  }

  static const bool flow = true;
};

namespace bolt {
struct SuccessorInfo {
  uint32_t Index{0};
  uint64_t Count{0};
  uint64_t Mispreds{0};

  bool operator==(const SuccessorInfo &Other) const {
    return Index == Other.Index;
  }
  bool operator!=(const SuccessorInfo &Other) const {
    return !(*this == Other);
  }
};
} // end namespace bolt

template <> struct MappingTraits<bolt::SuccessorInfo> {
  static void mapping(IO &YamlIO, bolt::SuccessorInfo &SI) {
    YamlIO.mapRequired("bid", SI.Index);
    YamlIO.mapRequired("cnt", SI.Count);
    YamlIO.mapOptional("mis", SI.Mispreds, (uint64_t)0);
  }

  static const bool flow = true;
};

namespace bolt {
struct PseudoProbeInfo {
  uint32_t InlineTreeIndex = 0;
  uint64_t BlockMask = 0;            // bitset with probe indices from 1 to 64
  std::vector<uint64_t> BlockProbes; // block probes with indices above 64
  std::vector<uint64_t> CallProbes;
  std::vector<uint64_t> IndCallProbes;
  std::vector<uint32_t> InlineTreeNodes;

  bool operator==(const PseudoProbeInfo &Other) const {
    return InlineTreeIndex == Other.InlineTreeIndex &&
           BlockProbes == Other.BlockProbes && CallProbes == Other.CallProbes &&
           IndCallProbes == Other.IndCallProbes;
  }
};
} // end namespace bolt

template <> struct MappingTraits<bolt::PseudoProbeInfo> {
  static void mapping(IO &YamlIO, bolt::PseudoProbeInfo &PI) {
    YamlIO.mapOptional("blx", PI.BlockMask, 0);
    YamlIO.mapOptional("blk", PI.BlockProbes, std::vector<uint64_t>());
    YamlIO.mapOptional("call", PI.CallProbes, std::vector<uint64_t>());
    YamlIO.mapOptional("icall", PI.IndCallProbes, std::vector<uint64_t>());
    YamlIO.mapOptional("id", PI.InlineTreeIndex, 0);
    YamlIO.mapOptional("ids", PI.InlineTreeNodes, std::vector<uint32_t>());
  }

  static const bool flow = true;
};
} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(llvm::yaml::bolt::CallSiteInfo)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(llvm::yaml::bolt::SuccessorInfo)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(llvm::yaml::bolt::PseudoProbeInfo)

namespace llvm {
namespace yaml {

namespace bolt {
struct BinaryBasicBlockProfile {
  uint32_t Index{0};
  uint32_t NumInstructions{0};
  llvm::yaml::Hex64 Hash{0};
  uint64_t ExecCount{0};
  uint64_t EventCount{0};
  std::vector<CallSiteInfo> CallSites;
  std::vector<SuccessorInfo> Successors;
  std::vector<PseudoProbeInfo> PseudoProbes;

  bool operator==(const BinaryBasicBlockProfile &Other) const {
    return Index == Other.Index;
  }
  bool operator!=(const BinaryBasicBlockProfile &Other) const {
    return !(*this == Other);
  }
};
} // end namespace bolt

template <> struct MappingTraits<bolt::BinaryBasicBlockProfile> {
  static void mapping(IO &YamlIO, bolt::BinaryBasicBlockProfile &BBP) {
    YamlIO.mapRequired("bid", BBP.Index);
    YamlIO.mapRequired("insns", BBP.NumInstructions);
    YamlIO.mapOptional("hash", BBP.Hash, (llvm::yaml::Hex64)0);
    YamlIO.mapOptional("exec", BBP.ExecCount, (uint64_t)0);
    YamlIO.mapOptional("events", BBP.EventCount, (uint64_t)0);
    YamlIO.mapOptional("calls", BBP.CallSites,
                       std::vector<bolt::CallSiteInfo>());
    YamlIO.mapOptional("succ", BBP.Successors,
                       std::vector<bolt::SuccessorInfo>());
    YamlIO.mapOptional("probes", BBP.PseudoProbes,
                       std::vector<bolt::PseudoProbeInfo>());
  }
};

namespace bolt {
struct InlineTreeNode {
  uint32_t ParentIndexDelta;
  uint32_t CallSiteProbe;
  // Index in PseudoProbeDesc.GUID, UINT32_MAX for same as previous (omitted)
  uint32_t GUIDIndex;
  // Decoded contents, ParentIndexDelta becomes absolute value.
  uint64_t GUID;
  uint64_t Hash;
  bool operator==(const InlineTreeNode &) const { return false; }
};
} // end namespace bolt

template <> struct MappingTraits<bolt::InlineTreeNode> {
  static void mapping(IO &YamlIO, bolt::InlineTreeNode &ITI) {
    YamlIO.mapOptional("g", ITI.GUIDIndex, UINT32_MAX);
    YamlIO.mapOptional("p", ITI.ParentIndexDelta, 0);
    YamlIO.mapOptional("cs", ITI.CallSiteProbe, 0);
  }

  static const bool flow = true;
};
} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::yaml::bolt::BinaryBasicBlockProfile)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(llvm::yaml::bolt::InlineTreeNode)

namespace llvm {
namespace yaml {

namespace bolt {
struct BinaryFunctionProfile {
  std::string Name;
  uint32_t NumBasicBlocks{0};
  uint32_t Id{0};
  llvm::yaml::Hex64 Hash{0};
  uint64_t ExecCount{0};
  std::vector<BinaryBasicBlockProfile> Blocks;
  std::vector<InlineTreeNode> InlineTree;
  bool Used{false};
};
} // end namespace bolt

template <> struct MappingTraits<bolt::BinaryFunctionProfile> {
  static void mapping(IO &YamlIO, bolt::BinaryFunctionProfile &BFP) {
    YamlIO.mapRequired("name", BFP.Name);
    YamlIO.mapRequired("fid", BFP.Id);
    YamlIO.mapRequired("hash", BFP.Hash);
    YamlIO.mapRequired("exec", BFP.ExecCount);
    YamlIO.mapRequired("nblocks", BFP.NumBasicBlocks);
    YamlIO.mapOptional("blocks", BFP.Blocks,
                       std::vector<bolt::BinaryBasicBlockProfile>());
    YamlIO.mapOptional("inline_tree", BFP.InlineTree,
                       std::vector<bolt::InlineTreeNode>());
  }
};

LLVM_YAML_STRONG_TYPEDEF(uint16_t, PROFILE_PF)

template <> struct ScalarBitSetTraits<PROFILE_PF> {
  static void bitset(IO &io, PROFILE_PF &value) {
    io.bitSetCase(value, "lbr", BinaryFunction::PF_LBR);
    io.bitSetCase(value, "sample", BinaryFunction::PF_SAMPLE);
    io.bitSetCase(value, "memevent", BinaryFunction::PF_MEMEVENT);
  }
};

template <> struct ScalarEnumerationTraits<llvm::bolt::HashFunction> {
  using HashFunction = llvm::bolt::HashFunction;
  static void enumeration(IO &io, HashFunction &value) {
    io.enumCase(value, "std-hash", HashFunction::StdHash);
    io.enumCase(value, "xxh3", HashFunction::XXH3);
  }
};

namespace bolt {
struct BinaryProfileHeader {
  uint32_t Version{1};
  std::string FileName; // Name of the profiled binary.
  std::string Id;       // BuildID.
  PROFILE_PF Flags{BinaryFunction::PF_NONE};
  // Type of the profile.
  std::string Origin;     // How the profile was obtained.
  std::string EventNames; // Events used for sample profile.
  bool IsDFSOrder{true};  // Whether using DFS block order in function profile
  llvm::bolt::HashFunction HashFunction; // Hash used for BB/BF hashing
};
} // end namespace bolt

template <> struct MappingTraits<bolt::BinaryProfileHeader> {
  static void mapping(IO &YamlIO, bolt::BinaryProfileHeader &Header) {
    YamlIO.mapRequired("profile-version", Header.Version);
    YamlIO.mapRequired("binary-name", Header.FileName);
    YamlIO.mapOptional("binary-build-id", Header.Id);
    YamlIO.mapRequired("profile-flags", Header.Flags);
    YamlIO.mapOptional("profile-origin", Header.Origin);
    YamlIO.mapOptional("profile-events", Header.EventNames);
    YamlIO.mapOptional("dfs-order", Header.IsDFSOrder);
    YamlIO.mapOptional("hash-func", Header.HashFunction,
                       llvm::bolt::HashFunction::StdHash);
  }
};

namespace bolt {
struct ProfilePseudoProbeDesc {
  std::vector<Hex64> GUID;
  std::vector<Hex64> Hash;
  std::vector<uint32_t> GUIDHashIdx; // Index of hash for that GUID in Hash

  bool operator==(const ProfilePseudoProbeDesc &Other) const {
    // Only treat empty Desc as equal
    return GUID.empty() && Other.GUID.empty() && Hash.empty() &&
           Other.Hash.empty() && GUIDHashIdx.empty() &&
           Other.GUIDHashIdx.empty();
  }
};
} // end namespace bolt

template <> struct MappingTraits<bolt::ProfilePseudoProbeDesc> {
  static void mapping(IO &YamlIO, bolt::ProfilePseudoProbeDesc &PD) {
    YamlIO.mapRequired("gs", PD.GUID);
    YamlIO.mapRequired("gh", PD.GUIDHashIdx);
    YamlIO.mapRequired("hs", PD.Hash);
  }
};
} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::yaml::bolt::BinaryFunctionProfile)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::yaml::bolt::ProfilePseudoProbeDesc)

namespace llvm {
namespace yaml {

namespace bolt {
struct BinaryProfile {
  BinaryProfileHeader Header;
  std::vector<BinaryFunctionProfile> Functions;
  ProfilePseudoProbeDesc PseudoProbeDesc;
};
} // namespace bolt

template <> struct MappingTraits<bolt::BinaryProfile> {
  static void mapping(IO &YamlIO, bolt::BinaryProfile &BP) {
    YamlIO.mapRequired("header", BP.Header);
    YamlIO.mapRequired("functions", BP.Functions);
    YamlIO.mapOptional("pseudo_probe_desc", BP.PseudoProbeDesc,
                       bolt::ProfilePseudoProbeDesc());
  }
};

} // end namespace yaml
} // end namespace llvm

#endif
