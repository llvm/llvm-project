//===- bolt/Profile/StaleProfileMatching.cpp - Profile data matching   ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BOLT often has to deal with profiles collected on binaries built from several
// revisions behind release. As a result, a certain percentage of functions is
// considered stale and not optimized. This file implements an ability to match
// profile to functions that are not 100% binary identical, and thus, increasing
// the optimization coverage and boost the performance of applications.
//
// The algorithm consists of two phases: matching and inference:
// - At the matching phase, we try to "guess" as many block and jump counts from
//   the stale profile as possible. To this end, the content of each basic block
//   is hashed and stored in the (yaml) profile. When BOLT optimizes a binary,
//   it computes block hashes and identifies the corresponding entries in the
//   stale profile. It yields a partial profile for every CFG in the binary.
// - At the inference phase, we employ a network flow-based algorithm (profi) to
//   reconstruct "realistic" block and jump counts from the partial profile
//   generated at the first stage. In practice, we don't always produce proper
//   profile data but the majority (e.g., >90%) of CFGs get the correct counts.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/HashUtilities.h"
#include "bolt/Profile/YAMLProfileReader.h"
#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Transforms/Utils/SampleProfileInference.h"

#include <queue>

using namespace llvm;

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

namespace opts {

extern cl::opt<bool> TimeRewrite;
extern cl::OptionCategory BoltOptCategory;

cl::opt<bool>
    InferStaleProfile("infer-stale-profile",
                      cl::desc("Infer counts from stale profile data."),
                      cl::init(false), cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingMinMatchedBlock(
    "stale-matching-min-matched-block",
    cl::desc("Percentage threshold of matched basic blocks at which stale "
             "profile inference is executed."),
    cl::init(0), cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingMaxFuncSize(
    "stale-matching-max-func-size",
    cl::desc("The maximum size of a function to consider for inference."),
    cl::init(10000), cl::Hidden, cl::cat(BoltOptCategory));

// Parameters of the profile inference algorithm. The default values are tuned
// on several benchmarks.
cl::opt<bool> StaleMatchingEvenFlowDistribution(
    "stale-matching-even-flow-distribution",
    cl::desc("Try to evenly distribute flow when there are multiple equally "
             "likely options."),
    cl::init(true), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<bool> StaleMatchingRebalanceUnknown(
    "stale-matching-rebalance-unknown",
    cl::desc("Evenly re-distribute flow among unknown subgraphs."),
    cl::init(false), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<bool> StaleMatchingJoinIslands(
    "stale-matching-join-islands",
    cl::desc("Join isolated components having positive flow."), cl::init(true),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockInc(
    "stale-matching-cost-block-inc",
    cl::desc("The cost of increasing a block count by one."), cl::init(150),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockDec(
    "stale-matching-cost-block-dec",
    cl::desc("The cost of decreasing a block count by one."), cl::init(150),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpInc(
    "stale-matching-cost-jump-inc",
    cl::desc("The cost of increasing a jump count by one."), cl::init(150),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpDec(
    "stale-matching-cost-jump-dec",
    cl::desc("The cost of decreasing a jump count by one."), cl::init(150),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockUnknownInc(
    "stale-matching-cost-block-unknown-inc",
    cl::desc("The cost of increasing an unknown block count by one."),
    cl::init(1), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpUnknownInc(
    "stale-matching-cost-jump-unknown-inc",
    cl::desc("The cost of increasing an unknown jump count by one."),
    cl::init(140), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpUnknownFTInc(
    "stale-matching-cost-jump-unknown-ft-inc",
    cl::desc(
        "The cost of increasing an unknown fall-through jump count by one."),
    cl::init(3), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<bool> StaleMatchingWithPseudoProbes(
    "stale-matching-with-pseudo-probes",
    cl::desc("Turns on stale matching with block pseudo probes."),
    cl::init(false), cl::ReallyHidden, cl::cat(BoltOptCategory));

} // namespace opts

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

/// The object is used to identify and match basic blocks in a BinaryFunction
/// given their hashes computed on a binary built from several revisions behind
/// release.
class StaleMatcher {
public:
  /// Initialize stale matcher.
  void init(const std::vector<FlowBlock *> &Blocks,
            const std::vector<BlendedBlockHash> &Hashes,
            const std::vector<uint64_t> &CallHashes) {
    assert(Blocks.size() == Hashes.size() &&
           Hashes.size() == CallHashes.size() &&
           "incorrect matcher initialization");
    for (size_t I = 0; I < Blocks.size(); I++) {
      FlowBlock *Block = Blocks[I];
      uint16_t OpHash = Hashes[I].OpcodeHash;
      OpHashToBlocks[OpHash].push_back(std::make_pair(Hashes[I], Block));
      if (CallHashes[I])
        CallHashToBlocks[CallHashes[I]].push_back(
            std::make_pair(Hashes[I], Block));
    }
  }

  /// Creates a mapping from a pseudo probe to a flow block.
  void mapProbeToBB(const MCDecodedPseudoProbe *Probe, FlowBlock *Block) {
    BBPseudoProbeToBlock[Probe] = Block;
  }

  enum MatchMethod : char {
    MATCH_EXACT = 0,
    MATCH_PROBE_EXACT,
    MATCH_PROBE_LOOSE,
    MATCH_OPCODE,
    MATCH_CALL,
    NO_MATCH
  };

  /// Find the most similar flow block for a profile block given blended hash.
  std::pair<const FlowBlock *, MatchMethod>
  matchBlockStrict(BlendedBlockHash BlendedHash) {
    const auto &[Block, ExactHash] = matchWithOpcodes(BlendedHash);
    if (Block && ExactHash)
      return {Block, MATCH_EXACT};
    return {nullptr, NO_MATCH};
  }

  /// Find the most similar flow block for a profile block given pseudo probes.
  std::pair<const FlowBlock *, MatchMethod> matchBlockProbe(
      const ArrayRef<yaml::bolt::PseudoProbeInfo> PseudoProbes,
      const YAMLProfileReader::InlineTreeNodeMapTy &InlineTreeNodeMap) {
    const auto &[ProbeBlock, ExactProbe] =
        matchWithPseudoProbes(PseudoProbes, InlineTreeNodeMap);
    if (ProbeBlock)
      return {ProbeBlock, ExactProbe ? MATCH_PROBE_EXACT : MATCH_PROBE_LOOSE};
    return {nullptr, NO_MATCH};
  }

  /// Find the most similar flow block for a profile block given its hashes.
  std::pair<const FlowBlock *, MatchMethod>
  matchBlockLoose(BlendedBlockHash BlendedHash, uint64_t CallHash) {
    if (const FlowBlock *CallBlock = matchWithCalls(BlendedHash, CallHash))
      return {CallBlock, MATCH_CALL};
    if (const FlowBlock *OpcodeBlock = matchWithOpcodes(BlendedHash).first)
      return {OpcodeBlock, MATCH_OPCODE};
    return {nullptr, NO_MATCH};
  }

  /// Returns true if the two basic blocks (in the binary and in the profile)
  /// corresponding to the given hashes are matched to each other with a high
  /// confidence.
  static bool isHighConfidenceMatch(BlendedBlockHash Hash1,
                                    BlendedBlockHash Hash2) {
    return Hash1.InstrHash == Hash2.InstrHash;
  }

private:
  using HashBlockPairType = std::pair<BlendedBlockHash, FlowBlock *>;
  std::unordered_map<uint16_t, std::vector<HashBlockPairType>> OpHashToBlocks;
  std::unordered_map<uint64_t, std::vector<HashBlockPairType>> CallHashToBlocks;
  DenseMap<const MCDecodedPseudoProbe *, FlowBlock *> BBPseudoProbeToBlock;

  // Uses OpcodeHash to find the most similar block for a given hash.
  std::pair<const FlowBlock *, bool>
  matchWithOpcodes(BlendedBlockHash BlendedHash) const {
    auto BlockIt = OpHashToBlocks.find(BlendedHash.OpcodeHash);
    if (BlockIt == OpHashToBlocks.end())
      return {nullptr, false};
    FlowBlock *BestBlock = nullptr;
    uint64_t BestDist = std::numeric_limits<uint64_t>::max();
    BlendedBlockHash BestHash;
    for (const auto &[Hash, Block] : BlockIt->second) {
      uint64_t Dist = Hash.distance(BlendedHash);
      if (BestBlock == nullptr || Dist < BestDist) {
        BestDist = Dist;
        BestBlock = Block;
        BestHash = Hash;
      }
    }
    return {BestBlock, isHighConfidenceMatch(BestHash, BlendedHash)};
  }

  // Uses CallHash to find the most similar block for a given hash.
  const FlowBlock *matchWithCalls(BlendedBlockHash BlendedHash,
                                  uint64_t CallHash) const {
    if (!CallHash)
      return nullptr;
    auto BlockIt = CallHashToBlocks.find(CallHash);
    if (BlockIt == CallHashToBlocks.end())
      return nullptr;
    FlowBlock *BestBlock = nullptr;
    uint64_t BestDist = std::numeric_limits<uint64_t>::max();
    for (const auto &[Hash, Block] : BlockIt->second) {
      uint64_t Dist = Hash.OpcodeHash > BlendedHash.OpcodeHash
                          ? Hash.OpcodeHash - BlendedHash.OpcodeHash
                          : BlendedHash.OpcodeHash - Hash.OpcodeHash;
      if (BestBlock == nullptr || Dist < BestDist) {
        BestDist = Dist;
        BestBlock = Block;
      }
    }
    return BestBlock;
  }

  /// Matches a profile block with a binary block based on pseudo probes.
  /// Returns the best matching block (or nullptr) and whether the match is
  /// unambiguous.
  std::pair<const FlowBlock *, bool> matchWithPseudoProbes(
      const ArrayRef<yaml::bolt::PseudoProbeInfo> BlockPseudoProbes,
      const YAMLProfileReader::InlineTreeNodeMapTy &InlineTreeNodeMap) const {

    if (!opts::StaleMatchingWithPseudoProbes)
      return {nullptr, false};

    DenseMap<const FlowBlock *, uint32_t> FlowBlockMatchCount;

    auto matchProfileProbeToBlock = [&](uint32_t NodeId,
                                        uint64_t ProbeId) -> const FlowBlock * {
      const MCDecodedPseudoProbeInlineTree *BinaryNode =
          InlineTreeNodeMap.getInlineTreeNode(NodeId);
      if (!BinaryNode)
        return nullptr;
      const MCDecodedPseudoProbe Dummy(0, ProbeId, PseudoProbeType::Block, 0, 0,
                                       nullptr);
      ArrayRef<MCDecodedPseudoProbe> BinaryProbes = BinaryNode->getProbes();
      auto BinaryProbeIt = llvm::lower_bound(
          BinaryProbes, Dummy, [](const auto &LHS, const auto &RHS) {
            return LHS.getIndex() < RHS.getIndex();
          });
      if (BinaryProbeIt == BinaryNode->getProbes().end() ||
          BinaryProbeIt->getIndex() != ProbeId)
        return nullptr;
      auto It = BBPseudoProbeToBlock.find(&*BinaryProbeIt);
      if (It == BBPseudoProbeToBlock.end())
        return nullptr;
      return It->second;
    };

    auto matchPseudoProbeInfo = [&](const yaml::bolt::PseudoProbeInfo
                                        &ProfileProbe,
                                    uint32_t NodeId) {
      for (uint64_t Index = 0; Index < 64; ++Index)
        if (ProfileProbe.BlockMask & 1ull << Index)
          ++FlowBlockMatchCount[matchProfileProbeToBlock(NodeId, Index + 1)];
      for (const auto &ProfileProbes :
           {ProfileProbe.BlockProbes, ProfileProbe.IndCallProbes,
            ProfileProbe.CallProbes})
        for (uint64_t ProfileProbe : ProfileProbes)
          ++FlowBlockMatchCount[matchProfileProbeToBlock(NodeId, ProfileProbe)];
    };

    for (const yaml::bolt::PseudoProbeInfo &ProfileProbe : BlockPseudoProbes) {
      if (!ProfileProbe.InlineTreeNodes.empty())
        for (uint32_t ProfileInlineTreeNode : ProfileProbe.InlineTreeNodes)
          matchPseudoProbeInfo(ProfileProbe, ProfileInlineTreeNode);
      else
        matchPseudoProbeInfo(ProfileProbe, ProfileProbe.InlineTreeIndex);
    }
    uint32_t BestMatchCount = 0;
    uint32_t TotalMatchCount = 0;
    const FlowBlock *BestMatchBlock = nullptr;
    for (const auto &[FlowBlock, Count] : FlowBlockMatchCount) {
      TotalMatchCount += Count;
      if (Count < BestMatchCount || (Count == BestMatchCount && BestMatchBlock))
        continue;
      BestMatchBlock = FlowBlock;
      BestMatchCount = Count;
    }
    return {BestMatchBlock, BestMatchCount == TotalMatchCount};
  }
};

void BinaryFunction::computeBlockHashes(HashFunction HashFunction) const {
  if (size() == 0)
    return;

  assert(hasCFG() && "the function is expected to have CFG");

  std::vector<BlendedBlockHash> BlendedHashes(BasicBlocks.size());
  std::vector<uint64_t> OpcodeHashes(BasicBlocks.size());
  // Initialize hash components.
  for (size_t I = 0; I < BasicBlocks.size(); I++) {
    const BinaryBasicBlock *BB = BasicBlocks[I];
    assert(BB->getIndex() == I && "incorrect block index");
    BlendedHashes[I].Offset = BB->getOffset();
    // Hashing complete instructions.
    std::string InstrHashStr = hashBlock(
        BC, *BB, [&](const MCOperand &Op) { return hashInstOperand(BC, Op); });
    if (HashFunction == HashFunction::StdHash) {
      uint64_t InstrHash = std::hash<std::string>{}(InstrHashStr);
      BlendedHashes[I].InstrHash = (uint16_t)hash_value(InstrHash);
    } else if (HashFunction == HashFunction::XXH3) {
      uint64_t InstrHash = llvm::xxh3_64bits(InstrHashStr);
      BlendedHashes[I].InstrHash = (uint16_t)InstrHash;
    } else {
      llvm_unreachable("Unhandled HashFunction");
    }
    // Hashing opcodes.
    std::string OpcodeHashStr = hashBlockLoose(BC, *BB);
    if (HashFunction == HashFunction::StdHash) {
      OpcodeHashes[I] = std::hash<std::string>{}(OpcodeHashStr);
      BlendedHashes[I].OpcodeHash = (uint16_t)hash_value(OpcodeHashes[I]);
    } else if (HashFunction == HashFunction::XXH3) {
      OpcodeHashes[I] = llvm::xxh3_64bits(OpcodeHashStr);
      BlendedHashes[I].OpcodeHash = (uint16_t)OpcodeHashes[I];
    } else {
      llvm_unreachable("Unhandled HashFunction");
    }
  }

  // Initialize neighbor hash.
  for (size_t I = 0; I < BasicBlocks.size(); I++) {
    const BinaryBasicBlock *BB = BasicBlocks[I];
    // Append hashes of successors.
    uint64_t Hash = 0;
    for (BinaryBasicBlock *SuccBB : BB->successors()) {
      uint64_t SuccHash = OpcodeHashes[SuccBB->getIndex()];
      Hash = hashing::detail::hash_16_bytes(Hash, SuccHash);
    }
    if (HashFunction == HashFunction::StdHash) {
      // Compatibility with old behavior.
      BlendedHashes[I].SuccHash = (uint8_t)hash_value(Hash);
    } else {
      BlendedHashes[I].SuccHash = (uint8_t)Hash;
    }

    // Append hashes of predecessors.
    Hash = 0;
    for (BinaryBasicBlock *PredBB : BB->predecessors()) {
      uint64_t PredHash = OpcodeHashes[PredBB->getIndex()];
      Hash = hashing::detail::hash_16_bytes(Hash, PredHash);
    }
    if (HashFunction == HashFunction::StdHash) {
      // Compatibility with old behavior.
      BlendedHashes[I].PredHash = (uint8_t)hash_value(Hash);
    } else {
      BlendedHashes[I].PredHash = (uint8_t)Hash;
    }
  }

  //  Assign hashes.
  for (size_t I = 0; I < BasicBlocks.size(); I++) {
    const BinaryBasicBlock *BB = BasicBlocks[I];
    BB->setHash(BlendedHashes[I].combine());
  }
}
// TODO: mediate the difference between flow function construction here in BOLT
// and in the compiler by splitting blocks with exception throwing calls at the
// call and adding the landing pad as the successor.
/// Create a wrapper flow function to use with the profile inference algorithm,
/// and initialize its jumps and metadata.
FlowFunction
createFlowFunction(const BinaryFunction::BasicBlockOrderType &BlockOrder) {
  FlowFunction Func;

  // Add a special "dummy" source so that there is always a unique entry point.
  FlowBlock EntryBlock;
  EntryBlock.Index = 0;
  Func.Blocks.push_back(EntryBlock);

  // Create FlowBlock for every basic block in the binary function.
  for (const BinaryBasicBlock *BB : BlockOrder) {
    Func.Blocks.emplace_back();
    FlowBlock &Block = Func.Blocks.back();
    Block.Index = Func.Blocks.size() - 1;
    (void)BB;
    assert(Block.Index == BB->getIndex() + 1 &&
           "incorrectly assigned basic block index");
  }

  // Add a special "dummy" sink block so there is always a unique sink.
  FlowBlock SinkBlock;
  SinkBlock.Index = Func.Blocks.size();
  Func.Blocks.push_back(SinkBlock);

  // Create FlowJump for each jump between basic blocks in the binary function.
  std::vector<uint64_t> InDegree(Func.Blocks.size(), 0);
  for (const BinaryBasicBlock *SrcBB : BlockOrder) {
    std::unordered_set<const BinaryBasicBlock *> UniqueSuccs;
    // Collect regular jumps
    for (const BinaryBasicBlock *DstBB : SrcBB->successors()) {
      // Ignoring parallel edges
      if (UniqueSuccs.find(DstBB) != UniqueSuccs.end())
        continue;

      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = SrcBB->getIndex() + 1;
      Jump.Target = DstBB->getIndex() + 1;
      InDegree[Jump.Target]++;
      UniqueSuccs.insert(DstBB);
    }
    // TODO: set jump from exit block to landing pad to Unlikely.
    // If the block is an exit, add a dummy edge from it to the sink block.
    if (UniqueSuccs.empty()) {
      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = SrcBB->getIndex() + 1;
      Jump.Target = Func.Blocks.size() - 1;
      InDegree[Jump.Target]++;
    }

    // Collect jumps to landing pads
    for (const BinaryBasicBlock *DstBB : SrcBB->landing_pads()) {
      // Ignoring parallel edges
      if (UniqueSuccs.find(DstBB) != UniqueSuccs.end())
        continue;

      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = SrcBB->getIndex() + 1;
      Jump.Target = DstBB->getIndex() + 1;
      InDegree[Jump.Target]++;
      UniqueSuccs.insert(DstBB);
    }
  }

  // Add dummy edges to the extra sources. If there are multiple entry blocks,
  // add an unlikely edge from 0 to the subsequent ones. Skips the sink block.
  assert(InDegree[0] == 0 && "dummy entry blocks shouldn't have predecessors");
  for (uint64_t I = 1; I < Func.Blocks.size() - 1; I++) {
    const BinaryBasicBlock *BB = BlockOrder[I - 1];
    if (BB->isEntryPoint() || InDegree[I] == 0) {
      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = 0;
      Jump.Target = I;
      if (!BB->isEntryPoint())
        Jump.IsUnlikely = true;
    }
  }

  // Create necessary metadata for the flow function
  for (FlowJump &Jump : Func.Jumps) {
    assert(Jump.Source < Func.Blocks.size());
    Func.Blocks[Jump.Source].SuccJumps.push_back(&Jump);
    assert(Jump.Target < Func.Blocks.size());
    Func.Blocks[Jump.Target].PredJumps.push_back(&Jump);
  }
  return Func;
}

/// Assign initial block/jump weights based on the stale profile data. The goal
/// is to extract as much information from the stale profile as possible. Here
/// we assume that each basic block is specified via a hash value computed from
/// its content and the hashes of the unchanged basic blocks stay the same
/// across different revisions of the binary. Blocks may also have pseudo probe
/// information in the profile and the binary which is used for matching.
/// Whenever there is a count in the profile with the hash corresponding to one
/// of the basic blocks in the binary, the count is "matched" to the block.
/// Similarly, if both the source and the target of a count in the profile are
/// matched to a jump in the binary, the count is recorded in CFG.
size_t matchWeights(
    BinaryContext &BC, const BinaryFunction::BasicBlockOrderType &BlockOrder,
    const yaml::bolt::BinaryFunctionProfile &YamlBF, FlowFunction &Func,
    HashFunction HashFunction, YAMLProfileReader::ProfileLookupMap &IdToYamlBF,
    const BinaryFunction &BF,
    const ArrayRef<YAMLProfileReader::ProbeMatchSpec> ProbeMatchSpecs) {

  assert(Func.Blocks.size() == BlockOrder.size() + 2);

  std::vector<uint64_t> CallHashes;
  std::vector<FlowBlock *> Blocks;
  std::vector<BlendedBlockHash> BlendedHashes;
  for (uint64_t I = 0; I < BlockOrder.size(); I++) {
    const BinaryBasicBlock *BB = BlockOrder[I];
    assert(BB->getHash() != 0 && "empty hash of BinaryBasicBlock");

    std::string CallHashStr = hashBlockCalls(BC, *BB);
    if (CallHashStr.empty()) {
      CallHashes.push_back(0);
    } else {
      if (HashFunction == HashFunction::StdHash)
        CallHashes.push_back(std::hash<std::string>{}(CallHashStr));
      else if (HashFunction == HashFunction::XXH3)
        CallHashes.push_back(llvm::xxh3_64bits(CallHashStr));
      else
        llvm_unreachable("Unhandled HashFunction");
    }

    Blocks.push_back(&Func.Blocks[I + 1]);
    BlendedBlockHash BlendedHash(BB->getHash());
    BlendedHashes.push_back(BlendedHash);
    LLVM_DEBUG(dbgs() << "BB with index " << I << " has hash = "
                      << Twine::utohexstr(BB->getHash()) << "\n");
  }
  StaleMatcher Matcher;
  // Collects function pseudo probes for use in the StaleMatcher.
  if (opts::StaleMatchingWithPseudoProbes) {
    const MCPseudoProbeDecoder *Decoder = BC.getPseudoProbeDecoder();
    assert(Decoder &&
           "If pseudo probes are in use, pseudo probe decoder should exist");
    const AddressProbesMap &ProbeMap = Decoder->getAddress2ProbesMap();
    const uint64_t FuncAddr = BF.getAddress();
    for (const MCDecodedPseudoProbe &Probe :
         ProbeMap.find(FuncAddr, FuncAddr + BF.getSize()))
      if (const BinaryBasicBlock *BB =
              BF.getBasicBlockContainingOffset(Probe.getAddress() - FuncAddr))
        Matcher.mapProbeToBB(&Probe, Blocks[BB->getIndex()]);
  }
  Matcher.init(Blocks, BlendedHashes, CallHashes);

  using FlowBlockTy =
      std::pair<const FlowBlock *, const yaml::bolt::BinaryBasicBlockProfile *>;
  using ProfileBlockMatchMap = DenseMap<uint32_t, FlowBlockTy>;
  // Binary profile => block index => matched block + its block profile
  DenseMap<const yaml::bolt::BinaryFunctionProfile *, ProfileBlockMatchMap>
      MatchedBlocks;

  // Map of FlowBlock and matching method.
  DenseMap<const FlowBlock *, StaleMatcher::MatchMethod> MatchedFlowBlocks;

  auto addMatchedBlock =
      [&](std::pair<const FlowBlock *, StaleMatcher::MatchMethod> BlockMethod,
          const yaml::bolt::BinaryFunctionProfile &YamlBP,
          const yaml::bolt::BinaryBasicBlockProfile &YamlBB) {
        const auto &[MatchedBlock, Method] = BlockMethod;
        if (!MatchedBlock)
          return;
        // Don't override earlier matches
        if (MatchedFlowBlocks.contains(MatchedBlock))
          return;
        MatchedFlowBlocks.try_emplace(MatchedBlock, Method);
        MatchedBlocks[&YamlBP][YamlBB.Index] = {MatchedBlock, &YamlBB};
      };

  // Match blocks from the profile to the blocks in CFG by strict hash.
  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks) {
    // Update matching stats.
    ++BC.Stats.NumStaleBlocks;
    BC.Stats.StaleSampleCount += YamlBB.ExecCount;

    assert(YamlBB.Hash != 0 && "empty hash of BinaryBasicBlockProfile");
    BlendedBlockHash YamlHash(YamlBB.Hash);
    addMatchedBlock(Matcher.matchBlockStrict(YamlHash), YamlBF, YamlBB);
  }
  // Match blocks from the profile to the blocks in CFG by pseudo probes.
  for (const auto &[InlineNodeMap, YamlBP] : ProbeMatchSpecs) {
    for (const yaml::bolt::BinaryBasicBlockProfile &BB : YamlBP.get().Blocks)
      if (!BB.PseudoProbes.empty())
        addMatchedBlock(Matcher.matchBlockProbe(BB.PseudoProbes, InlineNodeMap),
                        YamlBP, BB);
  }
  // Match blocks from the profile to the blocks in CFG with loose methods.
  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks) {
    assert(YamlBB.Hash != 0 && "empty hash of BinaryBasicBlockProfile");
    BlendedBlockHash YamlHash(YamlBB.Hash);

    std::string CallHashStr = hashBlockCalls(IdToYamlBF, YamlBB);
    uint64_t CallHash = 0;
    if (!CallHashStr.empty()) {
      if (HashFunction == HashFunction::StdHash)
        CallHash = std::hash<std::string>{}(CallHashStr);
      else if (HashFunction == HashFunction::XXH3)
        CallHash = llvm::xxh3_64bits(CallHashStr);
      else
        llvm_unreachable("Unhandled HashFunction");
    }
    auto [MatchedBlock, Method] = Matcher.matchBlockLoose(YamlHash, CallHash);
    if (MatchedBlock == nullptr && YamlBB.Index == 0) {
      MatchedBlock = Blocks[0];
      // Report as loose match
      Method = StaleMatcher::MATCH_OPCODE;
    }
    if (!MatchedBlock) {
      LLVM_DEBUG(dbgs() << "Couldn't match yaml block (bid = " << YamlBB.Index
                        << ")" << " with hash " << Twine::utohexstr(YamlBB.Hash)
                        << "\n");
      continue;
    }
    addMatchedBlock({MatchedBlock, Method}, YamlBF, YamlBB);
  }

  // Match jumps from the profile to the jumps from CFG
  std::vector<uint64_t> OutWeight(Func.Blocks.size(), 0);
  std::vector<uint64_t> InWeight(Func.Blocks.size(), 0);

  for (const auto &[YamlBF, MatchMap] : MatchedBlocks) {
    for (const auto &[YamlBBIdx, FlowBlockProfile] : MatchMap) {
      const auto &[MatchedBlock, YamlBB] = FlowBlockProfile;
      StaleMatcher::MatchMethod Method = MatchedFlowBlocks.lookup(MatchedBlock);
      BlendedBlockHash BinHash = BlendedHashes[MatchedBlock->Index - 1];
      LLVM_DEBUG(dbgs() << "Matched yaml block (bid = " << YamlBBIdx << ")"
                        << " with hash " << Twine::utohexstr(YamlBB->Hash)
                        << " to BB (index = " << MatchedBlock->Index - 1 << ")"
                        << " with hash " << Twine::utohexstr(BinHash.combine())
                        << "\n");
      (void)BinHash;
      uint64_t ExecCount = YamlBB->ExecCount;
      // Update matching stats accounting for the matched block.
      switch (Method) {
      case StaleMatcher::MATCH_EXACT:
        ++BC.Stats.NumExactMatchedBlocks;
        BC.Stats.ExactMatchedSampleCount += ExecCount;
        LLVM_DEBUG(dbgs() << "  exact match\n");
        break;
      case StaleMatcher::MATCH_PROBE_EXACT:
        ++BC.Stats.NumPseudoProbeExactMatchedBlocks;
        BC.Stats.PseudoProbeExactMatchedSampleCount += ExecCount;
        LLVM_DEBUG(dbgs() << "  exact pseudo probe match\n");
        break;
      case StaleMatcher::MATCH_PROBE_LOOSE:
        ++BC.Stats.NumPseudoProbeLooseMatchedBlocks;
        BC.Stats.PseudoProbeLooseMatchedSampleCount += ExecCount;
        LLVM_DEBUG(dbgs() << "  loose pseudo probe match\n");
        break;
      case StaleMatcher::MATCH_CALL:
        ++BC.Stats.NumCallMatchedBlocks;
        BC.Stats.CallMatchedSampleCount += ExecCount;
        LLVM_DEBUG(dbgs() << "  call match\n");
        break;
      case StaleMatcher::MATCH_OPCODE:
        ++BC.Stats.NumLooseMatchedBlocks;
        BC.Stats.LooseMatchedSampleCount += ExecCount;
        LLVM_DEBUG(dbgs() << "  loose match\n");
        break;
      case StaleMatcher::NO_MATCH:
        LLVM_DEBUG(dbgs() << "  no match\n");
      }
    }

    for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF->Blocks) {
      for (const yaml::bolt::SuccessorInfo &YamlSI : YamlBB.Successors) {
        if (YamlSI.Count == 0)
          continue;

        // Try to find the jump for a given (src, dst) pair from the profile and
        // assign the jump weight based on the profile count
        const uint64_t SrcIndex = YamlBB.Index;
        const uint64_t DstIndex = YamlSI.Index;

        const FlowBlock *MatchedSrcBlock = MatchMap.lookup(SrcIndex).first;
        const FlowBlock *MatchedDstBlock = MatchMap.lookup(DstIndex).first;

        if (MatchedSrcBlock != nullptr && MatchedDstBlock != nullptr) {
          // Find a jump between the two blocks
          FlowJump *Jump = nullptr;
          for (FlowJump *SuccJump : MatchedSrcBlock->SuccJumps) {
            if (SuccJump->Target == MatchedDstBlock->Index) {
              Jump = SuccJump;
              break;
            }
          }
          // Assign the weight, if the corresponding jump is found
          if (Jump != nullptr) {
            Jump->Weight = YamlSI.Count;
            Jump->HasUnknownWeight = false;
          }
        }
        // Assign the weight for the src block, if it is found
        if (MatchedSrcBlock != nullptr)
          OutWeight[MatchedSrcBlock->Index] += YamlSI.Count;
        // Assign the weight for the dst block, if it is found
        if (MatchedDstBlock != nullptr)
          InWeight[MatchedDstBlock->Index] += YamlSI.Count;
      }
    }
  }

  // Assign block counts based on in-/out- jumps
  for (FlowBlock &Block : Func.Blocks) {
    if (OutWeight[Block.Index] == 0 && InWeight[Block.Index] == 0) {
      assert(Block.HasUnknownWeight && "unmatched block with a positive count");
      continue;
    }
    Block.HasUnknownWeight = false;
    Block.Weight = std::max(OutWeight[Block.Index], InWeight[Block.Index]);
  }

  return MatchedBlocks[&YamlBF].size();
}

/// The function finds all blocks that are (i) reachable from the Entry block
/// and (ii) do not have a path to an exit, and marks all such blocks 'cold'
/// so that profi does not send any flow to such blocks.
void preprocessUnreachableBlocks(FlowFunction &Func) {
  const uint64_t NumBlocks = Func.Blocks.size();

  // Start bfs from the source
  std::queue<uint64_t> Queue;
  std::vector<bool> VisitedEntry(NumBlocks, false);
  for (uint64_t I = 0; I < NumBlocks; I++) {
    FlowBlock &Block = Func.Blocks[I];
    if (Block.isEntry()) {
      Queue.push(I);
      VisitedEntry[I] = true;
      break;
    }
  }
  while (!Queue.empty()) {
    const uint64_t Src = Queue.front();
    Queue.pop();
    for (FlowJump *Jump : Func.Blocks[Src].SuccJumps) {
      const uint64_t Dst = Jump->Target;
      if (!VisitedEntry[Dst]) {
        Queue.push(Dst);
        VisitedEntry[Dst] = true;
      }
    }
  }

  // Start bfs from all sinks
  std::vector<bool> VisitedExit(NumBlocks, false);
  for (uint64_t I = 0; I < NumBlocks; I++) {
    FlowBlock &Block = Func.Blocks[I];
    if (Block.isExit() && VisitedEntry[I]) {
      Queue.push(I);
      VisitedExit[I] = true;
    }
  }
  while (!Queue.empty()) {
    const uint64_t Src = Queue.front();
    Queue.pop();
    for (FlowJump *Jump : Func.Blocks[Src].PredJumps) {
      const uint64_t Dst = Jump->Source;
      if (!VisitedExit[Dst]) {
        Queue.push(Dst);
        VisitedExit[Dst] = true;
      }
    }
  }

  // Make all blocks of zero weight so that flow is not sent
  for (uint64_t I = 0; I < NumBlocks; I++) {
    FlowBlock &Block = Func.Blocks[I];
    if (Block.Weight == 0)
      continue;
    if (!VisitedEntry[I] || !VisitedExit[I]) {
      Block.Weight = 0;
      Block.HasUnknownWeight = true;
      Block.IsUnlikely = true;
      for (FlowJump *Jump : Block.SuccJumps) {
        if (Jump->Source == Block.Index && Jump->Target == Block.Index) {
          Jump->Weight = 0;
          Jump->HasUnknownWeight = true;
          Jump->IsUnlikely = true;
        }
      }
    }
  }
}

/// Decide if stale profile matching can be applied for a given function.
/// Currently we skip inference for (very) large instances and for instances
/// having "unexpected" control flow (e.g., having no sink basic blocks).
bool canApplyInference(const FlowFunction &Func,
                       const yaml::bolt::BinaryFunctionProfile &YamlBF,
                       const uint64_t &MatchedBlocks) {
  if (Func.Blocks.size() > opts::StaleMatchingMaxFuncSize)
    return false;

  if (MatchedBlocks * 100 <
      opts::StaleMatchingMinMatchedBlock * YamlBF.Blocks.size())
    return false;

  // Returns false if the artificial sink block has no predecessors meaning
  // there are no exit blocks.
  if (Func.Blocks[Func.Blocks.size() - 1].isEntry())
    return false;

  return true;
}

/// Apply the profile inference algorithm for a given flow function.
void applyInference(FlowFunction &Func) {
  ProfiParams Params;
  // Set the params from the command-line flags.
  Params.EvenFlowDistribution = opts::StaleMatchingEvenFlowDistribution;
  Params.RebalanceUnknown = opts::StaleMatchingRebalanceUnknown;
  Params.JoinIslands = opts::StaleMatchingJoinIslands;

  Params.CostBlockInc = opts::StaleMatchingCostBlockInc;
  Params.CostBlockEntryInc = opts::StaleMatchingCostBlockInc;
  Params.CostBlockDec = opts::StaleMatchingCostBlockDec;
  Params.CostBlockEntryDec = opts::StaleMatchingCostBlockDec;
  Params.CostBlockUnknownInc = opts::StaleMatchingCostBlockUnknownInc;

  Params.CostJumpInc = opts::StaleMatchingCostJumpInc;
  Params.CostJumpFTInc = opts::StaleMatchingCostJumpInc;
  Params.CostJumpDec = opts::StaleMatchingCostJumpDec;
  Params.CostJumpFTDec = opts::StaleMatchingCostJumpDec;
  Params.CostJumpUnknownInc = opts::StaleMatchingCostJumpUnknownInc;
  Params.CostJumpUnknownFTInc = opts::StaleMatchingCostJumpUnknownFTInc;

  applyFlowInference(Params, Func);
}

/// Collect inferred counts from the flow function and update annotations in
/// the binary function.
void assignProfile(BinaryFunction &BF,
                   const BinaryFunction::BasicBlockOrderType &BlockOrder,
                   FlowFunction &Func) {
  BinaryContext &BC = BF.getBinaryContext();

  assert(Func.Blocks.size() == BlockOrder.size() + 2);
  for (uint64_t I = 0; I < BlockOrder.size(); I++) {
    FlowBlock &Block = Func.Blocks[I + 1];
    BinaryBasicBlock *BB = BlockOrder[I];

    // Update block's count
    BB->setExecutionCount(Block.Flow);

    // Update jump counts: (i) clean existing counts and then (ii) set new ones
    auto BI = BB->branch_info_begin();
    for (const BinaryBasicBlock *DstBB : BB->successors()) {
      (void)DstBB;
      BI->Count = 0;
      BI->MispredictedCount = 0;
      ++BI;
    }
    for (FlowJump *Jump : Block.SuccJumps) {
      if (Jump->IsUnlikely)
        continue;
      if (Jump->Flow == 0)
        continue;

      // Skips the artificial sink block.
      if (Jump->Target == Func.Blocks.size() - 1)
        continue;
      BinaryBasicBlock &SuccBB = *BlockOrder[Jump->Target - 1];
      // Check if the edge corresponds to a regular jump or a landing pad
      if (BB->getSuccessor(SuccBB.getLabel())) {
        BinaryBasicBlock::BinaryBranchInfo &BI = BB->getBranchInfo(SuccBB);
        BI.Count += Jump->Flow;
      } else {
        BinaryBasicBlock *LP = BB->getLandingPad(SuccBB.getLabel());
        if (LP && LP->getKnownExecutionCount() < Jump->Flow)
          LP->setExecutionCount(Jump->Flow);
      }
    }

    // Update call-site annotations
    auto setOrUpdateAnnotation = [&](MCInst &Instr, StringRef Name,
                                     uint64_t Count) {
      if (BC.MIB->hasAnnotation(Instr, Name))
        BC.MIB->removeAnnotation(Instr, Name);
      // Do not add zero-count annotations
      if (Count == 0)
        return;
      BC.MIB->addAnnotation(Instr, Name, Count);
    };

    for (MCInst &Instr : *BB) {
      // Ignore pseudo instructions
      if (BC.MIB->isPseudo(Instr))
        continue;
      // Ignore jump tables
      const MCInst *LastInstr = BB->getLastNonPseudoInstr();
      if (BC.MIB->getJumpTable(*LastInstr) && LastInstr == &Instr)
        continue;

      if (BC.MIB->isIndirectCall(Instr) || BC.MIB->isIndirectBranch(Instr)) {
        auto &ICSP = BC.MIB->getOrCreateAnnotationAs<IndirectCallSiteProfile>(
            Instr, "CallProfile");
        if (!ICSP.empty()) {
          // Try to evenly distribute the counts among the call sites
          const uint64_t TotalCount = Block.Flow;
          const uint64_t NumSites = ICSP.size();
          for (uint64_t Idx = 0; Idx < ICSP.size(); Idx++) {
            IndirectCallProfile &CSP = ICSP[Idx];
            uint64_t CountPerSite = TotalCount / NumSites;
            // When counts cannot be exactly distributed, increase by 1 the
            // counts of the first (TotalCount % NumSites) call sites
            if (Idx < TotalCount % NumSites)
              CountPerSite++;
            CSP.Count = CountPerSite;
          }
        } else {
          ICSP.emplace_back(nullptr, Block.Flow, 0);
        }
      } else if (BC.MIB->getConditionalTailCall(Instr)) {
        // We don't know exactly the number of times the conditional tail call
        // is executed; conservatively, setting it to the count of the block
        setOrUpdateAnnotation(Instr, "CTCTakenCount", Block.Flow);
        BC.MIB->removeAnnotation(Instr, "CTCMispredCount");
      } else if (BC.MIB->isCall(Instr)) {
        setOrUpdateAnnotation(Instr, "Count", Block.Flow);
      }
    }
  }

  // Update function's execution count and mark the function inferred.
  BF.setExecutionCount(Func.Blocks[0].Flow);
  BF.setHasInferredProfile(true);
}

bool YAMLProfileReader::inferStaleProfile(
    BinaryFunction &BF, const yaml::bolt::BinaryFunctionProfile &YamlBF,
    const ArrayRef<ProbeMatchSpec> ProbeMatchSpecs) {

  NamedRegionTimer T("inferStaleProfile", "stale profile inference", "rewrite",
                     "Rewrite passes", opts::TimeRewrite);

  if (!BF.hasCFG())
    return false;

  LLVM_DEBUG(dbgs() << "BOLT-INFO: applying profile inference for "
                    << "\"" << BF.getPrintName() << "\"\n");

  // Make sure that block hashes are up to date.
  BF.computeBlockHashes(YamlBP.Header.HashFunction);

  const BinaryFunction::BasicBlockOrderType BlockOrder(
      BF.getLayout().block_begin(), BF.getLayout().block_end());

  // Tracks the number of matched blocks.

  // Create a wrapper flow function to use with the profile inference algorithm.
  FlowFunction Func = createFlowFunction(BlockOrder);

  // Match as many block/jump counts from the stale profile as possible
  size_t MatchedBlocks =
      matchWeights(BF.getBinaryContext(), BlockOrder, YamlBF, Func,
                   YamlBP.Header.HashFunction, IdToYamLBF, BF, ProbeMatchSpecs);

  // Adjust the flow function by marking unreachable blocks Unlikely so that
  // they don't get any counts assigned.
  preprocessUnreachableBlocks(Func);

  // Check if profile inference can be applied for the instance.
  if (!canApplyInference(Func, YamlBF, MatchedBlocks))
    return false;

  // Apply the profile inference algorithm.
  applyInference(Func);

  // Collect inferred counts and update function annotations.
  assignProfile(BF, BlockOrder, Func);

  // As of now, we always mark the binary function having "correct" profile.
  // In the future, we may discard the results for instances with poor inference
  // metrics and keep such functions un-optimized.
  return true;
}

} // end namespace bolt
} // end namespace llvm
