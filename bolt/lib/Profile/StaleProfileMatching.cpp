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
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/SampleProfileInference.h"

#include <queue>

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

cl::opt<bool>
    InferStaleProfile("infer-stale-profile",
                      cl::desc("Infer counts from stale profile data."),
                      cl::init(false), cl::Hidden, cl::cat(BoltOptCategory));

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
    cl::desc("The cost of increasing a block's count by one."), cl::init(110),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockDec(
    "stale-matching-cost-block-dec",
    cl::desc("The cost of decreasing a block's count by one."), cl::init(100),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockEntryInc(
    "stale-matching-cost-block-entry-inc",
    cl::desc("The cost of increasing the entry block's count by one."),
    cl::init(110), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockEntryDec(
    "stale-matching-cost-block-entry-dec",
    cl::desc("The cost of decreasing the entry block's count by one."),
    cl::init(100), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockZeroInc(
    "stale-matching-cost-block-zero-inc",
    cl::desc("The cost of increasing a count of zero-weight block by one."),
    cl::init(10), cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostBlockUnknownInc(
    "stale-matching-cost-block-unknown-inc",
    cl::desc("The cost of increasing an unknown block's count by one."),
    cl::init(10), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpInc(
    "stale-matching-cost-jump-inc",
    cl::desc("The cost of increasing a jump's count by one."), cl::init(100),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpFTInc(
    "stale-matching-cost-jump-ft-inc",
    cl::desc("The cost of increasing a fall-through jump's count by one."),
    cl::init(100), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpDec(
    "stale-matching-cost-jump-dec",
    cl::desc("The cost of decreasing a jump's count by one."), cl::init(110),
    cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpFTDec(
    "stale-matching-cost-jump-ft-dec",
    cl::desc("The cost of decreasing a fall-through jump's count by one."),
    cl::init(110), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpUnknownInc(
    "stale-matching-cost-jump-unknown-inc",
    cl::desc("The cost of increasing an unknown jump's count by one."),
    cl::init(50), cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<unsigned> StaleMatchingCostJumpUnknownFTInc(
    "stale-matching-cost-jump-unknown-ft-inc",
    cl::desc(
        "The cost of increasing an unknown fall-through jump's count by one."),
    cl::init(5), cl::ReallyHidden, cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

/// An object wrapping several components of a basic block hash. The combined
/// (blended) hash is represented and stored as one uint64_t, while individual
/// components are of smaller size (e.g., uint16_t or uint8_t).
struct BlendedBlockHash {
private:
  static uint64_t combineHashes(uint16_t Hash1, uint16_t Hash2, uint16_t Hash3,
                                uint16_t Hash4) {
    uint64_t Hash = 0;

    Hash |= uint64_t(Hash4);
    Hash <<= 16;

    Hash |= uint64_t(Hash3);
    Hash <<= 16;

    Hash |= uint64_t(Hash2);
    Hash <<= 16;

    Hash |= uint64_t(Hash1);

    return Hash;
  }

  static void parseHashes(uint64_t Hash, uint16_t &Hash1, uint16_t &Hash2,
                          uint16_t &Hash3, uint16_t &Hash4) {
    Hash1 = Hash & 0xffff;
    Hash >>= 16;

    Hash2 = Hash & 0xffff;
    Hash >>= 16;

    Hash3 = Hash & 0xffff;
    Hash >>= 16;

    Hash4 = Hash & 0xffff;
    Hash >>= 16;
  }

public:
  explicit BlendedBlockHash() {}

  explicit BlendedBlockHash(uint64_t CombinedHash) {
    parseHashes(CombinedHash, Offset, OpcodeHash, InstrHash, NeighborHash);
  }

  /// Combine the blended hash into uint64_t.
  uint64_t combine() const {
    return combineHashes(Offset, OpcodeHash, InstrHash, NeighborHash);
  }

  /// Compute a distance between two given blended hashes. The smaller the
  /// distance, the more similar two blocks are. For identical basic blocks,
  /// the distance is zero.
  uint64_t distance(const BlendedBlockHash &BBH) const {
    assert(OpcodeHash == BBH.OpcodeHash &&
           "incorrect blended hash distance computation");
    uint64_t Dist = 0;
    // Account for NeighborHash
    Dist += NeighborHash == BBH.NeighborHash ? 0 : 1;
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
  /// Hash of the (loose) basic block together with (loose) hashes of its
  /// successors and predecessors.
  uint16_t NeighborHash{0};
};

/// The object is used to identify and match basic blocks in a BinaryFunction
/// given their hashes computed on a binary built from several revisions behind
/// release.
class StaleMatcher {
public:
  /// Initialize stale matcher.
  void init(const std::vector<FlowBlock *> &Blocks,
            const std::vector<BlendedBlockHash> &Hashes) {
    assert(Blocks.size() == Hashes.size() &&
           "incorrect matcher initialization");
    for (size_t I = 0; I < Blocks.size(); I++) {
      FlowBlock *Block = Blocks[I];
      uint16_t OpHash = Hashes[I].OpcodeHash;
      OpHashToBlocks[OpHash].push_back(std::make_pair(Hashes[I], Block));
    }
  }

  /// Find the most similar block for a given hash.
  const FlowBlock *matchBlock(BlendedBlockHash BlendedHash) const {
    auto BlockIt = OpHashToBlocks.find(BlendedHash.OpcodeHash);
    if (BlockIt == OpHashToBlocks.end()) {
      return nullptr;
    }
    FlowBlock *BestBlock = nullptr;
    uint64_t BestDist = std::numeric_limits<uint64_t>::max();
    for (auto It : BlockIt->second) {
      FlowBlock *Block = It.second;
      BlendedBlockHash Hash = It.first;
      uint64_t Dist = Hash.distance(BlendedHash);
      if (BestBlock == nullptr || Dist < BestDist) {
        BestDist = Dist;
        BestBlock = Block;
      }
    }
    return BestBlock;
  }

private:
  using HashBlockPairType = std::pair<BlendedBlockHash, FlowBlock *>;
  std::unordered_map<uint16_t, std::vector<HashBlockPairType>> OpHashToBlocks;
};

void BinaryFunction::computeBlockHashes() const {
  if (size() == 0)
    return;

  assert(hasCFG() && "the function is expected to have CFG");

  std::vector<BlendedBlockHash> BlendedHashes(BasicBlocks.size());
  std::vector<uint64_t> OpcodeHashes(BasicBlocks.size());
  // Initialize hash components
  for (size_t I = 0; I < BasicBlocks.size(); I++) {
    const BinaryBasicBlock *BB = BasicBlocks[I];
    assert(BB->getIndex() == I && "incorrect block index");
    BlendedHashes[I].Offset = BB->getOffset();
    // Hashing complete instructions
    std::string InstrHashStr = hashBlock(
        BC, *BB, [&](const MCOperand &Op) { return hashInstOperand(BC, Op); });
    uint64_t InstrHash = std::hash<std::string>{}(InstrHashStr);
    BlendedHashes[I].InstrHash = hash_64_to_16(InstrHash);
    // Hashing opcodes
    std::string OpcodeHashStr =
        hashBlock(BC, *BB, [](const MCOperand &Op) { return std::string(); });
    OpcodeHashes[I] = std::hash<std::string>{}(OpcodeHashStr);
    BlendedHashes[I].OpcodeHash = hash_64_to_16(OpcodeHashes[I]);
  }

  // Initialize neighbor hash
  for (size_t I = 0; I < BasicBlocks.size(); I++) {
    const BinaryBasicBlock *BB = BasicBlocks[I];
    uint64_t Hash = OpcodeHashes[I];
    // Append hashes of successors
    for (BinaryBasicBlock *SuccBB : BB->successors()) {
      uint64_t SuccHash = OpcodeHashes[SuccBB->getIndex()];
      Hash = hashing::detail::hash_16_bytes(Hash, SuccHash);
    }
    // Append hashes of predecessors
    for (BinaryBasicBlock *PredBB : BB->predecessors()) {
      uint64_t PredHash = OpcodeHashes[PredBB->getIndex()];
      Hash = hashing::detail::hash_16_bytes(Hash, PredHash);
    }
    BlendedHashes[I].NeighborHash = hash_64_to_16(Hash);
  }

  //  Assign hashes
  for (size_t I = 0; I < BasicBlocks.size(); I++) {
    const BinaryBasicBlock *BB = BasicBlocks[I];
    BB->setHash(BlendedHashes[I].combine());
  }
}
/// Create a wrapper flow function to use with the profile inference algorithm,
/// and initialize its jumps and metadata.
FlowFunction
createFlowFunction(const BinaryFunction::BasicBlockOrderType &BlockOrder) {
  FlowFunction Func;

  // Add a special "dummy" source so that there is always a unique entry point.
  // Because of the extra source, for all other blocks in FlowFunction it holds
  // that Block.Index == BB->getLayoutIndex() + 1
  FlowBlock EntryBlock;
  EntryBlock.Index = 0;
  Func.Blocks.push_back(EntryBlock);

  // Create FlowBlock for every basic block in the binary function
  for (const BinaryBasicBlock *BB : BlockOrder) {
    Func.Blocks.emplace_back();
    FlowBlock &Block = Func.Blocks.back();
    Block.Index = Func.Blocks.size() - 1;
    (void)BB;
    assert(Block.Index == BB->getLayoutIndex() + 1 &&
           "incorrectly assigned basic block index");
  }

  // Create FlowJump for each jump between basic blocks in the binary function
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
      Jump.Source = SrcBB->getLayoutIndex() + 1;
      Jump.Target = DstBB->getLayoutIndex() + 1;
      InDegree[Jump.Target]++;
      UniqueSuccs.insert(DstBB);
    }
    // Collect jumps to landing pads
    for (const BinaryBasicBlock *DstBB : SrcBB->landing_pads()) {
      // Ignoring parallel edges
      if (UniqueSuccs.find(DstBB) != UniqueSuccs.end())
        continue;

      Func.Jumps.emplace_back();
      FlowJump &Jump = Func.Jumps.back();
      Jump.Source = SrcBB->getLayoutIndex() + 1;
      Jump.Target = DstBB->getLayoutIndex() + 1;
      InDegree[Jump.Target]++;
      UniqueSuccs.insert(DstBB);
    }
  }

  // Add dummy edges to the extra sources. If there are multiple entry blocks,
  // add an unlikely edge from 0 to the subsequent ones
  assert(InDegree[0] == 0 && "dummy entry blocks shouldn't have predecessors");
  for (uint64_t I = 1; I < Func.Blocks.size(); I++) {
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
    Func.Blocks.at(Jump.Source).SuccJumps.push_back(&Jump);
    Func.Blocks.at(Jump.Target).PredJumps.push_back(&Jump);
  }
  return Func;
}

/// Assign initial block/jump weights based on the stale profile data. The goal
/// is to extract as much information from the stale profile as possible. Here
/// we assume that each basic block is specified via a hash value computed from
/// its content and the hashes of the unchanged basic blocks stay the same
/// across different revisions of the binary.
/// Whenever there is a count in the profile with the hash corresponding to one
/// of the basic blocks in the binary, the count is "matched" to the block.
/// Similarly, if both the source and the target of a count in the profile are
/// matched to a jump in the binary, the count is recorded in CFG.
void matchWeightsByHashes(const BinaryFunction::BasicBlockOrderType &BlockOrder,
                          const yaml::bolt::BinaryFunctionProfile &YamlBF,
                          FlowFunction &Func) {
  assert(Func.Blocks.size() == BlockOrder.size() + 1);

  std::vector<FlowBlock *> Blocks;
  std::vector<BlendedBlockHash> BlendedHashes;
  for (uint64_t I = 0; I < BlockOrder.size(); I++) {
    const BinaryBasicBlock *BB = BlockOrder[I];
    assert(BB->getHash() != 0 && "empty hash of BinaryBasicBlock");
    Blocks.push_back(&Func.Blocks[I + 1]);
    BlendedBlockHash BlendedHash(BB->getHash());
    BlendedHashes.push_back(BlendedHash);
    LLVM_DEBUG(dbgs() << "BB with index " << I << " has hash = "
                      << Twine::utohexstr(BB->getHash()) << "\n");
  }
  StaleMatcher Matcher;
  Matcher.init(Blocks, BlendedHashes);

  // Index in yaml profile => corresponding (matched) block
  DenseMap<uint64_t, const FlowBlock *> MatchedBlocks;
  // Match blocks from the profile to the blocks in CFG
  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks) {
    assert(YamlBB.Hash != 0 && "empty hash of BinaryBasicBlockProfile");
    BlendedBlockHash BlendedHash(YamlBB.Hash);
    const FlowBlock *MatchedBlock = Matcher.matchBlock(BlendedHash);
    if (MatchedBlock != nullptr) {
      MatchedBlocks[YamlBB.Index] = MatchedBlock;
      LLVM_DEBUG(dbgs() << "Matched yaml block with bid = " << YamlBB.Index
                        << " and hash = " << Twine::utohexstr(YamlBB.Hash)
                        << " to BB with index = " << MatchedBlock->Index - 1
                        << "\n");
    } else {
      LLVM_DEBUG(
          dbgs() << "Couldn't match yaml block with bid = " << YamlBB.Index
                 << " and hash = " << Twine::utohexstr(YamlBB.Hash) << "\n");
    }
  }

  // Match jumps from the profile to the jumps from CFG
  std::vector<uint64_t> OutWeight(Func.Blocks.size(), 0);
  std::vector<uint64_t> InWeight(Func.Blocks.size(), 0);
  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks) {
    for (const yaml::bolt::SuccessorInfo &YamlSI : YamlBB.Successors) {
      if (YamlSI.Count == 0)
        continue;

      // Try to find the jump for a given (src, dst) pair from the profile and
      // assign the jump weight based on the profile count
      const uint64_t SrcIndex = YamlBB.Index;
      const uint64_t DstIndex = YamlSI.Index;

      const FlowBlock *MatchedSrcBlock = MatchedBlocks.lookup(SrcIndex);
      const FlowBlock *MatchedDstBlock = MatchedBlocks.lookup(DstIndex);

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

  // Assign block counts based on in-/out- jumps
  for (FlowBlock &Block : Func.Blocks) {
    if (OutWeight[Block.Index] == 0 && InWeight[Block.Index] == 0) {
      assert(Block.HasUnknownWeight && "unmatched block with positive count");
      continue;
    }
    Block.HasUnknownWeight = false;
    Block.Weight = std::max(OutWeight[Block.Index], InWeight[Block.Index]);
  }
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
bool canApplyInference(const FlowFunction &Func) {
  if (Func.Blocks.size() > opts::StaleMatchingMaxFuncSize)
    return false;

  bool HasExitBlocks = llvm::any_of(
      Func.Blocks, [&](const FlowBlock &Block) { return Block.isExit(); });
  if (!HasExitBlocks)
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
  Params.CostBlockDec = opts::StaleMatchingCostBlockDec;
  Params.CostBlockEntryInc = opts::StaleMatchingCostBlockEntryInc;
  Params.CostBlockEntryDec = opts::StaleMatchingCostBlockEntryDec;
  Params.CostBlockZeroInc = opts::StaleMatchingCostBlockZeroInc;
  Params.CostBlockUnknownInc = opts::StaleMatchingCostBlockUnknownInc;

  Params.CostJumpInc = opts::StaleMatchingCostJumpInc;
  Params.CostJumpFTInc = opts::StaleMatchingCostJumpFTInc;
  Params.CostJumpDec = opts::StaleMatchingCostJumpDec;
  Params.CostJumpFTDec = opts::StaleMatchingCostJumpFTDec;
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

  assert(Func.Blocks.size() == BlockOrder.size() + 1);
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
    BinaryFunction &BF, const yaml::bolt::BinaryFunctionProfile &YamlBF) {
  // Make sure that block indices and hashes are up to date
  BF.getLayout().updateLayoutIndices();
  BF.computeBlockHashes();

  const BinaryFunction::BasicBlockOrderType BlockOrder(
      BF.getLayout().block_begin(), BF.getLayout().block_end());

  // Create a wrapper flow function to use with the profile inference algorithm
  FlowFunction Func = createFlowFunction(BlockOrder);

  // Match as many block/jump counts from the stale profile as possible
  matchWeightsByHashes(BlockOrder, YamlBF, Func);

  // Adjust the flow function by marking unreachable blocks Unlikely so that
  // they don't get any counts assigned
  preprocessUnreachableBlocks(Func);

  // Check if profile inference can be applied for the instance
  if (!canApplyInference(Func))
    return false;

  // Apply the profile inference algorithm
  applyInference(Func);

  // Collect inferred counts and update function annotations
  assignProfile(BF, BlockOrder, Func);

  // As of now, we always mark the binary function having "correct" profile.
  // In the future, we may discard the results for instances with poor inference
  // metrics and keep such functions un-optimized.
  return true;
}

} // end namespace bolt
} // end namespace llvm
