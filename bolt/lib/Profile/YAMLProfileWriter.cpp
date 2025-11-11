//===- bolt/Profile/YAMLProfileWriter.cpp - YAML profile serializer -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/YAMLProfileWriter.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Profile/BoltAddressTranslation.h"
#include "bolt/Profile/DataAggregator.h"
#include "bolt/Profile/ProfileReaderBase.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

namespace opts {
using namespace llvm;
extern cl::opt<bool> ProfileUseDFS;
cl::opt<bool> ProfileWritePseudoProbes(
    "profile-write-pseudo-probes",
    cl::desc("Use pseudo probes in profile generation"), cl::Hidden,
    cl::cat(BoltOptCategory));
} // namespace opts

namespace llvm {
namespace bolt {

const BinaryFunction *YAMLProfileWriter::setCSIDestination(
    const BinaryContext &BC, yaml::bolt::CallSiteInfo &CSI,
    const MCSymbol *Symbol, const BoltAddressTranslation *BAT,
    uint32_t Offset) {
  CSI.DestId = 0; // designated for unknown functions
  CSI.EntryDiscriminator = 0;

  if (Symbol) {
    uint64_t EntryID = 0;
    if (const BinaryFunction *Callee =
            BC.getFunctionForSymbol(Symbol, &EntryID)) {
      if (BAT && BAT->isBATFunction(Callee->getAddress()))
        std::tie(Callee, EntryID) = BAT->translateSymbol(BC, *Symbol, Offset);
      else if (const BinaryBasicBlock *BB =
                   Callee->getBasicBlockContainingOffset(Offset))
        BC.getFunctionForSymbol(Callee->getSecondaryEntryPointSymbol(*BB),
                                &EntryID);
      CSI.DestId = Callee->getFunctionNumber();
      CSI.EntryDiscriminator = EntryID;
      return Callee;
    }
  }
  return nullptr;
}

std::vector<YAMLProfileWriter::InlineTreeNode>
YAMLProfileWriter::collectInlineTree(
    const MCPseudoProbeDecoder &Decoder,
    const MCDecodedPseudoProbeInlineTree &Root) {
  auto getHash = [&](const MCDecodedPseudoProbeInlineTree &Node) {
    return Decoder.getFuncDescForGUID(Node.Guid)->FuncHash;
  };
  std::vector<InlineTreeNode> InlineTree(
      {InlineTreeNode{&Root, Root.Guid, getHash(Root), 0, 0}});
  uint32_t ParentId = 0;
  while (ParentId != InlineTree.size()) {
    const MCDecodedPseudoProbeInlineTree *Cur = InlineTree[ParentId].InlineTree;
    for (const MCDecodedPseudoProbeInlineTree &Child : Cur->getChildren())
      InlineTree.emplace_back(
          InlineTreeNode{&Child, Child.Guid, getHash(Child), ParentId,
                         std::get<1>(Child.getInlineSite())});
    ++ParentId;
  }

  return InlineTree;
}

std::tuple<yaml::bolt::ProfilePseudoProbeDesc,
           YAMLProfileWriter::InlineTreeDesc>
YAMLProfileWriter::convertPseudoProbeDesc(const MCPseudoProbeDecoder &Decoder) {
  yaml::bolt::ProfilePseudoProbeDesc Desc;
  InlineTreeDesc InlineTree;

  for (const MCDecodedPseudoProbeInlineTree &TopLev :
       Decoder.getDummyInlineRoot().getChildren())
    InlineTree.TopLevelGUIDToInlineTree[TopLev.Guid] = &TopLev;

  for (const auto &FuncDesc : Decoder.getGUID2FuncDescMap())
    ++InlineTree.HashIdxMap[FuncDesc.FuncHash];

  InlineTree.GUIDIdxMap.reserve(Decoder.getGUID2FuncDescMap().size());
  for (const auto &Node : Decoder.getInlineTreeVec())
    ++InlineTree.GUIDIdxMap[Node.Guid];

  std::vector<std::pair<uint32_t, uint64_t>> GUIDFreqVec;
  GUIDFreqVec.reserve(InlineTree.GUIDIdxMap.size());
  for (const auto [GUID, Cnt] : InlineTree.GUIDIdxMap)
    GUIDFreqVec.emplace_back(Cnt, GUID);
  llvm::sort(GUIDFreqVec);

  std::vector<std::pair<uint32_t, uint64_t>> HashFreqVec;
  HashFreqVec.reserve(InlineTree.HashIdxMap.size());
  for (const auto [Hash, Cnt] : InlineTree.HashIdxMap)
    HashFreqVec.emplace_back(Cnt, Hash);
  llvm::sort(HashFreqVec);

  uint32_t Index = 0;
  Desc.Hash.reserve(HashFreqVec.size());
  for (uint64_t Hash : llvm::make_second_range(llvm::reverse(HashFreqVec))) {
    Desc.Hash.emplace_back(Hash);
    InlineTree.HashIdxMap[Hash] = Index++;
  }

  Index = 0;
  Desc.GUID.reserve(GUIDFreqVec.size());
  for (uint64_t GUID : llvm::make_second_range(llvm::reverse(GUIDFreqVec))) {
    Desc.GUID.emplace_back(GUID);
    InlineTree.GUIDIdxMap[GUID] = Index++;
    uint64_t Hash = Decoder.getFuncDescForGUID(GUID)->FuncHash;
    Desc.GUIDHashIdx.emplace_back(InlineTree.HashIdxMap[Hash]);
  }

  return {Desc, InlineTree};
}

void YAMLProfileWriter::BlockProbeCtx::addBlockProbe(
    const InlineTreeMapTy &Map, const MCDecodedPseudoProbe &Probe,
    uint32_t ProbeOffset) {
  auto It = Map.find(Probe.getInlineTreeNode());
  if (It == Map.end())
    return;
  auto NodeId = It->second;
  uint32_t Index = Probe.getIndex();
  if (Probe.isCall())
    CallProbes[ProbeOffset] =
        Call{Index, NodeId, Probe.isIndirectCall(), false};
  else
    NodeToProbes[NodeId].emplace_back(Index);
}

void YAMLProfileWriter::BlockProbeCtx::finalize(
    yaml::bolt::BinaryBasicBlockProfile &YamlBB) {
  // Hash block probes by vector
  struct ProbeHasher {
    size_t operator()(const ArrayRef<uint64_t> Probes) const {
      return llvm::hash_combine_range(Probes);
    }
  };

  // Check identical block probes and merge them
  std::unordered_map<std::vector<uint64_t>, std::vector<uint32_t>, ProbeHasher>
      ProbesToNodes;
  for (auto &[NodeId, Probes] : NodeToProbes) {
    llvm::sort(Probes);
    ProbesToNodes[Probes].emplace_back(NodeId);
  }
  for (auto &[Probes, Nodes] : ProbesToNodes) {
    llvm::sort(Nodes);
    YamlBB.PseudoProbes.emplace_back(
        yaml::bolt::PseudoProbeInfo{Probes, Nodes});
  }
  for (yaml::bolt::CallSiteInfo &CSI : YamlBB.CallSites) {
    auto It = CallProbes.find(CSI.Offset);
    if (It == CallProbes.end())
      continue;
    Call &Probe = It->second;
    CSI.Probe = Probe.Id;
    CSI.InlineTreeNode = Probe.Node;
    CSI.Indirect = Probe.Indirect;
    Probe.Used = true;
  }
  for (const auto &[Offset, Probe] : CallProbes) {
    if (Probe.Used)
      continue;
    yaml::bolt::CallSiteInfo CSI;
    CSI.Offset = Offset;
    CSI.Probe = Probe.Id;
    CSI.InlineTreeNode = Probe.Node;
    CSI.Indirect = Probe.Indirect;
    YamlBB.CallSites.emplace_back(CSI);
  }
}

std::tuple<std::vector<yaml::bolt::InlineTreeNode>,
           YAMLProfileWriter::InlineTreeMapTy>
YAMLProfileWriter::convertBFInlineTree(const MCPseudoProbeDecoder &Decoder,
                                       const InlineTreeDesc &InlineTree,
                                       uint64_t GUID) {
  DenseMap<const MCDecodedPseudoProbeInlineTree *, uint32_t> InlineTreeNodeId;
  std::vector<yaml::bolt::InlineTreeNode> YamlInlineTree;
  auto It = InlineTree.TopLevelGUIDToInlineTree.find(GUID);
  if (It == InlineTree.TopLevelGUIDToInlineTree.end())
    return {YamlInlineTree, InlineTreeNodeId};
  const MCDecodedPseudoProbeInlineTree *Root = It->second;
  assert(Root && "Malformed TopLevelGUIDToInlineTree");
  uint32_t Index = 0;
  uint32_t PrevParent = 0;
  uint32_t PrevGUIDIdx = 0;
  for (const auto &Node : collectInlineTree(Decoder, *Root)) {
    InlineTreeNodeId[Node.InlineTree] = Index++;
    auto GUIDIdxIt = InlineTree.GUIDIdxMap.find(Node.GUID);
    assert(GUIDIdxIt != InlineTree.GUIDIdxMap.end() && "Malformed GUIDIdxMap");
    uint32_t GUIDIdx = GUIDIdxIt->second;
    if (GUIDIdx == PrevGUIDIdx)
      GUIDIdx = UINT32_MAX;
    else
      PrevGUIDIdx = GUIDIdx;
    YamlInlineTree.emplace_back(yaml::bolt::InlineTreeNode{
        Node.ParentId - PrevParent, Node.InlineSite, GUIDIdx, 0, 0});
    PrevParent = Node.ParentId;
  }
  return {YamlInlineTree, InlineTreeNodeId};
}

yaml::bolt::BinaryFunctionProfile
YAMLProfileWriter::convert(const BinaryFunction &BF, bool UseDFS,
                           const InlineTreeDesc &InlineTree,
                           const BoltAddressTranslation *BAT) {
  yaml::bolt::BinaryFunctionProfile YamlBF;
  const BinaryContext &BC = BF.getBinaryContext();
  const MCPseudoProbeDecoder *PseudoProbeDecoder =
      opts::ProfileWritePseudoProbes ? BC.getPseudoProbeDecoder() : nullptr;

  const uint16_t LBRProfile = BF.getProfileFlags() & BinaryFunction::PF_BRANCH;

  // Prepare function and block hashes
  BF.computeHash(UseDFS);
  BF.computeBlockHashes();

  YamlBF.Name = DataAggregator::getLocationName(BF, BAT);
  YamlBF.Id = BF.getFunctionNumber();
  YamlBF.Hash = BF.getHash();
  YamlBF.NumBasicBlocks = BF.size();
  YamlBF.ExecCount = BF.getKnownExecutionCount();
  YamlBF.ExternEntryCount = BF.getExternEntryCount();
  DenseMap<const MCDecodedPseudoProbeInlineTree *, uint32_t> InlineTreeNodeId;
  if (PseudoProbeDecoder && BF.getGUID()) {
    std::tie(YamlBF.InlineTree, InlineTreeNodeId) =
        convertBFInlineTree(*PseudoProbeDecoder, InlineTree, BF.getGUID());
  }

  BinaryFunction::BasicBlockOrderType Order;
  llvm::copy(UseDFS ? BF.dfs() : BF.getLayout().blocks(),
             std::back_inserter(Order));

  const FunctionLayout Layout = BF.getLayout();
  Layout.updateLayoutIndices(Order);

  for (const BinaryBasicBlock *BB : Order) {
    yaml::bolt::BinaryBasicBlockProfile YamlBB;
    YamlBB.Index = BB->getLayoutIndex();
    YamlBB.NumInstructions = BB->getNumNonPseudos();
    YamlBB.Hash = BB->getHash();

    if (!LBRProfile) {
      YamlBB.EventCount = BB->getKnownExecutionCount();
      if (YamlBB.EventCount)
        YamlBF.Blocks.emplace_back(YamlBB);
      continue;
    }

    YamlBB.ExecCount = BB->getKnownExecutionCount();

    for (const MCInst &Instr : *BB) {
      if (!BC.MIB->isCall(Instr) && !BC.MIB->isIndirectBranch(Instr))
        continue;

      SmallVector<std::pair<StringRef, yaml::bolt::CallSiteInfo>> CSTargets;
      yaml::bolt::CallSiteInfo CSI;
      std::optional<uint32_t> Offset = BC.MIB->getOffset(Instr);
      if (!Offset || *Offset < BB->getInputOffset())
        continue;
      CSI.Offset = *Offset - BB->getInputOffset();

      if (BC.MIB->isIndirectCall(Instr) || BC.MIB->isIndirectBranch(Instr)) {
        const auto ICSP = BC.MIB->tryGetAnnotationAs<IndirectCallSiteProfile>(
            Instr, "CallProfile");
        if (!ICSP)
          continue;
        for (const IndirectCallProfile &CSP : ICSP.get()) {
          StringRef TargetName = "";
          const BinaryFunction *Callee =
              setCSIDestination(BC, CSI, CSP.Symbol, BAT);
          if (Callee)
            TargetName = Callee->getOneName();
          CSI.Count = CSP.Count;
          CSI.Mispreds = CSP.Mispreds;
          CSTargets.emplace_back(TargetName, CSI);
        }
      } else { // direct call or a tail call
        StringRef TargetName = "";
        const MCSymbol *CalleeSymbol = BC.MIB->getTargetSymbol(Instr);
        const BinaryFunction *const Callee =
            setCSIDestination(BC, CSI, CalleeSymbol, BAT);
        if (Callee)
          TargetName = Callee->getOneName();

        auto getAnnotationWithDefault = [&](const MCInst &Inst, StringRef Ann) {
          return BC.MIB->getAnnotationWithDefault(Instr, Ann, 0ull);
        };
        if (BC.MIB->getConditionalTailCall(Instr)) {
          CSI.Count = getAnnotationWithDefault(Instr, "CTCTakenCount");
          CSI.Mispreds = getAnnotationWithDefault(Instr, "CTCMispredCount");
        } else {
          CSI.Count = getAnnotationWithDefault(Instr, "Count");
        }

        if (CSI.Count)
          CSTargets.emplace_back(TargetName, CSI);
      }
      // Sort targets in a similar way to getBranchData, see Location::operator<
      llvm::sort(CSTargets, [](const auto &RHS, const auto &LHS) {
        return std::tie(RHS.first, RHS.second.Offset) <
               std::tie(LHS.first, LHS.second.Offset);
      });
      for (auto &KV : CSTargets)
        YamlBB.CallSites.push_back(KV.second);
    }

    // Skip printing if there's no profile data for non-entry basic block.
    // Include landing pads with non-zero execution count.
    if (YamlBB.CallSites.empty() && !BB->isEntryPoint() &&
        !(BB->isLandingPad() && BB->getKnownExecutionCount() != 0)) {
      // Include blocks having successors or predecessors with positive counts.
      uint64_t SuccessorExecCount = 0;
      for (const BinaryBasicBlock::BinaryBranchInfo &BranchInfo :
           BB->branch_info())
        SuccessorExecCount += BranchInfo.Count;
      uint64_t PredecessorExecCount = 0;
      for (auto Pred : BB->predecessors())
        PredecessorExecCount += Pred->getBranchInfo(*BB).Count;
      if (!SuccessorExecCount && !PredecessorExecCount)
        continue;
    }

    auto BranchInfo = BB->branch_info_begin();
    for (const BinaryBasicBlock *Successor : BB->successors()) {
      yaml::bolt::SuccessorInfo YamlSI;
      YamlSI.Index = Successor->getLayoutIndex();
      YamlSI.Count = BranchInfo->Count;
      YamlSI.Mispreds = BranchInfo->MispredictedCount;

      YamlBB.Successors.emplace_back(YamlSI);

      ++BranchInfo;
    }

    if (PseudoProbeDecoder) {
      const AddressProbesMap &ProbeMap =
          PseudoProbeDecoder->getAddress2ProbesMap();
      const uint64_t FuncAddr = BF.getAddress();
      auto [Start, End] = BB->getInputAddressRange();
      Start += FuncAddr;
      End += FuncAddr;
      BlockProbeCtx Ctx;
      for (const MCDecodedPseudoProbe &Probe : ProbeMap.find(Start, End))
        Ctx.addBlockProbe(InlineTreeNodeId, Probe, Probe.getAddress() - Start);
      Ctx.finalize(YamlBB);
    }

    YamlBF.Blocks.emplace_back(YamlBB);
  }
  return YamlBF;
}

std::error_code YAMLProfileWriter::writeProfile(const RewriteInstance &RI) {
  const BinaryContext &BC = RI.getBinaryContext();
  const auto &Functions = BC.getBinaryFunctions();

  std::error_code EC;
  OS = std::make_unique<raw_fd_ostream>(Filename, EC, sys::fs::OF_None);
  if (EC) {
    errs() << "BOLT-WARNING: " << EC.message() << " : unable to open "
           << Filename << " for output.\n";
    return EC;
  }

  yaml::bolt::BinaryProfile BP;

  // Fill out the header info.
  BP.Header.Version = 1;
  BP.Header.FileName = std::string(BC.getFilename());
  std::optional<StringRef> BuildID = BC.getFileBuildID();
  BP.Header.Id = BuildID ? std::string(*BuildID) : "<unknown>";
  BP.Header.Origin = std::string(RI.getProfileReader()->getReaderName());
  BP.Header.IsDFSOrder = opts::ProfileUseDFS;
  BP.Header.HashFunction = HashFunction::Default;

  StringSet<> EventNames = RI.getProfileReader()->getEventNames();
  if (!EventNames.empty()) {
    std::string Sep;
    for (const StringMapEntry<EmptyStringSetTag> &EventEntry : EventNames) {
      BP.Header.EventNames += Sep + EventEntry.first().str();
      Sep = ",";
    }
  }

  // Make sure the profile is consistent across all functions.
  uint16_t ProfileFlags = BinaryFunction::PF_NONE;
  for (const auto &BFI : Functions) {
    const BinaryFunction &BF = BFI.second;
    if (BF.hasProfile() && !BF.empty()) {
      assert(BF.getProfileFlags() != BinaryFunction::PF_NONE);
      if (ProfileFlags == BinaryFunction::PF_NONE)
        ProfileFlags = BF.getProfileFlags();

      assert(BF.getProfileFlags() == ProfileFlags &&
             "expected consistent profile flags across all functions");
    }
  }
  BP.Header.Flags = ProfileFlags;

  // Add probe inline tree nodes.
  InlineTreeDesc InlineTree;
  if (const MCPseudoProbeDecoder *Decoder =
          opts::ProfileWritePseudoProbes ? BC.getPseudoProbeDecoder() : nullptr)
    std::tie(BP.PseudoProbeDesc, InlineTree) = convertPseudoProbeDesc(*Decoder);

  // Add all function objects.
  for (const auto &BFI : Functions) {
    const BinaryFunction &BF = BFI.second;
    if (BF.hasProfile()) {
      if (!BF.hasValidProfile() && !RI.getProfileReader()->isTrustedSource())
        continue;

      BP.Functions.emplace_back(convert(BF, opts::ProfileUseDFS, InlineTree));
    }
  }

  // Write the profile.
  yaml::Output Out(*OS, nullptr, 0);
  Out << BP;

  return std::error_code();
}

} // namespace bolt
} // namespace llvm
