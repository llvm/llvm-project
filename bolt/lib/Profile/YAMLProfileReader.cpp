//===- bolt/Profile/YAMLProfileReader.cpp - YAML profile de-serializer ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/YAMLProfileReader.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/MCF.h"
#include "bolt/Profile/ProfileYAMLMapping.h"
#include "bolt/Utils/NameResolver.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/edit_distance.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace opts {

extern cl::opt<unsigned> Verbosity;
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> InferStaleProfile;
extern cl::opt<bool> Lite;

static cl::opt<unsigned> NameSimilarityFunctionMatchingThreshold(
    "name-similarity-function-matching-threshold",
    cl::desc("Match functions using namespace and edit distance"), cl::init(0),
    cl::Hidden, cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
    IgnoreHash("profile-ignore-hash",
               cl::desc("ignore hash while reading function profile"),
               cl::Hidden, cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
    MatchProfileWithFunctionHash("match-profile-with-function-hash",
                                 cl::desc("Match profile with function hash"),
                                 cl::Hidden, cl::cat(BoltOptCategory));
static llvm::cl::opt<bool>
    MatchWithCallGraph("match-with-call-graph",
                       cl::desc("Match functions with call graph"), cl::Hidden,
                       cl::cat(BoltOptCategory));

llvm::cl::opt<bool> ProfileUseDFS("profile-use-dfs",
                                  cl::desc("use DFS order for YAML profile"),
                                  cl::Hidden, cl::cat(BoltOptCategory));

extern llvm::cl::opt<bool> StaleMatchingWithPseudoProbes;
} // namespace opts

namespace llvm {
namespace bolt {

YAMLProfileReader::CallGraphMatcher::CallGraphMatcher(
    BinaryContext &BC, yaml::bolt::BinaryProfile &YamlBP,
    ProfileLookupMap &IdToYAMLBF) {
  constructBFCG(BC, YamlBP);
  constructYAMLFCG(YamlBP, IdToYAMLBF);
  computeBFNeighborHashes(BC);
}

void YAMLProfileReader::CallGraphMatcher::constructBFCG(
    BinaryContext &BC, yaml::bolt::BinaryProfile &YamlBP) {
  for (BinaryFunction *BF : BC.getAllBinaryFunctions()) {
    for (const BinaryBasicBlock &BB : BF->blocks()) {
      for (const MCInst &Instr : BB) {
        if (!BC.MIB->isCall(Instr))
          continue;
        const MCSymbol *CallSymbol = BC.MIB->getTargetSymbol(Instr);
        if (!CallSymbol)
          continue;
        BinaryData *BD = BC.getBinaryDataByName(CallSymbol->getName());
        if (!BD)
          continue;
        BinaryFunction *CalleeBF = BC.getFunctionForSymbol(BD->getSymbol());
        if (!CalleeBF)
          continue;

        BFAdjacencyMap[CalleeBF].insert(BF);
        BFAdjacencyMap[BF].insert(CalleeBF);
      }
    }
  }
}

void YAMLProfileReader::CallGraphMatcher::computeBFNeighborHashes(
    BinaryContext &BC) {
  for (BinaryFunction *BF : BC.getAllBinaryFunctions()) {
    auto It = BFAdjacencyMap.find(BF);
    if (It == BFAdjacencyMap.end())
      continue;
    auto &AdjacentBFs = It->second;
    std::string HashStr;
    for (BinaryFunction *BF : AdjacentBFs)
      HashStr += BF->getOneName();
    uint64_t Hash = std::hash<std::string>{}(HashStr);
    NeighborHashToBFs[Hash].push_back(BF);
  }
}

void YAMLProfileReader::CallGraphMatcher::constructYAMLFCG(
    yaml::bolt::BinaryProfile &YamlBP, ProfileLookupMap &IdToYAMLBF) {

  for (auto &CallerYamlBF : YamlBP.Functions) {
    for (auto &YamlBB : CallerYamlBF.Blocks) {
      for (auto &CallSite : YamlBB.CallSites) {
        auto IdToYAMLBFIt = IdToYAMLBF.find(CallSite.DestId);
        if (IdToYAMLBFIt == IdToYAMLBF.end())
          continue;
        YamlBFAdjacencyMap[&CallerYamlBF].insert(IdToYAMLBFIt->second);
        YamlBFAdjacencyMap[IdToYAMLBFIt->second].insert(&CallerYamlBF);
      }
    }
  }
}

bool YAMLProfileReader::isYAML(const StringRef Filename) {
  if (auto MB = MemoryBuffer::getFileOrSTDIN(Filename)) {
    StringRef Buffer = (*MB)->getBuffer();
    return Buffer.starts_with("---\n");
  } else {
    report_error(Filename, MB.getError());
  }
  return false;
}

void YAMLProfileReader::buildNameMaps(BinaryContext &BC) {
  auto lookupFunction = [&](StringRef Name) -> BinaryFunction * {
    if (BinaryData *BD = BC.getBinaryDataByName(Name))
      return BC.getFunctionForSymbol(BD->getSymbol());
    return nullptr;
  };

  ProfileBFs.reserve(YamlBP.Functions.size());

  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    StringRef Name = YamlBF.Name;
    const size_t Pos = Name.find("(*");
    if (Pos != StringRef::npos)
      Name = Name.substr(0, Pos);
    ProfileFunctionNames.insert(Name);
    ProfileBFs.push_back(lookupFunction(Name));
    if (const std::optional<StringRef> CommonName = getLTOCommonName(Name))
      LTOCommonNameMap[*CommonName].push_back(&YamlBF);
  }
  for (auto &[Symbol, BF] : BC.SymbolToFunctionMap) {
    StringRef Name = Symbol->getName();
    if (const std::optional<StringRef> CommonName = getLTOCommonName(Name))
      LTOCommonNameFunctionMap[*CommonName].insert(BF);
  }
}

bool YAMLProfileReader::hasLocalsWithFileName() const {
  return llvm::any_of(ProfileFunctionNames.keys(), [](StringRef FuncName) {
    return FuncName.count('/') == 2 && FuncName[0] != '/';
  });
}

bool YAMLProfileReader::parseFunctionProfile(
    BinaryFunction &BF, const yaml::bolt::BinaryFunctionProfile &YamlBF) {
  BinaryContext &BC = BF.getBinaryContext();

  const bool IsDFSOrder = YamlBP.Header.IsDFSOrder;
  const HashFunction HashFunction = YamlBP.Header.HashFunction;
  bool ProfileMatched = true;
  uint64_t MismatchedBlocks = 0;
  uint64_t MismatchedCalls = 0;
  uint64_t MismatchedEdges = 0;

  uint64_t FunctionExecutionCount = 0;

  BF.setExecutionCount(YamlBF.ExecCount);
  BF.setExternEntryCount(YamlBF.ExternEntryCount);

  uint64_t FuncRawBranchCount = 0;
  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks)
    for (const yaml::bolt::SuccessorInfo &YamlSI : YamlBB.Successors)
      FuncRawBranchCount += YamlSI.Count;
  BF.setRawSampleCount(FuncRawBranchCount);

  if (BF.empty())
    return true;

  if (!opts::IgnoreHash) {
    if (!BF.getHash())
      BF.computeHash(IsDFSOrder, HashFunction);
    if (YamlBF.Hash != BF.getHash()) {
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: function hash mismatch\n";
      ProfileMatched = false;
    }
  }

  if (YamlBF.NumBasicBlocks != BF.size()) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: number of basic blocks mismatch\n";
    ProfileMatched = false;
  }

  BinaryFunction::BasicBlockOrderType Order;
  if (IsDFSOrder)
    llvm::copy(BF.dfs(), std::back_inserter(Order));
  else
    llvm::copy(BF.getLayout().blocks(), std::back_inserter(Order));

  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks) {
    if (YamlBB.Index >= Order.size()) {
      if (opts::Verbosity >= 2)
        errs() << "BOLT-WARNING: index " << YamlBB.Index
               << " is out of bounds\n";
      ++MismatchedBlocks;
      continue;
    }

    BinaryBasicBlock &BB = *Order[YamlBB.Index];

    // Basic samples profile (without LBR) does not have branches information
    // and needs a special processing.
    if (YamlBP.Header.Flags & BinaryFunction::PF_BASIC) {
      if (!YamlBB.EventCount) {
        BB.setExecutionCount(0);
        continue;
      }
      uint64_t NumSamples = YamlBB.EventCount * 1000;
      if (NormalizeByInsnCount && BB.getNumNonPseudos())
        NumSamples /= BB.getNumNonPseudos();
      else if (NormalizeByCalls)
        NumSamples /= BB.getNumCalls() + 1;

      BB.setExecutionCount(NumSamples);
      if (BB.isEntryPoint())
        FunctionExecutionCount += NumSamples;
      continue;
    }

    BB.setExecutionCount(YamlBB.ExecCount);

    for (const yaml::bolt::CallSiteInfo &YamlCSI : YamlBB.CallSites) {
      BinaryFunction *Callee = YamlProfileToFunction.lookup(YamlCSI.DestId);
      bool IsFunction = Callee ? true : false;
      MCSymbol *CalleeSymbol = nullptr;
      if (IsFunction)
        CalleeSymbol = Callee->getSymbolForEntryID(YamlCSI.EntryDiscriminator);

      BF.getAllCallSites().emplace_back(CalleeSymbol, YamlCSI.Count,
                                        YamlCSI.Mispreds, YamlCSI.Offset);

      if (YamlCSI.Offset >= BB.getOriginalSize()) {
        if (opts::Verbosity >= 2)
          errs() << "BOLT-WARNING: offset " << YamlCSI.Offset
                 << " out of bounds in block " << BB.getName() << '\n';
        ++MismatchedCalls;
        continue;
      }

      MCInst *Instr =
          BF.getInstructionAtOffset(BB.getInputOffset() + YamlCSI.Offset);
      if (!Instr) {
        if (opts::Verbosity >= 2)
          errs() << "BOLT-WARNING: no instruction at offset " << YamlCSI.Offset
                 << " in block " << BB.getName() << '\n';
        ++MismatchedCalls;
        continue;
      }
      if (!BC.MIB->isCall(*Instr) && !BC.MIB->isIndirectBranch(*Instr)) {
        if (opts::Verbosity >= 2)
          errs() << "BOLT-WARNING: expected call at offset " << YamlCSI.Offset
                 << " in block " << BB.getName() << '\n';
        ++MismatchedCalls;
        continue;
      }

      auto setAnnotation = [&](StringRef Name, uint64_t Count) {
        if (BC.MIB->hasAnnotation(*Instr, Name)) {
          if (opts::Verbosity >= 1)
            errs() << "BOLT-WARNING: ignoring duplicate " << Name
                   << " info for offset 0x" << Twine::utohexstr(YamlCSI.Offset)
                   << " in function " << BF << '\n';
          return;
        }
        BC.MIB->addAnnotation(*Instr, Name, Count);
      };

      if (BC.MIB->isIndirectCall(*Instr) || BC.MIB->isIndirectBranch(*Instr)) {
        auto &CSP = BC.MIB->getOrCreateAnnotationAs<IndirectCallSiteProfile>(
            *Instr, "CallProfile");
        CSP.emplace_back(CalleeSymbol, YamlCSI.Count, YamlCSI.Mispreds);
      } else if (BC.MIB->getConditionalTailCall(*Instr)) {
        setAnnotation("CTCTakenCount", YamlCSI.Count);
        setAnnotation("CTCMispredCount", YamlCSI.Mispreds);
      } else {
        setAnnotation("Count", YamlCSI.Count);
      }
    }

    for (const yaml::bolt::SuccessorInfo &YamlSI : YamlBB.Successors) {
      if (YamlSI.Index >= Order.size()) {
        if (opts::Verbosity >= 1)
          errs() << "BOLT-WARNING: index out of bounds for profiled block\n";
        ++MismatchedEdges;
        continue;
      }

      BinaryBasicBlock *ToBB = Order[YamlSI.Index];
      if (!BB.getSuccessor(ToBB->getLabel())) {
        // Allow passthrough blocks.
        BinaryBasicBlock *FTSuccessor = BB.getConditionalSuccessor(false);
        if (FTSuccessor && FTSuccessor->succ_size() == 1 &&
            FTSuccessor->getSuccessor(ToBB->getLabel())) {
          BinaryBasicBlock::BinaryBranchInfo &FTBI =
              FTSuccessor->getBranchInfo(*ToBB);
          FTBI.Count += YamlSI.Count;
          FTBI.MispredictedCount += YamlSI.Mispreds;
          ToBB = FTSuccessor;
        } else {
          if (opts::Verbosity >= 1)
            errs() << "BOLT-WARNING: no successor for block " << BB.getName()
                   << " that matches index " << YamlSI.Index << " or block "
                   << ToBB->getName() << '\n';
          ++MismatchedEdges;
          continue;
        }
      }

      BinaryBasicBlock::BinaryBranchInfo &BI = BB.getBranchInfo(*ToBB);
      BI.Count += YamlSI.Count;
      BI.MispredictedCount += YamlSI.Mispreds;
    }
  }

  // If basic block profile wasn't read it should be 0.
  for (BinaryBasicBlock &BB : BF)
    if (BB.getExecutionCount() == BinaryBasicBlock::COUNT_NO_PROFILE)
      BB.setExecutionCount(0);

  if (YamlBP.Header.Flags & BinaryFunction::PF_BASIC)
    BF.setExecutionCount(FunctionExecutionCount);

  ProfileMatched &= !MismatchedBlocks && !MismatchedCalls && !MismatchedEdges;

  if (!ProfileMatched) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: " << MismatchedBlocks << " blocks, "
             << MismatchedCalls << " calls, and " << MismatchedEdges
             << " edges in profile did not match function " << BF << '\n';

    if (YamlBF.NumBasicBlocks != BF.size())
      ++BC.Stats.NumStaleFuncsWithEqualBlockCount;

    if (!opts::InferStaleProfile)
      return false;
    ArrayRef<ProbeMatchSpec> ProbeMatchSpecs;
    auto BFIt = BFToProbeMatchSpecs.find(&BF);
    if (BFIt != BFToProbeMatchSpecs.end())
      ProbeMatchSpecs = BFIt->second;
    ProfileMatched = inferStaleProfile(BF, YamlBF, ProbeMatchSpecs);
  }
  if (ProfileMatched)
    BF.markProfiled(YamlBP.Header.Flags);

  return ProfileMatched;
}

Error YAMLProfileReader::preprocessProfile(BinaryContext &BC) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = MB.getError()) {
    errs() << "ERROR: cannot open " << Filename << ": " << EC.message() << "\n";
    return errorCodeToError(EC);
  }
  yaml::Input YamlInput(MB.get()->getBuffer());
  YamlInput.setAllowUnknownKeys(true);

  // Consume YAML file.
  YamlInput >> YamlBP;
  if (YamlInput.error()) {
    errs() << "BOLT-ERROR: syntax error parsing profile in " << Filename
           << " : " << YamlInput.error().message() << '\n';
    return errorCodeToError(YamlInput.error());
  }

  // Sanity check.
  if (YamlBP.Header.Version != 1)
    return make_error<StringError>(
        Twine("cannot read profile : unsupported version"),
        inconvertibleErrorCode());

  if (YamlBP.Header.EventNames.find(',') != StringRef::npos)
    return make_error<StringError>(
        Twine("multiple events in profile are not supported"),
        inconvertibleErrorCode());

  // Match profile to function based on a function name.
  buildNameMaps(BC);

  // Preliminary assign function execution count.
  for (auto [YamlBF, BF] : llvm::zip_equal(YamlBP.Functions, ProfileBFs)) {
    if (!BF)
      continue;
    if (!BF->hasProfile()) {
      BF->setExecutionCount(YamlBF.ExecCount);
    } else {
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: dropping duplicate profile for " << YamlBF.Name
               << '\n';
      }
      BF = nullptr;
    }
  }

  return Error::success();
}

bool YAMLProfileReader::profileMatches(
    const yaml::bolt::BinaryFunctionProfile &Profile, const BinaryFunction &BF) {
  if (opts::IgnoreHash)
    return Profile.NumBasicBlocks == BF.size();
  return Profile.Hash == static_cast<uint64_t>(BF.getHash());
}

bool YAMLProfileReader::mayHaveProfileData(const BinaryFunction &BF) {
  if (opts::MatchProfileWithFunctionHash || opts::MatchWithCallGraph)
    return true;
  for (StringRef Name : BF.getNames())
    if (ProfileFunctionNames.contains(Name))
      return true;
  for (StringRef Name : BF.getNames()) {
    if (const std::optional<StringRef> CommonName = getLTOCommonName(Name)) {
      if (LTOCommonNameMap.contains(*CommonName))
        return true;
    }
  }

  return false;
}

size_t YAMLProfileReader::matchWithExactName() {
  size_t MatchedWithExactName = 0;
  // This first pass assigns profiles that match 100% by name and by hash.
  for (auto [YamlBF, BF] : llvm::zip_equal(YamlBP.Functions, ProfileBFs)) {
    if (!BF)
      continue;
    BinaryFunction &Function = *BF;
    // Clear function call count that may have been set while pre-processing
    // the profile.
    Function.setExecutionCount(BinaryFunction::COUNT_NO_PROFILE);

    if (profileMatches(YamlBF, Function)) {
      matchProfileToFunction(YamlBF, Function);
      ++MatchedWithExactName;
    }
  }
  return MatchedWithExactName;
}

size_t YAMLProfileReader::matchWithHash(BinaryContext &BC) {
  // Iterates through profiled functions to match the first binary function with
  // the same exact hash. Serves to match identical, renamed functions.
  // Collisions are possible where multiple functions share the same exact hash.
  size_t MatchedWithHash = 0;
  if (opts::MatchProfileWithFunctionHash) {
    DenseMap<size_t, BinaryFunction *> StrictHashToBF;
    StrictHashToBF.reserve(BC.getBinaryFunctions().size());

    for (auto &[_, BF] : BC.getBinaryFunctions())
      StrictHashToBF[BF.getHash()] = &BF;

    for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
      if (YamlBF.Used)
        continue;
      auto It = StrictHashToBF.find(YamlBF.Hash);
      if (It != StrictHashToBF.end() && !ProfiledFunctions.count(It->second)) {
        BinaryFunction *BF = It->second;
        matchProfileToFunction(YamlBF, *BF);
        ++MatchedWithHash;
      }
    }
  }
  return MatchedWithHash;
}

size_t YAMLProfileReader::matchWithLTOCommonName() {
  // This second pass allows name ambiguity for LTO private functions.
  size_t MatchedWithLTOCommonName = 0;
  for (const auto &[CommonName, LTOProfiles] : LTOCommonNameMap) {
    if (!LTOCommonNameFunctionMap.contains(CommonName))
      continue;
    std::unordered_set<BinaryFunction *> &Functions =
        LTOCommonNameFunctionMap[CommonName];
    // Return true if a given profile is matched to one of BinaryFunctions with
    // matching LTO common name.
    auto matchProfile = [&](yaml::bolt::BinaryFunctionProfile *YamlBF) {
      if (YamlBF->Used)
        return false;
      for (BinaryFunction *BF : Functions) {
        if (!ProfiledFunctions.count(BF) && profileMatches(*YamlBF, *BF)) {
          matchProfileToFunction(*YamlBF, *BF);
          ++MatchedWithLTOCommonName;
          return true;
        }
      }
      return false;
    };
    bool ProfileMatched = llvm::any_of(LTOProfiles, matchProfile);

    // If there's only one function with a given name, try to match it
    // partially.
    if (!ProfileMatched && LTOProfiles.size() == 1 && Functions.size() == 1 &&
        !LTOProfiles.front()->Used &&
        !ProfiledFunctions.count(*Functions.begin())) {
      matchProfileToFunction(*LTOProfiles.front(), **Functions.begin());
      ++MatchedWithLTOCommonName;
    }
  }
  return MatchedWithLTOCommonName;
}

size_t YAMLProfileReader::matchWithCallGraph(BinaryContext &BC) {
  if (!opts::MatchWithCallGraph)
    return 0;

  size_t MatchedWithCallGraph = 0;
  CallGraphMatcher CGMatcher(BC, YamlBP, IdToYamLBF);

  ItaniumPartialDemangler Demangler;
  auto GetBaseName = [&](std::string &FunctionName) {
    if (Demangler.partialDemangle(FunctionName.c_str()))
      return std::string("");
    size_t BufferSize = 1;
    char *Buffer = static_cast<char *>(std::malloc(BufferSize));
    char *BaseName = Demangler.getFunctionBaseName(Buffer, &BufferSize);
    if (!BaseName) {
      std::free(Buffer);
      return std::string("");
    }
    if (Buffer != BaseName)
      Buffer = BaseName;
    std::string BaseNameStr(Buffer, BufferSize);
    std::free(Buffer);
    return BaseNameStr;
  };

  // Matches YAMLBF to BFs with neighbor hashes.
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    if (YamlBF.Used)
      continue;
    auto AdjacentYamlBFsOpt = CGMatcher.getAdjacentYamlBFs(YamlBF);
    if (!AdjacentYamlBFsOpt)
      continue;
    std::set<yaml::bolt::BinaryFunctionProfile *> AdjacentYamlBFs =
        AdjacentYamlBFsOpt.value();
    std::string AdjacentYamlBFsHashStr;
    for (auto *AdjacentYamlBF : AdjacentYamlBFs)
      AdjacentYamlBFsHashStr += AdjacentYamlBF->Name;
    uint64_t Hash = std::hash<std::string>{}(AdjacentYamlBFsHashStr);
    auto BFsWithSameHashOpt = CGMatcher.getBFsWithNeighborHash(Hash);
    if (!BFsWithSameHashOpt)
      continue;
    std::vector<BinaryFunction *> BFsWithSameHash = BFsWithSameHashOpt.value();
    // Finds the binary function with the longest common prefix to the profiled
    // function and matches.
    BinaryFunction *ClosestBF = nullptr;
    size_t LCP = 0;
    std::string YamlBFBaseName = GetBaseName(YamlBF.Name);
    for (BinaryFunction *BF : BFsWithSameHash) {
      if (ProfiledFunctions.count(BF))
        continue;
      std::string BFName = std::string(BF->getOneName());
      std::string BFBaseName = GetBaseName(BFName);
      size_t PrefixLength = 0;
      size_t N = std::min(YamlBFBaseName.size(), BFBaseName.size());
      for (size_t I = 0; I < N; ++I) {
        if (YamlBFBaseName[I] != BFBaseName[I])
          break;
        ++PrefixLength;
      }
      if (PrefixLength >= LCP) {
        LCP = PrefixLength;
        ClosestBF = BF;
      }
    }
    if (ClosestBF) {
      matchProfileToFunction(YamlBF, *ClosestBF);
      ++MatchedWithCallGraph;
    }
  }

  return MatchedWithCallGraph;
}

size_t YAMLProfileReader::InlineTreeNodeMapTy::matchInlineTrees(
    const MCPseudoProbeDecoder &Decoder,
    const std::vector<yaml::bolt::InlineTreeNode> &DecodedInlineTree,
    const MCDecodedPseudoProbeInlineTree *Root) {
  // Match inline tree nodes by GUID, checksum, parent, and call site.
  for (const auto &[InlineTreeNodeId, InlineTreeNode] :
       llvm::enumerate(DecodedInlineTree)) {
    uint64_t GUID = InlineTreeNode.GUID;
    uint64_t Hash = InlineTreeNode.Hash;
    uint32_t ParentId = InlineTreeNode.ParentIndexDelta;
    uint32_t CallSiteProbe = InlineTreeNode.CallSiteProbe;
    const MCDecodedPseudoProbeInlineTree *Cur = nullptr;
    if (!InlineTreeNodeId) {
      Cur = Root;
    } else if (const MCDecodedPseudoProbeInlineTree *Parent =
                   getInlineTreeNode(ParentId)) {
      for (const MCDecodedPseudoProbeInlineTree &Child :
           Parent->getChildren()) {
        if (Child.Guid == GUID) {
          if (std::get<1>(Child.getInlineSite()) == CallSiteProbe)
            Cur = &Child;
          break;
        }
      }
    }
    // Don't match nodes if the profile is stale (mismatching binary FuncHash
    // and YAML Hash)
    if (Cur && Decoder.getFuncDescForGUID(Cur->Guid)->FuncHash == Hash)
      mapInlineTreeNode(InlineTreeNodeId, Cur);
  }
  return Map.size();
}

// Decode index deltas and indirection through \p YamlPD. Return modified copy
// of \p YamlInlineTree with populated decoded fields (GUID, Hash, ParentIndex).
static std::vector<yaml::bolt::InlineTreeNode>
decodeYamlInlineTree(const yaml::bolt::ProfilePseudoProbeDesc &YamlPD,
                     std::vector<yaml::bolt::InlineTreeNode> YamlInlineTree) {
  uint32_t ParentId = 0;
  uint32_t PrevGUIDIdx = 0;
  for (yaml::bolt::InlineTreeNode &InlineTreeNode : YamlInlineTree) {
    uint32_t GUIDIdx = InlineTreeNode.GUIDIndex;
    if (GUIDIdx != UINT32_MAX)
      PrevGUIDIdx = GUIDIdx;
    else
      GUIDIdx = PrevGUIDIdx;
    uint32_t HashIdx = YamlPD.GUIDHashIdx[GUIDIdx];
    ParentId += InlineTreeNode.ParentIndexDelta;
    InlineTreeNode.GUID = YamlPD.GUID[GUIDIdx];
    InlineTreeNode.Hash = YamlPD.Hash[HashIdx];
    InlineTreeNode.ParentIndexDelta = ParentId;
  }
  return YamlInlineTree;
}

size_t YAMLProfileReader::matchWithPseudoProbes(BinaryContext &BC) {
  if (!opts::StaleMatchingWithPseudoProbes)
    return 0;

  const MCPseudoProbeDecoder *Decoder = BC.getPseudoProbeDecoder();
  const yaml::bolt::ProfilePseudoProbeDesc &YamlPD = YamlBP.PseudoProbeDesc;

  // Set existing BF->YamlBF match into ProbeMatchSpecs for (local) probe
  // matching.
  assert(Decoder &&
         "If pseudo probes are in use, pseudo probe decoder should exist");
  for (auto [YamlBF, BF] : llvm::zip_equal(YamlBP.Functions, ProfileBFs)) {
    // BF is preliminary name-matched function to YamlBF
    // MatchedBF is final matched function
    BinaryFunction *MatchedBF = YamlProfileToFunction.lookup(YamlBF.Id);
    if (!BF)
      BF = MatchedBF;
    if (!BF)
      continue;
    uint64_t GUID = BF->getGUID();
    if (!GUID)
      continue;
    auto It = TopLevelGUIDToInlineTree.find(GUID);
    if (It == TopLevelGUIDToInlineTree.end())
      continue;
    const MCDecodedPseudoProbeInlineTree *Node = It->second;
    assert(Node && "Malformed TopLevelGUIDToInlineTree");
    auto &MatchSpecs = BFToProbeMatchSpecs[BF];
    auto &InlineTreeMap =
        MatchSpecs.emplace_back(InlineTreeNodeMapTy(), YamlBF).first;
    std::vector<yaml::bolt::InlineTreeNode> ProfileInlineTree =
        decodeYamlInlineTree(YamlPD, YamlBF.InlineTree);
    // Erase unsuccessful match
    if (!InlineTreeMap.matchInlineTrees(*Decoder, ProfileInlineTree, Node))
      MatchSpecs.pop_back();
  }

  return 0;
}

size_t YAMLProfileReader::matchWithNameSimilarity(BinaryContext &BC) {
  if (opts::NameSimilarityFunctionMatchingThreshold == 0)
    return 0;

  size_t MatchedWithNameSimilarity = 0;
  ItaniumPartialDemangler Demangler;

  // Demangle and derive namespace from function name.
  auto DemangleName = [&](std::string &FunctionName) {
    StringRef RestoredName = NameResolver::restore(FunctionName);
    return demangle(RestoredName);
  };
  auto DeriveNameSpace = [&](std::string &DemangledName) {
    if (Demangler.partialDemangle(DemangledName.c_str()))
      return std::string("");
    std::vector<char> Buffer(DemangledName.begin(), DemangledName.end());
    size_t BufferSize;
    char *NameSpace =
        Demangler.getFunctionDeclContextName(&Buffer[0], &BufferSize);
    return std::string(NameSpace, BufferSize);
  };

  // Maps namespaces to associated function block counts and gets profile
  // function names and namespaces to minimize the number of BFs to process and
  // avoid repeated name demangling/namespace derivation.
  StringMap<std::set<uint32_t>> NamespaceToProfiledBFSizes;
  std::vector<std::string> ProfileBFDemangledNames;
  ProfileBFDemangledNames.reserve(YamlBP.Functions.size());
  std::vector<std::string> ProfiledBFNamespaces;
  ProfiledBFNamespaces.reserve(YamlBP.Functions.size());

  for (auto &YamlBF : YamlBP.Functions) {
    std::string YamlBFDemangledName = DemangleName(YamlBF.Name);
    ProfileBFDemangledNames.push_back(YamlBFDemangledName);
    std::string YamlBFNamespace = DeriveNameSpace(YamlBFDemangledName);
    ProfiledBFNamespaces.push_back(YamlBFNamespace);
    NamespaceToProfiledBFSizes[YamlBFNamespace].insert(YamlBF.NumBasicBlocks);
  }

  StringMap<std::vector<BinaryFunction *>> NamespaceToBFs;

  // Maps namespaces to BFs excluding binary functions with no equal sized
  // profiled functions belonging to the same namespace.
  for (BinaryFunction *BF : BC.getAllBinaryFunctions()) {
    std::string DemangledName = BF->getDemangledName();
    std::string Namespace = DeriveNameSpace(DemangledName);

    auto NamespaceToProfiledBFSizesIt =
        NamespaceToProfiledBFSizes.find(Namespace);
    // Skip if there are no ProfileBFs with a given \p Namespace.
    if (NamespaceToProfiledBFSizesIt == NamespaceToProfiledBFSizes.end())
      continue;
    // Skip if there are no ProfileBFs in a given \p Namespace with
    // equal number of blocks.
    if (NamespaceToProfiledBFSizesIt->second.count(BF->size()) == 0)
      continue;
    NamespaceToBFs[Namespace].push_back(BF);
  }

  // Iterates through all profiled functions and binary functions belonging to
  // the same namespace and matches based on edit distance threshold.
  assert(YamlBP.Functions.size() == ProfiledBFNamespaces.size() &&
         ProfiledBFNamespaces.size() == ProfileBFDemangledNames.size());
  for (size_t I = 0; I < YamlBP.Functions.size(); ++I) {
    yaml::bolt::BinaryFunctionProfile &YamlBF = YamlBP.Functions[I];
    std::string &YamlBFNamespace = ProfiledBFNamespaces[I];
    if (YamlBF.Used)
      continue;
    // Skip if there are no BFs in a given \p Namespace.
    auto It = NamespaceToBFs.find(YamlBFNamespace);
    if (It == NamespaceToBFs.end())
      continue;

    std::string &YamlBFDemangledName = ProfileBFDemangledNames[I];
    std::vector<BinaryFunction *> BFs = It->second;
    unsigned MinEditDistance = UINT_MAX;
    BinaryFunction *ClosestNameBF = nullptr;

    // Determines BF the closest to the profiled function, in the
    // same namespace.
    for (BinaryFunction *BF : BFs) {
      if (ProfiledFunctions.count(BF))
        continue;
      if (BF->size() != YamlBF.NumBasicBlocks)
        continue;
      std::string BFDemangledName = BF->getDemangledName();
      unsigned BFEditDistance =
          StringRef(BFDemangledName).edit_distance(YamlBFDemangledName);
      if (BFEditDistance < MinEditDistance) {
        MinEditDistance = BFEditDistance;
        ClosestNameBF = BF;
      }
    }

    if (ClosestNameBF &&
        MinEditDistance <= opts::NameSimilarityFunctionMatchingThreshold) {
      matchProfileToFunction(YamlBF, *ClosestNameBF);
      ++MatchedWithNameSimilarity;
    }
  }

  return MatchedWithNameSimilarity;
}

Error YAMLProfileReader::readProfile(BinaryContext &BC) {
  if (opts::Verbosity >= 1) {
    outs() << "BOLT-INFO: YAML profile with hash: ";
    switch (YamlBP.Header.HashFunction) {
    case HashFunction::StdHash:
      outs() << "std::hash\n";
      break;
    case HashFunction::XXH3:
      outs() << "xxh3\n";
      break;
    }
  }
  YamlProfileToFunction.reserve(YamlBP.Functions.size());

  // Computes hash for binary functions.
  if (opts::MatchProfileWithFunctionHash) {
    for (auto &[_, BF] : BC.getBinaryFunctions()) {
      BF.computeHash(YamlBP.Header.IsDFSOrder, YamlBP.Header.HashFunction);
    }
  } else if (!opts::IgnoreHash) {
    for (BinaryFunction *BF : ProfileBFs) {
      if (!BF)
        continue;
      BF->computeHash(YamlBP.Header.IsDFSOrder, YamlBP.Header.HashFunction);
    }
  }

  if (opts::StaleMatchingWithPseudoProbes) {
    const MCPseudoProbeDecoder *Decoder = BC.getPseudoProbeDecoder();
    assert(Decoder &&
           "If pseudo probes are in use, pseudo probe decoder should exist");
    for (const MCDecodedPseudoProbeInlineTree &TopLev :
         Decoder->getDummyInlineRoot().getChildren())
      TopLevelGUIDToInlineTree[TopLev.Guid] = &TopLev;
  }

  // Map profiled function ids to names.
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions)
    IdToYamLBF[YamlBF.Id] = &YamlBF;

  const size_t MatchedWithExactName = matchWithExactName();
  const size_t MatchedWithHash = matchWithHash(BC);
  const size_t MatchedWithLTOCommonName = matchWithLTOCommonName();
  const size_t MatchedWithCallGraph = matchWithCallGraph(BC);
  const size_t MatchedWithNameSimilarity = matchWithNameSimilarity(BC);
  [[maybe_unused]] const size_t MatchedWithPseudoProbes =
      matchWithPseudoProbes(BC);

  for (auto [YamlBF, BF] : llvm::zip_equal(YamlBP.Functions, ProfileBFs))
    if (!YamlBF.Used && BF && !ProfiledFunctions.count(BF))
      matchProfileToFunction(YamlBF, *BF);


  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions)
    if (!YamlBF.Used && opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: profile ignored for function " << YamlBF.Name
             << '\n';

  if (opts::Verbosity >= 1) {
    outs() << "BOLT-INFO: matched " << MatchedWithExactName
           << " functions with identical names\n";
    outs() << "BOLT-INFO: matched " << MatchedWithHash
           << " functions with hash\n";
    outs() << "BOLT-INFO: matched " << MatchedWithLTOCommonName
           << " functions with matching LTO common names\n";
    outs() << "BOLT-INFO: matched " << MatchedWithCallGraph
           << " functions with call graph\n";
    outs() << "BOLT-INFO: matched " << MatchedWithNameSimilarity
           << " functions with similar names\n";
  }

  // Set for parseFunctionProfile().
  NormalizeByInsnCount = usesEvent("cycles") || usesEvent("instructions");
  NormalizeByCalls = usesEvent("branches");
  uint64_t NumUnused = 0;
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    if (BinaryFunction *BF = YamlProfileToFunction.lookup(YamlBF.Id))
      parseFunctionProfile(*BF, YamlBF);
    else
      ++NumUnused;
  }

  BC.setNumUnusedProfiledObjects(NumUnused);

  if (opts::Lite &&
      (opts::MatchProfileWithFunctionHash || opts::MatchWithCallGraph)) {
    for (BinaryFunction *BF : BC.getAllBinaryFunctions())
      if (!BF->hasProfile())
        BF->setIgnored();
  }

  return Error::success();
}

bool YAMLProfileReader::usesEvent(StringRef Name) const {
  return StringRef(YamlBP.Header.EventNames).contains(Name);
}

} // end namespace bolt
} // end namespace llvm
