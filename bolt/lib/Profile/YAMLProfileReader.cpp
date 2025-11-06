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
#include "bolt/Profile/ProfileYAMLMapping.h"
#include "bolt/Utils/NameResolver.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/Support/CommandLine.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

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

    if (!opts::InferStaleProfile)
      return false;
    ProfileMatched = inferStaleProfile(BF, YamlBF, BFToProbeMatchSpecs[&BF]);
  }
  if (ProfileMatched)
    BF.markProfiled(YamlBP.Header.Flags);

  return ProfileMatched;
}

// Inline tree: decode index deltas and indirection through \p YamlPD and
// set decoded fields (GUID, Hash, ParentIndex).
// Probe inline tree: move InlineTreeIndex into InlineTreeNodes.
static void
decodeYamlInlineTree(const yaml::bolt::ProfilePseudoProbeDesc &YamlPD,
                     yaml::bolt::BinaryFunctionProfile &YamlBF) {
  // Decompress inline tree
  uint32_t ParentId = 0;
  uint32_t PrevGUIDIdx = 0;
  for (yaml::bolt::InlineTreeNode &InlineTreeNode : YamlBF.InlineTree) {
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
  // Decompress probe descriptors
  for (yaml::bolt::BinaryBasicBlockProfile &BB : YamlBF.Blocks) {
    if (BB.PseudoProbesStr.empty())
      continue;
  }
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

  // Preliminary assign function execution count and decode pseudo probe info.
  const yaml::bolt::ProfilePseudoProbeDesc &YamlPD = YamlBP.PseudoProbeDesc;
  YamlGUIDs.insert(YamlPD.GUID.begin(), YamlPD.GUID.end());
  for (auto [YamlBF, BF] : llvm::zip_equal(YamlBP.Functions, ProfileBFs)) {
    decodeYamlInlineTree(YamlPD, YamlBF);
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
  if (opts::StaleMatchingWithPseudoProbes) {
    SmallVector<StringRef, 0> Suffixes(
        {".destroy", ".resume", ".llvm.", ".cold", ".warm"});
    for (const MCSymbol *Sym : BF.getSymbols()) {
      StringRef SymName = Sym->getName();
      for (auto Name : {std::optional(NameResolver::restore(SymName)),
                        getCommonName(SymName, false, Suffixes)}) {
        if (!Name)
          continue;
        SymName = *Name;
        uint64_t GUID = Function::getGUIDAssumingExternalLinkage(SymName);
        if (YamlGUIDs.count(GUID))
          return true;
      }
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

    // Allow mismatching profile when stale matching is enabled.
    if (profileMatches(YamlBF, Function) || opts::InferStaleProfile) {
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

const MCDecodedPseudoProbeInlineTree *
YAMLProfileReader::lookupTopLevelNode(const BinaryFunction &BF) {
  const BinaryContext &BC = BF.getBinaryContext();
  const MCPseudoProbeDecoder *Decoder = BC.getPseudoProbeDecoder();
  assert(Decoder &&
         "If pseudo probes are in use, pseudo probe decoder should exist");
  uint64_t Addr = BF.getAddress();
  uint64_t Size = BF.getSize();
  auto Probes = Decoder->getAddress2ProbesMap().find(Addr, Addr + Size);
  if (Probes.empty())
    return nullptr;
  const MCDecodedPseudoProbe &Probe = *Probes.begin();
  const MCDecodedPseudoProbeInlineTree *Root = Probe.getInlineTreeNode();
  while (Root->hasInlineSite())
    Root = (const MCDecodedPseudoProbeInlineTree *)Root->Parent;
  return Root;
}

size_t YAMLProfileReader::matchInlineTreesImpl(
    BinaryFunction &BF, yaml::bolt::BinaryFunctionProfile &YamlBF,
    const MCDecodedPseudoProbeInlineTree &Root, uint32_t RootIdx,
    ArrayRef<yaml::bolt::InlineTreeNode> ProfileInlineTree,
    MutableArrayRef<const MCDecodedPseudoProbeInlineTree *> Map, float Scale) {
  using namespace yaml::bolt;
  BinaryContext &BC = BF.getBinaryContext();
  const MCPseudoProbeDecoder &Decoder = *BC.getPseudoProbeDecoder();
  const InlineTreeNode &FuncNode = ProfileInlineTree[RootIdx];

  using ChildMapTy =
      std::unordered_map<InlineSite, const MCDecodedPseudoProbeInlineTree *,
                         InlineSiteHash>;
  using CallSiteInfoTy =
      std::unordered_map<InlineSite, const CallSiteInfo *, InlineSiteHash>;
  // Mapping from a parent node id to a map InlineSite -> Child node.
  DenseMap<uint32_t, ChildMapTy> ParentToChildren;
  // Collect calls in the profile: map from a parent node id to a map
  // InlineSite -> CallSiteInfo ptr.
  DenseMap<uint32_t, CallSiteInfoTy> ParentToCSI;
  for (const BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks) {
    // Collect callees for inlined profile matching, indexed by InlineSite.
    for (const CallSiteInfo &CSI : YamlBB.CallSites) {
      ProbeMatchingStats.TotalCallCount += CSI.Count;
      ++ProbeMatchingStats.TotalCallSites;
      if (CSI.Probe == 0) {
        LLVM_DEBUG(dbgs() << "no probe for " << CSI.DestId << " " << CSI.Count
                          << '\n');
        ++ProbeMatchingStats.MissingCallProbe;
        ProbeMatchingStats.MissingCallCount += CSI.Count;
        continue;
      }
      const BinaryFunctionProfile *Callee = IdToYamLBF.lookup(CSI.DestId);
      if (!Callee) {
        LLVM_DEBUG(dbgs() << "no callee for " << CSI.DestId << " " << CSI.Count
                          << '\n');
        ++ProbeMatchingStats.MissingCallee;
        ProbeMatchingStats.MissingCallCount += CSI.Count;
        continue;
      }
      // Get callee GUID
      if (Callee->InlineTree.empty()) {
        LLVM_DEBUG(dbgs() << "no inline tree for " << Callee->Name << '\n');
        ++ProbeMatchingStats.MissingInlineTree;
        ProbeMatchingStats.MissingCallCount += CSI.Count;
        continue;
      }
      uint64_t CalleeGUID = Callee->InlineTree.front().GUID;
      ParentToCSI[CSI.InlineTreeNode][InlineSite(CalleeGUID, CSI.Probe)] = &CSI;
    }
  }
  LLVM_DEBUG({
    for (auto &[ParentId, InlineSiteCSI] : ParentToCSI) {
      for (auto &[InlineSite, CSI] : InlineSiteCSI) {
        auto [CalleeGUID, CallSite] = InlineSite;
        errs() << ParentId << "@" << CallSite << "->"
               << Twine::utohexstr(CalleeGUID) << ": " << CSI->Count << ", "
               << Twine::utohexstr(CSI->Offset) << '\n';
      }
    }
  });

  assert(!Root.isRoot());
  LLVM_DEBUG(dbgs() << "matchInlineTreesImpl for " << BF << "@"
                    << Twine::utohexstr(Root.Guid) << " and " << YamlBF.Name
                    << "@" << Twine::utohexstr(FuncNode.GUID) << '\n');
  ++ProbeMatchingStats.AttemptedNodes;
  ++ProbeMatchingStats.AttemptedRoots;

  // Match profile function with a lead node (top-level function or inlinee)
  if (Root.Guid != FuncNode.GUID) {
    LLVM_DEBUG(dbgs() << "Mismatching root GUID\n");
    ++ProbeMatchingStats.MismatchingRootGUID;
    return 0;
  }
  {
    uint64_t BinaryHash = Decoder.getFuncDescForGUID(Root.Guid)->FuncHash;
    uint64_t ProfileHash = FuncNode.Hash;
    if (BinaryHash != ProfileHash) {
      LLVM_DEBUG(dbgs() << "Mismatching hashes: "
                        << Twine::utohexstr(BinaryHash) << " "
                        << Twine::utohexstr(ProfileHash) << '\n');
      ++ProbeMatchingStats.MismatchingRootHash;
      return 0;
    }
  }
  assert(!Map[RootIdx]);
  Map[RootIdx] = &Root;
  ++ProbeMatchingStats.MatchedRoots;

  uint64_t Matched = 1;
  for (const auto &[Idx, ProfileNode] :
       llvm::drop_begin(llvm::enumerate(ProfileInlineTree), RootIdx + 1)) {
    // Can't match children if parent is not matched.
    uint32_t ParentIdx = ProfileNode.ParentIndexDelta;
    const MCDecodedPseudoProbeInlineTree *Parent = Map[ParentIdx];
    // Exclude nodes with unmatched parent from attempted stats.
    if (!Parent)
      continue;
    ++ProbeMatchingStats.AttemptedNodes;
    if (!ParentToChildren.contains(ParentIdx))
      for (const MCDecodedPseudoProbeInlineTree &Child : Parent->getChildren())
        ParentToChildren[ParentIdx].emplace(InlineSite(Child.getInlineSite()),
                                            &Child);
    ChildMapTy &ChildMap = ParentToChildren[ParentIdx];
    InlineSite Site(ProfileNode.GUID, ProfileNode.CallSiteProbe);
    auto ChildIt = ChildMap.find(Site);
    if (ChildIt == ChildMap.end()) {
      // No binary inline tree node for a given profile inline tree node =>
      // the function is inlined in the profile and is outlined in the binary.
      // Look up the callee binary function and attach profile inline subtree
      // to it. The profile will be additively included into the function.
      LLVM_DEBUG({
        dbgs() << "No binary inline tree for "
               << Twine::utohexstr(ProfileNode.GUID);
        auto G2FDMap = Decoder.getGUID2FuncDescMap();
        auto FDIt = G2FDMap.find(ProfileNode.GUID);
        if (FDIt != G2FDMap.end())
          dbgs() << " " << FDIt->FuncName;
        dbgs() << '\n';
      });
      for (auto It : llvm::make_range(GUIDToBF.equal_range(ProfileNode.GUID))) {
        BinaryFunction *CalleeBF = It.second;
        LLVM_DEBUG(dbgs() << "Found outlined function "
                          << CalleeBF->getPrintName() << '\n');
        matchInlineTrees(*CalleeBF, nullptr, YamlBF, Idx, 1);
      }
      continue;
    }
    const MCDecodedPseudoProbeInlineTree *Child = ChildIt->second;
    uint64_t BinaryHash = Decoder.getFuncDescForGUID(Child->Guid)->FuncHash;
    if (BinaryHash == ProfileNode.Hash) {
      // Match inline trees.
      assert(!Map[Idx]);
      Map[Idx] = Child;
      ++Matched;
    } else {
      LLVM_DEBUG(dbgs() << "Mismatching hashes: "
                        << Twine::utohexstr(BinaryHash) << " "
                        << Twine::utohexstr(ProfileNode.Hash) << '\n');
      ++ProbeMatchingStats.MismatchingNodeHash;
    }
    ChildMap.erase(ChildIt);
  }

  LLVM_DEBUG(dbgs() << "Matching outlined binary inline tree nodes\n");
  // Binary inline trees without correspondence in the profile => nodes are
  // inlined in the binary and outlined in the profile. Match function-level
  // profiles to these nodes. The profile will be scaled down based on call
  // site count in the parent profile.
  for (const auto &[ParentIdx, ChildMap] : ParentToChildren) {
    const auto &CSI = ParentToCSI[ParentIdx];
    for (const auto &[CallSite, Node] : ChildMap) {
      uint64_t GUID = Node->Guid;
      assert(GUID == std::get<0>(CallSite));
      auto ChildCSIIt = CSI.find(CallSite);
      if (ChildCSIIt == CSI.end()) {
        LLVM_DEBUG({
          dbgs() << "Can't find profile call site info for "
                 << Twine::utohexstr(GUID);
          auto G2FDMap = Decoder.getGUID2FuncDescMap();
          auto FDIt = G2FDMap.find(GUID);
          if (FDIt != G2FDMap.end())
            dbgs() << " " << FDIt->FuncName << " H "
                   << Twine::utohexstr(FDIt->FuncHash);
          dbgs() << '\n';
        });
        ++ProbeMatchingStats.MissingProfileNode;
        continue;
      }
      const CallSiteInfo *ChildCSI = ChildCSIIt->second;
      assert(ChildCSI);
      BinaryFunctionProfile *ChildYamlBF = IdToYamLBF.lookup(ChildCSI->DestId);
      assert(ChildYamlBF);
      uint64_t ChildExecCount = ChildYamlBF->ExecCount;
      uint64_t Freq = ChildCSI->Count;
      float ChildScale = ChildExecCount ? 1.f * Freq / ChildExecCount : 1;
      LLVM_DEBUG({
        dbgs() << "Match inlined node " << Twine::utohexstr(GUID)
               << " at call site " << std::get<1>(CallSite) << " in parent id "
               << ParentIdx << " " << Twine::utohexstr(Map[ParentIdx]->Guid)
               << " with inline exec count " << Freq
               << " and outlined exec count " << ChildExecCount << '\n';
      });
      matchInlineTrees(BF, Node, *ChildYamlBF, 0, /*Scale * */ ChildScale);
    }
  }
  LLVM_DEBUG(dbgs() << "matchInlineTreesImpl done for " << BF << "@"
                    << Twine::utohexstr(Root.Guid) << " and " << YamlBF.Name
                    << "@" << Twine::utohexstr(FuncNode.GUID) << ": " << Matched
                    << "/" << Map.size() << '\n');
  ProbeMatchingStats.MatchedNodes += Matched;
  return Matched;
}

void YAMLProfileReader::matchInlineTrees(
    BinaryFunction &BF, const MCDecodedPseudoProbeInlineTree *Node,
    yaml::bolt::BinaryFunctionProfile &YamlBF, uint32_t RootIdx, float Scale) {
  ArrayRef Tree = YamlBF.InlineTree;
  size_t Size = Tree.size();
  if (!Size)
    return;
  BinaryContext &BC = BF.getBinaryContext();
  LLVM_DEBUG(dbgs() << "Match inline trees for " << BF << " @"
                    << (Node ? Twine::utohexstr(Node->Guid) : "root") << " and "
                    << YamlBF.Name << "@" << RootIdx << " with scale " << Scale
                    << '\n');

  if (!Node)
    Node = lookupTopLevelNode(BF);
  if (!Node)
    return;
  RootIdxToMapTy &FuncMatchSpecs = BFToProbeMatchSpecs[&BF][&YamlBF];
  auto MatchIt = FuncMatchSpecs.find(RootIdx);
  if (MatchIt != FuncMatchSpecs.end()) {
    LLVM_DEBUG(dbgs() << "Duplicate match attempt, skip\n");
    return;
  }
  const auto &[It, _] = FuncMatchSpecs.emplace(
      RootIdx, std::pair(InlineTreeNodeMapTy(Size), Scale));
  MutableArrayRef Map = It->second.first;
  size_t N = matchInlineTreesImpl(BF, YamlBF, *Node, RootIdx, Tree, Map, Scale);
  if (N)
    YamlBF.Used = true;
  LLVM_DEBUG({
    const MCPseudoProbeDecoder *Decoder = BC.getPseudoProbeDecoder();
    dbgs() << N << "/" << Size << " match with " << BF << " at ";
    // Print the inline context of head probe
    ListSeparator LS(" inlined into ");
    do {
      dbgs() << LS << Decoder->getFuncDescForGUID(Node->Guid)->FuncName << ":"
             << std::get<1>(Node->getInlineSite());
      Node = (const MCDecodedPseudoProbeInlineTree *)Node->Parent;
    } while (!Node->isRoot());
    dbgs() << '\n';
  });
}

size_t YAMLProfileReader::matchWithPseudoProbes(BinaryContext &BC) {
  if (!opts::StaleMatchingWithPseudoProbes)
    return 0;

  const MCPseudoProbeDecoder *Decoder = BC.getPseudoProbeDecoder();

  // Set existing BF->YamlBF match into ProbeMatchSpecs for (local) probe
  // matching.
  assert(Decoder &&
         "If pseudo probes are in use, pseudo probe decoder should exist");
  for (BinaryFunction *BF : BC.getAllBinaryFunctions())
    if (uint64_t GUID = BF->getGUID())
      GUIDToBF.emplace(GUID, BF);

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
    // Match YamlBF with BF at top-level inline tree level.
    matchInlineTrees(*BF, nullptr, YamlBF, 0, 1);
  }
  // Remove empty match specs
  for (auto &[BF, ProbeMatchSpecs] : BFToProbeMatchSpecs) {
    for (auto PMII = ProbeMatchSpecs.begin(); PMII != ProbeMatchSpecs.end();) {
      RootIdxToMapTy &RootIdxToMap = PMII->second;
      for (auto II = RootIdxToMap.begin(); II != RootIdxToMap.end();) {
        if (llvm::any_of(II->second.first, [](auto *P) { return P; }))
          ++II;
        else
          II = RootIdxToMap.erase(II);
      }
      if (RootIdxToMap.empty())
        PMII = ProbeMatchSpecs.erase(PMII);
      else
        ++PMII;
    }
    // Don't drop empty BFToProbeMatchSpecs - used during matching
  }
  outs() << "BOLT-INFO: pseudo probe profile matching matched "
         << formatv("{0:p} ({1}/{2}) ",
                    1.0 * ProbeMatchingStats.MatchedRoots /
                        ProbeMatchingStats.AttemptedRoots,
                    ProbeMatchingStats.MatchedRoots,
                    ProbeMatchingStats.AttemptedRoots)
         << "inline tree root(s) corresponding to profile or binary functions. "

         << "Couldn't match " << ProbeMatchingStats.MismatchingRootGUID
         << " roots with mismatching GUIDs, "
         << ProbeMatchingStats.MismatchingRootHash
         << " with mismatching hash.\n"

         << "BOLT-INFO: pseudo probe profile matching matched "
         << formatv("{0:p} ({1}/{2}) ",
                    1.0 * ProbeMatchingStats.MatchedNodes /
                        ProbeMatchingStats.AttemptedNodes,
                    ProbeMatchingStats.MatchedNodes,
                    ProbeMatchingStats.AttemptedNodes)
         << "inline tree node(s) corresponding to inlined source functions. "

         << "Couldn't match " << ProbeMatchingStats.MismatchingNodeHash
         << " nodes with mismatching hash, "
         << ProbeMatchingStats.MissingProfileNode
         << " with missing profile node.\n"

         << "BOLT-INFO: the profile has "
         << formatv("{0:p} ({1}/{2}) calls missing probe, ",
                    1.0 * ProbeMatchingStats.MissingCallProbe /
                        ProbeMatchingStats.TotalCallSites,
                    ProbeMatchingStats.MissingCallProbe,
                    ProbeMatchingStats.TotalCallSites)
         << formatv("{0:p} ({1}/{2}) calls missing callee, ",
                    1.0 * ProbeMatchingStats.MissingCallee /
                        ProbeMatchingStats.TotalCallSites,
                    ProbeMatchingStats.MissingCallee,
                    ProbeMatchingStats.TotalCallSites)
         << formatv("{0:p} ({1}/{2}) calls with callees missing inline trees, ",
                    1.0 * ProbeMatchingStats.MissingInlineTree /
                        ProbeMatchingStats.TotalCallSites,
                    ProbeMatchingStats.MissingInlineTree,
                    ProbeMatchingStats.TotalCallSites)
         << formatv("covering a total of {0:p} ({1}/{2}) call counts\n",
                    1.0 * ProbeMatchingStats.MissingCallCount /
                        ProbeMatchingStats.TotalCallCount,
                    ProbeMatchingStats.MissingCallCount,
                    ProbeMatchingStats.TotalCallCount);

  return BFToProbeMatchSpecs.size();
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

  // Map profiled function ids to names.
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions)
    IdToYamLBF[YamlBF.Id] = &YamlBF;

  const size_t MatchedWithExactName = matchWithExactName();
  const size_t MatchedWithHash = matchWithHash(BC);
  const size_t MatchedWithLTOCommonName = matchWithLTOCommonName();
  const size_t MatchedWithCallGraph = matchWithCallGraph(BC);
  const size_t MatchedWithNameSimilarity = matchWithNameSimilarity(BC);
  const size_t MatchedWithPseudoProbes = matchWithPseudoProbes(BC);

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
    outs() << "BOLT-INFO: matched " << MatchedWithPseudoProbes
           << " functions with pseudo probes\n";
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
      NumUnused += !YamlBF.Used;
  }

  // Partial matching: pass phony profile for binary functions without
  // corresponding real (non-partial) profile.
  yaml::bolt::BinaryFunctionProfile Phony;
  Phony.ExecCount = 1;
  for (BinaryFunction *BF : llvm::make_first_range(BFToProbeMatchSpecs))
    if (!ProfiledFunctions.count(BF))
      parseFunctionProfile(*BF, Phony);

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
