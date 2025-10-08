//===- bolt/Profile/YAMLProfileReader.cpp - YAML profile de-serializer ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/YAMLProfileReader.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/MCF.h"
#include "bolt/Profile/ProfileYAMLMapping.h"
#include "bolt/Utils/NameResolver.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/CoalescingBitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/edit_distance.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

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

llvm::cl::opt<bool>
    MatchWithPseudoProbes("match-with-pseudo-probes",
                          cl::desc("Match functions with pseudo probes"),
                          cl::Hidden, cl::cat(BoltOptCategory));

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

  std::vector<std::pair<StringRef, uint64_t>> NameToExecCount(
      YamlBP.Functions.size());
  // Map profiled function ids to names.
  IdToYamlBF.reserve(YamlBP.Functions.size());
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    IdToYamlBF[YamlBF.Id] = &YamlBF;
    NameToExecCount.emplace_back(YamlBF.Name, YamlBF.ExecCount);
  }
  llvm::sort(NameToExecCount, llvm::less_second());
  outs() << "Top 10 functions in the profile:\n";
  for (auto [Id, NameExecCount] :
       llvm::enumerate(llvm::reverse(NameToExecCount))) {
    outs() << NameExecCount.second << " " << NameExecCount.first << '\n';
    if (Id == 10)
      break;
  }

#ifndef MAX_PATH
#define MAX_PATH 255
#endif

  for (auto YamlBF : YamlBP.Functions) {
    std::error_code EC;
    std::string Filename = YamlBF.Name;
    StringRef Suffix(".yaml.dot");
    llvm::replace(Filename, '/', '-');
    if (Filename.size() + Suffix.size() > MAX_PATH)
      Filename.resize(MAX_PATH - Suffix.size());
    Filename += Suffix;

    raw_fd_ostream Of(Filename, EC, sys::fs::OF_None);
    if (EC) {
      BC.errs() << "Can't open " << Filename << " for output: " << EC.message()
                << "\n";
      continue;
    }
    dumpGraph(BC, Of, YamlBF);
  }

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

  // Construct GUID->YamlBF mapping
  if (!opts::MatchWithPseudoProbes)
    return Error::success();
  GUIDMap.reserve(YamlBP.Functions.size());
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    if (YamlBF.InlineTree.empty())
      continue;
    const yaml::bolt::InlineTreeNode &Node = YamlBF.InlineTree.front();
    uint32_t GUIDIdx = Node.GUIDIndex == UINT32_MAX ? 0 : Node.GUIDIndex;
    uint64_t GUID = YamlBP.PseudoProbeDesc.GUID[GUIDIdx];
    GUIDMap[GUID] = &YamlBF;
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
  if (!opts::MatchWithPseudoProbes)
    return false;

  SmallVector<StringRef> Suffixes(
      {".destroy", ".resume", ".llvm.", ".cold", ".warm"});
  for (const MCSymbol *Sym : BF.getSymbols()) {
    StringRef SymName = Sym->getName();
    for (auto Name : {std::optional(NameResolver::restore(SymName)),
                      getCommonName(SymName, false, Suffixes)}) {
      if (!Name)
        continue;
      SymName = *Name;
      uint64_t GUID = Function::getGUIDAssumingExternalLinkage(SymName);
      if (GUIDMap.count(GUID))
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
  CallGraphMatcher CGMatcher(BC, YamlBP, IdToYamlBF);

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

void YAMLProfileReader::dumpGraph(
    BinaryContext &BC, raw_ostream &OS,
    const yaml::bolt::BinaryFunctionProfile &BF) const {
  const yaml::bolt::ProfilePseudoProbeDesc &YamlPD = YamlBP.PseudoProbeDesc;
  std::vector<yaml::bolt::InlineTreeNode> ProfileInlineTree =
      decodeYamlInlineTree(YamlPD, BF.InlineTree);
  OS << "digraph \"" << BF.Name << "\" {\n"
     << "node [fontname=courier, shape=box, style=filled, colorscheme=brbg9]\n";
  for (const auto &BB : BF.Blocks) {
    OS << format("\"%d\" [label=\"%d\\n(C:%lu,H:%llx,I:%u)\\n", BB.Index,
                 BB.Index, BB.ExecCount, (uint64_t)BB.Hash, BB.NumInstructions);
    for (const auto &Callee : BB.CallSites) {
      OS << formatv("call {} C:{},M:{},O:{},E:{}\\n", Callee.DestId,
                    Callee.Count, Callee.Mispreds, Callee.Offset,
                    Callee.EntryDiscriminator);
    }
    OS << "probes:\\n";
    for (const auto &Probe : BB.PseudoProbes) {
      OS << "node";
      if (!Probe.InlineTreeNodes.empty()) {
        OS << "s ";
        CoalescingBitVector<uint32_t>::Allocator Alloc;
        CoalescingBitVector<uint32_t> BV(Alloc);
        for (uint64_t Node : Probe.InlineTreeNodes)
          BV.set(Node);
        BV.print(OS);
      } else
        OS << " " << Probe.InlineTreeIndex;
      OS << ": ";
      if (Probe.BlockMask || !Probe.BlockProbes.empty()) {
        OS << "blocks [";
        ListSeparator LS;
        for (uint64_t Index = 0; Index < 64; ++Index)
          if (Probe.BlockMask & 1ull << Index)
            OS << LS << Index + 1;
        for (auto Block : Probe.BlockProbes)
          OS << LS << Block;
        OS << "]";
      }
      if (!Probe.CallProbes.empty()) {
        OS << ", calls [";
        {
          ListSeparator LS;
          for (uint64_t Call : Probe.CallProbes)
            OS << LS << Call;
        }
        OS << "]";
      }
      if (!Probe.IndCallProbes.empty()) {
        OS << ", indcalls [";
        ListSeparator LS;
        for (uint64_t Call : Probe.IndCallProbes)
          OS << LS << Call;
        OS << "]";
      }
      OS << "\\n";
    }
    OS << "\"]\n";
    for (const yaml::bolt::SuccessorInfo &Succ : BB.Successors)
      OS << format("\"%d\" -> \"%d\" [label=\"%d\\n(C:%d,M:%d)\"]\n", BB.Index,
                   Succ.Index, Succ.Index, Succ.Count, Succ.Mispreds);
  }
  // Inline tree
  for (auto [Idx, Node] : llvm::enumerate(ProfileInlineTree)) {
    OS << format("\"IT%d\" [label=\"IT%d\\nG %llx\\nH %llx\"]\n",
        Idx, Idx, (uint64_t)Node.GUID, (uint64_t)Node.Hash);
    OS << format("\"IT%d\" -> \"IT%d\" [label=\"@%d\"]\n", Node.ParentIndexDelta,
                 Idx, Node.CallSiteProbe);
  }
  OS << "}\n";
}

size_t YAMLProfileReader::matchWithPseudoProbes(BinaryContext &BC) {
  if (!opts::StaleMatchingWithPseudoProbes && !opts::MatchWithPseudoProbes)
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

  if (!opts::MatchWithPseudoProbes)
    return 0;

  size_t MatchedWithPseudoProbes = 0;
  assert(Decoder &&
         "If pseudo probes are in use, pseudo probe decoder should exist");

  const auto &GUID2FuncDescMap = Decoder->getGUID2FuncDescMap();

  // Construct the mapping between GUID (non-toplev) and inline tree
  std::vector<std::pair<uint64_t, const MCDecodedPseudoProbeInlineTree *>>
      GUIDInlineTree;
  size_t TopLevCount = Decoder->getDummyInlineRoot().getChildren().size();
  for (const MCDecodedPseudoProbeInlineTree &Node :
       llvm::drop_begin(Decoder->getInlineTreeVec(), TopLevCount))
    GUIDInlineTree.emplace_back(Node.Guid, &Node);
  llvm::sort(GUIDInlineTree);

  // Construct the mapping from profile GUID to profile hash.
  std::unordered_map<uint64_t, uint64_t> ProfileGUIDHash;
  ProfileGUIDHash.reserve(YamlPD.GUID.size());
  for (const auto &[GUID, HashIdx] : llvm::zip(YamlPD.GUID, YamlPD.GUIDHashIdx))
    ProfileGUIDHash.emplace(GUID, YamlPD.Hash[HashIdx]);

  std::unordered_map<uint64_t, BinaryFunction *> GUIDToBF;
  for (BinaryFunction *BF : BC.getAllBinaryFunctions())
    if (uint64_t GUID = BF->getGUID())
      GUIDToBF[GUID] = BF;

  for (auto &[YamlGUID, YamlBFPtr] : GUIDMap) {
    LLVM_DEBUG(dbgs() << "Matching GUID " << Twine::utohexstr(YamlGUID)
                      << '\n');
    yaml::bolt::BinaryFunctionProfile &YamlBF = *YamlBFPtr;
    if (YamlBF.Used)
      continue;
    LLVM_DEBUG(dbgs() << "Attempting to match " << YamlBF.Name
                      << " using pseudo probes:\n");

    // Look up corresponding GUID in the binary.
    auto It = GUID2FuncDescMap.find(YamlGUID);
    if (It == GUID2FuncDescMap.end()) {
      LLVM_DEBUG(dbgs() << "no function with GUID=" << YamlGUID
                        << " in the binary\n");
      continue;
    }

    // Check if checksums match between profile and binary.
    uint64_t YamlHash = ProfileGUIDHash[YamlGUID];
    if (YamlHash != It->FuncHash) {
      LLVM_DEBUG(dbgs() << "hash mismatch\n");
      continue;
    }

    // Look for binary inline trees with match to YAML inline tree.
    auto Range = llvm::make_range(std::equal_range(
        GUIDInlineTree.begin(), GUIDInlineTree.end(),
        std::make_pair(YamlGUID,
                       (const MCDecodedPseudoProbeInlineTree *)nullptr),
        llvm::less_first()));

    size_t Matched = 0;
    for (const MCDecodedPseudoProbeInlineTree *Node :
         llvm::make_second_range(Range)) {
      // Find top-level function containing Node.
      const MCDecodedPseudoProbeInlineTree *Root = Node;
      while (Root->hasInlineSite())
        Root = (const MCDecodedPseudoProbeInlineTree *)Root->Parent;
      assert(Root && "Invalid pseudo probe inline tree");
      // Get binary function corresponding to that top-level function
      auto BFIt = GUIDToBF.find(Root->Guid);
      if (BFIt == GUIDToBF.end())
        continue;
      BinaryFunction *BF = BFIt->second;
      assert(BF && "Invalid GUIDToBF mapping");

      auto &BFMatchSpecs = BFToProbeMatchSpecs[BF];
      InlineTreeNodeMapTy &InlineTreeMap =
          BFMatchSpecs.emplace_back(InlineTreeNodeMapTy(), YamlBF).first;
      size_t MatchedNodes =
          InlineTreeMap.matchInlineTrees(*Decoder, YamlBF.InlineTree, Node);
      // Skip if inline tree didn't match
      if (!MatchedNodes)
        BFMatchSpecs.pop_back();
      ++Matched;
      const auto ProbeIt = Node->getProbes().begin();
      const auto *Probe =
          (ProbeIt != Node->getProbes().end()) ? &*ProbeIt : nullptr;
      LLVM_DEBUG(dbgs() << MatchedNodes << "/" << YamlBF.InlineTree.size()
                        << " match with " << *BF << " at "
                        << (Probe ? Probe->getInlineContextStr(GUID2FuncDescMap)
                                  : "(none)")
                        << '\n');
    }
    MatchedWithPseudoProbes += !!Matched;
    YamlBF.Used |= !!Matched;
    LLVM_DEBUG(dbgs() << "matched to " << Matched << " functions\n");
    LLVM_DEBUG(dbgs() << "done\n");
  }

  return MatchedWithPseudoProbes;
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

  if (opts::MatchWithPseudoProbes || opts::StaleMatchingWithPseudoProbes) {
    const MCPseudoProbeDecoder *Decoder = BC.getPseudoProbeDecoder();
    assert(Decoder &&
           "If pseudo probes are in use, pseudo probe decoder should exist");
    for (const MCDecodedPseudoProbeInlineTree &TopLev :
         Decoder->getDummyInlineRoot().getChildren())
      TopLevelGUIDToInlineTree[TopLev.Guid] = &TopLev;
  }

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

  if (opts::Verbosity >= 0) {
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
