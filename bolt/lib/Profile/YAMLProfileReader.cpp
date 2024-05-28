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
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace opts {

extern cl::opt<unsigned> Verbosity;
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<bool> InferStaleProfile;

static llvm::cl::opt<bool>
    IgnoreHash("profile-ignore-hash",
               cl::desc("ignore hash while reading function profile"),
               cl::Hidden, cl::cat(BoltOptCategory));

llvm::cl::opt<bool> ProfileUseDFS("profile-use-dfs",
                                  cl::desc("use DFS order for YAML profile"),
                                  cl::Hidden, cl::cat(BoltOptCategory));
} // namespace opts

namespace llvm {
namespace bolt {

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

  uint64_t FuncRawBranchCount = 0;
  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks)
    for (const yaml::bolt::SuccessorInfo &YamlSI : YamlBB.Successors)
      FuncRawBranchCount += YamlSI.Count;
  BF.setRawBranchCount(FuncRawBranchCount);

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
    if (YamlBP.Header.Flags & BinaryFunction::PF_SAMPLE) {
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
      BinaryFunction *Callee = YamlCSI.DestId < YamlProfileToFunction.size()
                                   ? YamlProfileToFunction[YamlCSI.DestId]
                                   : nullptr;
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

  if (YamlBP.Header.Flags & BinaryFunction::PF_SAMPLE)
    BF.setExecutionCount(FunctionExecutionCount);

  ProfileMatched &= !MismatchedBlocks && !MismatchedCalls && !MismatchedEdges;

  if (!ProfileMatched) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: " << MismatchedBlocks << " blocks, "
             << MismatchedCalls << " calls, and " << MismatchedEdges
             << " edges in profile did not match function " << BF << '\n';

    if (YamlBF.NumBasicBlocks != BF.size())
      ++BC.Stats.NumStaleFuncsWithEqualBlockCount;

    if (opts::InferStaleProfile && inferStaleProfile(BF, YamlBF))
      ProfileMatched = true;
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

bool YAMLProfileReader::mayHaveProfileData(const BinaryFunction &BF) {
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
  YamlProfileToFunction.resize(YamlBP.Functions.size() + 1);

  auto profileMatches = [](const yaml::bolt::BinaryFunctionProfile &Profile,
                           BinaryFunction &BF) {
    if (opts::IgnoreHash)
      return Profile.NumBasicBlocks == BF.size();
    return Profile.Hash == static_cast<uint64_t>(BF.getHash());
  };

  // We have to do 2 passes since LTO introduces an ambiguity in function
  // names. The first pass assigns profiles that match 100% by name and
  // by hash. The second pass allows name ambiguity for LTO private functions.
  for (auto [YamlBF, BF] : llvm::zip_equal(YamlBP.Functions, ProfileBFs)) {
    if (!BF)
      continue;
    BinaryFunction &Function = *BF;
    // Clear function call count that may have been set while pre-processing
    // the profile.
    Function.setExecutionCount(BinaryFunction::COUNT_NO_PROFILE);

    // Recompute hash once per function.
    if (!opts::IgnoreHash)
      Function.computeHash(YamlBP.Header.IsDFSOrder,
                           YamlBP.Header.HashFunction);

    if (profileMatches(YamlBF, Function))
      matchProfileToFunction(YamlBF, Function);
  }

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
        !ProfiledFunctions.count(*Functions.begin()))
      matchProfileToFunction(*LTOProfiles.front(), **Functions.begin());
  }

  for (auto [YamlBF, BF] : llvm::zip_equal(YamlBP.Functions, ProfileBFs))
    if (!YamlBF.Used && BF && !ProfiledFunctions.count(BF))
      matchProfileToFunction(YamlBF, *BF);

  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions)
    if (!YamlBF.Used && opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: profile ignored for function " << YamlBF.Name
             << '\n';

  // Set for parseFunctionProfile().
  NormalizeByInsnCount = usesEvent("cycles") || usesEvent("instructions");
  NormalizeByCalls = usesEvent("branches");

  uint64_t NumUnused = 0;
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    if (YamlBF.Id >= YamlProfileToFunction.size()) {
      // Such profile was ignored.
      ++NumUnused;
      continue;
    }
    if (BinaryFunction *BF = YamlProfileToFunction[YamlBF.Id])
      parseFunctionProfile(*BF, YamlBF);
    else
      ++NumUnused;
  }

  BC.setNumUnusedProfiledObjects(NumUnused);

  return Error::success();
}

bool YAMLProfileReader::usesEvent(StringRef Name) const {
  return YamlBP.Header.EventNames.find(std::string(Name)) != StringRef::npos;
}

} // end namespace bolt
} // end namespace llvm
