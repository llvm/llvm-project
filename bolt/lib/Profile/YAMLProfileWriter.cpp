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
#include "bolt/Profile/ProfileReaderBase.h"
#include "bolt/Profile/ProfileYAMLMapping.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

namespace llvm {
namespace bolt {

namespace {
void convert(const BinaryFunction &BF,
             yaml::bolt::BinaryFunctionProfile &YamlBF) {
  const BinaryContext &BC = BF.getBinaryContext();

  const uint16_t LBRProfile = BF.getProfileFlags() & BinaryFunction::PF_LBR;

  YamlBF.Name = BF.getPrintName();
  YamlBF.Id = BF.getFunctionNumber();
  YamlBF.Hash = BF.computeHash(/*UseDFS=*/true);
  YamlBF.NumBasicBlocks = BF.size();
  YamlBF.ExecCount = BF.getKnownExecutionCount();

  for (const BinaryBasicBlock *BB : BF.dfs()) {
    yaml::bolt::BinaryBasicBlockProfile YamlBB;
    YamlBB.Index = BB->getLayoutIndex();
    YamlBB.NumInstructions = BB->getNumNonPseudos();

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

      yaml::bolt::CallSiteInfo CSI;
      Optional<uint32_t> Offset = BC.MIB->getOffset(Instr);
      if (!Offset || *Offset < BB->getInputOffset())
        continue;
      CSI.Offset = *Offset - BB->getInputOffset();

      if (BC.MIB->isIndirectCall(Instr) || BC.MIB->isIndirectBranch(Instr)) {
        const auto ICSP = BC.MIB->tryGetAnnotationAs<IndirectCallSiteProfile>(
            Instr, "CallProfile");
        if (!ICSP)
          continue;
        for (const IndirectCallProfile &CSP : ICSP.get()) {
          CSI.DestId = 0; // designated for unknown functions
          CSI.EntryDiscriminator = 0;
          if (CSP.Symbol) {
            const BinaryFunction *Callee = BC.getFunctionForSymbol(CSP.Symbol);
            if (Callee)
              CSI.DestId = Callee->getFunctionNumber();
          }
          CSI.Count = CSP.Count;
          CSI.Mispreds = CSP.Mispreds;
          YamlBB.CallSites.push_back(CSI);
        }
      } else { // direct call or a tail call
        uint64_t EntryID = 0;
        const MCSymbol *CalleeSymbol = BC.MIB->getTargetSymbol(Instr);
        const BinaryFunction *const Callee =
            BC.getFunctionForSymbol(CalleeSymbol, &EntryID);
        if (Callee) {
          CSI.DestId = Callee->getFunctionNumber();
          CSI.EntryDiscriminator = EntryID;
        }

        if (BC.MIB->getConditionalTailCall(Instr)) {
          auto CTCCount =
              BC.MIB->tryGetAnnotationAs<uint64_t>(Instr, "CTCTakenCount");
          if (CTCCount) {
            CSI.Count = *CTCCount;
            auto CTCMispreds =
                BC.MIB->tryGetAnnotationAs<uint64_t>(Instr, "CTCMispredCount");
            if (CTCMispreds)
              CSI.Mispreds = *CTCMispreds;
          }
        } else {
          auto Count = BC.MIB->tryGetAnnotationAs<uint64_t>(Instr, "Count");
          if (Count)
            CSI.Count = *Count;
        }

        if (CSI.Count)
          YamlBB.CallSites.emplace_back(CSI);
      }
    }

    llvm::sort(YamlBB.CallSites);

    // Skip printing if there's no profile data for non-entry basic block.
    // Include landing pads with non-zero execution count.
    if (YamlBB.CallSites.empty() && !BB->isEntryPoint() &&
        !(BB->isLandingPad() && BB->getKnownExecutionCount() != 0)) {
      uint64_t SuccessorExecCount = 0;
      for (const BinaryBasicBlock::BinaryBranchInfo &BranchInfo :
           BB->branch_info())
        SuccessorExecCount += BranchInfo.Count;
      if (!SuccessorExecCount)
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

    YamlBF.Blocks.emplace_back(YamlBB);
  }
}
} // end anonymous namespace

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
  Optional<StringRef> BuildID = BC.getFileBuildID();
  BP.Header.Id = BuildID ? std::string(*BuildID) : "<unknown>";
  BP.Header.Origin = std::string(RI.getProfileReader()->getReaderName());

  StringSet<> EventNames = RI.getProfileReader()->getEventNames();
  if (!EventNames.empty()) {
    std::string Sep;
    for (const StringMapEntry<std::nullopt_t> &EventEntry : EventNames) {
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

  // Add all function objects.
  for (const auto &BFI : Functions) {
    const BinaryFunction &BF = BFI.second;
    if (BF.hasProfile()) {
      if (!BF.hasValidProfile() && !RI.getProfileReader()->isTrustedSource())
        continue;

      yaml::bolt::BinaryFunctionProfile YamlBF;
      convert(BF, YamlBF);
      BP.Functions.emplace_back(YamlBF);
    }
  }

  // Write the profile.
  yaml::Output Out(*OS, nullptr, 0);
  Out << BP;

  return std::error_code();
}

} // namespace bolt
} // namespace llvm
