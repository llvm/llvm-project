//===- bolt/Profile/DataReader.cpp - Perf data reader ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions reads profile data written by the perf2bolt
// utility and stores it in memory for llvm-bolt consumption.
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/DataReader.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/MCF.h"
#include "bolt/Utils/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltCategory;
extern llvm::cl::opt<unsigned> Verbosity;

static cl::opt<bool>
DumpData("dump-data",
  cl::desc("dump parsed bolt data for debugging"),
  cl::Hidden,
  cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

namespace {

/// Return true if the function name can change across compilations.
bool hasVolatileName(const BinaryFunction &BF) {
  for (const StringRef &Name : BF.getNames())
    if (getLTOCommonName(Name))
      return true;

  return false;
}

/// Return standard escaped name of the function possibly renamed by BOLT.
std::string normalizeName(StringRef NameRef) {
  // Strip "PG." prefix used for globalized locals.
  NameRef = NameRef.starts_with("PG.") ? NameRef.substr(2) : NameRef;
  return getEscapedName(NameRef);
}

} // anonymous namespace

raw_ostream &operator<<(raw_ostream &OS, const Location &Loc) {
  if (Loc.IsSymbol) {
    OS << Loc.Name;
    if (Loc.Offset)
      OS << "+" << Twine::utohexstr(Loc.Offset);
  } else {
    OS << Twine::utohexstr(Loc.Offset);
  }
  return OS;
}

void FuncBranchData::appendFrom(const FuncBranchData &FBD, uint64_t Offset) {
  Data.insert(Data.end(), FBD.Data.begin(), FBD.Data.end());
  for (auto I = Data.begin(), E = Data.end(); I != E; ++I) {
    if (I->From.Name == FBD.Name) {
      I->From.Name = this->Name;
      I->From.Offset += Offset;
    }
    if (I->To.Name == FBD.Name) {
      I->To.Name = this->Name;
      I->To.Offset += Offset;
    }
  }
  llvm::stable_sort(Data);
  ExecutionCount += FBD.ExecutionCount;
  ExternEntryCount += FBD.ExternEntryCount;
  for (auto I = FBD.EntryData.begin(), E = FBD.EntryData.end(); I != E; ++I) {
    assert(I->To.Name == FBD.Name);
    auto NewElmt = EntryData.insert(EntryData.end(), *I);
    NewElmt->To.Name = this->Name;
    NewElmt->To.Offset += Offset;
  }
}

uint64_t FuncBranchData::getNumExecutedBranches() const {
  uint64_t ExecutedBranches = 0;
  for (const BranchInfo &BI : Data) {
    int64_t BranchCount = BI.Branches;
    assert(BranchCount >= 0 && "branch execution count should not be negative");
    ExecutedBranches += BranchCount;
  }
  return ExecutedBranches;
}

void BasicSampleInfo::mergeWith(const BasicSampleInfo &SI) { Hits += SI.Hits; }

void BasicSampleInfo::print(raw_ostream &OS) const {
  OS << Loc.IsSymbol << " " << Loc.Name << " " << Twine::utohexstr(Loc.Offset)
     << " " << Hits << "\n";
}

uint64_t FuncBasicSampleData::getSamples(uint64_t Start, uint64_t End) const {
  assert(llvm::is_sorted(Data));
  struct Compare {
    bool operator()(const BasicSampleInfo &SI, const uint64_t Val) const {
      return SI.Loc.Offset < Val;
    }
    bool operator()(const uint64_t Val, const BasicSampleInfo &SI) const {
      return Val < SI.Loc.Offset;
    }
  };
  uint64_t Result = 0;
  for (auto I = llvm::lower_bound(Data, Start, Compare()),
            E = llvm::lower_bound(Data, End, Compare());
       I != E; ++I)
    Result += I->Hits;
  return Result;
}

uint64_t FuncBasicSampleData::getSamples() const {
  uint64_t Result = 0;
  for (const BasicSampleInfo &I : Data)
    Result += I.Hits;
  return Result;
}

void FuncBasicSampleData::bumpCount(uint64_t Offset, uint64_t Count) {
  auto Iter = Index.find(Offset);
  if (Iter == Index.end()) {
    Data.emplace_back(Location(true, Name, Offset), Count);
    Index[Offset] = Data.size() - 1;
    return;
  }
  BasicSampleInfo &SI = Data[Iter->second];
  SI.Hits += Count;
}

void FuncBranchData::bumpBranchCount(uint64_t OffsetFrom, uint64_t OffsetTo,
                                     uint64_t Count, uint64_t Mispreds) {
  auto Iter = IntraIndex[OffsetFrom].find(OffsetTo);
  if (Iter == IntraIndex[OffsetFrom].end()) {
    Data.emplace_back(Location(true, Name, OffsetFrom),
                      Location(true, Name, OffsetTo), Mispreds, Count);
    IntraIndex[OffsetFrom][OffsetTo] = Data.size() - 1;
    return;
  }
  BranchInfo &BI = Data[Iter->second];
  BI.Branches += Count;
  BI.Mispreds += Mispreds;
}

void FuncBranchData::bumpCallCount(uint64_t OffsetFrom, const Location &To,
                                   uint64_t Count, uint64_t Mispreds) {
  auto Iter = InterIndex[OffsetFrom].find(To);
  if (Iter == InterIndex[OffsetFrom].end()) {
    Data.emplace_back(Location(true, Name, OffsetFrom), To, Mispreds, Count);
    InterIndex[OffsetFrom][To] = Data.size() - 1;
    return;
  }
  BranchInfo &BI = Data[Iter->second];
  BI.Branches += Count;
  BI.Mispreds += Mispreds;
}

void FuncBranchData::bumpEntryCount(const Location &From, uint64_t OffsetTo,
                                    uint64_t Count, uint64_t Mispreds) {
  auto Iter = EntryIndex[OffsetTo].find(From);
  if (Iter == EntryIndex[OffsetTo].end()) {
    EntryData.emplace_back(From, Location(true, Name, OffsetTo), Mispreds,
                           Count);
    EntryIndex[OffsetTo][From] = EntryData.size() - 1;
    return;
  }
  BranchInfo &BI = EntryData[Iter->second];
  BI.Branches += Count;
  BI.Mispreds += Mispreds;
}

void BranchInfo::mergeWith(const BranchInfo &BI) {
  Branches += BI.Branches;
  Mispreds += BI.Mispreds;
}

void BranchInfo::print(raw_ostream &OS) const {
  OS << From.IsSymbol << " " << From.Name << " "
     << Twine::utohexstr(From.Offset) << " " << To.IsSymbol << " " << To.Name
     << " " << Twine::utohexstr(To.Offset) << " " << Mispreds << " " << Branches
     << '\n';
}

ErrorOr<const BranchInfo &> FuncBranchData::getBranch(uint64_t From,
                                                      uint64_t To) const {
  for (const BranchInfo &I : Data)
    if (I.From.Offset == From && I.To.Offset == To && I.From.Name == I.To.Name)
      return I;

  return make_error_code(llvm::errc::invalid_argument);
}

ErrorOr<const BranchInfo &>
FuncBranchData::getDirectCallBranch(uint64_t From) const {
  // Commented out because it can be expensive.
  // assert(std::is_sorted(Data.begin(), Data.end()));
  struct Compare {
    bool operator()(const BranchInfo &BI, const uint64_t Val) const {
      return BI.From.Offset < Val;
    }
    bool operator()(const uint64_t Val, const BranchInfo &BI) const {
      return Val < BI.From.Offset;
    }
  };
  auto Range = std::equal_range(Data.begin(), Data.end(), From, Compare());
  for (const auto &RI : llvm::make_range(Range))
    if (RI.From.Name != RI.To.Name)
      return RI;

  return make_error_code(llvm::errc::invalid_argument);
}

void MemInfo::print(raw_ostream &OS) const {
  OS << (Offset.IsSymbol + 3) << " " << Offset.Name << " "
     << Twine::utohexstr(Offset.Offset) << " " << (Addr.IsSymbol + 3) << " "
     << Addr.Name << " " << Twine::utohexstr(Addr.Offset) << " " << Count
     << "\n";
}

void MemInfo::prettyPrint(raw_ostream &OS) const {
  OS << "(PC: " << Offset << ", M: " << Addr << ", C: " << Count << ")";
}

void FuncMemData::update(const Location &Offset, const Location &Addr) {
  auto Iter = EventIndex[Offset.Offset].find(Addr);
  if (Iter == EventIndex[Offset.Offset].end()) {
    Data.emplace_back(MemInfo(Offset, Addr, 1));
    EventIndex[Offset.Offset][Addr] = Data.size() - 1;
    return;
  }
  ++Data[Iter->second].Count;
}

Error DataReader::preprocessProfile(BinaryContext &BC) {
  if (std::error_code EC = parseInput())
    return errorCodeToError(EC);

  if (opts::DumpData)
    dump();

  if (collectedInBoltedBinary())
    outs() << "BOLT-INFO: profile collection done on a binary already "
              "processed by BOLT\n";

  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (FuncMemData *MemData = getMemDataForNames(Function.getNames())) {
      setMemData(Function, MemData);
      MemData->Used = true;
    }
    if (FuncBranchData *FuncData = getBranchDataForNames(Function.getNames())) {
      setBranchData(Function, FuncData);
      Function.ExecutionCount = FuncData->ExecutionCount;
      Function.ExternEntryCount = FuncData->ExternEntryCount;
      FuncData->Used = true;
    }
  }

  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    matchProfileMemData(Function);
  }

  return Error::success();
}

Error DataReader::readProfilePreCFG(BinaryContext &BC) {
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    FuncMemData *MemoryData = getMemData(Function);
    if (!MemoryData)
      continue;

    for (MemInfo &MI : MemoryData->Data) {
      const uint64_t Offset = MI.Offset.Offset;
      auto II = Function.Instructions.find(Offset);
      if (II == Function.Instructions.end()) {
        // Ignore bad instruction address.
        continue;
      }

      auto &MemAccessProfile =
          BC.MIB->getOrCreateAnnotationAs<MemoryAccessProfile>(
              II->second, "MemoryAccessProfile");
      BinaryData *BD = nullptr;
      if (MI.Addr.IsSymbol)
        BD = BC.getBinaryDataByName(MI.Addr.Name);
      MemAccessProfile.AddressAccessInfo.push_back(
          {BD, MI.Addr.Offset, MI.Count});
      auto NextII = std::next(II);
      if (NextII == Function.Instructions.end())
        MemAccessProfile.NextInstrOffset = Function.getSize();
      else
        MemAccessProfile.NextInstrOffset = II->first;
    }
    Function.HasMemoryProfile = true;
  }

  return Error::success();
}

Error DataReader::readProfile(BinaryContext &BC) {
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    readProfile(Function);
  }

  uint64_t NumUnused = 0;
  for (const auto &KV : NamesToBranches) {
    const FuncBranchData &FBD = KV.second;
    if (!FBD.Used)
      ++NumUnused;
  }
  BC.setNumUnusedProfiledObjects(NumUnused);

  return Error::success();
}

std::error_code DataReader::parseInput() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = MB.getError()) {
    Diag << "cannot open " << Filename << ": " << EC.message() << "\n";
    return EC;
  }
  FileBuf = std::move(MB.get());
  ParsingBuf = FileBuf->getBuffer();
  if (std::error_code EC = parse())
    return EC;
  if (!ParsingBuf.empty())
    Diag << "WARNING: invalid profile data detected at line " << Line
         << ". Possibly corrupted profile.\n";

  buildLTONameMaps();

  return std::error_code();
}

void DataReader::readProfile(BinaryFunction &BF) {
  if (BF.empty())
    return;

  if (!hasLBR()) {
    BF.ProfileFlags = BinaryFunction::PF_BASIC;
    readBasicSampleData(BF);
    return;
  }

  BF.ProfileFlags = BinaryFunction::PF_BRANCH;

  // Possibly assign/re-assign branch profile data.
  matchProfileData(BF);

  FuncBranchData *FBD = getBranchData(BF);
  if (!FBD)
    return;

  // Assign basic block counts to function entry points. These only include
  // counts for outside entries.
  //
  // There is a slight skew introduced here as branches originated from RETs
  // may be accounted for in the execution count of an entry block if the last
  // instruction in a predecessor fall-through block is a call. This situation
  // should rarely happen because there are few multiple-entry functions.
  for (const BranchInfo &BI : FBD->EntryData) {
    BinaryBasicBlock *BB = BF.getBasicBlockAtOffset(BI.To.Offset);
    if (BB && (BB->isEntryPoint() || BB->isLandingPad())) {
      uint64_t Count = BB->getExecutionCount();
      if (Count == BinaryBasicBlock::COUNT_NO_PROFILE)
        Count = 0;
      BB->setExecutionCount(Count + BI.Branches);
    }
  }

  for (const BranchInfo &BI : FBD->Data) {
    if (BI.From.Name != BI.To.Name)
      continue;

    if (!recordBranch(BF, BI.From.Offset, BI.To.Offset, BI.Branches,
                      BI.Mispreds)) {
      LLVM_DEBUG(dbgs() << "bad branch : " << BI.From.Offset << " -> "
                        << BI.To.Offset << '\n');
    }
  }

  // Convert branch data into annotations.
  convertBranchData(BF);
}

void DataReader::matchProfileData(BinaryFunction &BF) {
  // This functionality is available for LBR-mode only
  // TODO: Implement evaluateProfileData() for samples, checking whether
  // sample addresses match instruction addresses in the function
  if (!hasLBR())
    return;

  FuncBranchData *FBD = getBranchData(BF);
  if (FBD) {
    BF.ProfileMatchRatio = evaluateProfileData(BF, *FBD);
    BF.RawSampleCount = FBD->getNumExecutedBranches();
    if (BF.ProfileMatchRatio == 1.0f) {
      if (fetchProfileForOtherEntryPoints(BF)) {
        BF.ProfileMatchRatio = evaluateProfileData(BF, *FBD);
        BF.ExecutionCount = FBD->ExecutionCount;
        BF.ExternEntryCount = FBD->ExternEntryCount;
        BF.RawSampleCount = FBD->getNumExecutedBranches();
      }
      return;
    }
  }

  // Check if the function name can fluctuate between several compilations
  // possibly triggered by minor unrelated code changes in the source code
  // of the input binary.
  if (!hasVolatileName(BF))
    return;

  // Check for a profile that matches with 100% confidence.
  const std::vector<FuncBranchData *> AllBranchData =
      getBranchDataForNamesRegex(BF.getNames());
  for (FuncBranchData *NewBranchData : AllBranchData) {
    // Prevent functions from sharing the same profile.
    if (NewBranchData->Used)
      continue;

    if (evaluateProfileData(BF, *NewBranchData) != 1.0f)
      continue;

    if (FBD)
      FBD->Used = false;

    // Update function profile data with the new set.
    setBranchData(BF, NewBranchData);
    NewBranchData->Used = true;
    BF.ExecutionCount = NewBranchData->ExecutionCount;
    BF.ExternEntryCount = NewBranchData->ExternEntryCount;
    BF.ProfileMatchRatio = 1.0f;
    break;
  }
}

void DataReader::matchProfileMemData(BinaryFunction &BF) {
  const std::vector<FuncMemData *> AllMemData =
      getMemDataForNamesRegex(BF.getNames());
  for (FuncMemData *NewMemData : AllMemData) {
    // Prevent functions from sharing the same profile.
    if (NewMemData->Used)
      continue;

    if (FuncMemData *MD = getMemData(BF))
      MD->Used = false;

    // Update function profile data with the new set.
    setMemData(BF, NewMemData);
    NewMemData->Used = true;
    break;
  }
}

bool DataReader::fetchProfileForOtherEntryPoints(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();

  FuncBranchData *FBD = getBranchData(BF);
  if (!FBD)
    return false;

  // Check if we are missing profiling data for secondary entry points
  bool First = true;
  bool Updated = false;
  for (BinaryBasicBlock *BB : BF.BasicBlocks) {
    if (First) {
      First = false;
      continue;
    }
    if (BB->isEntryPoint()) {
      uint64_t EntryAddress = BB->getOffset() + BF.getAddress();
      // Look for branch data associated with this entry point
      if (BinaryData *BD = BC.getBinaryDataAtAddress(EntryAddress)) {
        if (FuncBranchData *Data = getBranchDataForSymbols(BD->getSymbols())) {
          FBD->appendFrom(*Data, BB->getOffset());
          Data->Used = true;
          Updated = true;
        }
      }
    }
  }

  return Updated;
}

float DataReader::evaluateProfileData(BinaryFunction &BF,
                                      const FuncBranchData &BranchData) const {
  BinaryContext &BC = BF.getBinaryContext();

  // Until we define a minimal profile, we consider an empty branch data to be
  // a valid profile. It could happen to a function without branches when we
  // still have an EntryData for the execution count.
  if (BranchData.Data.empty())
    return 1.0f;

  uint64_t NumMatchedBranches = 0;
  for (const BranchInfo &BI : BranchData.Data) {
    bool IsValid = false;
    if (BI.From.Name == BI.To.Name) {
      // Try to record information with 0 count.
      IsValid = recordBranch(BF, BI.From.Offset, BI.To.Offset, 0);
    } else if (collectedInBoltedBinary()) {
      // We can't check branch source for collections in bolted binaries because
      // the source of the branch may be mapped to the first instruction in a BB
      // instead of the original branch (which may not exist in the source bin).
      IsValid = true;
    } else {
      // The branch has to originate from this function.
      // Check for calls, tail calls, rets and indirect branches.
      // When matching profiling info, we did not reach the stage
      // when we identify tail calls, so they are still represented
      // by regular branch instructions and we need isBranch() here.
      MCInst *Instr = BF.getInstructionAtOffset(BI.From.Offset);
      // If it's a prefix - skip it.
      if (Instr && BC.MIB->isPrefix(*Instr))
        Instr = BF.getInstructionAtOffset(BI.From.Offset + 1);
      if (Instr && (BC.MIB->isCall(*Instr) || BC.MIB->isBranch(*Instr) ||
                    BC.MIB->isReturn(*Instr)))
        IsValid = true;
    }

    if (IsValid) {
      ++NumMatchedBranches;
      continue;
    }

    LLVM_DEBUG(dbgs() << "\tinvalid branch in " << BF << " : 0x"
                      << Twine::utohexstr(BI.From.Offset) << " -> ";
               if (BI.From.Name == BI.To.Name) dbgs()
               << "0x" << Twine::utohexstr(BI.To.Offset) << '\n';
               else dbgs() << "<outbounds>\n";);
  }

  const float MatchRatio = (float)NumMatchedBranches / BranchData.Data.size();
  if (opts::Verbosity >= 2 && NumMatchedBranches < BranchData.Data.size())
    errs() << "BOLT-WARNING: profile branches match only "
           << format("%.1f%%", MatchRatio * 100.0f) << " ("
           << NumMatchedBranches << '/' << BranchData.Data.size()
           << ") for function " << BF << '\n';

  return MatchRatio;
}

void DataReader::readBasicSampleData(BinaryFunction &BF) {
  FuncBasicSampleData *SampleDataOrErr = getFuncBasicSampleData(BF.getNames());
  if (!SampleDataOrErr)
    return;

  // Basic samples mode territory (without LBR info)
  // First step is to assign BB execution count based on samples from perf
  BF.ProfileMatchRatio = 1.0f;
  BF.removeTagsFromProfile();
  bool NormalizeByInsnCount = usesEvent("cycles") || usesEvent("instructions");
  bool NormalizeByCalls = usesEvent("branches");
  static bool NagUser = true;
  if (NagUser) {
    outs()
        << "BOLT-INFO: operating with basic samples profiling data (no LBR).\n";
    if (NormalizeByInsnCount)
      outs() << "BOLT-INFO: normalizing samples by instruction count.\n";
    else if (NormalizeByCalls)
      outs() << "BOLT-INFO: normalizing samples by branches.\n";

    NagUser = false;
  }
  uint64_t LastOffset = BF.getSize();
  uint64_t TotalEntryCount = 0;
  for (BinaryFunction::BasicBlockOffset &BBOffset :
       llvm::reverse(BF.BasicBlockOffsets)) {
    uint64_t CurOffset = BBOffset.first;
    // Always work with samples multiplied by 1000 to avoid losing them if we
    // later need to normalize numbers
    uint64_t NumSamples =
        SampleDataOrErr->getSamples(CurOffset, LastOffset) * 1000;
    if (NormalizeByInsnCount && BBOffset.second->getNumNonPseudos()) {
      NumSamples /= BBOffset.second->getNumNonPseudos();
    } else if (NormalizeByCalls) {
      uint32_t NumCalls = BBOffset.second->getNumCalls();
      NumSamples /= NumCalls + 1;
    }
    BBOffset.second->setExecutionCount(NumSamples);
    if (BBOffset.second->isEntryPoint())
      TotalEntryCount += NumSamples;
    LastOffset = CurOffset;
  }

  BF.ExecutionCount = TotalEntryCount;
}

void DataReader::convertBranchData(BinaryFunction &BF) const {
  BinaryContext &BC = BF.getBinaryContext();

  if (BF.empty())
    return;

  FuncBranchData *FBD = getBranchData(BF);
  if (!FBD)
    return;

  // Profile information for calls.
  //
  // There are 3 cases that we annotate differently:
  //   1) Conditional tail calls that could be mispredicted.
  //   2) Indirect calls to multiple destinations with mispredictions.
  //      Before we validate CFG we have to handle indirect branches here too.
  //   3) Regular direct calls. The count could be different from containing
  //      basic block count. Keep this data in case we find it useful.
  //
  for (BranchInfo &BI : FBD->Data) {
    // Ignore internal branches.
    if (BI.To.IsSymbol && BI.To.Name == BI.From.Name && BI.To.Offset != 0)
      continue;

    MCInst *Instr = BF.getInstructionAtOffset(BI.From.Offset);
    if (!Instr ||
        (!BC.MIB->isCall(*Instr) && !BC.MIB->isIndirectBranch(*Instr)))
      continue;

    auto setOrUpdateAnnotation = [&](StringRef Name, uint64_t Count) {
      if (opts::Verbosity >= 1 && BC.MIB->hasAnnotation(*Instr, Name))
        errs() << "BOLT-WARNING: duplicate " << Name << " info for offset 0x"
               << Twine::utohexstr(BI.From.Offset) << " in function " << BF
               << '\n';
      auto &Value = BC.MIB->getOrCreateAnnotationAs<uint64_t>(*Instr, Name);
      Value += Count;
    };

    if (BC.MIB->isIndirectCall(*Instr) || BC.MIB->isIndirectBranch(*Instr)) {
      IndirectCallSiteProfile &CSP =
          BC.MIB->getOrCreateAnnotationAs<IndirectCallSiteProfile>(
              *Instr, "CallProfile");
      MCSymbol *CalleeSymbol = nullptr;
      if (BI.To.IsSymbol) {
        if (BinaryData *BD = BC.getBinaryDataByName(BI.To.Name))
          CalleeSymbol = BD->getSymbol();
      }
      CSP.emplace_back(CalleeSymbol, BI.Branches, BI.Mispreds);
    } else if (BC.MIB->getConditionalTailCall(*Instr)) {
      setOrUpdateAnnotation("CTCTakenCount", BI.Branches);
      setOrUpdateAnnotation("CTCMispredCount", BI.Mispreds);
    } else {
      setOrUpdateAnnotation("Count", BI.Branches);
    }
  }
}

bool DataReader::recordBranch(BinaryFunction &BF, uint64_t From, uint64_t To,
                              uint64_t Count, uint64_t Mispreds) const {
  BinaryContext &BC = BF.getBinaryContext();

  BinaryBasicBlock *FromBB = BF.getBasicBlockContainingOffset(From);
  const BinaryBasicBlock *ToBB = BF.getBasicBlockContainingOffset(To);

  if (!FromBB || !ToBB) {
    LLVM_DEBUG(dbgs() << "failed to get block for recorded branch\n");
    return false;
  }

  // Could be bad LBR data; ignore the branch. In the case of data collected
  // in binaries optimized by BOLT, a source BB may be mapped to two output
  // BBs as a result of optimizations. In that case, a branch between these
  // two will be recorded as a branch from A going to A in the source address
  // space. Keep processing.
  if (From == To)
    return true;

  // Return from a tail call.
  if (FromBB->succ_size() == 0)
    return true;

  // Very rarely we will see ignored branches. Do a linear check.
  for (std::pair<uint32_t, uint32_t> &Branch : BF.IgnoredBranches)
    if (Branch ==
        std::make_pair(static_cast<uint32_t>(From), static_cast<uint32_t>(To)))
      return true;

  bool OffsetMatches = !!(To == ToBB->getOffset());
  if (!OffsetMatches) {
    // Skip the nops to support old .fdata
    uint64_t Offset = ToBB->getOffset();
    for (const MCInst &Instr : *ToBB) {
      if (!BC.MIB->isNoop(Instr))
        break;

      if (std::optional<uint32_t> Size = BC.MIB->getSize(Instr))
        Offset += *Size;
    }

    if (To == Offset)
      OffsetMatches = true;
  }

  if (!OffsetMatches) {
    // "To" could be referring to nop instructions in between 2 basic blocks.
    // While building the CFG we make sure these nops are attributed to the
    // previous basic block, thus we check if the destination belongs to the
    // gap past the last instruction.
    const MCInst *LastInstr = ToBB->getLastNonPseudoInstr();
    if (LastInstr) {
      const uint32_t LastInstrOffset =
          BC.MIB->getOffsetWithDefault(*LastInstr, 0);

      // With old .fdata we are getting FT branches for "jcc,jmp" sequences.
      if (To == LastInstrOffset && BC.MIB->isUnconditionalBranch(*LastInstr))
        return true;

      if (To <= LastInstrOffset) {
        LLVM_DEBUG(dbgs() << "branch recorded into the middle of the block"
                          << " in " << BF << " : " << From << " -> " << To
                          << '\n');
        return false;
      }
    }

    // The real destination is the layout successor of the detected ToBB.
    if (ToBB == BF.getLayout().block_back())
      return false;
    const BinaryBasicBlock *NextBB =
        BF.getLayout().getBlock(ToBB->getIndex() + 1);
    assert((NextBB && NextBB->getOffset() > ToBB->getOffset()) && "bad layout");
    ToBB = NextBB;
  }

  // If there's no corresponding instruction for 'From', we have probably
  // discarded it as a FT from __builtin_unreachable.
  MCInst *FromInstruction = BF.getInstructionAtOffset(From);
  if (!FromInstruction) {
    // If the data was collected in a bolted binary, the From addresses may be
    // translated to the first instruction of the source BB if BOLT inserted
    // a new branch that did not exist in the source (we can't map it to the
    // source instruction, so we map it to the first instr of source BB).
    // We do not keep offsets for random instructions. So the check above will
    // evaluate to true if the first instr is not a branch (call/jmp/ret/etc)
    if (collectedInBoltedBinary()) {
      if (FromBB->getInputOffset() != From) {
        LLVM_DEBUG(dbgs() << "offset " << From << " does not match a BB in "
                          << BF << '\n');
        return false;
      }
      FromInstruction = nullptr;
    } else {
      LLVM_DEBUG(dbgs() << "no instruction for offset " << From << " in " << BF
                        << '\n');
      return false;
    }
  }

  if (!FromBB->getSuccessor(ToBB->getLabel())) {
    // Check if this is a recursive call or a return from a recursive call.
    if (FromInstruction && ToBB->isEntryPoint() &&
        (BC.MIB->isCall(*FromInstruction) ||
         BC.MIB->isIndirectBranch(*FromInstruction))) {
      // Execution count is already accounted for.
      return true;
    }
    // For data collected in a bolted binary, we may have created two output BBs
    // that map to one original block. Branches between these two blocks will
    // appear here as one BB jumping to itself, even though it has no loop
    // edges. Ignore these.
    if (collectedInBoltedBinary() && FromBB == ToBB)
      return true;

    // Allow passthrough blocks.
    BinaryBasicBlock *FTSuccessor = FromBB->getConditionalSuccessor(false);
    if (FTSuccessor && FTSuccessor->succ_size() == 1 &&
        FTSuccessor->getSuccessor(ToBB->getLabel())) {
      BinaryBasicBlock::BinaryBranchInfo &FTBI =
          FTSuccessor->getBranchInfo(*ToBB);
      FTBI.Count += Count;
      if (Count)
        FTBI.MispredictedCount += Mispreds;
      ToBB = FTSuccessor;
    } else {
      LLVM_DEBUG(dbgs() << "invalid branch in " << BF
                        << formatv(": {0:x} -> {1:x}\n", From, To));
      return false;
    }
  }

  BinaryBasicBlock::BinaryBranchInfo &BI = FromBB->getBranchInfo(*ToBB);
  BI.Count += Count;
  // Only update mispredicted count if it the count was real.
  if (Count) {
    BI.MispredictedCount += Mispreds;
  }

  return true;
}

void DataReader::reportError(StringRef ErrorMsg) {
  Diag << "Error reading BOLT data input file: line " << Line << ", column "
       << Col << ": " << ErrorMsg << '\n';
}

bool DataReader::expectAndConsumeFS() {
  if (ParsingBuf[0] != FieldSeparator) {
    reportError("expected field separator");
    return false;
  }
  ParsingBuf = ParsingBuf.drop_front(1);
  Col += 1;
  return true;
}

void DataReader::consumeAllRemainingFS() {
  while (ParsingBuf[0] == FieldSeparator) {
    ParsingBuf = ParsingBuf.drop_front(1);
    Col += 1;
  }
}

bool DataReader::checkAndConsumeNewLine() {
  if (ParsingBuf[0] != '\n')
    return false;

  ParsingBuf = ParsingBuf.drop_front(1);
  Col = 0;
  Line += 1;
  return true;
}

ErrorOr<StringRef> DataReader::parseString(char EndChar, bool EndNl) {
  if (EndChar == '\\') {
    reportError("EndChar could not be backslash");
    return make_error_code(llvm::errc::io_error);
  }

  std::string EndChars(1, EndChar);
  EndChars.push_back('\\');
  if (EndNl)
    EndChars.push_back('\n');

  size_t StringEnd = 0;
  do {
    StringEnd = ParsingBuf.find_first_of(EndChars, StringEnd);
    if (StringEnd == StringRef::npos ||
        (StringEnd == 0 && ParsingBuf[StringEnd] != '\\')) {
      reportError("malformed field");
      return make_error_code(llvm::errc::io_error);
    }

    if (ParsingBuf[StringEnd] != '\\')
      break;

    StringEnd += 2;
  } while (true);

  StringRef Str = ParsingBuf.substr(0, StringEnd);

  // If EndNl was set and nl was found instead of EndChar, do not consume the
  // new line.
  bool EndNlInsteadOfEndChar = ParsingBuf[StringEnd] == '\n' && EndChar != '\n';
  unsigned End = EndNlInsteadOfEndChar ? StringEnd : StringEnd + 1;

  ParsingBuf = ParsingBuf.drop_front(End);
  if (EndChar == '\n') {
    Col = 0;
    Line += 1;
  } else {
    Col += End;
  }
  return Str;
}

ErrorOr<int64_t> DataReader::parseNumberField(char EndChar, bool EndNl) {
  ErrorOr<StringRef> NumStrRes = parseString(EndChar, EndNl);
  if (std::error_code EC = NumStrRes.getError())
    return EC;
  StringRef NumStr = NumStrRes.get();
  int64_t Num;
  if (NumStr.getAsInteger(10, Num)) {
    reportError("expected decimal number");
    Diag << "Found: " << NumStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  return Num;
}

ErrorOr<uint64_t> DataReader::parseHexField(char EndChar, bool EndNl) {
  ErrorOr<StringRef> NumStrRes = parseString(EndChar, EndNl);
  if (std::error_code EC = NumStrRes.getError())
    return EC;
  StringRef NumStr = NumStrRes.get();
  uint64_t Num;
  if (NumStr.getAsInteger(16, Num)) {
    reportError("expected hexidecimal number");
    Diag << "Found: " << NumStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  return Num;
}

ErrorOr<Location> DataReader::parseLocation(char EndChar, bool EndNl,
                                            bool ExpectMemLoc) {
  // Read whether the location of the branch should be DSO or a symbol
  // 0 means it is a DSO. 1 means it is a global symbol. 2 means it is a local
  // symbol.
  // The symbol flag is also used to tag memory load events by adding 3 to the
  // base values, i.e. 3 not a symbol, 4 global symbol and 5 local symbol.
  if (!ExpectMemLoc && ParsingBuf[0] != '0' && ParsingBuf[0] != '1' &&
      ParsingBuf[0] != '2') {
    reportError("expected 0, 1 or 2");
    return make_error_code(llvm::errc::io_error);
  }

  if (ExpectMemLoc && ParsingBuf[0] != '3' && ParsingBuf[0] != '4' &&
      ParsingBuf[0] != '5') {
    reportError("expected 3, 4 or 5");
    return make_error_code(llvm::errc::io_error);
  }

  bool IsSymbol =
      (!ExpectMemLoc && (ParsingBuf[0] == '1' || ParsingBuf[0] == '2')) ||
      (ExpectMemLoc && (ParsingBuf[0] == '4' || ParsingBuf[0] == '5'));
  ParsingBuf = ParsingBuf.drop_front(1);
  Col += 1;

  if (!expectAndConsumeFS())
    return make_error_code(llvm::errc::io_error);
  consumeAllRemainingFS();

  // Read the string containing the symbol or the DSO name
  ErrorOr<StringRef> NameRes = parseString(FieldSeparator);
  if (std::error_code EC = NameRes.getError())
    return EC;
  StringRef Name = NameRes.get();
  consumeAllRemainingFS();

  // Read the offset
  ErrorOr<uint64_t> Offset = parseHexField(EndChar, EndNl);
  if (std::error_code EC = Offset.getError())
    return EC;

  return Location(IsSymbol, Name, Offset.get());
}

ErrorOr<BranchInfo> DataReader::parseBranchInfo() {
  ErrorOr<Location> Res = parseLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location From = Res.get();

  consumeAllRemainingFS();
  Res = parseLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location To = Res.get();

  consumeAllRemainingFS();
  ErrorOr<int64_t> MRes = parseNumberField(FieldSeparator);
  if (std::error_code EC = MRes.getError())
    return EC;
  int64_t NumMispreds = MRes.get();

  consumeAllRemainingFS();
  ErrorOr<int64_t> BRes = parseNumberField(FieldSeparator, /* EndNl = */ true);
  if (std::error_code EC = BRes.getError())
    return EC;
  int64_t NumBranches = BRes.get();

  consumeAllRemainingFS();
  if (!checkAndConsumeNewLine()) {
    reportError("expected end of line");
    return make_error_code(llvm::errc::io_error);
  }

  return BranchInfo(std::move(From), std::move(To), NumMispreds, NumBranches);
}

ErrorOr<MemInfo> DataReader::parseMemInfo() {
  ErrorOr<Location> Res = parseMemLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location Offset = Res.get();

  consumeAllRemainingFS();
  Res = parseMemLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location Addr = Res.get();

  consumeAllRemainingFS();
  ErrorOr<int64_t> CountRes = parseNumberField(FieldSeparator, true);
  if (std::error_code EC = CountRes.getError())
    return EC;

  consumeAllRemainingFS();
  if (!checkAndConsumeNewLine()) {
    reportError("expected end of line");
    return make_error_code(llvm::errc::io_error);
  }

  return MemInfo(Offset, Addr, CountRes.get());
}

ErrorOr<BasicSampleInfo> DataReader::parseSampleInfo() {
  ErrorOr<Location> Res = parseLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location Address = Res.get();

  consumeAllRemainingFS();
  ErrorOr<int64_t> BRes = parseNumberField(FieldSeparator, /* EndNl = */ true);
  if (std::error_code EC = BRes.getError())
    return EC;
  int64_t Occurrences = BRes.get();

  consumeAllRemainingFS();
  if (!checkAndConsumeNewLine()) {
    reportError("expected end of line");
    return make_error_code(llvm::errc::io_error);
  }

  return BasicSampleInfo(std::move(Address), Occurrences);
}

ErrorOr<bool> DataReader::maybeParseNoLBRFlag() {
  if (!ParsingBuf.consume_front("no_lbr"))
    return false;
  Col += 6;

  if (ParsingBuf.size() > 0 && ParsingBuf[0] == ' ')
    ParsingBuf = ParsingBuf.drop_front(1);

  while (ParsingBuf.size() > 0 && ParsingBuf[0] != '\n') {
    ErrorOr<StringRef> EventName = parseString(' ', true);
    if (!EventName)
      return make_error_code(llvm::errc::io_error);
    EventNames.insert(EventName.get());
  }

  if (!checkAndConsumeNewLine()) {
    reportError("malformed no_lbr line");
    return make_error_code(llvm::errc::io_error);
  }
  return true;
}

ErrorOr<bool> DataReader::maybeParseBATFlag() {
  if (!ParsingBuf.consume_front("boltedcollection"))
    return false;
  Col += 16;

  if (!checkAndConsumeNewLine()) {
    reportError("malformed boltedcollection line");
    return make_error_code(llvm::errc::io_error);
  }
  return true;
}

bool DataReader::hasBranchData() {
  if (ParsingBuf.size() == 0)
    return false;

  if (ParsingBuf[0] == '0' || ParsingBuf[0] == '1' || ParsingBuf[0] == '2')
    return true;
  return false;
}

bool DataReader::hasMemData() {
  if (ParsingBuf.size() == 0)
    return false;

  if (ParsingBuf[0] == '3' || ParsingBuf[0] == '4' || ParsingBuf[0] == '5')
    return true;
  return false;
}

std::error_code DataReader::parseInNoLBRMode() {
  auto GetOrCreateFuncEntry = [&](StringRef Name) {
    return NamesToBasicSamples.try_emplace(Name, Name).first;
  };

  auto GetOrCreateFuncMemEntry = [&](StringRef Name) {
    return NamesToMemEvents.try_emplace(Name, Name).first;
  };

  while (hasBranchData()) {
    ErrorOr<BasicSampleInfo> Res = parseSampleInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    BasicSampleInfo SI = Res.get();

    // Ignore samples not involving known locations
    if (!SI.Loc.IsSymbol)
      continue;

    auto I = GetOrCreateFuncEntry(SI.Loc.Name);
    I->second.Data.emplace_back(std::move(SI));
  }

  while (hasMemData()) {
    ErrorOr<MemInfo> Res = parseMemInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    MemInfo MI = Res.get();

    // Ignore memory events not involving known pc.
    if (!MI.Offset.IsSymbol)
      continue;

    auto I = GetOrCreateFuncMemEntry(MI.Offset.Name);
    I->second.Data.emplace_back(std::move(MI));
  }

  for (auto &FuncBasicSamples : NamesToBasicSamples)
    llvm::stable_sort(FuncBasicSamples.second.Data);

  for (auto &MemEvents : NamesToMemEvents)
    llvm::stable_sort(MemEvents.second.Data);

  return std::error_code();
}

std::error_code DataReader::parse() {
  auto GetOrCreateFuncEntry = [&](StringRef Name) {
    return NamesToBranches.try_emplace(Name, Name).first;
  };

  auto GetOrCreateFuncMemEntry = [&](StringRef Name) {
    return NamesToMemEvents.try_emplace(Name, Name).first;
  };

  Col = 0;
  Line = 1;
  ErrorOr<bool> FlagOrErr = maybeParseNoLBRFlag();
  if (!FlagOrErr)
    return FlagOrErr.getError();
  NoLBRMode = *FlagOrErr;

  ErrorOr<bool> BATFlagOrErr = maybeParseBATFlag();
  if (!BATFlagOrErr)
    return BATFlagOrErr.getError();
  BATMode = *BATFlagOrErr;

  if (!hasBranchData() && !hasMemData()) {
    Diag << "ERROR: no valid profile data found\n";
    return make_error_code(llvm::errc::io_error);
  }

  if (NoLBRMode)
    return parseInNoLBRMode();

  while (hasBranchData()) {
    ErrorOr<BranchInfo> Res = parseBranchInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    BranchInfo BI = Res.get();

    // Ignore branches not involving known location.
    if (!BI.From.IsSymbol && !BI.To.IsSymbol)
      continue;

    auto I = GetOrCreateFuncEntry(BI.From.Name);
    I->second.Data.emplace_back(std::move(BI));

    // Add entry data for branches to another function or branches
    // to entry points (including recursive calls)
    if (BI.To.IsSymbol && (BI.From.Name != BI.To.Name || BI.To.Offset == 0)) {
      I = GetOrCreateFuncEntry(BI.To.Name);
      I->second.EntryData.emplace_back(std::move(BI));
    }

    // If destination is the function start - update execution count.
    // NB: the data is skewed since we cannot tell tail recursion from
    //     branches to the function start.
    if (BI.To.IsSymbol && BI.To.Offset == 0) {
      I = GetOrCreateFuncEntry(BI.To.Name);
      I->second.ExecutionCount += BI.Branches;
      if (!BI.From.IsSymbol)
        I->second.ExternEntryCount += BI.Branches;
    }
  }

  while (hasMemData()) {
    ErrorOr<MemInfo> Res = parseMemInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    MemInfo MI = Res.get();

    // Ignore memory events not involving known pc.
    if (!MI.Offset.IsSymbol)
      continue;

    auto I = GetOrCreateFuncMemEntry(MI.Offset.Name);
    I->second.Data.emplace_back(std::move(MI));
  }

  for (auto &FuncBranches : NamesToBranches)
    llvm::stable_sort(FuncBranches.second.Data);

  for (auto &MemEvents : NamesToMemEvents)
    llvm::stable_sort(MemEvents.second.Data);

  return std::error_code();
}

void DataReader::buildLTONameMaps() {
  for (auto &FuncData : NamesToBranches) {
    const StringRef FuncName = FuncData.first;
    const std::optional<StringRef> CommonName = getLTOCommonName(FuncName);
    if (CommonName)
      LTOCommonNameMap[*CommonName].push_back(&FuncData.second);
  }

  for (auto &FuncData : NamesToMemEvents) {
    const StringRef FuncName = FuncData.first;
    const std::optional<StringRef> CommonName = getLTOCommonName(FuncName);
    if (CommonName)
      LTOCommonNameMemMap[*CommonName].push_back(&FuncData.second);
  }
}

template <typename MapTy>
static typename MapTy::mapped_type *
fetchMapEntry(MapTy &Map, const std::vector<MCSymbol *> &Symbols) {
  // Do a reverse order iteration since the name in profile has a higher chance
  // of matching a name at the end of the list.
  for (const MCSymbol *Symbol : llvm::reverse(Symbols)) {
    auto I = Map.find(normalizeName(Symbol->getName()));
    if (I != Map.end())
      return &I->second;
  }
  return nullptr;
}

template <typename MapTy>
static typename MapTy::mapped_type *
fetchMapEntry(MapTy &Map, const std::vector<StringRef> &FuncNames) {
  // Do a reverse order iteration since the name in profile has a higher chance
  // of matching a name at the end of the list.
  for (StringRef Name : llvm::reverse(FuncNames)) {
    auto I = Map.find(normalizeName(Name));
    if (I != Map.end())
      return &I->second;
  }
  return nullptr;
}

template <typename MapTy>
static std::vector<typename MapTy::mapped_type *>
fetchMapEntriesRegex(MapTy &Map,
                     const StringMap<std::vector<typename MapTy::mapped_type *>>
                         &LTOCommonNameMap,
                     const std::vector<StringRef> &FuncNames) {
  std::vector<typename MapTy::mapped_type *> AllData;
  // Do a reverse order iteration since the name in profile has a higher chance
  // of matching a name at the end of the list.
  for (StringRef FuncName : llvm::reverse(FuncNames)) {
    std::string Name = normalizeName(FuncName);
    const std::optional<StringRef> LTOCommonName = getLTOCommonName(Name);
    if (LTOCommonName) {
      auto I = LTOCommonNameMap.find(*LTOCommonName);
      if (I != LTOCommonNameMap.end()) {
        const std::vector<typename MapTy::mapped_type *> &CommonData =
            I->second;
        AllData.insert(AllData.end(), CommonData.begin(), CommonData.end());
      }
    } else {
      auto I = Map.find(Name);
      if (I != Map.end())
        return {&I->second};
    }
  }
  return AllData;
}

bool DataReader::mayHaveProfileData(const BinaryFunction &Function) {
  if (getBranchData(Function) || getMemData(Function))
    return true;

  if (getFuncBasicSampleData(Function.getNames()) ||
      getBranchDataForNames(Function.getNames()) ||
      getMemDataForNames(Function.getNames()))
    return true;

  if (!hasVolatileName(Function))
    return false;

  const std::vector<FuncBranchData *> AllBranchData =
      getBranchDataForNamesRegex(Function.getNames());
  if (!AllBranchData.empty())
    return true;

  const std::vector<FuncMemData *> AllMemData =
      getMemDataForNamesRegex(Function.getNames());
  if (!AllMemData.empty())
    return true;

  return false;
}

FuncBranchData *
DataReader::getBranchDataForNames(const std::vector<StringRef> &FuncNames) {
  return fetchMapEntry<NamesToBranchesMapTy>(NamesToBranches, FuncNames);
}

FuncBranchData *
DataReader::getBranchDataForSymbols(const std::vector<MCSymbol *> &Symbols) {
  return fetchMapEntry<NamesToBranchesMapTy>(NamesToBranches, Symbols);
}

FuncMemData *
DataReader::getMemDataForNames(const std::vector<StringRef> &FuncNames) {
  return fetchMapEntry<NamesToMemEventsMapTy>(NamesToMemEvents, FuncNames);
}

FuncBasicSampleData *
DataReader::getFuncBasicSampleData(const std::vector<StringRef> &FuncNames) {
  return fetchMapEntry<NamesToBasicSamplesMapTy>(NamesToBasicSamples,
                                                 FuncNames);
}

std::vector<FuncBranchData *> DataReader::getBranchDataForNamesRegex(
    const std::vector<StringRef> &FuncNames) {
  return fetchMapEntriesRegex(NamesToBranches, LTOCommonNameMap, FuncNames);
}

std::vector<FuncMemData *>
DataReader::getMemDataForNamesRegex(const std::vector<StringRef> &FuncNames) {
  return fetchMapEntriesRegex(NamesToMemEvents, LTOCommonNameMemMap, FuncNames);
}

bool DataReader::hasLocalsWithFileName() const {
  for (const auto &Func : NamesToBranches) {
    const StringRef &FuncName = Func.first;
    if (FuncName.count('/') == 2 && FuncName[0] != '/')
      return true;
  }
  return false;
}

void DataReader::dump() const {
  for (const auto &KV : NamesToBranches) {
    const StringRef Name = KV.first;
    const FuncBranchData &FBD = KV.second;
    Diag << Name << " branches:\n";
    for (const BranchInfo &BI : FBD.Data)
      Diag << BI.From.Name << " " << BI.From.Offset << " " << BI.To.Name << " "
           << BI.To.Offset << " " << BI.Mispreds << " " << BI.Branches << "\n";
    Diag << Name << " entry points:\n";
    for (const BranchInfo &BI : FBD.EntryData)
      Diag << BI.From.Name << " " << BI.From.Offset << " " << BI.To.Name << " "
           << BI.To.Offset << " " << BI.Mispreds << " " << BI.Branches << "\n";
  }

  for (auto I = EventNames.begin(), E = EventNames.end(); I != E; ++I) {
    StringRef Event = I->getKey();
    Diag << "Data was collected with event: " << Event << "\n";
  }
  for (const auto &KV : NamesToBasicSamples) {
    const StringRef Name = KV.first;
    const FuncBasicSampleData &FSD = KV.second;
    Diag << Name << " samples:\n";
    for (const BasicSampleInfo &SI : FSD.Data)
      Diag << SI.Loc.Name << " " << SI.Loc.Offset << " " << SI.Hits << "\n";
  }

  for (const auto &KV : NamesToMemEvents) {
    const StringRef Name = KV.first;
    const FuncMemData &FMD = KV.second;
    Diag << "Memory events for " << Name;
    Location LastOffset(0);
    for (const MemInfo &MI : FMD.Data) {
      if (MI.Offset == LastOffset)
        Diag << ", " << MI.Addr << "/" << MI.Count;
      else
        Diag << "\n" << MI.Offset << ": " << MI.Addr << "/" << MI.Count;
      LastOffset = MI.Offset;
    }
    Diag << "\n";
  }
}

} // namespace bolt
} // namespace llvm
