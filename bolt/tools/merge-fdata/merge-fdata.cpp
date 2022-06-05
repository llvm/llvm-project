//===- bolt/tools/merge-fdata/merge-fdata.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tool for merging profile in fdata format:
//
//   $ merge-fdata 1.fdata 2.fdata 3.fdata > merged.fdata
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/ProfileYAMLMapping.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include <unordered_map>

using namespace llvm;
using namespace llvm::yaml::bolt;

namespace opts {

cl::OptionCategory MergeFdataCategory("merge-fdata options");

enum SortType : char {
  ST_NONE,
  ST_EXEC_COUNT,      /// Sort based on function execution count.
  ST_TOTAL_BRANCHES,  /// Sort based on all branches in the function.
};

static cl::list<std::string>
InputDataFilenames(
  cl::Positional,
  cl::CommaSeparated,
  cl::desc("<fdata1> [<fdata2>]..."),
  cl::OneOrMore,
  cl::cat(MergeFdataCategory));

static cl::opt<SortType>
PrintFunctionList("print",
  cl::desc("print the list of objects with count to stderr"),
  cl::init(ST_NONE),
  cl::values(clEnumValN(ST_NONE,
      "none",
      "do not print objects/functions"),
    clEnumValN(ST_EXEC_COUNT,
      "exec",
      "print functions sorted by execution count"),
    clEnumValN(ST_TOTAL_BRANCHES,
      "branches",
      "print functions sorted by total branch count")),
  cl::cat(MergeFdataCategory));

static cl::opt<bool>
SuppressMergedDataOutput("q",
  cl::desc("do not print merged data to stdout"),
  cl::init(false),
  cl::Optional,
  cl::cat(MergeFdataCategory));

} // namespace opts

namespace {

static StringRef ToolName;

static void report_error(StringRef Message, std::error_code EC) {
  assert(EC);
  errs() << ToolName << ": '" << Message << "': " << EC.message() << ".\n";
  exit(1);
}

static void report_error(Twine Message, StringRef CustomError) {
  errs() << ToolName << ": '" << Message << "': " << CustomError << ".\n";
  exit(1);
}

void mergeProfileHeaders(BinaryProfileHeader &MergedHeader,
                         const BinaryProfileHeader &Header) {
  if (MergedHeader.FileName.empty())
    MergedHeader.FileName = Header.FileName;

  if (!MergedHeader.FileName.empty() &&
      MergedHeader.FileName != Header.FileName)
    errs() << "WARNING: merging profile from a binary for " << Header.FileName
           << " into a profile for binary " << MergedHeader.FileName << '\n';

  if (MergedHeader.Id.empty())
    MergedHeader.Id = Header.Id;

  if (!MergedHeader.Id.empty() && (MergedHeader.Id != Header.Id))
    errs() << "WARNING: build-ids in merged profiles do not match\n";

  // Cannot merge samples profile with LBR profile.
  if (!MergedHeader.Flags)
    MergedHeader.Flags = Header.Flags;

  constexpr auto Mask = llvm::bolt::BinaryFunction::PF_LBR |
                        llvm::bolt::BinaryFunction::PF_SAMPLE;
  if ((MergedHeader.Flags & Mask) != (Header.Flags & Mask)) {
    errs() << "ERROR: cannot merge LBR profile with non-LBR profile\n";
    exit(1);
  }
  MergedHeader.Flags = MergedHeader.Flags | Header.Flags;

  if (!Header.Origin.empty()) {
    if (MergedHeader.Origin.empty())
      MergedHeader.Origin = Header.Origin;
    else if (MergedHeader.Origin != Header.Origin)
      MergedHeader.Origin += "; " + Header.Origin;
  }

  if (MergedHeader.EventNames.empty())
    MergedHeader.EventNames = Header.EventNames;

  if (MergedHeader.EventNames != Header.EventNames) {
    errs() << "WARNING: merging profiles with different sampling events\n";
    MergedHeader.EventNames += "," + Header.EventNames;
  }
}

void mergeBasicBlockProfile(BinaryBasicBlockProfile &MergedBB,
                            BinaryBasicBlockProfile &&BB,
                            const BinaryFunctionProfile &BF) {
  // Verify that the blocks match.
  if (BB.NumInstructions != MergedBB.NumInstructions)
    report_error(BF.Name + " : BB #" + Twine(BB.Index),
                 "number of instructions in block mismatch");
  if (BB.Hash != MergedBB.Hash)
    report_error(BF.Name + " : BB #" + Twine(BB.Index),
                 "basic block hash mismatch");

  // Update the execution count.
  MergedBB.ExecCount += BB.ExecCount;

  // Update the event count.
  MergedBB.EventCount += BB.EventCount;

  // Merge calls sites.
  std::unordered_map<uint32_t, CallSiteInfo *> CSByOffset;
  for (CallSiteInfo &CS : BB.CallSites)
    CSByOffset.emplace(std::make_pair(CS.Offset, &CS));

  for (CallSiteInfo &MergedCS : MergedBB.CallSites) {
    auto CSI = CSByOffset.find(MergedCS.Offset);
    if (CSI == CSByOffset.end())
      continue;
    yaml::bolt::CallSiteInfo &CS = *CSI->second;
    if (CS != MergedCS)
      continue;

    MergedCS.Count += CS.Count;
    MergedCS.Mispreds += CS.Mispreds;

    CSByOffset.erase(CSI);
  }

  // Append the rest of call sites.
  for (std::pair<const uint32_t, CallSiteInfo *> CSI : CSByOffset)
    MergedBB.CallSites.emplace_back(std::move(*CSI.second));

  // Merge successor info.
  std::vector<SuccessorInfo *> SIByIndex(BF.NumBasicBlocks);
  for (SuccessorInfo &SI : BB.Successors) {
    if (SI.Index >= BF.NumBasicBlocks)
      report_error(BF.Name, "bad successor index");
    SIByIndex[SI.Index] = &SI;
  }
  for (SuccessorInfo &MergedSI : MergedBB.Successors) {
    if (!SIByIndex[MergedSI.Index])
      continue;
    SuccessorInfo &SI = *SIByIndex[MergedSI.Index];

    MergedSI.Count += SI.Count;
    MergedSI.Mispreds += SI.Mispreds;

    SIByIndex[MergedSI.Index] = nullptr;
  }
  for (SuccessorInfo *SI : SIByIndex)
    if (SI)
      MergedBB.Successors.emplace_back(std::move(*SI));
}

void mergeFunctionProfile(BinaryFunctionProfile &MergedBF,
                          BinaryFunctionProfile &&BF) {
  // Validate that we are merging the correct function.
  if (BF.NumBasicBlocks != MergedBF.NumBasicBlocks)
    report_error(BF.Name, "number of basic blocks mismatch");
  if (BF.Id != MergedBF.Id)
    report_error(BF.Name, "ID mismatch");
  if (BF.Hash != MergedBF.Hash)
    report_error(BF.Name, "hash mismatch");

  // Update the execution count.
  MergedBF.ExecCount += BF.ExecCount;

  // Merge basic blocks profile.
  std::vector<BinaryBasicBlockProfile *> BlockByIndex(BF.NumBasicBlocks);
  for (BinaryBasicBlockProfile &BB : BF.Blocks) {
    if (BB.Index >= BF.NumBasicBlocks)
      report_error(BF.Name + " : BB #" + Twine(BB.Index),
                   "bad basic block index");
    BlockByIndex[BB.Index] = &BB;
  }
  for (BinaryBasicBlockProfile &MergedBB : MergedBF.Blocks) {
    if (!BlockByIndex[MergedBB.Index])
      continue;
    BinaryBasicBlockProfile &BB = *BlockByIndex[MergedBB.Index];

    mergeBasicBlockProfile(MergedBB, std::move(BB), MergedBF);

    // Ignore this block in the future.
    BlockByIndex[MergedBB.Index] = nullptr;
  }

  // Append blocks unique to BF (i.e. those that are not in MergedBF).
  for (BinaryBasicBlockProfile *BB : BlockByIndex)
    if (BB)
      MergedBF.Blocks.emplace_back(std::move(*BB));
}

bool isYAML(const StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = MB.getError())
    report_error(Filename, EC);
  StringRef Buffer = MB.get()->getBuffer();
  if (Buffer.startswith("---\n"))
    return true;
  return false;
}

void mergeLegacyProfiles(const cl::list<std::string> &Filenames) {
  errs() << "Using legacy profile format.\n";
  bool BoltedCollection = false;
  bool First = true;
  StringMap<uint64_t> Entries;
  for (const std::string &Filename : Filenames) {
    if (isYAML(Filename))
      report_error(Filename, "cannot mix YAML and legacy formats");
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
        MemoryBuffer::getFileOrSTDIN(Filename);
    if (std::error_code EC = MB.getError())
      report_error(Filename, EC);
    errs() << "Merging data from " << Filename << "...\n";

    StringRef Buf = MB.get()->getBuffer();
    // Check if the string "boltedcollection" is in the first line
    if (Buf.startswith("boltedcollection\n")) {
      if (!First && !BoltedCollection)
        report_error(
            Filename,
            "cannot mix profile collected in BOLT and non-BOLT deployments");
      BoltedCollection = true;
      Buf = Buf.drop_front(17);
    } else {
      if (BoltedCollection)
        report_error(
            Filename,
            "cannot mix profile collected in BOLT and non-BOLT deployments");
    }

    SmallVector<StringRef> Lines;
    SplitString(Buf, Lines, "\n");
    for (StringRef Line : Lines) {
      size_t Pos = Line.rfind(" ");
      if (Pos == StringRef::npos)
        report_error(Filename, "Malformed / corrupted profile");
      StringRef Signature = Line.substr(0, Pos);
      uint64_t Count;
      if (Line.substr(Pos + 1, Line.size() - Pos).getAsInteger(10, Count))
        report_error(Filename, "Malformed / corrupted profile counter");
      Count += Entries.lookup(Signature);
      Entries.insert_or_assign(Signature, Count);
    }
    First = false;
  }

  if (BoltedCollection)
    outs() << "boltedcollection\n";
  for (const auto &Entry : Entries)
    outs() << Entry.getKey() << " " << Entry.getValue() << "\n";

  errs() << "Profile from " << Filenames.size() << " files merged.\n";
}

} // anonymous namespace

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::HideUnrelatedOptions(opts::MergeFdataCategory);

  cl::ParseCommandLineOptions(argc, argv,
                              "merge multiple fdata into a single file");

  ToolName = argv[0];

  if (!isYAML(opts::InputDataFilenames.front())) {
    mergeLegacyProfiles(opts::InputDataFilenames);
    return 0;
  }

  // Merged header.
  BinaryProfileHeader MergedHeader;
  MergedHeader.Version = 1;

  // Merged information for all functions.
  StringMap<BinaryFunctionProfile> MergedBFs;

  for (std::string &InputDataFilename : opts::InputDataFilenames) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
        MemoryBuffer::getFileOrSTDIN(InputDataFilename);
    if (std::error_code EC = MB.getError())
      report_error(InputDataFilename, EC);
    yaml::Input YamlInput(MB.get()->getBuffer());

    errs() << "Merging data from " << InputDataFilename << "...\n";

    BinaryProfile BP;
    YamlInput >> BP;
    if (YamlInput.error())
      report_error(InputDataFilename, YamlInput.error());

    // Sanity check.
    if (BP.Header.Version != 1) {
      errs() << "Unable to merge data from profile using version "
             << BP.Header.Version << '\n';
      exit(1);
    }

    // Merge the header.
    mergeProfileHeaders(MergedHeader, BP.Header);

    // Do the function merge.
    for (BinaryFunctionProfile &BF : BP.Functions) {
      if (!MergedBFs.count(BF.Name)) {
        MergedBFs.insert(std::make_pair(BF.Name, BF));
        continue;
      }

      BinaryFunctionProfile &MergedBF = MergedBFs.find(BF.Name)->second;
      mergeFunctionProfile(MergedBF, std::move(BF));
    }
  }

  if (!opts::SuppressMergedDataOutput) {
    yaml::Output YamlOut(outs());

    BinaryProfile MergedProfile;
    MergedProfile.Header = MergedHeader;
    MergedProfile.Functions.resize(MergedBFs.size());
    std::transform(
        MergedBFs.begin(), MergedBFs.end(), MergedProfile.Functions.begin(),
        [](StringMapEntry<BinaryFunctionProfile> &V) { return V.second; });

    // For consistency, sort functions by their IDs.
    std::sort(MergedProfile.Functions.begin(), MergedProfile.Functions.end(),
              [](const BinaryFunctionProfile &A,
                 const BinaryFunctionProfile &B) { return A.Id < B.Id; });

    YamlOut << MergedProfile;
  }

  errs() << "Data for " << MergedBFs.size()
         << " unique objects successfully merged.\n";

  if (opts::PrintFunctionList != opts::ST_NONE) {
    // List of function names with execution count.
    std::vector<std::pair<uint64_t, StringRef>> FunctionList(MergedBFs.size());
    using CountFuncType = std::function<std::pair<uint64_t, StringRef>(
        const StringMapEntry<BinaryFunctionProfile> &)>;
    CountFuncType ExecCountFunc =
        [](const StringMapEntry<BinaryFunctionProfile> &V) {
          return std::make_pair(V.second.ExecCount, StringRef(V.second.Name));
        };
    CountFuncType BranchCountFunc =
        [](const StringMapEntry<BinaryFunctionProfile> &V) {
          // Return total branch count.
          uint64_t BranchCount = 0;
          for (const BinaryBasicBlockProfile &BI : V.second.Blocks)
            for (const SuccessorInfo &SI : BI.Successors)
              BranchCount += SI.Count;
          return std::make_pair(BranchCount, StringRef(V.second.Name));
        };

    CountFuncType CountFunc = (opts::PrintFunctionList == opts::ST_EXEC_COUNT)
                                  ? ExecCountFunc
                                  : BranchCountFunc;
    std::transform(MergedBFs.begin(), MergedBFs.end(), FunctionList.begin(),
                   CountFunc);
    std::stable_sort(FunctionList.rbegin(), FunctionList.rend());
    errs() << "Functions sorted by "
           << (opts::PrintFunctionList == opts::ST_EXEC_COUNT ? "execution"
                                                              : "total branch")
           << " count:\n";
    for (std::pair<uint64_t, StringRef> &FI : FunctionList)
      errs() << FI.second << " : " << FI.first << '\n';
  }

  return 0;
}
