//===- bolt/Profile/DataAggregator.h - Perf data aggregator -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions reads profile data written by perf record,
// aggregates it and then writes it back to an output file.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_DATA_AGGREGATOR_H
#define BOLT_PROFILE_DATA_AGGREGATOR_H

#include "bolt/Profile/DataReader.h"
#include "bolt/Profile/YAMLProfileWriter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"
#include <unordered_map>

namespace llvm {
namespace bolt {

class BinaryFunction;
class BinaryContext;
class BoltAddressTranslation;

/// DataAggregator inherits all parsing logic from DataReader as well as
/// its data structures used to represent aggregated profile data in memory.
///
/// The aggregator works by dispatching two separate perf-script jobs that
/// read perf samples and perf task annotations. Later, we read the output
/// files to extract information about which PID was used for this binary.
/// With the PID, we filter the samples and extract all LBR entries.
///
/// To aggregate LBR entries, we rely on a BinaryFunction map to locate the
/// original function where the event happened. Then, we convert a raw address
/// to an offset relative to the start of this function and aggregate branch
/// information for each function.
///
/// This must be coordinated with RewriteInstance so we have BinaryFunctions in
/// State::Disassembled. After this state, BinaryFunction will drop the
/// instruction map with original addresses we rely on to validate the traces
/// found in the LBR.
///
/// The last step is to write the aggregated data to disk in the output file
/// specified by the user.
class DataAggregator : public DataReader {
public:
  explicit DataAggregator(StringRef Filename) : DataReader(Filename) {
    start();
  }

  ~DataAggregator();

  StringRef getReaderName() const override { return "perf data aggregator"; }

  bool isTrustedSource() const override { return true; }

  Error preprocessProfile(BinaryContext &BC) override;

  Error readProfilePreCFG(BinaryContext &BC) override {
    return Error::success();
  }

  Error readProfile(BinaryContext &BC) override;

  bool mayHaveProfileData(const BinaryFunction &BF) override;

  /// Set Bolt Address Translation Table when processing samples collected in
  /// bolted binaries
  void setBAT(BoltAddressTranslation *B) override { BAT = B; }

  /// Check whether \p FileName is a perf.data file
  static bool checkPerfDataMagic(StringRef FileName);

private:
  struct PerfBranchSample {
    SmallVector<LBREntry, 32> LBR;
  };

  struct PerfBasicSample {
    StringRef EventName;
    uint64_t PC;
  };

  struct PerfMemSample {
    uint64_t PC;
    uint64_t Addr;
  };

  /// Used for parsing specific pre-aggregated input files.
  struct AggregatedLBREntry {
    enum Type : char { BRANCH = 0, FT, FT_EXTERNAL_ORIGIN };
    Location From;
    Location To;
    uint64_t Count;
    uint64_t Mispreds;
    Type EntryType;
  };

  struct Trace {
    uint64_t From;
    uint64_t To;
    Trace(uint64_t From, uint64_t To) : From(From), To(To) {}
    bool operator==(const Trace &Other) const {
      return From == Other.From && To == Other.To;
    }
  };

  struct TraceHash {
    size_t operator()(const Trace &L) const {
      return std::hash<uint64_t>()(L.From << 32 | L.To);
    }
  };

  struct FTInfo {
    uint64_t InternCount{0};
    uint64_t ExternCount{0};
  };

  struct TakenBranchInfo {
    uint64_t TakenCount{0};
    uint64_t MispredCount{0};
  };

  /// Intermediate storage for profile data. We save the results of parsing
  /// and use them later for processing and assigning profile.
  std::unordered_map<Trace, TakenBranchInfo, TraceHash> BranchLBRs;
  std::unordered_map<Trace, FTInfo, TraceHash> FallthroughLBRs;
  std::vector<AggregatedLBREntry> AggregatedLBRs;
  std::unordered_map<uint64_t, uint64_t> BasicSamples;
  std::vector<PerfMemSample> MemSamples;

  template <typename T> void clear(T &Container) {
    T TempContainer;
    TempContainer.swap(Container);
  }

  /// Perf utility full path name
  std::string PerfPath;

  /// Perf process spawning bookkeeping
  struct PerfProcessInfo {
    bool IsFinished{false};
    sys::ProcessInfo PI;
    SmallVector<char, 256> StdoutPath;
    SmallVector<char, 256> StderrPath;
  };

  /// Process info for spawned processes
  PerfProcessInfo MainEventsPPI;
  PerfProcessInfo MemEventsPPI;
  PerfProcessInfo MMapEventsPPI;
  PerfProcessInfo TaskEventsPPI;

  /// Kernel VM starts at fixed based address
  /// https://www.kernel.org/doc/Documentation/x86/x86_64/mm.txt
  static constexpr uint64_t KernelBaseAddr = 0xffff800000000000;

  /// Current list of created temporary files
  std::vector<std::string> TempFiles;

  /// Name of the binary with matching build-id from perf.data if different
  /// from the file name in BC.
  std::string BuildIDBinaryName;

  /// Memory map info for a single file as recorded in perf.data
  /// When a binary has multiple text segments, the Size is computed as the
  /// difference of the last address of these segments from the BaseAddress.
  /// The base addresses of all text segments must be the same.
  struct MMapInfo {
    uint64_t BaseAddress{0}; /// Base address of the mapped binary.
    uint64_t MMapAddress{0}; /// Address of the executable segment.
    uint64_t Size{0};        /// Size of the mapping.
    uint64_t Offset{0};      /// File offset of the mapped segment.
    int32_t PID{-1};         /// Process ID.
    bool Forked{false};      /// Was the process forked?
    uint64_t Time{0ULL};     /// Time in micro seconds.
  };

  /// Per-PID map info for the binary
  std::unordered_map<uint64_t, MMapInfo> BinaryMMapInfo;

  /// Fork event info
  struct ForkInfo {
    int32_t ParentPID;
    int32_t ChildPID;
    uint64_t Time{0ULL};
  };

  /// References to core BOLT data structures
  BinaryContext *BC{nullptr};

  BoltAddressTranslation *BAT{nullptr};

  /// Update function execution profile with a recorded trace.
  /// A trace is region of code executed between two LBR entries supplied in
  /// execution order.
  ///
  /// Return a vector of offsets corresponding to a trace in a function
  /// if the trace is valid, std::nullopt otherwise.
  std::optional<SmallVector<std::pair<uint64_t, uint64_t>, 16>>
  getFallthroughsInTrace(BinaryFunction &BF, const LBREntry &First,
                         const LBREntry &Second, uint64_t Count = 1) const;

  /// Record external entry into the function \p BF.
  ///
  /// Return true if the entry is valid, false otherwise.
  bool recordEntry(BinaryFunction &BF, uint64_t To, bool Mispred,
                   uint64_t Count = 1) const;

  /// Record exit from the function \p BF via a call or return.
  ///
  /// Return true if the exit point is valid, false otherwise.
  bool recordExit(BinaryFunction &BF, uint64_t From, bool Mispred,
                  uint64_t Count = 1) const;

  /// Aggregation statistics
  uint64_t NumInvalidTraces{0};
  uint64_t NumLongRangeTraces{0};
  /// Specifies how many samples were recorded in cold areas if we are dealing
  /// with profiling data collected in a bolted binary. For LBRs, incremented
  /// for the source of the branch to avoid counting cold activity twice (one
  /// for source and another for destination).
  uint64_t NumColdSamples{0};

  /// Looks into system PATH for Linux Perf and set up the aggregator to use it
  void findPerfExecutable();

  /// Launch a perf subprocess with given args and save output for later
  /// parsing.
  void launchPerfProcess(StringRef Name, PerfProcessInfo &PPI,
                         const char *ArgsString, bool Wait);

  /// Delete all temporary files created to hold the output generated by spawned
  /// subprocesses during the aggregation job
  void deleteTempFiles();

  // Semantic pass helpers

  /// Look up which function contains an address by using out map of
  /// disassembled BinaryFunctions
  BinaryFunction *getBinaryFunctionContainingAddress(uint64_t Address) const;

  /// Perform BAT translation for a given \p Func and return the parent
  /// BinaryFunction or nullptr.
  BinaryFunction *getBATParentFunction(const BinaryFunction &Func) const;

  /// Retrieve the location name to be used for samples recorded in \p Func.
  static StringRef getLocationName(const BinaryFunction &Func, bool BAT);

  /// Semantic actions - parser hooks to interpret parsed perf samples
  /// Register a sample (non-LBR mode), i.e. a new hit at \p Address
  bool doSample(BinaryFunction &Func, const uint64_t Address, uint64_t Count);

  /// Register an intraprocedural branch \p Branch.
  bool doIntraBranch(BinaryFunction &Func, uint64_t From, uint64_t To,
                     uint64_t Count, uint64_t Mispreds);

  /// Register an interprocedural branch from \p FromFunc to \p ToFunc with
  /// offsets \p From and \p To, respectively.
  bool doInterBranch(BinaryFunction *FromFunc, BinaryFunction *ToFunc,
                     uint64_t From, uint64_t To, uint64_t Count,
                     uint64_t Mispreds);

  /// Register a \p Branch.
  bool doBranch(uint64_t From, uint64_t To, uint64_t Count, uint64_t Mispreds,
                bool IsPreagg);

  /// Register a trace between two LBR entries supplied in execution order.
  bool doTrace(const LBREntry &First, const LBREntry &Second,
               uint64_t Count = 1);

  /// Parser helpers
  /// Return false if we exhausted our parser buffer and finished parsing
  /// everything
  bool hasData() const { return !ParsingBuf.empty(); }

  /// Print heat map based on LBR samples.
  std::error_code printLBRHeatMap();

  /// Parse a single perf sample containing a PID associated with a sequence of
  /// LBR entries. If the PID does not correspond to the binary we are looking
  /// for, return std::errc::no_such_process. If other parsing errors occur,
  /// return the error. Otherwise, return the parsed sample.
  ErrorOr<PerfBranchSample> parseBranchSample();

  /// Parse a single perf sample containing a PID associated with an event name
  /// and a PC
  ErrorOr<PerfBasicSample> parseBasicSample();

  /// Parse a single perf sample containing a PID associated with an IP and
  /// address.
  ErrorOr<PerfMemSample> parseMemSample();

  /// Parse pre-aggregated LBR samples created by an external tool
  ErrorOr<AggregatedLBREntry> parseAggregatedLBREntry();

  /// Parse either buildid:offset or just offset, representing a location in the
  /// binary. Used exclusively for pre-aggregated LBR samples.
  ErrorOr<Location> parseLocationOrOffset();

  /// Check if a field separator is the next char to parse and, if yes, consume
  /// it and return true
  bool checkAndConsumeFS();

  /// Consume the entire line
  void consumeRestOfLine();

  /// True if the next token in the parsing buffer is a new line, but don't
  /// consume it (peek only).
  bool checkNewLine();

  using PerfProcessErrorCallbackTy = std::function<void(int, StringRef)>;
  /// Prepare to parse data from a given perf script invocation.
  /// Returns an invocation exit code.
  int prepareToParse(StringRef Name, PerfProcessInfo &Process,
                     PerfProcessErrorCallbackTy Callback);

  /// Parse a single LBR entry as output by perf script -Fbrstack
  ErrorOr<LBREntry> parseLBREntry();

  /// Parse LBR sample, returns the number of traces.
  uint64_t parseLBRSample(const PerfBranchSample &Sample, bool NeedsSkylakeFix);

  /// Parse and pre-aggregate branch events.
  std::error_code parseBranchEvents();

  /// Process all branch events.
  void processBranchEvents();

  /// Parse the full output generated by perf script to report non-LBR samples.
  std::error_code parseBasicEvents();

  /// Process non-LBR events.
  void processBasicEvents();

  /// Parse the full output generated by perf script to report memory events.
  std::error_code parseMemEvents();

  /// Process parsed memory events profile.
  void processMemEvents();

  /// Parse a single line of a PERF_RECORD_MMAP2 event looking for a mapping
  /// between the binary name and its memory layout in a process with a given
  /// PID.
  /// On success return a <FileName, MMapInfo> pair.
  ErrorOr<std::pair<StringRef, MMapInfo>> parseMMapEvent();

  /// Parse PERF_RECORD_FORK event.
  std::optional<ForkInfo> parseForkEvent();

  /// Parse 'PERF_RECORD_COMM exec'. Don't consume the string.
  std::optional<int32_t> parseCommExecEvent();

  /// Parse the full output generated by `perf script --show-mmap-events`
  /// to generate mapping between binary files and their memory mappings for
  /// all PIDs.
  std::error_code parseMMapEvents();

  /// Parse output of `perf script --show-task-events`, and forked processes
  /// to the set of tracked PIDs.
  std::error_code parseTaskEvents();

  /// Parse a single pair of binary full path and associated build-id
  std::optional<std::pair<StringRef, StringRef>> parseNameBuildIDPair();

  /// Coordinate reading and parsing of pre-aggregated file
  ///
  /// The regular perf2bolt aggregation job is to read perf output directly.
  /// However, if the data is coming from a database instead of perf, one could
  /// write a query to produce a pre-aggregated file. This function deals with
  /// this case.
  ///
  /// The pre-aggregated file contains aggregated LBR data, but without binary
  /// knowledge. BOLT will parse it and, using information from the disassembled
  /// binary, augment it with fall-through edge frequency information. After
  /// this step is finished, this data can be either written to disk to be
  /// consumed by BOLT later, or can be used by BOLT immediately if kept in
  /// memory.
  ///
  /// File format syntax:
  /// {B|F|f} [<start_id>:]<start_offset> [<end_id>:]<end_offset> <count>
  ///       [<mispred_count>]
  ///
  /// B - indicates an aggregated branch
  /// F - an aggregated fall-through
  /// f - an aggregated fall-through with external origin - used to disambiguate
  ///       between a return hitting a basic block head and a regular internal
  ///       jump to the block
  ///
  /// <start_id> - build id of the object containing the start address. We can
  /// skip it for the main binary and use "X" for an unknown object. This will
  /// save some space and facilitate human parsing.
  ///
  /// <start_offset> - hex offset from the object base load address (0 for the
  /// main executable unless it's PIE) to the start address.
  ///
  /// <end_id>, <end_offset> - same for the end address.
  ///
  /// <count> - total aggregated count of the branch or a fall-through.
  ///
  /// <mispred_count> - the number of times the branch was mispredicted.
  /// Omitted for fall-throughs.
  ///
  /// Example:
  /// F 41be50 41be50 3
  /// F 41be90 41be90 4
  /// B 4b1942 39b57f0 3 0
  /// B 4b196f 4b19e0 2 0
  void parsePreAggregated();

  /// Parse the full output of pre-aggregated LBR samples generated by
  /// an external tool.
  std::error_code parsePreAggregatedLBRSamples();

  /// Process parsed pre-aggregated data.
  void processPreAggregated();

  /// If \p Address falls into the binary address space based on memory
  /// mapping info \p MMI, then adjust it for further processing by subtracting
  /// the base load address. External addresses, i.e. addresses that do not
  /// correspond to the binary allocated address space, are adjusted to avoid
  /// conflicts.
  void adjustAddress(uint64_t &Address, const MMapInfo &MMI) const {
    if (Address >= MMI.MMapAddress && Address < MMI.MMapAddress + MMI.Size) {
      Address -= MMI.BaseAddress;
    } else if (Address < MMI.Size) {
      // Make sure the address is not treated as belonging to the binary.
      Address = (-1ULL);
    }
  }

  /// Adjust addresses in \p LBR entry.
  void adjustLBR(LBREntry &LBR, const MMapInfo &MMI) const {
    adjustAddress(LBR.From, MMI);
    adjustAddress(LBR.To, MMI);
  }

  /// Ignore kernel/user transition LBR if requested
  bool ignoreKernelInterrupt(LBREntry &LBR) const;

  /// Populate functions in \p BC with profile.
  void processProfile(BinaryContext &BC);

  /// Start an aggregation job asynchronously.
  void start();

  /// Returns true if this aggregation job is using a translation table to
  /// remap samples collected on binaries already processed by BOLT.
  bool usesBAT() const { return BAT; }

  /// Force all subprocesses to stop and cancel aggregation
  void abort();

  /// Dump data structures into a file readable by llvm-bolt
  std::error_code writeAggregatedFile(StringRef OutputFilename) const;

  /// Dump translated data structures into YAML
  std::error_code writeBATYAML(BinaryContext &BC,
                               StringRef OutputFilename) const;

  /// Filter out binaries based on PID
  void filterBinaryMMapInfo();

  /// If we have a build-id available for the input file, use it to assist
  /// matching profile to a binary.
  ///
  /// If the binary name changed after profile collection, use build-id
  /// to get the proper name in perf data when build-ids are available.
  /// If \p FileBuildID has no match, then issue an error and exit.
  void processFileBuildID(StringRef FileBuildID);

  /// Debugging dump methods
  void dump() const;
  void dump(const LBREntry &LBR) const;
  void dump(const PerfBranchSample &Sample) const;
  void dump(const PerfMemSample &Sample) const;

public:
  /// If perf.data was collected without build ids, the buildid-list may contain
  /// incomplete entries. Return true if the buffer containing
  /// "perf buildid-list" output has only valid entries and is non- empty.
  /// Return false otherwise.
  bool hasAllBuildIDs();

  /// Parse the output generated by "perf buildid-list" to extract build-ids
  /// and return a file name matching a given \p FileBuildID.
  std::optional<StringRef> getFileNameForBuildID(StringRef FileBuildID);

  /// Get a constant reference to the parsed binary mmap entries.
  const std::unordered_map<uint64_t, MMapInfo> &getBinaryMMapInfo() {
    return BinaryMMapInfo;
  }

  friend class YAMLProfileWriter;
};
} // namespace bolt
} // namespace llvm

#endif
