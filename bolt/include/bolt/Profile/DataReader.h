//===- bolt/Profile/DataReader.h - Perf data reader -------------*- C++ -*-===//
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

#ifndef BOLT_PROFILE_DATA_READER_H
#define BOLT_PROFILE_DATA_READER_H

#include "bolt/Profile/ProfileReaderBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>
#include <vector>

namespace llvm {
class MCSymbol;

namespace bolt {

class BinaryFunction;

struct LBREntry {
  uint64_t From;
  uint64_t To;
  bool Mispred;
};

inline raw_ostream &operator<<(raw_ostream &OS, const LBREntry &LBR) {
  OS << "0x" << Twine::utohexstr(LBR.From) << " -> 0x"
     << Twine::utohexstr(LBR.To);
  return OS;
}

struct Location {
  bool IsSymbol;
  StringRef Name;
  uint64_t Offset;

  explicit Location(uint64_t Offset)
      : IsSymbol(false), Name("[unknown]"), Offset(Offset) {}

  Location(bool IsSymbol, StringRef Name, uint64_t Offset)
      : IsSymbol(IsSymbol), Name(Name), Offset(Offset) {}

  bool operator==(const Location &RHS) const {
    return IsSymbol == RHS.IsSymbol && Name == RHS.Name &&
           (Name == "[heap]" || Offset == RHS.Offset);
  }

  bool operator<(const Location &RHS) const {
    if (IsSymbol != RHS.IsSymbol)
      return IsSymbol < RHS.IsSymbol;

    if (Name != RHS.Name)
      return Name < RHS.Name;

    return Name != "[heap]" && Offset < RHS.Offset;
  }

  friend raw_ostream &operator<<(raw_ostream &OS, const Location &Loc);
};

typedef std::vector<std::pair<Location, Location>> BranchContext;

struct BranchInfo {
  Location From;
  Location To;
  int64_t Mispreds;
  int64_t Branches;

  BranchInfo(Location From, Location To, int64_t Mispreds, int64_t Branches)
      : From(std::move(From)), To(std::move(To)), Mispreds(Mispreds),
        Branches(Branches) {}

  bool operator==(const BranchInfo &RHS) const {
    return From == RHS.From && To == RHS.To;
  }

  bool operator<(const BranchInfo &RHS) const {
    if (From < RHS.From)
      return true;

    if (From == RHS.From)
      return (To < RHS.To);

    return false;
  }

  /// Merges branch and misprediction counts of \p BI with those of this object.
  void mergeWith(const BranchInfo &BI);

  void print(raw_ostream &OS) const;
};

struct FuncBranchData {
  typedef std::vector<BranchInfo> ContainerTy;

  StringRef Name;
  ContainerTy Data;
  ContainerTy EntryData;

  /// Total execution count for the function.
  int64_t ExecutionCount{0};

  /// Indicate if the data was used.
  bool Used{false};

  FuncBranchData() {}

  FuncBranchData(StringRef Name, ContainerTy Data)
      : Name(Name), Data(std::move(Data)) {}

  FuncBranchData(StringRef Name, ContainerTy Data, ContainerTy EntryData)
      : Name(Name), Data(std::move(Data)), EntryData(std::move(EntryData)) {}

  ErrorOr<const BranchInfo &> getBranch(uint64_t From, uint64_t To) const;

  /// Returns the branch info object associated with a direct call originating
  /// from the given offset. If no branch info object is found, an error is
  /// returned. If the offset corresponds to an indirect call the behavior is
  /// undefined.
  ErrorOr<const BranchInfo &> getDirectCallBranch(uint64_t From) const;

  /// Append the branch data of another function located \p Offset bytes away
  /// from the entry of this function.
  void appendFrom(const FuncBranchData &FBD, uint64_t Offset);

  /// Returns the total number of executed branches in this function
  /// by counting the number of executed branches for each BranchInfo
  uint64_t getNumExecutedBranches() const;

  /// Aggregation helpers
  DenseMap<uint64_t, DenseMap<uint64_t, size_t>> IntraIndex;
  DenseMap<uint64_t, DenseMap<Location, size_t>> InterIndex;
  DenseMap<uint64_t, DenseMap<Location, size_t>> EntryIndex;

  void bumpBranchCount(uint64_t OffsetFrom, uint64_t OffsetTo, uint64_t Count,
                       uint64_t Mispreds);
  void bumpCallCount(uint64_t OffsetFrom, const Location &To, uint64_t Count,
                     uint64_t Mispreds);
  void bumpEntryCount(const Location &From, uint64_t OffsetTo, uint64_t Count,
                      uint64_t Mispreds);
};

/// MemInfo represents a single memory load from an address \p Addr at an \p
/// Offset within a function.  \p Count represents how many times a particular
/// address was seen.
struct MemInfo {
  Location Offset;
  Location Addr;
  uint64_t Count;

  bool operator==(const MemInfo &RHS) const {
    return Offset == RHS.Offset && Addr == RHS.Addr;
  }

  bool operator<(const MemInfo &RHS) const {
    if (Offset < RHS.Offset)
      return true;

    if (Offset == RHS.Offset)
      return (Addr < RHS.Addr);

    return false;
  }

  void mergeWith(const MemInfo &MI) { Count += MI.Count; }

  void print(raw_ostream &OS) const;
  void prettyPrint(raw_ostream &OS) const;

  MemInfo(const Location &Offset, const Location &Addr, uint64_t Count = 0)
      : Offset(Offset), Addr(Addr), Count(Count) {}

  friend raw_ostream &operator<<(raw_ostream &OS, const MemInfo &MI) {
    MI.prettyPrint(OS);
    return OS;
  }
};

/// Helper class to store memory load events recorded in the address space of
/// a given function, analogous to FuncBranchData but for memory load events
/// instead of branches.
struct FuncMemData {
  typedef std::vector<MemInfo> ContainerTy;

  StringRef Name;
  ContainerTy Data;

  /// Indicate if the data was used.
  bool Used{false};

  DenseMap<uint64_t, DenseMap<Location, size_t>> EventIndex;

  /// Update \p Data with a memory event.  Events with the same
  /// \p Offset and \p Addr will be coalesced.
  void update(const Location &Offset, const Location &Addr);

  FuncMemData() {}

  FuncMemData(StringRef Name, ContainerTy Data)
      : Name(Name), Data(std::move(Data)) {}
};

/// Similar to BranchInfo, but instead of recording from-to address (an edge),
/// it records the address of a perf event and the number of times samples hit
/// this address.
struct SampleInfo {
  Location Loc;
  int64_t Hits;

  SampleInfo(Location Loc, int64_t Hits) : Loc(std::move(Loc)), Hits(Hits) {}

  bool operator==(const SampleInfo &RHS) const { return Loc == RHS.Loc; }

  bool operator<(const SampleInfo &RHS) const {
    if (Loc < RHS.Loc)
      return true;

    return false;
  }

  void print(raw_ostream &OS) const;

  void mergeWith(const SampleInfo &SI);
};

/// Helper class to store samples recorded in the address space of a given
/// function, analogous to FuncBranchData but for samples instead of branches.
struct FuncSampleData {
  typedef std::vector<SampleInfo> ContainerTy;

  StringRef Name;
  ContainerTy Data;

  FuncSampleData(StringRef Name, ContainerTy Data)
      : Name(Name), Data(std::move(Data)) {}

  /// Get the number of samples recorded in [Start, End)
  uint64_t getSamples(uint64_t Start, uint64_t End) const;

  /// Aggregation helper
  DenseMap<uint64_t, size_t> Index;

  void bumpCount(uint64_t Offset, uint64_t Count);
};

/// DataReader Class
///
class DataReader : public ProfileReaderBase {
public:
  explicit DataReader(StringRef Filename)
      : ProfileReaderBase(Filename), Diag(errs()) {}

  StringRef getReaderName() const override { return "branch profile reader"; }

  bool isTrustedSource() const override { return false; }

  virtual Error preprocessProfile(BinaryContext &BC) override;

  virtual Error readProfilePreCFG(BinaryContext &BC) override;

  virtual Error readProfile(BinaryContext &BC) override;

  virtual bool hasLocalsWithFileName() const override;

  virtual bool mayHaveProfileData(const BinaryFunction &BF) override;

  /// Return all event names used to collect this profile
  virtual StringSet<> getEventNames() const override { return EventNames; }

protected:
  /// Read profile information available for the function.
  void readProfile(BinaryFunction &BF);

  /// In functions with multiple entry points, the profile collection records
  /// data for other entry points in a different function entry. This function
  /// attempts to fetch extra profile data for each secondary entry point.
  bool fetchProfileForOtherEntryPoints(BinaryFunction &BF);

  /// Find the best matching profile for a function after the creation of basic
  /// blocks.
  void matchProfileData(BinaryFunction &BF);

  /// Find the best matching memory data profile for a function before the
  /// creation of basic blocks.
  void matchProfileMemData(BinaryFunction &BF);

  /// Check how closely \p BranchData matches the function \p BF.
  /// Return accuracy (ranging from 0.0 to 1.0) of the matching.
  float evaluateProfileData(BinaryFunction &BF,
                            const FuncBranchData &BranchData) const;

  /// If our profile data comes from sample addresses instead of LBR entries,
  /// collect sample count for all addresses in this function address space,
  /// aggregating them per basic block and assigning an execution count to each
  /// basic block based on the number of samples recorded at those addresses.
  /// The last step is to infer edge counts based on BB execution count. Note
  /// this is the opposite of the LBR way, where we infer BB execution count
  /// based on edge counts.
  void readSampleData(BinaryFunction &BF);

  /// Convert function-level branch data into instruction annotations.
  void convertBranchData(BinaryFunction &BF) const;

  /// Update function \p BF profile with a taken branch.
  /// \p Count could be 0 if verification of the branch is required.
  ///
  /// Return true if the branch is valid, false otherwise.
  bool recordBranch(BinaryFunction &BF, uint64_t From, uint64_t To,
                    uint64_t Count = 1, uint64_t Mispreds = 0) const;

  /// Parses the input bolt data file into internal data structures. We expect
  /// the file format to follow the syntax below.
  ///
  /// <is symbol?> <closest elf symbol or DSO name> <relative FROM address>
  /// <is symbol?> <closest elf symbol or DSO name> <relative TO address>
  /// <number of mispredictions> <number of branches>
  ///
  /// In <is symbol?> field we record 0 if our closest address is a DSO load
  /// address or 1 if our closest address is an ELF symbol.
  ///
  /// Examples:
  ///
  ///  1 main 3fb 0 /lib/ld-2.21.so 12 4 221
  ///
  /// The example records branches from symbol main, offset 3fb, to DSO ld-2.21,
  /// offset 12, with 4 mispredictions and 221 branches.
  ///
  ///  2 t2.c/func 11 1 globalfunc 1d 0 1775 2
  ///  0 1002 2
  ///  2 t2.c/func 31 2 t2.c/func d
  ///  2 t2.c/func 18 2 t2.c/func 20
  ///  0 773 2
  ///  2 t2.c/func 71 2 t2.c/func d
  ///  2 t2.c/func 18 2 t2.c/func 60
  ///
  /// The examples records branches from local symbol func (from t2.c), offset
  /// 11, to global symbol globalfunc, offset 1d, with 1775 branches, no
  /// mispreds. Of these branches, 1002 were preceeded by a sequence of
  /// branches from func, offset 18 to offset 20 and then from offset 31 to
  /// offset d. The rest 773 branches were preceeded by a different sequence
  /// of branches, from func, offset 18 to offset 60 and then from offset 71 to
  /// offset d.
  std::error_code parse();

  /// When no_lbr is the first line of the file, activate No LBR mode. In this
  /// mode we read the addresses where samples were recorded directly instead of
  /// LBR entries. The line format is almost the same, except for a missing <to>
  /// triple and a missing mispredictions field:
  ///
  /// no_lbr
  /// <is symbol?> <closest elf symbol or DSO name> <relative address> <count>
  /// ...
  ///
  /// Example:
  ///
  /// no_lbr                           # First line of fdata file
  ///  1 BZ2_compressBlock 466c 3
  ///  1 BZ2_hbMakeCodeLengths 29c 1
  ///
  std::error_code parseInNoLBRMode();

  /// Return branch data matching one of the names in \p FuncNames.
  FuncBranchData *
  getBranchDataForNames(const std::vector<StringRef> &FuncNames);

  /// Return branch data matching one of the \p Symbols.
  FuncBranchData *
  getBranchDataForSymbols(const std::vector<MCSymbol *> &Symbols);

  /// Return mem data matching one of the names in \p FuncNames.
  FuncMemData *getMemDataForNames(const std::vector<StringRef> &FuncNames);

  FuncSampleData *getFuncSampleData(const std::vector<StringRef> &FuncNames);

  /// Return a vector of all FuncBranchData matching the list of names.
  /// Internally use fuzzy matching to match special names like LTO-generated
  /// function names.
  std::vector<FuncBranchData *>
  getBranchDataForNamesRegex(const std::vector<StringRef> &FuncNames);

  /// Return a vector of all FuncMemData matching the list of names.
  /// Internally use fuzzy matching to match special names like LTO-generated
  /// function names.
  std::vector<FuncMemData *>
  getMemDataForNamesRegex(const std::vector<StringRef> &FuncNames);

  /// Return branch data profile associated with function \p BF  or nullptr
  /// if the function has no associated profile.
  FuncBranchData *getBranchData(const BinaryFunction &BF) const {
    auto FBDI = FuncsToBranches.find(&BF);
    if (FBDI == FuncsToBranches.end())
      return nullptr;
    return FBDI->second;
  }

  /// Updates branch profile data associated with function \p BF.
  void setBranchData(const BinaryFunction &BF, FuncBranchData *FBD) {
    FuncsToBranches[&BF] = FBD;
  }

  /// Return memory profile data associated with function \p BF, or nullptr
  /// if the function has no associated profile.
  FuncMemData *getMemData(const BinaryFunction &BF) const {
    auto FMDI = FuncsToMemData.find(&BF);
    if (FMDI == FuncsToMemData.end())
      return nullptr;
    return FMDI->second;
  }

  /// Updates the memory profile data associated with function \p BF.
  void setMemData(const BinaryFunction &BF, FuncMemData *FMD) {
    FuncsToMemData[&BF] = FMD;
  }

  using NamesToBranchesMapTy = StringMap<FuncBranchData>;
  using NamesToSamplesMapTy = StringMap<FuncSampleData>;
  using NamesToMemEventsMapTy = StringMap<FuncMemData>;
  using FuncsToBranchesMapTy =
      std::unordered_map<const BinaryFunction *, FuncBranchData *>;
  using FuncsToMemDataMapTy =
      std::unordered_map<const BinaryFunction *, FuncMemData *>;

  /// Dumps the entire data structures parsed. Used for debugging.
  void dump() const;

  /// Return false only if we are running with profiling data that lacks LBR.
  bool hasLBR() const { return !NoLBRMode; }

  /// Return true if the profiling data was collected in a bolted binary. This
  /// means we lose the ability to identify stale data at some branch locations,
  /// since we have to be more permissive in some cases.
  bool collectedInBoltedBinary() const { return BATMode; }

  /// Return true if event named \p Name was used to collect this profile data.
  bool usesEvent(StringRef Name) const {
    for (auto I = EventNames.begin(), E = EventNames.end(); I != E; ++I) {
      StringRef Event = I->getKey();
      if (Event.find(Name) != StringRef::npos)
        return true;
    }
    return false;
  }

  /// Open the file and parse the contents.
  std::error_code parseInput();

  /// Build suffix map once the profile data is parsed.
  void buildLTONameMaps();

  void reportError(StringRef ErrorMsg);
  bool expectAndConsumeFS();
  void consumeAllRemainingFS();
  bool checkAndConsumeNewLine();
  ErrorOr<StringRef> parseString(char EndChar, bool EndNl = false);
  ErrorOr<int64_t> parseNumberField(char EndChar, bool EndNl = false);
  ErrorOr<uint64_t> parseHexField(char EndChar, bool EndNl = false);
  ErrorOr<Location> parseLocation(char EndChar, bool EndNl, bool ExpectMemLoc);
  ErrorOr<Location> parseLocation(char EndChar, bool EndNl = false) {
    return parseLocation(EndChar, EndNl, false);
  }
  ErrorOr<Location> parseMemLocation(char EndChar, bool EndNl = false) {
    return parseLocation(EndChar, EndNl, true);
  }
  ErrorOr<BranchInfo> parseBranchInfo();
  ErrorOr<SampleInfo> parseSampleInfo();
  ErrorOr<MemInfo> parseMemInfo();
  ErrorOr<bool> maybeParseNoLBRFlag();
  ErrorOr<bool> maybeParseBATFlag();
  bool hasBranchData();
  bool hasMemData();

  /// An in-memory copy of the input data file - owns strings used in reader.
  std::unique_ptr<MemoryBuffer> FileBuf;
  raw_ostream &Diag;
  StringRef ParsingBuf;
  unsigned Line{0};
  unsigned Col{0};
  NamesToBranchesMapTy NamesToBranches;
  NamesToSamplesMapTy NamesToSamples;
  NamesToMemEventsMapTy NamesToMemEvents;
  FuncsToBranchesMapTy FuncsToBranches;
  FuncsToMemDataMapTy FuncsToMemData;
  bool NoLBRMode{false};
  bool BATMode{false};
  StringSet<> EventNames;
  static const char FieldSeparator = ' ';

  /// Maps of common LTO names to possible matching profiles.
  StringMap<std::vector<FuncBranchData *>> LTOCommonNameMap;
  StringMap<std::vector<FuncMemData *>> LTOCommonNameMemMap;

public:
  void setParsingBuffer(StringRef Buffer) { ParsingBuf = Buffer; }
};

} // namespace bolt

/// DenseMapInfo allows us to use the DenseMap LLVM data structure to store
/// Locations
template <> struct DenseMapInfo<bolt::Location> {
  static inline bolt::Location getEmptyKey() {
    return bolt::Location(true, StringRef(), static_cast<uint64_t>(-1LL));
  }
  static inline bolt::Location getTombstoneKey() {
    return bolt::Location(true, StringRef(), static_cast<uint64_t>(-2LL));
    ;
  }
  static unsigned getHashValue(const bolt::Location &L) {
    return (unsigned(DenseMapInfo<StringRef>::getHashValue(L.Name)) >> 4) ^
           (unsigned(L.Offset));
  }
  static bool isEqual(const bolt::Location &LHS, const bolt::Location &RHS) {
    return LHS.IsSymbol == RHS.IsSymbol && LHS.Name == RHS.Name &&
           LHS.Offset == RHS.Offset;
  }
};

} // namespace llvm

#endif
