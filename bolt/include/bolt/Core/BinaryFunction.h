//===- bolt/Core/BinaryFunction.h - Low-level function ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the BinaryFunction class. It represents
// a function at the lowest IR level. Typically, a BinaryFunction represents a
// function object in a compiled and linked binary file. However, a
// BinaryFunction can also be constructed manually, e.g. for injecting into a
// binary file.
//
// A BinaryFunction could be in one of the several states described in
// BinaryFunction::State. While in the disassembled state, it will contain a
// list of instructions with their offsets. In the CFG state, it will contain a
// list of BinaryBasicBlocks that form a control-flow graph. This state is best
// suited for binary analysis and optimizations. However, sometimes it's
// impossible to build the precise CFG due to the ambiguity of indirect
// branches.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_BINARY_FUNCTION_H
#define BOLT_CORE_BINARY_FUNCTION_H

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryDomTree.h"
#include "bolt/Core/BinaryLoop.h"
#include "bolt/Core/BinarySection.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/JumpTable.h"
#include "bolt/Core/MCPlus.h"
#include "bolt/Utils/NameResolver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace llvm::object;

namespace llvm {

class DWARFUnit;

namespace bolt {

using InputOffsetToAddressMapTy = std::unordered_multimap<uint64_t, uint64_t>;

/// Types of macro-fusion alignment corrections.
enum MacroFusionType { MFT_NONE, MFT_HOT, MFT_ALL };

enum IndirectCallPromotionType : char {
  ICP_NONE,        /// Don't perform ICP.
  ICP_CALLS,       /// Perform ICP on indirect calls.
  ICP_JUMP_TABLES, /// Perform ICP on jump tables.
  ICP_ALL          /// Perform ICP on calls and jump tables.
};

/// Hash functions supported for BF/BB hashing.
enum class HashFunction : char {
  StdHash, /// std::hash, implementation is platform-dependent. Provided for
           /// backwards compatibility.
  XXH3,    /// llvm::xxh3_64bits, the default.
  Default = XXH3,
};

/// Information on a single indirect call to a particular callee.
struct IndirectCallProfile {
  MCSymbol *Symbol;
  uint32_t Offset;
  uint64_t Count;
  uint64_t Mispreds;

  IndirectCallProfile(MCSymbol *Symbol, uint64_t Count, uint64_t Mispreds,
                      uint32_t Offset = 0)
      : Symbol(Symbol), Offset(Offset), Count(Count), Mispreds(Mispreds) {}

  bool operator==(const IndirectCallProfile &Other) const {
    return Symbol == Other.Symbol && Offset == Other.Offset;
  }
};

/// Aggregated information for an indirect call site.
using IndirectCallSiteProfile = SmallVector<IndirectCallProfile, 4>;

inline raw_ostream &operator<<(raw_ostream &OS,
                               const bolt::IndirectCallSiteProfile &ICSP) {
  std::string TempString;
  raw_string_ostream SS(TempString);

  const char *Sep = "\n        ";
  uint64_t TotalCount = 0;
  uint64_t TotalMispreds = 0;
  for (const IndirectCallProfile &CSP : ICSP) {
    SS << Sep << "{ " << (CSP.Symbol ? CSP.Symbol->getName() : "<unknown>")
       << ": " << CSP.Count << " (" << CSP.Mispreds << " misses) }";
    Sep = ",\n        ";
    TotalCount += CSP.Count;
    TotalMispreds += CSP.Mispreds;
  }

  OS << TotalCount << " (" << TotalMispreds << " misses) :" << TempString;
  return OS;
}

/// BinaryFunction is a representation of machine-level function.
///
/// In the input binary, an instance of BinaryFunction can represent a fragment
/// of a function if the higher-level function was split, e.g. into hot and cold
/// parts. The fragment containing the main entry point is called a parent
/// or the main fragment.
class BinaryFunction {
public:
  enum class State : char {
    Empty = 0,     /// Function body is empty.
    Disassembled,  /// Function have been disassembled.
    CFG,           /// Control flow graph has been built.
    CFG_Finalized, /// CFG is finalized. No optimizations allowed.
    EmittedCFG,    /// Instructions have been emitted to output.
    Emitted,       /// Same as above plus CFG is destroyed.
  };

  /// Types of profile the function can use. Could be a combination.
  enum {
    PF_NONE = 0,     /// No profile.
    PF_BRANCH = 1,   /// Profile is based on branches or branch stacks.
    PF_BASIC = 2,    /// Non-branch IP sample-based profile.
    PF_MEMEVENT = 4, /// Profile has mem events.
  };

  void setContainedNegateRAState() { HadNegateRAState = true; }
  bool containedNegateRAState() const { return HadNegateRAState; }
  void setInitialRAState(bool State) { InitialRAState = State; }
  bool getInitialRAState() { return InitialRAState; }

  /// Struct for tracking exception handling ranges.
  struct CallSite {
    const MCSymbol *Start;
    const MCSymbol *End;
    const MCSymbol *LP;
    uint64_t Action;
  };

  using CallSitesList = SmallVector<std::pair<FragmentNum, CallSite>, 0>;
  using CallSitesRange = iterator_range<CallSitesList::const_iterator>;

  using IslandProxiesType =
      std::map<BinaryFunction *, std::map<const MCSymbol *, MCSymbol *>>;

  struct IslandInfo {
    /// Temporary holder of offsets that are data markers (used in AArch)
    /// It is possible to have data in code sections. To ease the identification
    /// of data in code sections, the ABI requires the symbol table to have
    /// symbols named "$d" identifying the start of data inside code and "$x"
    /// identifying the end of a chunk of data inside code. DataOffsets contain
    /// all offsets of $d symbols and CodeOffsets all offsets of $x symbols.
    std::set<uint64_t> DataOffsets;
    std::set<uint64_t> CodeOffsets;

    /// List of relocations associated with data in the constant island
    std::map<uint64_t, Relocation> Relocations;

    /// Set true if constant island contains dynamic relocations, which may
    /// happen if binary is linked with -z notext option.
    bool HasDynamicRelocations{false};

    /// Offsets in function that are data values in a constant island identified
    /// after disassembling
    std::map<uint64_t, MCSymbol *> Offsets;
    SmallPtrSet<MCSymbol *, 4> Symbols;
    DenseMap<const MCSymbol *, BinaryFunction *> ProxySymbols;
    DenseMap<const MCSymbol *, MCSymbol *> ColdSymbols;
    /// Keeps track of other functions we depend on because there is a reference
    /// to the constant islands in them.
    IslandProxiesType Proxies, ColdProxies;
    SmallPtrSet<BinaryFunction *, 1> Dependency; // The other way around

    mutable MCSymbol *FunctionConstantIslandLabel{nullptr};
    mutable MCSymbol *FunctionColdConstantIslandLabel{nullptr};
  };

  static constexpr uint64_t COUNT_NO_PROFILE =
      BinaryBasicBlock::COUNT_NO_PROFILE;

  static const char TimerGroupName[];
  static const char TimerGroupDesc[];

  using BasicBlockOrderType = SmallVector<BinaryBasicBlock *, 0>;

  /// Mark injected functions
  bool IsInjected = false;

  using LSDATypeTableTy = SmallVector<uint64_t, 0>;

  /// List of DWARF CFI instructions. Original CFI from the binary must be
  /// sorted w.r.t. offset that it appears. We rely on this to replay CFIs
  /// if needed (to fix state after reordering BBs).
  using CFIInstrMapType = SmallVector<MCCFIInstruction, 0>;
  using cfi_iterator = CFIInstrMapType::iterator;
  using const_cfi_iterator = CFIInstrMapType::const_iterator;

private:
  /// Current state of the function.
  State CurrentState{State::Empty};

  /// Indicates if the Function contained .cfi-negate-ra-state. These are not
  /// read from the binary. This boolean is used when deciding to run the
  /// .cfi-negate-ra-state rewriting passes on a function or not.
  bool HadNegateRAState{false};
  bool InitialRAState{false};

  /// A list of symbols associated with the function entry point.
  ///
  /// Multiple symbols would typically result from identical code-folding
  /// optimization.
  typedef SmallVector<MCSymbol *, 1> SymbolListTy;
  SymbolListTy Symbols;

  /// The list of names this function is known under. Used for fuzzy-matching
  /// the function to its name in a profile, command line, etc.
  SmallVector<std::string, 0> Aliases;

  /// Containing section in the input file.
  BinarySection *OriginSection = nullptr;

  /// Address of the function in memory. Also could be an offset from
  /// base address for position independent binaries.
  uint64_t Address;

  /// Original size of the function.
  uint64_t Size;

  /// Address of the function in output.
  uint64_t OutputAddress{0};

  /// Size of the function in the output file.
  uint64_t OutputSize{0};

  /// Maximum size this function is allowed to have.
  uint64_t MaxSize{std::numeric_limits<uint64_t>::max()};

  /// Alignment requirements for the function.
  uint16_t Alignment{2};

  /// Maximum number of bytes used for alignment of hot part of the function.
  uint16_t MaxAlignmentBytes{0};

  /// Maximum number of bytes used for alignment of cold part of the function.
  uint16_t MaxColdAlignmentBytes{0};

  const MCSymbol *PersonalityFunction{nullptr};
  uint8_t PersonalityEncoding{dwarf::DW_EH_PE_sdata4 | dwarf::DW_EH_PE_pcrel};

  BinaryContext &BC;

  std::unique_ptr<BinaryLoopInfo> BLI;
  std::unique_ptr<BinaryDominatorTree> BDT;

  /// All labels in the function that are referenced via relocations from
  /// data objects. Typically these are jump table destinations and computed
  /// goto labels.
  std::set<uint64_t> ExternallyReferencedOffsets;

  /// Offsets of indirect branches with unknown destinations.
  std::set<uint64_t> UnknownIndirectBranchOffsets;

  /// A set of local and global symbols corresponding to secondary entry points.
  /// Each additional function entry point has a corresponding entry in the map.
  /// The key is a local symbol corresponding to a basic block and the value
  /// is a global symbol corresponding to an external entry point.
  DenseMap<const MCSymbol *, MCSymbol *> SecondaryEntryPoints;

  /// False if the function is too complex to reconstruct its control
  /// flow graph.
  /// In relocation mode we still disassemble and re-assemble such functions.
  bool IsSimple{true};

  /// Indication that the function should be ignored for optimization purposes.
  /// If we can skip emission of some functions, then ignored functions could
  /// be not fully disassembled and will not be emitted.
  bool IsIgnored{false};

  /// Pseudo functions should not be disassembled or emitted.
  bool IsPseudo{false};

  /// True if the original function code has all necessary relocations to track
  /// addresses of functions emitted to new locations. Typically set for
  /// functions that we are not going to emit.
  bool HasExternalRefRelocations{false};

  /// True if the function has an indirect branch with unknown destination.
  bool HasUnknownControlFlow{false};

  /// The code from inside the function references one of the code locations
  /// from the same function as a data, i.e. it's possible the label is used
  /// inside an address calculation or could be referenced from outside.
  bool HasInternalLabelReference{false};

  /// In AArch64, preserve nops to maintain code equal to input (assuming no
  /// optimizations are done).
  bool PreserveNops{false};

  /// Indicate if this function has associated exception handling metadata.
  bool HasEHRanges{false};

  /// True if the function uses DW_CFA_GNU_args_size CFIs.
  bool UsesGnuArgsSize{false};

  /// True if the function might have a profile available externally.
  /// Used to check if processing of the function is required under certain
  /// conditions.
  bool HasProfileAvailable{false};

  bool HasMemoryProfile{false};

  /// Execution halts whenever this function is entered.
  bool TrapsOnEntry{false};

  /// True if the function is a fragment of another function. This means that
  /// this function could only be entered via its parent or one of its sibling
  /// fragments. It could be entered at any basic block. It can also return
  /// the control to any basic block of its parent or its sibling.
  bool IsFragment{false};

  /// Indicate that the function body has SDT marker
  bool HasSDTMarker{false};

  /// Indicate that the function body has Pseudo Probe
  bool HasPseudoProbe{BC.getUniqueSectionByName(".pseudo_probe_desc") &&
                      BC.getUniqueSectionByName(".pseudo_probe")};

  /// True if the function uses ORC format for stack unwinding.
  bool HasORC{false};

  /// True if the function contains explicit or implicit indirect branch to its
  /// split fragments, e.g., split jump table, landing pad in split fragment
  bool HasIndirectTargetToSplitFragment{false};

  /// True if there are no control-flow edges with successors in other functions
  /// (i.e. if tail calls have edges to function-local basic blocks).
  /// Set to false by SCTC. Dynostats can't be reliably computed for
  /// functions with non-canonical CFG.
  /// This attribute is only valid when hasCFG() == true.
  bool HasCanonicalCFG{true};

  /// True if another function body was merged into this one.
  bool HasFunctionsFoldedInto{false};

  /// True if the function is used for patching code at a fixed address.
  bool IsPatch{false};

  /// True if the original entry point of the function may get called, but the
  /// original body cannot be executed and needs to be patched with code that
  /// redirects execution to the new function body.
  bool NeedsPatch{false};

  /// True if the function should not have an associated symbol table entry.
  bool IsAnonymous{false};

  /// Name for the section this function code should reside in.
  std::string CodeSectionName;

  /// Name for the corresponding cold code section.
  std::string ColdCodeSectionName;

  /// Parent function fragment for split function fragments.
  using FragmentsSetTy = SmallPtrSet<BinaryFunction *, 1>;
  FragmentsSetTy ParentFragments;

  /// Indicate if the function body was folded into another function.
  /// Used by ICF optimization.
  BinaryFunction *FoldedIntoFunction{nullptr};

  /// All fragments for a parent function.
  FragmentsSetTy Fragments;

  /// The profile data for the number of times the function was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  /// Profile data for the number of times this function was entered from
  /// external code (DSO, JIT, etc).
  uint64_t ExternEntryCount{0};

  /// Profile match ratio.
  float ProfileMatchRatio{0.0f};

  /// Raw branch count for this function in the profile.
  uint64_t RawSampleCount{0};

  /// Dynamically executed function bytes, used for density computation.
  uint64_t SampleCountInBytes{0};

  /// Indicates the type of profile the function is using.
  uint16_t ProfileFlags{PF_NONE};

  /// True if the function's input profile data has been inaccurate but has
  /// been adjusted by the profile inference algorithm.
  bool HasInferredProfile{false};

  /// For functions with mismatched profile we store all call profile
  /// information at a function level (as opposed to tying it to
  /// specific call sites).
  IndirectCallSiteProfile AllCallSites;

  /// Score of the function (estimated number of instructions executed,
  /// according to profile data). -1 if the score has not been calculated yet.
  mutable int64_t FunctionScore{-1};

  /// Original LSDA address for the function.
  uint64_t LSDAAddress{0};

  /// Original LSDA type encoding
  unsigned LSDATypeEncoding{dwarf::DW_EH_PE_omit};

  /// All compilation units this function belongs to.
  /// Maps DWARF unit offset to the unit pointer.
  DenseMap<uint64_t, DWARFUnit *> DwarfUnitMap;

  /// Last computed hash value. Note that the value could be recomputed using
  /// different parameters by every pass.
  mutable uint64_t Hash{0};

  /// Function GUID assigned externally.
  uint64_t GUID{0};

  /// For PLT functions it contains a symbol associated with a function
  /// reference. It is nullptr for non-PLT functions.
  const MCSymbol *PLTSymbol{nullptr};

  /// Function order for streaming into the destination binary.
  uint32_t Index{-1U};

  /// Function is referenced by a non-control flow instruction.
  bool HasAddressTaken{false};

  /// Get basic block index assuming it belongs to this function.
  unsigned getIndex(const BinaryBasicBlock *BB) const {
    assert(BB->getIndex() < BasicBlocks.size());
    return BB->getIndex();
  }

  /// Release memory taken by the list.
  template <typename T> BinaryFunction &clearList(T &List) {
    T TempList;
    TempList.swap(List);
    return *this;
  }

  /// Update the indices of all the basic blocks starting at StartIndex.
  void updateBBIndices(const unsigned StartIndex);

  /// Annotate each basic block entry with its current CFI state. This is
  /// run right after the construction of CFG while basic blocks are in their
  /// original order.
  void annotateCFIState();

  /// Associate DW_CFA_GNU_args_size info with invoke instructions
  /// (call instructions with non-empty landing pad).
  void propagateGnuArgsSizeInfo(MCPlusBuilder::AllocatorIdTy AllocId);

  /// Synchronize branch instructions with CFG.
  void postProcessBranches();

  /// The address offset where we emitted the constant island, that is, the
  /// chunk of data in the function code area (AArch only)
  int64_t OutputDataOffset{0};
  int64_t OutputColdDataOffset{0};

  /// Map labels to corresponding basic blocks.
  DenseMap<const MCSymbol *, BinaryBasicBlock *> LabelToBB;

  using BranchListType = SmallVector<std::pair<uint32_t, uint32_t>, 0>;
  BranchListType TakenBranches;   /// All local taken branches.
  BranchListType IgnoredBranches; /// Branches ignored by CFG purposes.

  /// Map offset in the function to a label.
  /// Labels are used for building CFG for simple functions. For non-simple
  /// function in relocation mode we need to emit them for relocations
  /// referencing function internals to work (e.g. jump tables).
  using LabelsMapType = std::map<uint32_t, MCSymbol *>;
  LabelsMapType Labels;

  /// Temporary holder of instructions before CFG is constructed.
  /// Map offset in the function to MCInst.
  using InstrMapType = std::map<uint32_t, MCInst>;
  InstrMapType Instructions;

  /// We don't decode Call Frame Info encoded in DWARF program state
  /// machine. Instead we define a "CFI State" - a frame information that
  /// is a result of executing FDE CFI program up to a given point. The
  /// program consists of opaque Call Frame Instructions:
  ///
  ///   CFI #0
  ///   CFI #1
  ///   ....
  ///   CFI #N
  ///
  /// When we refer to "CFI State K" - it corresponds to a row in an abstract
  /// Call Frame Info table. This row is reached right before executing CFI #K.
  ///
  /// At any point of execution in a function we are in any one of (N + 2)
  /// states described in the original FDE program. We can't have more states
  /// without intelligent processing of CFIs.
  ///
  /// When the final layout of basic blocks is known, and we finalize CFG,
  /// we modify the original program to make sure the same state could be
  /// reached even when basic blocks containing CFI instructions are executed
  /// in a different order.
  CFIInstrMapType FrameInstructions;

  /// A map of restore state CFI instructions to their equivalent CFI
  /// instructions that produce the same state, in order to eliminate
  /// remember-restore CFI instructions when rewriting CFI.
  DenseMap<int32_t, SmallVector<int32_t, 4>> FrameRestoreEquivalents;

  // For tracking exception handling ranges.
  CallSitesList CallSites;

  /// Binary blobs representing action, type, and type index tables for this
  /// function' LSDA (exception handling).
  ArrayRef<uint8_t> LSDAActionTable;
  ArrayRef<uint8_t> LSDATypeIndexTable;

  /// Vector of addresses of types referenced by LSDA.
  LSDATypeTableTy LSDATypeTable;

  /// Vector of addresses of entries in LSDATypeTable used for indirect
  /// addressing.
  LSDATypeTableTy LSDATypeAddressTable;

  /// Marking for the beginnings of language-specific data areas for each
  /// fragment of the function.
  SmallVector<MCSymbol *, 0> LSDASymbols;

  /// Each function fragment may have another fragment containing all landing
  /// pads for it. If that's the case, the LP fragment will be stored in the
  /// vector below with indexing starting with the main fragment.
  SmallVector<std::optional<FragmentNum>, 0> LPFragments;

  /// Map to discover which CFIs are attached to a given instruction offset.
  /// Maps an instruction offset into a FrameInstructions offset.
  /// This is only relevant to the buildCFG phase and is discarded afterwards.
  std::multimap<uint32_t, uint32_t> OffsetToCFI;

  /// List of CFI instructions associated with the CIE (common to more than one
  /// function and that apply before the entry basic block).
  CFIInstrMapType CIEFrameInstructions;

  /// All compound jump tables for this function. This duplicates what's stored
  /// in the BinaryContext, but additionally it gives quick access for all
  /// jump tables used by this function.
  ///
  /// <OriginalAddress> -> <JumpTable *>
  std::map<uint64_t, JumpTable *> JumpTables;

  /// All jump table sites in the function before CFG is built.
  SmallVector<std::pair<uint64_t, uint64_t>, 0> JTSites;

  /// List of relocations in this function.
  std::map<uint64_t, Relocation> Relocations;

  /// Information on function constant islands.
  std::unique_ptr<IslandInfo> Islands;

  // Blocks are kept sorted in the layout order. If we need to change the
  // layout (if BasicBlocksLayout stores a different order than BasicBlocks),
  // the terminating instructions need to be modified.
  using BasicBlockListType = SmallVector<BinaryBasicBlock *, 0>;
  BasicBlockListType BasicBlocks;
  BasicBlockListType DeletedBasicBlocks;

  FunctionLayout Layout;

  /// BasicBlockOffsets are used during CFG construction to map from code
  /// offsets to BinaryBasicBlocks.  Any modifications made to the CFG
  /// after initial construction are not reflected in this data structure.
  using BasicBlockOffset = std::pair<uint64_t, BinaryBasicBlock *>;
  struct CompareBasicBlockOffsets {
    bool operator()(const BasicBlockOffset &A,
                    const BasicBlockOffset &B) const {
      return A.first < B.first;
    }
  };
  SmallVector<BasicBlockOffset, 0> BasicBlockOffsets;

  SmallVector<MCSymbol *, 0> ColdSymbols;

  /// Symbol at the end of each fragment of a split function.
  mutable SmallVector<MCSymbol *, 0> FunctionEndLabels;

  /// Unique number associated with the function.
  uint64_t FunctionNumber;

  /// Count the number of functions created.
  static uint64_t Count;

  /// Register alternative function name.
  void addAlternativeName(std::string NewName) {
    Aliases.push_back(std::move(NewName));
  }

  /// Return a label at a given \p Address in the function. If the label does
  /// not exist - create it. Assert if the \p Address does not belong to
  /// the function. If \p CreatePastEnd is true, then return the function
  /// end label when the \p Address points immediately past the last byte
  /// of the function.
  /// NOTE: the function always returns a local (temp) symbol, even if there's
  ///       a global symbol that corresponds to an entry at this address.
  MCSymbol *getOrCreateLocalLabel(uint64_t Address, bool CreatePastEnd = false);

  /// Register an data entry at a given \p Offset into the function.
  void markDataAtOffset(uint64_t Offset) {
    if (!Islands)
      Islands = std::make_unique<IslandInfo>();
    Islands->DataOffsets.emplace(Offset);
  }

  /// Register an entry point at a given \p Offset into the function.
  void markCodeAtOffset(uint64_t Offset) {
    if (!Islands)
      Islands = std::make_unique<IslandInfo>();
    Islands->CodeOffsets.emplace(Offset);
  }

  /// Register an internal offset in a function referenced from outside.
  void registerReferencedOffset(uint64_t Offset) {
    ExternallyReferencedOffsets.emplace(Offset);
  }

  /// True if there are references to internals of this function from data,
  /// e.g. from jump tables.
  bool hasInternalReference() const {
    return !ExternallyReferencedOffsets.empty();
  }

  /// Return an entry ID corresponding to a symbol known to belong to
  /// the function.
  ///
  /// Prefer to use BinaryContext::getFunctionForSymbol(EntrySymbol, &ID)
  /// instead of calling this function directly.
  uint64_t getEntryIDForSymbol(const MCSymbol *EntrySymbol) const;

  /// If the function represents a secondary split function fragment, set its
  /// parent fragment to \p BF.
  void addParentFragment(BinaryFunction &BF) {
    assert(this != &BF);
    assert(IsFragment && "function must be a fragment to have a parent");
    ParentFragments.insert(&BF);
  }

  /// Register a child fragment for the main fragment of a split function.
  void addFragment(BinaryFunction &BF) {
    assert(this != &BF);
    Fragments.insert(&BF);
  }

  void addInstruction(uint64_t Offset, MCInst &&Instruction) {
    Instructions.emplace(Offset, std::forward<MCInst>(Instruction));
  }

  /// Convert CFI instructions to a standard form (remove remember/restore).
  void normalizeCFIState();

  /// Analyze and process indirect branch \p Instruction before it is
  /// added to Instructions list.
  IndirectBranchType processIndirectBranch(MCInst &Instruction, unsigned Size,
                                           uint64_t Offset,
                                           uint64_t &TargetAddress);

  BinaryFunction &operator=(const BinaryFunction &) = delete;
  BinaryFunction(const BinaryFunction &) = delete;

  friend class MachORewriteInstance;
  friend class RewriteInstance;
  friend class BinaryContext;
  friend class DataReader;
  friend class DataAggregator;

  static std::string buildCodeSectionName(StringRef Name,
                                          const BinaryContext &BC);
  static std::string buildColdCodeSectionName(StringRef Name,
                                              const BinaryContext &BC);

  /// Creation should be handled by RewriteInstance or BinaryContext
  BinaryFunction(const std::string &Name, BinarySection &Section,
                 uint64_t Address, uint64_t Size, BinaryContext &BC)
      : OriginSection(&Section), Address(Address), Size(Size), BC(BC),
        CodeSectionName(buildCodeSectionName(Name, BC)),
        ColdCodeSectionName(buildColdCodeSectionName(Name, BC)),
        FunctionNumber(++Count) {
    Symbols.push_back(BC.Ctx->getOrCreateSymbol(Name));
  }

  /// This constructor is used to create an injected function
  BinaryFunction(const std::string &Name, BinaryContext &BC, bool IsSimple)
      : Address(0), Size(0), BC(BC), IsSimple(IsSimple),
        CodeSectionName(buildCodeSectionName(Name, BC)),
        ColdCodeSectionName(buildColdCodeSectionName(Name, BC)),
        FunctionNumber(++Count) {
    Symbols.push_back(BC.Ctx->getOrCreateSymbol(Name));
    IsInjected = true;
  }

  /// Create a basic block at a given \p Offset in the function and append it
  /// to the end of list of blocks. Used during CFG construction only.
  BinaryBasicBlock *addBasicBlockAt(uint64_t Offset, MCSymbol *Label) {
    assert(CurrentState == State::Disassembled &&
           "Cannot add block with an offset in non-disassembled state.");
    assert(!getBasicBlockAtOffset(Offset) &&
           "Basic block already exists at the offset.");

    BasicBlocks.emplace_back(createBasicBlock(Label).release());
    BinaryBasicBlock *BB = BasicBlocks.back();

    BB->setIndex(BasicBlocks.size() - 1);
    BB->setOffset(Offset);

    BasicBlockOffsets.emplace_back(Offset, BB);
    assert(llvm::is_sorted(BasicBlockOffsets, CompareBasicBlockOffsets()) &&
           llvm::is_sorted(blocks()));

    return BB;
  }

  /// Clear state of the function that could not be disassembled or if its
  /// disassembled state was later invalidated.
  void clearDisasmState();

  /// Release memory allocated for CFG and instructions.
  /// We still keep basic blocks for address translation/mapping purposes.
  void releaseCFG() {
    for (BinaryBasicBlock *BB : BasicBlocks)
      BB->releaseCFG();
    for (BinaryBasicBlock *BB : DeletedBasicBlocks)
      BB->releaseCFG();

    clearList(CallSites);
    clearList(LSDATypeTable);
    clearList(LSDATypeAddressTable);

    clearList(LabelToBB);

    if (!isMultiEntry())
      clearList(Labels);

    clearList(FrameInstructions);
    clearList(FrameRestoreEquivalents);
  }

public:
  BinaryFunction(BinaryFunction &&) = default;

  using iterator = pointee_iterator<BasicBlockListType::iterator>;
  using const_iterator = pointee_iterator<BasicBlockListType::const_iterator>;
  using reverse_iterator =
      pointee_iterator<BasicBlockListType::reverse_iterator>;
  using const_reverse_iterator =
      pointee_iterator<BasicBlockListType::const_reverse_iterator>;

  // CFG iterators.
  iterator                 begin()       { return BasicBlocks.begin(); }
  const_iterator           begin() const { return BasicBlocks.begin(); }
  iterator                 end  ()       { return BasicBlocks.end();   }
  const_iterator           end  () const { return BasicBlocks.end();   }

  reverse_iterator        rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator  rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator        rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator  rend  () const { return BasicBlocks.rend();   }

  size_t                    size() const { return BasicBlocks.size();}
  bool                     empty() const { return BasicBlocks.empty(); }
  const BinaryBasicBlock &front() const  { return *BasicBlocks.front(); }
        BinaryBasicBlock &front()        { return *BasicBlocks.front(); }
  const BinaryBasicBlock & back() const  { return *BasicBlocks.back(); }
        BinaryBasicBlock & back()        { return *BasicBlocks.back(); }
  inline iterator_range<iterator> blocks() {
    return iterator_range<iterator>(begin(), end());
  }
  inline iterator_range<const_iterator> blocks() const {
    return iterator_range<const_iterator>(begin(), end());
  }

  // Iterators by pointer.
  BasicBlockListType::iterator pbegin()  { return BasicBlocks.begin(); }
  BasicBlockListType::iterator pend()    { return BasicBlocks.end(); }

  cfi_iterator        cie_begin()       { return CIEFrameInstructions.begin(); }
  const_cfi_iterator  cie_begin() const { return CIEFrameInstructions.begin(); }
  cfi_iterator        cie_end()         { return CIEFrameInstructions.end(); }
  const_cfi_iterator  cie_end()   const { return CIEFrameInstructions.end(); }
  bool                cie_empty() const { return CIEFrameInstructions.empty(); }

  inline iterator_range<cfi_iterator> cie() {
    return iterator_range<cfi_iterator>(cie_begin(), cie_end());
  }
  inline iterator_range<const_cfi_iterator> cie() const {
    return iterator_range<const_cfi_iterator>(cie_begin(), cie_end());
  }

  /// Iterate over instructions (only if CFG is unavailable or not built yet).
  iterator_range<InstrMapType::iterator> instrs() {
    assert(!hasCFG() && "Iterate over basic blocks instead");
    return make_range(Instructions.begin(), Instructions.end());
  }
  iterator_range<InstrMapType::const_iterator> instrs() const {
    assert(!hasCFG() && "Iterate over basic blocks instead");
    return make_range(Instructions.begin(), Instructions.end());
  }

  /// Returns whether there are any labels at Offset.
  bool hasLabelAt(unsigned Offset) const { return Labels.count(Offset) != 0; }

  /// Iterate over all jump tables associated with this function.
  iterator_range<std::map<uint64_t, JumpTable *>::const_iterator>
  jumpTables() const {
    return make_range(JumpTables.begin(), JumpTables.end());
  }

  /// Return relocation associated with a given \p Offset in the function,
  /// or nullptr if no such relocation exists.
  const Relocation *getRelocationAt(uint64_t Offset) const {
    assert(CurrentState == State::Empty &&
           "Relocations unavailable in the current function state.");
    auto RI = Relocations.find(Offset);
    return (RI == Relocations.end()) ? nullptr : &RI->second;
  }

  /// Return the first relocation in the function that starts at an address in
  /// the [StartOffset, EndOffset) range. Return nullptr if no such relocation
  /// exists.
  const Relocation *getRelocationInRange(uint64_t StartOffset,
                                         uint64_t EndOffset) const {
    assert(CurrentState == State::Empty &&
           "Relocations unavailable in the current function state.");
    auto RI = Relocations.lower_bound(StartOffset);
    if (RI != Relocations.end() && RI->first < EndOffset)
      return &RI->second;

    return nullptr;
  }

  /// Return true if function is referenced in a non-control flow instruction.
  /// This flag is set when the code and relocation analyses are being
  /// performed, which occurs when safe ICF (Identical Code Folding) is enabled.
  bool hasAddressTaken() const { return HasAddressTaken; }

  /// Set whether function is referenced in a non-control flow instruction.
  void setHasAddressTaken(bool AddressTaken) { HasAddressTaken = AddressTaken; }

  /// Returns the raw binary encoding of this function.
  ErrorOr<ArrayRef<uint8_t>> getData() const;

  BinaryFunction &updateState(BinaryFunction::State State) {
    CurrentState = State;
    return *this;
  }

  FunctionLayout &getLayout() { return Layout; }

  const FunctionLayout &getLayout() const { return Layout; }

  /// Recompute landing pad information for the function and all its blocks.
  void recomputeLandingPads();

  /// Return a list of basic blocks sorted using DFS and update layout indices
  /// using the same order. Does not modify the current layout.
  BasicBlockListType dfs() const;

  /// Find the loops in the CFG of the function and store information about
  /// them.
  void calculateLoopInfo();

  /// Returns if BinaryDominatorTree has been constructed for this function.
  bool hasDomTree() const { return BDT != nullptr; }

  BinaryDominatorTree &getDomTree() { return *BDT; }

  /// Constructs DomTree for this function.
  void constructDomTree();

  /// Returns if loop detection has been run for this function.
  bool hasLoopInfo() const { return BLI != nullptr; }

  const BinaryLoopInfo &getLoopInfo() { return *BLI; }

  bool isLoopFree() {
    if (!hasLoopInfo())
      calculateLoopInfo();
    return BLI->empty();
  }

  /// Print loop information about the function.
  void printLoopInfo(raw_ostream &OS) const;

  /// View CFG in graphviz program
  void viewGraph() const;

  /// Dump CFG in graphviz format
  void dumpGraph(raw_ostream &OS) const;

  /// Dump CFG in graphviz format to file.
  void dumpGraphToFile(std::string Filename) const;

  /// Dump CFG in graphviz format to a file with a filename that is derived
  /// from the function name and Annotation strings.  Useful for dumping the
  /// CFG after an optimization pass.
  void dumpGraphForPass(std::string Annotation = "") const;

  /// Return BinaryContext for the function.
  const BinaryContext &getBinaryContext() const { return BC; }

  /// Return BinaryContext for the function.
  BinaryContext &getBinaryContext() { return BC; }

  /// Attempt to validate CFG invariants.
  bool validateCFG() const;

  BinaryBasicBlock *getBasicBlockForLabel(const MCSymbol *Label) {
    return LabelToBB.lookup(Label);
  }

  const BinaryBasicBlock *getBasicBlockForLabel(const MCSymbol *Label) const {
    return LabelToBB.lookup(Label);
  }

  /// Return basic block that originally contained offset \p Offset
  /// from the function start.
  BinaryBasicBlock *getBasicBlockContainingOffset(uint64_t Offset);

  const BinaryBasicBlock *getBasicBlockContainingOffset(uint64_t Offset) const {
    return const_cast<BinaryFunction *>(this)->getBasicBlockContainingOffset(
        Offset);
  }

  /// Return basic block that started at offset \p Offset.
  BinaryBasicBlock *getBasicBlockAtOffset(uint64_t Offset) {
    BinaryBasicBlock *BB = getBasicBlockContainingOffset(Offset);
    return BB && BB->getOffset() == Offset ? BB : nullptr;
  }

  const BinaryBasicBlock *getBasicBlockAtOffset(uint64_t Offset) const {
    return const_cast<BinaryFunction *>(this)->getBasicBlockAtOffset(Offset);
  }

  /// Retrieve the landing pad BB associated with invoke instruction \p Invoke
  /// that is in \p BB. Return nullptr if none exists
  BinaryBasicBlock *getLandingPadBBFor(const BinaryBasicBlock &BB,
                                       const MCInst &InvokeInst) const {
    assert(BC.MIB->isInvoke(InvokeInst) && "must be invoke instruction");
    const std::optional<MCPlus::MCLandingPad> LP =
        BC.MIB->getEHInfo(InvokeInst);
    if (LP && LP->first) {
      BinaryBasicBlock *LBB = BB.getLandingPad(LP->first);
      assert(LBB && "Landing pad should be defined");
      return LBB;
    }
    return nullptr;
  }

  /// Return instruction at a given offset in the function. Valid before
  /// CFG is constructed or while instruction offsets are available in CFG.
  MCInst *getInstructionAtOffset(uint64_t Offset);

  const MCInst *getInstructionAtOffset(uint64_t Offset) const {
    return const_cast<BinaryFunction *>(this)->getInstructionAtOffset(Offset);
  }

  /// When the function is in disassembled state, return an instruction that
  /// contains the \p Offset.
  MCInst *getInstructionContainingOffset(uint64_t Offset);

  std::optional<MCInst> disassembleInstructionAtOffset(uint64_t Offset) const;

  /// Return offset for the first instruction. If there is data at the
  /// beginning of a function then offset of the first instruction could
  /// be different from 0
  uint64_t getFirstInstructionOffset() const {
    if (Instructions.empty())
      return 0;
    return Instructions.begin()->first;
  }

  /// Return jump table that covers a given \p Address in memory.
  JumpTable *getJumpTableContainingAddress(uint64_t Address) {
    auto JTI = JumpTables.upper_bound(Address);
    if (JTI == JumpTables.begin())
      return nullptr;
    --JTI;
    if (JTI->first + JTI->second->getSize() > Address)
      return JTI->second;
    if (JTI->second->getSize() == 0 && JTI->first == Address)
      return JTI->second;
    return nullptr;
  }

  const JumpTable *getJumpTableContainingAddress(uint64_t Address) const {
    return const_cast<BinaryFunction *>(this)->getJumpTableContainingAddress(
        Address);
  }

  /// Return the name of the function if the function has just one name.
  /// If the function has multiple names - return one followed
  /// by "(*#<numnames>)".
  ///
  /// We should use getPrintName() for diagnostics and use
  /// hasName() to match function name against a given string.
  ///
  /// NOTE: for disambiguating names of local symbols we use the following
  ///       naming schemes:
  ///           primary:     <function>/<id>
  ///           alternative: <function>/<file>/<id2>
  std::string getPrintName() const {
    const size_t NumNames = Symbols.size() + Aliases.size();
    return NumNames == 1
               ? getOneName().str()
               : (getOneName().str() + "(*" + std::to_string(NumNames) + ")");
  }

  /// The function may have many names. For that reason, we avoid having
  /// getName() method as most of the time the user needs a different
  /// interface, such as forEachName(), hasName(), hasNameRegex(), etc.
  /// In some cases though, we need just a name uniquely identifying
  /// the function, and that's what this method is for.
  StringRef getOneName() const { return Symbols[0]->getName(); }

  /// Return the name of the function as getPrintName(), but also trying
  /// to demangle it.
  std::string getDemangledName() const;

  /// Call \p Callback for every name of this function as long as the Callback
  /// returns false. Stop if Callback returns true or all names have been used.
  /// Return the name for which the Callback returned true if any.
  template <typename FType>
  std::optional<StringRef> forEachName(FType Callback) const {
    for (MCSymbol *Symbol : Symbols)
      if (Callback(Symbol->getName()))
        return Symbol->getName();

    for (const std::string &Name : Aliases)
      if (Callback(StringRef(Name)))
        return StringRef(Name);

    return std::nullopt;
  }

  /// Check if (possibly one out of many) function name matches the given
  /// string. Use this member function instead of direct name comparison.
  bool hasName(const std::string &FunctionName) const {
    auto Res =
        forEachName([&](StringRef Name) { return Name == FunctionName; });
    return Res.has_value();
  }

  /// Check if any of function names matches the given regex.
  std::optional<StringRef> hasNameRegex(const StringRef NameRegex) const;

  /// Check if any of restored function names matches the given regex.
  /// Restored name means stripping BOLT-added suffixes like "/1",
  std::optional<StringRef>
  hasRestoredNameRegex(const StringRef NameRegex) const;

  /// Return a vector of all possible names for the function.
  std::vector<StringRef> getNames() const {
    std::vector<StringRef> AllNames;
    forEachName([&AllNames](StringRef Name) {
      AllNames.push_back(Name);
      return false;
    });

    return AllNames;
  }

  /// Return a state the function is in (see BinaryFunction::State definition
  /// for description).
  State getState() const { return CurrentState; }

  /// Return true if function has a control flow graph available.
  bool hasCFG() const {
    return getState() == State::CFG || getState() == State::CFG_Finalized ||
           getState() == State::EmittedCFG;
  }

  /// Return true if the function state implies that it includes instructions.
  bool hasInstructions() const {
    return getState() == State::Disassembled || hasCFG();
  }

  bool isEmitted() const {
    return getState() == State::EmittedCFG || getState() == State::Emitted;
  }

  /// Return the section in the input binary this function originated from or
  /// nullptr if the function did not originate from the file.
  BinarySection *getOriginSection() const { return OriginSection; }

  void setOriginSection(BinarySection *Section) { OriginSection = Section; }

  /// Return true if the function did not originate from the primary input file.
  bool isInjected() const { return IsInjected; }

  /// Return original address of the function (or offset from base for PIC).
  uint64_t getAddress() const { return Address; }

  uint64_t getOutputAddress() const { return OutputAddress; }

  uint64_t getOutputSize() const { return OutputSize; }

  /// Does this function have a valid streaming order index?
  bool hasValidIndex() const { return Index != -1U; }

  /// Get the streaming order index for this function.
  uint32_t getIndex() const { return Index; }

  /// Set the streaming order index for this function.
  void setIndex(uint32_t Idx) {
    assert(!hasValidIndex());
    Index = Idx;
  }

  /// Return offset of the function body in the binary file.
  uint64_t getFileOffset() const {
    return getLayout().getMainFragment().getFileOffset();
  }

  /// Return (original) byte size of the function.
  uint64_t getSize() const { return Size; }

  /// Return the maximum size the body of the function could have.
  uint64_t getMaxSize() const { return MaxSize; }

  /// Return the number of emitted instructions for this function.
  uint32_t getNumNonPseudos() const {
    uint32_t N = 0;
    for (const BinaryBasicBlock &BB : blocks())
      N += BB.getNumNonPseudos();
    return N;
  }

  /// Return true if function has instructions to emit.
  bool hasNonPseudoInstructions() const {
    for (const BinaryBasicBlock &BB : blocks())
      if (BB.getNumNonPseudos() > 0)
        return true;
    return false;
  }

  /// Return MC symbol associated with the function.
  /// All references to the function should use this symbol.
  MCSymbol *getSymbol(const FragmentNum Fragment = FragmentNum::main()) {
    if (Fragment == FragmentNum::main())
      return Symbols[0];

    size_t ColdSymbolIndex = Fragment.get() - 1;
    if (ColdSymbolIndex >= ColdSymbols.size())
      ColdSymbols.resize(ColdSymbolIndex + 1);

    MCSymbol *&ColdSymbol = ColdSymbols[ColdSymbolIndex];
    if (ColdSymbol == nullptr) {
      SmallString<10> Appendix = formatv(".cold.{0}", ColdSymbolIndex);
      ColdSymbol = BC.Ctx->getOrCreateSymbol(
          NameResolver::append(Symbols[0]->getName(), Appendix));
    }

    return ColdSymbol;
  }

  /// Return MC symbol associated with the function (const version).
  /// All references to the function should use this symbol.
  const MCSymbol *getSymbol() const { return Symbols[0]; }

  /// Return a list of symbols associated with the main entry of the function.
  SymbolListTy &getSymbols() { return Symbols; }
  const SymbolListTy &getSymbols() const { return Symbols; }

  /// If a local symbol \p BBLabel corresponds to a basic block that is a
  /// secondary entry point into the function, then return a global symbol
  /// that represents the secondary entry point. Otherwise return nullptr.
  MCSymbol *getSecondaryEntryPointSymbol(const MCSymbol *BBLabel) const {
    return SecondaryEntryPoints.lookup(BBLabel);
  }

  /// If the basic block serves as a secondary entry point to the function,
  /// return a global symbol representing the entry. Otherwise return nullptr.
  MCSymbol *getSecondaryEntryPointSymbol(const BinaryBasicBlock &BB) const {
    return getSecondaryEntryPointSymbol(BB.getLabel());
  }

  /// Return true if the basic block is an entry point into the function
  /// (either primary or secondary).
  bool isEntryPoint(const BinaryBasicBlock &BB) const {
    if (&BB == BasicBlocks.front())
      return true;
    return getSecondaryEntryPointSymbol(BB);
  }

  /// Return MC symbol corresponding to an enumerated entry for multiple-entry
  /// functions.
  MCSymbol *getSymbolForEntryID(uint64_t EntryNum);
  const MCSymbol *getSymbolForEntryID(uint64_t EntryNum) const {
    return const_cast<BinaryFunction *>(this)->getSymbolForEntryID(EntryNum);
  }

  using EntryPointCallbackTy = function_ref<bool(uint64_t, const MCSymbol *)>;

  /// Invoke \p Callback function for every entry point in the function starting
  /// with the main entry and using entries in the ascending address order.
  /// Stop calling the function after false is returned by the callback.
  ///
  /// Pass an offset of the entry point in the input binary and a corresponding
  /// global symbol to the callback function.
  ///
  /// Return true if all callbacks returned true, false otherwise.
  bool forEachEntryPoint(EntryPointCallbackTy Callback) const;

  /// Return MC symbol associated with the end of the function.
  MCSymbol *
  getFunctionEndLabel(const FragmentNum Fragment = FragmentNum::main()) const {
    assert(BC.Ctx && "cannot be called with empty context");

    size_t LabelIndex = Fragment.get();
    if (LabelIndex >= FunctionEndLabels.size()) {
      FunctionEndLabels.resize(LabelIndex + 1);
    }

    MCSymbol *&FunctionEndLabel = FunctionEndLabels[LabelIndex];
    if (!FunctionEndLabel) {
      std::unique_lock<llvm::sys::RWMutex> Lock(BC.CtxMutex);
      if (Fragment == FragmentNum::main())
        FunctionEndLabel = BC.Ctx->createNamedTempSymbol("func_end");
      else
        FunctionEndLabel = BC.Ctx->createNamedTempSymbol(
            formatv("func_cold_end.{0}", Fragment.get() - 1));
    }
    return FunctionEndLabel;
  }

  /// Return a label used to identify where the constant island was emitted
  /// (AArch only). This is used to update the symbol table accordingly,
  /// emitting data marker symbols as required by the ABI.
  MCSymbol *getFunctionConstantIslandLabel() const {
    assert(Islands && "function expected to have constant islands");

    if (!Islands->FunctionConstantIslandLabel) {
      Islands->FunctionConstantIslandLabel =
          BC.Ctx->getOrCreateSymbol("func_const_island@" + getOneName());
    }
    return Islands->FunctionConstantIslandLabel;
  }

  MCSymbol *getFunctionColdConstantIslandLabel() const {
    assert(Islands && "function expected to have constant islands");

    if (!Islands->FunctionColdConstantIslandLabel) {
      Islands->FunctionColdConstantIslandLabel =
          BC.Ctx->getOrCreateSymbol("func_cold_const_island@" + getOneName());
    }
    return Islands->FunctionColdConstantIslandLabel;
  }

  /// Return true if this is a function representing a PLT entry.
  bool isPLTFunction() const { return PLTSymbol != nullptr; }

  /// Return PLT function reference symbol for PLT functions and nullptr for
  /// non-PLT functions.
  const MCSymbol *getPLTSymbol() const { return PLTSymbol; }

  /// Set function PLT reference symbol for PLT functions.
  void setPLTSymbol(const MCSymbol *Symbol) {
    assert(Size == 0 && "function size should be 0 for PLT functions");
    PLTSymbol = Symbol;
    IsPseudo = true;
  }

  /// Update output values of the function based on the final \p Layout.
  void updateOutputValues(const BOLTLinker &Linker);

  /// Register relocation type \p RelType at a given \p Address in the function
  /// against \p Symbol.
  /// Assert if the \p Address is not inside this function.
  void addRelocation(uint64_t Address, MCSymbol *Symbol, uint32_t RelType,
                     uint64_t Addend, uint64_t Value);

  /// Return the name of the section this function originated from.
  std::optional<StringRef> getOriginSectionName() const {
    if (!OriginSection)
      return std::nullopt;
    return OriginSection->getName();
  }

  /// Return internal section name for this function.
  SmallString<32>
  getCodeSectionName(const FragmentNum Fragment = FragmentNum::main()) const {
    if (Fragment == FragmentNum::main())
      return SmallString<32>(CodeSectionName);
    if (Fragment == FragmentNum::cold())
      return SmallString<32>(ColdCodeSectionName);
    if (BC.HasWarmSection && Fragment == FragmentNum::warm())
      return SmallString<32>(BC.getWarmCodeSectionName());
    return formatv("{0}.{1}", ColdCodeSectionName, Fragment.get() - 1);
  }

  /// Assign a code section name to the function.
  void setCodeSectionName(const StringRef Name) {
    CodeSectionName = Name.str();
  }

  /// Get output code section.
  ErrorOr<BinarySection &>
  getCodeSection(const FragmentNum Fragment = FragmentNum::main()) const {
    return BC.getUniqueSectionByName(getCodeSectionName(Fragment));
  }

  /// Assign a section name for the cold part of the function.
  void setColdCodeSectionName(const StringRef Name) {
    ColdCodeSectionName = Name.str();
  }

  /// Return true if the function will halt execution on entry.
  bool trapsOnEntry() const { return TrapsOnEntry; }

  /// Make the function always trap on entry. Other than the trap instruction,
  /// the function body will be empty.
  void setTrapOnEntry();

  /// Return true if the function could be correctly processed.
  bool isSimple() const { return IsSimple; }

  /// Return true if the function should be ignored for optimization purposes.
  bool isIgnored() const { return IsIgnored; }

  /// Return true if the function should not be disassembled, emitted, or
  /// otherwise processed.
  bool isPseudo() const { return IsPseudo; }

  /// Return true if the function contains explicit or implicit indirect branch
  /// to its split fragments, e.g., split jump table, landing pad in split
  /// fragment.
  bool hasIndirectTargetToSplitFragment() const {
    return HasIndirectTargetToSplitFragment;
  }

  /// Return true if all CFG edges have local successors.
  bool hasCanonicalCFG() const { return HasCanonicalCFG; }

  /// Return true if the original function code has all necessary relocations
  /// to track addresses of functions emitted to new locations.
  bool hasExternalRefRelocations() const { return HasExternalRefRelocations; }

  /// Return true if the function has instruction(s) with unknown control flow.
  bool hasUnknownControlFlow() const { return HasUnknownControlFlow; }

  /// Return true if the function body is non-contiguous.
  bool isSplit() const { return isSimple() && getLayout().isSplit(); }

  bool shouldPreserveNops() const { return PreserveNops; }

  /// Return true if the function has exception handling tables.
  bool hasEHRanges() const { return HasEHRanges; }

  /// Return true if the function uses DW_CFA_GNU_args_size CFIs.
  bool usesGnuArgsSize() const { return UsesGnuArgsSize; }

  /// Return true if the function has more than one entry point.
  bool isMultiEntry() const { return !SecondaryEntryPoints.empty(); }

  /// Return true if the function might have a profile available externally,
  /// but not yet populated into the function.
  bool hasProfileAvailable() const { return HasProfileAvailable; }

  bool hasMemoryProfile() const { return HasMemoryProfile; }

  /// Return true if the body of the function was merged into another function.
  bool isFolded() const { return FoldedIntoFunction != nullptr; }

  /// Return true if other functions were folded into this one.
  bool hasFunctionsFoldedInto() const { return HasFunctionsFoldedInto; }

  /// Return true if this function is used for patching existing code.
  bool isPatch() const { return IsPatch; }

  /// Return true if the function requires a patch.
  bool needsPatch() const { return NeedsPatch; }

  /// Return true if the function should not have associated symbol table entry.
  bool isAnonymous() const { return IsAnonymous; }

  /// If this function was folded, return the function it was folded into.
  BinaryFunction *getFoldedIntoFunction() const { return FoldedIntoFunction; }

  /// Return true if the function uses jump tables.
  bool hasJumpTables() const { return !JumpTables.empty(); }

  /// Return true if the function has SDT marker
  bool hasSDTMarker() const { return HasSDTMarker; }

  /// Return true if the function has Pseudo Probe
  bool hasPseudoProbe() const { return HasPseudoProbe; }

  /// Return true if the function uses ORC format for stack unwinding.
  bool hasORC() const { return HasORC; }

  const JumpTable *getJumpTable(const MCInst &Inst) const {
    const uint64_t Address = BC.MIB->getJumpTable(Inst);
    return getJumpTableContainingAddress(Address);
  }

  JumpTable *getJumpTable(const MCInst &Inst) {
    const uint64_t Address = BC.MIB->getJumpTable(Inst);
    return getJumpTableContainingAddress(Address);
  }

  const MCSymbol *getPersonalityFunction() const { return PersonalityFunction; }

  uint8_t getPersonalityEncoding() const { return PersonalityEncoding; }

  CallSitesRange getCallSites(const FragmentNum F) const {
    return make_range(std::equal_range(CallSites.begin(), CallSites.end(),
                                       std::make_pair(F, CallSite()),
                                       llvm::less_first()));
  }

  void
  addCallSites(const ArrayRef<std::pair<FragmentNum, CallSite>> NewCallSites) {
    llvm::copy(NewCallSites, std::back_inserter(CallSites));
    llvm::stable_sort(CallSites, llvm::less_first());
  }

  ArrayRef<uint8_t> getLSDAActionTable() const { return LSDAActionTable; }

  const LSDATypeTableTy &getLSDATypeTable() const { return LSDATypeTable; }

  unsigned getLSDATypeEncoding() const { return LSDATypeEncoding; }

  const LSDATypeTableTy &getLSDATypeAddressTable() const {
    return LSDATypeAddressTable;
  }

  ArrayRef<uint8_t> getLSDATypeIndexTable() const { return LSDATypeIndexTable; }

  IslandInfo &getIslandInfo() {
    assert(Islands && "function expected to have constant islands");
    return *Islands;
  }

  const IslandInfo &getIslandInfo() const {
    assert(Islands && "function expected to have constant islands");
    return *Islands;
  }

  /// Return true if the function has CFI instructions
  bool hasCFI() const {
    return !FrameInstructions.empty() || !CIEFrameInstructions.empty() ||
           IsInjected;
  }

  /// Return unique number associated with the function.
  uint64_t getFunctionNumber() const { return FunctionNumber; }

  /// Return true if the given address \p PC is inside the function body.
  bool containsAddress(uint64_t PC, bool UseMaxSize = false) const {
    if (UseMaxSize)
      return Address <= PC && PC < Address + MaxSize;
    return Address <= PC && PC < Address + Size;
  }

  /// Create a basic block in the function. The new block is *NOT* inserted
  /// into the CFG. The caller must use insertBasicBlocks() to add any new
  /// blocks to the CFG.
  std::unique_ptr<BinaryBasicBlock>
  createBasicBlock(MCSymbol *Label = nullptr) {
    if (!Label) {
      std::unique_lock<llvm::sys::RWMutex> Lock(BC.CtxMutex);
      Label = BC.Ctx->createNamedTempSymbol("BB");
    }
    auto BB =
        std::unique_ptr<BinaryBasicBlock>(new BinaryBasicBlock(this, Label));

    LabelToBB[Label] = BB.get();

    return BB;
  }

  /// Create a new basic block with an optional \p Label and add it to the list
  /// of basic blocks of this function.
  BinaryBasicBlock *addBasicBlock(MCSymbol *Label = nullptr) {
    assert(CurrentState == State::CFG && "Can only add blocks in CFG state");

    BasicBlocks.emplace_back(createBasicBlock(Label).release());
    BinaryBasicBlock *BB = BasicBlocks.back();

    BB->setIndex(BasicBlocks.size() - 1);
    Layout.addBasicBlock(BB);

    return BB;
  }

  /// Add basic block \BB as an entry point to the function. Return global
  /// symbol associated with the entry.
  MCSymbol *addEntryPoint(const BinaryBasicBlock &BB);

  /// Register secondary entry point at a given \p Offset into the function.
  /// Return global symbol for use by extern function references.
  MCSymbol *addEntryPointAtOffset(uint64_t Offset);

  /// Mark all blocks that are unreachable from a root (entry point
  /// or landing pad) as invalid.
  void markUnreachableBlocks();

  /// Rebuilds BBs layout, ignoring dead BBs. Returns the number of removed
  /// BBs and the removed number of bytes of code.
  std::pair<unsigned, uint64_t>
  eraseInvalidBBs(const MCCodeEmitter *Emitter = nullptr);

  /// Get the relative order between two basic blocks in the original
  /// layout.  The result is > 0 if B occurs before A and < 0 if B
  /// occurs after A.  If A and B are the same block, the result is 0.
  signed getOriginalLayoutRelativeOrder(const BinaryBasicBlock *A,
                                        const BinaryBasicBlock *B) const {
    return getIndex(A) - getIndex(B);
  }

  /// Insert the BBs contained in NewBBs into the basic blocks for this
  /// function. Update the associated state of all blocks as needed, i.e.
  /// BB offsets and BB indices. The new BBs are inserted after Start.
  /// This operation could affect fallthrough branches for Start.
  ///
  void
  insertBasicBlocks(BinaryBasicBlock *Start,
                    std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
                    const bool UpdateLayout = true,
                    const bool UpdateCFIState = true,
                    const bool RecomputeLandingPads = true);

  iterator insertBasicBlocks(
      iterator StartBB, std::vector<std::unique_ptr<BinaryBasicBlock>> &&NewBBs,
      const bool UpdateLayout = true, const bool UpdateCFIState = true,
      const bool RecomputeLandingPads = true);

  /// Update the basic block layout for this function.  The BBs from
  /// [Start->Index, Start->Index + NumNewBlocks) are inserted into the
  /// layout after the BB indicated by Start.
  void updateLayout(BinaryBasicBlock *Start, const unsigned NumNewBlocks);

  /// Recompute the CFI state for NumNewBlocks following Start after inserting
  /// new blocks into the CFG.  This must be called after updateLayout.
  void updateCFIState(BinaryBasicBlock *Start, const unsigned NumNewBlocks);

  /// Return true if we detected ambiguous jump tables in this function, which
  /// happen when one JT is used in more than one indirect jumps. This precludes
  /// us from splitting edges for this JT unless we duplicate the JT (see
  /// disambiguateJumpTables).
  bool checkForAmbiguousJumpTables();

  /// Detect when two distinct indirect jumps are using the same jump table and
  /// duplicate it, allocating a separate JT for each indirect branch. This is
  /// necessary for code transformations on the CFG that change an edge induced
  /// by an indirect branch, e.g.: instrumentation or shrink wrapping. However,
  /// this is only possible if we are not updating jump tables in place, but are
  /// writing it to a new location (moving them).
  void disambiguateJumpTables(MCPlusBuilder::AllocatorIdTy AllocId);

  /// Change \p OrigDest to \p NewDest in the jump table used at the end of
  /// \p BB. Returns false if \p OrigDest couldn't be find as a valid target
  /// and no replacement took place.
  bool replaceJumpTableEntryIn(BinaryBasicBlock *BB, BinaryBasicBlock *OldDest,
                               BinaryBasicBlock *NewDest);

  /// Split the CFG edge <From, To> by inserting an intermediate basic block.
  /// Returns a pointer to this new intermediate basic block. BB "From" will be
  /// updated to jump to the intermediate block, which in turn will have an
  /// unconditional branch to BB "To".
  /// User needs to manually call fixBranches(). This function only creates the
  /// correct CFG edges.
  BinaryBasicBlock *splitEdge(BinaryBasicBlock *From, BinaryBasicBlock *To);

  /// We may have built an overly conservative CFG for functions with calls
  /// to functions that the compiler knows will never return. In this case,
  /// clear all successors from these blocks.
  void deleteConservativeEdges();

  /// Determine direction of the branch based on the current layout.
  /// Callee is responsible of updating basic block indices prior to using
  /// this function (e.g. by calling BinaryFunction::updateLayoutIndices()).
  static bool isForwardBranch(const BinaryBasicBlock *From,
                              const BinaryBasicBlock *To) {
    assert(From->getFunction() == To->getFunction() &&
           "basic blocks should be in the same function");
    return To->getLayoutIndex() > From->getLayoutIndex();
  }

  /// Determine direction of the call to callee symbol relative to the start
  /// of this function.
  /// Note: this doesn't take function splitting into account.
  bool isForwardCall(const MCSymbol *CalleeSymbol) const;

  /// Dump function information to debug output. If \p PrintInstructions
  /// is true - include instruction disassembly.
  void dump() const;

  /// Print function information to the \p OS stream.
  void print(raw_ostream &OS, std::string Annotation = "");

  /// Print all relocations between \p Offset and \p Offset + \p Size in
  /// this function.
  void printRelocations(raw_ostream &OS, uint64_t Offset, uint64_t Size) const;

  /// Return true if function has a profile, even if the profile does not
  /// match CFG 100%.
  bool hasProfile() const { return ExecutionCount != COUNT_NO_PROFILE; }

  /// Return true if function profile is present and accurate.
  bool hasValidProfile() const {
    return ExecutionCount != COUNT_NO_PROFILE && ProfileMatchRatio == 1.0f;
  }

  /// Mark this function as having a valid profile.
  void markProfiled(uint16_t Flags) {
    if (ExecutionCount == COUNT_NO_PROFILE)
      ExecutionCount = 0;
    ProfileFlags = Flags;
    ProfileMatchRatio = 1.0f;
  }

  /// Return flags describing a profile for this function.
  uint16_t getProfileFlags() const { return ProfileFlags; }

  /// Return true if the function's input profile data has been inaccurate but
  /// has been corrected by the profile inference algorithm.
  bool hasInferredProfile() const { return HasInferredProfile; }

  void setHasInferredProfile(bool Inferred) { HasInferredProfile = Inferred; }

  /// Find corrected offset the same way addCFIInstruction does it to skip NOPs.
  std::optional<uint64_t> getCorrectedCFIOffset(uint64_t Offset) {
    assert(!Instructions.empty());
    auto I = Instructions.lower_bound(Offset);
    if (Offset == getSize()) {
      assert(I == Instructions.end() && "unexpected iterator value");
      // Sometimes compiler issues restore_state after all instructions
      // in the function (even after nop).
      --I;
      Offset = I->first;
    }
    assert(I->first == Offset && "CFI pointing to unknown instruction");
    if (I == Instructions.begin())
      return {};

    --I;
    while (I != Instructions.begin() && BC.MIB->isNoop(I->second)) {
      Offset = I->first;
      --I;
    }
    return Offset;
  }

  void setInstModifiesRAState(uint8_t CFIOpcode, uint64_t Offset) {
    std::optional<uint64_t> CorrectedOffset = getCorrectedCFIOffset(Offset);
    if (CorrectedOffset) {
      auto I = Instructions.lower_bound(*CorrectedOffset);
      I--;

      switch (CFIOpcode) {
      case dwarf::DW_CFA_AARCH64_negate_ra_state:
        BC.MIB->setNegateRAState(I->second);
        break;
      case dwarf::DW_CFA_remember_state:
        BC.MIB->setRememberState(I->second);
        break;
      case dwarf::DW_CFA_restore_state:
        BC.MIB->setRestoreState(I->second);
        break;
      default:
        assert(0 && "CFI Opcode not covered by function");
      }
    }
  }

  void addCFIInstruction(uint64_t Offset, MCCFIInstruction &&Inst) {
    assert(!Instructions.empty());

    // Fix CFI instructions skipping NOPs. We need to fix this because changing
    // CFI state after a NOP, besides being wrong and inaccurate,  makes it
    // harder for us to recover this information, since we can create empty BBs
    // with NOPs and then reorder it away.
    // We fix this by moving the CFI instruction just before any NOPs.
    auto I = Instructions.lower_bound(Offset);
    if (Offset == getSize()) {
      assert(I == Instructions.end() && "unexpected iterator value");
      // Sometimes compiler issues restore_state after all instructions
      // in the function (even after nop).
      --I;
      Offset = I->first;
    }
    assert(I->first == Offset && "CFI pointing to unknown instruction");
    // When dealing with RememberState, we place this CFI in FrameInstructions.
    // We want to ensure RememberState and RestoreState CFIs are in the same
    // list in order to properly populate the StateStack.
    if (I == Instructions.begin() &&
        Inst.getOperation() != MCCFIInstruction::OpRememberState) {
      CIEFrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
      return;
    }

    --I;
    while (I != Instructions.begin() && BC.MIB->isNoop(I->second)) {
      Offset = I->first;
      --I;
    }
    OffsetToCFI.emplace(Offset, FrameInstructions.size());
    FrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
  }

  BinaryBasicBlock::iterator addCFIInstruction(BinaryBasicBlock *BB,
                                               BinaryBasicBlock::iterator Pos,
                                               MCCFIInstruction &&Inst) {
    size_t Idx = FrameInstructions.size();
    FrameInstructions.emplace_back(std::forward<MCCFIInstruction>(Inst));
    return addCFIPseudo(BB, Pos, Idx);
  }

  /// Insert a CFI pseudo instruction in a basic block. This pseudo instruction
  /// is a placeholder that refers to a real MCCFIInstruction object kept by
  /// this function that will be emitted at that position.
  BinaryBasicBlock::iterator addCFIPseudo(BinaryBasicBlock *BB,
                                          BinaryBasicBlock::iterator Pos,
                                          uint32_t Offset) {
    MCInst CFIPseudo;
    BC.MIB->createCFI(CFIPseudo, Offset);
    return BB->insertPseudoInstr(Pos, CFIPseudo);
  }

  /// Retrieve the MCCFIInstruction object associated with a CFI pseudo.
  const MCCFIInstruction *getCFIFor(const MCInst &Instr) const {
    if (!BC.MIB->isCFI(Instr))
      return nullptr;
    uint32_t Offset = Instr.getOperand(0).getImm();
    assert(Offset < FrameInstructions.size() && "Invalid CFI offset");
    return &FrameInstructions[Offset];
  }

  void setCFIFor(const MCInst &Instr, MCCFIInstruction &&CFIInst) {
    assert(BC.MIB->isCFI(Instr) &&
           "attempting to change CFI in a non-CFI inst");
    uint32_t Offset = Instr.getOperand(0).getImm();
    assert(Offset < FrameInstructions.size() && "Invalid CFI offset");
    FrameInstructions[Offset] = std::move(CFIInst);
  }

  void mutateCFIRegisterFor(const MCInst &Instr, MCPhysReg NewReg);

  const MCCFIInstruction *mutateCFIOffsetFor(const MCInst &Instr,
                                             int64_t NewOffset);

  BinaryFunction &setFileOffset(uint64_t Offset) {
    getLayout().getMainFragment().setFileOffset(Offset);
    return *this;
  }

  BinaryFunction &setSize(uint64_t S) {
    Size = S;
    return *this;
  }

  BinaryFunction &setMaxSize(uint64_t Size) {
    MaxSize = Size;
    return *this;
  }

  BinaryFunction &setOutputAddress(uint64_t Address) {
    OutputAddress = Address;
    return *this;
  }

  BinaryFunction &setOutputSize(uint64_t Size) {
    OutputSize = Size;
    return *this;
  }

  BinaryFunction &setSimple(bool Simple) {
    IsSimple = Simple;
    return *this;
  }

  void setPseudo(bool Pseudo) { IsPseudo = Pseudo; }

  void setPreserveNops(bool Value) { PreserveNops = Value; }

  BinaryFunction &setUsesGnuArgsSize(bool Uses = true) {
    UsesGnuArgsSize = Uses;
    return *this;
  }

  BinaryFunction &setHasProfileAvailable(bool V = true) {
    HasProfileAvailable = V;
    return *this;
  }

  /// Mark function that should not be emitted.
  void setIgnored();

  void setHasIndirectTargetToSplitFragment(bool V) {
    HasIndirectTargetToSplitFragment = V;
  }

  void setHasCanonicalCFG(bool V) { HasCanonicalCFG = V; }

  void setFolded(BinaryFunction *BF) { FoldedIntoFunction = BF; }

  /// Indicate that another function body was merged with this function.
  void setHasFunctionsFoldedInto() { HasFunctionsFoldedInto = true; }

  /// Indicate that this function is a patch.
  void setIsPatch(bool V) {
    assert(isInjected() && "Only injected functions can be used as patches");
    IsPatch = V;
  }

  /// Mark the function for patching.
  void setNeedsPatch(bool V) { NeedsPatch = V; }

  /// Indicate if the function should have a name in the symbol table.
  void setAnonymous(bool V) {
    assert(isInjected() && "Only injected functions could be anonymous");
    IsAnonymous = V;
  }

  void setHasSDTMarker(bool V) { HasSDTMarker = V; }

  /// Mark the function as using ORC format for stack unwinding.
  void setHasORC(bool V) { HasORC = V; }

  BinaryFunction &setPersonalityFunction(uint64_t Addr) {
    assert(!PersonalityFunction && "can't set personality function twice");
    PersonalityFunction = BC.getOrCreateGlobalSymbol(Addr, "FUNCat");
    return *this;
  }

  BinaryFunction &setPersonalityEncoding(uint8_t Encoding) {
    PersonalityEncoding = Encoding;
    return *this;
  }

  BinaryFunction &setAlignment(uint16_t Align) {
    Alignment = Align;
    return *this;
  }

  uint16_t getMinAlignment() const {
    // Align data in code BFs minimum to CI alignment
    if (!size() && hasIslandsInfo())
      return getConstantIslandAlignment();
    return BC.MIB->getMinFunctionAlignment();
  }

  Align getMinAlign() const { return Align(getMinAlignment()); }

  uint16_t getAlignment() const { return Alignment; }
  Align getAlign() const { return Align(getAlignment()); }

  BinaryFunction &setMaxAlignmentBytes(uint16_t MaxAlignBytes) {
    MaxAlignmentBytes = MaxAlignBytes;
    return *this;
  }

  uint16_t getMaxAlignmentBytes() const { return MaxAlignmentBytes; }

  BinaryFunction &setMaxColdAlignmentBytes(uint16_t MaxAlignBytes) {
    MaxColdAlignmentBytes = MaxAlignBytes;
    return *this;
  }

  uint16_t getMaxColdAlignmentBytes() const { return MaxColdAlignmentBytes; }

  BinaryFunction &setImageAddress(uint64_t Address) {
    getLayout().getMainFragment().setImageAddress(Address);
    return *this;
  }

  /// Return the address of this function' image in memory.
  uint64_t getImageAddress() const {
    return getLayout().getMainFragment().getImageAddress();
  }

  BinaryFunction &setImageSize(uint64_t Size) {
    getLayout().getMainFragment().setImageSize(Size);
    return *this;
  }

  /// Return the size of this function' image in memory.
  uint64_t getImageSize() const {
    return getLayout().getMainFragment().getImageSize();
  }

  /// Return true if the function is a secondary fragment of another function.
  bool isFragment() const { return IsFragment; }

  /// Returns if this function is a child of \p Other function.
  bool isChildOf(const BinaryFunction &Other) const {
    return ParentFragments.contains(&Other);
  }

  /// Return the child fragment form parent function
  iterator_range<FragmentsSetTy::const_iterator> getFragments() const {
    return iterator_range<FragmentsSetTy::const_iterator>(Fragments.begin(),
                                                          Fragments.end());
  }

  /// Return the parent function for split function fragments.
  FragmentsSetTy *getParentFragments() { return &ParentFragments; }

  /// Set the profile data for the number of times the function was called.
  BinaryFunction &setExecutionCount(uint64_t Count) {
    ExecutionCount = Count;
    return *this;
  }

  /// Set the profile data for the number of times the function was entered from
  /// external code (DSO/JIT).
  void setExternEntryCount(uint64_t Count) { ExternEntryCount = Count; }

  /// Adjust execution count for the function by a given \p Count. The value
  /// \p Count will be subtracted from the current function count.
  ///
  /// The function will proportionally adjust execution count for all
  /// basic blocks and edges in the control flow graph.
  void adjustExecutionCount(uint64_t Count);

  /// Set LSDA address for the function.
  BinaryFunction &setLSDAAddress(uint64_t Address) {
    LSDAAddress = Address;
    return *this;
  }

  /// Set main LSDA symbol for the function.
  BinaryFunction &setLSDASymbol(MCSymbol *Symbol) {
    if (LSDASymbols.empty())
      LSDASymbols.resize(1);
    LSDASymbols.front() = Symbol;
    return *this;
  }

  /// Return the profile information about the number of times
  /// the function was executed.
  ///
  /// Return COUNT_NO_PROFILE if there's no profile info.
  uint64_t getExecutionCount() const { return ExecutionCount; }

  /// Return the profile information about the number of times the function was
  /// entered from external code (DSO/JIT).
  uint64_t getExternEntryCount() const { return ExternEntryCount; }

  /// Return the raw profile information about the number of branch
  /// executions corresponding to this function.
  uint64_t getRawSampleCount() const { return RawSampleCount; }

  /// Set the profile data about the number of branch executions corresponding
  /// to this function.
  void setRawSampleCount(uint64_t Count) { RawSampleCount = Count; }

  /// Return the number of dynamically executed bytes, from raw perf data.
  uint64_t getSampleCountInBytes() const { return SampleCountInBytes; }

  /// Return the execution count for functions with known profile.
  /// Return 0 if the function has no profile.
  uint64_t getKnownExecutionCount() const {
    return ExecutionCount == COUNT_NO_PROFILE ? 0 : ExecutionCount;
  }

  /// Return original LSDA address for the function or NULL.
  uint64_t getLSDAAddress() const { return LSDAAddress; }

  /// Return symbol pointing to function's LSDA.
  MCSymbol *getLSDASymbol(const FragmentNum F) {
    if (F.get() < LSDASymbols.size() && LSDASymbols[F.get()] != nullptr)
      return LSDASymbols[F.get()];
    if (getCallSites(F).empty())
      return nullptr;

    if (F.get() >= LSDASymbols.size())
      LSDASymbols.resize(F.get() + 1);

    SmallString<256> SymbolName;
    if (F == FragmentNum::main())
      SymbolName = formatv("GCC_except_table{0:x-}", getFunctionNumber());
    else
      SymbolName = formatv("GCC_cold_except_table{0:x-}.{1}",
                           getFunctionNumber(), F.get());

    LSDASymbols[F.get()] = BC.Ctx->getOrCreateSymbol(SymbolName);

    return LSDASymbols[F.get()];
  }

  /// If all landing pads for the function fragment \p F are located in fragment
  /// \p LPF, designate \p LPF as a landing-pad fragment for \p F. Passing
  /// std::nullopt in LPF, means that landing pads for \p F are located in more
  /// than one fragment.
  void setLPFragment(const FragmentNum F, std::optional<FragmentNum> LPF) {
    if (F.get() >= LPFragments.size())
      LPFragments.resize(F.get() + 1);

    LPFragments[F.get()] = LPF;
  }

  /// If function fragment \p F has a designated landing pad fragment, i.e. a
  /// fragment that contains all landing pads for throwers in \p F, then return
  /// that landing pad fragment number. If \p F does not need landing pads,
  /// return \p F. Return nullptr if landing pads for \p F are scattered among
  /// several function fragments.
  std::optional<FragmentNum> getLPFragment(const FragmentNum F) {
    if (!isSplit()) {
      assert(F == FragmentNum::main() && "Invalid fragment number");
      return FragmentNum::main();
    }

    if (F.get() >= LPFragments.size())
      return std::nullopt;

    return LPFragments[F.get()];
  }

  /// Return a symbol corresponding to a landing pad fragment for fragment \p F.
  /// See getLPFragment().
  MCSymbol *getLPStartSymbol(const FragmentNum F) {
    if (std::optional<FragmentNum> LPFragment = getLPFragment(F))
      return getSymbol(*LPFragment);
    return nullptr;
  }

  void setOutputDataAddress(uint64_t Address) { OutputDataOffset = Address; }

  uint64_t getOutputDataAddress() const { return OutputDataOffset; }

  void setOutputColdDataAddress(uint64_t Address) {
    OutputColdDataOffset = Address;
  }

  uint64_t getOutputColdDataAddress() const { return OutputColdDataOffset; }

  /// If \p Address represents an access to a constant island managed by this
  /// function, return a symbol so code can safely refer to it. Otherwise,
  /// return nullptr. First return value is the symbol for reference in the
  /// hot code area while the second return value is the symbol for reference
  /// in the cold code area, as when the function is split the islands are
  /// duplicated.
  MCSymbol *getOrCreateIslandAccess(uint64_t Address) {
    if (!Islands)
      return nullptr;

    MCSymbol *Symbol;
    if (!isInConstantIsland(Address))
      return nullptr;

    // Register our island at global namespace
    Symbol = BC.getOrCreateGlobalSymbol(Address, "ISLANDat");

    // Internal bookkeeping
    const uint64_t Offset = Address - getAddress();
    assert((!Islands->Offsets.count(Offset) ||
            Islands->Offsets[Offset] == Symbol) &&
           "Inconsistent island symbol management");
    if (!Islands->Offsets.count(Offset)) {
      Islands->Offsets[Offset] = Symbol;
      Islands->Symbols.insert(Symbol);
    }
    return Symbol;
  }

  /// Support dynamic relocations in constant islands, which may happen if
  /// binary is linked with -z notext option.
  Error markIslandDynamicRelocationAtAddress(uint64_t Address) {
    if (!isInConstantIsland(Address))
      return createFatalBOLTError(
          Twine("dynamic relocation found for text section at 0x") +
          Twine::utohexstr(Address) + Twine("\n"));

    // Mark island to have dynamic relocation
    Islands->HasDynamicRelocations = true;

    // Create island access, so we would emit the label and
    // move binary data during updateOutputValues, making us emit
    // dynamic relocation with the right offset value.
    getOrCreateIslandAccess(Address);
    return Error::success();
  }

  bool hasDynamicRelocationAtIsland() const {
    return !!(Islands && Islands->HasDynamicRelocations);
  }

  /// Called by an external function which wishes to emit references to constant
  /// island symbols of this function. We create a proxy for it, so we emit
  /// separate symbols when emitting our constant island on behalf of this other
  /// function.
  MCSymbol *getOrCreateProxyIslandAccess(uint64_t Address,
                                         BinaryFunction &Referrer) {
    MCSymbol *Symbol = getOrCreateIslandAccess(Address);
    if (!Symbol)
      return nullptr;

    MCSymbol *Proxy;
    if (!Islands->Proxies[&Referrer].count(Symbol)) {
      Proxy = BC.Ctx->getOrCreateSymbol(Symbol->getName() + ".proxy.for." +
                                        Referrer.getPrintName());
      Islands->Proxies[&Referrer][Symbol] = Proxy;
      Islands->Proxies[&Referrer][Proxy] = Symbol;
    }
    Proxy = Islands->Proxies[&Referrer][Symbol];
    return Proxy;
  }

  /// Make this function depend on \p BF because we have a reference to its
  /// constant island. When emitting this function,  we will also emit
  //  \p BF's constants. This only happens in custom AArch64 assembly code.
  void createIslandDependency(MCSymbol *Island, BinaryFunction *BF) {
    if (!Islands)
      Islands = std::make_unique<IslandInfo>();

    Islands->Dependency.insert(BF);
    Islands->ProxySymbols[Island] = BF;
  }

  /// Detects whether \p Address is inside a data region in this function
  /// (constant islands).
  bool isInConstantIsland(uint64_t Address) const {
    if (!Islands)
      return false;

    if (Address < getAddress())
      return false;

    uint64_t Offset = Address - getAddress();

    if (Offset >= getMaxSize())
      return false;

    auto DataIter = Islands->DataOffsets.upper_bound(Offset);
    if (DataIter == Islands->DataOffsets.begin())
      return false;
    DataIter = std::prev(DataIter);

    auto CodeIter = Islands->CodeOffsets.upper_bound(Offset);
    if (CodeIter == Islands->CodeOffsets.begin())
      return true;

    return *std::prev(CodeIter) <= *DataIter;
  }

  uint16_t getConstantIslandAlignment() const;

  /// If there is a constant island in the range [StartOffset, EndOffset),
  /// return its address.
  std::optional<uint64_t> getIslandInRange(uint64_t StartOffset,
                                           uint64_t EndOffset) const;

  uint64_t
  estimateConstantIslandSize(const BinaryFunction *OnBehalfOf = nullptr) const {
    if (!Islands)
      return 0;

    uint64_t Size = 0;
    for (auto DataIter = Islands->DataOffsets.begin();
         DataIter != Islands->DataOffsets.end(); ++DataIter) {
      auto NextData = std::next(DataIter);
      auto CodeIter = Islands->CodeOffsets.lower_bound(*DataIter);
      if (CodeIter == Islands->CodeOffsets.end() &&
          NextData == Islands->DataOffsets.end()) {
        Size += getMaxSize() - *DataIter;
        continue;
      }

      uint64_t NextMarker;
      if (CodeIter == Islands->CodeOffsets.end())
        NextMarker = *NextData;
      else if (NextData == Islands->DataOffsets.end())
        NextMarker = *CodeIter;
      else
        NextMarker = (*CodeIter > *NextData) ? *NextData : *CodeIter;

      Size += NextMarker - *DataIter;
    }

    if (!OnBehalfOf) {
      for (BinaryFunction *ExternalFunc : Islands->Dependency) {
        Size = alignTo(Size, ExternalFunc->getConstantIslandAlignment());
        Size += ExternalFunc->estimateConstantIslandSize(this);
      }
    }

    return Size;
  }

  bool hasIslandsInfo() const {
    return Islands && (hasConstantIsland() || !Islands->Dependency.empty());
  }

  bool hasConstantIsland() const {
    return Islands && !Islands->DataOffsets.empty();
  }

  /// Return true if the whole function is a constant island.
  bool isDataObject() const {
    return Islands && Islands->CodeOffsets.size() == 0;
  }

  bool isStartOfConstantIsland(uint64_t Offset) const {
    return hasConstantIsland() && Islands->DataOffsets.count(Offset);
  }

  /// Return true iff the symbol could be seen inside this function otherwise
  /// it is probably another function.
  bool isSymbolValidInScope(const SymbolRef &Symbol, uint64_t SymbolSize) const;

  /// Disassemble function from raw data.
  /// If successful, this function will populate the list of instructions
  /// for this function together with offsets from the function start
  /// in the input. It will also populate Labels with destinations for
  /// local branches, and TakenBranches with [from, to] info.
  ///
  /// The Function should be properly initialized before this function
  /// is called. I.e. function address and size should be set.
  ///
  /// Returns true on successful disassembly, and updates the current
  /// state to State:Disassembled.
  ///
  /// Returns false if disassembly failed.
  Error disassemble();

  /// An external interface to register a branch while the function is in
  /// disassembled state. Allows to make custom modifications to the
  /// disassembler. E.g., a pre-CFG pass can add an instruction and register
  /// a branch that will later be used during the CFG construction.
  ///
  /// Return a label at the branch destination.
  MCSymbol *registerBranch(uint64_t Src, uint64_t Dst);

  Error handlePCRelOperand(MCInst &Instruction, uint64_t Address,
                           uint64_t Size);

  MCSymbol *handleExternalReference(MCInst &Instruction, uint64_t Size,
                                    uint64_t Offset, uint64_t TargetAddress,
                                    bool &IsCall);

  void handleIndirectBranch(MCInst &Instruction, uint64_t Size,
                            uint64_t Offset);

  // Check for linker veneers, which lack relocations and need manual
  // adjustments.
  void handleAArch64IndirectCall(MCInst &Instruction, const uint64_t Offset);

  /// Analyze instruction to identify a function reference.
  void analyzeInstructionForFuncReference(const MCInst &Inst);

  /// Scan function for references to other functions. In relocation mode,
  /// add relocations for external references. In non-relocation mode, detect
  /// and mark new entry points.
  ///
  /// Return true on success. False if the disassembly failed or relocations
  /// could not be created.
  bool scanExternalRefs();

  /// Return the size of a data object located at \p Offset in the function.
  /// Return 0 if there is no data object at the \p Offset.
  size_t getSizeOfDataInCodeAt(uint64_t Offset) const;

  /// Verify that starting at \p Offset function contents are filled with
  /// zero-value bytes.
  bool isZeroPaddingAt(uint64_t Offset) const;

  /// Check that entry points have an associated instruction at their
  /// offsets after disassembly.
  void postProcessEntryPoints();

  /// Post-processing for jump tables after disassembly. Since their
  /// boundaries are not known until all call sites are seen, we need this
  /// extra pass to perform any final adjustments.
  void postProcessJumpTables();

  /// Builds a list of basic blocks with successor and predecessor info.
  ///
  /// The function should in Disassembled state prior to call.
  ///
  /// Returns true on success and update the current function state to
  /// State::CFG. Returns false if CFG cannot be built.
  Error buildCFG(MCPlusBuilder::AllocatorIdTy);

  /// Perform post-processing of the CFG.
  void postProcessCFG();

  /// Verify that any assumptions we've made about indirect branches were
  /// correct and also make any necessary changes to unknown indirect branches.
  ///
  /// Catch-22: we need to know indirect branch targets to build CFG, and
  /// in order to determine the value for indirect branches we need to know CFG.
  ///
  /// As such, the process of decoding indirect branches is broken into 2 steps:
  /// first we make our best guess about a branch without knowing the CFG,
  /// and later after we have the CFG for the function, we verify our earlier
  /// assumptions and also do our best at processing unknown indirect branches.
  ///
  /// Return true upon successful processing, or false if the control flow
  /// cannot be statically evaluated for any given indirect branch.
  bool postProcessIndirectBranches(MCPlusBuilder::AllocatorIdTy AllocId);

  /// Validate that all data references to function offsets are claimed by
  /// recognized jump tables. Register externally referenced blocks as entry
  /// points. Returns true if there are no unclaimed externally referenced
  /// offsets.
  bool validateExternallyReferencedOffsets();

  /// Return all call site profile info for this function.
  IndirectCallSiteProfile &getAllCallSites() { return AllCallSites; }

  const IndirectCallSiteProfile &getAllCallSites() const {
    return AllCallSites;
  }

  /// Walks the list of basic blocks filling in missing information about
  /// edge frequency for fall-throughs.
  ///
  /// Assumes the CFG has been built and edge frequency for taken branches
  /// has been filled with LBR data.
  void inferFallThroughCounts();

  /// Clear execution profile of the function.
  void clearProfile();

  /// Converts conditional tail calls to unconditional tail calls. We do this to
  /// handle conditional tail calls correctly and to give a chance to the
  /// simplify conditional tail call pass to decide whether to re-optimize them
  /// using profile information.
  void removeConditionalTailCalls();

  // Convert COUNT_NO_PROFILE to 0
  void removeTagsFromProfile();

  /// Computes a function hotness score: the sum of the products of BB frequency
  /// and size.
  uint64_t getFunctionScore() const;

  /// Get the number of instructions within this function.
  uint64_t getInstructionCount() const;

  const CFIInstrMapType &getFDEProgram() const { return FrameInstructions; }

  void moveRememberRestorePair(BinaryBasicBlock *BB);

  bool replayCFIInstrs(int32_t FromState, int32_t ToState,
                       BinaryBasicBlock *InBB,
                       BinaryBasicBlock::iterator InsertIt);

  /// unwindCFIState is used to unwind from a higher to a lower state number
  /// without using remember-restore instructions. We do that by keeping track
  /// of what values have been changed from state A to B and emitting
  /// instructions that undo this change.
  SmallVector<int32_t, 4> unwindCFIState(int32_t FromState, int32_t ToState,
                                         BinaryBasicBlock *InBB,
                                         BinaryBasicBlock::iterator &InsertIt);

  /// After reordering, this function checks the state of CFI and fixes it if it
  /// is corrupted. If it is unable to fix it, it returns false.
  bool finalizeCFIState();

  /// Return true if this function needs an address-translation table after
  /// its code emission.
  bool requiresAddressTranslation() const;

  /// Return true if the linker needs to generate an address map for this
  /// function. Used for keeping track of the mapping from input to out
  /// addresses of basic blocks.
  bool requiresAddressMap() const;

  /// Adjust branch instructions to match the CFG.
  ///
  /// As it comes to internal branches, the CFG represents "the ultimate source
  /// of truth". Transformations on functions and blocks have to update the CFG
  /// and fixBranches() would make sure the correct branch instructions are
  /// inserted at the end of basic blocks.
  ///
  /// We do require a conditional branch at the end of the basic block if
  /// the block has 2 successors as CFG currently lacks the conditional
  /// code support (it will probably stay that way). We only use this
  /// branch instruction for its conditional code, the destination is
  /// determined by CFG - first successor representing true/taken branch,
  /// while the second successor - false/fall-through branch.
  ///
  /// When we reverse the branch condition, the CFG is updated accordingly.
  void fixBranches();

  /// Mark function as finalized. No further optimizations are permitted.
  void setFinalized() { CurrentState = State::CFG_Finalized; }

  void setEmitted(bool KeepCFG = false) {
    CurrentState = State::EmittedCFG;
    if (!KeepCFG) {
      releaseCFG();
      CurrentState = State::Emitted;
    }
    clearList(Relocations);
  }

  /// Process LSDA information for the function.
  Error parseLSDA(ArrayRef<uint8_t> LSDAData, uint64_t LSDAAddress);

  /// Update exception handling ranges for the function.
  void updateEHRanges();

  /// Traverse cold basic blocks and replace references to constants in islands
  /// with a proxy symbol for the duplicated constant island that is going to be
  /// emitted in the cold region.
  void duplicateConstantIslands();

  /// Merge profile data of this function into those of the given
  /// function. The functions should have been proven identical with
  /// isIdenticalWith.
  void mergeProfileDataInto(BinaryFunction &BF) const;

  /// Returns the last computed hash value of the function.
  size_t getHash() const { return Hash; }

  /// Returns the function GUID.
  uint64_t getGUID() const { return GUID; }

  void setGUID(uint64_t Id) { GUID = Id; }

  using OperandHashFuncTy =
      function_ref<typename std::string(const MCOperand &)>;

  /// Compute the hash value of the function based on its contents.
  ///
  /// If \p UseDFS is set, process basic blocks in DFS order. Otherwise, use
  /// the existing layout order.
  /// \p HashFunction specifies which function is used for BF hashing.
  ///
  /// By default, instruction operands are ignored while calculating the hash.
  /// The caller can change this via passing \p OperandHashFunc function.
  /// The return result of this function will be mixed with internal hash.
  size_t computeHash(
      bool UseDFS = false, HashFunction HashFunction = HashFunction::Default,
      OperandHashFuncTy OperandHashFunc = [](const MCOperand &) {
        return std::string();
      }) const;

  /// Compute hash values for each block of the function.
  /// \p HashFunction specifies which function is used for BB hashing.
  void
  computeBlockHashes(HashFunction HashFunction = HashFunction::Default) const;

  void addDWARFUnit(DWARFUnit *Unit) { DwarfUnitMap[Unit->getOffset()] = Unit; }

  void removeDWARFUnit(DWARFUnit *Unit) {
    DwarfUnitMap.erase(Unit->getOffset());
  }

  /// Return DWARF compile units for this function.
  /// Returns a reference to the map of DWARF unit offsets to units.
  const DenseMap<uint64_t, DWARFUnit *> &getDWARFUnits() const {
    return DwarfUnitMap;
  }

  const DWARFDebugLine::LineTable *
  getDWARFLineTableForUnit(DWARFUnit *Unit) const {
    return BC.DwCtx->getLineTableForUnit(Unit);
  }

  /// Finalize profile for the function.
  void postProcessProfile();

  /// Returns an estimate of the function's hot part after splitting.
  /// This is a very rough estimate, as with C++ exceptions there are
  /// blocks we don't move, and it makes no attempt at estimating the size
  /// of the added/removed branch instructions.
  /// Note that this size is optimistic and the actual size may increase
  /// after relaxation.
  size_t estimateHotSize(const bool UseSplitSize = true) const {
    size_t Estimate = 0;
    if (UseSplitSize && isSplit()) {
      for (const BinaryBasicBlock &BB : blocks())
        if (!BB.isCold())
          Estimate += BC.computeCodeSize(BB.begin(), BB.end());
    } else {
      for (const BinaryBasicBlock &BB : blocks())
        if (BB.getKnownExecutionCount() != 0)
          Estimate += BC.computeCodeSize(BB.begin(), BB.end());
    }
    return Estimate;
  }

  size_t estimateColdSize() const {
    if (!isSplit())
      return estimateSize();
    size_t Estimate = 0;
    for (const BinaryBasicBlock &BB : blocks())
      if (BB.isCold())
        Estimate += BC.computeCodeSize(BB.begin(), BB.end());
    return Estimate;
  }

  size_t estimateSize() const {
    size_t Estimate = 0;
    for (const BinaryBasicBlock &BB : blocks())
      Estimate += BC.computeCodeSize(BB.begin(), BB.end());
    return Estimate;
  }

  /// Return output address ranges for a function.
  DebugAddressRangesVector getOutputAddressRanges() const;

  /// Given an address corresponding to an instruction in the input binary,
  /// return an address of this instruction in output binary.
  ///
  /// Return 0 if no matching address could be found or the instruction was
  /// removed.
  uint64_t translateInputToOutputAddress(uint64_t Address) const;

  /// Translate a contiguous range of addresses in the input binary into a set
  /// of ranges in the output binary.
  DebugAddressRangesVector
  translateInputToOutputRange(DebugAddressRange InRange) const;

  /// Return true if the function is an AArch64 linker inserted veneer
  bool isAArch64Veneer() const;

  virtual ~BinaryFunction();
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               const BinaryFunction &Function) {
  OS << Function.getPrintName();
  return OS;
}

/// Compare function by index if it is valid, fall back to the original address
/// otherwise.
inline bool compareBinaryFunctionByIndex(const BinaryFunction *A,
                                         const BinaryFunction *B) {
  if (A->hasValidIndex() && B->hasValidIndex())
    return A->getIndex() < B->getIndex();
  if (A->hasValidIndex() && !B->hasValidIndex())
    return true;
  if (!A->hasValidIndex() && B->hasValidIndex())
    return false;
  return A->getAddress() < B->getAddress();
}

} // namespace bolt

// GraphTraits specializations for function basic block graphs (CFGs)
template <>
struct GraphTraits<bolt::BinaryFunction *>
    : public GraphTraits<bolt::BinaryBasicBlock *> {
  static NodeRef getEntryNode(bolt::BinaryFunction *F) {
    return F->getLayout().block_front();
  }

  using nodes_iterator = pointer_iterator<bolt::BinaryFunction::iterator>;

  static nodes_iterator nodes_begin(bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->begin());
  }
  static nodes_iterator nodes_end(bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->end());
  }
  static size_t size(bolt::BinaryFunction *F) { return F->size(); }
};

template <>
struct GraphTraits<const bolt::BinaryFunction *>
    : public GraphTraits<const bolt::BinaryBasicBlock *> {
  static NodeRef getEntryNode(const bolt::BinaryFunction *F) {
    return F->getLayout().block_front();
  }

  using nodes_iterator = pointer_iterator<bolt::BinaryFunction::const_iterator>;

  static nodes_iterator nodes_begin(const bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->begin());
  }
  static nodes_iterator nodes_end(const bolt::BinaryFunction *F) {
    llvm_unreachable("Not implemented");
    return nodes_iterator(F->end());
  }
  static size_t size(const bolt::BinaryFunction *F) { return F->size(); }
};

template <>
struct GraphTraits<Inverse<bolt::BinaryFunction *>>
    : public GraphTraits<Inverse<bolt::BinaryBasicBlock *>> {
  static NodeRef getEntryNode(Inverse<bolt::BinaryFunction *> G) {
    return G.Graph->getLayout().block_front();
  }
};

template <>
struct GraphTraits<Inverse<const bolt::BinaryFunction *>>
    : public GraphTraits<Inverse<const bolt::BinaryBasicBlock *>> {
  static NodeRef getEntryNode(Inverse<const bolt::BinaryFunction *> G) {
    return G.Graph->getLayout().block_front();
  }
};

} // namespace llvm

#endif
