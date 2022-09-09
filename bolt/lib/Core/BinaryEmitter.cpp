//===- bolt/Core/BinaryEmitter.cpp - Emit code and data -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the collection of functions and classes used for
// emission of code and data into object/binary file.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/SMLoc.h"

#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::opt<JumpTableSupportLevel> JumpTables;
extern cl::opt<bool> PreserveBlocksAlignment;

cl::opt<bool> AlignBlocks("align-blocks", cl::desc("align basic blocks"),
                          cl::cat(BoltOptCategory));

cl::opt<MacroFusionType>
AlignMacroOpFusion("align-macro-fusion",
  cl::desc("fix instruction alignment for macro-fusion (x86 relocation mode)"),
  cl::init(MFT_HOT),
  cl::values(clEnumValN(MFT_NONE, "none",
               "do not insert alignment no-ops for macro-fusion"),
             clEnumValN(MFT_HOT, "hot",
               "only insert alignment no-ops on hot execution paths (default)"),
             clEnumValN(MFT_ALL, "all",
               "always align instructions to allow macro-fusion")),
  cl::ZeroOrMore,
  cl::cat(BoltRelocCategory));

static cl::list<std::string>
BreakFunctionNames("break-funcs",
  cl::CommaSeparated,
  cl::desc("list of functions to core dump on (debugging)"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::list<std::string>
FunctionPadSpec("pad-funcs",
  cl::CommaSeparated,
  cl::desc("list of functions to pad with amount of bytes"),
  cl::value_desc("func1:pad1,func2:pad2,func3:pad3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool> MarkFuncs(
    "mark-funcs",
    cl::desc("mark function boundaries with break instruction to make "
             "sure we accidentally don't cross them"),
    cl::ReallyHidden, cl::cat(BoltCategory));

static cl::opt<bool> PrintJumpTables("print-jump-tables",
                                     cl::desc("print jump tables"), cl::Hidden,
                                     cl::cat(BoltCategory));

static cl::opt<bool>
X86AlignBranchBoundaryHotOnly("x86-align-branch-boundary-hot-only",
  cl::desc("only apply branch boundary alignment in hot code"),
  cl::init(true),
  cl::cat(BoltOptCategory));

size_t padFunction(const BinaryFunction &Function) {
  static std::map<std::string, size_t> FunctionPadding;

  if (FunctionPadding.empty() && !FunctionPadSpec.empty()) {
    for (std::string &Spec : FunctionPadSpec) {
      size_t N = Spec.find(':');
      if (N == std::string::npos)
        continue;
      std::string Name = Spec.substr(0, N);
      size_t Padding = std::stoull(Spec.substr(N + 1));
      FunctionPadding[Name] = Padding;
    }
  }

  for (auto &FPI : FunctionPadding) {
    std::string Name = FPI.first;
    size_t Padding = FPI.second;
    if (Function.hasNameRegex(Name))
      return Padding;
  }

  return 0;
}

} // namespace opts

namespace {
using JumpTable = bolt::JumpTable;

class BinaryEmitter {
private:
  BinaryEmitter(const BinaryEmitter &) = delete;
  BinaryEmitter &operator=(const BinaryEmitter &) = delete;

  MCStreamer &Streamer;
  BinaryContext &BC;

public:
  BinaryEmitter(MCStreamer &Streamer, BinaryContext &BC)
      : Streamer(Streamer), BC(BC) {}

  /// Emit all code and data.
  void emitAll(StringRef OrgSecPrefix);

  /// Emit function code. The caller is responsible for emitting function
  /// symbol(s) and setting the section to emit the code to.
  void emitFunctionBody(BinaryFunction &BF, FunctionFragment &FF,
                        bool EmitCodeOnly = false);

private:
  /// Emit function code.
  void emitFunctions();

  /// Emit a single function.
  bool emitFunction(BinaryFunction &BF, FunctionFragment &FF);

  /// Helper for emitFunctionBody to write data inside a function
  /// (used for AArch64)
  void emitConstantIslands(BinaryFunction &BF, bool EmitColdPart,
                           BinaryFunction *OnBehalfOf = nullptr);

  /// Emit jump tables for the function.
  void emitJumpTables(const BinaryFunction &BF);

  /// Emit jump table data. Callee supplies sections for the data.
  void emitJumpTable(const JumpTable &JT, MCSection *HotSection,
                     MCSection *ColdSection);

  void emitCFIInstruction(const MCCFIInstruction &Inst) const;

  /// Emit exception handling ranges for the function.
  void emitLSDA(BinaryFunction &BF, const FunctionFragment &FF);

  /// Emit line number information corresponding to \p NewLoc. \p PrevLoc
  /// provides a context for de-duplication of line number info.
  /// \p FirstInstr indicates if \p NewLoc represents the first instruction
  /// in a sequence, such as a function fragment.
  ///
  /// Return new current location which is either \p NewLoc or \p PrevLoc.
  SMLoc emitLineInfo(const BinaryFunction &BF, SMLoc NewLoc, SMLoc PrevLoc,
                     bool FirstInstr);

  /// Use \p FunctionEndSymbol to mark the end of the line info sequence.
  /// Note that it does not automatically result in the insertion of the EOS
  /// marker in the line table program, but provides one to the DWARF generator
  /// when it needs it.
  void emitLineInfoEnd(const BinaryFunction &BF, MCSymbol *FunctionEndSymbol);

  /// Emit debug line info for unprocessed functions from CUs that include
  /// emitted functions.
  void emitDebugLineInfoForOriginalFunctions();

  /// Emit debug line for CUs that were not modified.
  void emitDebugLineInfoForUnprocessedCUs();

  /// Emit data sections that have code references in them.
  void emitDataSections(StringRef OrgSecPrefix);
};

} // anonymous namespace

void BinaryEmitter::emitAll(StringRef OrgSecPrefix) {
  Streamer.initSections(false, *BC.STI);

  if (opts::UpdateDebugSections && BC.isELF()) {
    // Force the emission of debug line info into allocatable section to ensure
    // RuntimeDyld will process it without ProcessAllSections flag.
    //
    // NB: on MachO all sections are required for execution, hence no need
    //     to change flags/attributes.
    MCSectionELF *ELFDwarfLineSection =
        static_cast<MCSectionELF *>(BC.MOFI->getDwarfLineSection());
    ELFDwarfLineSection->setFlags(ELF::SHF_ALLOC);
  }

  if (RuntimeLibrary *RtLibrary = BC.getRuntimeLibrary())
    RtLibrary->emitBinary(BC, Streamer);

  BC.getTextSection()->setAlignment(Align(opts::AlignText));

  emitFunctions();

  if (opts::UpdateDebugSections) {
    emitDebugLineInfoForOriginalFunctions();
    DwarfLineTable::emit(BC, Streamer);
  }

  emitDataSections(OrgSecPrefix);

  Streamer.emitLabel(BC.Ctx->getOrCreateSymbol("_end"));
}

void BinaryEmitter::emitFunctions() {
  auto emit = [&](const std::vector<BinaryFunction *> &Functions) {
    const bool HasProfile = BC.NumProfiledFuncs > 0;
    const bool OriginalAllowAutoPadding = Streamer.getAllowAutoPadding();
    for (BinaryFunction *Function : Functions) {
      if (!BC.shouldEmit(*Function))
        continue;

      LLVM_DEBUG(dbgs() << "BOLT: generating code for function \"" << *Function
                        << "\" : " << Function->getFunctionNumber() << '\n');

      // Was any part of the function emitted.
      bool Emitted = false;

      // Turn off Intel JCC Erratum mitigation for cold code if requested
      if (HasProfile && opts::X86AlignBranchBoundaryHotOnly &&
          !Function->hasValidProfile())
        Streamer.setAllowAutoPadding(false);

      FunctionLayout &Layout = Function->getLayout();
      Emitted |= emitFunction(*Function, Layout.getMainFragment());

      if (Function->isSplit()) {
        if (opts::X86AlignBranchBoundaryHotOnly)
          Streamer.setAllowAutoPadding(false);

        assert((Layout.fragment_size() == 1 || Function->isSimple()) &&
               "Only simple functions can have fragments");
        for (FunctionFragment &FF : Layout.getSplitFragments()) {
          // Skip empty fragments so no symbols and sections for empty fragments
          // are generated
          if (FF.empty() && !Function->hasConstantIsland())
            continue;
          Emitted |= emitFunction(*Function, FF);
        }
      }

      Streamer.setAllowAutoPadding(OriginalAllowAutoPadding);

      if (Emitted)
        Function->setEmitted(/*KeepCFG=*/opts::PrintCacheMetrics);
    }
  };

  // Mark the start of hot text.
  if (opts::HotText) {
    Streamer.switchSection(BC.getTextSection());
    Streamer.emitLabel(BC.getHotTextStartSymbol());
  }

  // Emit functions in sorted order.
  std::vector<BinaryFunction *> SortedFunctions = BC.getSortedFunctions();
  emit(SortedFunctions);

  // Emit functions added by BOLT.
  emit(BC.getInjectedBinaryFunctions());

  // Mark the end of hot text.
  if (opts::HotText) {
    Streamer.switchSection(BC.getTextSection());
    Streamer.emitLabel(BC.getHotTextEndSymbol());
  }
}

bool BinaryEmitter::emitFunction(BinaryFunction &Function,
                                 FunctionFragment &FF) {
  if (Function.size() == 0 && !Function.hasIslandsInfo())
    return false;

  if (Function.getState() == BinaryFunction::State::Empty)
    return false;

  MCSection *Section =
      BC.getCodeSection(Function.getCodeSectionName(FF.getFragmentNum()));
  Streamer.switchSection(Section);
  Section->setHasInstructions(true);
  BC.Ctx->addGenDwarfSection(Section);

  if (BC.HasRelocations) {
    // Set section alignment to at least maximum possible object alignment.
    // We need this to support LongJmp and other passes that calculates
    // tentative layout.
    if (Section->getAlignment() < opts::AlignFunctions)
      Section->setAlignment(Align(opts::AlignFunctions));

    Streamer.emitCodeAlignment(BinaryFunction::MinAlign, &*BC.STI);
    uint16_t MaxAlignBytes = FF.isSplitFragment()
                                 ? Function.getMaxColdAlignmentBytes()
                                 : Function.getMaxAlignmentBytes();
    if (MaxAlignBytes > 0)
      Streamer.emitCodeAlignment(Function.getAlignment(), &*BC.STI,
                                 MaxAlignBytes);
  } else {
    Streamer.emitCodeAlignment(Function.getAlignment(), &*BC.STI);
  }

  MCContext &Context = Streamer.getContext();
  const MCAsmInfo *MAI = Context.getAsmInfo();

  MCSymbol *const StartSymbol = Function.getSymbol(FF.getFragmentNum());

  // Emit all symbols associated with the main function entry.
  if (FF.isMainFragment()) {
    for (MCSymbol *Symbol : Function.getSymbols()) {
      Streamer.emitSymbolAttribute(Symbol, MCSA_ELF_TypeFunction);
      Streamer.emitLabel(Symbol);
    }
  } else {
    Streamer.emitSymbolAttribute(StartSymbol, MCSA_ELF_TypeFunction);
    Streamer.emitLabel(StartSymbol);
  }

  // Emit CFI start
  if (Function.hasCFI()) {
    Streamer.emitCFIStartProc(/*IsSimple=*/false);
    if (Function.getPersonalityFunction() != nullptr)
      Streamer.emitCFIPersonality(Function.getPersonalityFunction(),
                                  Function.getPersonalityEncoding());
    MCSymbol *LSDASymbol = Function.getLSDASymbol(FF.getFragmentNum());
    if (LSDASymbol)
      Streamer.emitCFILsda(LSDASymbol, BC.LSDAEncoding);
    else
      Streamer.emitCFILsda(0, dwarf::DW_EH_PE_omit);
    // Emit CFI instructions relative to the CIE
    for (const MCCFIInstruction &CFIInstr : Function.cie()) {
      // Only write CIE CFI insns that LLVM will not already emit
      const std::vector<MCCFIInstruction> &FrameInstrs =
          MAI->getInitialFrameState();
      if (!llvm::is_contained(FrameInstrs, CFIInstr))
        emitCFIInstruction(CFIInstr);
    }
  }

  assert((Function.empty() || !(*Function.begin()).isCold()) &&
         "first basic block should never be cold");

  // Emit UD2 at the beginning if requested by user.
  if (!opts::BreakFunctionNames.empty()) {
    for (std::string &Name : opts::BreakFunctionNames) {
      if (Function.hasNameRegex(Name)) {
        Streamer.emitIntValue(0x0B0F, 2); // UD2: 0F 0B
        break;
      }
    }
  }

  // Emit code.
  emitFunctionBody(Function, FF, /*EmitCodeOnly=*/false);

  // Emit padding if requested.
  if (size_t Padding = opts::padFunction(Function)) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: padding function " << Function << " with "
                      << Padding << " bytes\n");
    Streamer.emitFill(Padding, MAI->getTextAlignFillValue());
  }

  if (opts::MarkFuncs)
    Streamer.emitIntValue(BC.MIB->getTrapFillValue(), 1);

  // Emit CFI end
  if (Function.hasCFI())
    Streamer.emitCFIEndProc();

  MCSymbol *EndSymbol = Function.getFunctionEndLabel(FF.getFragmentNum());
  Streamer.emitLabel(EndSymbol);

  if (MAI->hasDotTypeDotSizeDirective()) {
    const MCExpr *SizeExpr = MCBinaryExpr::createSub(
        MCSymbolRefExpr::create(EndSymbol, Context),
        MCSymbolRefExpr::create(StartSymbol, Context), Context);
    Streamer.emitELFSize(StartSymbol, SizeExpr);
  }

  if (opts::UpdateDebugSections && Function.getDWARFUnit())
    emitLineInfoEnd(Function, EndSymbol);

  // Exception handling info for the function.
  emitLSDA(Function, FF);

  if (FF.isMainFragment() && opts::JumpTables > JTS_NONE)
    emitJumpTables(Function);

  return true;
}

void BinaryEmitter::emitFunctionBody(BinaryFunction &BF, FunctionFragment &FF,
                                     bool EmitCodeOnly) {
  if (!EmitCodeOnly && FF.isSplitFragment() && BF.hasConstantIsland()) {
    assert(BF.getLayout().isHotColdSplit() &&
           "Constant island support only with hot/cold split");
    BF.duplicateConstantIslands();
  }

  if (!FF.empty() && FF.front()->isLandingPad()) {
    assert(!FF.front()->isEntryPoint() &&
           "Landing pad cannot be entry point of function");
    // If the first block of the fragment is a landing pad, it's offset from the
    // start of the area that the corresponding LSDA describes is zero. In this
    // case, the call site entries in that LSDA have 0 as offset to the landing
    // pad, which the runtime interprets as "no handler". To prevent this,
    // insert some padding.
    Streamer.emitIntValue(BC.MIB->getTrapFillValue(), 1);
  }

  // Track the first emitted instruction with debug info.
  bool FirstInstr = true;
  for (BinaryBasicBlock *const BB : FF) {
    if ((opts::AlignBlocks || opts::PreserveBlocksAlignment) &&
        BB->getAlignment() > 1)
      Streamer.emitCodeAlignment(BB->getAlignment(), &*BC.STI,
                                 BB->getAlignmentMaxBytes());
    Streamer.emitLabel(BB->getLabel());
    if (!EmitCodeOnly) {
      if (MCSymbol *EntrySymbol = BF.getSecondaryEntryPointSymbol(*BB))
        Streamer.emitLabel(EntrySymbol);
    }

    // Check if special alignment for macro-fusion is needed.
    bool MayNeedMacroFusionAlignment =
        (opts::AlignMacroOpFusion == MFT_ALL) ||
        (opts::AlignMacroOpFusion == MFT_HOT && BB->getKnownExecutionCount());
    BinaryBasicBlock::const_iterator MacroFusionPair;
    if (MayNeedMacroFusionAlignment) {
      MacroFusionPair = BB->getMacroOpFusionPair();
      if (MacroFusionPair == BB->end())
        MayNeedMacroFusionAlignment = false;
    }

    SMLoc LastLocSeen;
    // Remember if the last instruction emitted was a prefix.
    bool LastIsPrefix = false;
    for (auto I = BB->begin(), E = BB->end(); I != E; ++I) {
      MCInst &Instr = *I;

      if (EmitCodeOnly && BC.MIB->isPseudo(Instr))
        continue;

      // Handle pseudo instructions.
      if (BC.MIB->isEHLabel(Instr)) {
        const MCSymbol *Label = BC.MIB->getTargetSymbol(Instr);
        assert(Instr.getNumOperands() >= 1 && Label &&
               "bad EH_LABEL instruction");
        Streamer.emitLabel(const_cast<MCSymbol *>(Label));
        continue;
      }
      if (BC.MIB->isCFI(Instr)) {
        emitCFIInstruction(*BF.getCFIFor(Instr));
        continue;
      }

      // Handle macro-fusion alignment. If we emitted a prefix as
      // the last instruction, we should've already emitted the associated
      // alignment hint, so don't emit it twice.
      if (MayNeedMacroFusionAlignment && !LastIsPrefix &&
          I == MacroFusionPair) {
        // This assumes the second instruction in the macro-op pair will get
        // assigned to its own MCRelaxableFragment. Since all JCC instructions
        // are relaxable, we should be safe.
      }

      if (!EmitCodeOnly && opts::UpdateDebugSections && BF.getDWARFUnit()) {
        LastLocSeen = emitLineInfo(BF, Instr.getLoc(), LastLocSeen, FirstInstr);
        FirstInstr = false;
      }

      // Prepare to tag this location with a label if we need to keep track of
      // the location of calls/returns for BOLT address translation maps
      if (!EmitCodeOnly && BF.requiresAddressTranslation() &&
          BC.MIB->getOffset(Instr)) {
        const uint32_t Offset = *BC.MIB->getOffset(Instr);
        MCSymbol *LocSym = BC.Ctx->createTempSymbol();
        Streamer.emitLabel(LocSym);
        BB->getLocSyms().emplace_back(Offset, LocSym);
      }

      Streamer.emitInstruction(Instr, *BC.STI);
      LastIsPrefix = BC.MIB->isPrefix(Instr);
    }
  }

  if (!EmitCodeOnly)
    emitConstantIslands(BF, FF.isSplitFragment());
}

void BinaryEmitter::emitConstantIslands(BinaryFunction &BF, bool EmitColdPart,
                                        BinaryFunction *OnBehalfOf) {
  if (!BF.hasIslandsInfo())
    return;

  BinaryFunction::IslandInfo &Islands = BF.getIslandInfo();
  if (Islands.DataOffsets.empty() && Islands.Dependency.empty())
    return;

  // AArch64 requires CI to be aligned to 8 bytes due to access instructions
  // restrictions. E.g. the ldr with imm, where imm must be aligned to 8 bytes.
  const uint16_t Alignment = OnBehalfOf
                                 ? OnBehalfOf->getConstantIslandAlignment()
                                 : BF.getConstantIslandAlignment();
  Streamer.emitCodeAlignment(Alignment, &*BC.STI);

  if (!OnBehalfOf) {
    if (!EmitColdPart)
      Streamer.emitLabel(BF.getFunctionConstantIslandLabel());
    else
      Streamer.emitLabel(BF.getFunctionColdConstantIslandLabel());
  }

  assert((!OnBehalfOf || Islands.Proxies[OnBehalfOf].size() > 0) &&
         "spurious OnBehalfOf constant island emission");

  assert(!BF.isInjected() &&
         "injected functions should not have constant islands");
  // Raw contents of the function.
  StringRef SectionContents = BF.getOriginSection()->getContents();

  // Raw contents of the function.
  StringRef FunctionContents = SectionContents.substr(
      BF.getAddress() - BF.getOriginSection()->getAddress(), BF.getMaxSize());

  if (opts::Verbosity && !OnBehalfOf)
    outs() << "BOLT-INFO: emitting constant island for function " << BF << "\n";

  // We split the island into smaller blocks and output labels between them.
  auto IS = Islands.Offsets.begin();
  for (auto DataIter = Islands.DataOffsets.begin();
       DataIter != Islands.DataOffsets.end(); ++DataIter) {
    uint64_t FunctionOffset = *DataIter;
    uint64_t EndOffset = 0ULL;

    // Determine size of this data chunk
    auto NextData = std::next(DataIter);
    auto CodeIter = Islands.CodeOffsets.lower_bound(*DataIter);
    if (CodeIter == Islands.CodeOffsets.end() &&
        NextData == Islands.DataOffsets.end())
      EndOffset = BF.getMaxSize();
    else if (CodeIter == Islands.CodeOffsets.end())
      EndOffset = *NextData;
    else if (NextData == Islands.DataOffsets.end())
      EndOffset = *CodeIter;
    else
      EndOffset = (*CodeIter > *NextData) ? *NextData : *CodeIter;

    if (FunctionOffset == EndOffset)
      continue; // Size is zero, nothing to emit

    auto emitCI = [&](uint64_t &FunctionOffset, uint64_t EndOffset) {
      if (FunctionOffset >= EndOffset)
        return;

      for (auto It = Islands.Relocations.lower_bound(FunctionOffset);
           It != Islands.Relocations.end(); ++It) {
        if (It->first >= EndOffset)
          break;

        const Relocation &Relocation = It->second;
        if (FunctionOffset < Relocation.Offset) {
          Streamer.emitBytes(
              FunctionContents.slice(FunctionOffset, Relocation.Offset));
          FunctionOffset = Relocation.Offset;
        }

        LLVM_DEBUG(
            dbgs() << "BOLT-DEBUG: emitting constant island relocation"
                   << " for " << BF << " at offset 0x"
                   << Twine::utohexstr(Relocation.Offset) << " with size "
                   << Relocation::getSizeForType(Relocation.Type) << '\n');

        FunctionOffset += Relocation.emit(&Streamer);
      }

      assert(FunctionOffset <= EndOffset && "overflow error");
      if (FunctionOffset < EndOffset) {
        Streamer.emitBytes(FunctionContents.slice(FunctionOffset, EndOffset));
        FunctionOffset = EndOffset;
      }
    };

    // Emit labels, relocs and data
    while (IS != Islands.Offsets.end() && IS->first < EndOffset) {
      auto NextLabelOffset =
          IS == Islands.Offsets.end() ? EndOffset : IS->first;
      auto NextStop = std::min(NextLabelOffset, EndOffset);
      assert(NextStop <= EndOffset && "internal overflow error");
      emitCI(FunctionOffset, NextStop);
      if (IS != Islands.Offsets.end() && FunctionOffset == IS->first) {
        // This is a slightly complex code to decide which label to emit. We
        // have 4 cases to handle: regular symbol, cold symbol, regular or cold
        // symbol being emitted on behalf of an external function.
        if (!OnBehalfOf) {
          if (!EmitColdPart) {
            LLVM_DEBUG(dbgs() << "BOLT-DEBUG: emitted label "
                              << IS->second->getName() << " at offset 0x"
                              << Twine::utohexstr(IS->first) << '\n');
            if (IS->second->isUndefined())
              Streamer.emitLabel(IS->second);
            else
              assert(BF.hasName(std::string(IS->second->getName())));
          } else if (Islands.ColdSymbols.count(IS->second) != 0) {
            LLVM_DEBUG(dbgs()
                       << "BOLT-DEBUG: emitted label "
                       << Islands.ColdSymbols[IS->second]->getName() << '\n');
            if (Islands.ColdSymbols[IS->second]->isUndefined())
              Streamer.emitLabel(Islands.ColdSymbols[IS->second]);
          }
        } else {
          if (!EmitColdPart) {
            if (MCSymbol *Sym = Islands.Proxies[OnBehalfOf][IS->second]) {
              LLVM_DEBUG(dbgs() << "BOLT-DEBUG: emitted label "
                                << Sym->getName() << '\n');
              Streamer.emitLabel(Sym);
            }
          } else if (MCSymbol *Sym =
                         Islands.ColdProxies[OnBehalfOf][IS->second]) {
            LLVM_DEBUG(dbgs() << "BOLT-DEBUG: emitted label " << Sym->getName()
                              << '\n');
            Streamer.emitLabel(Sym);
          }
        }
        ++IS;
      }
    }
    assert(FunctionOffset <= EndOffset && "overflow error");
    emitCI(FunctionOffset, EndOffset);
  }
  assert(IS == Islands.Offsets.end() && "some symbols were not emitted!");

  if (OnBehalfOf)
    return;
  // Now emit constant islands from other functions that we may have used in
  // this function.
  for (BinaryFunction *ExternalFunc : Islands.Dependency)
    emitConstantIslands(*ExternalFunc, EmitColdPart, &BF);
}

SMLoc BinaryEmitter::emitLineInfo(const BinaryFunction &BF, SMLoc NewLoc,
                                  SMLoc PrevLoc, bool FirstInstr) {
  DWARFUnit *FunctionCU = BF.getDWARFUnit();
  const DWARFDebugLine::LineTable *FunctionLineTable = BF.getDWARFLineTable();
  assert(FunctionCU && "cannot emit line info for function without CU");

  DebugLineTableRowRef RowReference = DebugLineTableRowRef::fromSMLoc(NewLoc);

  // Check if no new line info needs to be emitted.
  if (RowReference == DebugLineTableRowRef::NULL_ROW ||
      NewLoc.getPointer() == PrevLoc.getPointer())
    return PrevLoc;

  unsigned CurrentFilenum = 0;
  const DWARFDebugLine::LineTable *CurrentLineTable = FunctionLineTable;

  // If the CU id from the current instruction location does not
  // match the CU id from the current function, it means that we
  // have come across some inlined code.  We must look up the CU
  // for the instruction's original function and get the line table
  // from that.
  const uint64_t FunctionUnitIndex = FunctionCU->getOffset();
  const uint32_t CurrentUnitIndex = RowReference.DwCompileUnitIndex;
  if (CurrentUnitIndex != FunctionUnitIndex) {
    CurrentLineTable = BC.DwCtx->getLineTableForUnit(
        BC.DwCtx->getCompileUnitForOffset(CurrentUnitIndex));
    // Add filename from the inlined function to the current CU.
    CurrentFilenum = BC.addDebugFilenameToUnit(
        FunctionUnitIndex, CurrentUnitIndex,
        CurrentLineTable->Rows[RowReference.RowIndex - 1].File);
  }

  const DWARFDebugLine::Row &CurrentRow =
      CurrentLineTable->Rows[RowReference.RowIndex - 1];
  if (!CurrentFilenum)
    CurrentFilenum = CurrentRow.File;

  unsigned Flags = (DWARF2_FLAG_IS_STMT * CurrentRow.IsStmt) |
                   (DWARF2_FLAG_BASIC_BLOCK * CurrentRow.BasicBlock) |
                   (DWARF2_FLAG_PROLOGUE_END * CurrentRow.PrologueEnd) |
                   (DWARF2_FLAG_EPILOGUE_BEGIN * CurrentRow.EpilogueBegin);

  // Always emit is_stmt at the beginning of function fragment.
  if (FirstInstr)
    Flags |= DWARF2_FLAG_IS_STMT;

  BC.Ctx->setCurrentDwarfLoc(CurrentFilenum, CurrentRow.Line, CurrentRow.Column,
                             Flags, CurrentRow.Isa, CurrentRow.Discriminator);
  const MCDwarfLoc &DwarfLoc = BC.Ctx->getCurrentDwarfLoc();
  BC.Ctx->clearDwarfLocSeen();

  MCSymbol *LineSym = BC.Ctx->createTempSymbol();
  Streamer.emitLabel(LineSym);

  BC.getDwarfLineTable(FunctionUnitIndex)
      .getMCLineSections()
      .addLineEntry(MCDwarfLineEntry(LineSym, DwarfLoc),
                    Streamer.getCurrentSectionOnly());

  return NewLoc;
}

void BinaryEmitter::emitLineInfoEnd(const BinaryFunction &BF,
                                    MCSymbol *FunctionEndLabel) {
  DWARFUnit *FunctionCU = BF.getDWARFUnit();
  assert(FunctionCU && "DWARF unit expected");
  BC.Ctx->setCurrentDwarfLoc(0, 0, 0, DWARF2_FLAG_END_SEQUENCE, 0, 0);
  const MCDwarfLoc &DwarfLoc = BC.Ctx->getCurrentDwarfLoc();
  BC.Ctx->clearDwarfLocSeen();
  BC.getDwarfLineTable(FunctionCU->getOffset())
      .getMCLineSections()
      .addLineEntry(MCDwarfLineEntry(FunctionEndLabel, DwarfLoc),
                    Streamer.getCurrentSectionOnly());
}

void BinaryEmitter::emitJumpTables(const BinaryFunction &BF) {
  MCSection *ReadOnlySection = BC.MOFI->getReadOnlySection();
  MCSection *ReadOnlyColdSection = BC.MOFI->getContext().getELFSection(
      ".rodata.cold", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);

  if (!BF.hasJumpTables())
    return;

  if (opts::PrintJumpTables)
    outs() << "BOLT-INFO: jump tables for function " << BF << ":\n";

  for (auto &JTI : BF.jumpTables()) {
    JumpTable &JT = *JTI.second;
    if (opts::PrintJumpTables)
      JT.print(outs());
    if ((opts::JumpTables == JTS_BASIC || !BF.isSimple()) &&
        BC.HasRelocations) {
      JT.updateOriginal();
    } else {
      MCSection *HotSection, *ColdSection;
      if (opts::JumpTables == JTS_BASIC) {
        // In non-relocation mode we have to emit jump tables in local sections.
        // This way we only overwrite them when the corresponding function is
        // overwritten.
        std::string Name = ".local." + JT.Labels[0]->getName().str();
        std::replace(Name.begin(), Name.end(), '/', '.');
        BinarySection &Section =
            BC.registerOrUpdateSection(Name, ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
        Section.setAnonymous(true);
        JT.setOutputSection(Section);
        HotSection = BC.getDataSection(Name);
        ColdSection = HotSection;
      } else {
        if (BF.isSimple()) {
          HotSection = ReadOnlySection;
          ColdSection = ReadOnlyColdSection;
        } else {
          HotSection = BF.hasProfile() ? ReadOnlySection : ReadOnlyColdSection;
          ColdSection = HotSection;
        }
      }
      emitJumpTable(JT, HotSection, ColdSection);
    }
  }
}

void BinaryEmitter::emitJumpTable(const JumpTable &JT, MCSection *HotSection,
                                  MCSection *ColdSection) {
  // Pre-process entries for aggressive splitting.
  // Each label represents a separate switch table and gets its own count
  // determining its destination.
  std::map<MCSymbol *, uint64_t> LabelCounts;
  if (opts::JumpTables > JTS_SPLIT && !JT.Counts.empty()) {
    MCSymbol *CurrentLabel = JT.Labels.at(0);
    uint64_t CurrentLabelCount = 0;
    for (unsigned Index = 0; Index < JT.Entries.size(); ++Index) {
      auto LI = JT.Labels.find(Index * JT.EntrySize);
      if (LI != JT.Labels.end()) {
        LabelCounts[CurrentLabel] = CurrentLabelCount;
        CurrentLabel = LI->second;
        CurrentLabelCount = 0;
      }
      CurrentLabelCount += JT.Counts[Index].Count;
    }
    LabelCounts[CurrentLabel] = CurrentLabelCount;
  } else {
    Streamer.switchSection(JT.Count > 0 ? HotSection : ColdSection);
    Streamer.emitValueToAlignment(JT.EntrySize);
  }
  MCSymbol *LastLabel = nullptr;
  uint64_t Offset = 0;
  for (MCSymbol *Entry : JT.Entries) {
    auto LI = JT.Labels.find(Offset);
    if (LI != JT.Labels.end()) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: emitting jump table "
                        << LI->second->getName()
                        << " (originally was at address 0x"
                        << Twine::utohexstr(JT.getAddress() + Offset)
                        << (Offset ? "as part of larger jump table\n" : "\n"));
      if (!LabelCounts.empty()) {
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: jump table count: "
                          << LabelCounts[LI->second] << '\n');
        if (LabelCounts[LI->second] > 0)
          Streamer.switchSection(HotSection);
        else
          Streamer.switchSection(ColdSection);
        Streamer.emitValueToAlignment(JT.EntrySize);
      }
      Streamer.emitLabel(LI->second);
      LastLabel = LI->second;
    }
    if (JT.Type == JumpTable::JTT_NORMAL) {
      Streamer.emitSymbolValue(Entry, JT.OutputEntrySize);
    } else { // JTT_PIC
      const MCSymbolRefExpr *JTExpr =
          MCSymbolRefExpr::create(LastLabel, Streamer.getContext());
      const MCSymbolRefExpr *E =
          MCSymbolRefExpr::create(Entry, Streamer.getContext());
      const MCBinaryExpr *Value =
          MCBinaryExpr::createSub(E, JTExpr, Streamer.getContext());
      Streamer.emitValue(Value, JT.EntrySize);
    }
    Offset += JT.EntrySize;
  }
}

void BinaryEmitter::emitCFIInstruction(const MCCFIInstruction &Inst) const {
  switch (Inst.getOperation()) {
  default:
    llvm_unreachable("Unexpected instruction");
  case MCCFIInstruction::OpDefCfaOffset:
    Streamer.emitCFIDefCfaOffset(Inst.getOffset());
    break;
  case MCCFIInstruction::OpAdjustCfaOffset:
    Streamer.emitCFIAdjustCfaOffset(Inst.getOffset());
    break;
  case MCCFIInstruction::OpDefCfa:
    Streamer.emitCFIDefCfa(Inst.getRegister(), Inst.getOffset());
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    Streamer.emitCFIDefCfaRegister(Inst.getRegister());
    break;
  case MCCFIInstruction::OpOffset:
    Streamer.emitCFIOffset(Inst.getRegister(), Inst.getOffset());
    break;
  case MCCFIInstruction::OpRegister:
    Streamer.emitCFIRegister(Inst.getRegister(), Inst.getRegister2());
    break;
  case MCCFIInstruction::OpWindowSave:
    Streamer.emitCFIWindowSave();
    break;
  case MCCFIInstruction::OpNegateRAState:
    Streamer.emitCFINegateRAState();
    break;
  case MCCFIInstruction::OpSameValue:
    Streamer.emitCFISameValue(Inst.getRegister());
    break;
  case MCCFIInstruction::OpGnuArgsSize:
    Streamer.emitCFIGnuArgsSize(Inst.getOffset());
    break;
  case MCCFIInstruction::OpEscape:
    Streamer.AddComment(Inst.getComment());
    Streamer.emitCFIEscape(Inst.getValues());
    break;
  case MCCFIInstruction::OpRestore:
    Streamer.emitCFIRestore(Inst.getRegister());
    break;
  case MCCFIInstruction::OpUndefined:
    Streamer.emitCFIUndefined(Inst.getRegister());
    break;
  }
}

// The code is based on EHStreamer::emitExceptionTable().
void BinaryEmitter::emitLSDA(BinaryFunction &BF, const FunctionFragment &FF) {
  const BinaryFunction::CallSitesRange Sites =
      BF.getCallSites(FF.getFragmentNum());
  if (llvm::empty(Sites))
    return;

  // Calculate callsite table size. Size of each callsite entry is:
  //
  //  sizeof(start) + sizeof(length) + sizeof(LP) + sizeof(uleb128(action))
  //
  // or
  //
  //  sizeof(dwarf::DW_EH_PE_data4) * 3 + sizeof(uleb128(action))
  uint64_t CallSiteTableLength = llvm::size(Sites) * 4 * 3;
  for (const auto &FragmentCallSite : Sites)
    CallSiteTableLength += getULEB128Size(FragmentCallSite.second.Action);

  Streamer.switchSection(BC.MOFI->getLSDASection());

  const unsigned TTypeEncoding = BC.TTypeEncoding;
  const unsigned TTypeEncodingSize = BC.getDWARFEncodingSize(TTypeEncoding);
  const uint16_t TTypeAlignment = 4;

  // Type tables have to be aligned at 4 bytes.
  Streamer.emitValueToAlignment(TTypeAlignment);

  // Emit the LSDA label.
  MCSymbol *LSDASymbol = BF.getLSDASymbol(FF.getFragmentNum());
  assert(LSDASymbol && "no LSDA symbol set");
  Streamer.emitLabel(LSDASymbol);

  // Corresponding FDE start.
  const MCSymbol *StartSymbol = BF.getSymbol(FF.getFragmentNum());

  // Emit the LSDA header.

  // If LPStart is omitted, then the start of the FDE is used as a base for
  // landing pad displacements. Then if a cold fragment starts with
  // a landing pad, this means that the first landing pad offset will be 0.
  // As a result, the exception handling runtime will ignore this landing pad
  // because zero offset denotes the absence of a landing pad.
  // For this reason, when the binary has fixed starting address we emit LPStart
  // as 0 and output the absolute value of the landing pad in the table.
  //
  // If the base address can change, we cannot use absolute addresses for
  // landing pads (at least not without runtime relocations). Hence, we fall
  // back to emitting landing pads relative to the FDE start.
  // As we are emitting label differences, we have to guarantee both labels are
  // defined in the same section and hence cannot place the landing pad into a
  // cold fragment when the corresponding call site is in the hot fragment.
  // Because of this issue and the previously described issue of possible
  // zero-offset landing pad we have to place landing pads in the same section
  // as the corresponding invokes for shared objects.
  std::function<void(const MCSymbol *)> emitLandingPad;
  if (BC.HasFixedLoadAddress) {
    Streamer.emitIntValue(dwarf::DW_EH_PE_udata4, 1); // LPStart format
    Streamer.emitIntValue(0, 4);                      // LPStart
    emitLandingPad = [&](const MCSymbol *LPSymbol) {
      if (!LPSymbol)
        Streamer.emitIntValue(0, 4);
      else
        Streamer.emitSymbolValue(LPSymbol, 4);
    };
  } else {
    Streamer.emitIntValue(dwarf::DW_EH_PE_omit, 1); // LPStart format
    emitLandingPad = [&](const MCSymbol *LPSymbol) {
      if (!LPSymbol)
        Streamer.emitIntValue(0, 4);
      else
        Streamer.emitAbsoluteSymbolDiff(LPSymbol, StartSymbol, 4);
    };
  }

  Streamer.emitIntValue(TTypeEncoding, 1); // TType format

  // See the comment in EHStreamer::emitExceptionTable() on to use
  // uleb128 encoding (which can use variable number of bytes to encode the same
  // value) to ensure type info table is properly aligned at 4 bytes without
  // iteratively fixing sizes of the tables.
  unsigned CallSiteTableLengthSize = getULEB128Size(CallSiteTableLength);
  unsigned TTypeBaseOffset =
      sizeof(int8_t) +                 // Call site format
      CallSiteTableLengthSize +        // Call site table length size
      CallSiteTableLength +            // Call site table length
      BF.getLSDAActionTable().size() + // Actions table size
      BF.getLSDATypeTable().size() * TTypeEncodingSize; // Types table size
  unsigned TTypeBaseOffsetSize = getULEB128Size(TTypeBaseOffset);
  unsigned TotalSize = sizeof(int8_t) +      // LPStart format
                       sizeof(int8_t) +      // TType format
                       TTypeBaseOffsetSize + // TType base offset size
                       TTypeBaseOffset;      // TType base offset
  unsigned SizeAlign = (4 - TotalSize) & 3;

  // Account for any extra padding that will be added to the call site table
  // length.
  Streamer.emitULEB128IntValue(TTypeBaseOffset,
                               /*PadTo=*/TTypeBaseOffsetSize + SizeAlign);

  // Emit the landing pad call site table. We use signed data4 since we can emit
  // a landing pad in a different part of the split function that could appear
  // earlier in the address space than LPStart.
  Streamer.emitIntValue(dwarf::DW_EH_PE_sdata4, 1);
  Streamer.emitULEB128IntValue(CallSiteTableLength);

  for (const auto &FragmentCallSite : Sites) {
    const BinaryFunction::CallSite &CallSite = FragmentCallSite.second;
    const MCSymbol *BeginLabel = CallSite.Start;
    const MCSymbol *EndLabel = CallSite.End;

    assert(BeginLabel && "start EH label expected");
    assert(EndLabel && "end EH label expected");

    // Start of the range is emitted relative to the start of current
    // function split part.
    Streamer.emitAbsoluteSymbolDiff(BeginLabel, StartSymbol, 4);
    Streamer.emitAbsoluteSymbolDiff(EndLabel, BeginLabel, 4);
    emitLandingPad(CallSite.LP);
    Streamer.emitULEB128IntValue(CallSite.Action);
  }

  // Write out action, type, and type index tables at the end.
  //
  // For action and type index tables there's no need to change the original
  // table format unless we are doing function splitting, in which case we can
  // split and optimize the tables.
  //
  // For type table we (re-)encode the table using TTypeEncoding matching
  // the current assembler mode.
  for (uint8_t const &Byte : BF.getLSDAActionTable())
    Streamer.emitIntValue(Byte, 1);

  const BinaryFunction::LSDATypeTableTy &TypeTable =
      (TTypeEncoding & dwarf::DW_EH_PE_indirect) ? BF.getLSDATypeAddressTable()
                                                 : BF.getLSDATypeTable();
  assert(TypeTable.size() == BF.getLSDATypeTable().size() &&
         "indirect type table size mismatch");

  for (int Index = TypeTable.size() - 1; Index >= 0; --Index) {
    const uint64_t TypeAddress = TypeTable[Index];
    switch (TTypeEncoding & 0x70) {
    default:
      llvm_unreachable("unsupported TTypeEncoding");
    case dwarf::DW_EH_PE_absptr:
      Streamer.emitIntValue(TypeAddress, TTypeEncodingSize);
      break;
    case dwarf::DW_EH_PE_pcrel: {
      if (TypeAddress) {
        const MCSymbol *TypeSymbol =
            BC.getOrCreateGlobalSymbol(TypeAddress, "TI", 0, TTypeAlignment);
        MCSymbol *DotSymbol = BC.Ctx->createNamedTempSymbol();
        Streamer.emitLabel(DotSymbol);
        const MCBinaryExpr *SubDotExpr = MCBinaryExpr::createSub(
            MCSymbolRefExpr::create(TypeSymbol, *BC.Ctx),
            MCSymbolRefExpr::create(DotSymbol, *BC.Ctx), *BC.Ctx);
        Streamer.emitValue(SubDotExpr, TTypeEncodingSize);
      } else {
        Streamer.emitIntValue(0, TTypeEncodingSize);
      }
      break;
    }
    }
  }
  for (uint8_t const &Byte : BF.getLSDATypeIndexTable())
    Streamer.emitIntValue(Byte, 1);
}

void BinaryEmitter::emitDebugLineInfoForOriginalFunctions() {
  // If a function is in a CU containing at least one processed function, we
  // have to rewrite the whole line table for that CU. For unprocessed functions
  // we use data from the input line table.
  for (auto &It : BC.getBinaryFunctions()) {
    const BinaryFunction &Function = It.second;

    // If the function was emitted, its line info was emitted with it.
    if (Function.isEmitted())
      continue;

    const DWARFDebugLine::LineTable *LineTable = Function.getDWARFLineTable();
    if (!LineTable)
      continue; // nothing to update for this function

    const uint64_t Address = Function.getAddress();
    std::vector<uint32_t> Results;
    if (!LineTable->lookupAddressRange(
            {Address, object::SectionedAddress::UndefSection},
            Function.getSize(), Results))
      continue;

    if (Results.empty())
      continue;

    // The first row returned could be the last row matching the start address.
    // Find the first row with the same address that is not the end of the
    // sequence.
    uint64_t FirstRow = Results.front();
    while (FirstRow > 0) {
      const DWARFDebugLine::Row &PrevRow = LineTable->Rows[FirstRow - 1];
      if (PrevRow.Address.Address != Address || PrevRow.EndSequence)
        break;
      --FirstRow;
    }

    const uint64_t EndOfSequenceAddress =
        Function.getAddress() + Function.getMaxSize();
    BC.getDwarfLineTable(Function.getDWARFUnit()->getOffset())
        .addLineTableSequence(LineTable, FirstRow, Results.back(),
                              EndOfSequenceAddress);
  }

  // For units that are completely unprocessed, use original debug line contents
  // eliminating the need to regenerate line info program.
  emitDebugLineInfoForUnprocessedCUs();
}

void BinaryEmitter::emitDebugLineInfoForUnprocessedCUs() {
  // Sorted list of section offsets provides boundaries for section fragments,
  // where each fragment is the unit's contribution to debug line section.
  std::vector<uint64_t> StmtListOffsets;
  StmtListOffsets.reserve(BC.DwCtx->getNumCompileUnits());
  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    DWARFDie CUDie = CU->getUnitDIE();
    auto StmtList = dwarf::toSectionOffset(CUDie.find(dwarf::DW_AT_stmt_list));
    if (!StmtList)
      continue;

    StmtListOffsets.push_back(*StmtList);
  }
  llvm::sort(StmtListOffsets);

  // For each CU that was not processed, emit its line info as a binary blob.
  for (const std::unique_ptr<DWARFUnit> &CU : BC.DwCtx->compile_units()) {
    if (BC.ProcessedCUs.count(CU.get()))
      continue;

    DWARFDie CUDie = CU->getUnitDIE();
    auto StmtList = dwarf::toSectionOffset(CUDie.find(dwarf::DW_AT_stmt_list));
    if (!StmtList)
      continue;

    StringRef DebugLineContents = CU->getLineSection().Data;

    const uint64_t Begin = *StmtList;

    // Statement list ends where the next unit contribution begins, or at the
    // end of the section.
    auto It = llvm::upper_bound(StmtListOffsets, Begin);
    const uint64_t End =
        It == StmtListOffsets.end() ? DebugLineContents.size() : *It;

    BC.getDwarfLineTable(CU->getOffset())
        .addRawContents(DebugLineContents.slice(Begin, End));
  }
}

void BinaryEmitter::emitDataSections(StringRef OrgSecPrefix) {
  for (BinarySection &Section : BC.sections()) {
    if (!Section.hasRelocations() || !Section.hasSectionRef())
      continue;

    StringRef SectionName = Section.getName();
    std::string EmitName = Section.isReordered()
                               ? std::string(Section.getOutputName())
                               : OrgSecPrefix.str() + std::string(SectionName);
    Section.emitAsData(Streamer, EmitName);
    Section.clearRelocations();
  }
}

namespace llvm {
namespace bolt {

void emitBinaryContext(MCStreamer &Streamer, BinaryContext &BC,
                       StringRef OrgSecPrefix) {
  BinaryEmitter(Streamer, BC).emitAll(OrgSecPrefix);
}

void emitFunctionBody(MCStreamer &Streamer, BinaryFunction &BF,
                      FunctionFragment &FF, bool EmitCodeOnly) {
  BinaryEmitter(Streamer, BF.getBinaryContext())
      .emitFunctionBody(BF, FF, EmitCodeOnly);
}

} // namespace bolt
} // namespace llvm
