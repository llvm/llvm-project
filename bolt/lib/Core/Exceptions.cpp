//===- bolt/Core/Exceptions.cpp - Helpers for C++ exceptions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions for handling C++ exception meta data.
//
// Some of the code is taken from examples/ExceptionDemo
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/Exceptions.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-exceptions"

using namespace llvm::dwarf;

namespace opts {

extern llvm::cl::OptionCategory BoltCategory;

extern llvm::cl::opt<unsigned> Verbosity;

static llvm::cl::opt<bool>
    PrintExceptions("print-exceptions",
                    llvm::cl::desc("print exception handling data"),
                    llvm::cl::Hidden, llvm::cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

// Read and dump the .gcc_exception_table section entry.
//
// .gcc_except_table section contains a set of Language-Specific Data Areas -
// a fancy name for exception handling tables. There's one  LSDA entry per
// function. However, we can't actually tell which function LSDA refers to
// unless we parse .eh_frame entry that refers to the LSDA.
// Then inside LSDA most addresses are encoded relative to the function start,
// so we need the function context in order to get to real addresses.
//
// The best visual representation of the tables comprising LSDA and
// relationships between them is illustrated at:
//   https://github.com/itanium-cxx-abi/cxx-abi/blob/master/exceptions.pdf
// Keep in mind that GCC implementation deviates slightly from that document.
//
// To summarize, there are 4 tables in LSDA: call site table, actions table,
// types table, and types index table (for indirection). The main table contains
// call site entries. Each call site includes a PC range that can throw an
// exception, a handler (landing pad), and a reference to an entry in the action
// table. The handler and/or action could be 0. The action entry is a head
// of a list of actions associated with a call site. The action table contains
// all such lists (it could be optimized to share list tails). Each action could
// be either to catch an exception of a given type, to perform a cleanup, or to
// propagate the exception after filtering it out (e.g. to make sure function
// exception specification is not violated). Catch action contains a reference
// to an entry in the type table, and filter action refers to an entry in the
// type index table to encode a set of types to filter.
//
// Call site table follows LSDA header. Action table immediately follows the
// call site table.
//
// Both types table and type index table start at the same location, but they
// grow in opposite directions (types go up, indices go down). The beginning of
// these tables is encoded in LSDA header. Sizes for both of the tables are not
// included anywhere.
//
// We have to parse all of the tables to determine their sizes. Then we have
// to parse the call site table and associate discovered information with
// actual call instructions and landing pad blocks.
//
// For the purpose of rewriting exception handling tables, we can reuse action,
// and type index tables in their original binary format.
//
// Type table could be encoded using position-independent references, and thus
// may require relocation.
//
// Ideally we should be able to re-write LSDA in-place, without the need to
// allocate a new space for it. Sadly there's no guarantee that the new call
// site table will be the same size as GCC uses uleb encodings for PC offsets.
//
// Note: some functions have LSDA entries with 0 call site entries.
void BinaryFunction::parseLSDA(ArrayRef<uint8_t> LSDASectionData,
                               uint64_t LSDASectionAddress) {
  assert(CurrentState == State::Disassembled && "unexpected function state");

  if (!getLSDAAddress())
    return;

  DWARFDataExtractor Data(
      StringRef(reinterpret_cast<const char *>(LSDASectionData.data()),
                LSDASectionData.size()),
      BC.DwCtx->getDWARFObj().isLittleEndian(), 8);
  uint64_t Offset = getLSDAAddress() - LSDASectionAddress;
  assert(Data.isValidOffset(Offset) && "wrong LSDA address");

  uint8_t LPStartEncoding = Data.getU8(&Offset);
  uint64_t LPStart = 0;
  // Convert to offset if LPStartEncoding is typed absptr DW_EH_PE_absptr
  if (std::optional<uint64_t> MaybeLPStart = Data.getEncodedPointer(
          &Offset, LPStartEncoding, Offset + LSDASectionAddress))
    LPStart = (LPStartEncoding && 0xFF == 0) ? *MaybeLPStart
                                             : *MaybeLPStart - Address;

  const uint8_t TTypeEncoding = Data.getU8(&Offset);
  LSDATypeEncoding = TTypeEncoding;
  size_t TTypeEncodingSize = 0;
  uintptr_t TTypeEnd = 0;
  if (TTypeEncoding != DW_EH_PE_omit) {
    TTypeEnd = Data.getULEB128(&Offset);
    TTypeEncodingSize = BC.getDWARFEncodingSize(TTypeEncoding);
  }

  if (opts::PrintExceptions) {
    outs() << "[LSDA at 0x" << Twine::utohexstr(getLSDAAddress())
           << " for function " << *this << "]:\n";
    outs() << "LPStart Encoding = 0x" << Twine::utohexstr(LPStartEncoding)
           << '\n';
    outs() << "LPStart = 0x" << Twine::utohexstr(LPStart) << '\n';
    outs() << "TType Encoding = 0x" << Twine::utohexstr(TTypeEncoding) << '\n';
    outs() << "TType End = " << TTypeEnd << '\n';
  }

  // Table to store list of indices in type table. Entries are uleb128 values.
  const uint64_t TypeIndexTableStart = Offset + TTypeEnd;

  // Offset past the last decoded index.
  uint64_t MaxTypeIndexTableOffset = 0;

  // Max positive index used in type table.
  unsigned MaxTypeIndex = 0;

  // The actual type info table starts at the same location, but grows in
  // opposite direction. TTypeEncoding is used to encode stored values.
  const uint64_t TypeTableStart = Offset + TTypeEnd;

  uint8_t CallSiteEncoding = Data.getU8(&Offset);
  uint32_t CallSiteTableLength = Data.getULEB128(&Offset);
  uint64_t CallSiteTableStart = Offset;
  uint64_t CallSiteTableEnd = CallSiteTableStart + CallSiteTableLength;
  uint64_t CallSitePtr = CallSiteTableStart;
  uint64_t ActionTableStart = CallSiteTableEnd;

  if (opts::PrintExceptions) {
    outs() << "CallSite Encoding = " << (unsigned)CallSiteEncoding << '\n';
    outs() << "CallSite table length = " << CallSiteTableLength << '\n';
    outs() << '\n';
  }

  this->HasEHRanges = CallSitePtr < CallSiteTableEnd;
  const uint64_t RangeBase = getAddress();
  while (CallSitePtr < CallSiteTableEnd) {
    uint64_t Start = *Data.getEncodedPointer(&CallSitePtr, CallSiteEncoding,
                                             CallSitePtr + LSDASectionAddress);
    uint64_t Length = *Data.getEncodedPointer(&CallSitePtr, CallSiteEncoding,
                                              CallSitePtr + LSDASectionAddress);
    uint64_t LandingPad = *Data.getEncodedPointer(
        &CallSitePtr, CallSiteEncoding, CallSitePtr + LSDASectionAddress);
    uint64_t ActionEntry = Data.getULEB128(&CallSitePtr);

    uint64_t LPOffset = LPStart + LandingPad;
    uint64_t LPAddress = Address + LPOffset;

    // Verify if landing pad code is located outside current function
    // Support landing pad to builtin_unreachable
    if (LPAddress < Address || LPAddress > Address + getSize()) {
      BinaryFunction *Fragment =
          BC.getBinaryFunctionContainingAddress(LPAddress);
      assert(Fragment != nullptr &&
             "BOLT-ERROR: cannot find landing pad fragment");
      BC.addInterproceduralReference(this, Fragment->getAddress());
      BC.processInterproceduralReferences();
      assert((isChildOf(*Fragment) || Fragment->isChildOf(*this)) &&
             "BOLT-ERROR: cannot have landing pads in different "
             "functions");
      setHasIndirectTargetToSplitFragment(true);
      BC.addFragmentsToSkip(this);
      return;
    }

    if (opts::PrintExceptions) {
      outs() << "Call Site: [0x" << Twine::utohexstr(RangeBase + Start)
             << ", 0x" << Twine::utohexstr(RangeBase + Start + Length)
             << "); landing pad: 0x" << Twine::utohexstr(LPOffset)
             << "; action entry: 0x" << Twine::utohexstr(ActionEntry) << "\n";
      outs() << "  current offset is " << (CallSitePtr - CallSiteTableStart)
             << '\n';
    }

    // Create a handler entry if necessary.
    MCSymbol *LPSymbol = nullptr;
    if (LPOffset) {
      if (!getInstructionAtOffset(LPOffset)) {
        if (opts::Verbosity >= 1)
          errs() << "BOLT-WARNING: landing pad " << Twine::utohexstr(LPOffset)
                 << " not pointing to an instruction in function " << *this
                 << " - ignoring.\n";
      } else {
        auto Label = Labels.find(LPOffset);
        if (Label != Labels.end()) {
          LPSymbol = Label->second;
        } else {
          LPSymbol = BC.Ctx->createNamedTempSymbol("LP");
          Labels[LPOffset] = LPSymbol;
        }
      }
    }

    // Mark all call instructions in the range.
    auto II = Instructions.find(Start);
    auto IE = Instructions.end();
    assert(II != IE && "exception range not pointing to an instruction");
    do {
      MCInst &Instruction = II->second;
      if (BC.MIB->isCall(Instruction) &&
          !BC.MIB->getConditionalTailCall(Instruction)) {
        assert(!BC.MIB->isInvoke(Instruction) &&
               "overlapping exception ranges detected");
        // Add extra operands to a call instruction making it an invoke from
        // now on.
        BC.MIB->addEHInfo(Instruction,
                          MCPlus::MCLandingPad(LPSymbol, ActionEntry));
      }
      ++II;
    } while (II != IE && II->first < Start + Length);

    if (ActionEntry != 0) {
      auto printType = [&](int Index, raw_ostream &OS) {
        assert(Index > 0 && "only positive indices are valid");
        uint64_t TTEntry = TypeTableStart - Index * TTypeEncodingSize;
        const uint64_t TTEntryAddress = TTEntry + LSDASectionAddress;
        uint64_t TypeAddress =
            *Data.getEncodedPointer(&TTEntry, TTypeEncoding, TTEntryAddress);
        if ((TTypeEncoding & DW_EH_PE_pcrel) && TypeAddress == TTEntryAddress)
          TypeAddress = 0;
        if (TypeAddress == 0) {
          OS << "<all>";
          return;
        }
        if (TTypeEncoding & DW_EH_PE_indirect) {
          ErrorOr<uint64_t> PointerOrErr = BC.getPointerAtAddress(TypeAddress);
          assert(PointerOrErr && "failed to decode indirect address");
          TypeAddress = *PointerOrErr;
        }
        if (BinaryData *TypeSymBD = BC.getBinaryDataAtAddress(TypeAddress))
          OS << TypeSymBD->getName();
        else
          OS << "0x" << Twine::utohexstr(TypeAddress);
      };
      if (opts::PrintExceptions)
        outs() << "    actions: ";
      uint64_t ActionPtr = ActionTableStart + ActionEntry - 1;
      int64_t ActionType;
      int64_t ActionNext;
      const char *Sep = "";
      do {
        ActionType = Data.getSLEB128(&ActionPtr);
        const uint32_t Self = ActionPtr;
        ActionNext = Data.getSLEB128(&ActionPtr);
        if (opts::PrintExceptions)
          outs() << Sep << "(" << ActionType << ", " << ActionNext << ") ";
        if (ActionType == 0) {
          if (opts::PrintExceptions)
            outs() << "cleanup";
        } else if (ActionType > 0) {
          // It's an index into a type table.
          MaxTypeIndex =
              std::max(MaxTypeIndex, static_cast<unsigned>(ActionType));
          if (opts::PrintExceptions) {
            outs() << "catch type ";
            printType(ActionType, outs());
          }
        } else { // ActionType < 0
          if (opts::PrintExceptions)
            outs() << "filter exception types ";
          const char *TSep = "";
          // ActionType is a negative *byte* offset into *uleb128-encoded* table
          // of indices with base 1.
          // E.g. -1 means offset 0, -2 is offset 1, etc. The indices are
          // encoded using uleb128 thus we cannot directly dereference them.
          uint64_t TypeIndexTablePtr = TypeIndexTableStart - ActionType - 1;
          while (uint64_t Index = Data.getULEB128(&TypeIndexTablePtr)) {
            MaxTypeIndex = std::max(MaxTypeIndex, static_cast<unsigned>(Index));
            if (opts::PrintExceptions) {
              outs() << TSep;
              printType(Index, outs());
              TSep = ", ";
            }
          }
          MaxTypeIndexTableOffset = std::max(
              MaxTypeIndexTableOffset, TypeIndexTablePtr - TypeIndexTableStart);
        }

        Sep = "; ";

        ActionPtr = Self + ActionNext;
      } while (ActionNext);
      if (opts::PrintExceptions)
        outs() << '\n';
    }
  }
  if (opts::PrintExceptions)
    outs() << '\n';

  assert(TypeIndexTableStart + MaxTypeIndexTableOffset <=
             Data.getData().size() &&
         "LSDA entry has crossed section boundary");

  if (TTypeEnd) {
    LSDAActionTable = LSDASectionData.slice(
        ActionTableStart, TypeIndexTableStart -
                              MaxTypeIndex * TTypeEncodingSize -
                              ActionTableStart);
    for (unsigned Index = 1; Index <= MaxTypeIndex; ++Index) {
      uint64_t TTEntry = TypeTableStart - Index * TTypeEncodingSize;
      const uint64_t TTEntryAddress = TTEntry + LSDASectionAddress;
      uint64_t TypeAddress =
          *Data.getEncodedPointer(&TTEntry, TTypeEncoding, TTEntryAddress);
      if ((TTypeEncoding & DW_EH_PE_pcrel) && (TypeAddress == TTEntryAddress))
        TypeAddress = 0;
      if (TTypeEncoding & DW_EH_PE_indirect) {
        LSDATypeAddressTable.emplace_back(TypeAddress);
        if (TypeAddress) {
          ErrorOr<uint64_t> PointerOrErr = BC.getPointerAtAddress(TypeAddress);
          assert(PointerOrErr && "failed to decode indirect address");
          TypeAddress = *PointerOrErr;
        }
      }
      LSDATypeTable.emplace_back(TypeAddress);
    }
    LSDATypeIndexTable =
        LSDASectionData.slice(TypeIndexTableStart, MaxTypeIndexTableOffset);
  }
}

void BinaryFunction::updateEHRanges() {
  if (getSize() == 0)
    return;

  assert(CurrentState == State::CFG_Finalized && "unexpected state");

  // Build call sites table.
  struct EHInfo {
    const MCSymbol *LP; // landing pad
    uint64_t Action;
  };

  // Sites to update.
  CallSitesList Sites;

  for (FunctionFragment &FF : getLayout().fragments()) {
    // If previous call can throw, this is its exception handler.
    EHInfo PreviousEH = {nullptr, 0};

    // Marker for the beginning of exceptions range.
    const MCSymbol *StartRange = nullptr;

    for (BinaryBasicBlock *const BB : FF) {
      for (auto II = BB->begin(); II != BB->end(); ++II) {
        if (!BC.MIB->isCall(*II))
          continue;

        // Instruction can throw an exception that should be handled.
        const bool Throws = BC.MIB->isInvoke(*II);

        // Ignore the call if it's a continuation of a no-throw gap.
        if (!Throws && !StartRange)
          continue;

        // Extract exception handling information from the instruction.
        const MCSymbol *LP = nullptr;
        uint64_t Action = 0;
        if (const std::optional<MCPlus::MCLandingPad> EHInfo =
                BC.MIB->getEHInfo(*II))
          std::tie(LP, Action) = *EHInfo;

        // No action if the exception handler has not changed.
        if (Throws && StartRange && PreviousEH.LP == LP &&
            PreviousEH.Action == Action)
          continue;

        // Same symbol is used for the beginning and the end of the range.
        const MCSymbol *EHSymbol;
        MCInst EHLabel;
        {
          std::unique_lock<llvm::sys::RWMutex> Lock(BC.CtxMutex);
          EHSymbol = BC.Ctx->createNamedTempSymbol("EH");
          BC.MIB->createEHLabel(EHLabel, EHSymbol, BC.Ctx.get());
        }

        II = std::next(BB->insertPseudoInstr(II, EHLabel));

        // At this point we could be in one of the following states:
        //
        // I. Exception handler has changed and we need to close previous range
        //    and start a new one.
        //
        // II. Start a new exception range after the gap.
        //
        // III. Close current exception range and start a new gap.
        const MCSymbol *EndRange;
        if (StartRange) {
          // I, III:
          EndRange = EHSymbol;
        } else {
          // II:
          StartRange = EHSymbol;
          EndRange = nullptr;
        }

        // Close the previous range.
        if (EndRange)
          Sites.emplace_back(
              FF.getFragmentNum(),
              CallSite{StartRange, EndRange, PreviousEH.LP, PreviousEH.Action});

        if (Throws) {
          // I, II:
          StartRange = EHSymbol;
          PreviousEH = EHInfo{LP, Action};
        } else {
          StartRange = nullptr;
        }
      }
    }

    // Check if we need to close the range.
    if (StartRange) {
      const MCSymbol *EndRange = getFunctionEndLabel(FF.getFragmentNum());
      Sites.emplace_back(
          FF.getFragmentNum(),
          CallSite{StartRange, EndRange, PreviousEH.LP, PreviousEH.Action});
    }
  }

  addCallSites(Sites);
}

const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;

CFIReaderWriter::CFIReaderWriter(const DWARFDebugFrame &EHFrame) {
  // Prepare FDEs for fast lookup
  for (const dwarf::FrameEntry &Entry : EHFrame.entries()) {
    const auto *CurFDE = dyn_cast<dwarf::FDE>(&Entry);
    // Skip CIEs.
    if (!CurFDE)
      continue;
    // There could me multiple FDEs with the same initial address, and perhaps
    // different sizes (address ranges). Use the first entry with non-zero size.
    auto FDEI = FDEs.lower_bound(CurFDE->getInitialLocation());
    if (FDEI != FDEs.end() && FDEI->first == CurFDE->getInitialLocation()) {
      if (CurFDE->getAddressRange()) {
        if (FDEI->second->getAddressRange() == 0) {
          FDEI->second = CurFDE;
        } else if (opts::Verbosity > 0) {
          errs() << "BOLT-WARNING: different FDEs for function at 0x"
                 << Twine::utohexstr(FDEI->first)
                 << " detected; sizes: " << FDEI->second->getAddressRange()
                 << " and " << CurFDE->getAddressRange() << '\n';
        }
      }
    } else {
      FDEs.emplace_hint(FDEI, CurFDE->getInitialLocation(), CurFDE);
    }
  }
}

bool CFIReaderWriter::fillCFIInfoFor(BinaryFunction &Function) const {
  uint64_t Address = Function.getAddress();
  auto I = FDEs.find(Address);
  // Ignore zero-length FDE ranges.
  if (I == FDEs.end() || !I->second->getAddressRange())
    return true;

  const FDE &CurFDE = *I->second;
  std::optional<uint64_t> LSDA = CurFDE.getLSDAAddress();
  Function.setLSDAAddress(LSDA ? *LSDA : 0);

  uint64_t Offset = Function.getFirstInstructionOffset();
  uint64_t CodeAlignment = CurFDE.getLinkedCIE()->getCodeAlignmentFactor();
  uint64_t DataAlignment = CurFDE.getLinkedCIE()->getDataAlignmentFactor();
  if (CurFDE.getLinkedCIE()->getPersonalityAddress()) {
    Function.setPersonalityFunction(
        *CurFDE.getLinkedCIE()->getPersonalityAddress());
    Function.setPersonalityEncoding(
        *CurFDE.getLinkedCIE()->getPersonalityEncoding());
  }

  auto decodeFrameInstruction = [&Function, &Offset, Address, CodeAlignment,
                                 DataAlignment](
                                    const CFIProgram::Instruction &Instr) {
    uint8_t Opcode = Instr.Opcode;
    if (Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK)
      Opcode &= DWARF_CFI_PRIMARY_OPCODE_MASK;
    switch (Instr.Opcode) {
    case DW_CFA_nop:
      break;
    case DW_CFA_advance_loc4:
    case DW_CFA_advance_loc2:
    case DW_CFA_advance_loc1:
    case DW_CFA_advance_loc:
      // Advance our current address
      Offset += CodeAlignment * int64_t(Instr.Ops[0]);
      break;
    case DW_CFA_offset_extended_sf:
      Function.addCFIInstruction(
          Offset,
          MCCFIInstruction::createOffset(
              nullptr, Instr.Ops[0], DataAlignment * int64_t(Instr.Ops[1])));
      break;
    case DW_CFA_offset_extended:
    case DW_CFA_offset:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createOffset(nullptr, Instr.Ops[0],
                                                 DataAlignment * Instr.Ops[1]));
      break;
    case DW_CFA_restore_extended:
    case DW_CFA_restore:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createRestore(nullptr, Instr.Ops[0]));
      break;
    case DW_CFA_set_loc:
      assert(Instr.Ops[0] >= Address && "set_loc out of function bounds");
      assert(Instr.Ops[0] <= Address + Function.getSize() &&
             "set_loc out of function bounds");
      Offset = Instr.Ops[0] - Address;
      break;

    case DW_CFA_undefined:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createUndefined(nullptr, Instr.Ops[0]));
      break;
    case DW_CFA_same_value:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createSameValue(nullptr, Instr.Ops[0]));
      break;
    case DW_CFA_register:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createRegister(nullptr, Instr.Ops[0],
                                                   Instr.Ops[1]));
      break;
    case DW_CFA_remember_state:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createRememberState(nullptr));
      break;
    case DW_CFA_restore_state:
      Function.addCFIInstruction(Offset,
                                 MCCFIInstruction::createRestoreState(nullptr));
      break;
    case DW_CFA_def_cfa:
      Function.addCFIInstruction(
          Offset,
          MCCFIInstruction::cfiDefCfa(nullptr, Instr.Ops[0], Instr.Ops[1]));
      break;
    case DW_CFA_def_cfa_sf:
      Function.addCFIInstruction(
          Offset,
          MCCFIInstruction::cfiDefCfa(nullptr, Instr.Ops[0],
                                      DataAlignment * int64_t(Instr.Ops[1])));
      break;
    case DW_CFA_def_cfa_register:
      Function.addCFIInstruction(Offset, MCCFIInstruction::createDefCfaRegister(
                                             nullptr, Instr.Ops[0]));
      break;
    case DW_CFA_def_cfa_offset:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::cfiDefCfaOffset(nullptr, Instr.Ops[0]));
      break;
    case DW_CFA_def_cfa_offset_sf:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::cfiDefCfaOffset(
                      nullptr, DataAlignment * int64_t(Instr.Ops[0])));
      break;
    case DW_CFA_GNU_args_size:
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createGnuArgsSize(nullptr, Instr.Ops[0]));
      Function.setUsesGnuArgsSize();
      break;
    case DW_CFA_val_offset_sf:
    case DW_CFA_val_offset:
      if (opts::Verbosity >= 1) {
        errs() << "BOLT-WARNING: DWARF val_offset() unimplemented\n";
      }
      return false;
    case DW_CFA_def_cfa_expression:
    case DW_CFA_val_expression:
    case DW_CFA_expression: {
      StringRef ExprBytes = Instr.Expression->getData();
      std::string Str;
      raw_string_ostream OS(Str);
      // Manually encode this instruction using CFI escape
      OS << Opcode;
      if (Opcode != DW_CFA_def_cfa_expression)
        encodeULEB128(Instr.Ops[0], OS);
      encodeULEB128(ExprBytes.size(), OS);
      OS << ExprBytes;
      Function.addCFIInstruction(
          Offset, MCCFIInstruction::createEscape(nullptr, OS.str()));
      break;
    }
    case DW_CFA_MIPS_advance_loc8:
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: DW_CFA_MIPS_advance_loc unimplemented\n";
      return false;
    case DW_CFA_GNU_window_save:
      // DW_CFA_GNU_window_save and DW_CFA_GNU_NegateRAState just use the same
      // id but mean different things. The latter is used in AArch64.
      if (Function.getBinaryContext().isAArch64()) {
        Function.addCFIInstruction(
            Offset, MCCFIInstruction::createNegateRAState(nullptr));
        break;
      }
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: DW_CFA_GNU_window_save unimplemented\n";
      return false;
    case DW_CFA_lo_user:
    case DW_CFA_hi_user:
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: DW_CFA_*_user unimplemented\n";
      return false;
    default:
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: Unrecognized CFI instruction: " << Instr.Opcode
               << '\n';
      return false;
    }

    return true;
  };

  for (const CFIProgram::Instruction &Instr : CurFDE.getLinkedCIE()->cfis())
    if (!decodeFrameInstruction(Instr))
      return false;

  for (const CFIProgram::Instruction &Instr : CurFDE.cfis())
    if (!decodeFrameInstruction(Instr))
      return false;

  return true;
}

std::vector<char> CFIReaderWriter::generateEHFrameHeader(
    const DWARFDebugFrame &OldEHFrame, const DWARFDebugFrame &NewEHFrame,
    uint64_t EHFrameHeaderAddress,
    std::vector<uint64_t> &FailedAddresses) const {
  // Common PC -> FDE map to be written into .eh_frame_hdr.
  std::map<uint64_t, uint64_t> PCToFDE;

  // Presort array for binary search.
  llvm::sort(FailedAddresses);

  // Initialize PCToFDE using NewEHFrame.
  for (dwarf::FrameEntry &Entry : NewEHFrame.entries()) {
    const dwarf::FDE *FDE = dyn_cast<dwarf::FDE>(&Entry);
    if (FDE == nullptr)
      continue;
    const uint64_t FuncAddress = FDE->getInitialLocation();
    const uint64_t FDEAddress =
        NewEHFrame.getEHFrameAddress() + FDE->getOffset();

    // Ignore unused FDEs.
    if (FuncAddress == 0)
      continue;

    // Add the address to the map unless we failed to write it.
    if (!std::binary_search(FailedAddresses.begin(), FailedAddresses.end(),
                            FuncAddress)) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: FDE for function at 0x"
                        << Twine::utohexstr(FuncAddress) << " is at 0x"
                        << Twine::utohexstr(FDEAddress) << '\n');
      PCToFDE[FuncAddress] = FDEAddress;
    }
  };

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: new .eh_frame contains "
                    << llvm::size(NewEHFrame.entries()) << " entries\n");

  // Add entries from the original .eh_frame corresponding to the functions
  // that we did not update.
  for (const dwarf::FrameEntry &Entry : OldEHFrame) {
    const dwarf::FDE *FDE = dyn_cast<dwarf::FDE>(&Entry);
    if (FDE == nullptr)
      continue;
    const uint64_t FuncAddress = FDE->getInitialLocation();
    const uint64_t FDEAddress =
        OldEHFrame.getEHFrameAddress() + FDE->getOffset();

    // Add the address if we failed to write it.
    if (PCToFDE.count(FuncAddress) == 0) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: old FDE for function at 0x"
                        << Twine::utohexstr(FuncAddress) << " is at 0x"
                        << Twine::utohexstr(FDEAddress) << '\n');
      PCToFDE[FuncAddress] = FDEAddress;
    }
  };

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: old .eh_frame contains "
                    << llvm::size(OldEHFrame.entries()) << " entries\n");

  // Generate a new .eh_frame_hdr based on the new map.

  // Header plus table of entries of size 8 bytes.
  std::vector<char> EHFrameHeader(12 + PCToFDE.size() * 8);

  // Version is 1.
  EHFrameHeader[0] = 1;
  // Encoding of the eh_frame pointer.
  EHFrameHeader[1] = DW_EH_PE_pcrel | DW_EH_PE_sdata4;
  // Encoding of the count field to follow.
  EHFrameHeader[2] = DW_EH_PE_udata4;
  // Encoding of the table entries - 4-byte offset from the start of the header.
  EHFrameHeader[3] = DW_EH_PE_datarel | DW_EH_PE_sdata4;

  // Address of eh_frame. Use the new one.
  support::ulittle32_t::ref(EHFrameHeader.data() + 4) =
      NewEHFrame.getEHFrameAddress() - (EHFrameHeaderAddress + 4);

  // Number of entries in the table (FDE count).
  support::ulittle32_t::ref(EHFrameHeader.data() + 8) = PCToFDE.size();

  // Write the table at offset 12.
  char *Ptr = EHFrameHeader.data();
  uint32_t Offset = 12;
  for (const auto &PCI : PCToFDE) {
    int64_t InitialPCOffset = PCI.first - EHFrameHeaderAddress;
    assert(isInt<32>(InitialPCOffset) && "PC offset out of bounds");
    support::ulittle32_t::ref(Ptr + Offset) = InitialPCOffset;
    Offset += 4;
    int64_t FDEOffset = PCI.second - EHFrameHeaderAddress;
    assert(isInt<32>(FDEOffset) && "FDE offset out of bounds");
    support::ulittle32_t::ref(Ptr + Offset) = FDEOffset;
    Offset += 4;
  }

  return EHFrameHeader;
}

Error EHFrameParser::parseCIE(uint64_t StartOffset) {
  uint8_t Version = Data.getU8(&Offset);
  const char *Augmentation = Data.getCStr(&Offset);
  StringRef AugmentationString(Augmentation ? Augmentation : "");
  uint8_t AddressSize =
      Version < 4 ? Data.getAddressSize() : Data.getU8(&Offset);
  Data.setAddressSize(AddressSize);
  // Skip segment descriptor size
  if (Version >= 4)
    Offset += 1;
  // Skip code alignment factor
  Data.getULEB128(&Offset);
  // Skip data alignment
  Data.getSLEB128(&Offset);
  // Skip return address register
  if (Version == 1)
    Offset += 1;
  else
    Data.getULEB128(&Offset);

  uint32_t FDEPointerEncoding = DW_EH_PE_absptr;
  uint32_t LSDAPointerEncoding = DW_EH_PE_omit;
  // Walk the augmentation string to get all the augmentation data.
  for (unsigned i = 0, e = AugmentationString.size(); i != e; ++i) {
    switch (AugmentationString[i]) {
    default:
      return createStringError(
          errc::invalid_argument,
          "unknown augmentation character in entry at 0x%" PRIx64, StartOffset);
    case 'L':
      LSDAPointerEncoding = Data.getU8(&Offset);
      break;
    case 'P': {
      uint32_t PersonalityEncoding = Data.getU8(&Offset);
      std::optional<uint64_t> Personality =
          Data.getEncodedPointer(&Offset, PersonalityEncoding,
                                 EHFrameAddress ? EHFrameAddress + Offset : 0);
      // Patch personality address
      if (Personality)
        PatcherCallback(*Personality, Offset, PersonalityEncoding);
      break;
    }
    case 'R':
      FDEPointerEncoding = Data.getU8(&Offset);
      break;
    case 'z':
      if (i)
        return createStringError(
            errc::invalid_argument,
            "'z' must be the first character at 0x%" PRIx64, StartOffset);
      // Skip augmentation length
      Data.getULEB128(&Offset);
      break;
    case 'S':
    case 'B':
      break;
    }
  }
  Entries.emplace_back(std::make_unique<CIEInfo>(
      FDEPointerEncoding, LSDAPointerEncoding, AugmentationString));
  CIEs[StartOffset] = &*Entries.back();
  return Error::success();
}

Error EHFrameParser::parseFDE(uint64_t CIEPointer,
                              uint64_t StartStructureOffset) {
  std::optional<uint64_t> LSDAAddress;
  CIEInfo *Cie = CIEs[StartStructureOffset - CIEPointer];

  // The address size is encoded in the CIE we reference.
  if (!Cie)
    return createStringError(errc::invalid_argument,
                             "parsing FDE data at 0x%" PRIx64
                             " failed due to missing CIE",
                             StartStructureOffset);
  // Patch initial location
  if (auto Val = Data.getEncodedPointer(&Offset, Cie->FDEPtrEncoding,
                                        EHFrameAddress + Offset)) {
    PatcherCallback(*Val, Offset, Cie->FDEPtrEncoding);
  }
  // Skip address range
  Data.getEncodedPointer(&Offset, Cie->FDEPtrEncoding, 0);

  // Process augmentation data for this FDE.
  StringRef AugmentationString = Cie->AugmentationString;
  if (!AugmentationString.empty() && Cie->LSDAPtrEncoding != DW_EH_PE_omit) {
    // Skip augmentation length
    Data.getULEB128(&Offset);
    LSDAAddress =
        Data.getEncodedPointer(&Offset, Cie->LSDAPtrEncoding,
                               EHFrameAddress ? Offset + EHFrameAddress : 0);
    // Patch LSDA address
    PatcherCallback(*LSDAAddress, Offset, Cie->LSDAPtrEncoding);
  }
  return Error::success();
}

Error EHFrameParser::parse() {
  while (Data.isValidOffset(Offset)) {
    const uint64_t StartOffset = Offset;

    uint64_t Length;
    DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(&Offset);

    // If the Length is 0, then this CIE is a terminator
    if (Length == 0)
      break;

    const uint64_t StartStructureOffset = Offset;
    const uint64_t EndStructureOffset = Offset + Length;

    Error Err = Error::success();
    const uint64_t Id = Data.getRelocatedValue(4, &Offset,
                                               /*SectionIndex=*/nullptr, &Err);
    if (Err)
      return Err;

    if (!Id) {
      if (Error Err = parseCIE(StartOffset))
        return Err;
    } else {
      if (Error Err = parseFDE(Id, StartStructureOffset))
        return Err;
    }
    Offset = EndStructureOffset;
  }

  return Error::success();
}

Error EHFrameParser::parse(DWARFDataExtractor Data, uint64_t EHFrameAddress,
                           PatcherCallbackTy PatcherCallback) {
  EHFrameParser Parser(Data, EHFrameAddress, PatcherCallback);
  return Parser.parse();
}

} // namespace bolt
} // namespace llvm
