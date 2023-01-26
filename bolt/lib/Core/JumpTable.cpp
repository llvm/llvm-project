//===- bolt/Core/JumpTable.cpp - Jump table at low-level IR ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the JumpTable class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/JumpTable.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinarySection.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace bolt;

using JumpTable = bolt::JumpTable;

namespace opts {
extern cl::opt<JumpTableSupportLevel> JumpTables;
extern cl::opt<unsigned> Verbosity;
} // namespace opts

bolt::JumpTable::JumpTable(MCSymbol &Symbol, uint64_t Address, size_t EntrySize,
                           JumpTableType Type, LabelMapType &&Labels,
                           BinarySection &Section)
    : BinaryData(Symbol, Address, 0, EntrySize, Section), EntrySize(EntrySize),
      OutputEntrySize(EntrySize), Type(Type), Labels(Labels) {}

std::pair<size_t, size_t>
bolt::JumpTable::getEntriesForAddress(const uint64_t Addr) const {
  // Check if this is not an address, but a cloned JT id
  if ((int64_t)Addr < 0ll)
    return std::make_pair(0, Entries.size());

  const uint64_t InstOffset = Addr - getAddress();
  size_t StartIndex = 0, EndIndex = 0;
  uint64_t Offset = 0;

  for (size_t I = 0; I < Entries.size(); ++I) {
    auto LI = Labels.find(Offset);
    if (LI != Labels.end()) {
      const auto NextLI = std::next(LI);
      const uint64_t NextOffset =
          NextLI == Labels.end() ? getSize() : NextLI->first;
      if (InstOffset >= LI->first && InstOffset < NextOffset) {
        StartIndex = I;
        EndIndex = I;
        while (Offset < NextOffset) {
          ++EndIndex;
          Offset += EntrySize;
        }
        break;
      }
    }
    Offset += EntrySize;
  }

  return std::make_pair(StartIndex, EndIndex);
}

bool bolt::JumpTable::replaceDestination(uint64_t JTAddress,
                                         const MCSymbol *OldDest,
                                         MCSymbol *NewDest) {
  bool Patched = false;
  const std::pair<size_t, size_t> Range = getEntriesForAddress(JTAddress);
  for (auto I = Range.first; I != Range.second; ++I) {
    if (Entries[I] == OldDest) {
      Patched = true;
      Entries[I] = NewDest;
    }
  }
  return Patched;
}

void bolt::JumpTable::updateOriginal() {
  BinaryContext &BC = getSection().getBinaryContext();
  const uint64_t BaseOffset = getAddress() - getSection().getAddress();
  uint64_t EntryOffset = BaseOffset;
  for (MCSymbol *Entry : Entries) {
    const uint64_t RelType =
        Type == JTT_NORMAL ? ELF::R_X86_64_64 : ELF::R_X86_64_PC32;
    const uint64_t RelAddend =
        Type == JTT_NORMAL ? 0 : EntryOffset - BaseOffset;
    // Replace existing relocation with the new one to allow any modifications
    // to the original jump table.
    if (BC.HasRelocations)
      getOutputSection().removeRelocationAt(EntryOffset);
    getOutputSection().addRelocation(EntryOffset, Entry, RelType, RelAddend);
    EntryOffset += EntrySize;
  }
}

void bolt::JumpTable::print(raw_ostream &OS) const {
  uint64_t Offset = 0;
  if (Type == JTT_PIC)
    OS << "PIC ";
  ListSeparator LS;

  OS << "Jump table " << getName() << " for function ";
  for (BinaryFunction *Frag : Parents)
    OS << LS << *Frag;
  OS << " at 0x" << Twine::utohexstr(getAddress()) << " with a total count of "
     << Count << ":\n";
  for (const uint64_t EntryAddress : EntriesAsAddress)
    OS << "  absolute offset: 0x" << Twine::utohexstr(EntryAddress) << '\n';
  for (const MCSymbol *Entry : Entries) {
    auto LI = Labels.find(Offset);
    if (Offset && LI != Labels.end()) {
      OS << "Jump Table " << LI->second->getName() << " at 0x"
         << Twine::utohexstr(getAddress() + Offset)
         << " (possibly part of larger jump table):\n";
    }
    OS << format("  0x%04" PRIx64 " : ", Offset) << Entry->getName();
    if (!Counts.empty()) {
      OS << " : " << Counts[Offset / EntrySize].Mispreds << "/"
         << Counts[Offset / EntrySize].Count;
    }
    OS << '\n';
    Offset += EntrySize;
  }
  OS << "\n\n";
}
