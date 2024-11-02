//===- bolt/Core/BinaryData.cpp - Objects in a binary file ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BinaryData class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryData.h"
#include "bolt/Core/BinarySection.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
using namespace bolt;

#define DEBUG_TYPE "bolt"

namespace opts {
extern cl::OptionCategory BoltCategory;
extern cl::opt<unsigned> Verbosity;

cl::opt<bool>
    PrintSymbolAliases("print-aliases",
                       cl::desc("print aliases when printing objects"),
                       cl::Hidden, cl::cat(BoltCategory));
}

bool BinaryData::isAbsolute() const { return Flags & SymbolRef::SF_Absolute; }

bool BinaryData::isMoveable() const {
  return (!isAbsolute() && (IsMoveable && (!Parent || isTopLevelJumpTable())));
}

void BinaryData::merge(const BinaryData *Other) {
  assert(!Size || !Other->Size || Size == Other->Size);
  assert(Address == Other->Address);
  assert(*Section == *Other->Section);
  assert(OutputOffset == Other->OutputOffset);
  assert(OutputSection == Other->OutputSection);
  Symbols.insert(Symbols.end(), Other->Symbols.begin(), Other->Symbols.end());
  Flags |= Other->Flags;
  if (!Size)
    Size = Other->Size;
}

bool BinaryData::hasName(StringRef Name) const {
  for (const MCSymbol *Symbol : Symbols)
    if (Name == Symbol->getName())
      return true;
  return false;
}

bool BinaryData::hasNameRegex(StringRef NameRegex) const {
  Regex MatchName(NameRegex);
  for (const MCSymbol *Symbol : Symbols)
    if (MatchName.match(Symbol->getName()))
      return true;
  return false;
}

bool BinaryData::nameStartsWith(StringRef Prefix) const {
  for (const MCSymbol *Symbol : Symbols)
    if (Symbol->getName().starts_with(Prefix))
      return true;
  return false;
}

StringRef BinaryData::getSectionName() const { return getSection().getName(); }

StringRef BinaryData::getOutputSectionName() const {
  return getOutputSection().getName();
}

uint64_t BinaryData::getOutputAddress() const {
  assert(OutputSection->getOutputAddress());
  return OutputSection->getOutputAddress() + OutputOffset;
}

uint64_t BinaryData::getOffset() const {
  return Address - getSection().getAddress();
}

void BinaryData::setSection(BinarySection &NewSection) {
  if (OutputSection == Section)
    OutputSection = &NewSection;
  Section = &NewSection;
}

bool BinaryData::isMoved() const {
  return (getOffset() != OutputOffset || OutputSection != Section);
}

void BinaryData::print(raw_ostream &OS) const { printBrief(OS); }

void BinaryData::printBrief(raw_ostream &OS) const {
  OS << "(";

  if (isJumpTable())
    OS << "jump-table: ";
  else
    OS << "object: ";

  OS << getName();

  if ((opts::PrintSymbolAliases || opts::Verbosity > 1) && Symbols.size() > 1) {
    OS << ", aliases:";
    for (unsigned I = 1u; I < Symbols.size(); ++I) {
      OS << (I == 1 ? " (" : ", ") << Symbols[I]->getName();
    }
    OS << ")";
  }

  if (Parent) {
    OS << " (parent: ";
    Parent->printBrief(OS);
    OS << ")";
  }

  OS << ", 0x" << Twine::utohexstr(getAddress()) << ":0x"
     << Twine::utohexstr(getEndAddress()) << "/" << getSize() << "/"
     << getAlignment() << "/0x" << Twine::utohexstr(Flags);

  OS << ")";
}

BinaryData::BinaryData(MCSymbol &Symbol, uint64_t Address, uint64_t Size,
                       uint16_t Alignment, BinarySection &Section,
                       unsigned Flags)
    : Section(&Section), Address(Address), Size(Size), Alignment(Alignment),
      Flags(Flags), OutputSection(&Section), OutputOffset(getOffset()) {
  Symbols.push_back(&Symbol);
}
