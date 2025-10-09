//===- bolt/Core/BinaryData.h - Objects in a binary file --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the BinaryData class, which represents
// an allocatable entity in a binary file, such as a data object, a jump table,
// or a function.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_BINARY_DATA_H
#define BOLT_CORE_BINARY_DATA_H

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace llvm {
namespace bolt {

class BinarySection;

/// \p BinaryData represents an indivisible part of a data section section.
/// BinaryData's may contain sub-components, e.g. jump tables but they are
/// considered to be part of the parent symbol in terms of divisibility and
/// reordering.
class BinaryData {
  friend class BinaryContext;
  /// Non-null if this BinaryData is contained in a larger BinaryData object,
  /// i.e. the start and end addresses are contained within another object.
  BinaryData *Parent{nullptr};

  // non-copyable
  BinaryData() = delete;
  BinaryData(const BinaryData &) = delete;
  BinaryData &operator=(const BinaryData &) = delete;

protected:
  /// All symbols associated with this data.
  std::vector<MCSymbol *> Symbols;

  /// Section this data belongs to.
  BinarySection *Section{nullptr};

  /// Start address of this symbol.
  uint64_t Address{0};
  /// Size of this data (can be 0).
  uint64_t Size{0};
  /// Alignment of this data.
  uint16_t Alignment{1};

  bool IsMoveable{true};

  /// Symbol flags (same as llvm::SymbolRef::Flags)
  unsigned Flags{0};

  /// Output section for this data if it has been moved from the original
  /// section.
  BinarySection *OutputSection{nullptr};

  /// The offset of this symbol in the output section.  This is different
  /// from \p Address - Section.getAddress() when the data has been reordered.
  uint64_t OutputOffset{0};

  BinaryData *getRootData() {
    BinaryData *BD = this;
    while (BD->Parent)
      BD = BD->Parent;
    return BD;
  }

public:
  BinaryData(BinaryData &&) = default;
  BinaryData(MCSymbol &Symbol, uint64_t Address, uint64_t Size,
             uint16_t Alignment, BinarySection &Section, unsigned Flags = 0);
  virtual ~BinaryData() {}

  virtual bool isJumpTable() const { return false; }
  virtual bool isObject() const { return !isJumpTable(); }
  virtual void merge(const BinaryData *Other);

  bool isTopLevelJumpTable() const {
    return (isJumpTable() &&
            (!Parent || (!Parent->Parent && Parent->isObject())));
  }

  // BinaryData that is considered atomic and potentially moveable.  All
  // MemInfo data and relocations should be wrt. to atomic data.
  bool isAtomic() const { return isTopLevelJumpTable() || !Parent; }

  iterator_range<std::vector<MCSymbol *>::const_iterator> symbols() const {
    return make_range(Symbols.begin(), Symbols.end());
  }

  StringRef getName() const { return getSymbol()->getName(); }

  MCSymbol *getSymbol() { return Symbols.front(); }
  const MCSymbol *getSymbol() const { return Symbols.front(); }

  const std::vector<MCSymbol *> &getSymbols() const { return Symbols; }
  std::vector<MCSymbol *> &getSymbols() { return Symbols; }

  bool hasName(StringRef Name) const;
  bool nameStartsWith(StringRef Prefix) const;

  bool hasSymbol(const MCSymbol *Symbol) const {
    return llvm::is_contained(Symbols, Symbol);
  }

  bool isAbsolute() const;
  bool isMoveable() const;

  uint64_t getAddress() const { return Address; }
  uint64_t getEndAddress() const { return Address + Size; }
  uint64_t getOffset() const;
  uint64_t getSize() const { return Size; }
  uint16_t getAlignment() const { return Alignment; }

  BinarySection &getSection() { return *Section; }
  const BinarySection &getSection() const { return *Section; }
  StringRef getSectionName() const;

  BinarySection &getOutputSection() { return *OutputSection; }
  const BinarySection &getOutputSection() const { return *OutputSection; }
  StringRef getOutputSectionName() const;
  uint64_t getOutputAddress() const;
  uint64_t getOutputOffset() const { return OutputOffset; }
  uint64_t getOutputSize() const { return Size; }

  bool isMoved() const;
  bool containsAddress(uint64_t Address) const {
    return ((getAddress() <= Address && Address < getEndAddress()) ||
            (getAddress() == Address && !getSize()));
  }
  bool containsRange(uint64_t Address, uint64_t Size) const {
    return containsAddress(Address) && Address + Size <= getEndAddress();
  }

  const BinaryData *getParent() const { return Parent; }

  const BinaryData *getRootData() const {
    const BinaryData *BD = this;
    while (BD->Parent)
      BD = BD->Parent;
    return BD;
  }

  BinaryData *getAtomicRoot() {
    BinaryData *BD = this;
    while (!BD->isAtomic() && BD->Parent)
      BD = BD->Parent;
    return BD;
  }

  const BinaryData *getAtomicRoot() const {
    const BinaryData *BD = this;
    while (!BD->isAtomic() && BD->Parent)
      BD = BD->Parent;
    return BD;
  }

  bool isAncestorOf(const BinaryData *BD) const {
    return Parent && (Parent == BD || Parent->isAncestorOf(BD));
  }

  void updateSize(uint64_t N) {
    if (N > Size)
      Size = N;
  }

  void setIsMoveable(bool Flag) { IsMoveable = Flag; }
  void setSection(BinarySection &NewSection);
  void setOutputSection(BinarySection &NewSection) {
    OutputSection = &NewSection;
  }
  void setOutputOffset(uint64_t Offset) { OutputOffset = Offset; }
  void setOutputLocation(BinarySection &NewSection, uint64_t NewOffset) {
    setOutputSection(NewSection);
    setOutputOffset(NewOffset);
  }

  virtual void printBrief(raw_ostream &OS) const;
  virtual void print(raw_ostream &OS) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const BinaryData &BD) {
  BD.printBrief(OS);
  return OS;
}

/// Address access info used for memory profiling.
struct AddressAccess {
  BinaryData *MemoryObject; /// Object accessed or nullptr
  uint64_t Offset;          /// Offset within the object or absolute address
  uint64_t Count;           /// Number of accesses
  bool operator==(const AddressAccess &Other) const {
    return MemoryObject == Other.MemoryObject && Offset == Other.Offset &&
           Count == Other.Count;
  }
};

/// Aggregated memory access info per instruction.
struct MemoryAccessProfile {
  uint64_t NextInstrOffset;
  SmallVector<AddressAccess, 4> AddressAccessInfo;
  bool operator==(const MemoryAccessProfile &Other) const {
    return NextInstrOffset == Other.NextInstrOffset &&
           AddressAccessInfo == Other.AddressAccessInfo;
  }
};

inline raw_ostream &operator<<(raw_ostream &OS,
                               const bolt::MemoryAccessProfile &MAP) {
  std::string TempString;
  raw_string_ostream SS(TempString);

  const char *Sep = "\n        ";
  uint64_t TotalCount = 0;
  for (const AddressAccess &AccessInfo : MAP.AddressAccessInfo) {
    SS << Sep << "{ ";
    if (AccessInfo.MemoryObject)
      SS << AccessInfo.MemoryObject->getName() << " + ";
    SS << "0x" << Twine::utohexstr(AccessInfo.Offset) << ": "
       << AccessInfo.Count << " }";
    Sep = ",\n        ";
    TotalCount += AccessInfo.Count;
  }

  OS << TotalCount << " total counts : " << TempString;
  return OS;
}

} // namespace bolt
} // namespace llvm

#endif
