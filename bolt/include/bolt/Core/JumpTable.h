//===- bolt/Core/JumpTable.h - Jump table at low-level IR -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the JumpTable class, which represents a jump table in a
// binary file.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_JUMP_TABLE_H
#define BOLT_CORE_JUMP_TABLE_H

#include "bolt/Core/BinaryData.h"
#include <map>
#include <vector>

namespace llvm {
class MCSymbol;
class raw_ostream;

namespace bolt {

enum JumpTableSupportLevel : char {
  JTS_NONE = 0,       /// Disable jump tables support.
  JTS_BASIC = 1,      /// Enable basic jump tables support (in-place).
  JTS_MOVE = 2,       /// Move jump tables to a separate section.
  JTS_SPLIT = 3,      /// Enable hot/cold splitting of jump tables.
  JTS_AGGRESSIVE = 4, /// Aggressive splitting of jump tables.
};

class BinaryFunction;

/// Representation of a jump table.
///
/// The jump table may include other jump tables that are referenced by
/// a different label at a different offset in this jump table.
class JumpTable : public BinaryData {
  friend class BinaryContext;

  JumpTable() = delete;
  JumpTable(const JumpTable &) = delete;
  JumpTable &operator=(const JumpTable &) = delete;

public:
  enum JumpTableType : char {
    JTT_NORMAL,
    JTT_PIC,
  };

  /// Branch statistics for jump table entries.
  struct JumpInfo {
    uint64_t Mispreds{0};
    uint64_t Count{0};
  };

  /// Size of the entry used for storage.
  size_t EntrySize;

  /// Size of the entry size we will write (we may use a more compact layout)
  size_t OutputEntrySize;

  /// The type of this jump table.
  JumpTableType Type;

  /// All the entries as labels.
  std::vector<MCSymbol *> Entries;

  /// All the entries as absolute addresses. Invalid after disassembly is done.
  using AddressesType = std::vector<uint64_t>;
  AddressesType EntriesAsAddress;

  /// Map <Offset> -> <Label> used for embedded jump tables. Label at 0 offset
  /// is the main label for the jump table.
  using LabelMapType = std::map<unsigned, MCSymbol *>;
  LabelMapType Labels;

  /// Dynamic number of times each entry in the table was referenced.
  /// Identical entries will have a shared count (identical for every
  /// entry in the set).
  std::vector<JumpInfo> Counts;

  /// Total number of times this jump table was used.
  uint64_t Count{0};

  /// BinaryFunction this jump tables belongs to.
  SmallVector<BinaryFunction *, 1> Parents;

private:
  /// Constructor should only be called by a BinaryContext.
  JumpTable(MCSymbol &Symbol, uint64_t Address, size_t EntrySize,
            JumpTableType Type, LabelMapType &&Labels, BinarySection &Section);

public:
  /// Return the size of the jump table.
  uint64_t getSize() const {
    return std::max(EntriesAsAddress.size(), Entries.size()) * EntrySize;
  }

  const MCSymbol *getFirstLabel() const {
    assert(Labels.count(0) != 0 && "labels must have an entry at 0");
    return Labels.find(0)->second;
  }

  /// Get the indexes for symbol entries that correspond to the jump table
  /// starting at (or containing) 'Addr'.
  std::pair<size_t, size_t> getEntriesForAddress(const uint64_t Addr) const;

  bool isJumpTable() const override { return true; }

  /// Change all entries of the jump table in \p JTAddress pointing to
  /// \p OldDest to \p NewDest. Return false if unsuccessful.
  bool replaceDestination(uint64_t JTAddress, const MCSymbol *OldDest,
                          MCSymbol *NewDest);

  /// Update jump table at its original location.
  void updateOriginal();

  /// Print for debugging purposes.
  void print(raw_ostream &OS) const override;
};

} // namespace bolt
} // namespace llvm

#endif
