//===- GdbIndex.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#ifndef LLD_ELF_GDB_INDEX_H
#define LLD_ELF_GDB_INDEX_H

#include "InputFiles.h"
#include "llvm/Object/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"

namespace lld {
namespace elf {

class InputSection;

// Struct represents single entry of address area of gdb index.
struct AddressEntry {
  InputSectionBase *Section;
  uint64_t LowAddress;
  uint64_t HighAddress;
  size_t CuIndex;
};

// Element of GdbHashTab hash table.
struct GdbSymbol {
  GdbSymbol(uint32_t Hash, size_t Offset)
      : NameHash(Hash), NameOffset(Offset) {}
  uint32_t NameHash;
  size_t NameOffset;
  size_t CuVectorIndex;
};

// This class manages the hashed symbol table for the .gdb_index section.
// The hash value for a table entry is computed by applying an iterative hash
// function to the symbol's name.
class GdbHashTab final {
public:
  std::pair<bool, GdbSymbol *> add(uint32_t Hash, size_t Offset);

  void finalizeContents();
  size_t getCapacity() { return Table.size(); }
  GdbSymbol *getSymbol(size_t I) { return Table[I]; }

private:
  llvm::DenseMap<size_t, GdbSymbol *> Map;
  std::vector<GdbSymbol *> Table;
};

} // namespace elf
} // namespace lld

#endif
