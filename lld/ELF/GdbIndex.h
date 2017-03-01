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
class ObjInfoTy;

// Struct represents single entry of address area of gdb index.
template <class ELFT> struct AddressEntry {
  InputSectionBase *Section;
  uint64_t LowAddress;
  uint64_t HighAddress;
  size_t CuIndex;
};

// GdbIndexBuilder is a helper class used for extracting data required
// for building .gdb_index section from objects.
template <class ELFT> class GdbIndexBuilder {
  typedef typename ELFT::uint uintX_t;

  InputSection *DebugInfoSec;
  std::unique_ptr<llvm::DWARFContext> Dwarf;
  std::unique_ptr<ObjInfoTy> ObjInfo;

public:
  GdbIndexBuilder(InputSection *DebugInfoSec);
  ~GdbIndexBuilder();

  // Extracts the compilation units. Each first element of pair is a offset of a
  // CU in the .debug_info section and second is the length of that CU.
  std::vector<std::pair<uintX_t, uintX_t>> readCUList();

  // Extracts the vector of address area entries. Accepts global index of last
  // parsed CU.
  std::vector<AddressEntry<ELFT>> readAddressArea(size_t CurrentCU);

  // Method extracts public names and types. It returns list of name and
  // gnu_pub* kind pairs.
  std::vector<std::pair<StringRef, uint8_t>> readPubNamesAndTypes();
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
  GdbSymbol **findSlot(uint32_t Hash, size_t Offset);

  llvm::DenseMap<std::pair<uint32_t, size_t>, GdbSymbol *> Map;
  std::vector<GdbSymbol *> Table;

  // Size keeps the amount of filled entries in Table.
  size_t Size = 0;
};

} // namespace elf
} // namespace lld

#endif
