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

template <class ELFT> class InputSection;

// Struct represents single entry of address area of gdb index.
template <class ELFT> struct AddressEntry {
  InputSectionBase<ELFT> *Section;
  uint64_t LowAddress;
  uint64_t HighAddress;
  size_t CuIndex;
};

// GdbIndexBuilder is a helper class used for extracting data required
// for building .gdb_index section from objects.
template <class ELFT> class GdbIndexBuilder : public llvm::LoadedObjectInfo {
  typedef typename ELFT::uint uintX_t;

  InputSection<ELFT> *DebugInfoSec;

  std::unique_ptr<llvm::DWARFContext> Dwarf;

public:
  GdbIndexBuilder(InputSection<ELFT> *DebugInfoSec);

  // Extracts the compilation units. Each first element of pair is a offset of a
  // CU in the .debug_info section and second is the length of that CU.
  std::vector<std::pair<uintX_t, uintX_t>> readCUList();

  // Extracts the vector of address area entries. Accepts global index of last
  // parsed CU.
  std::vector<AddressEntry<ELFT>> readAddressArea(size_t CurrentCU);

  // Method extracts public names and types. It returns list of name and
  // gnu_pub* kind pairs.
  std::vector<std::pair<StringRef, uint8_t>> readPubNamesAndTypes();

private:
  // Method returns section file offset as a load addres for DWARF parser. That
  // allows to find the target section index for address ranges.
  uint64_t
  getSectionLoadAddress(const llvm::object::SectionRef &Sec) const override;
  std::unique_ptr<llvm::LoadedObjectInfo> clone() const override;
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

  size_t getCapacity() { return Table.size(); }
  GdbSymbol *getSymbol(size_t I) { return Table[I]; }

private:
  void expand();

  GdbSymbol **findSlot(uint32_t Hash, size_t Offset);

  llvm::BumpPtrAllocator Alloc;
  std::vector<GdbSymbol *> Table;

  // Size keeps the amount of filled entries in Table.
  size_t Size = 0;

  // Initial size must be a power of 2.
  static const int32_t InitialSize = 1024;
};

} // namespace elf
} // namespace lld

#endif
