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

private:
  // Method returns section file offset as a load addres for DWARF parser. That
  // allows to find the target section index for address ranges.
  uint64_t
  getSectionLoadAddress(const llvm::object::SectionRef &Sec) const override;
  std::unique_ptr<llvm::LoadedObjectInfo> clone() const override;
};

} // namespace elf
} // namespace lld

#endif
