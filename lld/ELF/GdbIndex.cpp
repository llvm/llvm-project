//===- GdbIndex.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// File contains classes for implementation of --gdb-index command line option.
//
// If that option is used, linker should emit a .gdb_index section that allows
// debugger to locate and read .dwo files, containing neccessary debug
// information.
// More information about implementation can be found in DWARF specification,
// latest version is available at http://dwarfstd.org.
//
// .gdb_index section format:
//  (Information is based on/taken from
//  https://sourceware.org/gdb/onlinedocs/gdb/Index-Section-Format.html (*))
//
// A mapped index consists of several areas, laid out in order:
// 1) The file header.
// 2) "The CU (compilation unit) list. This is a sequence of pairs of 64-bit
//    little-endian values, sorted by the CU offset. The first element in each
//    pair is the offset of a CU in the .debug_info section. The second element
//    in each pair is the length of that CU. References to a CU elsewhere in the
//    map are done using a CU index, which is just the 0-based index into this
//    table. Note that if there are type CUs, then conceptually CUs and type CUs
//    form a single list for the purposes of CU indices."(*)
// 3) The types CU list. Depricated as .debug_types does not appear in the DWARF
//    v5 specification.
// 4) The address area. The address area is a sequence of address
//    entries, where each entrie contains low address, high address and CU
//    index.
// 5) "The symbol table. This is an open-addressed hash table. The size of the
//    hash table is always a power of 2. Each slot in the hash table consists of
//    a pair of offset_type values. The first value is the offset of the
//    symbol's name in the constant pool. The second value is the offset of the
//    CU vector in the constant pool."(*)
// 6) "The constant pool. This is simply a bunch of bytes. It is organized so
//    that alignment is correct: CU vectors are stored first, followed by
//    strings." (*)
//
// For constructing the .gdb_index section following steps should be performed:
// 1) For file header nothing special should be done. It contains the offsets to
//    the areas below.
// 2) Scan the compilation unit headers of the .debug_info sections to build a
//    list of compilation units.
// 3) CU Types are no longer needed as DWARF skeleton type units never made it
//    into the standard. lld does nothing to support parsing of .debug_types
//    and generates empty types CU area in .gdb_index section.
// 4) Address area entries are extracted from DW_TAG_compile_unit DIEs of
//   .debug_info sections.
// 5) For building the symbol table linker extracts the public names from the
//   .debug_gnu_pubnames and .debug_gnu_pubtypes sections. Then it builds the
//   hashtable in according to .gdb_index format specification.
// 6) Constant pool is populated at the same time as symbol table.
//
// Current version of implementation has 1, 2, 3 steps. So it writes .gdb_index
// header and list of compilation units. Since we so not plan to support types
// CU list area, it is also empty and so far is "implemented".
// Other data areas are not yet implemented.
//===----------------------------------------------------------------------===//

#include "GdbIndex.h"

#include "llvm/DebugInfo/DWARF/DWARFContext.h"

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
std::vector<std::pair<typename ELFT::uint, typename ELFT::uint>>
lld::elf::readCuList(InputSection<ELFT> *DebugInfoSec) {
  typedef typename ELFT::uint uintX_t;

  std::unique_ptr<DWARFContext> Dwarf;
  if (Expected<std::unique_ptr<object::ObjectFile>> Obj =
          object::ObjectFile::createObjectFile(DebugInfoSec->getFile()->MB))
    Dwarf.reset(new DWARFContextInMemory(*Obj.get()));

  if (!Dwarf) {
    error(toString(DebugInfoSec->getFile()) + ": error creating DWARF context");
    return {};
  }

  std::vector<std::pair<uintX_t, uintX_t>> Ret;
  for (std::unique_ptr<DWARFCompileUnit> &CU : Dwarf->compile_units())
    Ret.push_back(
        {DebugInfoSec->OutSecOff + CU->getOffset(), CU->getLength() + 4});
  return Ret;
}

template std::vector<std::pair<uint32_t, uint32_t>>
lld::elf::readCuList<ELF32LE>(InputSection<ELF32LE> *);
template std::vector<std::pair<uint32_t, uint32_t>>
lld::elf::readCuList<ELF32BE>(InputSection<ELF32BE> *);
template std::vector<std::pair<uint64_t, uint64_t>>
lld::elf::readCuList<ELF64LE>(InputSection<ELF64LE> *);
template std::vector<std::pair<uint64_t, uint64_t>>
lld::elf::readCuList<ELF64BE>(InputSection<ELF64BE> *);
