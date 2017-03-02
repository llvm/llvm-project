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
//===----------------------------------------------------------------------===//

#include "GdbIndex.h"
#include "Memory.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugPubTable.h"
#include "llvm/Object/ELFObjectFile.h"

using namespace llvm;
using namespace llvm::object;
using namespace lld;
using namespace lld::elf;

std::pair<bool, GdbSymbol *> GdbHashTab::add(uint32_t Hash, size_t Offset) {
  GdbSymbol *&Sym = Map[Offset];
  if (Sym)
    return {false, Sym};
  Sym = make<GdbSymbol>(Hash, Offset);
  return {true, Sym};
}

void GdbHashTab::finalizeContents() {
  uint32_t Size = std::max<uint32_t>(1024, NextPowerOf2(Map.size() * 4 / 3));
  uint32_t Mask = Size - 1;
  Table.resize(Size);

  for (auto &P : Map) {
    GdbSymbol *Sym = P.second;
    uint32_t I = Sym->NameHash & Mask;
    uint32_t Step = ((Sym->NameHash * 17) & Mask) | 1;

    while (Table[I])
      I = (I + Step) & Mask;
    Table[I] = Sym;
  }
}
