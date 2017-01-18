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
#include "llvm/DebugInfo/DWARF/DWARFDebugPubTable.h"
#include "llvm/Object/ELFObjectFile.h"

using namespace llvm;
using namespace llvm::object;
using namespace lld::elf;

template <class ELFT>
GdbIndexBuilder<ELFT>::GdbIndexBuilder(InputSection<ELFT> *DebugInfoSec)
    : DebugInfoSec(DebugInfoSec) {
  if (Expected<std::unique_ptr<object::ObjectFile>> Obj =
          object::ObjectFile::createObjectFile(DebugInfoSec->getFile()->MB))
    Dwarf.reset(new DWARFContextInMemory(*Obj.get(), this));
  else
    error(toString(DebugInfoSec->getFile()) + ": error creating DWARF context");
}

template <class ELFT>
std::vector<std::pair<typename ELFT::uint, typename ELFT::uint>>
GdbIndexBuilder<ELFT>::readCUList() {
  std::vector<std::pair<uintX_t, uintX_t>> Ret;
  for (std::unique_ptr<DWARFCompileUnit> &CU : Dwarf->compile_units())
    Ret.push_back(
        {DebugInfoSec->OutSecOff + CU->getOffset(), CU->getLength() + 4});
  return Ret;
}

template <class ELFT>
std::vector<std::pair<StringRef, uint8_t>>
GdbIndexBuilder<ELFT>::readPubNamesAndTypes() {
  const bool IsLE = ELFT::TargetEndianness == llvm::support::little;
  StringRef Data[] = {Dwarf->getGnuPubNamesSection(),
                      Dwarf->getGnuPubTypesSection()};

  std::vector<std::pair<StringRef, uint8_t>> Ret;
  for (StringRef D : Data) {
    DWARFDebugPubTable PubTable(D, IsLE, true);
    for (const DWARFDebugPubTable::Set &S : PubTable.getData())
      for (const DWARFDebugPubTable::Entry &E : S.Entries)
        Ret.push_back({E.Name, E.Descriptor.toBits()});
  }
  return Ret;
}

std::pair<bool, GdbSymbol *> GdbHashTab::add(uint32_t Hash, size_t Offset) {
  if (Size * 4 / 3 >= Table.size())
    expand();

  GdbSymbol **Slot = findSlot(Hash, Offset);
  bool New = false;
  if (*Slot == nullptr) {
    ++Size;
    *Slot = new (Alloc) GdbSymbol(Hash, Offset);
    New = true;
  }
  return {New, *Slot};
}

void GdbHashTab::expand() {
  if (Table.empty()) {
    Table.resize(InitialSize);
    return;
  }
  std::vector<GdbSymbol *> NewTable(Table.size() * 2);
  NewTable.swap(Table);

  for (GdbSymbol *Sym : NewTable) {
    if (!Sym)
      continue;
    GdbSymbol **Slot = findSlot(Sym->NameHash, Sym->NameOffset);
    *Slot = Sym;
  }
}

// Methods finds a slot for symbol with given hash. The step size used to find
// the next candidate slot when handling a hash collision is specified in
// .gdb_index section format. The hash value for a table entry is computed by
// applying an iterative hash function to the symbol's name.
GdbSymbol **GdbHashTab::findSlot(uint32_t Hash, size_t Offset) {
  uint32_t Index = Hash & (Table.size() - 1);
  uint32_t Step = ((Hash * 17) & (Table.size() - 1)) | 1;

  for (;;) {
    GdbSymbol *S = Table[Index];
    if (!S || ((S->NameOffset == Offset) && (S->NameHash == Hash)))
      return &Table[Index];
    Index = (Index + Step) & (Table.size() - 1);
  }
}

template <class ELFT>
static InputSectionBase<ELFT> *
findSection(ArrayRef<InputSectionBase<ELFT> *> Arr, uint64_t Offset) {
  for (InputSectionBase<ELFT> *S : Arr)
    if (S && S != &InputSection<ELFT>::Discarded)
      if (Offset >= S->Offset && Offset < S->Offset + S->getSize())
        return S;
  return nullptr;
}

template <class ELFT>
std::vector<AddressEntry<ELFT>>
GdbIndexBuilder<ELFT>::readAddressArea(size_t CurrentCU) {
  std::vector<AddressEntry<ELFT>> Ret;
  for (const auto &CU : Dwarf->compile_units()) {
    DWARFAddressRangesVector Ranges;
    CU->collectAddressRanges(Ranges);

    ArrayRef<InputSectionBase<ELFT> *> Sections =
        DebugInfoSec->getFile()->getSections();

    for (std::pair<uint64_t, uint64_t> &R : Ranges)
      if (InputSectionBase<ELFT> *S = findSection(Sections, R.first))
        Ret.push_back(
            {S, R.first - S->Offset, R.second - S->Offset, CurrentCU});
    ++CurrentCU;
  }
  return Ret;
}

// We return file offset as load address for allocatable sections. That is
// currently used for collecting address ranges in readAddressArea(). We are
// able then to find section index that range belongs to.
template <class ELFT>
uint64_t GdbIndexBuilder<ELFT>::getSectionLoadAddress(
    const object::SectionRef &Sec) const {
  if (static_cast<const ELFSectionRef &>(Sec).getFlags() & ELF::SHF_ALLOC)
    return static_cast<const ELFSectionRef &>(Sec).getOffset();
  return 0;
}

template <class ELFT>
std::unique_ptr<LoadedObjectInfo> GdbIndexBuilder<ELFT>::clone() const {
  return {};
}

namespace lld {
namespace elf {
template class GdbIndexBuilder<ELF32LE>;
template class GdbIndexBuilder<ELF32BE>;
template class GdbIndexBuilder<ELF64LE>;
template class GdbIndexBuilder<ELF64BE>;
}
}
