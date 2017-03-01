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

class lld::elf::ObjInfoTy : public llvm::LoadedObjectInfo {
  uint64_t getSectionLoadAddress(const object::SectionRef &Sec) const override {
    auto &S = static_cast<const ELFSectionRef &>(Sec);
    if (S.getFlags() & ELF::SHF_ALLOC)
      return S.getOffset();
    return 0;
  }

  std::unique_ptr<llvm::LoadedObjectInfo> clone() const override { return {}; }
};

template <class ELFT>
GdbIndexBuilder<ELFT>::GdbIndexBuilder(InputSection *Sec)
    : DebugInfoSec(Sec), ObjInfo(new ObjInfoTy) {
  elf::ObjectFile<ELFT> *File = Sec->template getFile<ELFT>();
  Expected<std::unique_ptr<object::ObjectFile>> Obj =
      object::ObjectFile::createObjectFile(File->MB);

  if (Obj)
    Dwarf.reset(new DWARFContextInMemory(*Obj.get(), ObjInfo.get()));
  else
    error(toString(File) + ": error creating DWARF context");
}

template <class ELFT> GdbIndexBuilder<ELFT>::~GdbIndexBuilder() {}

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
  GdbSymbol *&Sym = Map[{Hash, Offset}];
  if (Sym)
    return {false, Sym};
  ++Size;
  Sym = make<GdbSymbol>(Hash, Offset);
  return {true, Sym};
}

void GdbHashTab::finalizeContents() {
  Table.resize(std::max<uint64_t>(1024, NextPowerOf2(Map.size() * 4 / 3)));

  for (auto &P : Map) {
    GdbSymbol *Sym = P.second;

    uint32_t I = Sym->NameHash & (Table.size() - 1);
    uint32_t Step = ((Sym->NameHash * 17) & (Table.size() - 1)) | 1;

    for (;;) {
      if (Table[I]) {
        I = (I + Step) & (Table.size() - 1);
        continue;
      }

      Table[I] = Sym;
      break;
    }
  }
}

template <class ELFT>
static InputSectionBase *findSection(ArrayRef<InputSectionBase *> Arr,
                                     uint64_t Offset) {
  for (InputSectionBase *S : Arr)
    if (S && S != &InputSection::Discarded)
      if (Offset >= S->Offset && Offset < S->Offset + S->getSize<ELFT>())
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

    ArrayRef<InputSectionBase *> Sections =
        DebugInfoSec->template getFile<ELFT>()->getSections();

    for (std::pair<uint64_t, uint64_t> &R : Ranges)
      if (InputSectionBase *S = findSection<ELFT>(Sections, R.first))
        Ret.push_back(
            {S, R.first - S->Offset, R.second - S->Offset, CurrentCU});
    ++CurrentCU;
  }
  return Ret;
}

template class elf::GdbIndexBuilder<ELF32LE>;
template class elf::GdbIndexBuilder<ELF32BE>;
template class elf::GdbIndexBuilder<ELF64LE>;
template class elf::GdbIndexBuilder<ELF64BE>;
