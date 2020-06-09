//===- InputFiles.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to parse Mach-O object files. In this comment,
// we describe the Mach-O file structure and how we parse it.
//
// Mach-O is not very different from ELF or COFF. The notion of symbols,
// sections and relocations exists in Mach-O as it does in ELF and COFF.
//
// Perhaps the notion that is new to those who know ELF/COFF is "subsections".
// In ELF/COFF, sections are an atomic unit of data copied from input files to
// output files. When we merge or garbage-collect sections, we treat each
// section as an atomic unit. In Mach-O, that's not the case. Sections can
// consist of multiple subsections, and subsections are a unit of merging and
// garbage-collecting. Therefore, Mach-O's subsections are more similar to
// ELF/COFF's sections than Mach-O's sections are.
//
// A section can have multiple symbols. A symbol that does not have the
// N_ALT_ENTRY attribute indicates a beginning of a subsection. Therefore, by
// definition, a symbol is always present at the beginning of each subsection. A
// symbol with N_ALT_ENTRY attribute does not start a new subsection and can
// point to a middle of a subsection.
//
// The notion of subsections also affects how relocations are represented in
// Mach-O. All references within a section need to be explicitly represented as
// relocations if they refer to different subsections, because we obviously need
// to fix up addresses if subsections are laid out in an output file differently
// than they were in object files. To represent that, Mach-O relocations can
// refer to an unnamed location via its address. Scattered relocations (those
// with the R_SCATTERED bit set) always refer to unnamed locations.
// Non-scattered relocations refer to an unnamed location if r_extern is not set
// and r_symbolnum is zero.
//
// Without the above differences, I think you can use your knowledge about ELF
// and COFF for Mach-O.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Config.h"
#include "ExportTrie.h"
#include "InputSection.h"
#include "MachOStructs.h"
#include "OutputSection.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support::endian;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

std::vector<InputFile *> macho::inputFiles;

// Open a given file path and return it as a memory-mapped file.
Optional<MemoryBufferRef> macho::readFile(StringRef path) {
  // Open a file.
  auto mbOrErr = MemoryBuffer::getFile(path);
  if (auto ec = mbOrErr.getError()) {
    error("cannot open " + path + ": " + ec.message());
    return None;
  }

  std::unique_ptr<MemoryBuffer> &mb = *mbOrErr;
  MemoryBufferRef mbref = mb->getMemBufferRef();
  make<std::unique_ptr<MemoryBuffer>>(std::move(mb)); // take mb ownership

  // If this is a regular non-fat file, return it.
  const char *buf = mbref.getBufferStart();
  auto *hdr = reinterpret_cast<const MachO::fat_header *>(buf);
  if (read32be(&hdr->magic) != MachO::FAT_MAGIC)
    return mbref;

  // Object files and archive files may be fat files, which contains
  // multiple real files for different CPU ISAs. Here, we search for a
  // file that matches with the current link target and returns it as
  // a MemoryBufferRef.
  auto *arch = reinterpret_cast<const MachO::fat_arch *>(buf + sizeof(*hdr));

  for (uint32_t i = 0, n = read32be(&hdr->nfat_arch); i < n; ++i) {
    if (reinterpret_cast<const char *>(arch + i + 1) >
        buf + mbref.getBufferSize()) {
      error(path + ": fat_arch struct extends beyond end of file");
      return None;
    }

    if (read32be(&arch[i].cputype) != target->cpuType ||
        read32be(&arch[i].cpusubtype) != target->cpuSubtype)
      continue;

    uint32_t offset = read32be(&arch[i].offset);
    uint32_t size = read32be(&arch[i].size);
    if (offset + size > mbref.getBufferSize())
      error(path + ": slice extends beyond end of file");
    return MemoryBufferRef(StringRef(buf + offset, size), path.copy(bAlloc));
  }

  error("unable to find matching architecture in " + path);
  return None;
}

static const load_command *findCommand(const mach_header_64 *hdr,
                                       uint32_t type) {
  const uint8_t *p =
      reinterpret_cast<const uint8_t *>(hdr) + sizeof(mach_header_64);

  for (uint32_t i = 0, n = hdr->ncmds; i < n; ++i) {
    auto *cmd = reinterpret_cast<const load_command *>(p);
    if (cmd->cmd == type)
      return cmd;
    p += cmd->cmdsize;
  }
  return nullptr;
}

void InputFile::parseSections(ArrayRef<section_64> sections) {
  subsections.reserve(sections.size());
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());

  for (const section_64 &sec : sections) {
    InputSection *isec = make<InputSection>();
    isec->file = this;
    isec->name = StringRef(sec.sectname, strnlen(sec.sectname, 16));
    isec->segname = StringRef(sec.segname, strnlen(sec.segname, 16));
    isec->data = {buf + sec.offset, static_cast<size_t>(sec.size)};
    if (sec.align >= 32)
      error("alignment " + std::to_string(sec.align) + " of section " +
            isec->name + " is too large");
    else
      isec->align = 1 << sec.align;
    isec->flags = sec.flags;
    subsections.push_back({{0, isec}});
  }
}

// Find the subsection corresponding to the greatest section offset that is <=
// that of the given offset.
//
// offset: an offset relative to the start of the original InputSection (before
// any subsection splitting has occurred). It will be updated to represent the
// same location as an offset relative to the start of the containing
// subsection.
static InputSection *findContainingSubsection(SubsectionMap &map,
                                              uint32_t *offset) {
  auto it = std::prev(map.upper_bound(*offset));
  *offset -= it->first;
  return it->second;
}

void InputFile::parseRelocations(const section_64 &sec,
                                 SubsectionMap &subsecMap) {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  ArrayRef<any_relocation_info> relInfos(
      reinterpret_cast<const any_relocation_info *>(buf + sec.reloff),
      sec.nreloc);

  for (const any_relocation_info &anyRel : relInfos) {
    if (anyRel.r_word0 & R_SCATTERED)
      fatal("TODO: Scattered relocations not supported");

    auto rel = reinterpret_cast<const relocation_info &>(anyRel);

    Reloc r;
    r.type = rel.r_type;
    r.pcrel = rel.r_pcrel;
    uint64_t rawAddend = target->getImplicitAddend(mb, sec, rel);

    if (rel.r_extern) {
      r.target = symbols[rel.r_symbolnum];
      r.addend = rawAddend;
    } else {
      if (!rel.r_pcrel)
        fatal("TODO: Only pcrel section relocations are supported");

      if (rel.r_symbolnum == 0 || rel.r_symbolnum > subsections.size())
        fatal("invalid section index in relocation for offset " +
              std::to_string(r.offset) + " in section " + sec.sectname +
              " of " + getName());

      SubsectionMap &targetSubsecMap = subsections[rel.r_symbolnum - 1];
      const section_64 &targetSec = sectionHeaders[rel.r_symbolnum - 1];
      // The implicit addend for pcrel section relocations is the pcrel offset
      // in terms of the addresses in the input file. Here we adjust it so that
      // it describes the offset from the start of the target section.
      // TODO: Figure out what to do for non-pcrel section relocations.
      // TODO: The offset of 4 is probably not right for ARM64, nor for
      //       relocations with r_length != 2.
      uint32_t targetOffset =
          sec.addr + rel.r_address + 4 + rawAddend - targetSec.addr;
      r.target = findContainingSubsection(targetSubsecMap, &targetOffset);
      r.addend = targetOffset;
    }

    r.offset = rel.r_address;
    InputSection *subsec = findContainingSubsection(subsecMap, &r.offset);
    subsec->relocs.push_back(r);
  }
}

void InputFile::parseSymbols(ArrayRef<structs::nlist_64> nList,
                             const char *strtab, bool subsectionsViaSymbols) {
  // resize(), not reserve(), because we are going to create N_ALT_ENTRY symbols
  // out-of-sequence.
  symbols.resize(nList.size());
  std::vector<size_t> altEntrySymIdxs;

  auto createDefined = [&](const structs::nlist_64 &sym, InputSection *isec,
                           uint32_t value) -> Symbol * {
    StringRef name = strtab + sym.n_strx;
    if (sym.n_type & N_EXT)
      // Global defined symbol
      return symtab->addDefined(name, isec, value);
    else
      // Local defined symbol
      return make<Defined>(name, isec, value);
  };

  for (size_t i = 0, n = nList.size(); i < n; ++i) {
    const structs::nlist_64 &sym = nList[i];

    // Undefined symbol
    if (!sym.n_sect) {
      StringRef name = strtab + sym.n_strx;
      symbols[i] = symtab->addUndefined(name);
      continue;
    }

    const section_64 &sec = sectionHeaders[sym.n_sect - 1];
    SubsectionMap &subsecMap = subsections[sym.n_sect - 1];
    uint64_t offset = sym.n_value - sec.addr;

    // If the input file does not use subsections-via-symbols, all symbols can
    // use the same subsection. Otherwise, we must split the sections along
    // symbol boundaries.
    if (!subsectionsViaSymbols) {
      symbols[i] = createDefined(sym, subsecMap[0], offset);
      continue;
    }

    // nList entries aren't necessarily arranged in address order. Therefore,
    // we can't create alt-entry symbols at this point because a later symbol
    // may split its section, which may affect which subsection the alt-entry
    // symbol is assigned to. So we need to handle them in a second pass below.
    if (sym.n_desc & N_ALT_ENTRY) {
      altEntrySymIdxs.push_back(i);
      continue;
    }

    // Find the subsection corresponding to the greatest section offset that is
    // <= that of the current symbol. The subsection that we find either needs
    // to be used directly or split in two.
    uint32_t firstSize = offset;
    InputSection *firstIsec = findContainingSubsection(subsecMap, &firstSize);

    if (firstSize == 0) {
      // Alias of an existing symbol, or the first symbol in the section. These
      // are handled by reusing the existing section.
      symbols[i] = createDefined(sym, firstIsec, 0);
      continue;
    }

    // We saw a symbol definition at a new offset. Split the section into two
    // subsections. The new symbol uses the second subsection.
    auto *secondIsec = make<InputSection>(*firstIsec);
    secondIsec->data = firstIsec->data.slice(firstSize);
    firstIsec->data = firstIsec->data.slice(0, firstSize);
    // TODO: ld64 appears to preserve the original alignment as well as each
    // subsection's offset from the last aligned address. We should consider
    // emulating that behavior.
    secondIsec->align = MinAlign(firstIsec->align, offset);

    subsecMap[offset] = secondIsec;
    // By construction, the symbol will be at offset zero in the new section.
    symbols[i] = createDefined(sym, secondIsec, 0);
  }

  for (size_t idx : altEntrySymIdxs) {
    const structs::nlist_64 &sym = nList[idx];
    SubsectionMap &subsecMap = subsections[sym.n_sect - 1];
    uint32_t off = sym.n_value - sectionHeaders[sym.n_sect - 1].addr;
    InputSection *subsec = findContainingSubsection(subsecMap, &off);
    symbols[idx] = createDefined(sym, subsec, off);
  }
}

ObjFile::ObjFile(MemoryBufferRef mb) : InputFile(ObjKind, mb) {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const mach_header_64 *>(mb.getBufferStart());

  if (const load_command *cmd = findCommand(hdr, LC_SEGMENT_64)) {
    auto *c = reinterpret_cast<const segment_command_64 *>(cmd);
    sectionHeaders = ArrayRef<section_64>{
        reinterpret_cast<const section_64 *>(c + 1), c->nsects};
    parseSections(sectionHeaders);
  }

  // TODO: Error on missing LC_SYMTAB?
  if (const load_command *cmd = findCommand(hdr, LC_SYMTAB)) {
    auto *c = reinterpret_cast<const symtab_command *>(cmd);
    ArrayRef<structs::nlist_64> nList(
        reinterpret_cast<const structs::nlist_64 *>(buf + c->symoff), c->nsyms);
    const char *strtab = reinterpret_cast<const char *>(buf) + c->stroff;
    bool subsectionsViaSymbols = hdr->flags & MH_SUBSECTIONS_VIA_SYMBOLS;
    parseSymbols(nList, strtab, subsectionsViaSymbols);
  }

  // The relocations may refer to the symbols, so we parse them after we have
  // parsed all the symbols.
  for (size_t i = 0, n = subsections.size(); i < n; ++i)
    parseRelocations(sectionHeaders[i], subsections[i]);
}

DylibFile::DylibFile(MemoryBufferRef mb, DylibFile *umbrella)
    : InputFile(DylibKind, mb) {
  if (umbrella == nullptr)
    umbrella = this;

  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const mach_header_64 *>(mb.getBufferStart());

  // Initialize dylibName.
  if (const load_command *cmd = findCommand(hdr, LC_ID_DYLIB)) {
    auto *c = reinterpret_cast<const dylib_command *>(cmd);
    dylibName = reinterpret_cast<const char *>(cmd) + read32le(&c->dylib.name);
  } else {
    error("dylib " + getName() + " missing LC_ID_DYLIB load command");
    return;
  }

  // Initialize symbols.
  if (const load_command *cmd = findCommand(hdr, LC_DYLD_INFO_ONLY)) {
    auto *c = reinterpret_cast<const dyld_info_command *>(cmd);
    parseTrie(buf + c->export_off, c->export_size,
              [&](const Twine &name, uint64_t flags) {
                symbols.push_back(symtab->addDylib(saver.save(name), umbrella));
              });
  } else {
    error("LC_DYLD_INFO_ONLY not found in " + getName());
    return;
  }

  if (hdr->flags & MH_NO_REEXPORTED_DYLIBS)
    return;

  const uint8_t *p =
      reinterpret_cast<const uint8_t *>(hdr) + sizeof(mach_header_64);
  for (uint32_t i = 0, n = hdr->ncmds; i < n; ++i) {
    auto *cmd = reinterpret_cast<const load_command *>(p);
    p += cmd->cmdsize;
    if (cmd->cmd != LC_REEXPORT_DYLIB)
      continue;

    auto *c = reinterpret_cast<const dylib_command *>(cmd);
    StringRef reexportPath =
        reinterpret_cast<const char *>(c) + read32le(&c->dylib.name);
    // TODO: Expand @loader_path, @executable_path etc in reexportPath
    Optional<MemoryBufferRef> buffer = readFile(reexportPath);
    if (!buffer) {
      error("unable to read re-exported dylib at " + reexportPath);
      return;
    }
    reexported.push_back(make<DylibFile>(*buffer, umbrella));
  }
}

DylibFile::DylibFile(std::shared_ptr<llvm::MachO::InterfaceFile> interface,
                     DylibFile *umbrella)
    : InputFile(DylibKind, MemoryBufferRef()) {
  if (umbrella == nullptr)
    umbrella = this;

  dylibName = saver.save(interface->getInstallName());
  // TODO(compnerd) filter out symbols based on the target platform
  for (const auto symbol : interface->symbols())
    if (symbol->getArchitectures().has(config->arch))
      symbols.push_back(
          symtab->addDylib(saver.save(symbol->getName()), umbrella));
  // TODO(compnerd) properly represent the hierarchy of the documents as it is
  // in theory possible to have re-exported dylibs from re-exported dylibs which
  // should be parent'ed to the child.
  for (auto document : interface->documents())
    reexported.push_back(make<DylibFile>(document, umbrella));
}

DylibFile::DylibFile() : InputFile(DylibKind, MemoryBufferRef()) {}

DylibFile *DylibFile::createLibSystemMock() {
  auto *file = make<DylibFile>();
  file->mb = MemoryBufferRef("", "/usr/lib/libSystem.B.dylib");
  file->dylibName = "/usr/lib/libSystem.B.dylib";
  file->symbols.push_back(symtab->addDylib("dyld_stub_binder", file));
  return file;
}

ArchiveFile::ArchiveFile(std::unique_ptr<llvm::object::Archive> &&f)
    : InputFile(ArchiveKind, f->getMemoryBufferRef()), file(std::move(f)) {
  for (const object::Archive::Symbol &sym : file->symbols())
    symtab->addLazy(sym.getName(), this, sym);
}

void ArchiveFile::fetch(const object::Archive::Symbol &sym) {
  object::Archive::Child c =
      CHECK(sym.getMember(), toString(this) +
                                 ": could not get the member for symbol " +
                                 sym.getName());

  if (!seen.insert(c.getChildOffset()).second)
    return;

  MemoryBufferRef mb =
      CHECK(c.getMemoryBufferRef(),
            toString(this) +
                ": could not get the buffer for the member defining symbol " +
                sym.getName());
  auto file = make<ObjFile>(mb);
  symbols.insert(symbols.end(), file->symbols.begin(), file->symbols.end());
  subsections.insert(subsections.end(), file->subsections.begin(),
                     file->subsections.end());
}

// Returns "<internal>" or "baz.o".
std::string lld::toString(const InputFile *file) {
  return file ? std::string(file->getName()) : "<internal>";
}
