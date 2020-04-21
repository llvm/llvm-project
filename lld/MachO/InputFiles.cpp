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
#include "InputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support::endian;
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

  error("TODO: Add support for universal binaries");
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

std::vector<InputSection *>
InputFile::parseSections(ArrayRef<section_64> sections) {
  std::vector<InputSection *> ret;
  ret.reserve(sections.size());

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
    ret.push_back(isec);
  }

  return ret;
}

void InputFile::parseRelocations(const section_64 &sec,
                                 std::vector<Reloc> &relocs) {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  ArrayRef<any_relocation_info> relInfos(
      reinterpret_cast<const any_relocation_info *>(buf + sec.reloff),
      sec.nreloc);

  for (const any_relocation_info &anyRel : relInfos) {
    Reloc r;
    if (anyRel.r_word0 & R_SCATTERED) {
      error("TODO: Scattered relocations not supported");
    } else {
      auto rel = reinterpret_cast<const relocation_info &>(anyRel);
      r.type = rel.r_type;
      r.offset = rel.r_address;
      r.addend = target->getImplicitAddend(buf + sec.offset + r.offset, r.type);
      if (rel.r_extern)
        r.target = symbols[rel.r_symbolnum];
      else {
        error("TODO: Non-extern relocations are not supported");
        continue;
      }
    }
    relocs.push_back(r);
  }
}

ObjFile::ObjFile(MemoryBufferRef mb) : InputFile(ObjKind, mb) {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const mach_header_64 *>(mb.getBufferStart());
  ArrayRef<section_64> objSections;

  if (const load_command *cmd = findCommand(hdr, LC_SEGMENT_64)) {
    auto *c = reinterpret_cast<const segment_command_64 *>(cmd);
    objSections = ArrayRef<section_64>{
        reinterpret_cast<const section_64 *>(c + 1), c->nsects};
    sections = parseSections(objSections);
  }

  // TODO: Error on missing LC_SYMTAB?
  if (const load_command *cmd = findCommand(hdr, LC_SYMTAB)) {
    auto *c = reinterpret_cast<const symtab_command *>(cmd);
    const char *strtab = reinterpret_cast<const char *>(buf) + c->stroff;
    ArrayRef<const nlist_64> nList(
        reinterpret_cast<const nlist_64 *>(buf + c->symoff), c->nsyms);

    symbols.reserve(c->nsyms);

    for (const nlist_64 &sym : nList) {
      StringRef name = strtab + sym.n_strx;

      // Undefined symbol
      if (!sym.n_sect) {
        symbols.push_back(symtab->addUndefined(name));
        continue;
      }

      InputSection *isec = sections[sym.n_sect - 1];
      const section_64 &objSec = objSections[sym.n_sect - 1];
      uint64_t value = sym.n_value - objSec.addr;

      // Global defined symbol
      if (sym.n_type & N_EXT) {
        symbols.push_back(symtab->addDefined(name, isec, value));
        continue;
      }

      // Local defined symbol
      symbols.push_back(make<Defined>(name, isec, value));
    }
  }

  // The relocations may refer to the symbols, so we parse them after we have
  // the symbols loaded.
  if (!sections.empty()) {
    auto it = sections.begin();
    for (const section_64 &sec : objSections) {
      parseRelocations(sec, (*it)->relocs);
      ++it;
    }
  }
}

DylibFile::DylibFile(MemoryBufferRef mb) : InputFile(DylibKind, mb) {
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
  if (const load_command *cmd = findCommand(hdr, LC_SYMTAB)) {
    auto *c = reinterpret_cast<const symtab_command *>(cmd);
    const char *strtab = reinterpret_cast<const char *>(buf + c->stroff);
    ArrayRef<const nlist_64> nList(
        reinterpret_cast<const nlist_64 *>(buf + c->symoff), c->nsyms);

    symbols.reserve(c->nsyms);

    for (const nlist_64 &sym : nList) {
      StringRef name = strtab + sym.n_strx;
      // TODO: Figure out what to do about undefined symbols: ignore or warn
      // if unsatisfied? Also make sure we handle re-exported symbols
      // correctly.
      symbols.push_back(symtab->addDylib(name, this));
    }
  }
}

// Returns "<internal>" or "baz.o".
std::string lld::toString(const InputFile *file) {
  return file ? std::string(file->getName()) : "<internal>";
}
