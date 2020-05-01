//===- Writer.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

namespace {
class LCLinkEdit;
class LCDyldInfo;
class LCSymtab;

class Writer {
public:
  Writer() : buffer(errorHandler().outputBuffer) {}

  void scanRelocations();
  void createOutputSections();
  void createLoadCommands();
  void assignAddresses(OutputSegment *);
  void createSymtabContents();

  void openFile();
  void writeSections();

  void run();

  std::unique_ptr<FileOutputBuffer> &buffer;
  uint64_t addr = 0;
  uint64_t fileOff = 0;
  MachHeaderSection *headerSection = nullptr;
  BindingSection *bindingSection = nullptr;
  ExportSection *exportSection = nullptr;
  StringTableSection *stringTableSection = nullptr;
  SymtabSection *symtabSection = nullptr;
};

// LC_DYLD_INFO_ONLY stores the offsets of symbol import/export information.
class LCDyldInfo : public LoadCommand {
public:
  LCDyldInfo(BindingSection *bindingSection, ExportSection *exportSection)
      : bindingSection(bindingSection), exportSection(exportSection) {}

  uint32_t getSize() const override { return sizeof(dyld_info_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dyld_info_command *>(buf);
    c->cmd = LC_DYLD_INFO_ONLY;
    c->cmdsize = getSize();
    if (bindingSection->isNeeded()) {
      c->bind_off = bindingSection->fileOff;
      c->bind_size = bindingSection->getFileSize();
    }
    if (exportSection->isNeeded()) {
      c->export_off = exportSection->fileOff;
      c->export_size = exportSection->getFileSize();
    }
  }

  BindingSection *bindingSection;
  ExportSection *exportSection;
};

class LCDysymtab : public LoadCommand {
public:
  uint32_t getSize() const override { return sizeof(dysymtab_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dysymtab_command *>(buf);
    c->cmd = LC_DYSYMTAB;
    c->cmdsize = getSize();
  }
};

class LCSegment : public LoadCommand {
public:
  LCSegment(StringRef name, OutputSegment *seg) : name(name), seg(seg) {}

  uint32_t getSize() const override {
    return sizeof(segment_command_64) +
           seg->numNonHiddenSections() * sizeof(section_64);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<segment_command_64 *>(buf);
    buf += sizeof(segment_command_64);

    c->cmd = LC_SEGMENT_64;
    c->cmdsize = getSize();
    memcpy(c->segname, name.data(), name.size());
    c->fileoff = seg->fileOff;
    c->maxprot = seg->maxProt;
    c->initprot = seg->initProt;

    if (!seg->isNeeded())
      return;

    c->vmaddr = seg->firstSection()->addr;
    c->vmsize =
        seg->lastSection()->addr + seg->lastSection()->getSize() - c->vmaddr;
    c->nsects = seg->numNonHiddenSections();

    for (auto &p : seg->getSections()) {
      StringRef s = p.first;
      OutputSection *section = p.second;
      c->filesize += section->getFileSize();
      if (section->isHidden())
        continue;

      auto *sectHdr = reinterpret_cast<section_64 *>(buf);
      buf += sizeof(section_64);

      memcpy(sectHdr->sectname, s.data(), s.size());
      memcpy(sectHdr->segname, name.data(), name.size());

      sectHdr->addr = section->addr;
      sectHdr->offset = section->fileOff;
      sectHdr->align = Log2_32(section->align);
      sectHdr->flags = section->flags;
      sectHdr->size = section->getSize();
    }
  }

private:
  StringRef name;
  OutputSegment *seg;
};

class LCMain : public LoadCommand {
  uint32_t getSize() const override { return sizeof(entry_point_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<entry_point_command *>(buf);
    c->cmd = LC_MAIN;
    c->cmdsize = getSize();
    c->entryoff = config->entry->getVA() - ImageBase;
    c->stacksize = 0;
  }
};

class LCSymtab : public LoadCommand {
public:
  LCSymtab(SymtabSection *symtabSection, StringTableSection *stringTableSection)
      : symtabSection(symtabSection), stringTableSection(stringTableSection) {}

  uint32_t getSize() const override { return sizeof(symtab_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<symtab_command *>(buf);
    c->cmd = LC_SYMTAB;
    c->cmdsize = getSize();
    c->symoff = symtabSection->fileOff;
    c->nsyms = symtabSection->getNumSymbols();
    c->stroff = stringTableSection->fileOff;
    c->strsize = stringTableSection->getFileSize();
  }

  SymtabSection *symtabSection = nullptr;
  StringTableSection *stringTableSection = nullptr;
};

class LCLoadDylib : public LoadCommand {
public:
  LCLoadDylib(StringRef path) : path(path) {}

  uint32_t getSize() const override {
    return alignTo(sizeof(dylib_command) + path.size() + 1, 8);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dylib_command *>(buf);
    buf += sizeof(dylib_command);

    c->cmd = LC_LOAD_DYLIB;
    c->cmdsize = getSize();
    c->dylib.name = sizeof(dylib_command);

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

private:
  StringRef path;
};

class LCIdDylib : public LoadCommand {
public:
  LCIdDylib(StringRef name) : name(name) {}

  uint32_t getSize() const override {
    return alignTo(sizeof(dylib_command) + name.size() + 1, 8);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dylib_command *>(buf);
    buf += sizeof(dylib_command);

    c->cmd = LC_ID_DYLIB;
    c->cmdsize = getSize();
    c->dylib.name = sizeof(dylib_command);

    memcpy(buf, name.data(), name.size());
    buf[name.size()] = '\0';
  }

private:
  StringRef name;
};

class LCLoadDylinker : public LoadCommand {
public:
  uint32_t getSize() const override {
    return alignTo(sizeof(dylinker_command) + path.size() + 1, 8);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dylinker_command *>(buf);
    buf += sizeof(dylinker_command);

    c->cmd = LC_LOAD_DYLINKER;
    c->cmdsize = getSize();
    c->name = sizeof(dylinker_command);

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

private:
  // Recent versions of Darwin won't run any binary that has dyld at a
  // different location.
  const StringRef path = "/usr/lib/dyld";
};
} // namespace

void Writer::scanRelocations() {
  for (InputSection *sect : inputSections)
    for (Reloc &r : sect->relocs)
      if (auto *s = r.target.dyn_cast<Symbol *>())
        if (auto *dylibSymbol = dyn_cast<DylibSymbol>(s))
          in.got->addEntry(*dylibSymbol);
}

void Writer::createLoadCommands() {
  headerSection->addLoadCommand(
      make<LCDyldInfo>(bindingSection, exportSection));
  headerSection->addLoadCommand(
      make<LCSymtab>(symtabSection, stringTableSection));
  headerSection->addLoadCommand(make<LCDysymtab>());

  switch (config->outputType) {
  case MH_EXECUTE:
    headerSection->addLoadCommand(make<LCMain>());
    headerSection->addLoadCommand(make<LCLoadDylinker>());
    break;
  case MH_DYLIB:
    headerSection->addLoadCommand(make<LCIdDylib>(config->installName));
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }

  uint8_t segIndex = 0;
  for (OutputSegment *seg : outputSegments) {
    if (seg->isNeeded()) {
      headerSection->addLoadCommand(make<LCSegment>(seg->name, seg));
      seg->index = segIndex++;
    }
  }

  uint64_t dylibOrdinal = 1;
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      headerSection->addLoadCommand(make<LCLoadDylib>(dylibFile->dylibName));
      dylibFile->ordinal = dylibOrdinal++;
    }
  }

  // TODO: dyld requires libSystem to be loaded. libSystem is a universal
  // binary and we don't have support for that yet, so mock it out here.
  headerSection->addLoadCommand(
      make<LCLoadDylib>("/usr/lib/libSystem.B.dylib"));
}

void Writer::createOutputSections() {
  // First, create hidden sections
  headerSection = make<MachHeaderSection>();
  bindingSection = make<BindingSection>();
  stringTableSection = make<StringTableSection>();
  symtabSection = make<SymtabSection>(*stringTableSection);
  exportSection = make<ExportSection>();

  switch (config->outputType) {
  case MH_EXECUTE:
    make<PageZeroSection>();
    break;
  case MH_DYLIB:
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }

  // Then merge input sections into output sections/segments.
  for (InputSection *isec : inputSections) {
    getOrCreateOutputSegment(isec->segname)
        ->getOrCreateOutputSection(isec->name)
        ->mergeInput(isec);
  }
}

void Writer::assignAddresses(OutputSegment *seg) {
  addr = alignTo(addr, PageSize);
  fileOff = alignTo(fileOff, PageSize);
  seg->fileOff = fileOff;

  for (auto &p : seg->getSections()) {
    OutputSection *section = p.second;
    addr = alignTo(addr, section->align);
    // We must align the file offsets too to avoid misaligned writes of
    // structs.
    fileOff = alignTo(fileOff, section->align);
    section->addr = addr;
    section->fileOff = fileOff;
    section->finalize();

    addr += section->getSize();
    fileOff += section->getFileSize();
  }
}

void Writer::openFile() {
  Expected<std::unique_ptr<FileOutputBuffer>> bufferOrErr =
      FileOutputBuffer::create(config->outputFile, fileOff,
                               FileOutputBuffer::F_executable);

  if (!bufferOrErr)
    error("failed to open " + config->outputFile + ": " +
          llvm::toString(bufferOrErr.takeError()));
  else
    buffer = std::move(*bufferOrErr);
}

void Writer::writeSections() {
  uint8_t *buf = buffer->getBufferStart();
  for (OutputSegment *seg : outputSegments) {
    for (auto &p : seg->getSections()) {
      OutputSection *section = p.second;
      section->writeTo(buf + section->fileOff);
    }
  }
}

void Writer::run() {
  // dyld requires __LINKEDIT segment to always exist (even if empty).
  OutputSegment *linkEditSegment =
      getOrCreateOutputSegment(segment_names::linkEdit);

  scanRelocations();

  // Sort and assign sections to their respective segments. No more sections nor
  // segments may be created after this method runs.
  createOutputSections();
  sortOutputSegmentsAndSections();

  createLoadCommands();

  // Ensure that segments (and the sections they contain) are allocated
  // addresses in ascending order, which dyld requires.
  //
  // Note that at this point, __LINKEDIT sections are empty, but we need to
  // determine addresses of other segments/sections before generating its
  // contents.
  for (OutputSegment *seg : outputSegments)
    if (seg != linkEditSegment)
      assignAddresses(seg);

  // Fill __LINKEDIT contents.
  bindingSection->finalizeContents();
  exportSection->finalizeContents();
  symtabSection->finalizeContents();

  // Now that __LINKEDIT is filled out, do a proper calculation of its
  // addresses and offsets.
  assignAddresses(linkEditSegment);

  openFile();
  if (errorCount())
    return;

  writeSections();

  if (auto e = buffer->commit())
    error("failed to write to the output file: " + toString(std::move(e)));
}

void macho::writeResult() { Writer().run(); }

void macho::createSyntheticSections() { in.got = make<GotSection>(); }
