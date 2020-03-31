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
#include "Target.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support;
using namespace lld;
using namespace lld::macho;

namespace {
class LCLinkEdit;
class LCDyldInfo;
class LCSymtab;

class LoadCommand {
public:
  virtual ~LoadCommand() = default;
  virtual uint32_t getSize() const = 0;
  virtual void writeTo(uint8_t *buf) const = 0;
};

class Writer {
public:
  Writer() : buffer(errorHandler().outputBuffer) {}

  void createLoadCommands();
  void assignAddresses();

  void openFile();
  void writeHeader();
  void writeSections();

  void run();

  std::vector<LoadCommand *> loadCommands;
  std::unique_ptr<FileOutputBuffer> &buffer;
  uint64_t fileSize = 0;
  uint64_t sizeofCmds = 0;
  LCLinkEdit *linkEditSeg = nullptr;
  LCDyldInfo *dyldInfoSeg = nullptr;
  LCSymtab *symtabSeg = nullptr;
};

class LCPagezero : public LoadCommand {
public:
  uint32_t getSize() const override { return sizeof(segment_command_64); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<segment_command_64 *>(buf);
    c->cmd = LC_SEGMENT_64;
    c->cmdsize = getSize();
    strcpy(c->segname, "__PAGEZERO");
    c->vmsize = PageSize;
  }
};

class LCLinkEdit : public LoadCommand {
public:
  uint32_t getSize() const override { return sizeof(segment_command_64); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<segment_command_64 *>(buf);
    c->cmd = LC_SEGMENT_64;
    c->cmdsize = getSize();
    strcpy(c->segname, "__LINKEDIT");
    c->fileoff = fileOff;
    c->filesize = contents.size();
    c->maxprot = VM_PROT_READ | VM_PROT_WRITE;
    c->initprot = VM_PROT_READ;
  }

  uint64_t getOffset() const { return fileOff + contents.size(); }

  uint64_t fileOff = 0;
  SmallVector<char, 128> contents;
};

class LCDyldInfo : public LoadCommand {
public:
  uint32_t getSize() const override { return sizeof(dyld_info_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dyld_info_command *>(buf);
    c->cmd = LC_DYLD_INFO_ONLY;
    c->cmdsize = getSize();
    c->export_off = exportOff;
    c->export_size = exportSize;
  }

  uint64_t exportOff = 0;
  uint64_t exportSize = 0;
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
           seg->sections.size() * sizeof(section_64);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<segment_command_64 *>(buf);
    buf += sizeof(segment_command_64);

    c->cmd = LC_SEGMENT_64;
    c->cmdsize = getSize();
    memcpy(c->segname, name.data(), name.size());

    InputSection *firstSec = seg->sections.front().second[0];
    InputSection *lastSec = seg->sections.back().second.back();

    // dyld3's MachOLoaded::getSlide() assumes that the __TEXT segment starts
    // from the beginning of the file (i.e. the header).
    // TODO: replace this logic by creating a synthetic __TEXT,__mach_header
    // section instead.
    c->fileoff = name == "__TEXT" ? 0 : firstSec->addr - ImageBase;
    c->vmaddr = c->fileoff + ImageBase;
    c->vmsize = c->filesize = lastSec->addr + lastSec->data.size() - c->vmaddr;
    c->maxprot = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
    c->initprot = seg->perms;
    c->nsects = seg->sections.size();

    for (auto &p : seg->sections) {
      StringRef s = p.first;
      std::vector<InputSection *> &sections = p.second;

      auto *sectHdr = reinterpret_cast<section_64 *>(buf);
      buf += sizeof(section_64);

      memcpy(sectHdr->sectname, s.data(), s.size());
      memcpy(sectHdr->segname, name.data(), name.size());

      sectHdr->addr = sections[0]->addr;
      sectHdr->offset = sections[0]->addr - ImageBase;
      sectHdr->align = sections[0]->align;
      uint32_t maxAlign = 0;
      for (const InputSection *section : sections)
        maxAlign = std::max(maxAlign, section->align);
      sectHdr->align = Log2_32(maxAlign);
      sectHdr->flags = sections[0]->flags;
      sectHdr->size = sections.back()->addr + sections.back()->data.size() -
                      sections[0]->addr;
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
    c->entryoff = config->entry->getVA();
    c->stacksize = 0;
  }
};

class LCSymtab : public LoadCommand {
public:
  uint32_t getSize() const override { return sizeof(symtab_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<symtab_command *>(buf);
    c->cmd = LC_SYMTAB;
    c->cmdsize = getSize();
  }
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

void Writer::createLoadCommands() {
  linkEditSeg = make<LCLinkEdit>();
  dyldInfoSeg = make<LCDyldInfo>();
  symtabSeg = make<LCSymtab>();

  loadCommands.push_back(linkEditSeg);
  loadCommands.push_back(dyldInfoSeg);
  loadCommands.push_back(symtabSeg);
  loadCommands.push_back(make<LCPagezero>());
  loadCommands.push_back(make<LCLoadDylinker>());
  loadCommands.push_back(make<LCDysymtab>());
  loadCommands.push_back(make<LCMain>());
  // TODO: dyld requires libSystem to be loaded. libSystem is a universal
  // binary and we don't have support for that yet, so mock it out here.
  loadCommands.push_back(make<LCLoadDylib>("/usr/lib/libSystem.B.dylib"));

  for (OutputSegment *seg : outputSegments)
    if (!seg->sections.empty())
      loadCommands.push_back(make<LCSegment>(seg->name, seg));
}

void Writer::assignAddresses() {
  uint64_t addr = ImageBase + sizeof(mach_header_64);

  uint64_t size = 0;
  for (LoadCommand *lc : loadCommands)
    size += lc->getSize();
  sizeofCmds = size;
  addr += size;

  for (OutputSegment *seg : outputSegments) {
    addr = alignTo(addr, PageSize);

    for (auto &p : seg->sections) {
      ArrayRef<InputSection *> sections = p.second;
      for (InputSection *isec : sections) {
        addr = alignTo(addr, isec->align);
        isec->addr = addr;
        addr += isec->data.size();
      }
    }
  }

  linkEditSeg->fileOff = addr - ImageBase;
}

void Writer::openFile() {
  Expected<std::unique_ptr<FileOutputBuffer>> bufferOrErr =
      FileOutputBuffer::create(config->outputFile, fileSize,
                               FileOutputBuffer::F_executable);

  if (!bufferOrErr)
    error("failed to open " + config->outputFile + ": " +
          llvm::toString(bufferOrErr.takeError()));
  else
    buffer = std::move(*bufferOrErr);
}

void Writer::writeHeader() {
  auto *hdr = reinterpret_cast<mach_header_64 *>(buffer->getBufferStart());
  hdr->magic = MH_MAGIC_64;
  hdr->cputype = CPU_TYPE_X86_64;
  hdr->cpusubtype = CPU_SUBTYPE_X86_64_ALL | CPU_SUBTYPE_LIB64;
  hdr->filetype = MH_EXECUTE;
  hdr->ncmds = loadCommands.size();
  hdr->sizeofcmds = sizeofCmds;
  hdr->flags = MH_NOUNDEFS | MH_DYLDLINK | MH_TWOLEVEL;

  uint8_t *p = reinterpret_cast<uint8_t *>(hdr + 1);
  for (LoadCommand *lc : loadCommands) {
    lc->writeTo(p);
    p += lc->getSize();
  }
}

void Writer::writeSections() {
  uint8_t *buf = buffer->getBufferStart();

  for (OutputSegment *seg : outputSegments)
    for (auto &sect : seg->sections)
      for (InputSection *isec : sect.second)
        isec->writeTo(buf + isec->addr - ImageBase);

  memcpy(buf + linkEditSeg->fileOff, linkEditSeg->contents.data(),
         linkEditSeg->contents.size());
}

void Writer::run() {
  createLoadCommands();
  assignAddresses();
  fileSize = linkEditSeg->fileOff + linkEditSeg->contents.size();

  openFile();
  if (errorCount())
    return;

  writeHeader();
  writeSections();

  if (auto e = buffer->commit())
    error("failed to write to the output file: " + toString(std::move(e)));
}

void macho::writeResult() { Writer().run(); }
