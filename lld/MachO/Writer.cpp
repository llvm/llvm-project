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
  void createHiddenSections();
  void sortSections();
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
      c->bind_off = bindingSection->getFileOffset();
      c->bind_size = bindingSection->getFileSize();
    }
    if (exportSection->isNeeded()) {
      c->export_off = exportSection->getFileOffset();
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
           seg->numNonHiddenSections * sizeof(section_64);
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

    if (seg->getSections().empty())
      return;

    c->vmaddr = seg->firstSection()->addr;
    c->vmsize =
        seg->lastSection()->addr + seg->lastSection()->getSize() - c->vmaddr;
    c->nsects = seg->numNonHiddenSections;

    for (auto &p : seg->getSections()) {
      StringRef s = p.first;
      ArrayRef<InputSection *> sections = p.second;
      for (InputSection *isec : sections)
        c->filesize += isec->getFileSize();
      if (sections[0]->isHidden())
        continue;

      auto *sectHdr = reinterpret_cast<section_64 *>(buf);
      buf += sizeof(section_64);

      memcpy(sectHdr->sectname, s.data(), s.size());
      memcpy(sectHdr->segname, name.data(), name.size());

      sectHdr->addr = sections[0]->addr;
      sectHdr->offset = sections[0]->getFileOffset();
      sectHdr->align = sections[0]->align;
      uint32_t maxAlign = 0;
      for (const InputSection *section : sections)
        maxAlign = std::max(maxAlign, section->align);
      sectHdr->align = Log2_32(maxAlign);
      sectHdr->flags = sections[0]->flags;
      sectHdr->size = sections.back()->addr + sections.back()->getSize() -
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
    c->symoff = symtabSection->getFileOffset();
    c->nsyms = symtabSection->getNumSymbols();
    c->stroff = stringTableSection->getFileOffset();
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

class SectionComparator {
public:
  struct OrderInfo {
    uint32_t segmentOrder;
    DenseMap<StringRef, uint32_t> sectionOrdering;
  };

  SectionComparator() {
    // This defines the order of segments and the sections within each segment.
    // Segments that are not mentioned here will end up at defaultPosition;
    // sections that are not mentioned will end up at the end of the section
    // list for their given segment.
    std::vector<std::pair<StringRef, std::vector<StringRef>>> ordering{
        {segment_names::pageZero, {}},
        {segment_names::text, {section_names::header}},
        {defaultPosition, {}},
        // Make sure __LINKEDIT is the last segment (i.e. all its hidden
        // sections must be ordered after other sections).
        {segment_names::linkEdit,
         {
             section_names::binding,
             section_names::export_,
             section_names::symbolTable,
             section_names::stringTable,
         }},
    };

    for (uint32_t i = 0, n = ordering.size(); i < n; ++i) {
      auto &p = ordering[i];
      StringRef segname = p.first;
      const std::vector<StringRef> &sectOrdering = p.second;
      OrderInfo &info = orderMap[segname];
      info.segmentOrder = i;
      for (uint32_t j = 0, m = sectOrdering.size(); j < m; ++j)
        info.sectionOrdering[sectOrdering[j]] = j;
    }
  }

  // Return a {segmentOrder, sectionOrder} pair. Using this as a key will
  // ensure that all sections in the same segment are sorted contiguously.
  std::pair<uint32_t, uint32_t> order(const InputSection *isec) {
    auto it = orderMap.find(isec->segname);
    if (it == orderMap.end())
      return {orderMap[defaultPosition].segmentOrder, 0};
    OrderInfo &info = it->second;
    auto sectIt = info.sectionOrdering.find(isec->name);
    if (sectIt != info.sectionOrdering.end())
      return {info.segmentOrder, sectIt->second};
    return {info.segmentOrder, info.sectionOrdering.size()};
  }

  bool operator()(const InputSection *a, const InputSection *b) {
    return order(a) < order(b);
  }

private:
  const StringRef defaultPosition = StringRef();
  DenseMap<StringRef, OrderInfo> orderMap;
};

} // namespace

template <typename SectionType, typename... ArgT>
SectionType *createInputSection(ArgT &&... args) {
  auto *section = make<SectionType>(std::forward<ArgT>(args)...);
  inputSections.push_back(section);
  return section;
}

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

void Writer::createHiddenSections() {
  headerSection = createInputSection<MachHeaderSection>();
  bindingSection = createInputSection<BindingSection>();
  stringTableSection = createInputSection<StringTableSection>();
  symtabSection = createInputSection<SymtabSection>(*stringTableSection);
  exportSection = createInputSection<ExportSection>();

  switch (config->outputType) {
  case MH_EXECUTE:
    createInputSection<PageZeroSection>();
    break;
  case MH_DYLIB:
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }
}

void Writer::sortSections() {
  llvm::stable_sort(inputSections, SectionComparator());

  // TODO This is wrong; input sections ought to be grouped into
  // output sections, which are then organized like this.
  uint32_t sectionIndex = 0;
  // Add input sections to output segments.
  for (InputSection *isec : inputSections) {
    if (isec->isNeeded()) {
      if (!isec->isHidden())
        isec->sectionIndex = ++sectionIndex;
      getOrCreateOutputSegment(isec->segname)->addSection(isec);
    }
  }
}

void Writer::assignAddresses(OutputSegment *seg) {
  addr = alignTo(addr, PageSize);
  fileOff = alignTo(fileOff, PageSize);
  seg->fileOff = fileOff;

  for (auto &p : seg->getSections()) {
    ArrayRef<InputSection *> sections = p.second;
    for (InputSection *isec : sections) {
      addr = alignTo(addr, isec->align);
      // We must align the file offsets too to avoid misaligned writes of
      // structs.
      fileOff = alignTo(fileOff, isec->align);
      isec->addr = addr;
      addr += isec->getSize();
      fileOff += isec->getFileSize();
    }
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
    uint64_t fileOff = seg->fileOff;
    for (auto &sect : seg->getSections()) {
      for (InputSection *isec : sect.second) {
        fileOff = alignTo(fileOff, isec->align);
        isec->writeTo(buf + fileOff);
        fileOff += isec->getFileSize();
      }
    }
  }
}

void Writer::run() {
  scanRelocations();
  createHiddenSections();
  // Sort and assign sections to their respective segments. No more sections can
  // be created after this method runs.
  sortSections();
  // dyld requires __LINKEDIT segment to always exist (even if empty).
  getOrCreateOutputSegment(segment_names::linkEdit);
  // No more segments can be created after this method runs.
  createLoadCommands();

  // Ensure that segments (and the sections they contain) are allocated
  // addresses in ascending order, which dyld requires.
  //
  // Note that at this point, __LINKEDIT sections are empty, but we need to
  // determine addresses of other segments/sections before generating its
  // contents.
  for (OutputSegment *seg : outputSegments)
    assignAddresses(seg);

  // Fill __LINKEDIT contents.
  bindingSection->finalizeContents();
  exportSection->finalizeContents();
  symtabSection->finalizeContents();

  // Now that __LINKEDIT is filled out, do a proper calculation of its
  // addresses and offsets. We don't have to recalculate the other segments
  // since sortSections() ensures that __LINKEDIT is the last segment.
  assignAddresses(getOutputSegment(segment_names::linkEdit));

  openFile();
  if (errorCount())
    return;

  writeSections();

  if (auto e = buffer->commit())
    error("failed to write to the output file: " + toString(std::move(e)));
}

void macho::writeResult() { Writer().run(); }

void macho::createSyntheticSections() {
  in.got = createInputSection<GotSection>();
}
