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
#include "MergedOutputSection.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "UnwindInfoSection.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::MachO;
using namespace lld;
using namespace lld::macho;

namespace {
class LCUuid;

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
  void writeUuid();

  void run();

  std::unique_ptr<FileOutputBuffer> &buffer;
  uint64_t addr = 0;
  uint64_t fileOff = 0;
  MachHeaderSection *header = nullptr;
  StringTableSection *stringTableSection = nullptr;
  SymtabSection *symtabSection = nullptr;
  IndirectSymtabSection *indirectSymtabSection = nullptr;
  UnwindInfoSection *unwindInfoSection = nullptr;
  LCUuid *uuidCommand = nullptr;
};

// LC_DYLD_INFO_ONLY stores the offsets of symbol import/export information.
class LCDyldInfo : public LoadCommand {
public:
  LCDyldInfo(RebaseSection *rebaseSection, BindingSection *bindingSection,
             WeakBindingSection *weakBindingSection,
             LazyBindingSection *lazyBindingSection,
             ExportSection *exportSection)
      : rebaseSection(rebaseSection), bindingSection(bindingSection),
        weakBindingSection(weakBindingSection),
        lazyBindingSection(lazyBindingSection), exportSection(exportSection) {}

  uint32_t getSize() const override { return sizeof(dyld_info_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dyld_info_command *>(buf);
    c->cmd = LC_DYLD_INFO_ONLY;
    c->cmdsize = getSize();
    if (rebaseSection->isNeeded()) {
      c->rebase_off = rebaseSection->fileOff;
      c->rebase_size = rebaseSection->getFileSize();
    }
    if (bindingSection->isNeeded()) {
      c->bind_off = bindingSection->fileOff;
      c->bind_size = bindingSection->getFileSize();
    }
    if (weakBindingSection->isNeeded()) {
      c->weak_bind_off = weakBindingSection->fileOff;
      c->weak_bind_size = weakBindingSection->getFileSize();
    }
    if (lazyBindingSection->isNeeded()) {
      c->lazy_bind_off = lazyBindingSection->fileOff;
      c->lazy_bind_size = lazyBindingSection->getFileSize();
    }
    if (exportSection->isNeeded()) {
      c->export_off = exportSection->fileOff;
      c->export_size = exportSection->getFileSize();
    }
  }

  RebaseSection *rebaseSection;
  BindingSection *bindingSection;
  WeakBindingSection *weakBindingSection;
  LazyBindingSection *lazyBindingSection;
  ExportSection *exportSection;
};

class LCDysymtab : public LoadCommand {
public:
  LCDysymtab(IndirectSymtabSection *indirectSymtabSection)
      : indirectSymtabSection(indirectSymtabSection) {}

  uint32_t getSize() const override { return sizeof(dysymtab_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dysymtab_command *>(buf);
    c->cmd = LC_DYSYMTAB;
    c->cmdsize = getSize();
    c->indirectsymoff = indirectSymtabSection->fileOff;
    c->nindirectsyms = indirectSymtabSection->getNumSymbols();
  }

  IndirectSymtabSection *indirectSymtabSection = nullptr;
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

    if (seg->getSections().empty())
      return;

    c->vmaddr = seg->firstSection()->addr;
    c->vmsize =
        seg->lastSection()->addr + seg->lastSection()->getSize() - c->vmaddr;
    c->nsects = seg->numNonHiddenSections();

    for (OutputSection *osec : seg->getSections()) {
      if (!isZeroFill(osec->flags)) {
        assert(osec->fileOff >= seg->fileOff);
        c->filesize = std::max(
            c->filesize, osec->fileOff + osec->getFileSize() - seg->fileOff);
      }

      if (osec->isHidden())
        continue;

      auto *sectHdr = reinterpret_cast<section_64 *>(buf);
      buf += sizeof(section_64);

      memcpy(sectHdr->sectname, osec->name.data(), osec->name.size());
      memcpy(sectHdr->segname, name.data(), name.size());

      sectHdr->addr = osec->addr;
      sectHdr->offset = osec->fileOff;
      sectHdr->align = Log2_32(osec->align);
      sectHdr->flags = osec->flags;
      sectHdr->size = osec->getSize();
      sectHdr->reserved1 = osec->reserved1;
      sectHdr->reserved2 = osec->reserved2;
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

    if (config->entry->isInStubs())
      c->entryoff =
          in.stubs->fileOff + config->entry->stubsIndex * target->stubSize;
    else
      c->entryoff = config->entry->getFileOffset();

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

// There are several dylib load commands that share the same structure:
//   * LC_LOAD_DYLIB
//   * LC_ID_DYLIB
//   * LC_REEXPORT_DYLIB
class LCDylib : public LoadCommand {
public:
  LCDylib(LoadCommandType type, StringRef path) : type(type), path(path) {
    instanceCount++;
  }

  uint32_t getSize() const override {
    return alignTo(sizeof(dylib_command) + path.size() + 1, 8);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dylib_command *>(buf);
    buf += sizeof(dylib_command);

    c->cmd = type;
    c->cmdsize = getSize();
    c->dylib.name = sizeof(dylib_command);

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

  static uint32_t getInstanceCount() { return instanceCount; }

private:
  LoadCommandType type;
  StringRef path;
  static uint32_t instanceCount;
};

uint32_t LCDylib::instanceCount = 0;

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

class LCRPath : public LoadCommand {
public:
  LCRPath(StringRef path) : path(path) {}

  uint32_t getSize() const override {
    return alignTo(sizeof(rpath_command) + path.size() + 1, WordSize);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<rpath_command *>(buf);
    buf += sizeof(rpath_command);

    c->cmd = LC_RPATH;
    c->cmdsize = getSize();
    c->path = sizeof(rpath_command);

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

private:
  StringRef path;
};

class LCBuildVersion : public LoadCommand {
public:
  LCBuildVersion(const PlatformInfo &platform) : platform(platform) {}

  const int ntools = 1;

  uint32_t getSize() const override {
    return sizeof(build_version_command) + ntools * sizeof(build_tool_version);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<build_version_command *>(buf);
    c->cmd = LC_BUILD_VERSION;
    c->cmdsize = getSize();
    c->platform = static_cast<uint32_t>(platform.kind);
    c->minos = ((platform.minimum.getMajor() << 020) |
                (platform.minimum.getMinor().getValueOr(0) << 010) |
                platform.minimum.getSubminor().getValueOr(0));
    c->sdk = ((platform.sdk.getMajor() << 020) |
              (platform.sdk.getMinor().getValueOr(0) << 010) |
              platform.sdk.getSubminor().getValueOr(0));
    c->ntools = ntools;
    auto *t = reinterpret_cast<build_tool_version *>(&c[1]);
    t->tool = TOOL_LD;
    t->version = (LLVM_VERSION_MAJOR << 020) | (LLVM_VERSION_MINOR << 010) |
                 LLVM_VERSION_PATCH;
  }

  const PlatformInfo &platform;
};

// Stores a unique identifier for the output file based on an MD5 hash of its
// contents. In order to hash the contents, we must first write them, but
// LC_UUID itself must be part of the written contents in order for all the
// offsets to be calculated correctly. We resolve this circular paradox by
// first writing an LC_UUID with an all-zero UUID, then updating the UUID with
// its real value later.
class LCUuid : public LoadCommand {
public:
  uint32_t getSize() const override { return sizeof(uuid_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<uuid_command *>(buf);
    c->cmd = LC_UUID;
    c->cmdsize = getSize();
    uuidBuf = c->uuid;
  }

  void writeUuid(const std::array<uint8_t, 16> &uuid) const {
    memcpy(uuidBuf, uuid.data(), uuid.size());
  }

  mutable uint8_t *uuidBuf;
};

} // namespace

void Writer::scanRelocations() {
  for (InputSection *isec : inputSections) {
    // We do not wish to add rebase opcodes for __LD,__compact_unwind, because
    // it doesn't actually end up in the final binary. TODO: filtering it out
    // before Writer runs might be cleaner...
    if (isec->segname == segment_names::ld)
      continue;

    for (Reloc &r : isec->relocs) {
      if (auto *s = r.referent.dyn_cast<lld::macho::Symbol *>()) {
        if (isa<Undefined>(s))
          error("undefined symbol " + s->getName() + ", referenced from " +
                sys::path::filename(isec->file->getName()));
        else
          target->prepareSymbolRelocation(s, isec, r);
      } else {
        assert(r.referent.is<InputSection *>());
        if (!r.pcrel)
          in.rebase->addEntry(isec, r.offset);
      }
    }
  }
}

void Writer::createLoadCommands() {
  in.header->addLoadCommand(make<LCDyldInfo>(
      in.rebase, in.binding, in.weakBinding, in.lazyBinding, in.exports));
  in.header->addLoadCommand(make<LCSymtab>(symtabSection, stringTableSection));
  in.header->addLoadCommand(make<LCDysymtab>(indirectSymtabSection));
  for (StringRef path : config->runtimePaths)
    in.header->addLoadCommand(make<LCRPath>(path));

  switch (config->outputType) {
  case MH_EXECUTE:
    in.header->addLoadCommand(make<LCMain>());
    in.header->addLoadCommand(make<LCLoadDylinker>());
    break;
  case MH_DYLIB:
    in.header->addLoadCommand(make<LCDylib>(LC_ID_DYLIB, config->installName));
    break;
  case MH_BUNDLE:
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }

  in.header->addLoadCommand(make<LCBuildVersion>(config->platform));

  uuidCommand = make<LCUuid>();
  in.header->addLoadCommand(uuidCommand);

  uint8_t segIndex = 0;
  for (OutputSegment *seg : outputSegments) {
    in.header->addLoadCommand(make<LCSegment>(seg->name, seg));
    seg->index = segIndex++;
  }

  uint64_t dylibOrdinal = 1;
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      // TODO: dylibs that are only referenced by weak refs should also be
      // loaded via LC_LOAD_WEAK_DYLIB.
      LoadCommandType lcType =
          dylibFile->forceWeakImport ? LC_LOAD_WEAK_DYLIB : LC_LOAD_DYLIB;
      in.header->addLoadCommand(make<LCDylib>(lcType, dylibFile->dylibName));
      dylibFile->ordinal = dylibOrdinal++;

      if (dylibFile->reexport)
        in.header->addLoadCommand(
            make<LCDylib>(LC_REEXPORT_DYLIB, dylibFile->dylibName));
    }
  }

  const uint32_t MACOS_MAXPATHLEN = 1024;
  config->headerPad = std::max(
      config->headerPad, (config->headerPadMaxInstallNames
                              ? LCDylib::getInstanceCount() * MACOS_MAXPATHLEN
                              : 0));
}

static size_t getSymbolPriority(const SymbolPriorityEntry &entry,
                                const InputFile &file) {
  return std::max(entry.objectFiles.lookup(sys::path::filename(file.getName())),
                  entry.anyObjectFile);
}

// Each section gets assigned the priority of the highest-priority symbol it
// contains.
static DenseMap<const InputSection *, size_t> buildInputSectionPriorities() {
  DenseMap<const InputSection *, size_t> sectionPriorities;

  if (config->priorities.empty())
    return sectionPriorities;

  auto addSym = [&](Defined &sym) {
    auto it = config->priorities.find(sym.getName());
    if (it == config->priorities.end())
      return;

    SymbolPriorityEntry &entry = it->second;
    size_t &priority = sectionPriorities[sym.isec];
    priority = std::max(priority, getSymbolPriority(entry, *sym.isec->file));
  };

  // TODO: Make sure this handles weak symbols correctly.
  for (InputFile *file : inputFiles)
    if (isa<ObjFile>(file) || isa<ArchiveFile>(file))
      for (lld::macho::Symbol *sym : file->symbols)
        if (auto *d = dyn_cast<Defined>(sym))
          addSym(*d);

  return sectionPriorities;
}

static int segmentOrder(OutputSegment *seg) {
  return StringSwitch<int>(seg->name)
      .Case(segment_names::pageZero, -2)
      .Case(segment_names::text, -1)
      // Make sure __LINKEDIT is the last segment (i.e. all its hidden
      // sections must be ordered after other sections).
      .Case(segment_names::linkEdit, std::numeric_limits<int>::max())
      .Default(0);
}

static int sectionOrder(OutputSection *osec) {
  StringRef segname = osec->parent->name;
  // Sections are uniquely identified by their segment + section name.
  if (segname == segment_names::text) {
    return StringSwitch<int>(osec->name)
        .Case(section_names::header, -1)
        .Case(section_names::unwindInfo, std::numeric_limits<int>::max() - 1)
        .Case(section_names::ehFrame, std::numeric_limits<int>::max())
        .Default(0);
  } else if (segname == segment_names::linkEdit) {
    return StringSwitch<int>(osec->name)
        .Case(section_names::rebase, -8)
        .Case(section_names::binding, -7)
        .Case(section_names::weakBinding, -6)
        .Case(section_names::lazyBinding, -5)
        .Case(section_names::export_, -4)
        .Case(section_names::symbolTable, -3)
        .Case(section_names::indirectSymbolTable, -2)
        .Case(section_names::stringTable, -1)
        .Default(0);
  }
  // ZeroFill sections must always be the at the end of their segments,
  // otherwise subsequent sections may get overwritten with zeroes at runtime.
  if (isZeroFill(osec->flags))
    return std::numeric_limits<int>::max();
  return 0;
}

template <typename T, typename F>
static std::function<bool(T, T)> compareByOrder(F ord) {
  return [=](T a, T b) { return ord(a) < ord(b); };
}

// Sorting only can happen once all outputs have been collected. Here we sort
// segments, output sections within each segment, and input sections within each
// output segment.
static void sortSegmentsAndSections() {
  llvm::stable_sort(outputSegments,
                    compareByOrder<OutputSegment *>(segmentOrder));

  DenseMap<const InputSection *, size_t> isecPriorities =
      buildInputSectionPriorities();

  uint32_t sectionIndex = 0;
  for (OutputSegment *seg : outputSegments) {
    seg->sortOutputSections(compareByOrder<OutputSection *>(sectionOrder));
    for (auto *osec : seg->getSections()) {
      // Now that the output sections are sorted, assign the final
      // output section indices.
      if (!osec->isHidden())
        osec->index = ++sectionIndex;

      if (!isecPriorities.empty()) {
        if (auto *merged = dyn_cast<MergedOutputSection>(osec)) {
          llvm::stable_sort(merged->inputs,
                            [&](InputSection *a, InputSection *b) {
                              return isecPriorities[a] > isecPriorities[b];
                            });
        }
      }
    }
  }
}

void Writer::createOutputSections() {
  // First, create hidden sections
  stringTableSection = make<StringTableSection>();
  unwindInfoSection = make<UnwindInfoSection>(); // TODO(gkm): only when no -r
  symtabSection = make<SymtabSection>(*stringTableSection);
  indirectSymtabSection = make<IndirectSymtabSection>();

  switch (config->outputType) {
  case MH_EXECUTE:
    make<PageZeroSection>();
    break;
  case MH_DYLIB:
  case MH_BUNDLE:
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }

  // Then merge input sections into output sections.
  MapVector<std::pair<StringRef, StringRef>, MergedOutputSection *>
      mergedOutputSections;
  for (InputSection *isec : inputSections) {
    MergedOutputSection *&osec =
        mergedOutputSections[{isec->segname, isec->name}];
    if (osec == nullptr)
      osec = make<MergedOutputSection>(isec->name);
    osec->mergeInput(isec);
  }

  for (const auto &it : mergedOutputSections) {
    StringRef segname = it.first.first;
    MergedOutputSection *osec = it.second;
    if (unwindInfoSection && segname == segment_names::ld) {
      assert(osec->name == section_names::compactUnwind);
      unwindInfoSection->setCompactUnwindSection(osec);
    } else
      getOrCreateOutputSegment(segname)->addOutputSection(osec);
  }

  for (SyntheticSection *ssec : syntheticSections) {
    auto it = mergedOutputSections.find({ssec->segname, ssec->name});
    if (it == mergedOutputSections.end()) {
      if (ssec->isNeeded())
        getOrCreateOutputSegment(ssec->segname)->addOutputSection(ssec);
    } else {
      error("section from " + it->second->firstSection()->file->getName() +
            " conflicts with synthetic section " + ssec->segname + "," +
            ssec->name);
    }
  }
}

void Writer::assignAddresses(OutputSegment *seg) {
  addr = alignTo(addr, PageSize);
  fileOff = alignTo(fileOff, PageSize);
  seg->fileOff = fileOff;

  for (auto *osec : seg->getSections()) {
    if (!osec->isNeeded())
      continue;
    addr = alignTo(addr, osec->align);
    fileOff = alignTo(fileOff, osec->align);
    osec->addr = addr;
    osec->fileOff = isZeroFill(osec->flags) ? 0 : fileOff;
    osec->finalize();

    addr += osec->getSize();
    fileOff += osec->getFileSize();
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
  for (OutputSegment *seg : outputSegments)
    for (OutputSection *osec : seg->getSections())
      osec->writeTo(buf + osec->fileOff);
}

void Writer::writeUuid() {
  MD5 hash;
  const auto *bufStart = reinterpret_cast<char *>(buffer->getBufferStart());
  const auto *bufEnd = reinterpret_cast<char *>(buffer->getBufferEnd());
  hash.update(StringRef(bufStart, bufEnd - bufStart));
  MD5::MD5Result result;
  hash.final(result);
  // Conform to UUID version 4 & 5 as specified in RFC 4122:
  // 1. Set the version field to indicate that this is an MD5-based UUID.
  result.Bytes[6] = (result.Bytes[6] & 0xf) | 0x30;
  // 2. Set the two MSBs of uuid_t::clock_seq_hi_and_reserved to zero and one.
  result.Bytes[8] = (result.Bytes[8] & 0x3f) | 0x80;
  uuidCommand->writeUuid(result.Bytes);
}

void Writer::run() {
  // dyld requires __LINKEDIT segment to always exist (even if empty).
  OutputSegment *linkEditSegment =
      getOrCreateOutputSegment(segment_names::linkEdit);

  prepareBranchTarget(config->entry);
  scanRelocations();
  if (in.stubHelper->isNeeded())
    in.stubHelper->setup();

  for (const macho::Symbol *sym : symtab->getSymbols())
    if (const auto *defined = dyn_cast<Defined>(sym))
      if (defined->overridesWeakDef)
        in.weakBinding->addNonWeakDefinition(defined);

  // Sort and assign sections to their respective segments. No more sections nor
  // segments may be created after these methods run.
  createOutputSections();
  sortSegmentsAndSections();

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
  in.rebase->finalizeContents();
  in.binding->finalizeContents();
  in.weakBinding->finalizeContents();
  in.lazyBinding->finalizeContents();
  in.exports->finalizeContents();
  symtabSection->finalizeContents();
  indirectSymtabSection->finalizeContents();

  // Now that __LINKEDIT is filled out, do a proper calculation of its
  // addresses and offsets.
  assignAddresses(linkEditSegment);

  openFile();
  if (errorCount())
    return;

  writeSections();
  writeUuid();

  if (auto e = buffer->commit())
    error("failed to write to the output file: " + toString(std::move(e)));
}

void macho::writeResult() { Writer().run(); }

void macho::createSyntheticSections() {
  in.header = make<MachHeaderSection>();
  in.rebase = make<RebaseSection>();
  in.binding = make<BindingSection>();
  in.weakBinding = make<WeakBindingSection>();
  in.lazyBinding = make<LazyBindingSection>();
  in.exports = make<ExportSection>();
  in.got = make<GotSection>();
  in.tlvPointers = make<TlvPointerSection>();
  in.lazyPointers = make<LazyPointerSection>();
  in.stubs = make<StubsSection>();
  in.stubHelper = make<StubHelperSection>();
  in.imageLoaderCache = make<ImageLoaderCacheSection>();
}
