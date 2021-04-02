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
#include "MapFile.h"
#include "MergedOutputSection.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "UnwindInfoSection.h"

#include "lld/Common/Arrays.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

namespace {
class LCUuid;

class Writer {
public:
  Writer() : buffer(errorHandler().outputBuffer) {}

  void scanRelocations();
  void scanSymbols();
  void createOutputSections();
  void createLoadCommands();
  void finalizeAddressses();
  void finalizeLinkEditSegment();
  void assignAddresses(OutputSegment *);

  void openFile();
  void writeSections();
  void writeUuid();
  void writeCodeSignature();
  void writeOutputFile();

  void run();

  std::unique_ptr<FileOutputBuffer> &buffer;
  uint64_t addr = 0;
  uint64_t fileOff = 0;
  MachHeaderSection *header = nullptr;
  StringTableSection *stringTableSection = nullptr;
  SymtabSection *symtabSection = nullptr;
  IndirectSymtabSection *indirectSymtabSection = nullptr;
  CodeSignatureSection *codeSignatureSection = nullptr;
  UnwindInfoSection *unwindInfoSection = nullptr;
  FunctionStartsSection *functionStartsSection = nullptr;

  LCUuid *uuidCommand = nullptr;
  OutputSegment *linkEditSegment = nullptr;
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

class LCFunctionStarts : public LoadCommand {
public:
  explicit LCFunctionStarts(FunctionStartsSection *functionStartsSection)
      : functionStartsSection(functionStartsSection) {}

  uint32_t getSize() const override { return sizeof(linkedit_data_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<linkedit_data_command *>(buf);
    c->cmd = LC_FUNCTION_STARTS;
    c->cmdsize = getSize();
    c->dataoff = functionStartsSection->fileOff;
    c->datasize = functionStartsSection->getFileSize();
  }

private:
  FunctionStartsSection *functionStartsSection;
};

class LCDysymtab : public LoadCommand {
public:
  LCDysymtab(SymtabSection *symtabSection,
             IndirectSymtabSection *indirectSymtabSection)
      : symtabSection(symtabSection),
        indirectSymtabSection(indirectSymtabSection) {}

  uint32_t getSize() const override { return sizeof(dysymtab_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<dysymtab_command *>(buf);
    c->cmd = LC_DYSYMTAB;
    c->cmdsize = getSize();

    c->ilocalsym = 0;
    c->iextdefsym = c->nlocalsym = symtabSection->getNumLocalSymbols();
    c->nextdefsym = symtabSection->getNumExternalSymbols();
    c->iundefsym = c->iextdefsym + c->nextdefsym;
    c->nundefsym = symtabSection->getNumUndefinedSymbols();

    c->indirectsymoff = indirectSymtabSection->fileOff;
    c->nindirectsyms = indirectSymtabSection->getNumSymbols();
  }

  SymtabSection *symtabSection;
  IndirectSymtabSection *indirectSymtabSection;
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

    for (const OutputSection *osec : seg->getSections()) {
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
  LCDylib(LoadCommandType type, StringRef path,
          uint32_t compatibilityVersion = 0, uint32_t currentVersion = 0)
      : type(type), path(path), compatibilityVersion(compatibilityVersion),
        currentVersion(currentVersion) {
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
    c->dylib.timestamp = 0;
    c->dylib.compatibility_version = compatibilityVersion;
    c->dylib.current_version = currentVersion;

    memcpy(buf, path.data(), path.size());
    buf[path.size()] = '\0';
  }

  static uint32_t getInstanceCount() { return instanceCount; }

private:
  LoadCommandType type;
  StringRef path;
  uint32_t compatibilityVersion;
  uint32_t currentVersion;
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
  LCBuildVersion(PlatformKind platform, const PlatformInfo &platformInfo)
      : platform(platform), platformInfo(platformInfo) {}

  const int ntools = 1;

  uint32_t getSize() const override {
    return sizeof(build_version_command) + ntools * sizeof(build_tool_version);
  }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<build_version_command *>(buf);
    c->cmd = LC_BUILD_VERSION;
    c->cmdsize = getSize();
    c->platform = static_cast<uint32_t>(platform);
    c->minos = ((platformInfo.minimum.getMajor() << 020) |
                (platformInfo.minimum.getMinor().getValueOr(0) << 010) |
                platformInfo.minimum.getSubminor().getValueOr(0));
    c->sdk = ((platformInfo.sdk.getMajor() << 020) |
              (platformInfo.sdk.getMinor().getValueOr(0) << 010) |
              platformInfo.sdk.getSubminor().getValueOr(0));
    c->ntools = ntools;
    auto *t = reinterpret_cast<build_tool_version *>(&c[1]);
    t->tool = TOOL_LD;
    t->version = (LLVM_VERSION_MAJOR << 020) | (LLVM_VERSION_MINOR << 010) |
                 LLVM_VERSION_PATCH;
  }

  PlatformKind platform;
  const PlatformInfo &platformInfo;
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

  void writeUuid(uint64_t digest) const {
    // xxhash only gives us 8 bytes, so put some fixed data in the other half.
    static_assert(sizeof(uuid_command::uuid) == 16, "unexpected uuid size");
    memcpy(uuidBuf, "LLD\xa1UU1D", 8);
    memcpy(uuidBuf + 8, &digest, 8);

    // RFC 4122 conformance. We need to fix 4 bits in byte 6 and 2 bits in
    // byte 8. Byte 6 is already fine due to the fixed data we put in. We don't
    // want to lose bits of the digest in byte 8, so swap that with a byte of
    // fixed data that happens to have the right bits set.
    std::swap(uuidBuf[3], uuidBuf[8]);

    // Claim that this is an MD5-based hash. It isn't, but this signals that
    // this is not a time-based and not a random hash. MD5 seems like the least
    // bad lie we can put here.
    assert((uuidBuf[6] & 0xf0) == 0x30 && "See RFC 4122 Sections 4.2.2, 4.1.3");
    assert((uuidBuf[8] & 0xc0) == 0x80 && "See RFC 4122 Section 4.2.2");
  }

  mutable uint8_t *uuidBuf;
};

class LCCodeSignature : public LoadCommand {
public:
  LCCodeSignature(CodeSignatureSection *section) : section(section) {}

  uint32_t getSize() const override { return sizeof(linkedit_data_command); }

  void writeTo(uint8_t *buf) const override {
    auto *c = reinterpret_cast<linkedit_data_command *>(buf);
    c->cmd = LC_CODE_SIGNATURE;
    c->cmdsize = getSize();
    c->dataoff = static_cast<uint32_t>(section->fileOff);
    c->datasize = section->getSize();
  }

  CodeSignatureSection *section;
};

} // namespace

// Adds stubs and bindings where necessary (e.g. if the symbol is a
// DylibSymbol.)
static void prepareBranchTarget(Symbol *sym) {
  if (auto *dysym = dyn_cast<DylibSymbol>(sym)) {
    if (in.stubs->addEntry(dysym)) {
      if (sym->isWeakDef()) {
        in.binding->addEntry(dysym, in.lazyPointers->isec,
                             sym->stubsIndex * WordSize);
        in.weakBinding->addEntry(sym, in.lazyPointers->isec,
                                 sym->stubsIndex * WordSize);
      } else {
        in.lazyBinding->addEntry(dysym);
      }
    }
  } else if (auto *defined = dyn_cast<Defined>(sym)) {
    if (defined->isExternalWeakDef()) {
      if (in.stubs->addEntry(sym)) {
        in.rebase->addEntry(in.lazyPointers->isec, sym->stubsIndex * WordSize);
        in.weakBinding->addEntry(sym, in.lazyPointers->isec,
                                 sym->stubsIndex * WordSize);
      }
    }
  }
}

// Can a symbol's address can only be resolved at runtime?
static bool needsBinding(const Symbol *sym) {
  if (isa<DylibSymbol>(sym))
    return true;
  if (const auto *defined = dyn_cast<Defined>(sym))
    return defined->isExternalWeakDef();
  return false;
}

static void prepareSymbolRelocation(Symbol *sym, const InputSection *isec,
                                    const Reloc &r) {
  const RelocAttrs &relocAttrs = target->getRelocAttrs(r.type);

  if (relocAttrs.hasAttr(RelocAttrBits::BRANCH)) {
    prepareBranchTarget(sym);
  } else if (relocAttrs.hasAttr(RelocAttrBits::GOT)) {
    if (relocAttrs.hasAttr(RelocAttrBits::POINTER) || needsBinding(sym))
      in.got->addEntry(sym);
  } else if (relocAttrs.hasAttr(RelocAttrBits::TLV)) {
    if (needsBinding(sym))
      in.tlvPointers->addEntry(sym);
  } else if (relocAttrs.hasAttr(RelocAttrBits::UNSIGNED)) {
    // References from thread-local variable sections are treated as offsets
    // relative to the start of the referent section, and therefore have no
    // need of rebase opcodes.
    if (!(isThreadLocalVariables(isec->flags) && isa<Defined>(sym)))
      addNonLazyBindingEntries(sym, isec, r.offset, r.addend);
  }
}

void Writer::scanRelocations() {
  TimeTraceScope timeScope("Scan relocations");
  for (InputSection *isec : inputSections) {
    if (isec->segname == segment_names::ld) {
      prepareCompactUnwind(isec);
      continue;
    }

    for (auto it = isec->relocs.begin(); it != isec->relocs.end(); ++it) {
      Reloc &r = *it;
      if (target->hasAttr(r.type, RelocAttrBits::SUBTRAHEND)) {
        // Skip over the following UNSIGNED relocation -- it's just there as the
        // minuend, and doesn't have the usual UNSIGNED semantics. We don't want
        // to emit rebase opcodes for it.
        it = std::next(it);
        assert(isa<Defined>(it->referent.dyn_cast<Symbol *>()));
        continue;
      }
      if (auto *sym = r.referent.dyn_cast<Symbol *>()) {
        if (auto *undefined = dyn_cast<Undefined>(sym))
          treatUndefinedSymbol(*undefined);
        // treatUndefinedSymbol() can replace sym with a DylibSymbol; re-check.
        if (!isa<Undefined>(sym) && validateSymbolRelocation(sym, isec, r))
          prepareSymbolRelocation(sym, isec, r);
      } else {
        assert(r.referent.is<InputSection *>());
        if (!r.pcrel)
          in.rebase->addEntry(isec, r.offset);
      }
    }
  }
}

void Writer::scanSymbols() {
  TimeTraceScope timeScope("Scan symbols");
  for (const Symbol *sym : symtab->getSymbols()) {
    if (const auto *defined = dyn_cast<Defined>(sym)) {
      if (defined->overridesWeakDef)
        in.weakBinding->addNonWeakDefinition(defined);
    } else if (const auto *dysym = dyn_cast<DylibSymbol>(sym)) {
      if (dysym->isDynamicLookup())
        continue;
      dysym->getFile()->refState =
          std::max(dysym->getFile()->refState, dysym->refState);
    }
  }
}

void Writer::createLoadCommands() {
  uint8_t segIndex = 0;
  for (OutputSegment *seg : outputSegments) {
    in.header->addLoadCommand(make<LCSegment>(seg->name, seg));
    seg->index = segIndex++;
  }

  in.header->addLoadCommand(make<LCDyldInfo>(
      in.rebase, in.binding, in.weakBinding, in.lazyBinding, in.exports));
  in.header->addLoadCommand(make<LCSymtab>(symtabSection, stringTableSection));
  in.header->addLoadCommand(
      make<LCDysymtab>(symtabSection, indirectSymtabSection));
  if (functionStartsSection)
    in.header->addLoadCommand(make<LCFunctionStarts>(functionStartsSection));
  for (StringRef path : config->runtimePaths)
    in.header->addLoadCommand(make<LCRPath>(path));

  switch (config->outputType) {
  case MH_EXECUTE:
    in.header->addLoadCommand(make<LCLoadDylinker>());
    in.header->addLoadCommand(make<LCMain>());
    break;
  case MH_DYLIB:
    in.header->addLoadCommand(make<LCDylib>(LC_ID_DYLIB, config->installName,
                                            config->dylibCompatibilityVersion,
                                            config->dylibCurrentVersion));
    break;
  case MH_BUNDLE:
    break;
  default:
    llvm_unreachable("unhandled output file type");
  }

  uuidCommand = make<LCUuid>();
  in.header->addLoadCommand(uuidCommand);

  in.header->addLoadCommand(
      make<LCBuildVersion>(config->target.Platform, config->platformInfo));

  int64_t dylibOrdinal = 1;
  for (InputFile *file : inputFiles) {
    if (auto *dylibFile = dyn_cast<DylibFile>(file)) {
      if (dylibFile->isBundleLoader) {
        dylibFile->ordinal = BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE;
        // Shortcut since bundle-loader does not re-export the symbols.

        dylibFile->reexport = false;
        continue;
      }

      dylibFile->ordinal = dylibOrdinal++;
      LoadCommandType lcType =
          dylibFile->forceWeakImport || dylibFile->refState == RefState::Weak
              ? LC_LOAD_WEAK_DYLIB
              : LC_LOAD_DYLIB;
      in.header->addLoadCommand(make<LCDylib>(lcType, dylibFile->dylibName,
                                              dylibFile->compatibilityVersion,
                                              dylibFile->currentVersion));

      if (dylibFile->reexport)
        in.header->addLoadCommand(
            make<LCDylib>(LC_REEXPORT_DYLIB, dylibFile->dylibName));
    }
  }

  if (codeSignatureSection)
    in.header->addLoadCommand(make<LCCodeSignature>(codeSignatureSection));

  const uint32_t MACOS_MAXPATHLEN = 1024;
  config->headerPad = std::max(
      config->headerPad, (config->headerPadMaxInstallNames
                              ? LCDylib::getInstanceCount() * MACOS_MAXPATHLEN
                              : 0));
}

static size_t getSymbolPriority(const SymbolPriorityEntry &entry,
                                const InputFile *f) {
  // We don't use toString(InputFile *) here because it returns the full path
  // for object files, and we only want the basename.
  StringRef filename;
  if (f->archiveName.empty())
    filename = path::filename(f->getName());
  else
    filename = saver.save(path::filename(f->archiveName) + "(" +
                          path::filename(f->getName()) + ")");
  return std::max(entry.objectFiles.lookup(filename), entry.anyObjectFile);
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
    priority = std::max(priority, getSymbolPriority(entry, sym.isec->file));
  };

  // TODO: Make sure this handles weak symbols correctly.
  for (const InputFile *file : inputFiles) {
    if (isa<ObjFile>(file))
      for (Symbol *sym : file->symbols)
        if (auto *d = dyn_cast<Defined>(sym))
          addSym(*d);
  }

  return sectionPriorities;
}

static int segmentOrder(OutputSegment *seg) {
  return StringSwitch<int>(seg->name)
      .Case(segment_names::pageZero, -4)
      .Case(segment_names::text, -3)
      .Case(segment_names::dataConst, -2)
      .Case(segment_names::data, -1)
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
        .Case(section_names::header, -4)
        .Case(section_names::text, -3)
        .Case(section_names::stubs, -2)
        .Case(section_names::stubHelper, -1)
        .Case(section_names::unwindInfo, std::numeric_limits<int>::max() - 1)
        .Case(section_names::ehFrame, std::numeric_limits<int>::max())
        .Default(0);
  } else if (segname == segment_names::data) {
    // For each thread spawned, dyld will initialize its TLVs by copying the
    // address range from the start of the first thread-local data section to
    // the end of the last one. We therefore arrange these sections contiguously
    // to minimize the amount of memory used. Additionally, since zerofill
    // sections must be at the end of their segments, and since TLV data
    // sections can be zerofills, we end up putting all TLV data sections at the
    // end of the segment.
    switch (sectionType(osec->flags)) {
    case S_THREAD_LOCAL_REGULAR:
      return std::numeric_limits<int>::max() - 2;
    case S_THREAD_LOCAL_ZEROFILL:
      return std::numeric_limits<int>::max() - 1;
    case S_ZEROFILL:
      return std::numeric_limits<int>::max();
    default:
      return StringSwitch<int>(osec->name)
          .Case(section_names::laSymbolPtr, -2)
          .Case(section_names::data, -1)
          .Default(0);
    }
  } else if (segname == segment_names::linkEdit) {
    return StringSwitch<int>(osec->name)
        .Case(section_names::rebase, -9)
        .Case(section_names::binding, -8)
        .Case(section_names::weakBinding, -7)
        .Case(section_names::lazyBinding, -6)
        .Case(section_names::export_, -5)
        .Case(section_names::functionStarts, -4)
        .Case(section_names::symbolTable, -3)
        .Case(section_names::indirectSymbolTable, -2)
        .Case(section_names::stringTable, -1)
        .Case(section_names::codeSignature, std::numeric_limits<int>::max())
        .Default(0);
  }
  // ZeroFill sections must always be the at the end of their segments,
  // otherwise subsequent sections may get overwritten with zeroes at runtime.
  if (sectionType(osec->flags) == S_ZEROFILL)
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
  TimeTraceScope timeScope("Sort segments and sections");

  llvm::stable_sort(outputSegments,
                    compareByOrder<OutputSegment *>(segmentOrder));

  DenseMap<const InputSection *, size_t> isecPriorities =
      buildInputSectionPriorities();

  uint32_t sectionIndex = 0;
  for (OutputSegment *seg : outputSegments) {
    seg->sortOutputSections(compareByOrder<OutputSection *>(sectionOrder));
    for (OutputSection *osec : seg->getSections()) {
      // Now that the output sections are sorted, assign the final
      // output section indices.
      if (!osec->isHidden())
        osec->index = ++sectionIndex;
      if (!firstTLVDataSection && isThreadLocalData(osec->flags))
        firstTLVDataSection = osec;

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

static NamePair maybeRenameSection(NamePair key) {
  auto newNames = config->sectionRenameMap.find(key);
  if (newNames != config->sectionRenameMap.end())
    return newNames->second;
  auto newName = config->segmentRenameMap.find(key.first);
  if (newName != config->segmentRenameMap.end())
    return std::make_pair(newName->second, key.second);
  return key;
}

void Writer::createOutputSections() {
  TimeTraceScope timeScope("Create output sections");
  // First, create hidden sections
  stringTableSection = make<StringTableSection>();
  unwindInfoSection = make<UnwindInfoSection>(); // TODO(gkm): only when no -r
  symtabSection = make<SymtabSection>(*stringTableSection);
  indirectSymtabSection = make<IndirectSymtabSection>();
  if (config->adhocCodesign)
    codeSignatureSection = make<CodeSignatureSection>();
  if (config->emitFunctionStarts)
    functionStartsSection = make<FunctionStartsSection>();

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
  MapVector<NamePair, MergedOutputSection *> mergedOutputSections;
  for (InputSection *isec : inputSections) {
    NamePair names = maybeRenameSection({isec->segname, isec->name});
    MergedOutputSection *&osec = mergedOutputSections[names];
    if (osec == nullptr)
      osec = make<MergedOutputSection>(names.second);
    osec->mergeInput(isec);
  }

  for (const auto &it : mergedOutputSections) {
    StringRef segname = it.first.first;
    MergedOutputSection *osec = it.second;
    if (unwindInfoSection && segname == segment_names::ld) {
      assert(osec->name == section_names::compactUnwind);
      unwindInfoSection->setCompactUnwindSection(osec);
    } else {
      getOrCreateOutputSegment(segname)->addOutputSection(osec);
    }
  }

  for (SyntheticSection *ssec : syntheticSections) {
    auto it = mergedOutputSections.find({ssec->segname, ssec->name});
    if (it == mergedOutputSections.end()) {
      if (ssec->isNeeded())
        getOrCreateOutputSegment(ssec->segname)->addOutputSection(ssec);
    } else {
      error("section from " + toString(it->second->firstSection()->file) +
            " conflicts with synthetic section " + ssec->segname + "," +
            ssec->name);
    }
  }

  // dyld requires __LINKEDIT segment to always exist (even if empty).
  linkEditSegment = getOrCreateOutputSegment(segment_names::linkEdit);
}

void Writer::finalizeAddressses() {
  TimeTraceScope timeScope("Finalize addresses");
  // Ensure that segments (and the sections they contain) are allocated
  // addresses in ascending order, which dyld requires.
  //
  // Note that at this point, __LINKEDIT sections are empty, but we need to
  // determine addresses of other segments/sections before generating its
  // contents.
  for (OutputSegment *seg : outputSegments)
    if (seg != linkEditSegment)
      assignAddresses(seg);

  // FIXME(gkm): create branch-extension thunks here, then adjust addresses
}

void Writer::finalizeLinkEditSegment() {
  TimeTraceScope timeScope("Finalize __LINKEDIT segment");
  // Fill __LINKEDIT contents.
  in.rebase->finalizeContents();
  in.binding->finalizeContents();
  in.weakBinding->finalizeContents();
  in.lazyBinding->finalizeContents();
  in.exports->finalizeContents();
  symtabSection->finalizeContents();
  indirectSymtabSection->finalizeContents();

  if (functionStartsSection)
    functionStartsSection->finalizeContents();

  // Now that __LINKEDIT is filled out, do a proper calculation of its
  // addresses and offsets.
  assignAddresses(linkEditSegment);
}

void Writer::assignAddresses(OutputSegment *seg) {
  uint64_t pageSize = target->getPageSize();
  addr = alignTo(addr, pageSize);
  fileOff = alignTo(fileOff, pageSize);
  seg->fileOff = fileOff;

  for (OutputSection *osec : seg->getSections()) {
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
  seg->fileSize = fileOff - seg->fileOff;
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
  for (const OutputSegment *seg : outputSegments)
    for (const OutputSection *osec : seg->getSections())
      osec->writeTo(buf + osec->fileOff);
}

// In order to utilize multiple cores, we first split the buffer into chunks,
// compute a hash for each chunk, and then compute a hash value of the hash
// values.
void Writer::writeUuid() {
  TimeTraceScope timeScope("Computing UUID");
  ArrayRef<uint8_t> data{buffer->getBufferStart(), buffer->getBufferEnd()};
  unsigned chunkCount = parallel::strategy.compute_thread_count() * 10;
  // Round-up integer division
  size_t chunkSize = (data.size() + chunkCount - 1) / chunkCount;
  std::vector<ArrayRef<uint8_t>> chunks = split(data, chunkSize);
  std::vector<uint64_t> hashes(chunks.size());
  parallelForEachN(0, chunks.size(),
                   [&](size_t i) { hashes[i] = xxHash64(chunks[i]); });
  uint64_t digest = xxHash64({reinterpret_cast<uint8_t *>(hashes.data()),
                              hashes.size() * sizeof(uint64_t)});
  uuidCommand->writeUuid(digest);
}

void Writer::writeCodeSignature() {
  if (codeSignatureSection)
    codeSignatureSection->writeHashes(buffer->getBufferStart());
}

void Writer::writeOutputFile() {
  TimeTraceScope timeScope("Write output file");
  openFile();
  if (errorCount())
    return;
  writeSections();
  writeUuid();
  writeCodeSignature();

  if (auto e = buffer->commit())
    error("failed to write to the output file: " + toString(std::move(e)));
}

void Writer::run() {
  prepareBranchTarget(config->entry);
  scanRelocations();
  if (in.stubHelper->isNeeded())
    in.stubHelper->setup();
  scanSymbols();
  createOutputSections();
  // No more sections nor segments are created beyond this point.
  sortSegmentsAndSections();
  createLoadCommands();
  finalizeAddressses();
  finalizeLinkEditSegment();
  writeMapFile();
  writeOutputFile();
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

OutputSection *macho::firstTLVDataSection = nullptr;
