//===- lib/ReaderWriter/ELF/TargetLayout.cpp ------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TargetLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Errc.h"

namespace lld {
namespace elf {

template <class ELFT>
typename TargetLayout<ELFT>::SectionOrder
TargetLayout<ELFT>::getSectionOrder(StringRef name, int32_t contentType,
                                    int32_t contentPermissions) {
  switch (contentType) {
  case DefinedAtom::typeResolver:
  case DefinedAtom::typeCode:
    return llvm::StringSwitch<typename TargetLayout<ELFT>::SectionOrder>(name)
        .StartsWith(".eh_frame_hdr", ORDER_EH_FRAMEHDR)
        .StartsWith(".eh_frame", ORDER_EH_FRAME)
        .StartsWith(".init", ORDER_INIT)
        .StartsWith(".fini", ORDER_FINI)
        .StartsWith(".hash", ORDER_HASH)
        .Default(ORDER_TEXT);

  case DefinedAtom::typeConstant:
    return ORDER_RODATA;

  case DefinedAtom::typeData:
  case DefinedAtom::typeDataFast:
    return llvm::StringSwitch<typename TargetLayout<ELFT>::SectionOrder>(name)
        .StartsWith(".init_array", ORDER_INIT_ARRAY)
        .StartsWith(".fini_array", ORDER_FINI_ARRAY)
        .StartsWith(".dynamic", ORDER_DYNAMIC)
        .StartsWith(".ctors", ORDER_CTORS)
        .StartsWith(".dtors", ORDER_DTORS)
        .Default(ORDER_DATA);

  case DefinedAtom::typeZeroFill:
  case DefinedAtom::typeZeroFillFast:
    return ORDER_BSS;

  case DefinedAtom::typeGOT:
    return llvm::StringSwitch<typename TargetLayout<ELFT>::SectionOrder>(name)
        .StartsWith(".got.plt", ORDER_GOT_PLT)
        .Default(ORDER_GOT);

  case DefinedAtom::typeStub:
    return ORDER_PLT;

  case DefinedAtom::typeRONote:
    return ORDER_RO_NOTE;

  case DefinedAtom::typeRWNote:
    return ORDER_RW_NOTE;

  case DefinedAtom::typeNoAlloc:
    return ORDER_NOALLOC;

  case DefinedAtom::typeThreadData:
    return ORDER_TDATA;
  case DefinedAtom::typeThreadZeroFill:
    return ORDER_TBSS;
  default:
    // If we get passed in a section push it to OTHER
    if (contentPermissions == DefinedAtom::perm___)
      return ORDER_OTHER;

    return ORDER_NOT_DEFINED;
  }
}

/// \brief This maps the input sections to the output section names
template <class ELFT>
StringRef TargetLayout<ELFT>::getInputSectionName(const DefinedAtom *da) const {
  if (da->sectionChoice() == DefinedAtom::sectionBasedOnContent) {
    switch (da->contentType()) {
    case DefinedAtom::typeCode:
      return ".text";
    case DefinedAtom::typeData:
      return ".data";
    case DefinedAtom::typeConstant:
      return ".rodata";
    case DefinedAtom::typeZeroFill:
      return ".bss";
    case DefinedAtom::typeThreadData:
      return ".tdata";
    case DefinedAtom::typeThreadZeroFill:
      return ".tbss";
    default:
      break;
    }
  }
  return da->customSectionName();
}

/// \brief This maps the input sections to the output section names.
template <class ELFT>
StringRef
TargetLayout<ELFT>::getOutputSectionName(StringRef archivePath,
                                         StringRef memberPath,
                                         StringRef inputSectionName) const {
  StringRef outputSectionName;
  if (_linkerScriptSema.hasLayoutCommands()) {
    script::Sema::SectionKey key = {archivePath, memberPath, inputSectionName};
    outputSectionName = _linkerScriptSema.getOutputSection(key);
    if (!outputSectionName.empty())
      return outputSectionName;
  }
  return llvm::StringSwitch<StringRef>(inputSectionName)
      .StartsWith(".text", ".text")
      .StartsWith(".ctors", ".ctors")
      .StartsWith(".dtors", ".dtors")
      .StartsWith(".rodata", ".rodata")
      .StartsWith(".gcc_except_table", ".gcc_except_table")
      .StartsWith(".data.rel.ro", ".data.rel.ro")
      .StartsWith(".data.rel.local", ".data.rel.local")
      .StartsWith(".data", ".data")
      .StartsWith(".tdata", ".tdata")
      .StartsWith(".tbss", ".tbss")
      .StartsWith(".init_array", ".init_array")
      .StartsWith(".fini_array", ".fini_array")
      .Default(inputSectionName);
}

/// \brief Gets the segment for a output section
template <class ELFT>
typename TargetLayout<ELFT>::SegmentType
TargetLayout<ELFT>::getSegmentType(const Section<ELFT> *section) const {
  switch (section->order()) {
  case ORDER_INTERP:
    return llvm::ELF::PT_INTERP;

  case ORDER_TEXT:
  case ORDER_HASH:
  case ORDER_DYNAMIC_SYMBOLS:
  case ORDER_DYNAMIC_STRINGS:
  case ORDER_DYNAMIC_RELOCS:
  case ORDER_DYNAMIC_PLT_RELOCS:
  case ORDER_REL:
  case ORDER_INIT:
  case ORDER_PLT:
  case ORDER_FINI:
  case ORDER_RODATA:
  case ORDER_EH_FRAME:
  case ORDER_CTORS:
  case ORDER_DTORS:
    return llvm::ELF::PT_LOAD;

  case ORDER_RO_NOTE:
  case ORDER_RW_NOTE:
    return llvm::ELF::PT_NOTE;

  case ORDER_DYNAMIC:
    return llvm::ELF::PT_DYNAMIC;

  case ORDER_EH_FRAMEHDR:
    return llvm::ELF::PT_GNU_EH_FRAME;

  case ORDER_GOT:
  case ORDER_GOT_PLT:
  case ORDER_DATA:
  case ORDER_BSS:
  case ORDER_INIT_ARRAY:
  case ORDER_FINI_ARRAY:
    return llvm::ELF::PT_LOAD;

  case ORDER_TDATA:
  case ORDER_TBSS:
    return llvm::ELF::PT_TLS;

  default:
    return llvm::ELF::PT_NULL;
  }
}

template <class ELFT>
bool TargetLayout<ELFT>::hasOutputSegment(Section<ELFT> *section) {
  switch (section->order()) {
  case ORDER_INTERP:
  case ORDER_HASH:
  case ORDER_DYNAMIC_SYMBOLS:
  case ORDER_DYNAMIC_STRINGS:
  case ORDER_DYNAMIC_RELOCS:
  case ORDER_DYNAMIC_PLT_RELOCS:
  case ORDER_REL:
  case ORDER_INIT:
  case ORDER_PLT:
  case ORDER_TEXT:
  case ORDER_FINI:
  case ORDER_RODATA:
  case ORDER_EH_FRAME:
  case ORDER_EH_FRAMEHDR:
  case ORDER_TDATA:
  case ORDER_TBSS:
  case ORDER_RO_NOTE:
  case ORDER_RW_NOTE:
  case ORDER_DYNAMIC:
  case ORDER_CTORS:
  case ORDER_DTORS:
  case ORDER_GOT:
  case ORDER_GOT_PLT:
  case ORDER_DATA:
  case ORDER_INIT_ARRAY:
  case ORDER_FINI_ARRAY:
  case ORDER_BSS:
  case ORDER_NOALLOC:
    return true;
  default:
    return section->hasOutputSegment();
  }
}

template <class ELFT>
AtomSection<ELFT> *
TargetLayout<ELFT>::createSection(StringRef sectionName, int32_t contentType,
                                  DefinedAtom::ContentPermissions permissions,
                                  SectionOrder sectionOrder) {
  return new (_allocator) AtomSection<ELFT>(_ctx, sectionName, contentType,
                                            permissions, sectionOrder);
}

template <class ELFT>
AtomSection<ELFT> *
TargetLayout<ELFT>::getSection(StringRef sectionName, int32_t contentType,
                               DefinedAtom::ContentPermissions permissions,
                               const DefinedAtom *da) {
  const SectionKey sectionKey(sectionName, permissions, da->file().path());
  SectionOrder sectionOrder =
      getSectionOrder(sectionName, contentType, permissions);
  auto sec = _sectionMap.find(sectionKey);
  if (sec != _sectionMap.end())
    return sec->second;
  AtomSection<ELFT> *newSec =
      createSection(sectionName, contentType, permissions, sectionOrder);

  newSec->setOutputSectionName(getOutputSectionName(
      da->file().archivePath(), da->file().memberPath(), sectionName));
  newSec->setOrder(sectionOrder);
  newSec->setArchiveNameOrPath(da->file().archivePath());
  newSec->setMemberNameOrPath(da->file().memberPath());
  _sections.push_back(newSec);
  _sectionMap.insert(std::make_pair(sectionKey, newSec));
  return newSec;
}

template <class ELFT>
ErrorOr<const AtomLayout *> TargetLayout<ELFT>::addAtom(const Atom *atom) {
  if (const DefinedAtom *definedAtom = dyn_cast<DefinedAtom>(atom)) {
    // HACK: Ignore undefined atoms. We need to adjust the interface so that
    // undefined atoms can still be included in the output symbol table for
    // -noinhibit-exec.
    if (definedAtom->contentType() == DefinedAtom::typeUnknown)
      return make_error_code(llvm::errc::invalid_argument);
    const DefinedAtom::ContentPermissions permissions =
        definedAtom->permissions();
    const DefinedAtom::ContentType contentType = definedAtom->contentType();

    StringRef sectionName = getInputSectionName(definedAtom);
    AtomSection<ELFT> *section =
        getSection(sectionName, contentType, permissions, definedAtom);

    // Add runtime relocations to the .rela section.
    for (const auto &reloc : *definedAtom) {
      bool isLocalReloc = true;
      if (_ctx.isDynamicRelocation(*reloc)) {
        getDynamicRelocationTable()->addRelocation(*definedAtom, *reloc);
        isLocalReloc = false;
      } else if (_ctx.isPLTRelocation(*reloc)) {
        getPLTRelocationTable()->addRelocation(*definedAtom, *reloc);
        isLocalReloc = false;
      }

      if (!reloc->target())
        continue;

      // Ignore undefined atoms that are not target of dynamic relocations
      if (isa<UndefinedAtom>(reloc->target()) && isLocalReloc)
        continue;

      if (_ctx.isCopyRelocation(*reloc)) {
        _copiedDynSymNames.insert(definedAtom->name());
        continue;
      }

      _referencedDynAtoms.insert(reloc->target());
    }
    return section->appendAtom(atom);
  }

  const AbsoluteAtom *absoluteAtom = cast<AbsoluteAtom>(atom);
  // Absolute atoms are not part of any section, they are global for the whole
  // link
  _absoluteAtoms.push_back(
      new (_allocator) AtomLayout(absoluteAtom, 0, absoluteAtom->value()));
  return _absoluteAtoms.back();
}

/// Output sections with the same name into a OutputSection
template <class ELFT> void TargetLayout<ELFT>::createOutputSections() {
  OutputSection<ELFT> *outputSection;

  for (auto &si : _sections) {
    Section<ELFT> *section = dyn_cast<Section<ELFT>>(si);
    if (!section)
      continue;
    const std::pair<StringRef, OutputSection<ELFT> *> currentOutputSection(
        section->outputSectionName(), nullptr);
    std::pair<typename OutputSectionMapT::iterator, bool> outputSectionInsert(
        _outputSectionMap.insert(currentOutputSection));
    if (!outputSectionInsert.second) {
      outputSection = outputSectionInsert.first->second;
    } else {
      outputSection = new (_allocator.Allocate<OutputSection<ELFT>>())
          OutputSection<ELFT>(section->outputSectionName());
      _outputSections.push_back(outputSection);
      outputSectionInsert.first->second = outputSection;
    }
    outputSection->appendSection(section);
  }
}

template <class ELFT>
std::vector<typename TargetLayout<ELFT>::SegmentKey>
TargetLayout<ELFT>::getSegmentsForSection(const OutputSection<ELFT> *os,
                                          const Section<ELFT> *sec) const {
  std::vector<SegmentKey> segKeys;
  auto phdrs = _linkerScriptSema.getPHDRsForOutputSection(os->name());
  if (!phdrs.empty()) {
    if (phdrs.size() == 1 && phdrs[0]->isNone()) {
      segKeys.emplace_back("NONE", llvm::ELF::PT_NULL, 0, false);
      return segKeys;
    }

    for (auto phdr : phdrs) {
      segKeys.emplace_back(phdr->name(), phdr->type(), phdr->flags(), true);
    }
    return segKeys;
  }

  uint64_t flags = getLookupSectionFlags(os);
  int64_t segmentType = getSegmentType(sec);
  StringRef segmentName = sec->segmentKindToStr();

  // We need a separate segment for sections that don't have
  // the segment type to be PT_LOAD
  if (segmentType != llvm::ELF::PT_LOAD)
    segKeys.emplace_back(segmentName, segmentType, flags, false);

  if (segmentType == llvm::ELF::PT_NULL)
    return segKeys;

  // If the output magic is set to OutputMagic::NMAGIC or
  // OutputMagic::OMAGIC, Place the data alongside text in one single
  // segment
  ELFLinkingContext::OutputMagic outputMagic = _ctx.getOutputMagic();
  if (outputMagic == ELFLinkingContext::OutputMagic::NMAGIC ||
      outputMagic == ELFLinkingContext::OutputMagic::OMAGIC)
    flags =
        llvm::ELF::SHF_EXECINSTR | llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE;

  segKeys.emplace_back("LOAD", llvm::ELF::PT_LOAD, flags, false);
  return segKeys;
}

template <class ELFT>
uint64_t
TargetLayout<ELFT>::getLookupSectionFlags(const OutputSection<ELFT> *os) const {
  uint64_t flags = os->flags();
  if (!(flags & llvm::ELF::SHF_WRITE) && _ctx.mergeRODataToTextSegment())
    flags &= ~llvm::ELF::SHF_EXECINSTR;

  // Merge string sections into Data segment itself
  flags &= ~(llvm::ELF::SHF_STRINGS | llvm::ELF::SHF_MERGE);

  // Merge the TLS section into the DATA segment itself
  flags &= ~(llvm::ELF::SHF_TLS);
  return flags;
}

template <class ELFT> void TargetLayout<ELFT>::assignSectionsToSegments() {
  ScopedTask task(getDefaultDomain(), "assignSectionsToSegments");
  // sort the sections by their order as defined by the layout
  sortInputSections();

  // Create output sections.
  createOutputSections();

  // Finalize output section layout.
  finalizeOutputSectionLayout();

  // Set the ordinal after sorting the sections
  int ordinal = 1;
  for (auto osi : _outputSections) {
    osi->setOrdinal(ordinal);
    for (auto ai : osi->sections()) {
      ai->setOrdinal(ordinal);
    }
    ++ordinal;
  }
  for (auto osi : _outputSections) {
    for (auto section : osi->sections()) {
      if (!hasOutputSegment(section))
        continue;

      osi->setLoadableSection(section->isLoadableSection());
      osi->setHasSegment();

      auto segKeys = getSegmentsForSection(osi, section);
      assert(!segKeys.empty() && "Must always be at least one segment");
      section->setSegmentType(segKeys[0]._type);

      for (auto key : segKeys) {
        // Try to find non-load (real) segment type if possible
        if (key._type != llvm::ELF::PT_LOAD)
          section->setSegmentType(key._type);

        const std::pair<SegmentKey, Segment<ELFT> *> currentSegment(key,
                                                                    nullptr);
        std::pair<typename SegmentMapT::iterator, bool> segmentInsert(
            _segmentMap.insert(currentSegment));
        Segment<ELFT> *segment;
        if (!segmentInsert.second) {
          segment = segmentInsert.first->second;
        } else {
          segment = new (_allocator) Segment<ELFT>(_ctx, key._name, key._type);
          if (key._segmentFlags)
            segment->setSegmentFlags(key._flags);
          segmentInsert.first->second = segment;
          _segments.push_back(segment);
        }
        if (key._type == llvm::ELF::PT_LOAD) {
          // Insert chunks with linker script expressions that occur at this
          // point, just before appending a new input section
          addExtraChunksToSegment(segment, section->archivePath(),
                                  section->memberPath(),
                                  section->inputSectionName());
        }
        segment->append(section);
      }
    }
  }

  // Default values if no linker script is available
  bool hasProgramSegment = _ctx.isDynamic() && !_ctx.isDynamicLibrary();
  bool hasElfHeader = true;
  bool hasProgramHeader = true;
  uint64_t segmentFlags = 0;

  // Check if linker script has PHDRS and program segment defined
  if (_linkerScriptSema.hasPHDRs()) {
    if (auto p = _linkerScriptSema.getProgramPHDR()) {
      hasProgramSegment = true;
      hasElfHeader = p->hasFileHdr();
      hasProgramHeader = p->hasPHDRs();
      segmentFlags = p->flags();
    } else {
      hasProgramSegment = false;
      hasElfHeader = false;
      hasProgramHeader = false;
    }
  }

  if (hasProgramSegment) {
    Segment<ELFT> *segment = new (_allocator) ProgramHeaderSegment<ELFT>(_ctx);
    _segments.push_back(segment);
    if (segmentFlags)
      segment->setSegmentFlags(segmentFlags);
    if (hasElfHeader)
      segment->append(_elfHeader);
    if (hasProgramHeader)
      segment->append(_programHeader);
  }
}

template <class ELFT> void TargetLayout<ELFT>::sortSegments() {
  std::sort(_segments.begin(), _segments.end(), Segment<ELFT>::compareSegments);
}

template <class ELFT> void TargetLayout<ELFT>::assignVirtualAddress() {
  if (_segments.empty())
    return;

  sortSegments();

  uint64_t baseAddress = _ctx.getBaseAddress();

  // HACK: This is a super dirty hack. The elf header and program header are
  // not part of a section, but we need them to be loaded at the base address
  // so that AT_PHDR is set correctly by the loader and so they are accessible
  // at runtime. To do this we simply prepend them to the first loadable Segment
  // and let the layout logic take care of it.
  Segment<ELFT> *firstLoadSegment = nullptr;
  for (auto si : _segments) {
    if (si->segmentType() == llvm::ELF::PT_LOAD) {
      firstLoadSegment = si;
      si->firstSection()->setAlign(si->alignment());
      break;
    }
  }
  assert(firstLoadSegment != nullptr && "No loadable segment!");
  firstLoadSegment->prepend(_programHeader);
  firstLoadSegment->prepend(_elfHeader);
  bool newSegmentHeaderAdded = true;
  bool virtualAddressAssigned = false;
  bool fileOffsetAssigned = false;
  while (true) {
    for (auto si : _segments) {
      si->finalize();
      // Don't add PT_NULL segments into the program header
      if (si->segmentType() != llvm::ELF::PT_NULL)
        newSegmentHeaderAdded = _programHeader->addSegment(si);
    }
    if (!newSegmentHeaderAdded && virtualAddressAssigned)
      break;
    uint64_t address = baseAddress;
    // start assigning virtual addresses
    for (auto &si : _segments) {
      if ((si->segmentType() != llvm::ELF::PT_LOAD) &&
          (si->segmentType() != llvm::ELF::PT_NULL))
        continue;

      if (si->segmentType() == llvm::ELF::PT_NULL) {
        si->assignVirtualAddress(0 /*non loadable*/);
      } else {
        if (virtualAddressAssigned && (address != baseAddress) &&
            (address == si->virtualAddr()))
          break;
        si->assignVirtualAddress(address);
      }
      address = si->virtualAddr() + si->memSize();
    }
    uint64_t baseFileOffset = 0;
    uint64_t fileoffset = baseFileOffset;
    for (auto &si : _segments) {
      if ((si->segmentType() != llvm::ELF::PT_LOAD) &&
          (si->segmentType() != llvm::ELF::PT_NULL))
        continue;
      if (fileOffsetAssigned && (fileoffset != baseFileOffset) &&
          (fileoffset == si->fileOffset()))
        break;
      si->assignFileOffsets(fileoffset);
      fileoffset = si->fileOffset() + si->fileSize();
    }
    virtualAddressAssigned = true;
    fileOffsetAssigned = true;
    _programHeader->resetProgramHeaders();
  }
  Section<ELFT> *section;
  // Fix the offsets of all the atoms within a section
  for (auto &si : _sections) {
    section = dyn_cast<Section<ELFT>>(si);
    if (section && TargetLayout<ELFT>::hasOutputSegment(section))
      section->assignFileOffsets(section->fileOffset());
  }
  // Set the size of the merged Sections
  for (auto osi : _outputSections) {
    uint64_t sectionfileoffset = 0;
    uint64_t startFileOffset = 0;
    uint64_t sectionsize = 0;
    bool isFirstSection = true;
    for (auto si : osi->sections()) {
      if (isFirstSection) {
        startFileOffset = si->fileOffset();
        isFirstSection = false;
      }
      sectionfileoffset = si->fileOffset();
      sectionsize = si->fileSize();
    }
    sectionsize = (sectionfileoffset - startFileOffset) + sectionsize;
    osi->setFileOffset(startFileOffset);
    osi->setSize(sectionsize);
  }
  // Set the virtual addr of the merged Sections
  for (auto osi : _outputSections) {
    uint64_t sectionstartaddr = 0;
    uint64_t startaddr = 0;
    uint64_t sectionsize = 0;
    bool isFirstSection = true;
    for (auto si : osi->sections()) {
      if (isFirstSection) {
        startaddr = si->virtualAddr();
        isFirstSection = false;
      }
      sectionstartaddr = si->virtualAddr();
      sectionsize = si->memSize();
    }
    sectionsize = (sectionstartaddr - startaddr) + sectionsize;
    osi->setMemSize(sectionsize);
    osi->setAddr(startaddr);
  }
}

template <class ELFT>
void TargetLayout<ELFT>::assignFileOffsetsForMiscSections() {
  uint64_t fileoffset = 0;
  uint64_t size = 0;
  for (auto si : _segments) {
    // Don't calculate offsets from non loadable segments
    if ((si->segmentType() != llvm::ELF::PT_LOAD) &&
        (si->segmentType() != llvm::ELF::PT_NULL))
      continue;
    fileoffset = si->fileOffset();
    size = si->fileSize();
  }
  fileoffset = fileoffset + size;
  Section<ELFT> *section;
  for (auto si : _sections) {
    section = dyn_cast<Section<ELFT>>(si);
    if (section && TargetLayout<ELFT>::hasOutputSegment(section))
      continue;
    fileoffset = llvm::alignTo(fileoffset, si->alignment());
    si->setFileOffset(fileoffset);
    si->setVirtualAddr(0);
    fileoffset += si->fileSize();
  }
}

template <class ELFT> void TargetLayout<ELFT>::sortInputSections() {
  // First, sort according to default layout's order
  std::stable_sort(
      _sections.begin(), _sections.end(),
      [](Chunk<ELFT> *A, Chunk<ELFT> *B) { return A->order() < B->order(); });

  if (!_linkerScriptSema.hasLayoutCommands())
    return;

  // Sort the sections by their order as defined by the linker script
  std::stable_sort(
      this->_sections.begin(), this->_sections.end(),
      [this](Chunk<ELFT> *A, Chunk<ELFT> *B) {
        auto *a = dyn_cast<Section<ELFT>>(A);
        auto *b = dyn_cast<Section<ELFT>>(B);

        if (a == nullptr)
          return false;
        if (b == nullptr)
          return true;

        return _linkerScriptSema.less(
            {a->archivePath(), a->memberPath(), a->inputSectionName()},
            {b->archivePath(), b->memberPath(), b->inputSectionName()});
      });
  // Now try to arrange sections with no mapping rules to sections with
  // similar content
  auto p = this->_sections.begin();
  // Find first section that has no assigned rule id
  while (p != this->_sections.end()) {
    auto *sect = dyn_cast<AtomSection<ELFT>>(*p);
    if (!sect)
      break;

    if (!_linkerScriptSema.hasMapping({sect->archivePath(), sect->memberPath(),
                                       sect->inputSectionName()}))
      break;

    ++p;
  }
  // For all sections that have no assigned rule id, try to move them near a
  // section with similar contents
  if (p != this->_sections.begin()) {
    for (; p != this->_sections.end(); ++p) {
      auto q = p;
      --q;
      while (q != this->_sections.begin() &&
             (*q)->getContentType() != (*p)->getContentType())
        --q;
      if ((*q)->getContentType() != (*p)->getContentType())
        continue;
      ++q;
      for (auto i = p; i != q;) {
        auto next = i--;
        std::iter_swap(i, next);
      }
    }
  }
}

template <class ELFT>
const AtomLayout *
TargetLayout<ELFT>::findAtomLayoutByName(StringRef name) const {
  for (auto sec : _sections)
    if (auto section = dyn_cast<Section<ELFT>>(sec))
      if (auto *al = section->findAtomLayoutByName(name))
        return al;
  return nullptr;
}

template <class ELFT>
void TargetLayout<ELFT>::addExtraChunksToSegment(Segment<ELFT> *segment,
                                                 StringRef archivePath,
                                                 StringRef memberPath,
                                                 StringRef sectionName) {
  if (!_linkerScriptSema.hasLayoutCommands())
    return;
  std::vector<const script::SymbolAssignment *> exprs =
      _linkerScriptSema.getExprs({archivePath, memberPath, sectionName});
  for (auto expr : exprs) {
    auto expChunk =
        new (this->_allocator) ExpressionChunk<ELFT>(this->_ctx, expr);
    segment->append(expChunk);
  }
}

template <class ELFT>
RelocationTable<ELFT> *TargetLayout<ELFT>::getDynamicRelocationTable() {
  if (!_dynamicRelocationTable) {
    _dynamicRelocationTable = createRelocationTable(
        _ctx.isRelaOutputFormat() ? ".rela.dyn" : ".rel.dyn",
        ORDER_DYNAMIC_RELOCS);
    addSection(_dynamicRelocationTable.get());
  }
  return _dynamicRelocationTable.get();
}

template <class ELFT>
RelocationTable<ELFT> *TargetLayout<ELFT>::getPLTRelocationTable() {
  if (!_pltRelocationTable) {
    _pltRelocationTable = createRelocationTable(
        _ctx.isRelaOutputFormat() ? ".rela.plt" : ".rel.plt",
        ORDER_DYNAMIC_PLT_RELOCS);
    addSection(_pltRelocationTable.get());
  }
  return _pltRelocationTable.get();
}

template <class ELFT> uint64_t TargetLayout<ELFT>::getTLSSize() const {
  for (const auto &phdr : *_programHeader)
    if (phdr->p_type == llvm::ELF::PT_TLS)
      return phdr->p_memsz;
  return 0;
}

template class TargetLayout<ELF32LE>;
template class TargetLayout<ELF32BE>;
template class TargetLayout<ELF64LE>;
template class TargetLayout<ELF64BE>;

} // end namespace elf
} // end namespace lld
