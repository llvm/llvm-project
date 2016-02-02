//===- lib/ReaderWriter/MachO/MachONormalizedFileFromAtoms.cpp ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

///
/// \file Converts from in-memory Atoms to in-memory normalized mach-o.
///
///                  +------------+
///                  | normalized |
///                  +------------+
///                        ^
///                        |
///                        |
///                    +-------+
///                    | Atoms |
///                    +-------+

#include "MachONormalizedFile.h"
#include "ArchHandler.h"
#include "MachONormalizedFileBinaryUtils.h"
#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MachO.h"
#include <map>
#include <system_error>

using llvm::StringRef;
using llvm::isa;
using namespace llvm::MachO;
using namespace lld::mach_o::normalized;
using namespace lld;

namespace {

struct AtomInfo {
  const DefinedAtom  *atom;
  uint64_t            offsetInSection;
};

struct SectionInfo {
  SectionInfo(StringRef seg, StringRef sect, SectionType type,
              const MachOLinkingContext &ctxt, uint32_t attr,
              bool relocsToDefinedCanBeImplicit);

  StringRef                 segmentName;
  StringRef                 sectionName;
  SectionType               type;
  uint32_t                  attributes;
  uint64_t                  address;
  uint64_t                  size;
  uint16_t                  alignment;

  /// If this is set, the any relocs in this section which point to defined
  /// addresses can be implicitly generated.  This is the case for the
  /// __eh_frame section where references to the function can be implicit if the
  /// function is defined.
  bool                      relocsToDefinedCanBeImplicit;


  std::vector<AtomInfo>     atomsAndOffsets;
  uint32_t                  normalizedSectionIndex;
  uint32_t                  finalSectionIndex;
};

SectionInfo::SectionInfo(StringRef sg, StringRef sct, SectionType t,
                         const MachOLinkingContext &ctxt, uint32_t attrs,
                         bool relocsToDefinedCanBeImplicit)
 : segmentName(sg), sectionName(sct), type(t), attributes(attrs),
                 address(0), size(0), alignment(1),
                 relocsToDefinedCanBeImplicit(relocsToDefinedCanBeImplicit),
                 normalizedSectionIndex(0), finalSectionIndex(0) {
  uint16_t align = 1;
  if (ctxt.sectionAligned(segmentName, sectionName, align)) {
    alignment = align;
  }
}

struct SegmentInfo {
  SegmentInfo(StringRef name);

  StringRef                  name;
  uint64_t                   address;
  uint64_t                   size;
  uint32_t                   access;
  std::vector<SectionInfo*>  sections;
  uint32_t                   normalizedSegmentIndex;
};

SegmentInfo::SegmentInfo(StringRef n)
 : name(n), address(0), size(0), access(0), normalizedSegmentIndex(0) {
}

class Util {
public:
  Util(const MachOLinkingContext &ctxt)
      : _ctx(ctxt), _archHandler(ctxt.archHandler()), _entryAtom(nullptr),
        _hasTLVDescriptors(false), _subsectionsViaSymbols(true) {}
  ~Util();

  void      processDefinedAtoms(const lld::File &atomFile);
  void      processAtomAttributes(const DefinedAtom *atom);
  void      assignAtomToSection(const DefinedAtom *atom);
  void      organizeSections();
  void      assignAddressesToSections(const NormalizedFile &file);
  uint32_t  fileFlags();
  void      copySegmentInfo(NormalizedFile &file);
  void      copySectionInfo(NormalizedFile &file);
  void      updateSectionInfo(NormalizedFile &file);
  void      buildAtomToAddressMap();
  std::error_code addSymbols(const lld::File &atomFile, NormalizedFile &file);
  void      addIndirectSymbols(const lld::File &atomFile, NormalizedFile &file);
  void      addRebaseAndBindingInfo(const lld::File &, NormalizedFile &file);
  void      addExportInfo(const lld::File &, NormalizedFile &file);
  void      addSectionRelocs(const lld::File &, NormalizedFile &file);
  void      buildDataInCodeArray(const lld::File &, NormalizedFile &file);
  void      addDependentDylibs(const lld::File &, NormalizedFile &file);
  void      copyEntryPointAddress(NormalizedFile &file);
  void      copySectionContent(NormalizedFile &file);

private:
  typedef std::map<DefinedAtom::ContentType, SectionInfo*> TypeToSection;
  typedef llvm::DenseMap<const Atom*, uint64_t> AtomToAddress;

  struct DylibInfo { int ordinal; bool hasWeak; bool hasNonWeak; };
  typedef llvm::StringMap<DylibInfo> DylibPathToInfo;

  SectionInfo *sectionForAtom(const DefinedAtom*);
  SectionInfo *getRelocatableSection(DefinedAtom::ContentType type);
  SectionInfo *getFinalSection(DefinedAtom::ContentType type);
  void         appendAtom(SectionInfo *sect, const DefinedAtom *atom);
  SegmentInfo *segmentForName(StringRef segName);
  void         layoutSectionsInSegment(SegmentInfo *seg, uint64_t &addr);
  void         layoutSectionsInTextSegment(size_t, SegmentInfo *, uint64_t &);
  void         copySectionContent(SectionInfo *si, ContentBytes &content);
  uint16_t     descBits(const DefinedAtom* atom);
  int          dylibOrdinal(const SharedLibraryAtom *sa);
  void         segIndexForSection(const SectionInfo *sect,
                             uint8_t &segmentIndex, uint64_t &segmentStartAddr);
  const Atom  *targetOfLazyPointer(const DefinedAtom *lpAtom);
  const Atom  *targetOfStub(const DefinedAtom *stubAtom);
  std::error_code getSymbolTableRegion(const DefinedAtom* atom,
                                       bool &inGlobalsRegion,
                                       SymbolScope &symbolScope);
  void         appendSection(SectionInfo *si, NormalizedFile &file);
  uint32_t     sectionIndexForAtom(const Atom *atom);

  typedef llvm::DenseMap<const Atom*, uint32_t> AtomToIndex;
  struct AtomAndIndex { const Atom *atom; uint32_t index; SymbolScope scope; };
  struct AtomSorter {
    bool operator()(const AtomAndIndex &left, const AtomAndIndex &right);
  };
  struct SegmentSorter {
    bool operator()(const SegmentInfo *left, const SegmentInfo *right);
    static unsigned weight(const SegmentInfo *);
  };
  struct TextSectionSorter {
    bool operator()(const SectionInfo *left, const SectionInfo *right);
    static unsigned weight(const SectionInfo *);
  };

  const MachOLinkingContext &_ctx;
  mach_o::ArchHandler          &_archHandler;
  llvm::BumpPtrAllocator        _allocator;
  std::vector<SectionInfo*>     _sectionInfos;
  std::vector<SegmentInfo*>     _segmentInfos;
  TypeToSection                 _sectionMap;
  std::vector<SectionInfo*>     _customSections;
  AtomToAddress                 _atomToAddress;
  DylibPathToInfo               _dylibInfo;
  const DefinedAtom            *_entryAtom;
  AtomToIndex                   _atomToSymbolIndex;
  std::vector<const Atom *>     _machHeaderAliasAtoms;
  bool                          _hasTLVDescriptors;
  bool                          _subsectionsViaSymbols;
};

Util::~Util() {
  // The SectionInfo structs are BumpPtr allocated, but atomsAndOffsets needs
  // to be deleted.
  for (SectionInfo *si : _sectionInfos) {
    // clear() destroys vector elements, but does not deallocate.
    // Instead use swap() to deallocate vector buffer.
    std::vector<AtomInfo> empty;
    si->atomsAndOffsets.swap(empty);
  }
  // The SegmentInfo structs are BumpPtr allocated, but sections needs
  // to be deleted.
  for (SegmentInfo *sgi : _segmentInfos) {
    std::vector<SectionInfo*> empty2;
    sgi->sections.swap(empty2);
  }
}

SectionInfo *Util::getRelocatableSection(DefinedAtom::ContentType type) {
  StringRef segmentName;
  StringRef sectionName;
  SectionType sectionType;
  SectionAttr sectionAttrs;
  bool relocsToDefinedCanBeImplicit;

  // Use same table used by when parsing .o files.
  relocatableSectionInfoForContentType(type, segmentName, sectionName,
                                       sectionType, sectionAttrs,
                                       relocsToDefinedCanBeImplicit);
  // If we already have a SectionInfo with this name, re-use it.
  // This can happen if two ContentType map to the same mach-o section.
  for (auto sect : _sectionMap) {
    if (sect.second->sectionName.equals(sectionName) &&
        sect.second->segmentName.equals(segmentName)) {
      return sect.second;
    }
  }
  // Otherwise allocate new SectionInfo object.
  auto *sect = new (_allocator)
      SectionInfo(segmentName, sectionName, sectionType, _ctx, sectionAttrs,
                  relocsToDefinedCanBeImplicit);
  _sectionInfos.push_back(sect);
  _sectionMap[type] = sect;
  return sect;
}

#define ENTRY(seg, sect, type, atomType) \
  {seg, sect, type, DefinedAtom::atomType }

struct MachOFinalSectionFromAtomType {
  StringRef                 segmentName;
  StringRef                 sectionName;
  SectionType               sectionType;
  DefinedAtom::ContentType  atomType;
};

const MachOFinalSectionFromAtomType sectsToAtomType[] = {
  ENTRY("__TEXT", "__text",           S_REGULAR,          typeCode),
  ENTRY("__TEXT", "__text",           S_REGULAR,          typeMachHeader),
  ENTRY("__TEXT", "__cstring",        S_CSTRING_LITERALS, typeCString),
  ENTRY("__TEXT", "__ustring",        S_REGULAR,          typeUTF16String),
  ENTRY("__TEXT", "__const",          S_REGULAR,          typeConstant),
  ENTRY("__TEXT", "__const",          S_4BYTE_LITERALS,   typeLiteral4),
  ENTRY("__TEXT", "__const",          S_8BYTE_LITERALS,   typeLiteral8),
  ENTRY("__TEXT", "__const",          S_16BYTE_LITERALS,  typeLiteral16),
  ENTRY("__TEXT", "__stubs",          S_SYMBOL_STUBS,     typeStub),
  ENTRY("__TEXT", "__stub_helper",    S_REGULAR,          typeStubHelper),
  ENTRY("__TEXT", "__gcc_except_tab", S_REGULAR,          typeLSDA),
  ENTRY("__TEXT", "__eh_frame",       S_COALESCED,        typeCFI),
  ENTRY("__TEXT", "__unwind_info",    S_REGULAR,          typeProcessedUnwindInfo),
  ENTRY("__DATA", "__data",           S_REGULAR,          typeData),
  ENTRY("__DATA", "__const",          S_REGULAR,          typeConstData),
  ENTRY("__DATA", "__cfstring",       S_REGULAR,          typeCFString),
  ENTRY("__DATA", "__la_symbol_ptr",  S_LAZY_SYMBOL_POINTERS,
                                                          typeLazyPointer),
  ENTRY("__DATA", "__mod_init_func",  S_MOD_INIT_FUNC_POINTERS,
                                                          typeInitializerPtr),
  ENTRY("__DATA", "__mod_term_func",  S_MOD_TERM_FUNC_POINTERS,
                                                          typeTerminatorPtr),
  ENTRY("__DATA", "__got",            S_NON_LAZY_SYMBOL_POINTERS,
                                                          typeGOT),
  ENTRY("__DATA", "__thread_vars",    S_THREAD_LOCAL_VARIABLES,
                                                          typeThunkTLV),
  ENTRY("__DATA", "__thread_data",    S_THREAD_LOCAL_REGULAR,
                                                          typeTLVInitialData),
  ENTRY("__DATA", "__thread_ptrs",    S_THREAD_LOCAL_VARIABLE_POINTERS,
                                                          typeTLVInitializerPtr),
  ENTRY("__DATA", "__thread_bss",     S_THREAD_LOCAL_ZEROFILL,
                                                         typeTLVInitialZeroFill),
  ENTRY("__DATA", "__bss",            S_ZEROFILL,         typeZeroFill),
  ENTRY("__DATA", "__interposing",    S_INTERPOSING,      typeInterposingTuples),
};
#undef ENTRY

SectionInfo *Util::getFinalSection(DefinedAtom::ContentType atomType) {
  for (auto &p : sectsToAtomType) {
    if (p.atomType != atomType)
      continue;
    SectionAttr sectionAttrs = 0;
    switch (atomType) {
    case DefinedAtom::typeCode:
    case DefinedAtom::typeStub:
    case DefinedAtom::typeStubHelper:
      sectionAttrs = S_ATTR_PURE_INSTRUCTIONS;
      break;
    case DefinedAtom::typeThunkTLV:
      _hasTLVDescriptors = true;
      break;
    default:
      break;
    }
    // If we already have a SectionInfo with this name, re-use it.
    // This can happen if two ContentType map to the same mach-o section.
    for (auto sect : _sectionMap) {
      if (sect.second->sectionName.equals(p.sectionName) &&
          sect.second->segmentName.equals(p.segmentName)) {
        return sect.second;
      }
    }
    // Otherwise allocate new SectionInfo object.
    auto *sect = new (_allocator) SectionInfo(
        p.segmentName, p.sectionName, p.sectionType, _ctx, sectionAttrs,
        /* relocsToDefinedCanBeImplicit */ false);
    _sectionInfos.push_back(sect);
    _sectionMap[atomType] = sect;
    return sect;
  }
  llvm_unreachable("content type not yet supported");
}

SectionInfo *Util::sectionForAtom(const DefinedAtom *atom) {
  if (atom->sectionChoice() == DefinedAtom::sectionBasedOnContent) {
    // Section for this atom is derived from content type.
    DefinedAtom::ContentType type = atom->contentType();
    auto pos = _sectionMap.find(type);
    if ( pos != _sectionMap.end() )
      return pos->second;
    bool rMode = (_ctx.outputMachOType() == llvm::MachO::MH_OBJECT);
    return rMode ? getRelocatableSection(type) : getFinalSection(type);
  } else {
    // This atom needs to be in a custom section.
    StringRef customName = atom->customSectionName();
    // Look to see if we have already allocated the needed custom section.
    for(SectionInfo *sect : _customSections) {
      const DefinedAtom *firstAtom = sect->atomsAndOffsets.front().atom;
      if (firstAtom->customSectionName().equals(customName)) {
        return sect;
      }
    }
    // Not found, so need to create a new custom section.
    size_t seperatorIndex = customName.find('/');
    assert(seperatorIndex != StringRef::npos);
    StringRef segName = customName.slice(0, seperatorIndex);
    StringRef sectName = customName.drop_front(seperatorIndex + 1);
    auto *sect =
        new (_allocator) SectionInfo(segName, sectName, S_REGULAR, _ctx,
                                     0, /* relocsToDefinedCanBeImplicit */ false);
    _customSections.push_back(sect);
    _sectionInfos.push_back(sect);
    return sect;
  }
}

void Util::appendAtom(SectionInfo *sect, const DefinedAtom *atom) {
  // Figure out offset for atom in this section given alignment constraints.
  uint64_t offset = sect->size;
  DefinedAtom::Alignment atomAlign = atom->alignment();
  uint64_t align = atomAlign.value;
  uint64_t requiredModulus = atomAlign.modulus;
  uint64_t currentModulus = (offset % align);
  if ( currentModulus != requiredModulus ) {
    if ( requiredModulus > currentModulus )
      offset += requiredModulus-currentModulus;
    else
      offset += align+requiredModulus-currentModulus;
  }
  // Record max alignment of any atom in this section.
  if (align > sect->alignment)
    sect->alignment = atomAlign.value;
  // Assign atom to this section with this offset.
  AtomInfo ai = {atom, offset};
  sect->atomsAndOffsets.push_back(ai);
  // Update section size to include this atom.
  sect->size = offset + atom->size();
}

void Util::processDefinedAtoms(const lld::File &atomFile) {
  for (const DefinedAtom *atom : atomFile.defined()) {
    processAtomAttributes(atom);
    assignAtomToSection(atom);
  }
}

void Util::processAtomAttributes(const DefinedAtom *atom) {
  // If the file doesn't use subsections via symbols, then make sure we don't
  // add that flag to the final output file if we have a relocatable file.
  if (auto *machoFile = dyn_cast<mach_o::MachOFile>(&atom->file()))
    if (!machoFile->subsectionsViaSymbols())
      _subsectionsViaSymbols = false;
}

void Util::assignAtomToSection(const DefinedAtom *atom) {
  if (atom->contentType() == DefinedAtom::typeMachHeader) {
    _machHeaderAliasAtoms.push_back(atom);
    // Assign atom to this section with this offset.
    AtomInfo ai = {atom, 0};
    sectionForAtom(atom)->atomsAndOffsets.push_back(ai);
  } else if (atom->contentType() == DefinedAtom::typeDSOHandle)
    _machHeaderAliasAtoms.push_back(atom);
  else
    appendAtom(sectionForAtom(atom), atom);
}

SegmentInfo *Util::segmentForName(StringRef segName) {
  for (SegmentInfo *si : _segmentInfos) {
    if ( si->name.equals(segName) )
      return si;
  }
  auto *info = new (_allocator) SegmentInfo(segName);
  if (segName.equals("__TEXT"))
    info->access = VM_PROT_READ | VM_PROT_EXECUTE;
  else if (segName.equals("__DATA"))
    info->access = VM_PROT_READ | VM_PROT_WRITE;
  else if (segName.equals("__PAGEZERO"))
    info->access = 0;
  _segmentInfos.push_back(info);
  return info;
}

unsigned Util::SegmentSorter::weight(const SegmentInfo *seg) {
 return llvm::StringSwitch<unsigned>(seg->name)
    .Case("__PAGEZERO",  1)
    .Case("__TEXT",      2)
    .Case("__DATA",      3)
    .Default(100);
}

bool Util::SegmentSorter::operator()(const SegmentInfo *left,
                                  const SegmentInfo *right) {
  return (weight(left) < weight(right));
}

unsigned Util::TextSectionSorter::weight(const SectionInfo *sect) {
 return llvm::StringSwitch<unsigned>(sect->sectionName)
    .Case("__text",         1)
    .Case("__stubs",        2)
    .Case("__stub_helper",  3)
    .Case("__const",        4)
    .Case("__cstring",      5)
    .Case("__unwind_info",  98)
    .Case("__eh_frame",     99)
    .Default(10);
}

bool Util::TextSectionSorter::operator()(const SectionInfo *left,
                                         const SectionInfo *right) {
  return (weight(left) < weight(right));
}

void Util::organizeSections() {
  // NOTE!: Keep this in sync with assignAddressesToSections.
  switch (_ctx.outputMachOType()) {
    case llvm::MachO::MH_EXECUTE:
      // Main executables, need a zero-page segment
      segmentForName("__PAGEZERO");
      // Fall into next case.
    case llvm::MachO::MH_DYLIB:
    case llvm::MachO::MH_BUNDLE:
      // All dynamic code needs TEXT segment to hold the load commands.
      segmentForName("__TEXT");
      break;
    default:
      break;
  }
  // Group sections into segments.
  for (SectionInfo *si : _sectionInfos) {
    SegmentInfo *seg = segmentForName(si->segmentName);
    seg->sections.push_back(si);
  }
  // Sort segments.
  std::sort(_segmentInfos.begin(), _segmentInfos.end(), SegmentSorter());

  // Sort sections within segments.
  for (SegmentInfo *seg : _segmentInfos) {
    if (seg->name.equals("__TEXT")) {
      std::sort(seg->sections.begin(), seg->sections.end(),
                TextSectionSorter());
    }
  }

  // Record final section indexes.
  uint32_t segmentIndex = 0;
  uint32_t sectionIndex = 1;
  for (SegmentInfo *seg : _segmentInfos) {
    seg->normalizedSegmentIndex = segmentIndex++;
    for (SectionInfo *sect : seg->sections)
      sect->finalSectionIndex = sectionIndex++;
  }
}

void Util::layoutSectionsInSegment(SegmentInfo *seg, uint64_t &addr) {
  seg->address = addr;
  for (SectionInfo *sect : seg->sections) {
    sect->address = llvm::alignTo(addr, sect->alignment);
    addr = sect->address + sect->size;
  }
  seg->size = llvm::alignTo(addr - seg->address, _ctx.pageSize());
}

// __TEXT segment lays out backwards so padding is at front after load commands.
void Util::layoutSectionsInTextSegment(size_t hlcSize, SegmentInfo *seg,
                                                               uint64_t &addr) {
  seg->address = addr;
  // Walks sections starting at end to calculate padding for start.
  int64_t taddr = 0;
  for (auto it = seg->sections.rbegin(); it != seg->sections.rend(); ++it) {
    SectionInfo *sect = *it;
    taddr -= sect->size;
    taddr = taddr & (0 - sect->alignment);
  }
  int64_t padding = taddr - hlcSize;
  while (padding < 0)
    padding += _ctx.pageSize();
  // Start assigning section address starting at padded offset.
  addr += (padding + hlcSize);
  for (SectionInfo *sect : seg->sections) {
    sect->address = llvm::alignTo(addr, sect->alignment);
    addr = sect->address + sect->size;
  }
  seg->size = llvm::alignTo(addr - seg->address, _ctx.pageSize());
}

void Util::assignAddressesToSections(const NormalizedFile &file) {
  // NOTE!: Keep this in sync with organizeSections.
  size_t hlcSize = headerAndLoadCommandsSize(file);
  uint64_t address = 0;
  for (SegmentInfo *seg : _segmentInfos) {
    if (seg->name.equals("__PAGEZERO")) {
      seg->size = _ctx.pageZeroSize();
      address += seg->size;
    }
    else if (seg->name.equals("__TEXT")) {
      // _ctx.baseAddress()  == 0 implies it was either unspecified or
      // pageZeroSize is also 0. In either case resetting address is safe.
      address = _ctx.baseAddress() ? _ctx.baseAddress() : address;
      layoutSectionsInTextSegment(hlcSize, seg, address);
    } else
      layoutSectionsInSegment(seg, address);

    address = llvm::alignTo(address, _ctx.pageSize());
  }
  DEBUG_WITH_TYPE("WriterMachO-norm",
    llvm::dbgs() << "assignAddressesToSections()\n";
    for (SegmentInfo *sgi : _segmentInfos) {
      llvm::dbgs()  << "   address=" << llvm::format("0x%08llX", sgi->address)
                    << ", size="  << llvm::format("0x%08llX", sgi->size)
                    << ", segment-name='" << sgi->name
                    << "'\n";
      for (SectionInfo *si : sgi->sections) {
        llvm::dbgs()<< "      addr="  << llvm::format("0x%08llX", si->address)
                    << ", size="  << llvm::format("0x%08llX", si->size)
                    << ", section-name='" << si->sectionName
                    << "\n";
      }
    }
  );
}

void Util::copySegmentInfo(NormalizedFile &file) {
  for (SegmentInfo *sgi : _segmentInfos) {
    Segment seg;
    seg.name    = sgi->name;
    seg.address = sgi->address;
    seg.size    = sgi->size;
    seg.access  = sgi->access;
    file.segments.push_back(seg);
  }
}

void Util::appendSection(SectionInfo *si, NormalizedFile &file) {
   // Add new empty section to end of file.sections.
  Section temp;
  file.sections.push_back(std::move(temp));
  Section* normSect = &file.sections.back();
  // Copy fields to normalized section.
  normSect->segmentName   = si->segmentName;
  normSect->sectionName   = si->sectionName;
  normSect->type          = si->type;
  normSect->attributes    = si->attributes;
  normSect->address       = si->address;
  normSect->alignment     = si->alignment;
  // Record where normalized section is.
  si->normalizedSectionIndex = file.sections.size()-1;
}

void Util::copySectionContent(NormalizedFile &file) {
  const bool r = (_ctx.outputMachOType() == llvm::MachO::MH_OBJECT);

  // Utility function for ArchHandler to find address of atom in output file.
  auto addrForAtom = [&] (const Atom &atom) -> uint64_t {
    auto pos = _atomToAddress.find(&atom);
    assert(pos != _atomToAddress.end());
    return pos->second;
  };

  auto sectionAddrForAtom = [&] (const Atom &atom) -> uint64_t {
    for (const SectionInfo *sectInfo : _sectionInfos)
      for (const AtomInfo &atomInfo : sectInfo->atomsAndOffsets)
        if (atomInfo.atom == &atom)
          return sectInfo->address;
    llvm_unreachable("atom not assigned to section");
  };

  for (SectionInfo *si : _sectionInfos) {
    Section *normSect = &file.sections[si->normalizedSectionIndex];
    if (isZeroFillSection(si->type)) {
      const uint8_t *empty = nullptr;
      normSect->content = llvm::makeArrayRef(empty, si->size);
      continue;
    }
    // Copy content from atoms to content buffer for section.
    uint8_t *sectionContent = file.ownedAllocations.Allocate<uint8_t>(si->size);
    normSect->content = llvm::makeArrayRef(sectionContent, si->size);
    for (AtomInfo &ai : si->atomsAndOffsets) {
      uint8_t *atomContent = reinterpret_cast<uint8_t*>
                                          (&sectionContent[ai.offsetInSection]);
      _archHandler.generateAtomContent(*ai.atom, r, addrForAtom,
                                       sectionAddrForAtom, _ctx.baseAddress(),
                                       atomContent);
    }
  }
}

void Util::copySectionInfo(NormalizedFile &file) {
  file.sections.reserve(_sectionInfos.size());
  // Write sections grouped by segment.
  for (SegmentInfo *sgi : _segmentInfos) {
    for (SectionInfo *si : sgi->sections) {
      appendSection(si, file);
    }
  }
}

void Util::updateSectionInfo(NormalizedFile &file) {
  file.sections.reserve(_sectionInfos.size());
  // sections grouped by segment.
  for (SegmentInfo *sgi : _segmentInfos) {
    Segment *normSeg = &file.segments[sgi->normalizedSegmentIndex];
    normSeg->address = sgi->address;
    normSeg->size = sgi->size;
    for (SectionInfo *si : sgi->sections) {
      Section *normSect = &file.sections[si->normalizedSectionIndex];
      normSect->address = si->address;
    }
  }
}

void Util::copyEntryPointAddress(NormalizedFile &nFile) {
  if (_ctx.outputTypeHasEntry()) {
    if (_archHandler.isThumbFunction(*_entryAtom))
      nFile.entryAddress = (_atomToAddress[_entryAtom] | 1);
    else
      nFile.entryAddress = _atomToAddress[_entryAtom];
  }
}

void Util::buildAtomToAddressMap() {
  DEBUG_WITH_TYPE("WriterMachO-address", llvm::dbgs()
                   << "assign atom addresses:\n");
  const bool lookForEntry = _ctx.outputTypeHasEntry();
  for (SectionInfo *sect : _sectionInfos) {
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      _atomToAddress[info.atom] = sect->address + info.offsetInSection;
      if (lookForEntry && (info.atom->contentType() == DefinedAtom::typeCode) &&
          (info.atom->size() != 0) &&
          info.atom->name() == _ctx.entrySymbolName()) {
        _entryAtom = info.atom;
      }
      DEBUG_WITH_TYPE("WriterMachO-address", llvm::dbgs()
                      << "   address="
                      << llvm::format("0x%016X", _atomToAddress[info.atom])
                      << llvm::format("    0x%09lX", info.atom)
                      << ", file=#"
                      << info.atom->file().ordinal()
                      << ", atom=#"
                      << info.atom->ordinal()
                      << ", name="
                      << info.atom->name()
                      << ", type="
                      << info.atom->contentType()
                      << "\n");
    }
  }
  DEBUG_WITH_TYPE("WriterMachO-address", llvm::dbgs()
                  << "assign header alias atom addresses:\n");
  for (const Atom *atom : _machHeaderAliasAtoms) {
    _atomToAddress[atom] = _ctx.baseAddress();
#ifndef NDEBUG
    if (auto *definedAtom = dyn_cast<DefinedAtom>(atom)) {
      DEBUG_WITH_TYPE("WriterMachO-address", llvm::dbgs()
                      << "   address="
                      << llvm::format("0x%016X", _atomToAddress[atom])
                      << llvm::format("    0x%09lX", atom)
                      << ", file=#"
                      << definedAtom->file().ordinal()
                      << ", atom=#"
                      << definedAtom->ordinal()
                      << ", name="
                      << definedAtom->name()
                      << ", type="
                      << definedAtom->contentType()
                      << "\n");
    } else {
      DEBUG_WITH_TYPE("WriterMachO-address", llvm::dbgs()
                      << "   address="
                      << llvm::format("0x%016X", _atomToAddress[atom])
                      << " atom=" << atom
                      << " name=" << atom->name() << "\n");
    }
#endif
  }
}

uint16_t Util::descBits(const DefinedAtom* atom) {
  uint16_t desc = 0;
  switch (atom->merge()) {
  case lld::DefinedAtom::mergeNo:
  case lld::DefinedAtom::mergeAsTentative:
    break;
  case lld::DefinedAtom::mergeAsWeak:
  case lld::DefinedAtom::mergeAsWeakAndAddressUsed:
    desc |= N_WEAK_DEF;
    break;
  case lld::DefinedAtom::mergeSameNameAndSize:
  case lld::DefinedAtom::mergeByLargestSection:
  case lld::DefinedAtom::mergeByContent:
    llvm_unreachable("Unsupported DefinedAtom::merge()");
    break;
  }
  if (atom->contentType() == lld::DefinedAtom::typeResolver)
    desc |= N_SYMBOL_RESOLVER;
  if (_archHandler.isThumbFunction(*atom))
    desc |= N_ARM_THUMB_DEF;
  if (atom->deadStrip() == DefinedAtom::deadStripNever) {
    if ((atom->contentType() != DefinedAtom::typeInitializerPtr)
     && (atom->contentType() != DefinedAtom::typeTerminatorPtr))
    desc |= N_NO_DEAD_STRIP;
  }
  return desc;
}

bool Util::AtomSorter::operator()(const AtomAndIndex &left,
                                  const AtomAndIndex &right) {
  return (left.atom->name().compare(right.atom->name()) < 0);
}

std::error_code Util::getSymbolTableRegion(const DefinedAtom* atom,
                                           bool &inGlobalsRegion,
                                           SymbolScope &scope) {
  bool rMode = (_ctx.outputMachOType() == llvm::MachO::MH_OBJECT);
  switch (atom->scope()) {
  case Atom::scopeTranslationUnit:
    scope = 0;
    inGlobalsRegion = false;
    return std::error_code();
  case Atom::scopeLinkageUnit:
    if ((_ctx.exportMode() == MachOLinkingContext::ExportMode::whiteList) &&
        _ctx.exportSymbolNamed(atom->name())) {
      return make_dynamic_error_code(Twine("cannot export hidden symbol ")
                                    + atom->name());
    }
    if (rMode) {
      if (_ctx.keepPrivateExterns()) {
        // -keep_private_externs means keep in globals region as N_PEXT.
        scope = N_PEXT | N_EXT;
        inGlobalsRegion = true;
        return std::error_code();
      }
    }
    // scopeLinkageUnit symbols are no longer global once linked.
    scope = N_PEXT;
    inGlobalsRegion = false;
    return std::error_code();
  case Atom::scopeGlobal:
    if (_ctx.exportRestrictMode()) {
      if (_ctx.exportSymbolNamed(atom->name())) {
        scope = N_EXT;
        inGlobalsRegion = true;
        return std::error_code();
      } else {
        scope = N_PEXT;
        inGlobalsRegion = false;
        return std::error_code();
      }
    } else {
      scope = N_EXT;
      inGlobalsRegion = true;
      return std::error_code();
    }
    break;
  }
  llvm_unreachable("atom->scope() unknown enum value");
}

std::error_code Util::addSymbols(const lld::File &atomFile,
                                 NormalizedFile &file) {
  bool rMode = (_ctx.outputMachOType() == llvm::MachO::MH_OBJECT);
  // Mach-O symbol table has three regions: locals, globals, undefs.

  // Add all local (non-global) symbols in address order
  std::vector<AtomAndIndex> globals;
  globals.reserve(512);
  for (SectionInfo *sect : _sectionInfos) {
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      if (!atom->name().empty()) {
        SymbolScope symbolScope;
        bool inGlobalsRegion;
        if (auto ec = getSymbolTableRegion(atom, inGlobalsRegion, symbolScope)){
          return ec;
        }
        if (inGlobalsRegion) {
          AtomAndIndex ai = { atom, sect->finalSectionIndex, symbolScope };
          globals.push_back(ai);
        } else {
          Symbol sym;
          sym.name  = atom->name();
          sym.type  = N_SECT;
          sym.scope = symbolScope;
          sym.sect  = sect->finalSectionIndex;
          sym.desc  = descBits(atom);
          sym.value = _atomToAddress[atom];
          _atomToSymbolIndex[atom] = file.localSymbols.size();
          file.localSymbols.push_back(sym);
        }
      } else if (rMode && _archHandler.needsLocalSymbolInRelocatableFile(atom)){
        // Create 'Lxxx' labels for anonymous atoms if archHandler says so.
        static unsigned tempNum = 1;
        char tmpName[16];
        sprintf(tmpName, "L%04u", tempNum++);
        StringRef tempRef(tmpName);
        Symbol sym;
        sym.name  = tempRef.copy(file.ownedAllocations);
        sym.type  = N_SECT;
        sym.scope = 0;
        sym.sect  = sect->finalSectionIndex;
        sym.desc  = 0;
        sym.value = _atomToAddress[atom];
        _atomToSymbolIndex[atom] = file.localSymbols.size();
        file.localSymbols.push_back(sym);
      }
    }
  }

  // Sort global symbol alphabetically, then add to symbol table.
  std::sort(globals.begin(), globals.end(), AtomSorter());
  const uint32_t globalStartIndex = file.localSymbols.size();
  for (AtomAndIndex &ai : globals) {
    Symbol sym;
    sym.name  = ai.atom->name();
    sym.type  = N_SECT;
    sym.scope = ai.scope;
    sym.sect  = ai.index;
    sym.desc  = descBits(static_cast<const DefinedAtom*>(ai.atom));
    sym.value = _atomToAddress[ai.atom];
    _atomToSymbolIndex[ai.atom] = globalStartIndex + file.globalSymbols.size();
    file.globalSymbols.push_back(sym);
  }

  // Sort undefined symbol alphabetically, then add to symbol table.
  std::vector<AtomAndIndex> undefs;
  undefs.reserve(128);
  for (const UndefinedAtom *atom : atomFile.undefined()) {
    AtomAndIndex ai = { atom, 0, N_EXT };
    undefs.push_back(ai);
  }
  for (const SharedLibraryAtom *atom : atomFile.sharedLibrary()) {
    AtomAndIndex ai = { atom, 0, N_EXT };
    undefs.push_back(ai);
  }
  std::sort(undefs.begin(), undefs.end(), AtomSorter());
  const uint32_t start = file.globalSymbols.size() + file.localSymbols.size();
  for (AtomAndIndex &ai : undefs) {
    Symbol sym;
    uint16_t desc = 0;
    if (!rMode) {
      uint8_t ordinal = 0;
      if (!_ctx.useFlatNamespace())
        ordinal = dylibOrdinal(dyn_cast<SharedLibraryAtom>(ai.atom));
      llvm::MachO::SET_LIBRARY_ORDINAL(desc, ordinal);
    }
    sym.name  = ai.atom->name();
    sym.type  = N_UNDF;
    sym.scope = ai.scope;
    sym.sect  = 0;
    sym.desc  = desc;
    sym.value = 0;
    _atomToSymbolIndex[ai.atom] = file.undefinedSymbols.size() + start;
    file.undefinedSymbols.push_back(sym);
  }

  return std::error_code();
}

const Atom *Util::targetOfLazyPointer(const DefinedAtom *lpAtom) {
  for (const Reference *ref : *lpAtom) {
    if (_archHandler.isLazyPointer(*ref)) {
      return ref->target();
    }
  }
  return nullptr;
}

const Atom *Util::targetOfStub(const DefinedAtom *stubAtom) {
  for (const Reference *ref : *stubAtom) {
    if (const Atom *ta = ref->target()) {
      if (const DefinedAtom *lpAtom = dyn_cast<DefinedAtom>(ta)) {
        const Atom *target = targetOfLazyPointer(lpAtom);
        if (target)
          return target;
      }
    }
  }
  return nullptr;
}

void Util::addIndirectSymbols(const lld::File &atomFile, NormalizedFile &file) {
  for (SectionInfo *si : _sectionInfos) {
    Section &normSect = file.sections[si->normalizedSectionIndex];
    switch (si->type) {
    case llvm::MachO::S_NON_LAZY_SYMBOL_POINTERS:
      for (const AtomInfo &info : si->atomsAndOffsets) {
        bool foundTarget = false;
        for (const Reference *ref : *info.atom) {
          const Atom *target = ref->target();
          if (target) {
            if (isa<const SharedLibraryAtom>(target)) {
              uint32_t index = _atomToSymbolIndex[target];
              normSect.indirectSymbols.push_back(index);
              foundTarget = true;
            } else {
              normSect.indirectSymbols.push_back(
                                            llvm::MachO::INDIRECT_SYMBOL_LOCAL);
            }
          }
        }
        if (!foundTarget) {
          normSect.indirectSymbols.push_back(
                                             llvm::MachO::INDIRECT_SYMBOL_ABS);
        }
      }
      break;
    case llvm::MachO::S_LAZY_SYMBOL_POINTERS:
      for (const AtomInfo &info : si->atomsAndOffsets) {
        const Atom *target = targetOfLazyPointer(info.atom);
        if (target) {
          uint32_t index = _atomToSymbolIndex[target];
          normSect.indirectSymbols.push_back(index);
        }
      }
      break;
    case llvm::MachO::S_SYMBOL_STUBS:
      for (const AtomInfo &info : si->atomsAndOffsets) {
        const Atom *target = targetOfStub(info.atom);
        if (target) {
          uint32_t index = _atomToSymbolIndex[target];
          normSect.indirectSymbols.push_back(index);
        }
      }
      break;
    default:
      break;
    }
  }
}

void Util::addDependentDylibs(const lld::File &atomFile,NormalizedFile &nFile) {
  // Scan all imported symbols and build up list of dylibs they are from.
  int ordinal = 1;
  for (const SharedLibraryAtom *slAtom : atomFile.sharedLibrary()) {
    StringRef loadPath = slAtom->loadName();
    DylibPathToInfo::iterator pos = _dylibInfo.find(loadPath);
    if (pos == _dylibInfo.end()) {
      DylibInfo info;
      bool flatNamespaceAtom = &slAtom->file() == _ctx.flatNamespaceFile();

      // If we're in -flat_namespace mode (or this atom came from the flat
      // namespace file under -undefined dynamic_lookup) then use the flat
      // lookup ordinal.
      if (flatNamespaceAtom || _ctx.useFlatNamespace())
        info.ordinal = BIND_SPECIAL_DYLIB_FLAT_LOOKUP;
      else
        info.ordinal = ordinal++;
      info.hasWeak = slAtom->canBeNullAtRuntime();
      info.hasNonWeak = !info.hasWeak;
      _dylibInfo[loadPath] = info;

      // Unless this was a flat_namespace atom, record the source dylib.
      if (!flatNamespaceAtom) {
        DependentDylib depInfo;
        depInfo.path = loadPath;
        depInfo.kind = llvm::MachO::LC_LOAD_DYLIB;
        depInfo.currentVersion = _ctx.dylibCurrentVersion(loadPath);
        depInfo.compatVersion = _ctx.dylibCompatVersion(loadPath);
        nFile.dependentDylibs.push_back(depInfo);
      }
    } else {
      if ( slAtom->canBeNullAtRuntime() )
        pos->second.hasWeak = true;
      else
        pos->second.hasNonWeak = true;
    }
  }
  // Automatically weak link dylib in which all symbols are weak (canBeNull).
  for (DependentDylib &dep : nFile.dependentDylibs) {
    DylibInfo &info = _dylibInfo[dep.path];
    if (info.hasWeak && !info.hasNonWeak)
      dep.kind = llvm::MachO::LC_LOAD_WEAK_DYLIB;
    else if (_ctx.isUpwardDylib(dep.path))
      dep.kind = llvm::MachO::LC_LOAD_UPWARD_DYLIB;
  }
}

int Util::dylibOrdinal(const SharedLibraryAtom *sa) {
  return _dylibInfo[sa->loadName()].ordinal;
}

void Util::segIndexForSection(const SectionInfo *sect, uint8_t &segmentIndex,
                                                  uint64_t &segmentStartAddr) {
  segmentIndex = 0;
  for (const SegmentInfo *seg : _segmentInfos) {
    if ((seg->address <= sect->address)
      && (seg->address+seg->size >= sect->address+sect->size)) {
      segmentStartAddr = seg->address;
      return;
    }
    ++segmentIndex;
  }
  llvm_unreachable("section not in any segment");
}

uint32_t Util::sectionIndexForAtom(const Atom *atom) {
  uint64_t address = _atomToAddress[atom];
  for (const SectionInfo *si : _sectionInfos) {
    if ((si->address <= address) && (address < si->address+si->size))
      return si->finalSectionIndex;
  }
  llvm_unreachable("atom not in any section");
}

void Util::addSectionRelocs(const lld::File &, NormalizedFile &file) {
  if (_ctx.outputMachOType() != llvm::MachO::MH_OBJECT)
    return;

  // Utility function for ArchHandler to find symbol index for an atom.
  auto symIndexForAtom = [&] (const Atom &atom) -> uint32_t {
    auto pos = _atomToSymbolIndex.find(&atom);
    assert(pos != _atomToSymbolIndex.end());
    return pos->second;
  };

  // Utility function for ArchHandler to find section index for an atom.
  auto sectIndexForAtom = [&] (const Atom &atom) -> uint32_t {
    return sectionIndexForAtom(&atom);
  };

  // Utility function for ArchHandler to find address of atom in output file.
  auto addressForAtom = [&] (const Atom &atom) -> uint64_t {
    auto pos = _atomToAddress.find(&atom);
    assert(pos != _atomToAddress.end());
    return pos->second;
  };

  for (SectionInfo *si : _sectionInfos) {
    Section &normSect = file.sections[si->normalizedSectionIndex];
    for (const AtomInfo &info : si->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      for (const Reference *ref : *atom) {
        // Skip emitting relocs for sections which are always able to be
        // implicitly regenerated and where the relocation targets an address
        // which is defined.
        if (si->relocsToDefinedCanBeImplicit && isa<DefinedAtom>(ref->target()))
          continue;
        _archHandler.appendSectionRelocations(*atom, info.offsetInSection, *ref,
                                              symIndexForAtom,
                                              sectIndexForAtom,
                                              addressForAtom,
                                              normSect.relocations);
      }
    }
  }
}

void Util::buildDataInCodeArray(const lld::File &, NormalizedFile &file) {
  for (SectionInfo *si : _sectionInfos) {
    for (const AtomInfo &info : si->atomsAndOffsets) {
      // Atoms that contain data-in-code have "transition" references
      // which mark a point where the embedded data starts of ends.
      // This needs to be converted to the mach-o format which is an array
      // of data-in-code ranges.
      uint32_t startOffset = 0;
      DataRegionType mode = DataRegionType(0);
      for (const Reference *ref : *info.atom) {
        if (ref->kindNamespace() != Reference::KindNamespace::mach_o)
          continue;
        if (_archHandler.isDataInCodeTransition(ref->kindValue())) {
          DataRegionType nextMode = (DataRegionType)ref->addend();
          if (mode != nextMode) {
            if (mode != 0) {
              // Found end data range, so make range entry.
              DataInCode entry;
              entry.offset = si->address + info.offsetInSection + startOffset;
              entry.length = ref->offsetInAtom() - startOffset;
              entry.kind   = mode;
              file.dataInCode.push_back(entry);
            }
          }
          mode = nextMode;
          startOffset = ref->offsetInAtom();
        }
      }
      if (mode != 0) {
        // Function ends with data (no end transition).
        DataInCode entry;
        entry.offset = si->address + info.offsetInSection + startOffset;
        entry.length = info.atom->size() - startOffset;
        entry.kind   = mode;
        file.dataInCode.push_back(entry);
      }
    }
  }
}

void Util::addRebaseAndBindingInfo(const lld::File &atomFile,
                                                        NormalizedFile &nFile) {
  if (_ctx.outputMachOType() == llvm::MachO::MH_OBJECT)
    return;

  uint8_t segmentIndex;
  uint64_t segmentStartAddr;
  for (SectionInfo *sect : _sectionInfos) {
    segIndexForSection(sect, segmentIndex, segmentStartAddr);
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      for (const Reference *ref : *atom) {
        uint64_t segmentOffset = _atomToAddress[atom] + ref->offsetInAtom()
                                - segmentStartAddr;
        const Atom* targ = ref->target();
        if (_archHandler.isPointer(*ref)) {
          // A pointer to a DefinedAtom requires rebasing.
          if (isa<DefinedAtom>(targ)) {
            RebaseLocation rebase;
            rebase.segIndex = segmentIndex;
            rebase.segOffset = segmentOffset;
            rebase.kind = llvm::MachO::REBASE_TYPE_POINTER;
            nFile.rebasingInfo.push_back(rebase);
          }
          // A pointer to an SharedLibraryAtom requires binding.
          if (const SharedLibraryAtom *sa = dyn_cast<SharedLibraryAtom>(targ)) {
            BindLocation bind;
            bind.segIndex = segmentIndex;
            bind.segOffset = segmentOffset;
            bind.kind = llvm::MachO::BIND_TYPE_POINTER;
            bind.canBeNull = sa->canBeNullAtRuntime();
            bind.ordinal = dylibOrdinal(sa);
            bind.symbolName = targ->name();
            bind.addend = ref->addend();
            nFile.bindingInfo.push_back(bind);
          }
        }
        else if (_archHandler.isLazyPointer(*ref)) {
          BindLocation bind;
          if (const SharedLibraryAtom *sa = dyn_cast<SharedLibraryAtom>(targ)) {
            bind.ordinal = dylibOrdinal(sa);
          } else {
            bind.ordinal = llvm::MachO::BIND_SPECIAL_DYLIB_SELF;
          }
          bind.segIndex = segmentIndex;
          bind.segOffset = segmentOffset;
          bind.kind = llvm::MachO::BIND_TYPE_POINTER;
          bind.canBeNull = false; //sa->canBeNullAtRuntime();
          bind.symbolName = targ->name();
          bind.addend = ref->addend();
          nFile.lazyBindingInfo.push_back(bind);
        }
      }
    }
  }
}

void Util::addExportInfo(const lld::File &atomFile, NormalizedFile &nFile) {
  if (_ctx.outputMachOType() == llvm::MachO::MH_OBJECT)
    return;

  for (SectionInfo *sect : _sectionInfos) {
    for (const AtomInfo &info : sect->atomsAndOffsets) {
      const DefinedAtom *atom = info.atom;
      if (atom->scope() != Atom::scopeGlobal)
        continue;
      if (_ctx.exportRestrictMode()) {
        if (!_ctx.exportSymbolNamed(atom->name()))
          continue;
      }
      Export exprt;
      exprt.name = atom->name();
      exprt.offset = _atomToAddress[atom] - _ctx.baseAddress();
      exprt.kind = EXPORT_SYMBOL_FLAGS_KIND_REGULAR;
      if (atom->merge() == DefinedAtom::mergeAsWeak)
        exprt.flags = EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION;
      else
        exprt.flags = 0;
      exprt.otherOffset = 0;
      exprt.otherName = StringRef();
      nFile.exportInfo.push_back(exprt);
    }
  }
}

uint32_t Util::fileFlags() {
  // FIXME: these need to determined at runtime.
  if (_ctx.outputMachOType() == MH_OBJECT) {
    return _subsectionsViaSymbols ? MH_SUBSECTIONS_VIA_SYMBOLS : 0;
  } else {
    uint32_t flags = MH_DYLDLINK;
    if (!_ctx.useFlatNamespace())
        flags |= MH_TWOLEVEL | MH_NOUNDEFS;
    if ((_ctx.outputMachOType() == MH_EXECUTE) && _ctx.PIE())
      flags |= MH_PIE;
    if (_hasTLVDescriptors)
      flags |= (MH_PIE | MH_HAS_TLV_DESCRIPTORS);
    return flags;
  }
}

} // end anonymous namespace

namespace lld {
namespace mach_o {
namespace normalized {

/// Convert a set of Atoms into a normalized mach-o file.
ErrorOr<std::unique_ptr<NormalizedFile>>
normalizedFromAtoms(const lld::File &atomFile,
                                           const MachOLinkingContext &context) {
  // The util object buffers info until the normalized file can be made.
  Util util(context);
  util.processDefinedAtoms(atomFile);
  util.organizeSections();

  std::unique_ptr<NormalizedFile> f(new NormalizedFile());
  NormalizedFile &normFile = *f.get();
  normFile.arch = context.arch();
  normFile.fileType = context.outputMachOType();
  normFile.flags = util.fileFlags();
  normFile.stackSize = context.stackSize();
  normFile.installName = context.installName();
  normFile.currentVersion = context.currentVersion();
  normFile.compatVersion = context.compatibilityVersion();
  normFile.pageSize = context.pageSize();
  normFile.rpaths = context.rpaths();
  util.addDependentDylibs(atomFile, normFile);
  util.copySegmentInfo(normFile);
  util.copySectionInfo(normFile);
  util.assignAddressesToSections(normFile);
  util.buildAtomToAddressMap();
  util.updateSectionInfo(normFile);
  util.copySectionContent(normFile);
  if (auto ec = util.addSymbols(atomFile, normFile)) {
    return ec;
  }
  util.addIndirectSymbols(atomFile, normFile);
  util.addRebaseAndBindingInfo(atomFile, normFile);
  util.addExportInfo(atomFile, normFile);
  util.addSectionRelocs(atomFile, normFile);
  util.buildDataInCodeArray(atomFile, normFile);
  util.copyEntryPointAddress(normFile);

  return std::move(f);
}

} // namespace normalized
} // namespace mach_o
} // namespace lld
