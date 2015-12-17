//===- lib/ReaderWriter/MachO/MachONormalizedFileToAtoms.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

///
/// \file Converts from in-memory normalized mach-o to in-memory Atoms.
///
///                  +------------+
///                  | normalized |
///                  +------------+
///                        |
///                        |
///                        v
///                    +-------+
///                    | Atoms |
///                    +-------+

#include "MachONormalizedFile.h"
#include "ArchHandler.h"
#include "Atoms.h"
#include "File.h"
#include "MachONormalizedFileBinaryUtils.h"
#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::MachO;
using namespace lld::mach_o::normalized;

#define DEBUG_TYPE "normalized-file-to-atoms"

namespace lld {
namespace mach_o {


namespace { // anonymous


#define ENTRY(seg, sect, type, atomType) \
  {seg, sect, type, DefinedAtom::atomType }

struct MachORelocatableSectionToAtomType {
  StringRef                 segmentName;
  StringRef                 sectionName;
  SectionType               sectionType;
  DefinedAtom::ContentType  atomType;
};

const MachORelocatableSectionToAtomType sectsToAtomType[] = {
  ENTRY("__TEXT", "__text",           S_REGULAR,          typeCode),
  ENTRY("__TEXT", "__text",           S_REGULAR,          typeResolver),
  ENTRY("__TEXT", "__cstring",        S_CSTRING_LITERALS, typeCString),
  ENTRY("",       "",                 S_CSTRING_LITERALS, typeCString),
  ENTRY("__TEXT", "__ustring",        S_REGULAR,          typeUTF16String),
  ENTRY("__TEXT", "__const",          S_REGULAR,          typeConstant),
  ENTRY("__TEXT", "__const_coal",     S_COALESCED,        typeConstant),
  ENTRY("__TEXT", "__eh_frame",       S_COALESCED,        typeCFI),
  ENTRY("__TEXT", "__eh_frame",       S_REGULAR,          typeCFI),
  ENTRY("__TEXT", "__literal4",       S_4BYTE_LITERALS,   typeLiteral4),
  ENTRY("__TEXT", "__literal8",       S_8BYTE_LITERALS,   typeLiteral8),
  ENTRY("__TEXT", "__literal16",      S_16BYTE_LITERALS,  typeLiteral16),
  ENTRY("__TEXT", "__gcc_except_tab", S_REGULAR,          typeLSDA),
  ENTRY("__DATA", "__data",           S_REGULAR,          typeData),
  ENTRY("__DATA", "__datacoal_nt",    S_COALESCED,        typeData),
  ENTRY("__DATA", "__const",          S_REGULAR,          typeConstData),
  ENTRY("__DATA", "__cfstring",       S_REGULAR,          typeCFString),
  ENTRY("__DATA", "__mod_init_func",  S_MOD_INIT_FUNC_POINTERS,
                                                          typeInitializerPtr),
  ENTRY("__DATA", "__mod_term_func",  S_MOD_TERM_FUNC_POINTERS,
                                                          typeTerminatorPtr),
  ENTRY("__DATA", "__got",            S_NON_LAZY_SYMBOL_POINTERS,
                                                          typeGOT),
  ENTRY("__DATA", "__bss",            S_ZEROFILL,         typeZeroFill),
  ENTRY("",       "",                 S_NON_LAZY_SYMBOL_POINTERS,
                                                          typeGOT),
  ENTRY("__DATA", "__interposing",    S_INTERPOSING,      typeInterposingTuples),
  ENTRY("__DATA", "__thread_vars",    S_THREAD_LOCAL_VARIABLES,
                                                          typeThunkTLV),
  ENTRY("__DATA", "__thread_data", S_THREAD_LOCAL_REGULAR, typeTLVInitialData),
  ENTRY("__DATA", "__thread_bss",     S_THREAD_LOCAL_ZEROFILL,
                                                        typeTLVInitialZeroFill),
  ENTRY("",       "",                 S_INTERPOSING,      typeInterposingTuples),
  ENTRY("__LD",   "__compact_unwind", S_REGULAR,
                                                         typeCompactUnwindInfo),
  ENTRY("",       "",                 S_REGULAR,          typeUnknown)
};
#undef ENTRY


/// Figures out ContentType of a mach-o section.
DefinedAtom::ContentType atomTypeFromSection(const Section &section,
                                             bool &customSectionName) {
  // First look for match of name and type. Empty names in table are wildcards.
  customSectionName = false;
  for (const MachORelocatableSectionToAtomType *p = sectsToAtomType ;
                                 p->atomType != DefinedAtom::typeUnknown; ++p) {
    if (p->sectionType != section.type)
      continue;
    if (!p->segmentName.equals(section.segmentName) && !p->segmentName.empty())
      continue;
    if (!p->sectionName.equals(section.sectionName) && !p->sectionName.empty())
      continue;
    customSectionName = p->segmentName.empty() && p->sectionName.empty();
    return p->atomType;
  }
  // Look for code denoted by section attributes
  if (section.attributes & S_ATTR_PURE_INSTRUCTIONS)
    return DefinedAtom::typeCode;

  return DefinedAtom::typeUnknown;
}

enum AtomizeModel {
  atomizeAtSymbols,
  atomizeFixedSize,
  atomizePointerSize,
  atomizeUTF8,
  atomizeUTF16,
  atomizeCFI,
  atomizeCU,
  atomizeCFString
};

/// Returns info on how to atomize a section of the specified ContentType.
void sectionParseInfo(DefinedAtom::ContentType atomType,
                      unsigned int &sizeMultiple,
                      DefinedAtom::Scope &scope,
                      DefinedAtom::Merge &merge,
                      AtomizeModel &atomizeModel) {
  struct ParseInfo {
    DefinedAtom::ContentType  atomType;
    unsigned int              sizeMultiple;
    DefinedAtom::Scope        scope;
    DefinedAtom::Merge        merge;
    AtomizeModel              atomizeModel;
  };

  #define ENTRY(type, size, scope, merge, model) \
    {DefinedAtom::type, size, DefinedAtom::scope, DefinedAtom::merge, model }

  static const ParseInfo parseInfo[] = {
    ENTRY(typeCode,              1, scopeGlobal,          mergeNo,
                                                            atomizeAtSymbols),
    ENTRY(typeData,              1, scopeGlobal,          mergeNo,
                                                            atomizeAtSymbols),
    ENTRY(typeConstData,         1, scopeGlobal,          mergeNo,
                                                            atomizeAtSymbols),
    ENTRY(typeZeroFill,          1, scopeGlobal,          mergeNo,
                                                            atomizeAtSymbols),
    ENTRY(typeConstant,          1, scopeGlobal,          mergeNo,
                                                            atomizeAtSymbols),
    ENTRY(typeCString,           1, scopeLinkageUnit,     mergeByContent,
                                                            atomizeUTF8),
    ENTRY(typeUTF16String,       1, scopeLinkageUnit,     mergeByContent,
                                                            atomizeUTF16),
    ENTRY(typeCFI,               4, scopeTranslationUnit, mergeNo,
                                                            atomizeCFI),
    ENTRY(typeLiteral4,          4, scopeLinkageUnit,     mergeByContent,
                                                            atomizeFixedSize),
    ENTRY(typeLiteral8,          8, scopeLinkageUnit,     mergeByContent,
                                                            atomizeFixedSize),
    ENTRY(typeLiteral16,        16, scopeLinkageUnit,     mergeByContent,
                                                            atomizeFixedSize),
    ENTRY(typeCFString,          4, scopeLinkageUnit,     mergeByContent,
                                                            atomizeCFString),
    ENTRY(typeInitializerPtr,    4, scopeTranslationUnit, mergeNo,
                                                            atomizePointerSize),
    ENTRY(typeTerminatorPtr,     4, scopeTranslationUnit, mergeNo,
                                                            atomizePointerSize),
    ENTRY(typeCompactUnwindInfo, 4, scopeTranslationUnit, mergeNo,
                                                            atomizeCU),
    ENTRY(typeGOT,               4, scopeLinkageUnit,     mergeByContent,
                                                            atomizePointerSize),
    ENTRY(typeUnknown,           1, scopeGlobal,          mergeNo,
                                                            atomizeAtSymbols)
  };
  #undef ENTRY
  const int tableLen = sizeof(parseInfo) / sizeof(ParseInfo);
  for (int i=0; i < tableLen; ++i) {
    if (parseInfo[i].atomType == atomType) {
      sizeMultiple = parseInfo[i].sizeMultiple;
      scope        = parseInfo[i].scope;
      merge        = parseInfo[i].merge;
      atomizeModel = parseInfo[i].atomizeModel;
      return;
    }
  }

  // Unknown type is atomized by symbols.
  sizeMultiple = 1;
  scope = DefinedAtom::scopeGlobal;
  merge = DefinedAtom::mergeNo;
  atomizeModel = atomizeAtSymbols;
}


Atom::Scope atomScope(uint8_t scope) {
  switch (scope) {
  case N_EXT:
    return Atom::scopeGlobal;
  case N_PEXT:
  case N_PEXT | N_EXT:
    return Atom::scopeLinkageUnit;
  case 0:
    return Atom::scopeTranslationUnit;
  }
  llvm_unreachable("unknown scope value!");
}

void appendSymbolsInSection(const std::vector<Symbol> &inSymbols,
                            uint32_t sectionIndex,
                            SmallVector<const Symbol *, 64> &outSyms) {
  for (const Symbol &sym : inSymbols) {
    // Only look at definition symbols.
    if ((sym.type & N_TYPE) != N_SECT)
      continue;
    if (sym.sect != sectionIndex)
      continue;
    outSyms.push_back(&sym);
  }
}

void atomFromSymbol(DefinedAtom::ContentType atomType, const Section &section,
                    MachOFile &file, uint64_t symbolAddr, StringRef symbolName,
                    uint16_t symbolDescFlags, Atom::Scope symbolScope,
                    uint64_t nextSymbolAddr, bool scatterable, bool copyRefs) {
  // Mach-O symbol table does have size in it. Instead the size is the
  // difference between this and the next symbol.
  uint64_t size = nextSymbolAddr - symbolAddr;
  uint64_t offset = symbolAddr - section.address;
  bool noDeadStrip = (symbolDescFlags & N_NO_DEAD_STRIP) || !scatterable;
  if (isZeroFillSection(section.type)) {
    file.addZeroFillDefinedAtom(symbolName, symbolScope, offset, size,
                                noDeadStrip, copyRefs, &section);
  } else {
    DefinedAtom::Merge merge = (symbolDescFlags & N_WEAK_DEF)
                              ? DefinedAtom::mergeAsWeak : DefinedAtom::mergeNo;
    bool thumb = (symbolDescFlags & N_ARM_THUMB_DEF);
    if (atomType == DefinedAtom::typeUnknown) {
      // Mach-O needs a segment and section name.  Concatentate those two
      // with a / separator (e.g. "seg/sect") to fit into the lld model
      // of just a section name.
      std::string segSectName = section.segmentName.str()
                                + "/" + section.sectionName.str();
      file.addDefinedAtomInCustomSection(symbolName, symbolScope, atomType,
                                         merge, thumb, noDeadStrip, offset,
                                         size, segSectName, true, &section);
    } else {
      if ((atomType == lld::DefinedAtom::typeCode) &&
          (symbolDescFlags & N_SYMBOL_RESOLVER)) {
        atomType = lld::DefinedAtom::typeResolver;
      }
      file.addDefinedAtom(symbolName, symbolScope, atomType, merge,
                          offset, size, thumb, noDeadStrip, copyRefs, &section);
    }
  }
}

std::error_code processSymboledSection(DefinedAtom::ContentType atomType,
                                       const Section &section,
                                       const NormalizedFile &normalizedFile,
                                       MachOFile &file, bool scatterable,
                                       bool copyRefs) {
  // Find section's index.
  uint32_t sectIndex = 1;
  for (auto &sect : normalizedFile.sections) {
    if (&sect == &section)
      break;
    ++sectIndex;
  }

  // Find all symbols in this section.
  SmallVector<const Symbol *, 64> symbols;
  appendSymbolsInSection(normalizedFile.globalSymbols, sectIndex, symbols);
  appendSymbolsInSection(normalizedFile.localSymbols,  sectIndex, symbols);

  // Sort symbols.
  std::sort(symbols.begin(), symbols.end(),
            [](const Symbol *lhs, const Symbol *rhs) -> bool {
              if (lhs == rhs)
                return false;
              // First by address.
              uint64_t lhsAddr = lhs->value;
              uint64_t rhsAddr = rhs->value;
              if (lhsAddr != rhsAddr)
                return lhsAddr < rhsAddr;
               // If same address, one is an alias so sort by scope.
              Atom::Scope lScope = atomScope(lhs->scope);
              Atom::Scope rScope = atomScope(rhs->scope);
              if (lScope != rScope)
                return lScope < rScope;
              // If same address and scope, see if one might be better as
              // the alias.
              bool lPrivate = (lhs->name.front() == 'l');
              bool rPrivate = (rhs->name.front() == 'l');
              if (lPrivate != rPrivate)
                return lPrivate;
              // If same address and scope, sort by name.
              return lhs->name < rhs->name;
            });

  // Debug logging of symbols.
  //for (const Symbol *sym : symbols)
  //  llvm::errs() << "  sym: "
  //    << llvm::format("0x%08llx ", (uint64_t)sym->value)
  //    << ", " << sym->name << "\n";

  // If section has no symbols and no content, there are no atoms.
  if (symbols.empty() && section.content.empty())
    return std::error_code();

  if (symbols.empty()) {
    // Section has no symbols, put all content in one anoymous atom.
    atomFromSymbol(atomType, section, file, section.address, StringRef(),
                  0, Atom::scopeTranslationUnit,
                  section.address + section.content.size(),
                  scatterable, copyRefs);
  }
  else if (symbols.front()->value != section.address) {
    // Section has anonymous content before first symbol.
    atomFromSymbol(atomType, section, file, section.address, StringRef(),
                   0, Atom::scopeTranslationUnit, symbols.front()->value,
                   scatterable, copyRefs);
  }

  const Symbol *lastSym = nullptr;
  for (const Symbol *sym : symbols) {
    if (lastSym != nullptr) {
      // Ignore any assembler added "ltmpNNN" symbol at start of section
      // if there is another symbol at the start.
      if ((lastSym->value != sym->value)
          || lastSym->value != section.address
          || !lastSym->name.startswith("ltmp")) {
        atomFromSymbol(atomType, section, file, lastSym->value, lastSym->name,
                       lastSym->desc, atomScope(lastSym->scope), sym->value,
                       scatterable, copyRefs);
      }
    }
    lastSym = sym;
  }
  if (lastSym != nullptr) {
    atomFromSymbol(atomType, section, file, lastSym->value, lastSym->name,
                   lastSym->desc, atomScope(lastSym->scope),
                   section.address + section.content.size(),
                   scatterable, copyRefs);
  }

  // If object built without .subsections_via_symbols, add reference chain.
  if (!scatterable) {
    MachODefinedAtom *prevAtom = nullptr;
    file.eachAtomInSection(section,
                           [&](MachODefinedAtom *atom, uint64_t offset)->void {
      if (prevAtom)
        prevAtom->addReference(0, Reference::kindLayoutAfter, atom, 0,
                               Reference::KindArch::all,
                               Reference::KindNamespace::all);
      prevAtom = atom;
    });
  }

  return std::error_code();
}

std::error_code processSection(DefinedAtom::ContentType atomType,
                               const Section &section,
                               bool customSectionName,
                               const NormalizedFile &normalizedFile,
                               MachOFile &file, bool scatterable,
                               bool copyRefs) {
  const bool is64 = MachOLinkingContext::is64Bit(normalizedFile.arch);
  const bool isBig = MachOLinkingContext::isBigEndian(normalizedFile.arch);

  // Get info on how to atomize section.
  unsigned int       sizeMultiple;
  DefinedAtom::Scope scope;
  DefinedAtom::Merge merge;
  AtomizeModel       atomizeModel;
  sectionParseInfo(atomType, sizeMultiple, scope, merge, atomizeModel);

  // Validate section size.
  if ((section.content.size() % sizeMultiple) != 0)
    return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName
                                     + " has size ("
                                     + Twine(section.content.size())
                                     + ") which is not a multiple of "
                                     + Twine(sizeMultiple) );

  if (atomizeModel == atomizeAtSymbols) {
    // Break section up into atoms each with a fixed size.
    return processSymboledSection(atomType, section, normalizedFile, file,
                                  scatterable, copyRefs);
  } else {
    unsigned int size;
    for (unsigned int offset = 0, e = section.content.size(); offset != e;) {
      switch (atomizeModel) {
      case atomizeFixedSize:
        // Break section up into atoms each with a fixed size.
        size = sizeMultiple;
        break;
      case atomizePointerSize:
        // Break section up into atoms each the size of a pointer.
        size = is64 ? 8 : 4;
        break;
      case atomizeUTF8:
        // Break section up into zero terminated c-strings.
        size = 0;
        for (unsigned int i = offset; i < e; ++i) {
          if (section.content[i] == 0) {
            size = i + 1 - offset;
            break;
          }
        }
        break;
      case atomizeUTF16:
        // Break section up into zero terminated UTF16 strings.
        size = 0;
        for (unsigned int i = offset; i < e; i += 2) {
          if ((section.content[i] == 0) && (section.content[i + 1] == 0)) {
            size = i + 2 - offset;
            break;
          }
        }
        break;
      case atomizeCFI:
        // Break section up into dwarf unwind CFIs (FDE or CIE).
        size = read32(&section.content[offset], isBig) + 4;
        if (offset+size > section.content.size()) {
          return make_dynamic_error_code(Twine(Twine("Section ")
                                         + section.segmentName
                                         + "/" + section.sectionName
                                         + " is malformed.  Size of CFI "
                                         "starting at offset ("
                                         + Twine(offset)
                                         + ") is past end of section."));
        }
        break;
      case atomizeCU:
        // Break section up into compact unwind entries.
        size = is64 ? 32 : 20;
        break;
      case atomizeCFString:
        // Break section up into NS/CFString objects.
        size = is64 ? 32 : 16;
        break;
      case atomizeAtSymbols:
        break;
      }
      if (size == 0) {
        return make_dynamic_error_code(Twine("Section ") + section.segmentName
                                     + "/" + section.sectionName
                                     + " is malformed.  The last atom is "
                                     "not zero terminated.");
      }
      if (customSectionName) {
        // Mach-O needs a segment and section name.  Concatentate those two
        // with a / separator (e.g. "seg/sect") to fit into the lld model
        // of just a section name.
        std::string segSectName = section.segmentName.str()
                                  + "/" + section.sectionName.str();
        file.addDefinedAtomInCustomSection(StringRef(), scope, atomType,
                                           merge, false, false, offset,
                                           size, segSectName, true, &section);
      } else {
        file.addDefinedAtom(StringRef(), scope, atomType, merge, offset, size,
                            false, false, copyRefs, &section);
      }
      offset += size;
    }
  }
  return std::error_code();
}

const Section* findSectionCoveringAddress(const NormalizedFile &normalizedFile,
                                          uint64_t address) {
  for (const Section &s : normalizedFile.sections) {
    uint64_t sAddr = s.address;
    if ((sAddr <= address) && (address < sAddr+s.content.size())) {
      return &s;
    }
  }
  return nullptr;
}

const MachODefinedAtom *
findAtomCoveringAddress(const NormalizedFile &normalizedFile, MachOFile &file,
                        uint64_t addr, Reference::Addend *addend) {
  const Section *sect = nullptr;
  sect = findSectionCoveringAddress(normalizedFile, addr);
  if (!sect)
    return nullptr;

  uint32_t offsetInTarget;
  uint64_t offsetInSect = addr - sect->address;
  auto atom =
      file.findAtomCoveringAddress(*sect, offsetInSect, &offsetInTarget);
  *addend = offsetInTarget;
  return atom;
}

// Walks all relocations for a section in a normalized .o file and
// creates corresponding lld::Reference objects.
std::error_code convertRelocs(const Section &section,
                              const NormalizedFile &normalizedFile,
                              bool scatterable,
                              MachOFile &file,
                              ArchHandler &handler) {
  // Utility function for ArchHandler to find atom by its address.
  auto atomByAddr = [&] (uint32_t sectIndex, uint64_t addr,
                         const lld::Atom **atom, Reference::Addend *addend)
                         -> std::error_code {
    if (sectIndex > normalizedFile.sections.size())
      return make_dynamic_error_code(Twine("out of range section "
                                     "index (") + Twine(sectIndex) + ")");
    const Section *sect = nullptr;
    if (sectIndex == 0) {
      sect = findSectionCoveringAddress(normalizedFile, addr);
      if (!sect)
        return make_dynamic_error_code(Twine("address (" + Twine(addr)
                                       + ") is not in any section"));
    } else {
      sect = &normalizedFile.sections[sectIndex-1];
    }
    uint32_t offsetInTarget;
    uint64_t offsetInSect = addr - sect->address;
    *atom = file.findAtomCoveringAddress(*sect, offsetInSect, &offsetInTarget);
    *addend = offsetInTarget;
    return std::error_code();
  };

  // Utility function for ArchHandler to find atom by its symbol index.
  auto atomBySymbol = [&] (uint32_t symbolIndex, const lld::Atom **result)
                           -> std::error_code {
    // Find symbol from index.
    const Symbol *sym = nullptr;
    uint32_t numLocal  = normalizedFile.localSymbols.size();
    uint32_t numGlobal = normalizedFile.globalSymbols.size();
    uint32_t numUndef  = normalizedFile.undefinedSymbols.size();
    if (symbolIndex < numLocal) {
      sym = &normalizedFile.localSymbols[symbolIndex];
    } else if (symbolIndex < numLocal+numGlobal) {
      sym = &normalizedFile.globalSymbols[symbolIndex-numLocal];
    } else if (symbolIndex < numLocal+numGlobal+numUndef) {
      sym = &normalizedFile.undefinedSymbols[symbolIndex-numLocal-numGlobal];
    } else {
      return make_dynamic_error_code(Twine("symbol index (")
                                     + Twine(symbolIndex) + ") out of range");
    }
    // Find atom from symbol.
    if ((sym->type & N_TYPE) == N_SECT) {
      if (sym->sect > normalizedFile.sections.size())
        return make_dynamic_error_code(Twine("symbol section index (")
                                        + Twine(sym->sect) + ") out of range ");
      const Section &symSection = normalizedFile.sections[sym->sect-1];
      uint64_t targetOffsetInSect = sym->value - symSection.address;
      MachODefinedAtom *target = file.findAtomCoveringAddress(symSection,
                                                            targetOffsetInSect);
      if (target) {
        *result = target;
        return std::error_code();
      }
      return make_dynamic_error_code("no atom found for defined symbol");
    } else if ((sym->type & N_TYPE) == N_UNDF) {
      const lld::Atom *target = file.findUndefAtom(sym->name);
      if (target) {
        *result = target;
        return std::error_code();
      }
      return make_dynamic_error_code("no undefined atom found for sym");
    } else {
      // Search undefs
      return make_dynamic_error_code("no atom found for symbol");
    }
  };

  const bool isBig = MachOLinkingContext::isBigEndian(normalizedFile.arch);
  // Use old-school iterator so that paired relocations can be grouped.
  for (auto it=section.relocations.begin(), e=section.relocations.end();
                                                                it != e; ++it) {
    const Relocation &reloc = *it;
    // Find atom this relocation is in.
    if (reloc.offset > section.content.size())
      return make_dynamic_error_code(Twine("r_address (") + Twine(reloc.offset)
                                    + ") is larger than section size ("
                                    + Twine(section.content.size()) + ")");
    uint32_t offsetInAtom;
    MachODefinedAtom *inAtom = file.findAtomCoveringAddress(section,
                                                            reloc.offset,
                                                            &offsetInAtom);
    assert(inAtom && "r_address in range, should have found atom");
    uint64_t fixupAddress = section.address + reloc.offset;

    const lld::Atom *target = nullptr;
    Reference::Addend addend = 0;
    Reference::KindValue kind;
    std::error_code relocErr;
    if (handler.isPairedReloc(reloc)) {
      // Handle paired relocations together.
      const Relocation &reloc2 = *++it;
      relocErr = handler.getPairReferenceInfo(
          reloc, reloc2, inAtom, offsetInAtom, fixupAddress, isBig, scatterable,
          atomByAddr, atomBySymbol, &kind, &target, &addend);
      if (relocErr) {
        return make_dynamic_error_code(
          Twine("bad relocation (") + relocErr.message()
           + ") in section "
           + section.segmentName + "/" + section.sectionName
           + " (r1_address=" + Twine::utohexstr(reloc.offset)
           + ", r1_type=" + Twine(reloc.type)
           + ", r1_extern=" + Twine(reloc.isExtern)
           + ", r1_length=" + Twine((int)reloc.length)
           + ", r1_pcrel=" + Twine(reloc.pcRel)
           + (!reloc.scattered ? (Twine(", r1_symbolnum=")
                                  + Twine(reloc.symbol))
                               : (Twine(", r1_scattered=1, r1_value=")
                                  + Twine(reloc.value)))
           + ")"
           + ", (r2_address=" + Twine::utohexstr(reloc2.offset)
           + ", r2_type=" + Twine(reloc2.type)
           + ", r2_extern=" + Twine(reloc2.isExtern)
           + ", r2_length=" + Twine((int)reloc2.length)
           + ", r2_pcrel=" + Twine(reloc2.pcRel)
           + (!reloc2.scattered ? (Twine(", r2_symbolnum=")
                                   + Twine(reloc2.symbol))
                                : (Twine(", r2_scattered=1, r2_value=")
                                   + Twine(reloc2.value)))
           + ")" );
      }
    }
    else {
      // Use ArchHandler to convert relocation record into information
      // needed to instantiate an lld::Reference object.
      relocErr = handler.getReferenceInfo(
          reloc, inAtom, offsetInAtom, fixupAddress, isBig, atomByAddr,
          atomBySymbol, &kind, &target, &addend);
      if (relocErr) {
        return make_dynamic_error_code(
          Twine("bad relocation (") + relocErr.message()
           + ") in section "
           + section.segmentName + "/" + section.sectionName
           + " (r_address=" + Twine::utohexstr(reloc.offset)
           + ", r_type=" + Twine(reloc.type)
           + ", r_extern=" + Twine(reloc.isExtern)
           + ", r_length=" + Twine((int)reloc.length)
           + ", r_pcrel=" + Twine(reloc.pcRel)
           + (!reloc.scattered ? (Twine(", r_symbolnum=") + Twine(reloc.symbol))
                               : (Twine(", r_scattered=1, r_value=")
                                  + Twine(reloc.value)))
           + ")" );
      }
    }
    // Instantiate an lld::Reference object and add to its atom.
    inAtom->addReference(offsetInAtom, kind, target, addend,
                         handler.kindArch());
  }

  return std::error_code();
}

bool isDebugInfoSection(const Section &section) {
  if ((section.attributes & S_ATTR_DEBUG) == 0)
    return false;
  return section.segmentName.equals("__DWARF");
}

static int64_t readSPtr(bool is64, bool isBig, const uint8_t *addr) {
  if (is64)
    return read64(addr, isBig);

  int32_t res = read32(addr, isBig);
  return res;
}

/// --- Augmentation String Processing ---

struct CIEInfo {
  bool _augmentationDataPresent = false;
  bool _mayHaveLSDA = false;
};

typedef llvm::DenseMap<const MachODefinedAtom*, CIEInfo> CIEInfoMap;

static std::error_code processAugmentationString(const uint8_t *augStr,
                                                 CIEInfo &cieInfo,
                                                 unsigned *len = nullptr) {

  if (augStr[0] == '\0') {
    if (len)
      *len = 1;
    return std::error_code();
  }

  if (augStr[0] != 'z')
    return make_dynamic_error_code("expected 'z' at start of augmentation "
                                   "string");

  cieInfo._augmentationDataPresent = true;
  uint64_t idx = 1;

  while (augStr[idx] != '\0') {
    if (augStr[idx] == 'L') {
      cieInfo._mayHaveLSDA = true;
      ++idx;
    } else
      ++idx;
  }

  if (len)
    *len = idx + 1;
  return std::error_code();
}

static std::error_code processCIE(const NormalizedFile &normalizedFile,
                                  MachODefinedAtom *atom,
                                  CIEInfoMap &cieInfos) {
  const bool isBig = MachOLinkingContext::isBigEndian(normalizedFile.arch);
  const uint8_t *frameData = atom->rawContent().data();

  CIEInfo cieInfo;

  uint32_t size = read32(frameData, isBig);
  uint64_t cieIDField = size == 0xffffffffU
                          ? sizeof(uint32_t) + sizeof(uint64_t)
                          : sizeof(uint32_t);
  uint64_t versionField = cieIDField + sizeof(uint32_t);
  uint64_t augmentationStringField = versionField + sizeof(uint8_t);

  if (auto err = processAugmentationString(frameData + augmentationStringField,
                                           cieInfo))
    return err;

  cieInfos[atom] = std::move(cieInfo);

  return std::error_code();
}

static std::error_code processFDE(const NormalizedFile &normalizedFile,
                                  MachOFile &file,
                                  mach_o::ArchHandler &handler,
                                  const Section *ehFrameSection,
                                  MachODefinedAtom *atom,
                                  uint64_t offset,
                                  const CIEInfoMap &cieInfos) {

  const bool isBig = MachOLinkingContext::isBigEndian(normalizedFile.arch);
  const bool is64 = MachOLinkingContext::is64Bit(normalizedFile.arch);

  // Compiler wasn't lazy and actually told us what it meant.
  if (atom->begin() != atom->end())
    return std::error_code();

  const uint8_t *frameData = atom->rawContent().data();
  uint32_t size = read32(frameData, isBig);
  uint64_t cieFieldInFDE = size == 0xffffffffU
    ? sizeof(uint32_t) + sizeof(uint64_t)
    : sizeof(uint32_t);

  // Linker needs to fixup a reference from the FDE to its parent CIE (a
  // 32-bit byte offset backwards in the __eh_frame section).
  uint32_t cieDelta = read32(frameData + cieFieldInFDE, isBig);
  uint64_t cieAddress = ehFrameSection->address + offset + cieFieldInFDE;
  cieAddress -= cieDelta;

  Reference::Addend addend;
  const MachODefinedAtom *cie =
    findAtomCoveringAddress(normalizedFile, file, cieAddress, &addend);
  atom->addReference(cieFieldInFDE, handler.unwindRefToCIEKind(), cie,
                     addend, handler.kindArch());

  assert(cie && cie->contentType() == DefinedAtom::typeCFI && !addend &&
         "FDE's CIE field does not point at the start of a CIE.");

  const CIEInfo &cieInfo = cieInfos.find(cie)->second;

  // Linker needs to fixup reference from the FDE to the function it's
  // describing. FIXME: there are actually different ways to do this, and the
  // particular method used is specified in the CIE's augmentation fields
  // (hopefully)
  uint64_t rangeFieldInFDE = cieFieldInFDE + sizeof(uint32_t);

  int64_t functionFromFDE = readSPtr(is64, isBig,
                                     frameData + rangeFieldInFDE);
  uint64_t rangeStart = ehFrameSection->address + offset + rangeFieldInFDE;
  rangeStart += functionFromFDE;

  const Atom *func =
    findAtomCoveringAddress(normalizedFile, file, rangeStart, &addend);
  atom->addReference(rangeFieldInFDE, handler.unwindRefToFunctionKind(),
                     func, addend, handler.kindArch());

  // Handle the augmentation data if there is any.
  if (cieInfo._augmentationDataPresent) {
    // First process the augmentation data length field.
    uint64_t augmentationDataLengthFieldInFDE =
      rangeFieldInFDE + 2 * (is64 ? sizeof(uint64_t) : sizeof(uint32_t));
    unsigned lengthFieldSize = 0;
    uint64_t augmentationDataLength =
      llvm::decodeULEB128(frameData + augmentationDataLengthFieldInFDE,
                          &lengthFieldSize);

    if (cieInfo._mayHaveLSDA && augmentationDataLength > 0) {

      // Look at the augmentation data field.
      uint64_t augmentationDataFieldInFDE =
        augmentationDataLengthFieldInFDE + lengthFieldSize;

      int64_t lsdaFromFDE = readSPtr(is64, isBig,
                                     frameData + augmentationDataFieldInFDE);
      uint64_t lsdaStart =
        ehFrameSection->address + offset + augmentationDataFieldInFDE +
        lsdaFromFDE;
      const Atom *lsda =
        findAtomCoveringAddress(normalizedFile, file, lsdaStart, &addend);
      atom->addReference(augmentationDataFieldInFDE,
                         handler.unwindRefToFunctionKind(),
                         lsda, addend, handler.kindArch());
    }
  }

  return std::error_code();
}

std::error_code addEHFrameReferences(const NormalizedFile &normalizedFile,
                                     MachOFile &file,
                                     mach_o::ArchHandler &handler) {

  const Section *ehFrameSection = nullptr;
  for (auto &section : normalizedFile.sections)
    if (section.segmentName == "__TEXT" &&
        section.sectionName == "__eh_frame") {
      ehFrameSection = &section;
      break;
    }

  // No __eh_frame so nothing to do.
  if (!ehFrameSection)
    return std::error_code();

  std::error_code ehFrameErr;
  CIEInfoMap cieInfos;

  file.eachAtomInSection(*ehFrameSection,
                         [&](MachODefinedAtom *atom, uint64_t offset) -> void {
    assert(atom->contentType() == DefinedAtom::typeCFI);

    // Bail out if we've encountered an error.
    if (ehFrameErr)
      return;

    const bool isBig = MachOLinkingContext::isBigEndian(normalizedFile.arch);
    if (ArchHandler::isDwarfCIE(isBig, atom))
      ehFrameErr = processCIE(normalizedFile, atom, cieInfos);
    else
      ehFrameErr = processFDE(normalizedFile, file, handler, ehFrameSection,
                              atom, offset, cieInfos);
  });

  return ehFrameErr;
}


/// Converts normalized mach-o file into an lld::File and lld::Atoms.
ErrorOr<std::unique_ptr<lld::File>>
objectToAtoms(const NormalizedFile &normalizedFile, StringRef path,
              bool copyRefs) {
  std::unique_ptr<MachOFile> file(new MachOFile(path));
  if (std::error_code ec = normalizedObjectToAtoms(
        file.get(), normalizedFile, copyRefs))
    return ec;
  return std::unique_ptr<File>(std::move(file));
}

ErrorOr<std::unique_ptr<lld::File>>
dylibToAtoms(const NormalizedFile &normalizedFile, StringRef path,
             bool copyRefs) {
  // Instantiate SharedLibraryFile object.
  std::unique_ptr<MachODylibFile> file(new MachODylibFile(path));
  normalizedDylibToAtoms(file.get(), normalizedFile, copyRefs);
  return std::unique_ptr<File>(std::move(file));
}

} // anonymous namespace

namespace normalized {

std::error_code
normalizedObjectToAtoms(MachOFile *file,
                        const NormalizedFile &normalizedFile,
                        bool copyRefs) {
  DEBUG(llvm::dbgs() << "******** Normalizing file to atoms: "
                    << file->path() << "\n");
  bool scatterable = ((normalizedFile.flags & MH_SUBSECTIONS_VIA_SYMBOLS) != 0);

  // Create atoms from each section.
  for (auto &sect : normalizedFile.sections) {
    DEBUG(llvm::dbgs() << "Creating atoms: "; sect.dump());
    if (isDebugInfoSection(sect))
      continue;
    bool customSectionName;
    DefinedAtom::ContentType atomType = atomTypeFromSection(sect,
                                                            customSectionName);
    if (std::error_code ec =
            processSection(atomType, sect, customSectionName, normalizedFile,
                           *file, scatterable, copyRefs))
      return ec;
  }
  // Create atoms from undefined symbols.
  for (auto &sym : normalizedFile.undefinedSymbols) {
    // Undefinded symbols with n_value != 0 are actually tentative definitions.
    if (sym.value == Hex64(0)) {
      file->addUndefinedAtom(sym.name, copyRefs);
    } else {
      file->addTentativeDefAtom(sym.name, atomScope(sym.scope), sym.value,
                                DefinedAtom::Alignment(1 << (sym.desc >> 8)),
                                copyRefs);
    }
  }

  // Convert mach-o relocations to References
  std::unique_ptr<mach_o::ArchHandler> handler
                                     = ArchHandler::create(normalizedFile.arch);
  for (auto &sect : normalizedFile.sections) {
    if (isDebugInfoSection(sect))
      continue;
    if (std::error_code ec = convertRelocs(sect, normalizedFile, scatterable,
                                           *file, *handler))
        return ec;
  }

  // Add additional arch-specific References
  file->eachDefinedAtom([&](MachODefinedAtom* atom) -> void {
    handler->addAdditionalReferences(*atom);
  });

  // Each __eh_frame section needs references to both __text (the function we're
  // providing unwind info for) and itself (FDE -> CIE). These aren't
  // represented in the relocations on some architectures, so we have to add
  // them back in manually there.
  if (std::error_code ec = addEHFrameReferences(normalizedFile, *file, *handler))
    return ec;

  // Process mach-o data-in-code regions array. That information is encoded in
  // atoms as References at each transition point.
  unsigned nextIndex = 0;
  for (const DataInCode &entry : normalizedFile.dataInCode) {
    ++nextIndex;
    const Section* s = findSectionCoveringAddress(normalizedFile, entry.offset);
    if (!s) {
      return make_dynamic_error_code(Twine("LC_DATA_IN_CODE address ("
                                     + Twine(entry.offset)
                                     + ") is not in any section"));
    }
    uint64_t offsetInSect = entry.offset - s->address;
    uint32_t offsetInAtom;
    MachODefinedAtom *atom = file->findAtomCoveringAddress(*s, offsetInSect,
                                                           &offsetInAtom);
    if (offsetInAtom + entry.length > atom->size()) {
      return make_dynamic_error_code(Twine("LC_DATA_IN_CODE entry (offset="
                                     + Twine(entry.offset)
                                     + ", length="
                                     + Twine(entry.length)
                                     + ") crosses atom boundary."));
    }
    // Add reference that marks start of data-in-code.
    atom->addReference(offsetInAtom,
                       handler->dataInCodeTransitionStart(*atom), atom,
                       entry.kind, handler->kindArch());

    // Peek at next entry, if it starts where this one ends, skip ending ref.
    if (nextIndex < normalizedFile.dataInCode.size()) {
      const DataInCode &nextEntry = normalizedFile.dataInCode[nextIndex];
      if (nextEntry.offset == (entry.offset + entry.length))
        continue;
    }

    // If data goes to end of function, skip ending ref.
    if ((offsetInAtom + entry.length) == atom->size())
      continue;

    // Add reference that marks end of data-in-code.
    atom->addReference(offsetInAtom+entry.length,
                       handler->dataInCodeTransitionEnd(*atom), atom, 0,
                       handler->kindArch());
  }

  // Sort references in each atom to their canonical order.
  for (const DefinedAtom* defAtom : file->defined()) {
    reinterpret_cast<const SimpleDefinedAtom*>(defAtom)->sortReferences();
  }
  return std::error_code();
}

std::error_code
normalizedDylibToAtoms(MachODylibFile *file,
                       const NormalizedFile &normalizedFile,
                       bool copyRefs) {
  file->setInstallName(normalizedFile.installName);
  file->setCompatVersion(normalizedFile.compatVersion);
  file->setCurrentVersion(normalizedFile.currentVersion);

  // Tell MachODylibFile object about all symbols it exports.
  if (!normalizedFile.exportInfo.empty()) {
    // If exports trie exists, use it instead of traditional symbol table.
    for (const Export &exp : normalizedFile.exportInfo) {
      bool weakDef = (exp.flags & EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION);
      // StringRefs from export iterator are ephemeral, so force copy.
      file->addExportedSymbol(exp.name, weakDef, true);
    }
  } else {
    for (auto &sym : normalizedFile.globalSymbols) {
      assert((sym.scope & N_EXT) && "only expect external symbols here");
      bool weakDef = (sym.desc & N_WEAK_DEF);
      file->addExportedSymbol(sym.name, weakDef, copyRefs);
    }
  }
  // Tell MachODylibFile object about all dylibs it re-exports.
  for (const DependentDylib &dep : normalizedFile.dependentDylibs) {
    if (dep.kind == llvm::MachO::LC_REEXPORT_DYLIB)
      file->addReExportedDylib(dep.path);
  }
  return std::error_code();
}

void relocatableSectionInfoForContentType(DefinedAtom::ContentType atomType,
                                          StringRef &segmentName,
                                          StringRef &sectionName,
                                          SectionType &sectionType,
                                          SectionAttr &sectionAttrs) {

  for (const MachORelocatableSectionToAtomType *p = sectsToAtomType ;
                                 p->atomType != DefinedAtom::typeUnknown; ++p) {
    if (p->atomType != atomType)
      continue;
    // Wild carded entries are ignored for reverse lookups.
    if (p->segmentName.empty() || p->sectionName.empty())
      continue;
    segmentName = p->segmentName;
    sectionName = p->sectionName;
    sectionType = p->sectionType;
    sectionAttrs = 0;
    if (atomType == DefinedAtom::typeCode)
      sectionAttrs = S_ATTR_PURE_INSTRUCTIONS;
    return;
  }
  llvm_unreachable("content type not yet supported");
}

ErrorOr<std::unique_ptr<lld::File>>
normalizedToAtoms(const NormalizedFile &normalizedFile, StringRef path,
                  bool copyRefs) {
  switch (normalizedFile.fileType) {
  case MH_DYLIB:
  case MH_DYLIB_STUB:
    return dylibToAtoms(normalizedFile, path, copyRefs);
  case MH_OBJECT:
    return objectToAtoms(normalizedFile, path, copyRefs);
  default:
    llvm_unreachable("unhandled MachO file type!");
  }
}

#ifndef NDEBUG
void Section::dump(llvm::raw_ostream &OS) const {
  OS << "Section (\"" << segmentName << ", " << sectionName << "\"";
  OS << ", addr: " << llvm::format_hex(address, 16, true);
  OS << ", size: " << llvm::format_hex(content.size(), 8, true) << ")\n";
}
#endif

} // namespace normalized
} // namespace mach_o
} // namespace lld
