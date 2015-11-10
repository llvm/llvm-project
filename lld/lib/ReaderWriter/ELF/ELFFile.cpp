//===- lib/ReaderWriter/ELF/ELFFile.cpp -------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ELFFile.h"
#include "FileCommon.h"
#include "llvm/ADT/STLExtras.h"

namespace lld {
namespace elf {

template <typename ELFT>
ELFFile<ELFT>::ELFFile(StringRef name, ELFLinkingContext &ctx)
    : SimpleFile(name), _ordinal(0), _doStringsMerge(ctx.mergeCommonStrings()),
      _useWrap(false), _ctx(ctx) {
  setLastError(std::error_code());
}

template <typename ELFT>
ELFFile<ELFT>::ELFFile(std::unique_ptr<MemoryBuffer> mb, ELFLinkingContext &ctx)
    : SimpleFile(mb->getBufferIdentifier()), _mb(std::move(mb)), _ordinal(0),
      _doStringsMerge(ctx.mergeCommonStrings()),
      _useWrap(ctx.wrapCalls().size()), _ctx(ctx) {}

template <typename ELFT>
std::error_code ELFFile<ELFT>::isCompatible(MemoryBufferRef mb,
                                            ELFLinkingContext &ctx) {
  return elf::isCompatible<ELFT>(mb, ctx);
}

template <typename ELFT>
Atom *ELFFile<ELFT>::findAtom(const Elf_Sym *sourceSym,
                              const Elf_Sym *targetSym) {
  // Return the atom for targetSym if we can do so.
  Atom *target = _symbolToAtomMapping.lookup(targetSym);
  if (!target)
    // Some realocations (R_ARM_V4BX) do not have a defined
    // target.  For this cases make it points to itself.
    target = _symbolToAtomMapping.lookup(sourceSym);

  if (target->definition() != Atom::definitionRegular)
    return target;
  Atom::Scope scope = llvm::cast<DefinedAtom>(target)->scope();
  if (scope == DefinedAtom::scopeTranslationUnit)
    return target;
  if (!redirectReferenceUsingUndefAtom(sourceSym, targetSym))
    return target;

  // Otherwise, create a new undefined symbol and returns it.
  StringRef targetName = target->name();
  auto it = _undefAtomsForGroupChild.find(targetName);
  if (it != _undefAtomsForGroupChild.end())
    return it->getValue();
  auto atom = new (_readerStorage) SimpleUndefinedAtom(*this, targetName);
  _undefAtomsForGroupChild[targetName] = atom;
  addAtom(*atom);
  return atom;
}

template <typename ELFT>
ErrorOr<StringRef> ELFFile<ELFT>::getSectionName(const Elf_Shdr *shdr) const {
  if (!shdr)
    return StringRef();
  return _objFile->getSectionName(shdr);
}

template <class ELFT> std::error_code ELFFile<ELFT>::doParse() {
  std::error_code ec;
  _objFile.reset(new llvm::object::ELFFile<ELFT>(_mb->getBuffer(), ec));
  if (ec)
    return ec;

  if ((ec = createAtomsFromContext()))
    return ec;

  // Read input sections from the input file that need to be converted to
  // atoms
  if ((ec = createAtomizableSections()))
    return ec;

  // For mergeable strings, we would need to split the section into various
  // atoms
  if ((ec = createMergeableAtoms()))
    return ec;

  // Create the necessary symbols that are part of the section that we
  // created in createAtomizableSections function
  if ((ec = createSymbolsFromAtomizableSections()))
    return ec;

  // Create the appropriate atoms from the file
  if ((ec = createAtoms()))
    return ec;
  return std::error_code();
}

template <class ELFT> Reference::KindArch ELFFile<ELFT>::kindArch() {
  switch (_objFile->getHeader()->e_machine) {
  case llvm::ELF::EM_X86_64:
    return Reference::KindArch::x86_64;
  case llvm::ELF::EM_386:
    return Reference::KindArch::x86;
  case llvm::ELF::EM_ARM:
    return Reference::KindArch::ARM;
  case llvm::ELF::EM_HEXAGON:
    return Reference::KindArch::Hexagon;
  case llvm::ELF::EM_MIPS:
    return Reference::KindArch::Mips;
  case llvm::ELF::EM_AARCH64:
    return Reference::KindArch::AArch64;
  }
  llvm_unreachable("unsupported e_machine value");
}

template <class ELFT>
std::error_code ELFFile<ELFT>::createAtomizableSections() {
  // Handle: SHT_REL and SHT_RELA sections:
  // Increment over the sections, when REL/RELA section types are found add
  // the contents to the RelocationReferences map.
  // Record the number of relocs to guess at preallocating the buffer.
  uint64_t totalRelocs = 0;
  for (const Elf_Shdr &section : _objFile->sections()) {
    switch (section.sh_type) {
    case llvm::ELF::SHT_SYMTAB:
      _symtab = &section;
      continue;
    case llvm::ELF::SHT_SYMTAB_SHNDX: {
      ErrorOr<ArrayRef<Elf_Word>> tableOrErr = _objFile->getSHNDXTable(section);
      if (std::error_code ec = tableOrErr.getError())
        return ec;
      _shndxTable = *tableOrErr;
      continue;
    }
    }

    if (isIgnoredSection(&section))
      continue;

    if (isMergeableStringSection(&section)) {
      _mergeStringSections.push_back(&section);
      continue;
    }

    if (section.sh_type == llvm::ELF::SHT_RELA) {
      auto sHdrOrErr = _objFile->getSection(section.sh_info);
      if (std::error_code ec = sHdrOrErr.getError())
        return ec;
      auto sHdr = *sHdrOrErr;
      auto rai = _objFile->rela_begin(&section);
      auto rae = _objFile->rela_end(&section);
      _relocationAddendReferences[sHdr] = make_range(rai, rae);
      totalRelocs += std::distance(rai, rae);
    } else if (section.sh_type == llvm::ELF::SHT_REL) {
      auto sHdrOrErr = _objFile->getSection(section.sh_info);
      if (std::error_code ec = sHdrOrErr.getError())
        return ec;
      auto sHdr = *sHdrOrErr;
      auto ri = _objFile->rel_begin(&section);
      auto re = _objFile->rel_end(&section);
      _relocationReferences[sHdr] = &section;
      totalRelocs += std::distance(ri, re);
    } else {
      auto sectionName = _objFile->getSectionName(&section);
      if (std::error_code ec = sectionName.getError())
        return ec;
      _ctx.notifyInputSectionName(*sectionName);
      _sectionSymbols[&section];
    }
  }
  _references.reserve(totalRelocs);
  return std::error_code();
}

template <class ELFT> std::error_code ELFFile<ELFT>::createMergeableAtoms() {
  // Divide the section that contains mergeable strings into tokens
  // TODO
  // a) add resolver support to recognize multibyte chars
  // b) Create a separate section chunk to write mergeable atoms
  std::vector<MergeString *> tokens;
  for (const Elf_Shdr *msi : _mergeStringSections) {
    auto sectionName = getSectionName(msi);
    if (std::error_code ec = sectionName.getError())
      return ec;

    auto sectionContents = getSectionContents(msi);
    if (std::error_code ec = sectionContents.getError())
      return ec;

    StringRef secCont(reinterpret_cast<const char *>(sectionContents->begin()),
                      sectionContents->size());

    unsigned int prev = 0;
    for (std::size_t i = 0, e = sectionContents->size(); i != e; ++i) {
      if ((*sectionContents)[i] == '\0') {
        tokens.push_back(new (_readerStorage) MergeString(
            prev, secCont.slice(prev, i + 1), msi, *sectionName));
        prev = i + 1;
      }
    }
  }

  // Create Mergeable atoms
  for (const MergeString *tai : tokens) {
    ArrayRef<uint8_t> content((const uint8_t *)tai->_string.data(),
                              tai->_string.size());
    ELFMergeAtom<ELFT> *atom = createMergedString(tai->_sectionName, tai->_shdr,
                                                  content, tai->_offset);
    atom->setOrdinal(++_ordinal);
    addAtom(*atom);
    _mergeAtoms.push_back(atom);
  }
  return std::error_code();
}

template <class ELFT>
std::error_code ELFFile<ELFT>::createSymbolsFromAtomizableSections() {
  // Increment over all the symbols collecting atoms and symbol names for
  // later use.
  if (!_symtab)
    return std::error_code();

  ErrorOr<StringRef> strTableOrErr =
      _objFile->getStringTableForSymtab(*_symtab);
  if (std::error_code ec = strTableOrErr.getError())
    return ec;
  StringRef strTable = *strTableOrErr;

  auto SymI = _objFile->symbol_begin(_symtab),
       SymE = _objFile->symbol_end(_symtab);
  // Skip over dummy sym.
  ++SymI;

  for (; SymI != SymE; ++SymI) {
    ErrorOr<const Elf_Shdr *> section =
        _objFile->getSection(SymI, _symtab, _shndxTable);
    if (std::error_code ec = section.getError())
      return ec;

    auto symbolName = SymI->getName(strTable);
    if (std::error_code ec = symbolName.getError())
      return ec;

    if (SymI->isAbsolute()) {
      ELFAbsoluteAtom<ELFT> *absAtom = createAbsoluteAtom(
          *symbolName, &*SymI, (int64_t)getSymbolValue(&*SymI));
      addAtom(*absAtom);
      _symbolToAtomMapping.insert(std::make_pair(&*SymI, absAtom));
    } else if (SymI->isUndefined()) {
      if (_useWrap &&
          (_wrapSymbolMap.find(*symbolName) != _wrapSymbolMap.end())) {
        auto wrapAtom = _wrapSymbolMap.find(*symbolName);
        _symbolToAtomMapping.insert(
            std::make_pair(&*SymI, wrapAtom->getValue()));
        continue;
      }
      ELFUndefinedAtom<ELFT> *undefAtom =
          createUndefinedAtom(*symbolName, &*SymI);
      addAtom(*undefAtom);
      _symbolToAtomMapping.insert(std::make_pair(&*SymI, undefAtom));
    } else if (isCommonSymbol(&*SymI)) {
      ELFCommonAtom<ELFT> *commonAtom = createCommonAtom(*symbolName, &*SymI);
      commonAtom->setOrdinal(++_ordinal);
      addAtom(*commonAtom);
      _symbolToAtomMapping.insert(std::make_pair(&*SymI, commonAtom));
    } else if (SymI->isDefined()) {
      _sectionSymbols[*section].push_back(SymI);
    } else {
      llvm::errs() << "Unable to create atom for: " << *symbolName << "\n";
      return llvm::object::object_error::parse_failed;
    }
  }

  return std::error_code();
}

template <class ELFT> std::error_code ELFFile<ELFT>::createAtoms() {
  // Holds all the atoms that are part of the section. They are the targets of
  // the kindGroupChild reference.
  llvm::StringMap<std::vector<ELFDefinedAtom<ELFT> *>> atomsForSection;

  // Contains a list of comdat sections for a group.
  for (auto &i : _sectionSymbols) {
    const Elf_Shdr *section = i.first;
    std::vector<const Elf_Sym *> &symbols = i.second;

    // Sort symbols by position.
    std::stable_sort(symbols.begin(), symbols.end(),
                     [this](const Elf_Sym *a, const Elf_Sym *b) {
                       return getSymbolValue(&*a) < getSymbolValue(&*b);
                     });

    ErrorOr<StringRef> sectionName = this->getSectionName(section);
    if (std::error_code ec = sectionName.getError())
      return ec;

    auto sectionContents = getSectionContents(section);
    if (std::error_code ec = sectionContents.getError())
      return ec;

    // SHT_GROUP sections are handled in the following loop.
    if (isGroupSection(section))
      continue;

    bool addAtoms = (!isGnuLinkOnceSection(*sectionName) &&
                     !isSectionMemberOfGroup(section));

    if (handleSectionWithNoSymbols(section, symbols)) {
      ELFDefinedAtom<ELFT> *newAtom =
          createSectionAtom(section, *sectionName, *sectionContents);
      newAtom->setOrdinal(++_ordinal);
      if (addAtoms)
        addAtom(*newAtom);
      else
        atomsForSection[*sectionName].push_back(newAtom);
      continue;
    }

    ELFDefinedAtom<ELFT> *previousAtom = nullptr;
    ELFReference<ELFT> *anonFollowedBy = nullptr;

    if (!_symtab)
      continue;
    ErrorOr<StringRef> strTableOrErr =
        _objFile->getStringTableForSymtab(*_symtab);
    if (std::error_code ec = strTableOrErr.getError())
      return ec;
    StringRef strTable = *strTableOrErr;
    for (auto si = symbols.begin(), se = symbols.end(); si != se; ++si) {
      auto symbol = *si;
      StringRef symbolName = "";
      if (symbol->getType() != llvm::ELF::STT_SECTION) {
        auto symName = symbol->getName(strTable);
        if (std::error_code ec = symName.getError())
          return ec;
        symbolName = *symName;
      }

      uint64_t contentSize = symbolContentSize(
          section, &*symbol, (si + 1 == se) ? nullptr : &**(si + 1));

      // Check to see if we need to add the FollowOn Reference
      ELFReference<ELFT> *followOn = nullptr;
      if (previousAtom) {
        // Replace the followon atom with the anonymous atom that we created,
        // so that the next symbol that we create is a followon from the
        // anonymous atom.
        if (anonFollowedBy) {
          followOn = anonFollowedBy;
        } else {
          followOn = new (_readerStorage)
              ELFReference<ELFT>(Reference::kindLayoutAfter);
          previousAtom->addReference(followOn);
        }
      }

      ArrayRef<uint8_t> symbolData((const uint8_t *)sectionContents->data() +
                                       getSymbolValue(&*symbol),
                                   contentSize);

      // If the linker finds that a section has global atoms that are in a
      // mergeable section, treat them as defined atoms as they shouldn't be
      // merged away as well as these symbols have to be part of symbol
      // resolution
      if (isMergeableStringSection(section)) {
        if (symbol->getBinding() != llvm::ELF::STB_GLOBAL)
          continue;
        ELFDefinedAtom<ELFT> *atom = createDefinedAtom(
            symbolName, *sectionName, &**si, section, symbolData,
            _references.size(), _references.size(), _references);
        atom->setOrdinal(++_ordinal);
        if (addAtoms)
          addAtom(*atom);
        else
          atomsForSection[*sectionName].push_back(atom);
        continue;
      }

      // Don't allocate content to a weak symbol, as they may be merged away.
      // Create an anonymous atom to hold the data.
      ELFDefinedAtom<ELFT> *anonAtom = nullptr;
      anonFollowedBy = nullptr;
      if (symbol->getBinding() == llvm::ELF::STB_WEAK) {
        // Create anonymous new non-weak ELF symbol that holds the symbol
        // data.
        auto sym = new (_readerStorage) Elf_Sym(*symbol);
        sym->setBinding(llvm::ELF::STB_GLOBAL);
        anonAtom = createDefinedAtomAndAssignRelocations(
            "", *sectionName, sym, section, symbolData, *sectionContents);
        symbolData = ArrayRef<uint8_t>();

        // If this is the last atom, let's not create a followon reference.
        if (anonAtom && (si + 1) != se) {
          anonFollowedBy = new (_readerStorage)
              ELFReference<ELFT>(Reference::kindLayoutAfter);
          anonAtom->addReference(anonFollowedBy);
        }
      }

      ELFDefinedAtom<ELFT> *newAtom = createDefinedAtomAndAssignRelocations(
          symbolName, *sectionName, &*symbol, section, symbolData,
          *sectionContents);
      newAtom->setOrdinal(++_ordinal);

      // If the atom was a weak symbol, let's create a followon reference to
      // the anonymous atom that we created.
      if (anonAtom)
        createEdge(newAtom, anonAtom, Reference::kindLayoutAfter);

      if (previousAtom) {
        // Set the followon atom to the weak atom that we have created, so
        // that they would alias when the file gets written.
        followOn->setTarget(anonAtom ? anonAtom : newAtom);
      }

      // The previous atom is always the atom created before unless the atom
      // is a weak atom.
      previousAtom = anonAtom ? anonAtom : newAtom;

      if (addAtoms)
        addAtom(*newAtom);
      else
        atomsForSection[*sectionName].push_back(newAtom);

      _symbolToAtomMapping.insert(std::make_pair(&*symbol, newAtom));
      if (anonAtom) {
        anonAtom->setOrdinal(++_ordinal);
        if (addAtoms)
          addAtom(*anonAtom);
        else
          atomsForSection[*sectionName].push_back(anonAtom);
      }
    }
  }

  for (auto &i : _sectionSymbols)
    if (std::error_code ec = handleSectionGroup(i.first, atomsForSection))
      return ec;
  for (auto &i : _sectionSymbols)
    if (std::error_code ec = handleGnuLinkOnceSection(i.first, atomsForSection))
      return ec;

  updateReferences();
  return std::error_code();
}

template <class ELFT>
std::error_code ELFFile<ELFT>::handleGnuLinkOnceSection(
    const Elf_Shdr *section,
    llvm::StringMap<std::vector<ELFDefinedAtom<ELFT> *>> &atomsForSection) {
  ErrorOr<StringRef> sectionName = this->getSectionName(section);
  if (std::error_code ec = sectionName.getError())
    return ec;
  if (!isGnuLinkOnceSection(*sectionName))
    return std::error_code();

  unsigned int referenceStart = _references.size();
  std::vector<ELFReference<ELFT> *> refs;
  for (auto ha : atomsForSection[*sectionName]) {
    _groupChild[ha->symbol()] = std::make_pair(*sectionName, section);
    auto *ref =
        new (_readerStorage) ELFReference<ELFT>(Reference::kindGroupChild);
    ref->setTarget(ha);
    refs.push_back(ref);
  }
  atomsForSection[*sectionName].clear();
  // Create a gnu linkonce atom.
  ELFDefinedAtom<ELFT> *atom = createDefinedAtom(
      *sectionName, *sectionName, nullptr, section, ArrayRef<uint8_t>(),
      referenceStart, _references.size(), _references);
  atom->setOrdinal(++_ordinal);
  addAtom(*atom);
  for (auto reference : refs)
    atom->addReference(reference);
  return std::error_code();
}

template <class ELFT>
std::error_code ELFFile<ELFT>::handleSectionGroup(
    const Elf_Shdr *section,
    llvm::StringMap<std::vector<ELFDefinedAtom<ELFT> *>> &atomsForSection) {
  ErrorOr<StringRef> sectionName = this->getSectionName(section);
  if (std::error_code ec = sectionName.getError())
    return ec;
  if (!isGroupSection(section))
    return std::error_code();

  auto sectionContents = getSectionContents(section);
  if (std::error_code ec = sectionContents.getError())
    return ec;

  // A section of type SHT_GROUP defines a grouping of sections. The
  // name of a symbol from one of the containing object's symbol tables
  // provides a signature for the section group. The section header of
  // the SHT_GROUP section specifies the identifying symbol entry, as
  // described: the sh_link member contains the section header index of
  // the symbol table section that contains the entry. The sh_info
  // member contains the symbol table index of the identifying entry.
  // The sh_flags member of the section header contains 0. The name of
  // the section (sh_name) is not specified.
  std::vector<StringRef> sectionNames;
  const Elf_Word *groupMembers =
      reinterpret_cast<const Elf_Word *>(sectionContents->data());
  const size_t count = section->sh_size / sizeof(Elf_Word);
  for (size_t i = 1; i < count; i++) {
    ErrorOr<const Elf_Shdr *> shdr = _objFile->getSection(groupMembers[i]);
    if (std::error_code ec = shdr.getError())
      return ec;
    ErrorOr<StringRef> sectionName = _objFile->getSectionName(*shdr);
    if (std::error_code ec = sectionName.getError())
      return ec;
    sectionNames.push_back(*sectionName);
  }
  ErrorOr<const Elf_Shdr *> symtab = _objFile->getSection(section->sh_link);
  if (std::error_code ec = symtab.getError())
    return ec;
  const Elf_Sym *symbol = _objFile->getSymbol(*symtab, section->sh_info);
  ErrorOr<const Elf_Shdr *> strtab_sec =
      _objFile->getSection((*symtab)->sh_link);
  if (std::error_code ec = strtab_sec.getError())
    return ec;
  ErrorOr<StringRef> strtab_or_err = _objFile->getStringTable(*strtab_sec);
  if (std::error_code ec = strtab_or_err.getError())
    return ec;
  StringRef strtab = *strtab_or_err;
  ErrorOr<StringRef> symbolName = symbol->getName(strtab);
  if (std::error_code ec = symbolName.getError())
    return ec;

  unsigned int referenceStart = _references.size();
  std::vector<ELFReference<ELFT> *> refs;
  for (auto name : sectionNames) {
    for (auto ha : atomsForSection[name]) {
      _groupChild[ha->symbol()] = std::make_pair(*symbolName, section);
      auto *ref =
          new (_readerStorage) ELFReference<ELFT>(Reference::kindGroupChild);
      ref->setTarget(ha);
      refs.push_back(ref);
    }
    atomsForSection[name].clear();
  }

  // Create an atom for comdat signature.
  ELFDefinedAtom<ELFT> *atom = createDefinedAtom(
      *symbolName, *sectionName, nullptr, section, ArrayRef<uint8_t>(),
      referenceStart, _references.size(), _references);
  atom->setOrdinal(++_ordinal);
  addAtom(*atom);
  for (auto reference : refs)
    atom->addReference(reference);
  return std::error_code();
}

template <class ELFT> std::error_code ELFFile<ELFT>::createAtomsFromContext() {
  if (!_useWrap)
    return std::error_code();
  // Steps:
  // a) Create an undefined atom for the symbol specified by the --wrap option,
  //    as that may be needed to be pulled from an archive.
  // b) Create an undefined atom for __wrap_<symbolname>.
  // c) All references to the symbol specified by wrap should point to
  //    __wrap_<symbolname>
  // d) All references to __real_symbol should point to the <symbol>
  for (auto &wrapsym : _ctx.wrapCalls()) {
    StringRef wrapStr = wrapsym.getKey();
    // Create a undefined symbol fror the wrap symbol.
    UndefinedAtom *wrapSymAtom =
        new (_readerStorage) SimpleUndefinedAtom(*this, wrapStr);
    StringRef wrapCallSym =
        _ctx.allocateString((llvm::Twine("__wrap_") + wrapStr).str());
    StringRef realCallSym =
        _ctx.allocateString((llvm::Twine("__real_") + wrapStr).str());
    UndefinedAtom *wrapCallAtom =
        new (_readerStorage) SimpleUndefinedAtom(*this, wrapCallSym);
    // Create maps, when there is call to sym, it should point to wrapCallSym.
    _wrapSymbolMap.insert(std::make_pair(wrapStr, wrapCallAtom));
    // Whenever there is a reference to realCall it should point to the symbol
    // created for each wrap usage.
    _wrapSymbolMap.insert(std::make_pair(realCallSym, wrapSymAtom));
    addAtom(*wrapSymAtom);
    addAtom(*wrapCallAtom);
  }
  return std::error_code();
}

template <class ELFT>
ELFDefinedAtom<ELFT> *ELFFile<ELFT>::createDefinedAtomAndAssignRelocations(
    StringRef symbolName, StringRef sectionName, const Elf_Sym *symbol,
    const Elf_Shdr *section, ArrayRef<uint8_t> symContent,
    ArrayRef<uint8_t> secContent) {
  unsigned int referenceStart = _references.size();

  // Add Rela (those with r_addend) references:
  auto rari = _relocationAddendReferences.find(section);
  if (rari != _relocationAddendReferences.end())
    createRelocationReferences(symbol, symContent, rari->second);

  // Add Rel references.
  auto rri = _relocationReferences.find(section);
  if (rri != _relocationReferences.end())
    createRelocationReferences(symbol, symContent, secContent, rri->second);

  // Create the DefinedAtom and add it to the list of DefinedAtoms.
  return createDefinedAtom(symbolName, sectionName, symbol, section, symContent,
                           referenceStart, _references.size(), _references);
}

template <class ELFT>
void ELFFile<ELFT>::createRelocationReferences(const Elf_Sym *symbol,
                                               ArrayRef<uint8_t> content,
                                               range<const Elf_Rela *> rels) {
  bool isMips64EL = _objFile->isMips64EL();
  const auto symValue = getSymbolValue(symbol);
  for (const auto &rel : rels) {
    if (rel.r_offset < symValue || symValue + content.size() <= rel.r_offset)
      continue;
    auto elfRelocation = new (_readerStorage)
        ELFReference<ELFT>(&rel, rel.r_offset - symValue, kindArch(),
                           rel.getType(isMips64EL), rel.getSymbol(isMips64EL));
    addReferenceToSymbol(elfRelocation, symbol);
    _references.push_back(elfRelocation);
  }
}

template <class ELFT>
void ELFFile<ELFT>::createRelocationReferences(const Elf_Sym *symbol,
                                               ArrayRef<uint8_t> symContent,
                                               ArrayRef<uint8_t> secContent,
                                               const Elf_Shdr *relSec) {
  auto rels = _objFile->rels(relSec);
  bool isMips64EL = _objFile->isMips64EL();
  const auto symValue = getSymbolValue(symbol);
  for (const auto &rel : rels) {
    if (rel.r_offset < symValue || symValue + symContent.size() <= rel.r_offset)
      continue;
    auto elfRelocation = new (_readerStorage)
        ELFReference<ELFT>(rel.r_offset - symValue, kindArch(),
                           rel.getType(isMips64EL), rel.getSymbol(isMips64EL));
    Reference::Addend addend = getInitialAddend(symContent, symValue, rel);
    elfRelocation->setAddend(addend);
    addReferenceToSymbol(elfRelocation, symbol);
    _references.push_back(elfRelocation);
  }
}

template <class ELFT>
void ELFFile<ELFT>::updateReferenceForMergeStringAccess(ELFReference<ELFT> *ref,
                                                        const Elf_Sym *symbol,
                                                        const Elf_Shdr *shdr) {
  // If the target atom is mergeable strefng atom, the atom might have been
  // merged with other atom having the same contents. Try to find the
  // merged one if that's the case.
  int64_t addend = ref->addend();
  if (addend < 0)
    addend = 0;

  const MergeSectionKey ms = {shdr, addend};
  auto msec = _mergedSectionMap.find(ms);
  if (msec != _mergedSectionMap.end()) {
    ref->setTarget(msec->second);
    return;
  }

  // The target atom was not merged. Mergeable atoms are not in
  // _symbolToAtomMapping, so we cannot find it by calling findAtom(). We
  // instead call findMergeAtom().
  if (symbol->getType() != llvm::ELF::STT_SECTION)
    addend = getSymbolValue(symbol) + addend;
  ELFMergeAtom<ELFT> *mergedAtom = findMergeAtom(shdr, addend);
  ref->setOffset(addend - mergedAtom->offset());
  ref->setAddend(0);
  ref->setTarget(mergedAtom);
}

template <class ELFT> void ELFFile<ELFT>::updateReferences() {
  for (auto &ri : _references) {
    if (ri->kindNamespace() != Reference::KindNamespace::ELF)
      continue;
    const Elf_Sym *symbol =
        _objFile->getSymbol(_symtab, ri->targetSymbolIndex());
    ErrorOr<const Elf_Shdr *> shdr =
        _objFile->getSection(symbol, _symtab, _shndxTable);

    // If the atom is not in mergeable string section, the target atom is
    // simply that atom.
    if (isMergeableStringSection(*shdr))
      updateReferenceForMergeStringAccess(ri, symbol, *shdr);
    else
      ri->setTarget(findAtom(findSymbolForReference(ri), symbol));
  }
}

template <class ELFT>
bool ELFFile<ELFT>::isIgnoredSection(const Elf_Shdr *section) {
  switch (section->sh_type) {
  case llvm::ELF::SHT_NULL:
  case llvm::ELF::SHT_STRTAB:
  case llvm::ELF::SHT_SYMTAB:
  case llvm::ELF::SHT_SYMTAB_SHNDX:
    return true;
  default:
    break;
  }
  return false;
}

template <class ELFT>
bool ELFFile<ELFT>::isMergeableStringSection(const Elf_Shdr *section) {
  if (_doStringsMerge && section) {
    int64_t sectionFlags = section->sh_flags;
    sectionFlags &= ~llvm::ELF::SHF_ALLOC;
    // Mergeable string sections have both SHF_MERGE and SHF_STRINGS flags
    // set. sh_entsize is the size of each character which is normally 1.
    if ((section->sh_entsize < 2) &&
        (sectionFlags == (llvm::ELF::SHF_MERGE | llvm::ELF::SHF_STRINGS))) {
      return true;
    }
  }
  return false;
}

template <class ELFT>
ELFDefinedAtom<ELFT> *
ELFFile<ELFT>::createSectionAtom(const Elf_Shdr *section, StringRef sectionName,
                                 ArrayRef<uint8_t> content) {
  auto *sym = new (_readerStorage) Elf_Sym;
  sym->st_name = 0;
  sym->setBindingAndType(llvm::ELF::STB_LOCAL, llvm::ELF::STT_SECTION);
  sym->st_other = 0;
  sym->st_shndx = 0;
  sym->st_value = 0;
  sym->st_size = 0;
  auto *newAtom = createDefinedAtomAndAssignRelocations(
      "", sectionName, sym, section, content, content);
  newAtom->setOrdinal(++_ordinal);
  return newAtom;
}

template <class ELFT>
uint64_t ELFFile<ELFT>::symbolContentSize(const Elf_Shdr *section,
                                          const Elf_Sym *symbol,
                                          const Elf_Sym *nextSymbol) {
  const auto symValue = getSymbolValue(symbol);
  // if this is the last symbol, take up the remaining data.
  return nextSymbol ? getSymbolValue(nextSymbol) - symValue
                    : section->sh_size - symValue;
}

template <class ELFT>
void ELFFile<ELFT>::createEdge(ELFDefinedAtom<ELFT> *from,
                               ELFDefinedAtom<ELFT> *to, uint32_t edgeKind) {
  auto reference = new (_readerStorage) ELFReference<ELFT>(edgeKind);
  reference->setTarget(to);
  from->addReference(reference);
}

/// Does the atom need to be redirected using a separate undefined atom?
template <class ELFT>
bool ELFFile<ELFT>::redirectReferenceUsingUndefAtom(
    const Elf_Sym *sourceSymbol, const Elf_Sym *targetSymbol) const {
  auto groupChildTarget = _groupChild.find(targetSymbol);

  // If the reference is not to a group child atom, there is no need to redirect
  // using a undefined atom. Its also not needed if the source and target are
  // from the same section.
  if ((groupChildTarget == _groupChild.end()) ||
      (sourceSymbol->st_shndx == targetSymbol->st_shndx))
    return false;

  auto groupChildSource = _groupChild.find(sourceSymbol);

  // If the source symbol is not in a group, use a undefined symbol too.
  if (groupChildSource == _groupChild.end())
    return true;

  // If the source and child are from the same group, we dont need the
  // relocation to go through a undefined symbol.
  if (groupChildSource->second.second == groupChildTarget->second.second)
    return false;
  return true;
}

template <class ELFT>
void RuntimeFile<ELFT>::addAbsoluteAtom(StringRef symbolName, bool isHidden) {
  assert(!symbolName.empty() && "AbsoluteAtoms must have a name");
  auto *sym = new (this->_readerStorage) Elf_Sym;
  sym->st_name = 0;
  sym->st_value = 0;
  sym->st_shndx = llvm::ELF::SHN_ABS;
  sym->setBindingAndType(llvm::ELF::STB_GLOBAL, llvm::ELF::STT_OBJECT);
  if (isHidden)
    sym->setVisibility(llvm::ELF::STV_HIDDEN);
  else
    sym->setVisibility(llvm::ELF::STV_DEFAULT);
  sym->st_size = 0;
  ELFAbsoluteAtom<ELFT> *atom = this->createAbsoluteAtom(symbolName, sym, -1);
  this->addAtom(*atom);
}

template <class ELFT>
void RuntimeFile<ELFT>::addUndefinedAtom(StringRef symbolName) {
  assert(!symbolName.empty() && "UndefinedAtoms must have a name");
  auto *sym = new (this->_readerStorage) Elf_Sym;
  sym->st_name = 0;
  sym->st_value = 0;
  sym->st_shndx = llvm::ELF::SHN_UNDEF;
  sym->setBindingAndType(llvm::ELF::STB_GLOBAL, llvm::ELF::STT_NOTYPE);
  sym->setVisibility(llvm::ELF::STV_DEFAULT);
  sym->st_size = 0;
  ELFUndefinedAtom<ELFT> *atom = this->createUndefinedAtom(symbolName, sym);
  this->addAtom(*atom);
}

template class ELFFile<ELF32LE>;
template class ELFFile<ELF32BE>;
template class ELFFile<ELF64LE>;
template class ELFFile<ELF64BE>;

template class RuntimeFile<ELF32LE>;
template class RuntimeFile<ELF32BE>;
template class RuntimeFile<ELF64LE>;
template class RuntimeFile<ELF64BE>;

} // end namespace elf
} // end namespace lld
