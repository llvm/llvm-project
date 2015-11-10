//===- lib/ReaderWriter/ELF/ELFFile.h ---------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_FILE_H
#define LLD_READER_WRITER_ELF_FILE_H

#include "Atoms.h"
#include "FileCommon.h"
#include "llvm/ADT/MapVector.h"
#include <map>
#include <unordered_map>

namespace lld {

namespace elf {
/// \brief Read a binary, find out based on the symbol table contents what kind
/// of symbol it is and create corresponding atoms for it
template <class ELFT> class ELFFile : public SimpleFile {
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef llvm::object::Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef llvm::object::Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef llvm::object::Elf_Rel_Impl<ELFT, true> Elf_Rela;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Word Elf_Word;

  // A Map is used to hold the atoms that have been divided up
  // after reading the section that contains Merge String attributes
  struct MergeSectionKey {
    const Elf_Shdr *_shdr;
    int64_t _offset;
  };

  struct MergeSectionEq {
    int64_t operator()(const MergeSectionKey &k) const {
      return llvm::hash_combine((int64_t)(k._shdr->sh_name),
                                (int64_t)k._offset);
    }
    bool operator()(const MergeSectionKey &lhs,
                    const MergeSectionKey &rhs) const {
      return ((lhs._shdr->sh_name == rhs._shdr->sh_name) &&
              (lhs._offset == rhs._offset));
    }
  };

  struct MergeString {
    MergeString(int64_t offset, StringRef str, const Elf_Shdr *shdr,
                StringRef sectionName)
        : _offset(offset), _string(str), _shdr(shdr),
          _sectionName(sectionName) {}
    // the offset of this atom
    int64_t _offset;
    // The content
    StringRef _string;
    // Section header
    const Elf_Shdr *_shdr;
    // Section name
    StringRef _sectionName;
  };

  // This is used to find the MergeAtom given a relocation
  // offset
  typedef std::vector<ELFMergeAtom<ELFT> *> MergeAtomsT;

  /// \brief find a merge atom given a offset
  ELFMergeAtom<ELFT> *findMergeAtom(const Elf_Shdr *shdr, int64_t offset) {
    auto it = std::find_if(_mergeAtoms.begin(), _mergeAtoms.end(),
                           [=](const ELFMergeAtom<ELFT> *a) {
                             int64_t off = a->offset();
                             return shdr->sh_name == a->section() &&
                                    offset >= off &&
                                    offset <= off + (int64_t)a->size();
                           });
    assert(it != _mergeAtoms.end());
    return *it;
  }

  typedef std::unordered_map<MergeSectionKey, DefinedAtom *, MergeSectionEq,
                             MergeSectionEq> MergedSectionMapT;
  typedef typename MergedSectionMapT::iterator MergedSectionMapIterT;

public:
  ELFFile(StringRef name, ELFLinkingContext &ctx);
  ELFFile(std::unique_ptr<MemoryBuffer> mb, ELFLinkingContext &ctx);

  static std::error_code isCompatible(MemoryBufferRef mb,
                                      ELFLinkingContext &ctx);

  static bool canParse(file_magic magic) {
    return magic == file_magic::elf_relocatable;
  }

  virtual Reference::KindArch kindArch();

  /// \brief Create symbols from LinkingContext.
  std::error_code createAtomsFromContext();

  /// \brief Read input sections and populate necessary data structures
  /// to read them later and create atoms
  std::error_code createAtomizableSections();

  /// \brief Create mergeable atoms from sections that have the merge attribute
  /// set
  std::error_code createMergeableAtoms();

  /// \brief Add the symbols that the sections contain. The symbols will be
  /// converted to atoms for
  /// Undefined symbols, absolute symbols
  std::error_code createSymbolsFromAtomizableSections();

  /// \brief Create individual atoms
  std::error_code createAtoms();

  // Assuming sourceSymbol has a reference to targetSym, find an atom
  // for targetSym. Usually it's just the atom for targetSym.
  // However, if an atom is in a section group, we may want to return an
  // undefined atom for targetSym to let the resolver to resolve the
  // symbol. (It's because if targetSym is in a section group A, and the
  // group A is not linked in because other file already provides a
  // section group B, we want to resolve references to B, not to A.)
  Atom *findAtom(const Elf_Sym *sourceSym, const Elf_Sym *targetSym);

protected:
  ELFDefinedAtom<ELFT> *createDefinedAtomAndAssignRelocations(
      StringRef symbolName, StringRef sectionName, const Elf_Sym *symbol,
      const Elf_Shdr *section, ArrayRef<uint8_t> symContent,
      ArrayRef<uint8_t> secContent);

  std::error_code doParse() override;

  /// \brief Iterate over Elf_Rela relocations list and create references.
  virtual void createRelocationReferences(const Elf_Sym *symbol,
                                          ArrayRef<uint8_t> content,
                                          range<const Elf_Rela *> rels);

  /// \brief Iterate over Elf_Rel relocations list and create references.
  virtual void createRelocationReferences(const Elf_Sym *symbol,
                                          ArrayRef<uint8_t> symContent,
                                          ArrayRef<uint8_t> secContent,
                                          const Elf_Shdr *relSec);

  /// \brief After all the Atoms and References are created, update each
  /// Reference's target with the Atom pointer it refers to.
  void updateReferences();

  /// \brief Update the reference if the access corresponds to a merge string
  /// section.
  void updateReferenceForMergeStringAccess(ELFReference<ELFT> *ref,
                                           const Elf_Sym *symbol,
                                           const Elf_Shdr *shdr);

  /// \brief Do we want to ignore the section. Ignored sections are
  /// not processed to create atoms
  bool isIgnoredSection(const Elf_Shdr *section);

  /// \brief Is the current section be treated as a mergeable string section.
  /// The contents of a mergeable string section are null-terminated strings.
  /// If the section have mergeable strings, the linker would need to split
  /// the section into multiple atoms and mark them mergeByContent.
  bool isMergeableStringSection(const Elf_Shdr *section);

  /// \brief Returns a new anonymous atom whose size is equal to the
  /// section size. That atom will be used to represent the entire
  /// section that have no symbols.
  ELFDefinedAtom<ELFT> *createSectionAtom(const Elf_Shdr *section,
                                          StringRef sectionName,
                                          ArrayRef<uint8_t> contents);

  /// Returns the symbol's content size. The nextSymbol should be null if the
  /// symbol is the last one in the section.
  uint64_t symbolContentSize(const Elf_Shdr *section,
                             const Elf_Sym *symbol,
                             const Elf_Sym *nextSymbol);

  void createEdge(ELFDefinedAtom<ELFT> *from, ELFDefinedAtom<ELFT> *to,
                  uint32_t edgeKind);

  /// Get the section name for a section.
  ErrorOr<StringRef> getSectionName(const Elf_Shdr *shdr) const;

  /// Determines if the section occupy memory space.
  bool sectionOccupiesMemorySpace(const Elf_Shdr *shdr) const {
    return (shdr->sh_type != llvm::ELF::SHT_NOBITS);
  }

  /// Return the section contents.
  ErrorOr<ArrayRef<uint8_t>> getSectionContents(const Elf_Shdr *shdr) const {
    if (!shdr || !sectionOccupiesMemorySpace(shdr))
      return ArrayRef<uint8_t>();
    return _objFile->getSectionContents(shdr);
  }

  /// Determines if the target wants to create an atom for a section that has no
  /// symbol references.
  bool
  handleSectionWithNoSymbols(const Elf_Shdr *shdr,
                             std::vector<const Elf_Sym *> &syms) const {
    return shdr &&
           (shdr->sh_type == llvm::ELF::SHT_PROGBITS ||
            shdr->sh_type == llvm::ELF::SHT_INIT_ARRAY ||
            shdr->sh_type == llvm::ELF::SHT_FINI_ARRAY ||
            shdr->sh_type == llvm::ELF::SHT_NOTE) &&
           syms.empty();
  }

  /// Handle creation of atoms for .gnu.linkonce sections.
  std::error_code handleGnuLinkOnceSection(
      const Elf_Shdr *section,
      llvm::StringMap<std::vector<ELFDefinedAtom<ELFT> *>> &atomsForSection);

  // Handle COMDAT scetions.
  std::error_code handleSectionGroup(
      const Elf_Shdr *section,
      llvm::StringMap<std::vector<ELFDefinedAtom<ELFT> *>> &atomsForSection);

  /// Process the Undefined symbol and create an atom for it.
  ELFUndefinedAtom<ELFT> *createUndefinedAtom(StringRef symName,
                                              const Elf_Sym *sym) {
    return new (_readerStorage) ELFUndefinedAtom<ELFT>(*this, symName, sym);
  }

  /// Process the Absolute symbol and create an atom for it.
  ELFAbsoluteAtom<ELFT> *createAbsoluteAtom(StringRef symName,
                                            const Elf_Sym *sym, int64_t value) {
    return new (_readerStorage)
        ELFAbsoluteAtom<ELFT>(*this, symName, sym, value);
  }

  /// Returns true if the symbol is common symbol. A common symbol represents a
  /// tentive definition in C. It has name, size and alignment constraint, but
  /// actual storage has not yet been allocated. (The linker will allocate
  /// storage for them in the later pass after coalescing tentative symbols by
  /// name.)
  virtual bool isCommonSymbol(const Elf_Sym *symbol) const {
    return symbol->getType() == llvm::ELF::STT_COMMON ||
           symbol->st_shndx == llvm::ELF::SHN_COMMON;
  }

  /// Returns true if the section is a gnulinkonce section.
  bool isGnuLinkOnceSection(StringRef sectionName) const {
    return sectionName.startswith(".gnu.linkonce.");
  }

  /// Returns true if the section is a COMDAT group section.
  bool isGroupSection(const Elf_Shdr *shdr) const {
    return (shdr->sh_type == llvm::ELF::SHT_GROUP);
  }

  /// Returns true if the section is a member of some group.
  bool isSectionMemberOfGroup(const Elf_Shdr *shdr) const {
    return (shdr->sh_flags & llvm::ELF::SHF_GROUP);
  }

  /// Returns correct st_value for the symbol depending on the architecture.
  /// For most architectures it's just a regular st_value with no changes.
  virtual uint64_t getSymbolValue(const Elf_Sym *symbol) const {
    return symbol->st_value;
  }

  /// Returns initial addend
  virtual Reference::Addend getInitialAddend(ArrayRef<uint8_t> symContent,
                                  uint64_t symbolValue,
                                  const Elf_Rel& reference) const {
    return *(symContent.data() + reference.r_offset - symbolValue);
  }

  /// Process the common symbol and create an atom for it.
  virtual ELFCommonAtom<ELFT> *createCommonAtom(StringRef symName,
                                                const Elf_Sym *sym) {
    return new (_readerStorage) ELFCommonAtom<ELFT>(*this, symName, sym);
  }

  /// Creates an atom for a given defined symbol.
  virtual ELFDefinedAtom<ELFT> *
  createDefinedAtom(StringRef symName, StringRef sectionName,
                    const Elf_Sym *sym, const Elf_Shdr *sectionHdr,
                    ArrayRef<uint8_t> contentData, unsigned int referenceStart,
                    unsigned int referenceEnd,
                    std::vector<ELFReference<ELFT> *> &referenceList) {
    return new (_readerStorage) ELFDefinedAtom<ELFT>(
        *this, symName, sectionName, sym, sectionHdr, contentData,
        referenceStart, referenceEnd, referenceList);
  }

  /// Process the Merge string and create an atom for it.
  ELFMergeAtom<ELFT> *createMergedString(StringRef sectionName,
                                         const Elf_Shdr *sectionHdr,
                                         ArrayRef<uint8_t> contentData,
                                         unsigned int offset) {
    auto *mergeAtom = new (_readerStorage)
        ELFMergeAtom<ELFT>(*this, sectionName, sectionHdr, contentData, offset);
    const MergeSectionKey mergedSectionKey = {sectionHdr, offset};
    if (_mergedSectionMap.find(mergedSectionKey) == _mergedSectionMap.end())
      _mergedSectionMap.insert(std::make_pair(mergedSectionKey, mergeAtom));
    return mergeAtom;
  }

  /// References to the sections comprising a group, from sections
  /// outside the group, must be made via global UNDEF symbols,
  /// referencing global symbols defined as addresses in the group
  /// sections. They may not reference local symbols for addresses in
  /// the group's sections, including section symbols.
  /// ABI Doc : https://mentorembedded.github.io/cxx-abi/abi/prop-72-comdat.html
  /// Does the atom need to be redirected using a separate undefined atom?
  bool redirectReferenceUsingUndefAtom(const Elf_Sym *sourceSymbol,
                                       const Elf_Sym *targetSymbol) const;

  void addReferenceToSymbol(const ELFReference<ELFT> *r, const Elf_Sym *sym) {
    _referenceToSymbol[r] = sym;
  }

  const Elf_Sym *findSymbolForReference(const ELFReference<ELFT> *r) const {
    auto elfReferenceToSymbol = _referenceToSymbol.find(r);
    if (elfReferenceToSymbol != _referenceToSymbol.end())
      return elfReferenceToSymbol->second;
    return nullptr;
  }

  llvm::BumpPtrAllocator _readerStorage;
  std::unique_ptr<llvm::object::ELFFile<ELFT> > _objFile;
  const Elf_Shdr *_symtab = nullptr;
  ArrayRef<Elf_Word> _shndxTable;

  /// \brief _relocationAddendReferences and _relocationReferences contain the
  /// list of relocations references.  In ELF, if a section named, ".text" has
  /// relocations will also have a section named ".rel.text" or ".rela.text"
  /// which will hold the entries.
  std::unordered_map<const Elf_Shdr *, range<const Elf_Rela *>>
      _relocationAddendReferences;
  MergedSectionMapT _mergedSectionMap;
  std::unordered_map<const Elf_Shdr *, const Elf_Shdr *> _relocationReferences;
  std::vector<ELFReference<ELFT> *> _references;
  llvm::DenseMap<const Elf_Sym *, Atom *> _symbolToAtomMapping;
  llvm::DenseMap<const ELFReference<ELFT> *, const Elf_Sym *>
  _referenceToSymbol;
  // Group child atoms have a pair corresponding to the signature and the
  // section header of the section that was used for generating the signature.
  llvm::DenseMap<const Elf_Sym *, std::pair<StringRef, const Elf_Shdr *>>
      _groupChild;
  llvm::StringMap<Atom *> _undefAtomsForGroupChild;

  /// \brief Atoms that are created for a section that has the merge property
  /// set
  MergeAtomsT _mergeAtoms;

  /// \brief the section and the symbols that are contained within it to create
  /// used to create atoms
  llvm::MapVector<const Elf_Shdr *, std::vector<const Elf_Sym *>>
      _sectionSymbols;

  /// \brief Sections that have merge string property
  std::vector<const Elf_Shdr *> _mergeStringSections;

  std::unique_ptr<MemoryBuffer> _mb;
  int64_t _ordinal;

  /// \brief the cached options relevant while reading the ELF File
  bool _doStringsMerge;

  /// \brief Is --wrap on?
  bool _useWrap;

  /// \brief The LinkingContext.
  ELFLinkingContext &_ctx;

  // Wrap map
  llvm::StringMap<UndefinedAtom *> _wrapSymbolMap;
};

/// \brief All atoms are owned by a File. To add linker specific atoms
/// the atoms need to be inserted to a file called (RuntimeFile) which
/// are basically additional symbols required by libc and other runtime
/// libraries part of executing a program. This class provides support
/// for adding absolute symbols and undefined symbols
template <class ELFT> class RuntimeFile : public ELFFile<ELFT> {
public:
  typedef llvm::object::Elf_Sym_Impl<ELFT> Elf_Sym;
  RuntimeFile(ELFLinkingContext &ctx, StringRef name)
      : ELFFile<ELFT>(name, ctx) {}

  /// \brief add a global absolute atom
  virtual void addAbsoluteAtom(StringRef symbolName, bool isHidden = false);

  /// \brief add an undefined atom
  virtual void addUndefinedAtom(StringRef symbolName);
};

} // end namespace elf
} // end namespace lld

#endif // LLD_READER_WRITER_ELF_FILE_H
