//===- lld/Core/Simple.h - Simple implementations of Atom and File --------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provide simple implementations for Atoms and File.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_SIMPLE_H
#define LLD_CORE_SIMPLE_H

#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Reference.h"
#include "lld/Core/UndefinedAtom.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include <atomic>

namespace lld {

class SimpleFile : public File {
public:
  SimpleFile(StringRef path, File::Kind kind)
    : File(path, kind) {}

  void addAtom(const DefinedAtom &a) { _defined.push_back(&a); }
  void addAtom(const UndefinedAtom &a) { _undefined.push_back(&a); }
  void addAtom(const SharedLibraryAtom &a) { _shared.push_back(&a); }
  void addAtom(const AbsoluteAtom &a) { _absolute.push_back(&a); }

  void addAtom(const Atom &atom) {
    if (auto *p = dyn_cast<DefinedAtom>(&atom)) {
      _defined.push_back(p);
    } else if (auto *p = dyn_cast<UndefinedAtom>(&atom)) {
      _undefined.push_back(p);
    } else if (auto *p = dyn_cast<SharedLibraryAtom>(&atom)) {
      _shared.push_back(p);
    } else if (auto *p = dyn_cast<AbsoluteAtom>(&atom)) {
      _absolute.push_back(p);
    } else {
      llvm_unreachable("atom has unknown definition kind");
    }
  }

  void removeDefinedAtomsIf(std::function<bool(const DefinedAtom *)> pred) {
    auto &atoms = _defined;
    auto newEnd = std::remove_if(atoms.begin(), atoms.end(), pred);
    atoms.erase(newEnd, atoms.end());
  }

  const AtomVector<DefinedAtom> &defined() const override { return _defined; }

  const AtomVector<UndefinedAtom> &undefined() const override {
    return _undefined;
  }

  const AtomVector<SharedLibraryAtom> &sharedLibrary() const override {
    return _shared;
  }

  const AtomVector<AbsoluteAtom> &absolute() const override {
    return _absolute;
  }

  typedef range<std::vector<const DefinedAtom *>::iterator> DefinedAtomRange;
  DefinedAtomRange definedAtoms() { return make_range(_defined); }

private:
  AtomVector<DefinedAtom> _defined;
  AtomVector<UndefinedAtom> _undefined;
  AtomVector<SharedLibraryAtom> _shared;
  AtomVector<AbsoluteAtom> _absolute;
};

/// \brief Archive library file that may be used as a virtual container
/// for symbols that should be added dynamically in response to
/// call to find() method.
class SimpleArchiveLibraryFile : public ArchiveLibraryFile {
public:
  SimpleArchiveLibraryFile(StringRef filename)
      : ArchiveLibraryFile(filename) {}

  const AtomVector<DefinedAtom> &defined() const override {
    return _definedAtoms;
  }

  const AtomVector<UndefinedAtom> &undefined() const override {
    return _undefinedAtoms;
  }

  const AtomVector<SharedLibraryAtom> &sharedLibrary() const override {
    return _sharedLibraryAtoms;
  }

  const AtomVector<AbsoluteAtom> &absolute() const override {
    return _absoluteAtoms;
  }

  File *find(StringRef sym, bool dataSymbolOnly) override {
    // For descendants:
    // do some checks here and return dynamically generated files with atoms.
    return nullptr;
  }

  std::error_code
  parseAllMembers(std::vector<std::unique_ptr<File>> &result) override {
    return std::error_code();
  }

private:
  AtomVector<DefinedAtom> _definedAtoms;
  AtomVector<UndefinedAtom> _undefinedAtoms;
  AtomVector<SharedLibraryAtom> _sharedLibraryAtoms;
  AtomVector<AbsoluteAtom> _absoluteAtoms;
};

class SimpleReference : public Reference {
public:
  SimpleReference(Reference::KindNamespace ns, Reference::KindArch arch,
                  Reference::KindValue value, uint64_t off, const Atom *t,
                  Reference::Addend a)
      : Reference(ns, arch, value), _target(t), _offsetInAtom(off), _addend(a),
        _next(nullptr), _prev(nullptr) {
  }
  SimpleReference()
      : Reference(Reference::KindNamespace::all, Reference::KindArch::all, 0),
        _target(nullptr), _offsetInAtom(0), _addend(0), _next(nullptr),
        _prev(nullptr) {
  }

  uint64_t offsetInAtom() const override { return _offsetInAtom; }

  const Atom *target() const override {
    assert(_target);
    return _target;
  }

  Addend addend() const override { return _addend; }
  void setAddend(Addend a) override { _addend = a; }
  void setTarget(const Atom *newAtom) override { _target = newAtom; }
  SimpleReference *getNext() const { return _next; }
  SimpleReference *getPrev() const { return _prev; }
  void setNext(SimpleReference *n) { _next = n; }
  void setPrev(SimpleReference *p) { _prev = p; }

private:
  const Atom *_target;
  uint64_t _offsetInAtom;
  Addend _addend;
  SimpleReference *_next;
  SimpleReference *_prev;
};

}

// ilist will lazily create a sentinal (so end() can return a node past the
// end of the list). We need this trait so that the sentinal is allocated
// via the BumpPtrAllocator.
namespace llvm {
template<>
struct ilist_sentinel_traits<lld::SimpleReference> {

  ilist_sentinel_traits() : _allocator(nullptr) { }

  void setAllocator(llvm::BumpPtrAllocator *alloc) {
    _allocator = alloc;
  }

  lld::SimpleReference *createSentinel() const {
    return new (*_allocator) lld::SimpleReference();
  }

  static void destroySentinel(lld::SimpleReference*) {}

  static lld::SimpleReference *provideInitialHead() { return nullptr; }

  lld::SimpleReference *ensureHead(lld::SimpleReference *&head) const {
    if (!head) {
      head = createSentinel();
      noteHead(head, head);
      ilist_traits<lld::SimpleReference>::setNext(head, nullptr);
      return head;
    }
    return ilist_traits<lld::SimpleReference>::getPrev(head);
  }

  void noteHead(lld::SimpleReference *newHead,
                lld::SimpleReference *sentinel) const {
    ilist_traits<lld::SimpleReference>::setPrev(newHead, sentinel);
  }

private:
  mutable llvm::BumpPtrAllocator *_allocator;
};
}

namespace lld {

class SimpleDefinedAtom : public DefinedAtom {
public:
  explicit SimpleDefinedAtom(const File &f)
    : _file(f), _ordinal(f.getNextAtomOrdinalAndIncrement()) {
    _references.setAllocator(&f.allocator());
  }

  const File &file() const override { return _file; }

  StringRef name() const override { return StringRef(); }

  uint64_t ordinal() const override { return _ordinal; }

  Scope scope() const override { return DefinedAtom::scopeLinkageUnit; }

  Interposable interposable() const override {
    return DefinedAtom::interposeNo;
  }

  Merge merge() const override { return DefinedAtom::mergeNo; }

  Alignment alignment() const override { return 1; }

  SectionChoice sectionChoice() const override {
    return DefinedAtom::sectionBasedOnContent;
  }

  StringRef customSectionName() const override { return StringRef(); }
  DeadStripKind deadStrip() const override {
    return DefinedAtom::deadStripNormal;
  }

  DefinedAtom::reference_iterator begin() const override {
    const void *it = reinterpret_cast<const void *>(&*_references.begin());
    return reference_iterator(*this, it);
  }

  DefinedAtom::reference_iterator end() const override {
    const void *it = reinterpret_cast<const void *>(&*_references.end());
    return reference_iterator(*this, it);
  }

  const Reference *derefIterator(const void *it) const override {
    return reinterpret_cast<const Reference*>(it);
  }

  void incrementIterator(const void *&it) const override {
    const SimpleReference* node = reinterpret_cast<const SimpleReference*>(it);
    const SimpleReference* next = node->getNext();
    it = reinterpret_cast<const void*>(next);
  }

  void addReference(Reference::KindNamespace ns,
                    Reference::KindArch arch,
                    Reference::KindValue kindValue, uint64_t off,
                    const Atom *target, Reference::Addend a) override {
    assert(target && "trying to create reference to nothing");
    auto node = new (_file.allocator())
        SimpleReference(ns, arch, kindValue, off, target, a);
    _references.push_back(node);
  }

  /// Sort references in a canonical order (by offset, then by kind).
  void sortReferences() const {
    // Cannot sort a linked  list, so move elements into a temporary vector,
    // sort the vector, then reconstruct the list.
    llvm::SmallVector<SimpleReference *, 16> elements;
    for (SimpleReference &node : _references) {
      elements.push_back(&node);
    }
    std::sort(elements.begin(), elements.end(),
        [] (const SimpleReference *lhs, const SimpleReference *rhs) -> bool {
          uint64_t lhsOffset = lhs->offsetInAtom();
          uint64_t rhsOffset = rhs->offsetInAtom();
          if (rhsOffset != lhsOffset)
            return (lhsOffset < rhsOffset);
          if (rhs->kindNamespace() != lhs->kindNamespace())
            return (lhs->kindNamespace() < rhs->kindNamespace());
          if (rhs->kindArch() != lhs->kindArch())
            return (lhs->kindArch() < rhs->kindArch());
          return (lhs->kindValue() < rhs->kindValue());
        });
    _references.clearAndLeakNodesUnsafely();
    for (SimpleReference *node : elements) {
      _references.push_back(node);
    }
  }
  void setOrdinal(uint64_t ord) { _ordinal = ord; }

private:
  typedef llvm::ilist<SimpleReference> RefList;

  const File &_file;
  uint64_t _ordinal;
  mutable RefList _references;
};

class SimpleUndefinedAtom : public UndefinedAtom {
public:
  SimpleUndefinedAtom(const File &f, StringRef name) : _file(f), _name(name) {
    assert(!name.empty() && "UndefinedAtoms must have a name");
  }

  /// file - returns the File that produced/owns this Atom
  const File &file() const override { return _file; }

  /// name - The name of the atom. For a function atom, it is the (mangled)
  /// name of the function.
  StringRef name() const override { return _name; }

  CanBeNull canBeNull() const override { return UndefinedAtom::canBeNullNever; }

private:
  const File &_file;
  StringRef _name;
};

class SimpleAbsoluteAtom : public AbsoluteAtom {
public:
  SimpleAbsoluteAtom(const File &f, StringRef name, Scope s, uint64_t value)
      : _file(f), _name(name), _scope(s), _value(value) {}

  const File &file() const override { return _file; }
  StringRef name() const override { return _name; }
  uint64_t value() const override { return _value; }
  Scope scope() const override { return _scope; }

private:
  const File &_file;
  StringRef _name;
  Scope _scope;
  uint64_t _value;
};

} // end namespace lld

#endif
