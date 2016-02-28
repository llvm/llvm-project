//===- Core/Resolver.cpp - Resolves Atom References -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Atom.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/File.h"
#include "lld/Core/Instrumentation.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/SharedLibraryFile.h"
#include "lld/Core/SymbolTable.h"
#include "lld/Core/UndefinedAtom.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

namespace lld {

ErrorOr<bool> Resolver::handleFile(File &file) {
  if (auto ec = _ctx.handleLoadedFile(file))
    return ec;
  bool undefAdded = false;
  for (const DefinedAtom *atom : file.defined())
    doDefinedAtom(*atom);
  for (const UndefinedAtom *atom : file.undefined()) {
    if (doUndefinedAtom(*atom))
      undefAdded = true;
  }
  for (const SharedLibraryAtom *atom : file.sharedLibrary())
    doSharedLibraryAtom(*atom);
  for (const AbsoluteAtom *atom : file.absolute())
    doAbsoluteAtom(*atom);
  return undefAdded;
}

ErrorOr<bool> Resolver::forEachUndefines(File &file, bool searchForOverrides,
                                         UndefCallback callback) {
  size_t i = _undefineIndex[&file];
  bool undefAdded = false;
  do {
    for (; i < _undefines.size(); ++i) {
      StringRef undefName = _undefines[i];
      if (undefName.empty())
        continue;
      const Atom *atom = _symbolTable.findByName(undefName);
      if (!isa<UndefinedAtom>(atom) || _symbolTable.isCoalescedAway(atom)) {
        // The symbol was resolved by some other file. Cache the result.
        _undefines[i] = "";
        continue;
      }
      auto undefAddedOrError = callback(undefName, false);
      if (undefAddedOrError.getError())
        return undefAddedOrError;
      undefAdded |= undefAddedOrError.get();
    }
    if (!searchForOverrides)
      continue;
    for (StringRef tentDefName : _symbolTable.tentativeDefinitions()) {
      // Load for previous tentative may also have loaded
      // something that overrode this tentative, so always check.
      const Atom *curAtom = _symbolTable.findByName(tentDefName);
      assert(curAtom != nullptr);
      if (const DefinedAtom *curDefAtom = dyn_cast<DefinedAtom>(curAtom)) {
        if (curDefAtom->merge() == DefinedAtom::mergeAsTentative) {
          auto undefAddedOrError = callback(tentDefName, true);
          if (undefAddedOrError.getError())
            return undefAddedOrError;
          undefAdded |= undefAddedOrError.get();
        }
      }
    }
  } while (i < _undefines.size());
  _undefineIndex[&file] = i;
  return undefAdded;
}

ErrorOr<bool> Resolver::handleArchiveFile(File &file) {
  ArchiveLibraryFile *archiveFile = cast<ArchiveLibraryFile>(&file);
  bool searchForOverrides =
      _ctx.searchArchivesToOverrideTentativeDefinitions();
  return forEachUndefines(file, searchForOverrides,
                          [&](StringRef undefName,
                              bool dataSymbolOnly)->ErrorOr<bool> {
    if (File *member = archiveFile->find(undefName, dataSymbolOnly)) {
      member->setOrdinal(_ctx.getNextOrdinalAndIncrement());
      return handleFile(*member);
    }
    return false;
  });
}

std::error_code Resolver::handleSharedLibrary(File &file) {
  // Add all the atoms from the shared library
  SharedLibraryFile *sharedLibrary = cast<SharedLibraryFile>(&file);
  auto undefAddedOrError = handleFile(*sharedLibrary);
  if (undefAddedOrError.getError())
    return undefAddedOrError.getError();
  bool searchForOverrides =
      _ctx.searchSharedLibrariesToOverrideTentativeDefinitions();
  undefAddedOrError = forEachUndefines(file, searchForOverrides,
                                       [&](StringRef undefName,
                                           bool dataSymbolOnly)->ErrorOr<bool> {
    if (const SharedLibraryAtom *atom =
            sharedLibrary->exports(undefName, dataSymbolOnly))
      doSharedLibraryAtom(*atom);
    return false;
  });

  if (undefAddedOrError.getError())
    return undefAddedOrError.getError();
  return std::error_code();
}

bool Resolver::doUndefinedAtom(const UndefinedAtom &atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "       UndefinedAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name=" << atom.name() << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  bool newUndefAdded = _symbolTable.add(atom);
  if (newUndefAdded)
    _undefines.push_back(atom.name());

  return newUndefAdded;
}

// Called on each atom when a file is added. Returns true if a given
// atom is added to the symbol table.
void Resolver::doDefinedAtom(const DefinedAtom &atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "         DefinedAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", file=#"
                    << atom.file().ordinal()
                    << ", atom=#"
                    << atom.ordinal()
                    << ", name="
                    << atom.name()
                    << ", type="
                    << atom.contentType()
                    << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);
  _symbolTable.add(atom);

  // An atom that should never be dead-stripped is a dead-strip root.
  if (_ctx.deadStrip() && atom.deadStrip() == DefinedAtom::deadStripNever) {
    _deadStripRoots.insert(&atom);
  }
}

void Resolver::doSharedLibraryAtom(const SharedLibraryAtom &atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "   SharedLibraryAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name="
                    << atom.name()
                    << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  _symbolTable.add(atom);
}

void Resolver::doAbsoluteAtom(const AbsoluteAtom &atom) {
  DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                    << "       AbsoluteAtom: "
                    << llvm::format("0x%09lX", &atom)
                    << ", name="
                    << atom.name()
                    << "\n");

  // add to list of known atoms
  _atoms.push_back(&atom);

  // tell symbol table
  if (atom.scope() != Atom::scopeTranslationUnit)
    _symbolTable.add(atom);
}

// utility to add a vector of atoms
void Resolver::addAtoms(const std::vector<const DefinedAtom *> &newAtoms) {
  for (const DefinedAtom *newAtom : newAtoms)
    doDefinedAtom(*newAtom);
}

// Returns true if at least one of N previous files has created an
// undefined symbol.
bool Resolver::undefinesAdded(int begin, int end) {
  std::vector<std::unique_ptr<Node>> &inputs = _ctx.getNodes();
  for (int i = begin; i < end; ++i)
    if (FileNode *node = dyn_cast<FileNode>(inputs[i].get()))
      if (_newUndefinesAdded[node->getFile()])
        return true;
  return false;
}

File *Resolver::getFile(int &index) {
  std::vector<std::unique_ptr<Node>> &inputs = _ctx.getNodes();
  if ((size_t)index >= inputs.size())
    return nullptr;
  if (GroupEnd *group = dyn_cast<GroupEnd>(inputs[index].get())) {
    // We are at the end of the current group. If one or more new
    // undefined atom has been added in the last groupSize files, we
    // reiterate over the files.
    int size = group->getSize();
    if (undefinesAdded(index - size, index)) {
      index -= size;
      return getFile(index);
    }
    ++index;
    return getFile(index);
  }
  return cast<FileNode>(inputs[index++].get())->getFile();
}

// Keep adding atoms until _ctx.getNextFile() returns an error. This
// function is where undefined atoms are resolved.
bool Resolver::resolveUndefines() {
  DEBUG_WITH_TYPE("resolver",
                  llvm::dbgs() << "******** Resolving undefines:\n");
  ScopedTask task(getDefaultDomain(), "resolveUndefines");
  int index = 0;
  std::set<File *> seen;
  for (;;) {
    bool undefAdded = false;
    DEBUG_WITH_TYPE("resolver",
                    llvm::dbgs() << "Loading file #" << index << "\n");
    File *file = getFile(index);
    if (!file)
      return true;
    if (std::error_code ec = file->parse()) {
      llvm::errs() << "Cannot open " + file->path()
                   << ": " << ec.message() << "\n";
      return false;
    }
    DEBUG_WITH_TYPE("resolver",
                    llvm::dbgs() << "Loaded file: " << file->path() << "\n");
    switch (file->kind()) {
    case File::kindErrorObject:
    case File::kindNormalizedObject:
    case File::kindMachObject:
    case File::kindCEntryObject:
    case File::kindHeaderObject:
    case File::kindEntryObject:
    case File::kindUndefinedSymsObject:
    case File::kindAliasSymsObject:
    case File::kindStubHelperObject:
    case File::kindResolverMergedObject:
    case File::kindSectCreateObject: {
      // The same file may be visited more than once if the file is
      // in --start-group and --end-group. Only library files should
      // be processed more than once.
      if (seen.count(file))
        break;
      seen.insert(file);
      assert(!file->hasOrdinal());
      file->setOrdinal(_ctx.getNextOrdinalAndIncrement());
      auto undefAddedOrError = handleFile(*file);
      if (undefAddedOrError.getError()) {
        llvm::errs() << "Error in " + file->path()
                     << ": " << undefAddedOrError.getError().message() << "\n";
        return false;
      }
      undefAdded = undefAddedOrError.get();
      break;
    }
    case File::kindArchiveLibrary: {
      if (!file->hasOrdinal())
        file->setOrdinal(_ctx.getNextOrdinalAndIncrement());
      auto undefAddedOrError = handleArchiveFile(*file);
      if (undefAddedOrError.getError()) {
        llvm::errs() << "Error in " + file->path()
                     << ": " << undefAddedOrError.getError().message() << "\n";
        return false;
      }
      undefAdded = undefAddedOrError.get();
      break;
    }
    case File::kindSharedLibrary:
      if (!file->hasOrdinal())
        file->setOrdinal(_ctx.getNextOrdinalAndIncrement());
      if (auto EC = handleSharedLibrary(*file)) {
        llvm::errs() << "Error in " + file->path()
                     << ": " << EC.message() << "\n";
        return false;
      }
      break;
    }
    _newUndefinesAdded[file] = undefAdded;
  }
}

// switch all references to undefined or coalesced away atoms
// to the new defined atom
void Resolver::updateReferences() {
  DEBUG_WITH_TYPE("resolver",
                  llvm::dbgs() << "******** Updating references:\n");
  ScopedTask task(getDefaultDomain(), "updateReferences");
  for (const Atom *atom : _atoms) {
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom)) {
      for (const Reference *ref : *defAtom) {
        // A reference of type kindAssociate should't be updated.
        // Instead, an atom having such reference will be removed
        // if the target atom is coalesced away, so that they will
        // go away as a group.
        if (ref->kindNamespace() == lld::Reference::KindNamespace::all &&
            ref->kindValue() == lld::Reference::kindAssociate) {
          if (_symbolTable.isCoalescedAway(atom))
            _deadAtoms.insert(ref->target());
          continue;
        }
        const Atom *newTarget = _symbolTable.replacement(ref->target());
        const_cast<Reference *>(ref)->setTarget(newTarget);
      }
    }
  }
}

// For dead code stripping, recursively mark atoms "live"
void Resolver::markLive(const Atom *atom) {
  // Mark the atom is live. If it's already marked live, then stop recursion.
  auto exists = _liveAtoms.insert(atom);
  if (!exists.second)
    return;

  // Mark all atoms it references as live
  if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom)) {
    for (const Reference *ref : *defAtom)
      markLive(ref->target());
    for (auto &p : llvm::make_range(_reverseRef.equal_range(defAtom))) {
      const Atom *target = p.second;
      markLive(target);
    }
  }
}

static bool isBackref(const Reference *ref) {
  if (ref->kindNamespace() != lld::Reference::KindNamespace::all)
    return false;
  return (ref->kindValue() == lld::Reference::kindLayoutAfter);
}

// remove all atoms not actually used
void Resolver::deadStripOptimize() {
  DEBUG_WITH_TYPE("resolver",
                  llvm::dbgs() << "******** Dead stripping unused atoms:\n");
  ScopedTask task(getDefaultDomain(), "deadStripOptimize");
  // only do this optimization with -dead_strip
  if (!_ctx.deadStrip())
    return;

  // Some type of references prevent referring atoms to be dead-striped.
  // Make a reverse map of such references before traversing the graph.
  // While traversing the list of atoms, mark AbsoluteAtoms as live
  // in order to avoid reclaim.
  for (const Atom *atom : _atoms) {
    if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom))
      for (const Reference *ref : *defAtom)
        if (isBackref(ref))
          _reverseRef.insert(std::make_pair(ref->target(), atom));
    if (const AbsoluteAtom *absAtom = dyn_cast<AbsoluteAtom>(atom))
      markLive(absAtom);
  }

  // By default, shared libraries are built with all globals as dead strip roots
  if (_ctx.globalsAreDeadStripRoots())
    for (const Atom *atom : _atoms)
      if (const DefinedAtom *defAtom = dyn_cast<DefinedAtom>(atom))
        if (defAtom->scope() == DefinedAtom::scopeGlobal)
          _deadStripRoots.insert(defAtom);

  // Or, use list of names that are dead strip roots.
  for (const StringRef &name : _ctx.deadStripRoots()) {
    const Atom *symAtom = _symbolTable.findByName(name);
    assert(symAtom);
    _deadStripRoots.insert(symAtom);
  }

  // mark all roots as live, and recursively all atoms they reference
  for (const Atom *dsrAtom : _deadStripRoots)
    markLive(dsrAtom);

  // now remove all non-live atoms from _atoms
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(), [&](const Atom *a) {
                 return _liveAtoms.count(a) == 0;
               }),
               _atoms.end());
}

// error out if some undefines remain
bool Resolver::checkUndefines() {
  DEBUG_WITH_TYPE("resolver",
                  llvm::dbgs() << "******** Checking for undefines:\n");

  // build vector of remaining undefined symbols
  std::vector<const UndefinedAtom *> undefinedAtoms = _symbolTable.undefines();
  if (_ctx.deadStrip()) {
    // When dead code stripping, we don't care if dead atoms are undefined.
    undefinedAtoms.erase(
        std::remove_if(undefinedAtoms.begin(), undefinedAtoms.end(),
                       [&](const Atom *a) { return _liveAtoms.count(a) == 0; }),
        undefinedAtoms.end());
  }

  if (undefinedAtoms.empty())
    return false;

  // Warn about unresolved symbols.
  bool foundUndefines = false;
  for (const UndefinedAtom *undef : undefinedAtoms) {
    // Skip over a weak symbol.
    if (undef->canBeNull() != UndefinedAtom::canBeNullNever)
      continue;

    // If this is a library and undefined symbols are allowed on the
    // target platform, skip over it.
    if (isa<SharedLibraryFile>(undef->file()) && _ctx.allowShlibUndefines())
      continue;

    // If the undefine is coalesced away, skip over it.
    if (_symbolTable.isCoalescedAway(undef))
      continue;

    // Seems like this symbol is undefined. Warn that.
    foundUndefines = true;
    if (_ctx.printRemainingUndefines()) {
      llvm::errs() << "Undefined symbol: " << undef->file().path()
                   << ": " << _ctx.demangle(undef->name())
                   << "\n";
    }
  }
  if (!foundUndefines)
    return false;
  if (_ctx.printRemainingUndefines())
    llvm::errs() << "symbol(s) not found\n";
  return true;
}

// remove from _atoms all coaleseced away atoms
void Resolver::removeCoalescedAwayAtoms() {
  DEBUG_WITH_TYPE("resolver",
                  llvm::dbgs() << "******** Removing coalesced away atoms:\n");
  ScopedTask task(getDefaultDomain(), "removeCoalescedAwayAtoms");
  _atoms.erase(std::remove_if(_atoms.begin(), _atoms.end(), [&](const Atom *a) {
                 return _symbolTable.isCoalescedAway(a) || _deadAtoms.count(a);
               }),
               _atoms.end());
}

bool Resolver::resolve() {
  DEBUG_WITH_TYPE("resolver",
                  llvm::dbgs() << "******** Resolving atom references:\n");
  if (!resolveUndefines())
    return false;
  updateReferences();
  deadStripOptimize();
  if (checkUndefines()) {
    DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "Found undefines... ");
    if (!_ctx.allowRemainingUndefines()) {
      DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "which we don't allow\n");
      return false;
    }
    DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "which we are ok with\n");
  }
  removeCoalescedAwayAtoms();
  _result->addAtoms(_atoms);
  DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "******** Finished resolver\n");
  return true;
}

void Resolver::MergedFile::addAtoms(std::vector<const Atom *> &all) {
  ScopedTask task(getDefaultDomain(), "addAtoms");
  DEBUG_WITH_TYPE("resolver", llvm::dbgs() << "Resolver final atom list:\n");

  for (const Atom *atom : all) {
#ifndef NDEBUG
    if (auto *definedAtom = dyn_cast<DefinedAtom>(atom)) {
      DEBUG_WITH_TYPE("resolver", llvm::dbgs()
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
      DEBUG_WITH_TYPE("resolver", llvm::dbgs()
                      << llvm::format("    0x%09lX", atom)
                      << ", name="
                      << atom->name()
                      << "\n");
    }
#endif
    addAtom(*atom);
  }
}

} // namespace lld
