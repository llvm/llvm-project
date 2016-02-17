//===- Core/File.h - A Container of Atoms ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_FILE_H
#define LLD_CORE_FILE_H

#include "lld/Core/AbsoluteAtom.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/SharedLibraryAtom.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/Core/range.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace lld {

class LinkingContext;

/// Every Atom is owned by some File. A common scenario is for a single
/// object file (.o) to be parsed by some reader and produce a single
/// File object that represents the content of that object file.
///
/// To iterate through the Atoms in a File there are four methods that
/// return collections.  For instance to iterate through all the DefinedAtoms
/// in a File object use:
///      for (const DefinedAtoms *atom : file->defined()) {
///      }
///
/// The Atom objects in a File are owned by the File object.  The Atom objects
/// are destroyed when the File object is destroyed.
class File {
public:
  virtual ~File();

  /// \brief Kinds of files that are supported.
  enum Kind {
    kindErrorObject,          ///< a error object file (.o)
    kindNormalizedObject,     ///< a normalized file (.o)
    kindMachObject,           ///< a MachO object file (.o)
    kindELFObject,            ///< a ELF object file (.o)
    kindCEntryObject,         ///< a file for CEntries
    kindHeaderObject,         ///< a file for file headers
    kindEntryObject,          ///< a file for the entry
    kindUndefinedSymsObject,  ///< a file for undefined symbols
    kindAliasSymsObject,      ///< a file for alias symbols
    kindStubHelperObject,     ///< a file for stub helpers
    kindResolverMergedObject, ///< the resolver merged file.
    kindSectCreateObject,     ///< a sect create object file (.o)
    kindSharedLibrary,        ///< shared library (.so)
    kindArchiveLibrary        ///< archive (.a)
  };

  /// \brief Returns file kind.  Need for dyn_cast<> on File objects.
  Kind kind() const {
    return _kind;
  }

  /// This returns the path to the file which was used to create this object
  /// (e.g. "/tmp/foo.o"). If the file is a member of an archive file, the
  /// returned string includes the archive file name.
  StringRef path() const {
    if (_archivePath.empty())
      return _path;
    if (_archiveMemberPath.empty())
      _archiveMemberPath = (_archivePath + "(" + _path + ")").str();
    return _archiveMemberPath;
  }

  /// Returns the path of the archive file name if this file is instantiated
  /// from an archive file. Otherwise returns the empty string.
  StringRef archivePath() const { return _archivePath; }
  void setArchivePath(StringRef path) { _archivePath = path; }

  /// Returns the path name of this file. It doesn't include archive file name.
  StringRef memberPath() const { return _path; }

  /// Returns the command line order of the file.
  uint64_t ordinal() const {
    assert(_ordinal != UINT64_MAX);
    return _ordinal;
  }

  /// Returns true/false depending on whether an ordinal has been set.
  bool hasOrdinal() const { return (_ordinal != UINT64_MAX); }

  /// Sets the command line order of the file.
  void setOrdinal(uint64_t ordinal) const { _ordinal = ordinal; }

  /// Returns the ordinal for the next atom to be defined in this file.
  uint64_t getNextAtomOrdinalAndIncrement() const {
    return _nextAtomOrdinal++;
  }

  /// For allocating any objects owned by this File.
  llvm::BumpPtrAllocator &allocator() const {
    return _allocator;
  }

  /// The type of atom mutable container.
  template <typename T> using AtomVector = std::vector<const T *>;

  /// The range type for the atoms. It's backed by a std::vector, but hides
  /// its member functions so that you can only call begin or end.
  template <typename T> class AtomRange {
  public:
    AtomRange(AtomVector<T> v) : _v(v) {}
    typename AtomVector<T>::const_iterator begin() const { return _v.begin(); }
    typename AtomVector<T>::const_iterator end() const { return _v.end(); }
    typename AtomVector<T>::iterator begin() { return _v.begin(); }
    typename AtomVector<T>::iterator end() { return _v.end(); }

  private:
    AtomVector<T> &_v;
  };

  /// \brief Must be implemented to return the AtomVector object for
  /// all DefinedAtoms in this File.
  virtual const AtomVector<DefinedAtom> &defined() const = 0;

  /// \brief Must be implemented to return the AtomVector object for
  /// all UndefinedAtomw in this File.
  virtual const AtomVector<UndefinedAtom> &undefined() const = 0;

  /// \brief Must be implemented to return the AtomVector object for
  /// all SharedLibraryAtoms in this File.
  virtual const AtomVector<SharedLibraryAtom> &sharedLibrary() const = 0;

  /// \brief Must be implemented to return the AtomVector object for
  /// all AbsoluteAtoms in this File.
  virtual const AtomVector<AbsoluteAtom> &absolute() const = 0;

  /// \brief If a file is parsed using a different method than doParse(),
  /// one must use this method to set the last error status, so that
  /// doParse will not be called twice. Only YAML reader uses this
  /// (because YAML reader does not read blobs but structured data).
  void setLastError(std::error_code err) { _lastError = err; }

  std::error_code parse();

  // This function is called just before the core linker tries to use
  // a file. Currently the PECOFF reader uses this to trigger the
  // driver to parse .drectve section (which contains command line options).
  // If you want to do something having side effects, don't do that in
  // doParse() because a file could be pre-loaded speculatively.
  // Use this hook instead.
  virtual void beforeLink() {}

  // Usually each file owns a std::unique_ptr<MemoryBuffer>.
  // However, there's one special case. If a file is an archive file,
  // the archive file and its children all shares the same memory buffer.
  // This method is used by the ArchiveFile to give its children
  // co-ownership of the buffer.
  void setSharedMemoryBuffer(std::shared_ptr<MemoryBuffer> mb) {
    _sharedMemoryBuffer = mb;
  }

protected:
  /// \brief only subclasses of File can be instantiated
  File(StringRef p, Kind kind)
    : _path(p), _kind(kind), _ordinal(UINT64_MAX),
      _nextAtomOrdinal(0) {}

  /// \brief Subclasses should override this method to parse the
  /// memory buffer passed to this file's constructor.
  virtual std::error_code doParse() { return std::error_code(); }

  static AtomVector<DefinedAtom> _noDefinedAtoms;
  static AtomVector<UndefinedAtom> _noUndefinedAtoms;
  static AtomVector<SharedLibraryAtom> _noSharedLibraryAtoms;
  static AtomVector<AbsoluteAtom> _noAbsoluteAtoms;
  mutable llvm::BumpPtrAllocator _allocator;

private:
  StringRef _path;
  std::string _archivePath;
  mutable std::string _archiveMemberPath;
  Kind              _kind;
  mutable uint64_t  _ordinal;
  mutable uint64_t _nextAtomOrdinal;
  std::shared_ptr<MemoryBuffer> _sharedMemoryBuffer;
  llvm::Optional<std::error_code> _lastError;
  std::mutex _parseMutex;
};

/// An ErrorFile represents a file that doesn't exist.
/// If you try to parse a file which doesn't exist, an instance of this
/// class will be returned. That's parse method always returns an error.
/// This is useful to delay erroring on non-existent files, so that we
/// can do unit testing a driver using non-existing file paths.
class ErrorFile : public File {
public:
  ErrorFile(StringRef path, std::error_code ec)
      : File(path, kindErrorObject), _ec(ec) {}

  std::error_code doParse() override { return _ec; }

  const AtomVector<DefinedAtom> &defined() const override {
    llvm_unreachable("internal error");
  }
  const AtomVector<UndefinedAtom> &undefined() const override {
    llvm_unreachable("internal error");
  }
  const AtomVector<SharedLibraryAtom> &sharedLibrary() const override {
    llvm_unreachable("internal error");
  }
  const AtomVector<AbsoluteAtom> &absolute() const override {
    llvm_unreachable("internal error");
  }

private:
  std::error_code _ec;
};

} // end namespace lld

#endif
