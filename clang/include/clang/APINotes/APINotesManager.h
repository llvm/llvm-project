//===--- APINotesManager.h - Manage API Notes Files -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the APINotesManager interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APINOTES_APINOTESMANAGER_H
#define LLVM_CLANG_APINOTES_APINOTESMANAGER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/VersionTuple.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace clang {

class DirectoryEntry;
class FileEntry;
class LangOptions;
class SourceManager;

namespace api_notes {

class APINotesReader;

/// The API notes manager helps find API notes associated with declarations.
///
/// API notes are externally-provided annotations for declarations that can
/// introduce new attributes (covering availability, nullability of
/// parameters/results, and so on) for specific declarations without directly
/// modifying the headers that contain those declarations.
///
/// The API notes manager is responsible for finding and loading the
/// external API notes files that correspond to a given header. Its primary
/// operation is \c findAPINotes(), which finds the API notes reader that
/// provides information about the declarations at that location.
class APINotesManager {
  typedef llvm::PointerUnion<const DirectoryEntry *, APINotesReader *>
    ReaderEntry;

  SourceManager &SourceMgr;

  /// Whether to implicitly search for API notes files based on the
  /// source file from which an entity was declared.
  bool ImplicitAPINotes;

  /// The Swift version to use when interpreting versioned API notes.
  VersionTuple SwiftVersion;

  /// API notes readers for the current module.
  ///
  /// There can be up to two of these, one for public headers and one
  /// for private headers.
  APINotesReader *CurrentModuleReaders[2] = { nullptr, nullptr };

  /// Whether we have already pruned the API notes cache.
  bool PrunedCache;

  /// A mapping from header file directories to the API notes reader for
  /// that directory, or a redirection to another directory entry that may
  /// have more information, or NULL to indicate that there is no API notes
  /// reader for this directory.
  llvm::DenseMap<const DirectoryEntry *, ReaderEntry> Readers;

  /// Load the API notes associated with the given file, whether it is
  /// the binary or source form of API notes.
  ///
  /// \returns the API notes reader for this file, or null if there is
  /// a failure.
  std::unique_ptr<APINotesReader> loadAPINotes(const FileEntry *apiNotesFile);

  /// Load the given API notes file for the given header directory.
  ///
  /// \param HeaderDir The directory at which we
  ///
  /// \returns true if an error occurred.
  bool loadAPINotes(const DirectoryEntry *HeaderDir,
                    const FileEntry *APINotesFile);

  /// Look for API notes in the given directory.
  ///
  /// This might find either a binary or source API notes.
  const FileEntry *findAPINotesFile(const DirectoryEntry *directory,
                                    StringRef filename,
                                    bool wantPublic = true);

  /// Attempt to load API notes for the given framework.
  ///
  /// \param FrameworkPath The path to the framework.
  /// \param Public Whether to load the public API notes. Otherwise, attempt
  /// to load the private API notes.
  ///
  /// \returns the header directory entry (e.g., for Headers or PrivateHeaders)
  /// for which the API notes were successfully loaded, or NULL if API notes
  /// could not be loaded for any reason.
  const DirectoryEntry *loadFrameworkAPINotes(llvm::StringRef FrameworkPath,
                                              llvm::StringRef FrameworkName,
                                              bool Public);

public:
  APINotesManager(SourceManager &sourceMgr, const LangOptions &langOpts);
  ~APINotesManager();

  /// Set the Swift version to use when filtering API notes.
  void setSwiftVersion(VersionTuple swiftVersion) {
    SwiftVersion = swiftVersion;
  }

  /// Load the API notes for the current module.
  ///
  /// \param module The current module.
  /// \param lookInModule Whether to look inside the module itself.
  /// \param searchPaths The paths in which we should search for API notes
  /// for the current module.
  ///
  /// \returns true if API notes were successfully loaded, \c false otherwise.
  bool loadCurrentModuleAPINotes(const Module *module,
                                 bool lookInModule,
                                 ArrayRef<std::string> searchPaths);

  /// Retrieve the set of API notes readers for the current module.
  ArrayRef<APINotesReader *> getCurrentModuleReaders() const {
    unsigned numReaders = static_cast<unsigned>(CurrentModuleReaders[0] != nullptr) +
      static_cast<unsigned>(CurrentModuleReaders[1] != nullptr);
    return llvm::makeArrayRef(CurrentModuleReaders).slice(0, numReaders);
  }

  /// Find the API notes readers that correspond to the given source location.
  llvm::SmallVector<APINotesReader *, 2> findAPINotes(SourceLocation Loc);
};

} // end namespace api_notes
} // end namespace clang

#endif
