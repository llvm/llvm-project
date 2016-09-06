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

  /// The API notes reader for the current module.
  std::unique_ptr<APINotesReader> CurrentModuleReader;

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

  /// Load the API notes for the current module.
  ///
  /// \param moduleName The name of the current module.
  /// \param searchPaths The paths in which we should search for API notes
  /// for the current module.
  ///
  /// \returns the file entry for the API notes file loaded, or nullptr if
  /// no API notes were found.
  const FileEntry *loadCurrentModuleAPINotes(StringRef moduleName,
                                             ArrayRef<std::string> searchPaths);

  /// Find the API notes reader that corresponds to the given source location.
  APINotesReader *findAPINotes(SourceLocation Loc);

  APINotesReader *getCurrentModuleReader() {
    return CurrentModuleReader.get();
  }
};

} // end namespace api_notes
} // end namespace clang

#endif
