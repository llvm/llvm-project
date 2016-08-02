//===--- APINotesManager.h - Manage API Notes Files -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the HeaderSearch interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APINOTES_APINOTESMANAGER_H
#define LLVM_CLANG_APINOTES_APINOTESMANAGER_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace clang {

class DirectoryEntry;
class FileEntry;
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

  /// Whether we have already pruned the API notes cache.
  bool PrunedCache;

  /// A mapping from header file directories to the API notes reader for
  /// that directory, or a redirection to another directory entry that may
  /// have more information, or NULL to indicate that there is no API notes
  /// reader for this directory.
  llvm::DenseMap<const DirectoryEntry *, ReaderEntry> Readers;

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
  APINotesManager(SourceManager &SourceMgr);
  ~APINotesManager();

  /// Find the API notes reader that corresponds to the given source location.
  APINotesReader *findAPINotes(SourceLocation Loc);
};

} // end namespace api_notes
} // end namespace clang

#endif
