//===- PTHManager.h - Manager object for PTH processing ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PTHManager interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PTHMANAGER_H
#define LLVM_CLANG_LEX_PTHMANAGER_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/OnDiskHashTable.h"
#include <memory>

namespace llvm {

class MemoryBuffer;

} // namespace llvm

namespace clang {

class DiagnosticsEngine;
class FileSystemStatCache;
class Preprocessor;
class PTHLexer;

class PTHManager : public IdentifierInfoLookup {
  friend class PTHLexer;
  friend class PTHStatCache;

  class PTHFileLookupTrait;
  class PTHStringLookupTrait;

  using PTHStringIdLookup = llvm::OnDiskChainedHashTable<PTHStringLookupTrait>;
  using PTHFileLookup = llvm::OnDiskChainedHashTable<PTHFileLookupTrait>;

  /// The memory mapped PTH file.
  std::unique_ptr<const llvm::MemoryBuffer> Buf;

  /// Alloc - Allocator used for IdentifierInfo objects.
  llvm::BumpPtrAllocator Alloc;

  /// IdMap - A lazily generated cache mapping from persistent identifiers to
  ///  IdentifierInfo*.
  std::unique_ptr<IdentifierInfo *[], llvm::FreeDeleter> PerIDCache;

  /// FileLookup - Abstract data structure used for mapping between files
  ///  and token data in the PTH file.
  std::unique_ptr<PTHFileLookup> FileLookup;

  /// IdDataTable - Array representing the mapping from persistent IDs to the
  ///  data offset within the PTH file containing the information to
  ///  reconsitute an IdentifierInfo.
  const unsigned char* const IdDataTable;

  /// SortedIdTable - Abstract data structure mapping from strings to
  ///  persistent IDs.  This is used by get().
  std::unique_ptr<PTHStringIdLookup> StringIdLookup;

  /// NumIds - The number of identifiers in the PTH file.
  const unsigned NumIds;

  /// PP - The Preprocessor object that will use this PTHManager to create
  ///  PTHLexer objects.
  Preprocessor* PP = nullptr;

  /// SpellingBase - The base offset within the PTH memory buffer that
  ///  contains the cached spellings for literals.
  const unsigned char* const SpellingBase;

  /// OriginalSourceFile - A null-terminated C-string that specifies the name
  ///  if the file (if any) that was to used to generate the PTH cache.
  const char* OriginalSourceFile;

  /// This constructor is intended to only be called by the static 'Create'
  /// method.
  PTHManager(std::unique_ptr<const llvm::MemoryBuffer> buf,
             std::unique_ptr<PTHFileLookup> fileLookup,
             const unsigned char *idDataTable,
             std::unique_ptr<IdentifierInfo *[], llvm::FreeDeleter> perIDCache,
             std::unique_ptr<PTHStringIdLookup> stringIdLookup, unsigned numIds,
             const unsigned char *spellingBase, const char *originalSourceFile);

  /// getSpellingAtPTHOffset - Used by PTHLexer classes to get the cached
  ///  spelling for a token.
  unsigned getSpellingAtPTHOffset(unsigned PTHOffset, const char*& Buffer);

  /// GetIdentifierInfo - Used to reconstruct IdentifierInfo objects from the
  ///  PTH file.
  IdentifierInfo *GetIdentifierInfo(unsigned PersistentID) {
    // Check if the IdentifierInfo has already been resolved.
    if (IdentifierInfo* II = PerIDCache[PersistentID])
      return II;
    return LazilyCreateIdentifierInfo(PersistentID);
  }
  IdentifierInfo* LazilyCreateIdentifierInfo(unsigned PersistentID);

public:
  // The current PTH version.
  enum { Version = 10 };

  PTHManager(const PTHManager &) = delete;
  PTHManager &operator=(const PTHManager &) = delete;
  ~PTHManager() override;

  /// getOriginalSourceFile - Return the full path to the original header
  ///  file name that was used to generate the PTH cache.
  const char* getOriginalSourceFile() const {
    return OriginalSourceFile;
  }

  /// get - Return the identifier token info for the specified named identifier.
  ///  Unlike the version in IdentifierTable, this returns a pointer instead
  ///  of a reference.  If the pointer is nullptr then the IdentifierInfo cannot
  ///  be found.
  IdentifierInfo *get(StringRef Name) override;

  /// Create - This method creates PTHManager objects.  The 'file' argument
  ///  is the name of the PTH file.  This method returns nullptr upon failure.
  static PTHManager *Create(StringRef file, DiagnosticsEngine &Diags);

  void setPreprocessor(Preprocessor *pp) { PP = pp; }

  /// CreateLexer - Return a PTHLexer that "lexes" the cached tokens for the
  ///  specified file.  This method returns nullptr if no cached tokens exist.
  ///  It is the responsibility of the caller to 'delete' the returned object.
  PTHLexer *CreateLexer(FileID FID);

  /// createStatCache - Returns a FileSystemStatCache object for use with
  ///  FileManager objects.  These objects use the PTH data to speed up
  ///  calls to stat by memoizing their results from when the PTH file
  ///  was generated.
  std::unique_ptr<FileSystemStatCache> createStatCache();
};

} // namespace clang

#endif // LLVM_CLANG_LEX_PTHMANAGER_H
