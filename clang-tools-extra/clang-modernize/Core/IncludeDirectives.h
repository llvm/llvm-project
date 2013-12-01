//===-- Core/IncludeDirectives.h - Include directives handling --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares the IncludeDirectives class that helps with
/// detecting and modifying \#include directives.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_INCLUDE_DIRECTIVES_H
#define CLANG_MODERNIZE_INCLUDE_DIRECTIVES_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <vector>

namespace clang {
class Preprocessor;
} // namespace clang

/// \brief Support for include directives handling.
///
/// This class should be created with a \c clang::CompilerInstance before the
/// file is preprocessed in order to collect the inclusion information. It can
/// be queried as long as the compiler instance is valid.
class IncludeDirectives {
public:
  IncludeDirectives(clang::CompilerInstance &CI);

  /// \brief Add an angled include to a the given file.
  ///
  /// \param File A file accessible by a SourceManager
  /// \param Include The include file as it should be written in the code.
  ///
  /// \returns
  /// \li A null Replacement (check using \c Replacement::isApplicable()), if
  ///     the \c Include is already visible from \c File.
  /// \li Otherwise, a non-null Replacement that, when applied, inserts an
  ///     \c \#include into \c File.
  clang::tooling::Replacement addAngledInclude(llvm::StringRef File,
                                               llvm::StringRef Include);
  clang::tooling::Replacement addAngledInclude(const clang::FileEntry *File,
                                               llvm::StringRef Include);

  /// \brief Check if \p Include is included by \p File or any of the files
  /// \p File includes.
  bool hasInclude(const clang::FileEntry *File, llvm::StringRef Include) const;

private:
  friend class IncludeDirectivesPPCallback;

  /// \brief Contains information about an inclusion.
  class Entry {
  public:
    Entry(clang::SourceLocation HashLoc, const clang::FileEntry *IncludedFile,
          bool Angled)
        : HashLoc(HashLoc), IncludedFile(IncludedFile), Angled(Angled) {}

    /// \brief The location of the '#'.
    clang::SourceLocation getHashLocation() const { return HashLoc; }

    /// \brief The file included by this include directive.
    const clang::FileEntry *getIncludedFile() const { return IncludedFile; }

    /// \brief \c true if the include use angle brackets, \c false otherwise
    /// when using of quotes.
    bool isAngled() const { return Angled; }

  private:
    clang::SourceLocation HashLoc;
    const clang::FileEntry *IncludedFile;
    bool Angled;
  };

  // A list of entries.
  typedef std::vector<Entry> EntryVec;

  // A list of source locations.
  typedef std::vector<clang::SourceLocation> LocationVec;

  // Associates files to their includes.
  typedef llvm::DenseMap<const clang::FileEntry *, EntryVec> FileToEntriesMap;

  // Associates headers to their include guards if any. The location is the
  // location of the hash from the #define.
  typedef llvm::DenseMap<const clang::FileEntry *, clang::SourceLocation>
  HeaderToGuardMap;

  /// \brief Type used by \c lookForInclude() to keep track of the files that
  /// have already been processed.
  typedef llvm::SmallPtrSet<const clang::FileEntry *, 32> SeenFilesSet;

  /// \brief Recursively look if an include is included by \p File or any of the
  /// headers \p File includes.
  ///
  /// \param File The file where to start the search.
  /// \param IncludeLocs These are the hash locations of the \#include
  /// directives we are looking for.
  /// \param Seen Used to avoid visiting a same file more than once during the
  /// recursion.
  bool lookForInclude(const clang::FileEntry *File,
                      const LocationVec &IncludeLocs, SeenFilesSet &Seen) const;

  /// \brief Find the end of a file header and returns a pair (FileOffset,
  /// NewLineFlags).
  ///
  /// Source files often contain a file header (copyright, license, explanation
  /// of the file content). An \#include should preferably be put after this.
  std::pair<unsigned, unsigned>
  findFileHeaderEndOffset(clang::FileID FID) const;

  /// \brief Finds the offset where an angled include should be added and
  /// returns a pair (FileOffset, NewLineFlags).
  std::pair<unsigned, unsigned>
  angledIncludeInsertionOffset(clang::FileID FID) const;

  /// \brief Find the location of an include directive that can be used to
  /// insert an inclusion after.
  ///
  /// If no such include exists returns a null SourceLocation.
  clang::SourceLocation angledIncludeHintLoc(clang::FileID FID) const;

  clang::CompilerInstance &CI;
  clang::SourceManager &Sources;
  FileToEntriesMap FileToEntries;
  // maps include filename as written in the source code to the source locations
  // where it appears
  llvm::StringMap<LocationVec> IncludeAsWrittenToLocationsMap;
  HeaderToGuardMap HeaderToGuard;
};

#endif // CLANG_MODERNIZE_INCLUDE_DIRECTIVES_H
