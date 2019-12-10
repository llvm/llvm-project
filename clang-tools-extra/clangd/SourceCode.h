//===--- SourceCode.h - Manipulating source code as strings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various code that examines C++ source code without using heavy AST machinery
// (and often not even the lexer). To be used sparingly!
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SOURCECODE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SOURCECODE_H

#include "Context.h"
#include "Protocol.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA1.h"
#include <string>

namespace clang {
class SourceManager;

namespace clangd {

// We tend to generate digests for source codes in a lot of different places.
// This represents the type for those digests to prevent us hard coding details
// of hashing function at every place that needs to store this information.
using FileDigest = std::array<uint8_t, 8>;
FileDigest digest(StringRef Content);
Optional<FileDigest> digestFile(const SourceManager &SM, FileID FID);

// This context variable controls the behavior of functions in this file
// that convert between LSP offsets and native clang byte offsets.
// If not set, defaults to UTF-16 for backwards-compatibility.
extern Key<OffsetEncoding> kCurrentOffsetEncoding;

// Counts the number of UTF-16 code units needed to represent a string (LSP
// specifies string lengths in UTF-16 code units).
// Use of UTF-16 may be overridden by kCurrentOffsetEncoding.
size_t lspLength(StringRef Code);

/// Turn a [line, column] pair into an offset in Code.
///
/// If P.character exceeds the line length, returns the offset at end-of-line.
/// (If !AllowColumnsBeyondLineLength, then returns an error instead).
/// If the line number is out of range, returns an error.
///
/// The returned value is in the range [0, Code.size()].
llvm::Expected<size_t>
positionToOffset(llvm::StringRef Code, Position P,
                 bool AllowColumnsBeyondLineLength = true);

/// Turn an offset in Code into a [line, column] pair.
/// The offset must be in range [0, Code.size()].
Position offsetToPosition(llvm::StringRef Code, size_t Offset);

/// Turn a SourceLocation into a [line, column] pair.
/// FIXME: This should return an error if the location is invalid.
Position sourceLocToPosition(const SourceManager &SM, SourceLocation Loc);

/// Returns the taken range at \p TokLoc.
llvm::Optional<Range> getTokenRange(const SourceManager &SM,
                                    const LangOptions &LangOpts,
                                    SourceLocation TokLoc);

/// Return the file location, corresponding to \p P. Note that one should take
/// care to avoid comparing the result with expansion locations.
llvm::Expected<SourceLocation> sourceLocationInMainFile(const SourceManager &SM,
                                                        Position P);

/// Get the beginning SourceLocation at a specified \p Pos in the main file.
/// May be invalid if Pos is, or if there's no identifier or operators.
/// The returned position is in the main file, callers may prefer to
/// obtain the macro expansion location.
SourceLocation getBeginningOfIdentifier(const Position &Pos,
                                        const SourceManager &SM,
                                        const LangOptions &LangOpts);

/// Returns true iff \p Loc is inside the main file. This function handles
/// file & macro locations. For macro locations, returns iff the macro is being
/// expanded inside the main file.
///
/// The function is usually used to check whether a declaration is inside the
/// the main file.
bool isInsideMainFile(SourceLocation Loc, const SourceManager &SM);

/// Returns the #include location through which IncludedFIle was loaded.
/// Where SM.getIncludeLoc() returns the location of the *filename*, which may
/// be in a macro, includeHashLoc() returns the location of the #.
SourceLocation includeHashLoc(FileID IncludedFile, const SourceManager &SM);

/// Returns true if the token at Loc is spelled in the source code.
/// This is not the case for:
///   * symbols formed via macro concatenation, the spelling location will
///     be "<scratch space>"
///   * symbols controlled and defined by a compile command-line option
///     `-DName=foo`, the spelling location will be "<command line>".
bool isSpelledInSource(SourceLocation Loc, const SourceManager &SM);

/// Turns a token range into a half-open range and checks its correctness.
/// The resulting range will have only valid source location on both sides, both
/// of which are file locations.
///
/// File locations always point to a particular offset in a file, i.e. they
/// never refer to a location inside a macro expansion. Turning locations from
/// macro expansions into file locations is ambiguous - one can use
/// SourceManager::{getExpansion|getFile|getSpelling}Loc. This function
/// calls SourceManager::getFileLoc on both ends of \p R to do the conversion.
///
/// User input (e.g. cursor position) is expressed as a file location, so this
/// function can be viewed as a way to normalize the ranges used in the clang
/// AST so that they are comparable with ranges coming from the user input.
llvm::Optional<SourceRange> toHalfOpenFileRange(const SourceManager &Mgr,
                                                const LangOptions &LangOpts,
                                                SourceRange R);

/// Returns true iff all of the following conditions hold:
///   - start and end locations are valid,
///   - start and end locations are file locations from the same file
///     (i.e. expansion locations are not taken into account).
///   - start offset <= end offset.
/// FIXME: introduce a type for source range with this invariant.
bool isValidFileRange(const SourceManager &Mgr, SourceRange R);

/// Returns true iff \p L is contained in \p R.
/// EXPECTS: isValidFileRange(R) == true, L is a file location.
bool halfOpenRangeContains(const SourceManager &Mgr, SourceRange R,
                           SourceLocation L);

/// Returns true iff \p L is contained in \p R or \p L is equal to the end point
/// of \p R.
/// EXPECTS: isValidFileRange(R) == true, L is a file location.
bool halfOpenRangeTouches(const SourceManager &Mgr, SourceRange R,
                          SourceLocation L);

/// Returns the source code covered by the source range.
/// EXPECTS: isValidFileRange(R) == true.
llvm::StringRef toSourceCode(const SourceManager &SM, SourceRange R);

// Converts a half-open clang source range to an LSP range.
// Note that clang also uses closed source ranges, which this can't handle!
Range halfOpenToRange(const SourceManager &SM, CharSourceRange R);

// Converts an offset to a clang line/column (1-based, columns are bytes).
// The offset must be in range [0, Code.size()].
// Prefer to use SourceManager if one is available.
std::pair<size_t, size_t> offsetToClangLineColumn(llvm::StringRef Code,
                                                  size_t Offset);

/// From "a::b::c", return {"a::b::", "c"}. Scope is empty if there's no
/// qualifier.
std::pair<llvm::StringRef, llvm::StringRef>
splitQualifiedName(llvm::StringRef QName);

TextEdit replacementToEdit(StringRef Code, const tooling::Replacement &R);

std::vector<TextEdit> replacementsToEdits(StringRef Code,
                                          const tooling::Replacements &Repls);

TextEdit toTextEdit(const FixItHint &FixIt, const SourceManager &M,
                    const LangOptions &L);

/// Get the canonical path of \p F.  This means:
///
///   - Absolute path
///   - Symlinks resolved
///   - No "." or ".." component
///   - No duplicate or trailing directory separator
///
/// This function should be used when paths needs to be used outside the
/// component that generate it, so that paths are normalized as much as
/// possible.
llvm::Optional<std::string> getCanonicalPath(const FileEntry *F,
                                             const SourceManager &SourceMgr);

bool isRangeConsecutive(const Range &Left, const Range &Right);

/// Choose the clang-format style we should apply to a certain file.
/// This will usually use FS to look for .clang-format directories.
/// FIXME: should we be caching the .clang-format file search?
/// This uses format::DefaultFormatStyle and format::DefaultFallbackStyle,
/// though the latter may have been overridden in main()!
format::FormatStyle getFormatStyleForFile(llvm::StringRef File,
                                          llvm::StringRef Content,
                                          llvm::vfs::FileSystem *FS);

/// Cleanup and format the given replacements.
llvm::Expected<tooling::Replacements>
cleanupAndFormat(StringRef Code, const tooling::Replacements &Replaces,
                 const format::FormatStyle &Style);

/// A set of edits generated for a single file. Can verify whether it is safe to
/// apply these edits to a code block.
struct Edit {
  tooling::Replacements Replacements;
  std::string InitialCode;

  Edit(llvm::StringRef Code, tooling::Replacements Reps)
      : Replacements(std::move(Reps)), InitialCode(Code) {}

  /// Returns the file contents after changes are applied.
  llvm::Expected<std::string> apply() const;

  /// Represents Replacements as TextEdits that are available for use in LSP.
  std::vector<TextEdit> asTextEdits() const;

  /// Checks whether the Replacements are applicable to given Code.
  bool canApplyTo(llvm::StringRef Code) const;
};
/// A mapping from absolute file path (the one used for accessing the underlying
/// VFS) to edits.
using FileEdits = llvm::StringMap<Edit>;

/// Formats the edits and code around it according to Style. Changes
/// Replacements to formatted ones if succeeds.
llvm::Error reformatEdit(Edit &E, const format::FormatStyle &Style);

/// Collects identifiers with counts in the source code.
llvm::StringMap<unsigned> collectIdentifiers(llvm::StringRef Content,
                                             const format::FormatStyle &Style);

/// Collects all ranges of the given identifier in the source code.
std::vector<Range> collectIdentifierRanges(llvm::StringRef Identifier,
                                           llvm::StringRef Content,
                                           const LangOptions &LangOpts);

/// Collects words from the source code.
/// Unlike collectIdentifiers:
/// - also finds text in comments:
/// - splits text into words
/// - drops stopwords like "get" and "for"
llvm::StringSet<> collectWords(llvm::StringRef Content);

/// Heuristically determine namespaces visible at a point, without parsing Code.
/// This considers using-directives and enclosing namespace-declarations that
/// are visible (and not obfuscated) in the file itself (not headers).
/// Code should be truncated at the point of interest.
///
/// The returned vector is always non-empty.
/// - The first element is the namespace that encloses the point: a declaration
///   near the point would be within this namespace.
/// - The elements are the namespaces in scope at the point: an unqualified
///   lookup would search within these namespaces.
///
/// Using directives are resolved against all enclosing scopes, but no other
/// namespace directives.
///
/// example:
///   using namespace a;
///   namespace foo {
///     using namespace b;
///
/// visibleNamespaces are {"foo::", "", "a::", "b::", "foo::b::"}, not "a::b::".
std::vector<std::string> visibleNamespaces(llvm::StringRef Code,
                                           const format::FormatStyle &Style);

/// Represents locations that can accept a definition.
struct EligibleRegion {
  /// Namespace that owns all of the EligiblePoints, e.g.
  /// namespace a{ namespace b {^ void foo();^} }
  /// It will be “a::b” for both carrot locations.
  std::string EnclosingNamespace;
  /// Offsets into the code marking eligible points to insert a function
  /// definition.
  std::vector<Position> EligiblePoints;
};

/// Returns most eligible region to insert a definition for \p
/// FullyQualifiedName in the \p Code.
/// Pseudo parses \pCode under the hood to determine namespace decls and
/// possible insertion points. Choses the region that matches the longest prefix
/// of \p FullyQualifiedName. Returns EOF if there are no shared namespaces.
/// \p FullyQualifiedName should not contain anonymous namespaces.
EligibleRegion getEligiblePoints(llvm::StringRef Code,
                                 llvm::StringRef FullyQualifiedName,
                                 const format::FormatStyle &Style);

struct DefinedMacro {
  llvm::StringRef Name;
  const MacroInfo *Info;
};
/// Gets the macro at a specified \p Loc.
llvm::Optional<DefinedMacro> locateMacroAt(SourceLocation Loc,
                                           Preprocessor &PP);

/// Infers whether this is a header from the FileName and LangOpts (if
/// presents).
bool isHeaderFile(llvm::StringRef FileName,
                  llvm::Optional<LangOptions> LangOpts = llvm::None);

} // namespace clangd
} // namespace clang
#endif
