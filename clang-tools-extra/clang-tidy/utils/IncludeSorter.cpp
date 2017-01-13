//===---------- IncludeSorter.cpp - clang-tidy ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IncludeSorter.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace tidy {
namespace utils {

namespace {

StringRef RemoveFirstSuffix(StringRef Str, ArrayRef<const char *> Suffixes) {
  for (StringRef Suffix : Suffixes) {
    if (Str.endswith(Suffix)) {
      return Str.substr(0, Str.size() - Suffix.size());
    }
  }
  return Str;
}

StringRef MakeCanonicalName(StringRef Str, IncludeSorter::IncludeStyle Style) {
  // The list of suffixes to remove from source file names to get the
  // "canonical" file names.
  // E.g. tools/sort_includes.cc and tools/sort_includes_test.cc
  // would both canonicalize to tools/sort_includes and tools/sort_includes.h
  // (once canonicalized) will match as being the main include file associated
  // with the source files.
  if (Style == IncludeSorter::IS_LLVM) {
    return RemoveFirstSuffix(
        RemoveFirstSuffix(Str, {".cc", ".cpp", ".c", ".h", ".hpp"}), {"Test"});
  }
  return RemoveFirstSuffix(
      RemoveFirstSuffix(Str, {".cc", ".cpp", ".c", ".h", ".hpp"}),
      {"_unittest", "_regtest", "_test"});
}

// Scan to the end of the line and return the offset of the next line.
size_t FindNextLine(const char *Text) {
  size_t EOLIndex = std::strcspn(Text, "\n");
  return Text[EOLIndex] == '\0' ? EOLIndex : EOLIndex + 1;
}

IncludeSorter::IncludeKinds
DetermineIncludeKind(StringRef CanonicalFile, StringRef IncludeFile,
                     bool IsAngled, IncludeSorter::IncludeStyle Style) {
  // Compute the two "canonical" forms of the include's filename sans extension.
  // The first form is the include's filename without ".h" or "-inl.h" at the
  // end. The second form is the first form with "/public/" in the file path
  // replaced by "/internal/".
  if (IsAngled) {
    // If the system include (<foo>) ends with ".h", then it is a normal C-style
    // include. Otherwise assume it is a C++-style extensionless include.
    return IncludeFile.endswith(".h") ? IncludeSorter::IK_CSystemInclude
                                      : IncludeSorter::IK_CXXSystemInclude;
  }
  StringRef CanonicalInclude = MakeCanonicalName(IncludeFile, Style);
  if (CanonicalFile.endswith(CanonicalInclude)
      || CanonicalInclude.endswith(CanonicalFile)) {
    return IncludeSorter::IK_MainTUInclude;
  }
  if (Style == IncludeSorter::IS_Google) {
    std::pair<StringRef, StringRef> Parts = CanonicalInclude.split("/public/");
    std::string AltCanonicalInclude =
        Parts.first.str() + "/internal/" + Parts.second.str();
    std::string ProtoCanonicalInclude =
        Parts.first.str() + "/proto/" + Parts.second.str();

    // Determine the kind of this inclusion.
    if (CanonicalFile.equals(AltCanonicalInclude) ||
        CanonicalFile.equals(ProtoCanonicalInclude)) {
      return IncludeSorter::IK_MainTUInclude;
    }
  }
  return IncludeSorter::IK_NonSystemInclude;
}

} // namespace

IncludeSorter::IncludeSorter(const SourceManager *SourceMgr,
                             const LangOptions *LangOpts, const FileID FileID,
                             StringRef FileName, IncludeStyle Style)
    : SourceMgr(SourceMgr), LangOpts(LangOpts), Style(Style),
      CurrentFileID(FileID), CanonicalFile(MakeCanonicalName(FileName, Style)) {
}

void IncludeSorter::AddInclude(StringRef FileName, bool IsAngled,
                               SourceLocation HashLocation,
                               SourceLocation EndLocation) {
  int Offset = FindNextLine(SourceMgr->getCharacterData(EndLocation));

  // Record the relevant location information for this inclusion directive.
  IncludeLocations[FileName].push_back(
      SourceRange(HashLocation, EndLocation.getLocWithOffset(Offset)));
  SourceLocations.push_back(IncludeLocations[FileName].back());

  // Stop if this inclusion is a duplicate.
  if (IncludeLocations[FileName].size() > 1)
    return;

  // Add the included file's name to the appropriate bucket.
  IncludeKinds Kind =
      DetermineIncludeKind(CanonicalFile, FileName, IsAngled, Style);
  if (Kind != IK_InvalidInclude)
    IncludeBucket[Kind].push_back(FileName.str());
}

Optional<FixItHint> IncludeSorter::CreateIncludeInsertion(StringRef FileName,
                                                          bool IsAngled) {
  std::string IncludeStmt =
      IsAngled ? llvm::Twine("#include <" + FileName + ">\n").str()
               : llvm::Twine("#include \"" + FileName + "\"\n").str();
  if (SourceLocations.empty()) {
    // If there are no includes in this file, add it in the first line.
    // FIXME: insert after the file comment or the header guard, if present.
    IncludeStmt.append("\n");
    return FixItHint::CreateInsertion(
        SourceMgr->getLocForStartOfFile(CurrentFileID), IncludeStmt);
  }

  auto IncludeKind =
      DetermineIncludeKind(CanonicalFile, FileName, IsAngled, Style);

  if (!IncludeBucket[IncludeKind].empty()) {
    for (const std::string &IncludeEntry : IncludeBucket[IncludeKind]) {
      if (FileName < IncludeEntry) {
        const auto &Location = IncludeLocations[IncludeEntry][0];
        return FixItHint::CreateInsertion(Location.getBegin(), IncludeStmt);
      } else if (FileName == IncludeEntry) {
        return llvm::None;
      }
    }
    // FileName comes after all include entries in bucket, insert it after
    // last.
    const std::string &LastInclude = IncludeBucket[IncludeKind].back();
    SourceRange LastIncludeLocation = IncludeLocations[LastInclude].back();
    return FixItHint::CreateInsertion(LastIncludeLocation.getEnd(),
                                      IncludeStmt);
  }
  // Find the non-empty include bucket to be sorted directly above
  // 'IncludeKind'. If such a bucket exists, we'll want to sort the include
  // after that bucket. If no such bucket exists, find the first non-empty
  // include bucket in the file. In that case, we'll want to sort the include
  // before that bucket.
  IncludeKinds NonEmptyKind = IK_InvalidInclude;
  for (int i = IK_InvalidInclude - 1; i >= 0; --i) {
    if (!IncludeBucket[i].empty()) {
      NonEmptyKind = static_cast<IncludeKinds>(i);
      if (NonEmptyKind < IncludeKind)
        break;
    }
  }
  if (NonEmptyKind == IK_InvalidInclude) {
    return llvm::None;
  }

  if (NonEmptyKind < IncludeKind) {
    // Create a block after.
    const std::string &LastInclude = IncludeBucket[NonEmptyKind].back();
    SourceRange LastIncludeLocation = IncludeLocations[LastInclude].back();
    IncludeStmt = '\n' + IncludeStmt;
    return FixItHint::CreateInsertion(LastIncludeLocation.getEnd(),
                                      IncludeStmt);
  }
  // Create a block before.
  const std::string &FirstInclude = IncludeBucket[NonEmptyKind][0];
  SourceRange FirstIncludeLocation = IncludeLocations[FirstInclude].back();
  IncludeStmt.append("\n");
  return FixItHint::CreateInsertion(FirstIncludeLocation.getBegin(),
                                    IncludeStmt);
}

std::vector<FixItHint> IncludeSorter::GetEdits() {
  if (SourceLocations.empty())
    return {};

  typedef std::map<int, std::pair<SourceRange, std::string>>
      FileLineToSourceEditMap;
  FileLineToSourceEditMap Edits;
  auto SourceLocationIterator = SourceLocations.begin();
  auto SourceLocationIteratorEnd = SourceLocations.end();

  // Compute the Edits that need to be done to each line to add, replace, or
  // delete inclusions.
  for (int IncludeKind = 0; IncludeKind < IK_InvalidInclude; ++IncludeKind) {
    std::sort(IncludeBucket[IncludeKind].begin(),
              IncludeBucket[IncludeKind].end());
    for (const auto &IncludeEntry : IncludeBucket[IncludeKind]) {
      auto &Location = IncludeLocations[IncludeEntry];
      SourceRangeVector::iterator LocationIterator = Location.begin();
      SourceRangeVector::iterator LocationIteratorEnd = Location.end();
      SourceRange FirstLocation = *LocationIterator;

      // If the first occurrence of a particular include is on the current
      // source line we are examining, leave it alone.
      if (FirstLocation == *SourceLocationIterator)
        ++LocationIterator;

      // Add the deletion Edits for any (remaining) instances of this inclusion,
      // and remove their Locations from the source Locations to be processed.
      for (; LocationIterator != LocationIteratorEnd; ++LocationIterator) {
        int LineNumber =
            SourceMgr->getSpellingLineNumber(LocationIterator->getBegin());
        Edits[LineNumber] = std::make_pair(*LocationIterator, "");
        SourceLocationIteratorEnd =
            std::remove(SourceLocationIterator, SourceLocationIteratorEnd,
                        *LocationIterator);
      }

      if (FirstLocation == *SourceLocationIterator) {
        // Do nothing except move to the next source Location (Location of an
        // inclusion in the original, unchanged source file).
        ++SourceLocationIterator;
        continue;
      }

      // Add (or append to) the replacement text for this line in source file.
      int LineNumber =
          SourceMgr->getSpellingLineNumber(SourceLocationIterator->getBegin());
      if (Edits.find(LineNumber) == Edits.end()) {
        Edits[LineNumber].first =
            SourceRange(SourceLocationIterator->getBegin());
      }
      StringRef SourceText = Lexer::getSourceText(
          CharSourceRange::getCharRange(FirstLocation), *SourceMgr, *LangOpts);
      Edits[LineNumber].second.append(SourceText.data(), SourceText.size());
    }

    // Clear the bucket.
    IncludeBucket[IncludeKind].clear();
  }

  // Go through the single-line Edits and combine them into blocks of Edits.
  int CurrentEndLine = 0;
  SourceRange CurrentRange;
  std::string CurrentText;
  std::vector<FixItHint> Fixes;
  for (const auto &LineEdit : Edits) {
    // If the current edit is on the next line after the previous edit, add it
    // to the current block edit.
    if (LineEdit.first == CurrentEndLine + 1 &&
        CurrentRange.getBegin() != CurrentRange.getEnd()) {
      SourceRange EditRange = LineEdit.second.first;
      if (EditRange.getBegin() != EditRange.getEnd()) {
        ++CurrentEndLine;
        CurrentRange.setEnd(EditRange.getEnd());
      }
      CurrentText += LineEdit.second.second;
      // Otherwise report the current block edit and start a new block.
    } else {
      if (CurrentEndLine) {
        Fixes.push_back(FixItHint::CreateReplacement(
            CharSourceRange::getCharRange(CurrentRange), CurrentText));
      }

      CurrentEndLine = LineEdit.first;
      CurrentRange = LineEdit.second.first;
      CurrentText = LineEdit.second.second;
    }
  }
  // Finally, report the current block edit if there is one.
  if (CurrentEndLine) {
    Fixes.push_back(FixItHint::CreateReplacement(
        CharSourceRange::getCharRange(CurrentRange), CurrentText));
  }

  // Reset the remaining internal state.
  SourceLocations.clear();
  IncludeLocations.clear();
  return Fixes;
}

IncludeSorter::IncludeStyle
IncludeSorter::parseIncludeStyle(const std::string &Value) {
  return Value == "llvm" ? IS_LLVM : IS_Google;
}

StringRef IncludeSorter::toString(IncludeStyle Style) {
  return Style == IS_LLVM ? "llvm" : "google";
}

} // namespace utils
} // namespace tidy
} // namespace clang
