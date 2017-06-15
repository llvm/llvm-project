//===--- SourceLocationUtilities.h - Source location helper functions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_REFACTOR_SOURCE_LOCATION_UTILITIES_H
#define LLVM_CLANG_LIB_TOOLING_REFACTOR_SOURCE_LOCATION_UTILITIES_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"

namespace clang {

class Stmt;
class LangOptions;

namespace tooling {

inline bool isPairOfFileLocations(SourceLocation Start, SourceLocation End) {
  return Start.isValid() && Start.isFileID() && End.isValid() && End.isFileID();
}

/// Return true if the Point is within Start and End.
inline bool isPointWithin(SourceLocation Location, SourceLocation Start,
                          SourceLocation End, const SourceManager &SM) {
  return Location == Start || Location == End ||
         (SM.isBeforeInTranslationUnit(Start, Location) &&
          SM.isBeforeInTranslationUnit(Location, End));
}

/// Return true if the two given ranges overlap with each other.
inline bool areRangesOverlapping(SourceRange R1, SourceRange R2,
                                 const SourceManager &SM) {
  return isPointWithin(R1.getBegin(), R2.getBegin(), R2.getEnd(), SM) ||
         isPointWithin(R2.getBegin(), R1.getBegin(), R1.getEnd(), SM);
}

/// \brief Return the source location that can be considered the last location
/// of the source construct before its body.
///
/// The returned location is determined using the following rules:
///
/// 1) If the source construct has a compound body that starts on the same line,
///    then this function will return the location of the opening '{'.
///
///    if (condition) {
///                   ^
///
/// 2) If the source construct's body is not a compound statement that starts
///    on the same line, then this function will return the location just before
///    the starting location of the body.
///
///    if (condition) foo()
///                  ^
///
/// 3) Otherwise, this function will return the last location on the line prior
///    to the the line on which the body starts.
///
///    if (condition)
///                  ^
///      foo()
///
/// \param HeaderEnd The last known location of the pre-body portion of the
/// source construct. For example, for an if statement, HeaderEnd should
/// be the ending location of its conditional expression.
SourceLocation findLastLocationOfSourceConstruct(SourceLocation HeaderEnd,
                                                 const Stmt *Body,
                                                 const SourceManager &SM);

/// \brief Return the source location that can be considered the first location
/// of the source construct prior to the previous portion of its body.
///
/// The returned location is determined using the following rules:
///
/// 1) If the source construct's body is a compound statement that ends
///    on the same line, then this function will return the location of the
///    closing '}'.
///
///    } else if (condition)
///    ^
///
/// 2) Otherwise, this function will return the starting location of the source
///    construct.
///
///      foo();
///    else if (condition)
///    ^
///
///    }
///    else if (condition)
///    ^
///
/// \param HeaderStart The first known location of the post-body portion of the
/// source construct. For example, for an if statement, HeaderStart should
/// be the starting location of the if keyword.
SourceLocation findFirstLocationOfSourceConstruct(SourceLocation HeaderStart,
                                                  const Stmt *PreviousBody,
                                                  const SourceManager &SM);

/// Return true if the given \p Location is within any range.
bool isLocationInAnyRange(SourceLocation Location, ArrayRef<SourceRange> Ranges,
                          const SourceManager &SM);

/// Return the precise end location for the given token.
SourceLocation getPreciseTokenLocEnd(SourceLocation Loc,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts);

/// \brief Find the source location right after the location of the next ')'.
///
/// If the token that's located after \p LastKnownLoc isn't ')', then this
/// function returns an invalid source location.
SourceLocation findClosingParenLocEnd(SourceLocation LastKnownLoc,
                                      const SourceManager &SM,
                                      const LangOptions &LangOpts);

/// Return the range of the next token if it has the given kind.
SourceRange getRangeOfNextToken(SourceLocation Loc, tok::TokenKind Kind,
                                const SourceManager &SM,
                                const LangOptions &LangOpts);

/// Return the end location of the body when \p S is a compound statement or an
/// invalid location when \p S is an empty compound statement. Otherwise,
/// return the end location of the given statement \p S.
SourceLocation findLastNonCompoundLocation(const Stmt *S);

/// Return true if the two locations are on the same line and aren't
///  macro locations.
bool areOnSameLine(SourceLocation Loc1, SourceLocation Loc2,
                   const SourceManager &SM);

/// Return the last location of the line which contains the given spellling
/// location \p SpellingLoc unless that line has other tokens after the given
/// location.
SourceLocation
getLastLineLocationUnlessItHasOtherTokens(SourceLocation SpellingLoc,
                                          const SourceManager &SM,
                                          const LangOptions &LangOpts);

/// Return true if the token at the given location is a semicolon.
bool isSemicolonAtLocation(SourceLocation TokenLoc, const SourceManager &SM,
                           const LangOptions &LangOpts);

/// Shrink the given range by ignoring leading whitespace and trailing
/// whitespace and semicolons.
///
/// Returns an invalid source range if the source range consists of whitespace
/// or semicolons only.
SourceRange trimSelectionRange(SourceRange Range, const SourceManager &SM,
                               const LangOptions &LangOpts);

/// Return the source location of the conjoined comment(s) that precede the
/// given location \p Loc, or the same location if there's no comment before
/// \p Loc.
SourceLocation getLocationOfPrecedingComment(SourceLocation Loc,
                                             const SourceManager &SM,
                                             const LangOptions &LangOpts);

/// Return the source location of the token that comes before the token at the
/// given location.
SourceLocation getLocationOfPrecedingToken(SourceLocation Loc,
                                           const SourceManager &SM,
                                           const LangOptions &LangOpts);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_LIB_TOOLING_REFACTOR_SOURCE_LOCATION_UTILITIES_H
