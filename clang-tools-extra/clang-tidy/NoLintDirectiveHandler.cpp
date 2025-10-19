//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements the NoLintDirectiveHandler class, which is used
///  to locate NOLINT comments in the file being analyzed, to decide whether a
///  diagnostic should be suppressed.
///
//===----------------------------------------------------------------------===//

#include "NoLintDirectiveHandler.h"
#include "GlobList.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace clang::tidy {

//===----------------------------------------------------------------------===//
// NoLintType
//===----------------------------------------------------------------------===//

// The type - one of NOLINT[NEXTLINE/BEGIN/END].
enum class NoLintType { NoLint, NoLintNextLine, NoLintBegin, NoLintEnd };

// Convert a string like "NOLINTNEXTLINE" to its enum `Type::NoLintNextLine`.
// Return `std::nullopt` if the string is unrecognized.
static std::optional<NoLintType> strToNoLintType(StringRef Str) {
  auto Type = llvm::StringSwitch<std::optional<NoLintType>>(Str)
                  .Case("NOLINT", NoLintType::NoLint)
                  .Case("NOLINTNEXTLINE", NoLintType::NoLintNextLine)
                  .Case("NOLINTBEGIN", NoLintType::NoLintBegin)
                  .Case("NOLINTEND", NoLintType::NoLintEnd)
                  .Default(std::nullopt);
  return Type;
}

//===----------------------------------------------------------------------===//
// NoLintToken
//===----------------------------------------------------------------------===//

// Whitespace within a NOLINT's check list shall be ignored.
// "NOLINT( check1, check2 )" is equivalent to "NOLINT(check1,check2)".
// Return the check list with all extraneous whitespace removed.
static std::string trimWhitespace(StringRef Checks) {
  SmallVector<StringRef> Split;
  Checks.split(Split, ',');
  for (StringRef &Check : Split)
    Check = Check.trim();
  return llvm::join(Split, ",");
}

namespace {

// Record the presence of a NOLINT comment - its type, location, checks -
// as parsed from the file's character contents.
class NoLintToken {
public:
  // \param Checks:
  // - If unspecified (i.e. `None`) then ALL checks are suppressed - equivalent
  //   to NOLINT(*).
  // - An empty string means nothing is suppressed - equivalent to NOLINT().
  // - Negative globs ignored (which would effectively disable the suppression).
  NoLintToken(NoLintType Type, size_t Pos,
              const std::optional<StringRef> &Checks)
      : Type(Type), Pos(Pos), ChecksGlob(std::make_unique<CachedGlobList>(
                                  Checks.value_or("*"),
                                  /*KeepNegativeGlobs=*/false)) {
    if (Checks)
      this->Checks = trimWhitespace(*Checks);
  }

  // The type - one of NOLINT[NEXTLINE/BEGIN/END].
  NoLintType Type;

  // The location of the first character, "N", in "NOLINT".
  size_t Pos;

  // A glob of the checks this NOLINT token disables.
  std::unique_ptr<CachedGlobList> ChecksGlob;

  // If this NOLINT specifies checks, return the checks.
  const std::optional<std::string> &checks() const { return Checks; }

  // Whether this NOLINT applies to the provided check.
  bool suppresses(StringRef Check) const { return ChecksGlob->contains(Check); }

private:
  std::optional<std::string> Checks;
};

} // namespace

// Consume the entire buffer and return all `NoLintToken`s that were found.
static SmallVector<NoLintToken> getNoLints(StringRef Buffer) {
  static constexpr llvm::StringLiteral NOLINT = "NOLINT";
  SmallVector<NoLintToken> NoLints;

  size_t Pos = 0;
  while (Pos < Buffer.size()) {
    // Find NOLINT:
    const size_t NoLintPos = Buffer.find(NOLINT, Pos);
    if (NoLintPos == StringRef::npos)
      break; // Buffer exhausted

    // Read [A-Z] characters immediately after "NOLINT", e.g. the "NEXTLINE" in
    // "NOLINTNEXTLINE".
    Pos = NoLintPos + NOLINT.size();
    while (Pos < Buffer.size() && llvm::isAlpha(Buffer[Pos]))
      ++Pos;

    // Is this a recognized NOLINT type?
    const std::optional<NoLintType> NoLintType =
        strToNoLintType(Buffer.slice(NoLintPos, Pos));
    if (!NoLintType)
      continue;

    // Get checks, if specified.
    std::optional<StringRef> Checks;
    if (Pos < Buffer.size() && Buffer[Pos] == '(') {
      size_t ClosingBracket = Buffer.find_first_of("\n)", ++Pos);
      if (ClosingBracket != StringRef::npos && Buffer[ClosingBracket] == ')') {
        Checks = Buffer.slice(Pos, ClosingBracket);
        Pos = ClosingBracket + 1;
      }
    }

    NoLints.emplace_back(*NoLintType, NoLintPos, Checks);
  }

  return NoLints;
}

//===----------------------------------------------------------------------===//
// NoLintBlockToken
//===----------------------------------------------------------------------===//

namespace {

// Represents a source range within a pair of NOLINT(BEGIN/END) comments.
class NoLintBlockToken {
public:
  NoLintBlockToken(size_t BeginPos, size_t EndPos,
                   std::unique_ptr<CachedGlobList> ChecksGlob)
      : BeginPos(BeginPos), EndPos(EndPos), ChecksGlob(std::move(ChecksGlob)) {}

  // Whether the provided diagnostic is within and is suppressible by this block
  // of NOLINT(BEGIN/END) comments.
  bool suppresses(size_t DiagPos, StringRef DiagName) const {
    return (BeginPos < DiagPos) && (DiagPos < EndPos) &&
           ChecksGlob->contains(DiagName);
  }

private:
  size_t BeginPos;
  size_t EndPos;
  std::unique_ptr<CachedGlobList> ChecksGlob;
};

} // namespace

// Construct a [clang-tidy-nolint] diagnostic to do with the unmatched
// NOLINT(BEGIN/END) pair.
static tooling::Diagnostic makeNoLintError(const SourceManager &SrcMgr,
                                           FileID File,
                                           const NoLintToken &NoLint) {
  tooling::Diagnostic Error;
  Error.DiagLevel = tooling::Diagnostic::Error;
  Error.DiagnosticName = "clang-tidy-nolint";
  StringRef Message =
      (NoLint.Type == NoLintType::NoLintBegin)
          ? ("unmatched 'NOLINTBEGIN' comment without a subsequent 'NOLINT"
             "END' comment")
          : ("unmatched 'NOLINTEND' comment without a previous 'NOLINT"
             "BEGIN' comment");
  SourceLocation Loc = SrcMgr.getComposedLoc(File, NoLint.Pos);
  Error.Message = tooling::DiagnosticMessage(Message, SrcMgr, Loc);
  return Error;
}

// Match NOLINTBEGINs with their corresponding NOLINTENDs and move them into
// `NoLintBlockToken`s. If any BEGINs or ENDs are left over, a diagnostic is
// written to `NoLintErrors`.
static SmallVector<NoLintBlockToken>
formNoLintBlocks(SmallVector<NoLintToken> NoLints, const SourceManager &SrcMgr,
                 FileID File,
                 SmallVectorImpl<tooling::Diagnostic> &NoLintErrors) {
  SmallVector<NoLintBlockToken> CompletedBlocks;
  SmallVector<NoLintToken> Stack;

  // Nested blocks must be fully contained within their parent block. What this
  // means is that when you have a series of nested BEGIN tokens, the END tokens
  // shall appear in the reverse order, starting with the closing of the
  // inner-most block first, then the next level up, and so on. This is
  // essentially a last-in-first-out/stack system.
  for (NoLintToken &NoLint : NoLints) {
    if (NoLint.Type == NoLintType::NoLintBegin)
      // A new block is being started. Add it to the stack.
      Stack.emplace_back(std::move(NoLint));
    else if (NoLint.Type == NoLintType::NoLintEnd) {
      if (!Stack.empty() && Stack.back().checks() == NoLint.checks()) {
        // The previous block is being closed. Pop one element off the stack.
        CompletedBlocks.emplace_back(Stack.back().Pos, NoLint.Pos,
                                     std::move(Stack.back().ChecksGlob));
        Stack.pop_back();
      } else
        // Trying to close the wrong block.
        NoLintErrors.emplace_back(makeNoLintError(SrcMgr, File, NoLint));
    }
  }

  for (const NoLintToken &NoLint : Stack)
    NoLintErrors.emplace_back(makeNoLintError(SrcMgr, File, NoLint));

  return CompletedBlocks;
}

//===----------------------------------------------------------------------===//
// NoLintDirectiveHandler::Impl
//===----------------------------------------------------------------------===//

class NoLintDirectiveHandler::Impl {
public:
  bool shouldSuppress(DiagnosticsEngine::Level DiagLevel,
                      const Diagnostic &Diag, StringRef DiagName,
                      SmallVectorImpl<tooling::Diagnostic> &NoLintErrors,
                      bool AllowIO, bool EnableNoLintBlocks);

private:
  bool diagHasNoLintInMacro(const Diagnostic &Diag, StringRef DiagName,
                            SmallVectorImpl<tooling::Diagnostic> &NoLintErrors,
                            bool AllowIO, bool EnableNoLintBlocks);

  bool diagHasNoLint(StringRef DiagName, SourceLocation DiagLoc,
                     const SourceManager &SrcMgr,
                     SmallVectorImpl<tooling::Diagnostic> &NoLintErrors,
                     bool AllowIO, bool EnableNoLintBlocks);

  void generateCache(const SourceManager &SrcMgr, StringRef FileName,
                     FileID File, StringRef Buffer,
                     SmallVectorImpl<tooling::Diagnostic> &NoLintErrors);

  llvm::StringMap<SmallVector<NoLintBlockToken>> Cache;
};

bool NoLintDirectiveHandler::Impl::shouldSuppress(
    DiagnosticsEngine::Level DiagLevel, const Diagnostic &Diag,
    StringRef DiagName, SmallVectorImpl<tooling::Diagnostic> &NoLintErrors,
    bool AllowIO, bool EnableNoLintBlocks) {
  if (DiagLevel >= DiagnosticsEngine::Error)
    return false;
  return diagHasNoLintInMacro(Diag, DiagName, NoLintErrors, AllowIO,
                              EnableNoLintBlocks);
}

// Look at the macro's spelling location for a NOLINT. If none is found, keep
// looking up the call stack.
bool NoLintDirectiveHandler::Impl::diagHasNoLintInMacro(
    const Diagnostic &Diag, StringRef DiagName,
    SmallVectorImpl<tooling::Diagnostic> &NoLintErrors, bool AllowIO,
    bool EnableNoLintBlocks) {
  SourceLocation DiagLoc = Diag.getLocation();
  if (DiagLoc.isInvalid())
    return false;
  const SourceManager &SrcMgr = Diag.getSourceManager();
  while (true) {
    if (diagHasNoLint(DiagName, DiagLoc, SrcMgr, NoLintErrors, AllowIO,
                      EnableNoLintBlocks))
      return true;
    if (!DiagLoc.isMacroID())
      return false;
    DiagLoc = SrcMgr.getImmediateExpansionRange(DiagLoc).getBegin();
  }
  return false;
}

// Look behind and ahead for '\n' characters. These mark the start and end of
// this line.
static std::pair<size_t, size_t> getLineStartAndEnd(StringRef Buffer,
                                                    size_t From) {
  size_t StartPos = Buffer.find_last_of('\n', From) + 1;
  size_t EndPos = std::min(Buffer.find('\n', From), Buffer.size());
  return {StartPos, EndPos};
}

// Whether the line has a NOLINT of type = `Type` that can suppress the
// diagnostic `DiagName`.
static bool lineHasNoLint(StringRef Buffer,
                          std::pair<size_t, size_t> LineStartAndEnd,
                          NoLintType Type, StringRef DiagName) {
  // Get all NOLINTs on the line.
  Buffer = Buffer.slice(LineStartAndEnd.first, LineStartAndEnd.second);
  SmallVector<NoLintToken> NoLints = getNoLints(Buffer);

  // Do any of these NOLINTs match the desired type and diag name?
  return llvm::any_of(NoLints, [&](const NoLintToken &NoLint) {
    return NoLint.Type == Type && NoLint.suppresses(DiagName);
  });
}

// Whether the provided diagnostic is located within and is suppressible by a
// block of NOLINT(BEGIN/END) comments.
static bool withinNoLintBlock(ArrayRef<NoLintBlockToken> NoLintBlocks,
                              size_t DiagPos, StringRef DiagName) {
  return llvm::any_of(NoLintBlocks, [&](const NoLintBlockToken &NoLintBlock) {
    return NoLintBlock.suppresses(DiagPos, DiagName);
  });
}

// Get the file contents as a string.
static std::optional<StringRef> getBuffer(const SourceManager &SrcMgr,
                                          FileID File, bool AllowIO) {
  return AllowIO ? SrcMgr.getBufferDataOrNone(File)
                 : SrcMgr.getBufferDataIfLoaded(File);
}

// We will check for NOLINTs and NOLINTNEXTLINEs first. Checking for these is
// not so expensive (just need to parse the current and previous lines). Only if
// that fails do we look for NOLINT(BEGIN/END) blocks (which requires reading
// the entire file).
bool NoLintDirectiveHandler::Impl::diagHasNoLint(
    StringRef DiagName, SourceLocation DiagLoc, const SourceManager &SrcMgr,
    SmallVectorImpl<tooling::Diagnostic> &NoLintErrors, bool AllowIO,
    bool EnableNoLintBlocks) {
  // Translate the diagnostic's SourceLocation to a raw file + offset pair.
  const auto [File, Pos] = SrcMgr.getDecomposedSpellingLoc(DiagLoc);

  // We will only see NOLINTs in user-authored sources. No point reading the
  // file if it is a <built-in>.
  std::optional<StringRef> FileName = SrcMgr.getNonBuiltinFilenameForID(File);
  if (!FileName)
    return false;

  // Get file contents.
  std::optional<StringRef> Buffer = getBuffer(SrcMgr, File, AllowIO);
  if (!Buffer)
    return false;

  // Check if there's a NOLINT on this line.
  auto ThisLine = getLineStartAndEnd(*Buffer, Pos);
  if (lineHasNoLint(*Buffer, ThisLine, NoLintType::NoLint, DiagName))
    return true;

  // Check if there's a NOLINTNEXTLINE on the previous line.
  if (ThisLine.first > 0) {
    auto PrevLine = getLineStartAndEnd(*Buffer, ThisLine.first - 1);
    if (lineHasNoLint(*Buffer, PrevLine, NoLintType::NoLintNextLine, DiagName))
      return true;
  }

  // Check if this line is within a NOLINT(BEGIN/END) block.
  if (!EnableNoLintBlocks)
    return false;

  // Do we have cached NOLINT block locations for this file?
  if (!Cache.contains(*FileName))
    // Warning: heavy operation - need to read entire file.
    generateCache(SrcMgr, *FileName, File, *Buffer, NoLintErrors);

  return withinNoLintBlock(Cache[*FileName], Pos, DiagName);
}

// Find all NOLINT(BEGIN/END) blocks in a file and store in the cache.
void NoLintDirectiveHandler::Impl::generateCache(
    const SourceManager &SrcMgr, StringRef FileName, FileID File,
    StringRef Buffer, SmallVectorImpl<tooling::Diagnostic> &NoLintErrors) {
  // Read entire file to get all NOLINTs and match each BEGIN with its
  // corresponding END, raising errors for any BEGIN or END that is unmatched.
  Cache.try_emplace(FileName, formNoLintBlocks(getNoLints(Buffer), SrcMgr, File,
                                               NoLintErrors));
}

//===----------------------------------------------------------------------===//
// NoLintDirectiveHandler
//===----------------------------------------------------------------------===//

NoLintDirectiveHandler::NoLintDirectiveHandler()
    : PImpl(std::make_unique<Impl>()) {}

NoLintDirectiveHandler::~NoLintDirectiveHandler() = default;

bool NoLintDirectiveHandler::shouldSuppress(
    DiagnosticsEngine::Level DiagLevel, const Diagnostic &Diag,
    StringRef DiagName, SmallVectorImpl<tooling::Diagnostic> &NoLintErrors,
    bool AllowIO, bool EnableNoLintBlocks) {
  return PImpl->shouldSuppress(DiagLevel, Diag, DiagName, NoLintErrors, AllowIO,
                               EnableNoLintBlocks);
}

} // namespace clang::tidy
