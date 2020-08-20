//===--- tools/extra/clang-tidy/ClangTidyDiagnosticConsumer.cpp ----------=== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file This file implements ClangTidyDiagnosticConsumer, ClangTidyContext
///  and ClangTidyError classes.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyOptions.h"
#include "GlobList.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DiagnosticRenderer.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include <tuple>
#include <vector>
using namespace clang;
using namespace tidy;

namespace {
class ClangTidyDiagnosticRenderer : public DiagnosticRenderer {
public:
  ClangTidyDiagnosticRenderer(const LangOptions &LangOpts,
                              DiagnosticOptions *DiagOpts,
                              ClangTidyError &Error)
      : DiagnosticRenderer(LangOpts, DiagOpts), Error(Error) {}

protected:
  void emitDiagnosticMessage(FullSourceLoc Loc, PresumedLoc PLoc,
                             DiagnosticsEngine::Level Level, StringRef Message,
                             ArrayRef<CharSourceRange> Ranges,
                             DiagOrStoredDiag Info) override {
    // Remove check name from the message.
    // FIXME: Remove this once there's a better way to pass check names than
    // appending the check name to the message in ClangTidyContext::diag and
    // using getCustomDiagID.
    std::string CheckNameInMessage = " [" + Error.DiagnosticName + "]";
    if (Message.endswith(CheckNameInMessage))
      Message = Message.substr(0, Message.size() - CheckNameInMessage.size());

    auto TidyMessage =
        Loc.isValid()
            ? tooling::DiagnosticMessage(Message, Loc.getManager(), Loc)
            : tooling::DiagnosticMessage(Message);
    if (Level == DiagnosticsEngine::Note) {
      Error.Notes.push_back(TidyMessage);
      return;
    }
    assert(Error.Message.Message.empty() && "Overwriting a diagnostic message");
    Error.Message = TidyMessage;
    for (const CharSourceRange &SourceRange : Ranges) {
      Error.Ranges.emplace_back(Loc.getManager(), SourceRange);
    }
  }

  void emitDiagnosticLoc(FullSourceLoc Loc, PresumedLoc PLoc,
                         DiagnosticsEngine::Level Level,
                         ArrayRef<CharSourceRange> Ranges) override {}

  void emitCodeContext(FullSourceLoc Loc, DiagnosticsEngine::Level Level,
                       SmallVectorImpl<CharSourceRange> &Ranges,
                       ArrayRef<FixItHint> Hints) override {
    assert(Loc.isValid());
    tooling::DiagnosticMessage *DiagWithFix =
        Level == DiagnosticsEngine::Note ? &Error.Notes.back() : &Error.Message;

    for (const auto &FixIt : Hints) {
      CharSourceRange Range = FixIt.RemoveRange;
      assert(Range.getBegin().isValid() && Range.getEnd().isValid() &&
             "Invalid range in the fix-it hint.");
      assert(Range.getBegin().isFileID() && Range.getEnd().isFileID() &&
             "Only file locations supported in fix-it hints.");

      tooling::Replacement Replacement(Loc.getManager(), Range,
                                       FixIt.CodeToInsert);
      llvm::Error Err =
          DiagWithFix->Fix[Replacement.getFilePath()].add(Replacement);
      // FIXME: better error handling (at least, don't let other replacements be
      // applied).
      if (Err) {
        llvm::errs() << "Fix conflicts with existing fix! "
                     << llvm::toString(std::move(Err)) << "\n";
        assert(false && "Fix conflicts with existing fix!");
      }
    }
  }

  void emitIncludeLocation(FullSourceLoc Loc, PresumedLoc PLoc) override {}

  void emitImportLocation(FullSourceLoc Loc, PresumedLoc PLoc,
                          StringRef ModuleName) override {}

  void emitBuildingModuleLocation(FullSourceLoc Loc, PresumedLoc PLoc,
                                  StringRef ModuleName) override {}

  void endDiagnostic(DiagOrStoredDiag D,
                     DiagnosticsEngine::Level Level) override {
    assert(!Error.Message.Message.empty() && "Message has not been set");
  }

private:
  ClangTidyError &Error;
};
} // end anonymous namespace

ClangTidyError::ClangTidyError(StringRef CheckName,
                               ClangTidyError::Level DiagLevel,
                               StringRef BuildDirectory, bool IsWarningAsError)
    : tooling::Diagnostic(CheckName, DiagLevel, BuildDirectory),
      IsWarningAsError(IsWarningAsError) {}

class ClangTidyContext::CachedGlobList {
public:
  CachedGlobList(StringRef Globs) : Globs(Globs) {}

  bool contains(StringRef S) {
    switch (auto &Result = Cache[S]) {
    case Yes:
      return true;
    case No:
      return false;
    case None:
      Result = Globs.contains(S) ? Yes : No;
      return Result == Yes;
    }
    llvm_unreachable("invalid enum");
  }

private:
  GlobList Globs;
  enum Tristate { None, Yes, No };
  llvm::StringMap<Tristate> Cache;
};

ClangTidyContext::ClangTidyContext(
    std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider,
    bool AllowEnablingAnalyzerAlphaCheckers)
    : DiagEngine(nullptr), OptionsProvider(std::move(OptionsProvider)),
      Profile(false),
      AllowEnablingAnalyzerAlphaCheckers(AllowEnablingAnalyzerAlphaCheckers) {
  // Before the first translation unit we can get errors related to command-line
  // parsing, use empty string for the file name in this case.
  setCurrentFile("");
}

ClangTidyContext::~ClangTidyContext() = default;

DiagnosticBuilder ClangTidyContext::diag(
    StringRef CheckName, SourceLocation Loc, StringRef Description,
    DiagnosticIDs::Level Level /* = DiagnosticIDs::Warning*/) {
  assert(Loc.isValid());
  unsigned ID = DiagEngine->getDiagnosticIDs()->getCustomDiagID(
      Level, (Description + " [" + CheckName + "]").str());
  CheckNamesByDiagnosticID.try_emplace(ID, CheckName);
  return DiagEngine->Report(Loc, ID);
}

void ClangTidyContext::setSourceManager(SourceManager *SourceMgr) {
  DiagEngine->setSourceManager(SourceMgr);
}

void ClangTidyContext::setCurrentFile(StringRef File) {
  CurrentFile = std::string(File);
  CurrentOptions = getOptionsForFile(CurrentFile);
  CheckFilter = std::make_unique<CachedGlobList>(*getOptions().Checks);
  WarningAsErrorFilter =
      std::make_unique<CachedGlobList>(*getOptions().WarningsAsErrors);
}

void ClangTidyContext::setASTContext(ASTContext *Context) {
  DiagEngine->SetArgToStringFn(&FormatASTNodeDiagnosticArgument, Context);
  LangOpts = Context->getLangOpts();
}

const ClangTidyGlobalOptions &ClangTidyContext::getGlobalOptions() const {
  return OptionsProvider->getGlobalOptions();
}

const ClangTidyOptions &ClangTidyContext::getOptions() const {
  return CurrentOptions;
}

ClangTidyOptions ClangTidyContext::getOptionsForFile(StringRef File) const {
  // Merge options on top of getDefaults() as a safeguard against options with
  // unset values.
  return ClangTidyOptions::getDefaults().mergeWith(
      OptionsProvider->getOptions(File), 0);
}

void ClangTidyContext::setEnableProfiling(bool P) { Profile = P; }

void ClangTidyContext::setProfileStoragePrefix(StringRef Prefix) {
  ProfilePrefix = std::string(Prefix);
}

llvm::Optional<ClangTidyProfiling::StorageParams>
ClangTidyContext::getProfileStorageParams() const {
  if (ProfilePrefix.empty())
    return llvm::None;

  return ClangTidyProfiling::StorageParams(ProfilePrefix, CurrentFile);
}

bool ClangTidyContext::isCheckEnabled(StringRef CheckName) const {
  assert(CheckFilter != nullptr);
  return CheckFilter->contains(CheckName);
}

bool ClangTidyContext::treatAsError(StringRef CheckName) const {
  assert(WarningAsErrorFilter != nullptr);
  return WarningAsErrorFilter->contains(CheckName);
}

std::string ClangTidyContext::getCheckName(unsigned DiagnosticID) const {
  std::string ClangWarningOption = std::string(
      DiagEngine->getDiagnosticIDs()->getWarningOptionForDiag(DiagnosticID));
  if (!ClangWarningOption.empty())
    return "clang-diagnostic-" + ClangWarningOption;
  llvm::DenseMap<unsigned, std::string>::const_iterator I =
      CheckNamesByDiagnosticID.find(DiagnosticID);
  if (I != CheckNamesByDiagnosticID.end())
    return I->second;
  return "";
}

ClangTidyDiagnosticConsumer::ClangTidyDiagnosticConsumer(
    ClangTidyContext &Ctx, DiagnosticsEngine *ExternalDiagEngine,
    bool RemoveIncompatibleErrors)
    : Context(Ctx), ExternalDiagEngine(ExternalDiagEngine),
      RemoveIncompatibleErrors(RemoveIncompatibleErrors),
      LastErrorRelatesToUserCode(false), LastErrorPassesLineFilter(false),
      LastErrorWasIgnored(false) {}

void ClangTidyDiagnosticConsumer::finalizeLastError() {
  if (!Errors.empty()) {
    ClangTidyError &Error = Errors.back();
    if (!Context.isCheckEnabled(Error.DiagnosticName) &&
        Error.DiagLevel != ClangTidyError::Error) {
      ++Context.Stats.ErrorsIgnoredCheckFilter;
      Errors.pop_back();
    } else if (!LastErrorRelatesToUserCode) {
      ++Context.Stats.ErrorsIgnoredNonUserCode;
      Errors.pop_back();
    } else if (!LastErrorPassesLineFilter) {
      ++Context.Stats.ErrorsIgnoredLineFilter;
      Errors.pop_back();
    } else {
      ++Context.Stats.ErrorsDisplayed;
    }
  }
  LastErrorRelatesToUserCode = false;
  LastErrorPassesLineFilter = false;
}

static bool IsNOLINTFound(StringRef NolintDirectiveText, StringRef Line,
                          unsigned DiagID, const ClangTidyContext &Context) {
  const size_t NolintIndex = Line.find(NolintDirectiveText);
  if (NolintIndex == StringRef::npos)
    return false;

  size_t BracketIndex = NolintIndex + NolintDirectiveText.size();
  // Check if the specific checks are specified in brackets.
  if (BracketIndex < Line.size() && Line[BracketIndex] == '(') {
    ++BracketIndex;
    const size_t BracketEndIndex = Line.find(')', BracketIndex);
    if (BracketEndIndex != StringRef::npos) {
      StringRef ChecksStr =
          Line.substr(BracketIndex, BracketEndIndex - BracketIndex);
      // Allow disabling all the checks with "*".
      if (ChecksStr != "*") {
        std::string CheckName = Context.getCheckName(DiagID);
        // Allow specifying a few check names, delimited with comma.
        SmallVector<StringRef, 1> Checks;
        ChecksStr.split(Checks, ',', -1, false);
        llvm::transform(Checks, Checks.begin(),
                        [](StringRef S) { return S.trim(); });
        return llvm::find(Checks, CheckName) != Checks.end();
      }
    }
  }
  return true;
}

static llvm::Optional<StringRef> getBuffer(const SourceManager &SM, FileID File,
                                           bool AllowIO) {
  // This is similar to the implementation of SourceManager::getBufferData(),
  // but uses ContentCache::getRawBuffer() rather than getBuffer() if
  // AllowIO=false, to avoid triggering file I/O if the file contents aren't
  // already mapped.
  bool CharDataInvalid = false;
  const SrcMgr::SLocEntry &Entry = SM.getSLocEntry(File, &CharDataInvalid);
  if (CharDataInvalid || !Entry.isFile())
    return llvm::None;
  const SrcMgr::ContentCache *Cache = Entry.getFile().getContentCache();
  const llvm::MemoryBuffer *Buffer =
      AllowIO ? Cache->getBuffer(SM.getDiagnostics(), SM.getFileManager(),
                                 SourceLocation(), &CharDataInvalid)
              : Cache->getRawBuffer();
  if (!Buffer || CharDataInvalid)
    return llvm::None;
  return Buffer->getBuffer();
}

static bool LineIsMarkedWithNOLINT(const SourceManager &SM, SourceLocation Loc,
                                   unsigned DiagID,
                                   const ClangTidyContext &Context,
                                   bool AllowIO) {
  FileID File;
  unsigned Offset;
  std::tie(File, Offset) = SM.getDecomposedSpellingLoc(Loc);
  llvm::Optional<StringRef> Buffer = getBuffer(SM, File, AllowIO);
  if (!Buffer)
    return false;

  // Check if there's a NOLINT on this line.
  StringRef RestOfLine = Buffer->substr(Offset).split('\n').first;
  if (IsNOLINTFound("NOLINT", RestOfLine, DiagID, Context))
    return true;

  // Check if there's a NOLINTNEXTLINE on the previous line.
  StringRef PrevLine =
      Buffer->substr(0, Offset).rsplit('\n').first.rsplit('\n').second;
  return IsNOLINTFound("NOLINTNEXTLINE", PrevLine, DiagID, Context);
}

static bool LineIsMarkedWithNOLINTinMacro(const SourceManager &SM,
                                          SourceLocation Loc, unsigned DiagID,
                                          const ClangTidyContext &Context,
                                          bool AllowIO) {
  while (true) {
    if (LineIsMarkedWithNOLINT(SM, Loc, DiagID, Context, AllowIO))
      return true;
    if (!Loc.isMacroID())
      return false;
    Loc = SM.getImmediateExpansionRange(Loc).getBegin();
  }
  return false;
}

namespace clang {
namespace tidy {

bool shouldSuppressDiagnostic(DiagnosticsEngine::Level DiagLevel,
                              const Diagnostic &Info, ClangTidyContext &Context,
                              bool AllowIO) {
  return Info.getLocation().isValid() &&
         DiagLevel != DiagnosticsEngine::Error &&
         DiagLevel != DiagnosticsEngine::Fatal &&
         LineIsMarkedWithNOLINTinMacro(Info.getSourceManager(),
                                       Info.getLocation(), Info.getID(),
                                       Context, AllowIO);
}

} // namespace tidy
} // namespace clang

void ClangTidyDiagnosticConsumer::HandleDiagnostic(
    DiagnosticsEngine::Level DiagLevel, const Diagnostic &Info) {
  if (LastErrorWasIgnored && DiagLevel == DiagnosticsEngine::Note)
    return;

  if (shouldSuppressDiagnostic(DiagLevel, Info, Context)) {
    ++Context.Stats.ErrorsIgnoredNOLINT;
    // Ignored a warning, should ignore related notes as well
    LastErrorWasIgnored = true;
    return;
  }

  LastErrorWasIgnored = false;
  // Count warnings/errors.
  DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);

  if (DiagLevel == DiagnosticsEngine::Note) {
    assert(!Errors.empty() &&
           "A diagnostic note can only be appended to a message.");
  } else {
    finalizeLastError();
    std::string CheckName = Context.getCheckName(Info.getID());
    if (CheckName.empty()) {
      // This is a compiler diagnostic without a warning option. Assign check
      // name based on its level.
      switch (DiagLevel) {
      case DiagnosticsEngine::Error:
      case DiagnosticsEngine::Fatal:
        CheckName = "clang-diagnostic-error";
        break;
      case DiagnosticsEngine::Warning:
        CheckName = "clang-diagnostic-warning";
        break;
      default:
        CheckName = "clang-diagnostic-unknown";
        break;
      }
    }

    ClangTidyError::Level Level = ClangTidyError::Warning;
    if (DiagLevel == DiagnosticsEngine::Error ||
        DiagLevel == DiagnosticsEngine::Fatal) {
      // Force reporting of Clang errors regardless of filters and non-user
      // code.
      Level = ClangTidyError::Error;
      LastErrorRelatesToUserCode = true;
      LastErrorPassesLineFilter = true;
    }
    bool IsWarningAsError = DiagLevel == DiagnosticsEngine::Warning &&
                            Context.treatAsError(CheckName);
    Errors.emplace_back(CheckName, Level, Context.getCurrentBuildDirectory(),
                        IsWarningAsError);
  }

  if (ExternalDiagEngine) {
    // If there is an external diagnostics engine, like in the
    // ClangTidyPluginAction case, forward the diagnostics to it.
    forwardDiagnostic(Info);
  } else {
    ClangTidyDiagnosticRenderer Converter(
        Context.getLangOpts(), &Context.DiagEngine->getDiagnosticOptions(),
        Errors.back());
    SmallString<100> Message;
    Info.FormatDiagnostic(Message);
    FullSourceLoc Loc;
    if (Info.getLocation().isValid() && Info.hasSourceManager())
      Loc = FullSourceLoc(Info.getLocation(), Info.getSourceManager());
    Converter.emitDiagnostic(Loc, DiagLevel, Message, Info.getRanges(),
                             Info.getFixItHints());
  }

  if (Info.hasSourceManager())
    checkFilters(Info.getLocation(), Info.getSourceManager());
}

bool ClangTidyDiagnosticConsumer::passesLineFilter(StringRef FileName,
                                                   unsigned LineNumber) const {
  if (Context.getGlobalOptions().LineFilter.empty())
    return true;
  for (const FileFilter &Filter : Context.getGlobalOptions().LineFilter) {
    if (FileName.endswith(Filter.Name)) {
      if (Filter.LineRanges.empty())
        return true;
      for (const FileFilter::LineRange &Range : Filter.LineRanges) {
        if (Range.first <= LineNumber && LineNumber <= Range.second)
          return true;
      }
      return false;
    }
  }
  return false;
}

void ClangTidyDiagnosticConsumer::forwardDiagnostic(const Diagnostic &Info) {
  // Acquire a diagnostic ID also in the external diagnostics engine.
  auto DiagLevelAndFormatString =
      Context.getDiagLevelAndFormatString(Info.getID(), Info.getLocation());
  unsigned ExternalID = ExternalDiagEngine->getDiagnosticIDs()->getCustomDiagID(
      DiagLevelAndFormatString.first, DiagLevelAndFormatString.second);

  // Forward the details.
  auto Builder = ExternalDiagEngine->Report(Info.getLocation(), ExternalID);
  for (auto Hint : Info.getFixItHints())
    Builder << Hint;
  for (auto Range : Info.getRanges())
    Builder << Range;
  for (unsigned Index = 0; Index < Info.getNumArgs(); ++Index) {
    DiagnosticsEngine::ArgumentKind Kind = Info.getArgKind(Index);
    switch (Kind) {
    case clang::DiagnosticsEngine::ak_std_string:
      Builder << Info.getArgStdStr(Index);
      break;
    case clang::DiagnosticsEngine::ak_c_string:
      Builder << Info.getArgCStr(Index);
      break;
    case clang::DiagnosticsEngine::ak_sint:
      Builder << Info.getArgSInt(Index);
      break;
    case clang::DiagnosticsEngine::ak_uint:
      Builder << Info.getArgUInt(Index);
      break;
    case clang::DiagnosticsEngine::ak_tokenkind:
      Builder << static_cast<tok::TokenKind>(Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_identifierinfo:
      Builder << Info.getArgIdentifier(Index);
      break;
    case clang::DiagnosticsEngine::ak_qual:
      Builder << Qualifiers::fromOpaqueValue(Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_qualtype:
      Builder << QualType::getFromOpaquePtr((void *)Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_declarationname:
      Builder << DeclarationName::getFromOpaqueInteger(Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_nameddecl:
      Builder << reinterpret_cast<const NamedDecl *>(Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_nestednamespec:
      Builder << reinterpret_cast<NestedNameSpecifier *>(Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_declcontext:
      Builder << reinterpret_cast<DeclContext *>(Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_qualtype_pair:
      assert(false); // This one is not passed around.
      break;
    case clang::DiagnosticsEngine::ak_attr:
      Builder << reinterpret_cast<Attr *>(Info.getRawArg(Index));
      break;
    case clang::DiagnosticsEngine::ak_addrspace:
      Builder << static_cast<LangAS>(Info.getRawArg(Index));
      break;
    }
  }
}

void ClangTidyDiagnosticConsumer::checkFilters(SourceLocation Location,
                                               const SourceManager &Sources) {
  // Invalid location may mean a diagnostic in a command line, don't skip these.
  if (!Location.isValid()) {
    LastErrorRelatesToUserCode = true;
    LastErrorPassesLineFilter = true;
    return;
  }

  if (!*Context.getOptions().SystemHeaders &&
      Sources.isInSystemHeader(Location))
    return;

  // FIXME: We start with a conservative approach here, but the actual type of
  // location needed depends on the check (in particular, where this check wants
  // to apply fixes).
  FileID FID = Sources.getDecomposedExpansionLoc(Location).first;
  const FileEntry *File = Sources.getFileEntryForID(FID);

  // -DMACRO definitions on the command line have locations in a virtual buffer
  // that doesn't have a FileEntry. Don't skip these as well.
  if (!File) {
    LastErrorRelatesToUserCode = true;
    LastErrorPassesLineFilter = true;
    return;
  }

  StringRef FileName(File->getName());
  LastErrorRelatesToUserCode = LastErrorRelatesToUserCode ||
                               Sources.isInMainFile(Location) ||
                               getHeaderFilter()->match(FileName);

  unsigned LineNumber = Sources.getExpansionLineNumber(Location);
  LastErrorPassesLineFilter =
      LastErrorPassesLineFilter || passesLineFilter(FileName, LineNumber);
}

llvm::Regex *ClangTidyDiagnosticConsumer::getHeaderFilter() {
  if (!HeaderFilter)
    HeaderFilter =
        std::make_unique<llvm::Regex>(*Context.getOptions().HeaderFilterRegex);
  return HeaderFilter.get();
}

void ClangTidyDiagnosticConsumer::removeIncompatibleErrors() {
  // Each error is modelled as the set of intervals in which it applies
  // replacements. To detect overlapping replacements, we use a sweep line
  // algorithm over these sets of intervals.
  // An event here consists of the opening or closing of an interval. During the
  // process, we maintain a counter with the amount of open intervals. If we
  // find an endpoint of an interval and this counter is different from 0, it
  // means that this interval overlaps with another one, so we set it as
  // inapplicable.
  struct Event {
    // An event can be either the begin or the end of an interval.
    enum EventType {
      ET_Begin = 1,
      ET_Insert = 0,
      ET_End = -1,
    };

    Event(unsigned Begin, unsigned End, EventType Type, unsigned ErrorId,
          unsigned ErrorSize)
        : Type(Type), ErrorId(ErrorId) {
      // The events are going to be sorted by their position. In case of draw:
      //
      // * If an interval ends at the same position at which other interval
      //   begins, this is not an overlapping, so we want to remove the ending
      //   interval before adding the starting one: end events have higher
      //   priority than begin events.
      //
      // * If we have several begin points at the same position, we will mark as
      //   inapplicable the ones that we process later, so the first one has to
      //   be the one with the latest end point, because this one will contain
      //   all the other intervals. For the same reason, if we have several end
      //   points in the same position, the last one has to be the one with the
      //   earliest begin point. In both cases, we sort non-increasingly by the
      //   position of the complementary.
      //
      // * In case of two equal intervals, the one whose error is bigger can
      //   potentially contain the other one, so we want to process its begin
      //   points before and its end points later.
      //
      // * Finally, if we have two equal intervals whose errors have the same
      //   size, none of them will be strictly contained inside the other.
      //   Sorting by ErrorId will guarantee that the begin point of the first
      //   one will be processed before, disallowing the second one, and the
      //   end point of the first one will also be processed before,
      //   disallowing the first one.
      switch (Type) {
      case ET_Begin:
        Priority = std::make_tuple(Begin, Type, -End, -ErrorSize, ErrorId);
        break;
      case ET_Insert:
        Priority = std::make_tuple(Begin, Type, -End, ErrorSize, ErrorId);
        break;
      case ET_End:
        Priority = std::make_tuple(End, Type, -Begin, ErrorSize, ErrorId);
        break;
      }
    }

    bool operator<(const Event &Other) const {
      return Priority < Other.Priority;
    }

    // Determines if this event is the begin or the end of an interval.
    EventType Type;
    // The index of the error to which the interval that generated this event
    // belongs.
    unsigned ErrorId;
    // The events will be sorted based on this field.
    std::tuple<unsigned, EventType, int, int, unsigned> Priority;
  };

  removeDuplicatedDiagnosticsOfAliasCheckers();

  // Compute error sizes.
  std::vector<int> Sizes;
  std::vector<
      std::pair<ClangTidyError *, llvm::StringMap<tooling::Replacements> *>>
      ErrorFixes;
  for (auto &Error : Errors) {
    if (const auto *Fix = tooling::selectFirstFix(Error))
      ErrorFixes.emplace_back(
          &Error, const_cast<llvm::StringMap<tooling::Replacements> *>(Fix));
  }
  for (const auto &ErrorAndFix : ErrorFixes) {
    int Size = 0;
    for (const auto &FileAndReplaces : *ErrorAndFix.second) {
      for (const auto &Replace : FileAndReplaces.second)
        Size += Replace.getLength();
    }
    Sizes.push_back(Size);
  }

  // Build events from error intervals.
  llvm::StringMap<std::vector<Event>> FileEvents;
  for (unsigned I = 0; I < ErrorFixes.size(); ++I) {
    for (const auto &FileAndReplace : *ErrorFixes[I].second) {
      for (const auto &Replace : FileAndReplace.second) {
        unsigned Begin = Replace.getOffset();
        unsigned End = Begin + Replace.getLength();
        auto &Events = FileEvents[Replace.getFilePath()];
        if (Begin == End) {
          Events.emplace_back(Begin, End, Event::ET_Insert, I, Sizes[I]);
        } else {
          Events.emplace_back(Begin, End, Event::ET_Begin, I, Sizes[I]);
          Events.emplace_back(Begin, End, Event::ET_End, I, Sizes[I]);
        }
      }
    }
  }

  std::vector<bool> Apply(ErrorFixes.size(), true);
  for (auto &FileAndEvents : FileEvents) {
    std::vector<Event> &Events = FileAndEvents.second;
    // Sweep.
    llvm::sort(Events);
    int OpenIntervals = 0;
    for (const auto &Event : Events) {
      switch (Event.Type) {
      case Event::ET_Begin:
        if (OpenIntervals++ != 0)
          Apply[Event.ErrorId] = false;
        break;
      case Event::ET_Insert:
        if (OpenIntervals != 0)
          Apply[Event.ErrorId] = false;
        break;
      case Event::ET_End:
        if (--OpenIntervals != 0)
          Apply[Event.ErrorId] = false;
        break;
      }
    }
    assert(OpenIntervals == 0 && "Amount of begin/end points doesn't match");
  }

  for (unsigned I = 0; I < ErrorFixes.size(); ++I) {
    if (!Apply[I]) {
      ErrorFixes[I].second->clear();
      ErrorFixes[I].first->Notes.emplace_back(
          "this fix will not be applied because it overlaps with another fix");
    }
  }
}

namespace {
struct LessClangTidyError {
  bool operator()(const ClangTidyError &LHS, const ClangTidyError &RHS) const {
    const tooling::DiagnosticMessage &M1 = LHS.Message;
    const tooling::DiagnosticMessage &M2 = RHS.Message;

    return std::tie(M1.FilePath, M1.FileOffset, LHS.DiagnosticName,
                    M1.Message) <
           std::tie(M2.FilePath, M2.FileOffset, RHS.DiagnosticName, M2.Message);
  }
};
struct EqualClangTidyError {
  bool operator()(const ClangTidyError &LHS, const ClangTidyError &RHS) const {
    LessClangTidyError Less;
    return !Less(LHS, RHS) && !Less(RHS, LHS);
  }
};
} // end anonymous namespace

std::vector<ClangTidyError> ClangTidyDiagnosticConsumer::take() {
  finalizeLastError();

  llvm::stable_sort(Errors, LessClangTidyError());
  Errors.erase(std::unique(Errors.begin(), Errors.end(), EqualClangTidyError()),
               Errors.end());
  if (RemoveIncompatibleErrors)
    removeIncompatibleErrors();
  return std::move(Errors);
}

namespace {
struct LessClangTidyErrorWithoutDiagnosticName {
  bool operator()(const ClangTidyError *LHS, const ClangTidyError *RHS) const {
    const tooling::DiagnosticMessage &M1 = LHS->Message;
    const tooling::DiagnosticMessage &M2 = RHS->Message;

    return std::tie(M1.FilePath, M1.FileOffset, M1.Message) <
           std::tie(M2.FilePath, M2.FileOffset, M2.Message);
  }
};
} // end anonymous namespace

void ClangTidyDiagnosticConsumer::removeDuplicatedDiagnosticsOfAliasCheckers() {
  using UniqueErrorSet =
      std::set<ClangTidyError *, LessClangTidyErrorWithoutDiagnosticName>;
  UniqueErrorSet UniqueErrors;

  auto IT = Errors.begin();
  while (IT != Errors.end()) {
    ClangTidyError &Error = *IT;
    std::pair<UniqueErrorSet::iterator, bool> Inserted =
        UniqueErrors.insert(&Error);

    // Unique error, we keep it and move along.
    if (Inserted.second) {
      ++IT;
    } else {
      ClangTidyError &ExistingError = **Inserted.first;
      const llvm::StringMap<tooling::Replacements> &CandidateFix =
          Error.Message.Fix;
      const llvm::StringMap<tooling::Replacements> &ExistingFix =
          (*Inserted.first)->Message.Fix;

      if (CandidateFix != ExistingFix) {

        // In case of a conflict, don't suggest any fix-it.
        ExistingError.Message.Fix.clear();
        ExistingError.Notes.emplace_back(
            llvm::formatv("cannot apply fix-it because an alias checker has "
                          "suggested a different fix-it; please remove one of "
                          "the checkers ('{0}', '{1}') or "
                          "ensure they are both configured the same",
                          ExistingError.DiagnosticName, Error.DiagnosticName)
                .str());
      }

      if (Error.IsWarningAsError)
        ExistingError.IsWarningAsError = true;

      // Since it is the same error, we should take it as alias and remove it.
      ExistingError.EnabledDiagnosticAliases.emplace_back(Error.DiagnosticName);
      IT = Errors.erase(IT);
    }
  }
}
