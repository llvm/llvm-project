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
#include "NoLintDirectiveHandler.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DiagnosticRenderer.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include <optional>
#include <tuple>
#include <utility>
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
    Message.consume_back(CheckNameInMessage);

    auto TidyMessage =
        Loc.isValid()
            ? tooling::DiagnosticMessage(Message, Loc.getManager(), Loc)
            : tooling::DiagnosticMessage(Message);

    // Make sure that if a TokenRange is received from the check it is unfurled
    // into a real CharRange for the diagnostic printer later.
    // Whatever we store here gets decoupled from the current SourceManager, so
    // we **have to** know the exact position and length of the highlight.
    auto ToCharRange = [this, &Loc](const CharSourceRange &SourceRange) {
      if (SourceRange.isCharRange())
        return SourceRange;
      assert(SourceRange.isTokenRange());
      SourceLocation End = Lexer::getLocForEndOfToken(
          SourceRange.getEnd(), 0, Loc.getManager(), LangOpts);
      return CharSourceRange::getCharRange(SourceRange.getBegin(), End);
    };

    // We are only interested in valid ranges.
    auto ValidRanges =
        llvm::make_filter_range(Ranges, [](const CharSourceRange &R) {
          return R.getAsRange().isValid();
        });

    if (Level == DiagnosticsEngine::Note) {
      Error.Notes.push_back(TidyMessage);
      for (const CharSourceRange &SourceRange : ValidRanges)
        Error.Notes.back().Ranges.emplace_back(Loc.getManager(),
                                               ToCharRange(SourceRange));
      return;
    }
    assert(Error.Message.Message.empty() && "Overwriting a diagnostic message");
    Error.Message = TidyMessage;
    for (const CharSourceRange &SourceRange : ValidRanges)
      Error.Message.Ranges.emplace_back(Loc.getManager(),
                                        ToCharRange(SourceRange));
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

ClangTidyContext::ClangTidyContext(
    std::unique_ptr<ClangTidyOptionsProvider> OptionsProvider,
    bool AllowEnablingAnalyzerAlphaCheckers, bool EnableModuleHeadersParsing)
    : OptionsProvider(std::move(OptionsProvider)),

      AllowEnablingAnalyzerAlphaCheckers(AllowEnablingAnalyzerAlphaCheckers),
      EnableModuleHeadersParsing(EnableModuleHeadersParsing) {
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

DiagnosticBuilder ClangTidyContext::diag(
    StringRef CheckName, StringRef Description,
    DiagnosticIDs::Level Level /* = DiagnosticIDs::Warning*/) {
  unsigned ID = DiagEngine->getDiagnosticIDs()->getCustomDiagID(
      Level, (Description + " [" + CheckName + "]").str());
  CheckNamesByDiagnosticID.try_emplace(ID, CheckName);
  return DiagEngine->Report(ID);
}

DiagnosticBuilder ClangTidyContext::diag(const tooling::Diagnostic &Error) {
  SourceManager &SM = DiagEngine->getSourceManager();
  FileManager &FM = SM.getFileManager();
  FileEntryRef File = llvm::cantFail(FM.getFileRef(Error.Message.FilePath));
  FileID ID = SM.getOrCreateFileID(File, SrcMgr::C_User);
  SourceLocation FileStartLoc = SM.getLocForStartOfFile(ID);
  SourceLocation Loc = FileStartLoc.getLocWithOffset(
      static_cast<SourceLocation::IntTy>(Error.Message.FileOffset));
  return diag(Error.DiagnosticName, Loc, Error.Message.Message,
              static_cast<DiagnosticIDs::Level>(Error.DiagLevel));
}

DiagnosticBuilder ClangTidyContext::configurationDiag(
    StringRef Message,
    DiagnosticIDs::Level Level /* = DiagnosticIDs::Warning*/) {
  return diag("clang-tidy-config", Message, Level);
}

bool ClangTidyContext::shouldSuppressDiagnostic(
    DiagnosticsEngine::Level DiagLevel, const Diagnostic &Info,
    SmallVectorImpl<tooling::Diagnostic> &NoLintErrors, bool AllowIO,
    bool EnableNoLintBlocks) {
  std::string CheckName = getCheckName(Info.getID());
  return NoLintHandler.shouldSuppress(DiagLevel, Info, CheckName, NoLintErrors,
                                      AllowIO, EnableNoLintBlocks);
}

void ClangTidyContext::setSourceManager(SourceManager *SourceMgr) {
  DiagEngine->setSourceManager(SourceMgr);
}

static bool parseFileExtensions(llvm::ArrayRef<std::string> AllFileExtensions,
                                FileExtensionsSet &FileExtensions) {
  FileExtensions.clear();
  for (StringRef Suffix : AllFileExtensions) {
    StringRef Extension = Suffix.trim();
    if (!llvm::all_of(Extension, isAlphanumeric))
      return false;
    FileExtensions.insert(Extension);
  }
  return true;
}

void ClangTidyContext::setCurrentFile(StringRef File) {
  CurrentFile = std::string(File);
  CurrentOptions = getOptionsForFile(CurrentFile);
  CheckFilter = std::make_unique<CachedGlobList>(*getOptions().Checks);
  WarningAsErrorFilter =
      std::make_unique<CachedGlobList>(*getOptions().WarningsAsErrors);
  if (!parseFileExtensions(*getOptions().HeaderFileExtensions,
                           HeaderFileExtensions))
    this->configurationDiag("Invalid header file extensions");
  if (!parseFileExtensions(*getOptions().ImplementationFileExtensions,
                           ImplementationFileExtensions))
    this->configurationDiag("Invalid implementation file extensions");
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
  return ClangTidyOptions::getDefaults().merge(
      OptionsProvider->getOptions(File), 0);
}

void ClangTidyContext::setEnableProfiling(bool P) { Profile = P; }

void ClangTidyContext::setProfileStoragePrefix(StringRef Prefix) {
  ProfilePrefix = std::string(Prefix);
}

std::optional<ClangTidyProfiling::StorageParams>
ClangTidyContext::getProfileStorageParams() const {
  if (ProfilePrefix.empty())
    return std::nullopt;

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
    bool RemoveIncompatibleErrors, bool GetFixesFromNotes,
    bool EnableNolintBlocks)
    : Context(Ctx), ExternalDiagEngine(ExternalDiagEngine),
      RemoveIncompatibleErrors(RemoveIncompatibleErrors),
      GetFixesFromNotes(GetFixesFromNotes),
      EnableNolintBlocks(EnableNolintBlocks) {

  if (Context.getOptions().HeaderFilterRegex &&
      !Context.getOptions().HeaderFilterRegex->empty())
    HeaderFilter =
        std::make_unique<llvm::Regex>(*Context.getOptions().HeaderFilterRegex);

  if (Context.getOptions().ExcludeHeaderFilterRegex &&
      !Context.getOptions().ExcludeHeaderFilterRegex->empty())
    ExcludeHeaderFilter = std::make_unique<llvm::Regex>(
        *Context.getOptions().ExcludeHeaderFilterRegex);
}

void ClangTidyDiagnosticConsumer::finalizeLastError() {
  if (!Errors.empty()) {
    ClangTidyError &Error = Errors.back();
    if (Error.DiagnosticName == "clang-tidy-config") {
      // Never ignore these.
    } else if (!Context.isCheckEnabled(Error.DiagnosticName) &&
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

namespace clang::tidy {

const llvm::StringMap<tooling::Replacements> *
getFixIt(const tooling::Diagnostic &Diagnostic, bool AnyFix) {
  if (!Diagnostic.Message.Fix.empty())
    return &Diagnostic.Message.Fix;
  if (!AnyFix)
    return nullptr;
  const llvm::StringMap<tooling::Replacements> *Result = nullptr;
  for (const auto &Note : Diagnostic.Notes) {
    if (!Note.Fix.empty()) {
      if (Result)
        // We have 2 different fixes in notes, bail out.
        return nullptr;
      Result = &Note.Fix;
    }
  }
  return Result;
}

} // namespace clang::tidy

void ClangTidyDiagnosticConsumer::HandleDiagnostic(
    DiagnosticsEngine::Level DiagLevel, const Diagnostic &Info) {
  if (LastErrorWasIgnored && DiagLevel == DiagnosticsEngine::Note)
    return;

  SmallVector<tooling::Diagnostic, 1> SuppressionErrors;
  if (Context.shouldSuppressDiagnostic(DiagLevel, Info, SuppressionErrors,
                                       EnableNolintBlocks)) {
    ++Context.Stats.ErrorsIgnoredNOLINT;
    // Ignored a warning, should ignore related notes as well
    LastErrorWasIgnored = true;
    for (const auto &Error : SuppressionErrors)
      Context.diag(Error);
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
      case DiagnosticsEngine::Remark:
        CheckName = "clang-diagnostic-remark";
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
    } else if (DiagLevel == DiagnosticsEngine::Remark) {
      Level = ClangTidyError::Remark;
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
    if (Info.hasSourceManager())
      Loc = FullSourceLoc(Info.getLocation(), Info.getSourceManager());
    else if (Context.DiagEngine->hasSourceManager())
      Loc = FullSourceLoc(Info.getLocation(),
                          Context.DiagEngine->getSourceManager());
    Converter.emitDiagnostic(Loc, DiagLevel, Message, Info.getRanges(),
                             Info.getFixItHints());
  }

  if (Info.hasSourceManager())
    checkFilters(Info.getLocation(), Info.getSourceManager());

  for (const auto &Error : SuppressionErrors)
    Context.diag(Error);
}

bool ClangTidyDiagnosticConsumer::passesLineFilter(StringRef FileName,
                                                   unsigned LineNumber) const {
  if (Context.getGlobalOptions().LineFilter.empty())
    return true;
  for (const FileFilter &Filter : Context.getGlobalOptions().LineFilter) {
    if (FileName.ends_with(Filter.Name)) {
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
  for (const FixItHint &Hint : Info.getFixItHints())
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
      (Sources.isInSystemHeader(Location) || Sources.isInSystemMacro(Location)))
    return;

  // FIXME: We start with a conservative approach here, but the actual type of
  // location needed depends on the check (in particular, where this check wants
  // to apply fixes).
  FileID FID = Sources.getDecomposedExpansionLoc(Location).first;
  OptionalFileEntryRef File = Sources.getFileEntryRefForID(FID);

  // -DMACRO definitions on the command line have locations in a virtual buffer
  // that doesn't have a FileEntry. Don't skip these as well.
  if (!File) {
    LastErrorRelatesToUserCode = true;
    LastErrorPassesLineFilter = true;
    return;
  }

  StringRef FileName(File->getName());
  LastErrorRelatesToUserCode =
      LastErrorRelatesToUserCode || Sources.isInMainFile(Location) ||
      (HeaderFilter &&
       (HeaderFilter->match(FileName) &&
        !(ExcludeHeaderFilter && ExcludeHeaderFilter->match(FileName))));

  unsigned LineNumber = Sources.getExpansionLineNumber(Location);
  LastErrorPassesLineFilter =
      LastErrorPassesLineFilter || passesLineFilter(FileName, LineNumber);
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
    if (const auto *Fix = getFixIt(Error, GetFixesFromNotes))
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

  llvm::BitVector Apply(ErrorFixes.size(), true);
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
