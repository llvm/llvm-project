//===--- ParsedAST.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ParsedAST.h"
#include "../clang-tidy/ClangTidyCheck.h"
#include "../clang-tidy/ClangTidyDiagnosticConsumer.h"
#include "../clang-tidy/ClangTidyModuleRegistry.h"
#include "../clang-tidy/NoLintDirectiveHandler.h"
#include "AST.h"
#include "Compiler.h"
#include "Config.h"
#include "Diagnostics.h"
#include "Feature.h"
#include "FeatureModule.h"
#include "Headers.h"
#include "HeuristicResolver.h"
#include "IncludeCleaner.h"
#include "IncludeFixer.h"
#include "Preamble.h"
#include "SourceCode.h"
#include "TidyProvider.h"
#include "index/CanonicalIncludes.h"
#include "index/Index.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Serialization/ASTWriter.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <memory>
#include <vector>

// Force the linker to link in Clang-tidy modules.
// clangd doesn't support the static analyzer.
#if CLANGD_TIDY_CHECKS
#define CLANG_TIDY_DISABLE_STATIC_ANALYZER_CHECKS
#include "../clang-tidy/ClangTidyForceLinker.h"
#endif

namespace clang {
namespace clangd {
namespace {

template <class T> std::size_t getUsedBytes(const std::vector<T> &Vec) {
  return Vec.capacity() * sizeof(T);
}

class DeclTrackingASTConsumer : public ASTConsumer {
public:
  DeclTrackingASTConsumer(std::vector<Decl *> &TopLevelDecls)
      : TopLevelDecls(TopLevelDecls) {}

  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    for (Decl *D : DG) {
      auto &SM = D->getASTContext().getSourceManager();
      if (!isInsideMainFile(D->getLocation(), SM))
        continue;
      if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
        if (isImplicitTemplateInstantiation(ND))
          continue;

      // ObjCMethodDecl are not actually top-level decls.
      if (isa<ObjCMethodDecl>(D))
        continue;

      TopLevelDecls.push_back(D);
    }
    return true;
  }

private:
  std::vector<Decl *> &TopLevelDecls;
};

class ClangdFrontendAction : public SyntaxOnlyAction {
public:
  std::vector<Decl *> takeTopLevelDecls() { return std::move(TopLevelDecls); }

protected:
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, llvm::StringRef InFile) override {
    return std::make_unique<DeclTrackingASTConsumer>(/*ref*/ TopLevelDecls);
  }

private:
  std::vector<Decl *> TopLevelDecls;
};

// When using a preamble, only preprocessor events outside its bounds are seen.
// This is almost what we want: replaying transitive preprocessing wastes time.
// However this confuses clang-tidy checks: they don't see any #includes!
// So we replay the *non-transitive* #includes that appear in the main-file.
// It would be nice to replay other events (macro definitions, ifdefs etc) but
// this addresses the most common cases fairly cheaply.
class ReplayPreamble : private PPCallbacks {
public:
  // Attach preprocessor hooks such that preamble events will be injected at
  // the appropriate time.
  // Events will be delivered to the *currently registered* PP callbacks.
  static void attach(std::vector<Inclusion> Includes, CompilerInstance &Clang,
                     const PreambleBounds &PB) {
    auto &PP = Clang.getPreprocessor();
    auto *ExistingCallbacks = PP.getPPCallbacks();
    // No need to replay events if nobody is listening.
    if (!ExistingCallbacks)
      return;
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(new ReplayPreamble(
        std::move(Includes), ExistingCallbacks, Clang.getSourceManager(), PP,
        Clang.getLangOpts(), PB)));
    // We're relying on the fact that addPPCallbacks keeps the old PPCallbacks
    // around, creating a chaining wrapper. Guard against other implementations.
    assert(PP.getPPCallbacks() != ExistingCallbacks &&
           "Expected chaining implementation");
  }

private:
  ReplayPreamble(std::vector<Inclusion> Includes, PPCallbacks *Delegate,
                 const SourceManager &SM, Preprocessor &PP,
                 const LangOptions &LangOpts, const PreambleBounds &PB)
      : Includes(std::move(Includes)), Delegate(Delegate), SM(SM), PP(PP) {
    // Only tokenize the preamble section of the main file, as we are not
    // interested in the rest of the tokens.
    MainFileTokens = syntax::tokenize(
        syntax::FileRange(SM.getMainFileID(), 0, PB.Size), SM, LangOpts);
  }

  // In a normal compile, the preamble traverses the following structure:
  //
  // mainfile.cpp
  //   <built-in>
  //     ... macro definitions like __cplusplus ...
  //     <command-line>
  //       ... macro definitions for args like -Dfoo=bar ...
  //   "header1.h"
  //     ... header file contents ...
  //   "header2.h"
  //     ... header file contents ...
  //   ... main file contents ...
  //
  // When using a preamble, the "header1" and "header2" subtrees get skipped.
  // We insert them right after the built-in header, which still appears.
  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind Kind, FileID PrevFID) override {
    // It'd be nice if there was a better way to identify built-in headers...
    if (Reason == FileChangeReason::ExitFile &&
        SM.getBufferOrFake(PrevFID).getBufferIdentifier() == "<built-in>")
      replay();
  }

  void replay() {
    for (const auto &Inc : Includes) {
      llvm::Optional<FileEntryRef> File;
      if (Inc.Resolved != "")
        File = expectedToOptional(SM.getFileManager().getFileRef(Inc.Resolved));

      // Re-lex the #include directive to find its interesting parts.
      auto HashLoc = SM.getComposedLoc(SM.getMainFileID(), Inc.HashOffset);
      auto HashTok = llvm::partition_point(MainFileTokens,
                                           [&HashLoc](const syntax::Token &T) {
                                             return T.location() < HashLoc;
                                           });
      assert(HashTok != MainFileTokens.end() && HashTok->kind() == tok::hash);

      auto IncludeTok = std::next(HashTok);
      assert(IncludeTok != MainFileTokens.end());

      auto FileTok = std::next(IncludeTok);
      assert(FileTok != MainFileTokens.end());

      // Create a fake import/include token, none of the callers seem to care
      // about clang::Token::Flags.
      Token SynthesizedIncludeTok;
      SynthesizedIncludeTok.startToken();
      SynthesizedIncludeTok.setLocation(IncludeTok->location());
      SynthesizedIncludeTok.setLength(IncludeTok->length());
      SynthesizedIncludeTok.setKind(tok::raw_identifier);
      SynthesizedIncludeTok.setRawIdentifierData(IncludeTok->text(SM).data());
      PP.LookUpIdentifierInfo(SynthesizedIncludeTok);

      // Same here, create a fake one for Filename, including angles or quotes.
      Token SynthesizedFilenameTok;
      SynthesizedFilenameTok.startToken();
      SynthesizedFilenameTok.setLocation(FileTok->location());
      // Note that we can't make use of FileTok->length/text in here as in the
      // case of angled includes this will contain tok::less instead of
      // filename. Whereas Inc.Written contains the full header name including
      // quotes/angles.
      SynthesizedFilenameTok.setLength(Inc.Written.length());
      SynthesizedFilenameTok.setKind(tok::header_name);
      SynthesizedFilenameTok.setLiteralData(Inc.Written.data());

      const FileEntry *FE = File ? &File->getFileEntry() : nullptr;
      llvm::StringRef WrittenFilename =
          llvm::StringRef(Inc.Written).drop_front().drop_back();
      Delegate->InclusionDirective(
          HashTok->location(), SynthesizedIncludeTok, WrittenFilename,
          Inc.Written.front() == '<',
          syntax::FileRange(SM, SynthesizedFilenameTok.getLocation(),
                            SynthesizedFilenameTok.getEndLoc())
              .toCharRange(SM),
          FE, "SearchPath", "RelPath",
          /*Imported=*/nullptr, Inc.FileKind);
      if (File)
        Delegate->FileSkipped(*File, SynthesizedFilenameTok, Inc.FileKind);
    }
  }

  const std::vector<Inclusion> Includes;
  PPCallbacks *Delegate;
  const SourceManager &SM;
  Preprocessor &PP;
  std::vector<syntax::Token> MainFileTokens;
};

// Find -W<group> and -Wno-<group> options in ExtraArgs and apply them to Diags.
//
// This is used to handle ExtraArgs in clang-tidy configuration.
// We don't use clang's standard handling of this as we want slightly different
// behavior (e.g. we want to exclude these from -Wno-error).
void applyWarningOptions(llvm::ArrayRef<std::string> ExtraArgs,
                         DiagnosticsEngine &Diags) {
  for (llvm::StringRef Group : ExtraArgs) {
    // Only handle args that are of the form -W[no-]<group>.
    // Other flags are possible but rare and deliberately out of scope.
    llvm::SmallVector<diag::kind> Members;
    if (!Group.consume_front("-W") || Group.empty())
      continue;
    bool Enable = !Group.consume_front("no-");
    if (Diags.getDiagnosticIDs()->getDiagnosticsInGroup(
            diag::Flavor::WarningOrError, Group, Members))
      continue;

    // Upgrade (or downgrade) the severity of each diagnostic in the group.
    // If -Werror is on, newly added warnings will be treated as errors.
    // We don't want this, so keep track of them to fix afterwards.
    bool NeedsWerrorExclusion = false;
    for (diag::kind ID : Members) {
      if (Enable) {
        if (Diags.getDiagnosticLevel(ID, SourceLocation()) <
            DiagnosticsEngine::Warning) {
          Diags.setSeverity(ID, diag::Severity::Warning, SourceLocation());
          if (Diags.getWarningsAsErrors())
            NeedsWerrorExclusion = true;
        }
      } else {
        Diags.setSeverity(ID, diag::Severity::Ignored, SourceLocation());
      }
    }
    if (NeedsWerrorExclusion) {
      // FIXME: there's no API to suppress -Werror for single diagnostics.
      // In some cases with sub-groups, we may end up erroneously
      // downgrading diagnostics that were -Werror in the compile command.
      Diags.setDiagnosticGroupWarningAsError(Group, false);
    }
  }
}

} // namespace

llvm::Optional<ParsedAST>
ParsedAST::build(llvm::StringRef Filename, const ParseInputs &Inputs,
                 std::unique_ptr<clang::CompilerInvocation> CI,
                 llvm::ArrayRef<Diag> CompilerInvocationDiags,
                 std::shared_ptr<const PreambleData> Preamble) {
  trace::Span Tracer("BuildAST");
  SPAN_ATTACH(Tracer, "File", Filename);

  auto VFS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  if (Preamble && Preamble->StatCache)
    VFS = Preamble->StatCache->getConsumingFS(std::move(VFS));

  assert(CI);
  // Command-line parsing sets DisableFree to true by default, but we don't want
  // to leak memory in clangd.
  CI->getFrontendOpts().DisableFree = false;
  const PrecompiledPreamble *PreamblePCH =
      Preamble ? &Preamble->Preamble : nullptr;

  // This is on-by-default in windows to allow parsing SDK headers, but it
  // breaks many features. Disable it for the main-file (not preamble).
  CI->getLangOpts()->DelayedTemplateParsing = false;

  std::vector<std::unique_ptr<FeatureModule::ASTListener>> ASTListeners;
  if (Inputs.FeatureModules) {
    for (auto &M : *Inputs.FeatureModules) {
      if (auto Listener = M.astListeners())
        ASTListeners.emplace_back(std::move(Listener));
    }
  }
  StoreDiags ASTDiags;
  ASTDiags.setDiagCallback(
      [&ASTListeners](const clang::Diagnostic &D, clangd::Diag &Diag) {
        llvm::for_each(ASTListeners,
                       [&](const auto &L) { L->sawDiagnostic(D, Diag); });
      });

  llvm::Optional<PreamblePatch> Patch;
  bool PreserveDiags = true;
  // We might use an ignoring diagnostic consumer if they are going to be
  // dropped later on to not pay for extra latency by processing them.
  DiagnosticConsumer *DiagConsumer = &ASTDiags;
  IgnoreDiagnostics DropDiags;
  if (Preamble) {
    Patch = PreamblePatch::createFullPatch(Filename, Inputs, *Preamble);
    Patch->apply(*CI);
    PreserveDiags = Patch->preserveDiagnostics();
    if (!PreserveDiags)
      DiagConsumer = &DropDiags;
  }
  auto Clang = prepareCompilerInstance(
      std::move(CI), PreamblePCH,
      llvm::MemoryBuffer::getMemBufferCopy(Inputs.Contents, Filename), VFS,
      *DiagConsumer);
  if (!Clang) {
    // The last diagnostic contains information about the reason of this
    // failure.
    std::vector<Diag> Diags(ASTDiags.take());
    elog("Failed to prepare a compiler instance: {0}",
         !Diags.empty() ? static_cast<DiagBase &>(Diags.back()).Message
                        : "unknown error");
    return None;
  }
  tidy::ClangTidyOptions ClangTidyOpts;
  if (PreserveDiags) {
    trace::Span Tracer("ClangTidyOpts");
    ClangTidyOpts = getTidyOptionsForFile(Inputs.ClangTidyProvider, Filename);
    dlog("ClangTidy configuration for file {0}: {1}", Filename,
         tidy::configurationAsText(ClangTidyOpts));

    // If clang-tidy is configured to emit clang warnings, we should too.
    //
    // Such clang-tidy configuration consists of two parts:
    //   - ExtraArgs: ["-Wfoo"] causes clang to produce the warnings
    //   - Checks: "clang-diagnostic-foo" prevents clang-tidy filtering them out
    //
    // We treat these as clang warnings, so the Checks part is not relevant.
    // We must enable the warnings specified in ExtraArgs.
    //
    // We *don't* want to change the compile command directly. this can have
    // too many unexpected effects: breaking the command, interactions with
    // -- and -Werror, etc. Besides, we've already parsed the command.
    // Instead we parse the -W<group> flags and handle them directly.
    auto &Diags = Clang->getDiagnostics();
    if (ClangTidyOpts.ExtraArgsBefore)
      applyWarningOptions(*ClangTidyOpts.ExtraArgsBefore, Diags);
    if (ClangTidyOpts.ExtraArgs)
      applyWarningOptions(*ClangTidyOpts.ExtraArgs, Diags);
  } else {
    // Skips some analysis.
    Clang->getDiagnosticOpts().IgnoreWarnings = true;
  }

  auto Action = std::make_unique<ClangdFrontendAction>();
  const FrontendInputFile &MainInput = Clang->getFrontendOpts().Inputs[0];
  if (!Action->BeginSourceFile(*Clang, MainInput)) {
    log("BeginSourceFile() failed when building AST for {0}",
        MainInput.getFile());
    return None;
  }
  // If we saw an include guard in the preamble section of the main file,
  // mark the main-file as include-guarded.
  // This information is part of the HeaderFileInfo but is not loaded from the
  // preamble as the file's size is part of its identity and may have changed.
  // (The rest of HeaderFileInfo is not relevant for our purposes).
  if (Preamble && Preamble->MainIsIncludeGuarded) {
    const SourceManager &SM = Clang->getSourceManager();
    const FileEntry *MainFE = SM.getFileEntryForID(SM.getMainFileID());
    Clang->getPreprocessor().getHeaderSearchInfo().MarkFileIncludeOnce(MainFE);
  }

  // Set up ClangTidy. Must happen after BeginSourceFile() so ASTContext exists.
  // Clang-tidy has some limitations to ensure reasonable performance:
  //  - checks don't see all preprocessor events in the preamble
  //  - matchers run only over the main-file top-level decls (and can't see
  //    ancestors outside this scope).
  // In practice almost all checks work well without modifications.
  std::vector<std::unique_ptr<tidy::ClangTidyCheck>> CTChecks;
  ast_matchers::MatchFinder CTFinder;
  llvm::Optional<tidy::ClangTidyContext> CTContext;
  llvm::Optional<IncludeFixer> FixIncludes;
  llvm::DenseMap<diag::kind, DiagnosticsEngine::Level> OverriddenSeverity;
  // No need to run clang-tidy or IncludeFixerif we are not going to surface
  // diagnostics.
  if (PreserveDiags) {
    trace::Span Tracer("ClangTidyInit");
    tidy::ClangTidyCheckFactories CTFactories;
    for (const auto &E : tidy::ClangTidyModuleRegistry::entries())
      E.instantiate()->addCheckFactories(CTFactories);
    CTContext.emplace(std::make_unique<tidy::DefaultOptionsProvider>(
        tidy::ClangTidyGlobalOptions(), ClangTidyOpts));
    CTContext->setDiagnosticsEngine(&Clang->getDiagnostics());
    CTContext->setASTContext(&Clang->getASTContext());
    CTContext->setCurrentFile(Filename);
    CTChecks = CTFactories.createChecks(CTContext.getPointer());
    llvm::erase_if(CTChecks, [&](const auto &Check) {
      return !Check->isLanguageVersionSupported(CTContext->getLangOpts());
    });
    Preprocessor *PP = &Clang->getPreprocessor();
    for (const auto &Check : CTChecks) {
      Check->registerPPCallbacks(Clang->getSourceManager(), PP, PP);
      Check->registerMatchers(&CTFinder);
    }

    // Clang only corrects typos for use of undeclared functions in C if that
    // use is an error. Include fixer relies on typo correction, so pretend
    // this is an error. (The actual typo correction is nice too).
    // We restore the original severity in the level adjuster.
    // FIXME: It would be better to have a real API for this, but what?
    for (auto ID : {diag::ext_implicit_function_decl,
                    diag::warn_implicit_function_decl}) {
      OverriddenSeverity.try_emplace(
          ID, Clang->getDiagnostics().getDiagnosticLevel(ID, SourceLocation()));
      Clang->getDiagnostics().setSeverity(ID, diag::Severity::Error,
                                          SourceLocation());
    }

    const Config &Cfg = Config::current();
    ASTDiags.setLevelAdjuster([&](DiagnosticsEngine::Level DiagLevel,
                                  const clang::Diagnostic &Info) {
      if (Cfg.Diagnostics.SuppressAll ||
          isBuiltinDiagnosticSuppressed(Info.getID(), Cfg.Diagnostics.Suppress,
                                        Clang->getLangOpts()))
        return DiagnosticsEngine::Ignored;

      auto It = OverriddenSeverity.find(Info.getID());
      if (It != OverriddenSeverity.end())
        DiagLevel = It->second;

      if (!CTChecks.empty()) {
        std::string CheckName = CTContext->getCheckName(Info.getID());
        bool IsClangTidyDiag = !CheckName.empty();
        if (IsClangTidyDiag) {
          if (Cfg.Diagnostics.Suppress.contains(CheckName))
            return DiagnosticsEngine::Ignored;
          // Check for suppression comment. Skip the check for diagnostics not
          // in the main file, because we don't want that function to query the
          // source buffer for preamble files. For the same reason, we ask
          // shouldSuppressDiagnostic to avoid I/O.
          // We let suppression comments take precedence over warning-as-error
          // to match clang-tidy's behaviour.
          bool IsInsideMainFile =
              Info.hasSourceManager() &&
              isInsideMainFile(Info.getLocation(), Info.getSourceManager());
          SmallVector<tooling::Diagnostic, 1> TidySuppressedErrors;
          if (IsInsideMainFile && CTContext->shouldSuppressDiagnostic(
                                      DiagLevel, Info, TidySuppressedErrors,
                                      /*AllowIO=*/false,
                                      /*EnableNolintBlocks=*/false)) {
            // FIXME: should we expose the suppression error (invalid use of
            // NOLINT comments)?
            return DiagnosticsEngine::Ignored;
          }

          // Check for warning-as-error.
          if (DiagLevel == DiagnosticsEngine::Warning &&
              CTContext->treatAsError(CheckName)) {
            return DiagnosticsEngine::Error;
          }
        }
      }
      return DiagLevel;
    });

    // Add IncludeFixer which can recover diagnostics caused by missing includes
    // (e.g. incomplete type) and attach include insertion fixes to diagnostics.
    auto BuildDir = VFS->getCurrentWorkingDirectory();
    if (Inputs.Index && !BuildDir.getError()) {
      auto Style =
          getFormatStyleForFile(Filename, Inputs.Contents, *Inputs.TFS);
      auto Inserter = std::make_shared<IncludeInserter>(
          Filename, Inputs.Contents, Style, BuildDir.get(),
          &Clang->getPreprocessor().getHeaderSearchInfo());
      if (Preamble) {
        for (const auto &Inc : Preamble->Includes.MainFileIncludes)
          Inserter->addExisting(Inc);
      }
      FixIncludes.emplace(Filename, Inserter, *Inputs.Index,
                          /*IndexRequestLimit=*/5);
      ASTDiags.contributeFixes([&FixIncludes](DiagnosticsEngine::Level DiagLevl,
                                              const clang::Diagnostic &Info) {
        return FixIncludes->fix(DiagLevl, Info);
      });
      Clang->setExternalSemaSource(FixIncludes->unresolvedNameRecorder());
    }
  }

  IncludeStructure Includes;
  // If we are using a preamble, copy existing includes.
  if (Preamble) {
    Includes = Preamble->Includes;
    Includes.MainFileIncludes = Patch->preambleIncludes();
    // Replay the preamble includes so that clang-tidy checks can see them.
    ReplayPreamble::attach(Patch->preambleIncludes(), *Clang,
                           Patch->modifiedBounds());
  }
  // Important: collectIncludeStructure is registered *after* ReplayPreamble!
  // Otherwise we would collect the replayed includes again...
  // (We can't *just* use the replayed includes, they don't have Resolved path).
  Includes.collect(*Clang);
  // Copy over the macros in the preamble region of the main file, and combine
  // with non-preamble macros below.
  MainFileMacros Macros;
  if (Preamble)
    Macros = Preamble->Macros;
  Clang->getPreprocessor().addPPCallbacks(
      std::make_unique<CollectMainFileMacros>(Clang->getSourceManager(),
                                              Macros));

  std::vector<PragmaMark> Marks;
  // FIXME: We need to patch the marks for stale preambles.
  if (Preamble)
    Marks = Preamble->Marks;
  Clang->getPreprocessor().addPPCallbacks(
      collectPragmaMarksCallback(Clang->getSourceManager(), Marks));

  // Copy over the includes from the preamble, then combine with the
  // non-preamble includes below.
  CanonicalIncludes CanonIncludes;
  if (Preamble)
    CanonIncludes = Preamble->CanonIncludes;
  else
    CanonIncludes.addSystemHeadersMapping(Clang->getLangOpts());
  std::unique_ptr<CommentHandler> IWYUHandler =
      collectIWYUHeaderMaps(&CanonIncludes);
  Clang->getPreprocessor().addCommentHandler(IWYUHandler.get());

  // Collect tokens of the main file.
  syntax::TokenCollector CollectTokens(Clang->getPreprocessor());

  if (llvm::Error Err = Action->Execute())
    log("Execute() failed when building AST for {0}: {1}", MainInput.getFile(),
        toString(std::move(Err)));

  // We have to consume the tokens before running clang-tidy to avoid collecting
  // tokens from running the preprocessor inside the checks (only
  // modernize-use-trailing-return-type does that today).
  syntax::TokenBuffer Tokens = std::move(CollectTokens).consume();
  // Makes SelectionTree build much faster.
  Tokens.indexExpandedTokens();
  std::vector<Decl *> ParsedDecls = Action->takeTopLevelDecls();
  // AST traversals should exclude the preamble, to avoid performance cliffs.
  Clang->getASTContext().setTraversalScope(ParsedDecls);
  if (!CTChecks.empty()) {
    // Run the AST-dependent part of the clang-tidy checks.
    // (The preprocessor part ran already, via PPCallbacks).
    trace::Span Tracer("ClangTidyMatch");
    CTFinder.matchAST(Clang->getASTContext());
  }

  // XXX: This is messy: clang-tidy checks flush some diagnostics at EOF.
  // However Action->EndSourceFile() would destroy the ASTContext!
  // So just inform the preprocessor of EOF, while keeping everything alive.
  Clang->getPreprocessor().EndSourceFile();
  // UnitDiagsConsumer is local, we can not store it in CompilerInstance that
  // has a longer lifetime.
  Clang->getDiagnostics().setClient(new IgnoreDiagnostics);
  // CompilerInstance won't run this callback, do it directly.
  ASTDiags.EndSourceFile();

  llvm::Optional<std::vector<Diag>> Diags;
  // FIXME: Also skip generation of diagnostics alltogether to speed up ast
  // builds when we are patching a stale preamble.
  if (PreserveDiags) {
    Diags = CompilerInvocationDiags;
    // Add diagnostics from the preamble, if any.
    if (Preamble)
      Diags->insert(Diags->end(), Preamble->Diags.begin(),
                    Preamble->Diags.end());
    // Finally, add diagnostics coming from the AST.
    {
      std::vector<Diag> D = ASTDiags.take(CTContext.getPointer());
      Diags->insert(Diags->end(), D.begin(), D.end());
    }
  }
  ParsedAST Result(Inputs.Version, std::move(Preamble), std::move(Clang),
                   std::move(Action), std::move(Tokens), std::move(Macros),
                   std::move(Marks), std::move(ParsedDecls), std::move(Diags),
                   std::move(Includes), std::move(CanonIncludes));
  if (Result.Diags) {
    auto UnusedHeadersDiags =
        issueUnusedIncludesDiagnostics(Result, Inputs.Contents);
    Result.Diags->insert(Result.Diags->end(),
                         make_move_iterator(UnusedHeadersDiags.begin()),
                         make_move_iterator(UnusedHeadersDiags.end()));
  }
  return Result;
}

ParsedAST::ParsedAST(ParsedAST &&Other) = default;

ParsedAST &ParsedAST::operator=(ParsedAST &&Other) = default;

ParsedAST::~ParsedAST() {
  if (Action) {
    // We already notified the PP of end-of-file earlier, so detach it first.
    // We must keep it alive until after EndSourceFile(), Sema relies on this.
    auto PP = Clang->getPreprocessorPtr(); // Keep PP alive for now.
    Clang->setPreprocessor(nullptr);       // Detach so we don't send EOF again.
    Action->EndSourceFile();               // Destroy ASTContext and Sema.
    // Now Sema is gone, it's safe for PP to go out of scope.
  }
}

ASTContext &ParsedAST::getASTContext() { return Clang->getASTContext(); }

const ASTContext &ParsedAST::getASTContext() const {
  return Clang->getASTContext();
}

Preprocessor &ParsedAST::getPreprocessor() { return Clang->getPreprocessor(); }

std::shared_ptr<Preprocessor> ParsedAST::getPreprocessorPtr() {
  return Clang->getPreprocessorPtr();
}

const Preprocessor &ParsedAST::getPreprocessor() const {
  return Clang->getPreprocessor();
}

llvm::ArrayRef<Decl *> ParsedAST::getLocalTopLevelDecls() {
  return LocalTopLevelDecls;
}

const MainFileMacros &ParsedAST::getMacros() const { return Macros; }
const std::vector<PragmaMark> &ParsedAST::getMarks() const { return Marks; }

std::size_t ParsedAST::getUsedBytes() const {
  auto &AST = getASTContext();
  // FIXME(ibiryukov): we do not account for the dynamically allocated part of
  // Message and Fixes inside each diagnostic.
  std::size_t Total = clangd::getUsedBytes(LocalTopLevelDecls) +
                      (Diags ? clangd::getUsedBytes(*Diags) : 0);

  // FIXME: the rest of the function is almost a direct copy-paste from
  // libclang's clang_getCXTUResourceUsage. We could share the implementation.

  // Sum up various allocators inside the ast context and the preprocessor.
  Total += AST.getASTAllocatedMemory();
  Total += AST.getSideTableAllocatedMemory();
  Total += AST.Idents.getAllocator().getTotalMemory();
  Total += AST.Selectors.getTotalMemory();

  Total += AST.getSourceManager().getContentCacheSize();
  Total += AST.getSourceManager().getDataStructureSizes();
  Total += AST.getSourceManager().getMemoryBufferSizes().malloc_bytes;

  if (ExternalASTSource *Ext = AST.getExternalSource())
    Total += Ext->getMemoryBufferSizes().malloc_bytes;

  const Preprocessor &PP = getPreprocessor();
  Total += PP.getTotalMemory();
  if (PreprocessingRecord *PRec = PP.getPreprocessingRecord())
    Total += PRec->getTotalMemory();
  Total += PP.getHeaderSearchInfo().getTotalMemory();

  return Total;
}

const IncludeStructure &ParsedAST::getIncludeStructure() const {
  return Includes;
}

const CanonicalIncludes &ParsedAST::getCanonicalIncludes() const {
  return CanonIncludes;
}

ParsedAST::ParsedAST(llvm::StringRef Version,
                     std::shared_ptr<const PreambleData> Preamble,
                     std::unique_ptr<CompilerInstance> Clang,
                     std::unique_ptr<FrontendAction> Action,
                     syntax::TokenBuffer Tokens, MainFileMacros Macros,
                     std::vector<PragmaMark> Marks,
                     std::vector<Decl *> LocalTopLevelDecls,
                     llvm::Optional<std::vector<Diag>> Diags,
                     IncludeStructure Includes, CanonicalIncludes CanonIncludes)
    : Version(Version), Preamble(std::move(Preamble)), Clang(std::move(Clang)),
      Action(std::move(Action)), Tokens(std::move(Tokens)),
      Macros(std::move(Macros)), Marks(std::move(Marks)),
      Diags(std::move(Diags)),
      LocalTopLevelDecls(std::move(LocalTopLevelDecls)),
      Includes(std::move(Includes)), CanonIncludes(std::move(CanonIncludes)) {
  Resolver = std::make_unique<HeuristicResolver>(getASTContext());
  assert(this->Clang);
  assert(this->Action);
}

llvm::Optional<llvm::StringRef> ParsedAST::preambleVersion() const {
  if (!Preamble)
    return llvm::None;
  return llvm::StringRef(Preamble->Version);
}

} // namespace clangd
} // namespace clang
