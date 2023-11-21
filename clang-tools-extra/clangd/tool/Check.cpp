//===--- Check.cpp - clangd self-diagnostics ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Many basic problems can occur processing a file in clangd, e.g.:
//  - system includes are not found
//  - crash when indexing its AST
// clangd --check provides a simplified, isolated way to reproduce these,
// with no editor, LSP, threads, background indexing etc to contend with.
//
// One important use case is gathering information for bug reports.
// Another is reproducing crashes, and checking which setting prevent them.
//
// It simulates opening a file (determining compile command, parsing, indexing)
// and then running features at many locations.
//
// Currently it adds some basic logging of progress and results.
// We should consider extending it to also recognize common symptoms and
// recommend solutions (e.g. standard library installation issues).
//
//===----------------------------------------------------------------------===//

#include "../clang-tidy/ClangTidyModule.h"
#include "../clang-tidy/ClangTidyModuleRegistry.h"
#include "../clang-tidy/ClangTidyOptions.h"
#include "../clang-tidy/GlobList.h"
#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "CodeComplete.h"
#include "CompileCommands.h"
#include "Compiler.h"
#include "Config.h"
#include "ConfigFragment.h"
#include "ConfigProvider.h"
#include "Diagnostics.h"
#include "Feature.h"
#include "GlobalCompilationDatabase.h"
#include "Hover.h"
#include "InlayHints.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "Protocol.h"
#include "Selection.h"
#include "SemanticHighlighting.h"
#include "SourceCode.h"
#include "TidyProvider.h"
#include "XRefs.h"
#include "clang-include-cleaner/Record.h"
#include "index/FileIndex.h"
#include "refactor/Tweak.h"
#include "support/Context.h"
#include "support/Logger.h"
#include "support/ThreadsafeFS.h"
#include "support/Trace.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// These will never be shown in --help, ClangdMain doesn't list the category.
llvm::cl::opt<std::string> CheckTidyTime{
    "check-tidy-time",
    llvm::cl::desc("Print the overhead of checks matching this glob"),
    llvm::cl::init("")};
llvm::cl::opt<std::string> CheckFileLines{
    "check-lines",
    llvm::cl::desc(
        "Limits the range of tokens in -check file on which "
        "various features are tested. Example --check-lines=3-7 restricts "
        "testing to lines 3 to 7 (inclusive) or --check-lines=5 to restrict "
        "to one line. Default is testing entire file."),
    llvm::cl::init("")};
llvm::cl::opt<bool> CheckLocations{
    "check-locations",
    llvm::cl::desc(
        "Runs certain features (e.g. hover) at each point in the file. "
        "Somewhat slow."),
    llvm::cl::init(true)};
llvm::cl::opt<bool> CheckCompletion{
    "check-completion",
    llvm::cl::desc("Run code-completion at each point (slow)"),
    llvm::cl::init(false)};
llvm::cl::opt<bool> CheckWarnings{
    "check-warnings",
    llvm::cl::desc("Print warnings as well as errors"),
    llvm::cl::init(false)};

// Print the diagnostics meeting severity threshold, and return count of errors.
unsigned showErrors(llvm::ArrayRef<Diag> Diags) {
  unsigned ErrCount = 0;
  for (const auto &D : Diags) {
    if (D.Severity >= DiagnosticsEngine::Error || CheckWarnings)
      elog("[{0}] Line {1}: {2}", D.Name, D.Range.start.line + 1, D.Message);
    if (D.Severity >= DiagnosticsEngine::Error)
      ++ErrCount;
  }
  return ErrCount;
}

std::vector<std::string> listTidyChecks(llvm::StringRef Glob) {
  tidy::GlobList G(Glob);
  tidy::ClangTidyCheckFactories CTFactories;
  for (const auto &E : tidy::ClangTidyModuleRegistry::entries())
    E.instantiate()->addCheckFactories(CTFactories);
  std::vector<std::string> Result;
  for (const auto &E : CTFactories)
    if (G.contains(E.getKey()))
      Result.push_back(E.getKey().str());
  llvm::sort(Result);
  return Result;
}

// This class is just a linear pipeline whose functions get called in sequence.
// Each exercises part of clangd's logic on our test file and logs results.
// Later steps depend on state built in earlier ones (such as the AST).
// Many steps can fatally fail (return false), then subsequent ones cannot run.
// Nonfatal failures are logged and tracked in ErrCount.
class Checker {
  // from constructor
  std::string File;
  ClangdLSPServer::Options Opts;
  // from buildCommand
  tooling::CompileCommand Cmd;
  // from buildInvocation
  ParseInputs Inputs;
  std::unique_ptr<CompilerInvocation> Invocation;
  format::FormatStyle Style;
  // from buildAST
  std::shared_ptr<const PreambleData> Preamble;
  std::optional<ParsedAST> AST;
  FileIndex Index;

public:
  // Number of non-fatal errors seen.
  unsigned ErrCount = 0;

  Checker(llvm::StringRef File, const ClangdLSPServer::Options &Opts)
      : File(File), Opts(Opts) {}

  // Read compilation database and choose a compile command for the file.
  bool buildCommand(const ThreadsafeFS &TFS) {
    log("Loading compilation database...");
    DirectoryBasedGlobalCompilationDatabase::Options CDBOpts(TFS);
    CDBOpts.CompileCommandsDir =
        Config::current().CompileFlags.CDBSearch.FixedCDBPath;
    std::unique_ptr<GlobalCompilationDatabase> BaseCDB =
        std::make_unique<DirectoryBasedGlobalCompilationDatabase>(CDBOpts);
    auto Mangler = CommandMangler::detect();
    Mangler.SystemIncludeExtractor =
        getSystemIncludeExtractor(llvm::ArrayRef(Opts.QueryDriverGlobs));
    if (Opts.ResourceDir)
      Mangler.ResourceDir = *Opts.ResourceDir;
    auto CDB = std::make_unique<OverlayCDB>(
        BaseCDB.get(), std::vector<std::string>{}, std::move(Mangler));

    if (auto TrueCmd = CDB->getCompileCommand(File)) {
      Cmd = std::move(*TrueCmd);
      log("Compile command {0} is: [{1}] {2}",
          Cmd.Heuristic.empty() ? "from CDB" : Cmd.Heuristic, Cmd.Directory,
          printArgv(Cmd.CommandLine));
    } else {
      Cmd = CDB->getFallbackCommand(File);
      log("Generic fallback command is: [{0}] {1}", Cmd.Directory,
          printArgv(Cmd.CommandLine));
    }

    return true;
  }

  // Prepare inputs and build CompilerInvocation (parsed compile command).
  bool buildInvocation(const ThreadsafeFS &TFS,
                       std::optional<std::string> Contents) {
    StoreDiags CaptureInvocationDiags;
    std::vector<std::string> CC1Args;
    Inputs.CompileCommand = Cmd;
    Inputs.TFS = &TFS;
    Inputs.ClangTidyProvider = Opts.ClangTidyProvider;
    Inputs.Opts.PreambleParseForwardingFunctions =
        Opts.PreambleParseForwardingFunctions;
    if (Contents) {
      Inputs.Contents = *Contents;
      log("Imaginary source file contents:\n{0}", Inputs.Contents);
    } else {
      if (auto Contents = TFS.view(std::nullopt)->getBufferForFile(File)) {
        Inputs.Contents = Contents->get()->getBuffer().str();
      } else {
        elog("Couldn't read {0}: {1}", File, Contents.getError().message());
        return false;
      }
    }
    log("Parsing command...");
    Invocation =
        buildCompilerInvocation(Inputs, CaptureInvocationDiags, &CC1Args);
    auto InvocationDiags = CaptureInvocationDiags.take();
    ErrCount += showErrors(InvocationDiags);
    log("internal (cc1) args are: {0}", printArgv(CC1Args));
    if (!Invocation) {
      elog("Failed to parse command line");
      return false;
    }

    // FIXME: Check that resource-dir/built-in-headers exist?

    Style = getFormatStyleForFile(File, Inputs.Contents, TFS);

    return true;
  }

  // Build preamble and AST, and index them.
  bool buildAST() {
    log("Building preamble...");
    Preamble = buildPreamble(
        File, *Invocation, Inputs, /*StoreInMemory=*/true,
        [&](CapturedASTCtx Ctx,
            std::shared_ptr<const include_cleaner::PragmaIncludes> PI) {
          if (!Opts.BuildDynamicSymbolIndex)
            return;
          log("Indexing headers...");
          Index.updatePreamble(File, /*Version=*/"null", Ctx.getASTContext(),
                               Ctx.getPreprocessor(), *PI);
        });
    if (!Preamble) {
      elog("Failed to build preamble");
      return false;
    }
    ErrCount += showErrors(Preamble->Diags);

    log("Building AST...");
    AST = ParsedAST::build(File, Inputs, std::move(Invocation),
                           /*InvocationDiags=*/std::vector<Diag>{}, Preamble);
    if (!AST) {
      elog("Failed to build AST");
      return false;
    }
    ErrCount +=
        showErrors(AST->getDiagnostics().drop_front(Preamble->Diags.size()));

    if (Opts.BuildDynamicSymbolIndex) {
      log("Indexing AST...");
      Index.updateMain(File, *AST);
    }

    if (!CheckTidyTime.empty()) {
      if (!CLANGD_TIDY_CHECKS) {
        elog("-{0} requires -DCLANGD_TIDY_CHECKS!", CheckTidyTime.ArgStr);
        return false;
      }
      #ifndef NDEBUG
      elog("Timing clang-tidy checks in asserts-mode is not representative!");
      #endif
      checkTidyTimes();
    }

    return true;
  }

  // For each check foo, we want to build with checks=-* and checks=-*,foo.
  // (We do a full build rather than just AST matchers to meausre PPCallbacks).
  //
  // However, performance has both random noise and systematic changes, such as
  // step-function slowdowns due to CPU scaling.
  // We take the median of 5 measurements, and after every check discard the
  // measurement if the baseline changed by >3%.
  void checkTidyTimes() {
    double Stability = 0.03;
    log("Timing AST build with individual clang-tidy checks (target accuracy "
        "{0:P0})",
        Stability);

    using Duration = std::chrono::nanoseconds;
    // Measure time elapsed by a block of code. Currently: user CPU time.
    auto Time = [&](auto &&Run) -> Duration {
      llvm::sys::TimePoint<> Elapsed;
      std::chrono::nanoseconds UserBegin, UserEnd, System;
      llvm::sys::Process::GetTimeUsage(Elapsed, UserBegin, System);
      Run();
      llvm::sys::Process::GetTimeUsage(Elapsed, UserEnd, System);
      return UserEnd - UserBegin;
    };
    auto Change = [&](Duration Exp, Duration Base) -> double {
      return (double)(Exp.count() - Base.count()) / Base.count();
    };
    // Build ParsedAST with a fixed check glob, and return the time taken.
    auto Build = [&](llvm::StringRef Checks) -> Duration {
      TidyProvider CTProvider = [&](tidy::ClangTidyOptions &Opts,
                                    llvm::StringRef) {
        Opts.Checks = Checks.str();
      };
      Inputs.ClangTidyProvider = CTProvider;
      // Sigh, can't reuse the CompilerInvocation.
      IgnoringDiagConsumer IgnoreDiags;
      auto Invocation = buildCompilerInvocation(Inputs, IgnoreDiags);
      Duration Val = Time([&] {
        ParsedAST::build(File, Inputs, std::move(Invocation), {}, Preamble);
      });
      vlog("    Measured {0} ==> {1}", Checks, Val);
      return Val;
    };
    // Measure several times, return the median.
    auto MedianTime = [&](llvm::StringRef Checks) -> Duration {
      std::array<Duration, 5> Measurements;
      for (auto &M : Measurements)
        M = Build(Checks);
      llvm::sort(Measurements);
      return Measurements[Measurements.size() / 2];
    };
    Duration Baseline = MedianTime("-*");
    log("  Baseline = {0}", Baseline);
    // Attempt to time a check, may update Baseline if it is unstable.
    auto Measure = [&](llvm::StringRef Check) -> double {
      for (;;) {
        Duration Median = MedianTime(("-*," + Check).str());
        Duration NewBase = MedianTime("-*");

        // Value only usable if baseline is fairly consistent before/after.
        double DeltaFraction = Change(NewBase, Baseline);
        Baseline = NewBase;
        vlog("  Baseline = {0}", Baseline);
        if (DeltaFraction < -Stability || DeltaFraction > Stability) {
          elog("  Speed unstable, discarding measurement.");
          continue;
        }
        return Change(Median, Baseline);
      }
    };

    for (const auto& Check : listTidyChecks(CheckTidyTime)) {
      // vlog the check name in case we crash!
      vlog("  Timing {0}", Check);
      double Fraction = Measure(Check);
      log("  {0} = {1:P0}", Check, Fraction);
    }
    log("Finished individual clang-tidy checks");

    // Restore old options.
    Inputs.ClangTidyProvider = Opts.ClangTidyProvider;
  }

  // Build Inlay Hints for the entire AST or the specified range
  void buildInlayHints(std::optional<Range> LineRange) {
    log("Building inlay hints");
    auto Hints = inlayHints(*AST, LineRange);

    for (const auto &Hint : Hints) {
      vlog("  {0} {1} {2}", Hint.kind, Hint.position, Hint.label);
    }
  }

  void buildSemanticHighlighting(std::optional<Range> LineRange) {
    log("Building semantic highlighting");
    auto Highlights =
        getSemanticHighlightings(*AST, /*IncludeInactiveRegionTokens=*/true);
    for (const auto HL : Highlights)
      if (!LineRange || LineRange->contains(HL.R))
        vlog(" {0} {1} {2}", HL.R, HL.Kind, HL.Modifiers);
  }

  // Run AST-based features at each token in the file.
  void testLocationFeatures(std::optional<Range> LineRange) {
    trace::Span Trace("testLocationFeatures");
    log("Testing features at each token (may be slow in large files)");
    auto &SM = AST->getSourceManager();
    auto SpelledTokens = AST->getTokens().spelledTokens(SM.getMainFileID());

    CodeCompleteOptions CCOpts = Opts.CodeComplete;
    CCOpts.Index = &Index;

    for (const auto &Tok : SpelledTokens) {
      unsigned Start = AST->getSourceManager().getFileOffset(Tok.location());
      unsigned End = Start + Tok.length();
      Position Pos = offsetToPosition(Inputs.Contents, Start);

      if (LineRange && !LineRange->contains(Pos))
        continue;

      trace::Span Trace("Token");
      SPAN_ATTACH(Trace, "pos", Pos);
      SPAN_ATTACH(Trace, "text", Tok.text(AST->getSourceManager()));

      // FIXME: dumping the tokens may leak sensitive code into bug reports.
      // Add an option to turn this off, once we decide how options work.
      vlog("  {0} {1}", Pos, Tok.text(AST->getSourceManager()));
      auto Tree = SelectionTree::createRight(AST->getASTContext(),
                                             AST->getTokens(), Start, End);
      Tweak::Selection Selection(&Index, *AST, Start, End, std::move(Tree),
                                 nullptr);
      // FS is only populated when applying a tweak, not during prepare as
      // prepare should not do any I/O to be fast.
      auto Tweaks =
          prepareTweaks(Selection, Opts.TweakFilter, Opts.FeatureModules);
      Selection.FS =
          &AST->getSourceManager().getFileManager().getVirtualFileSystem();
      for (const auto &T : Tweaks) {
        auto Result = T->apply(Selection);
        if (!Result) {
          elog("    tweak: {0} ==> FAIL: {1}", T->id(), Result.takeError());
          ++ErrCount;
        } else {
          vlog("    tweak: {0}", T->id());
        }
      }
      unsigned Definitions = locateSymbolAt(*AST, Pos, &Index).size();
      vlog("    definition: {0}", Definitions);

      auto Hover = getHover(*AST, Pos, Style, &Index);
      vlog("    hover: {0}", Hover.has_value());

      unsigned DocHighlights = findDocumentHighlights(*AST, Pos).size();
      vlog("    documentHighlight: {0}", DocHighlights);

      if (CheckCompletion) {
        Position EndPos = offsetToPosition(Inputs.Contents, End);
        auto CC = codeComplete(File, EndPos, Preamble.get(), Inputs, CCOpts);
        vlog("    code completion: {0}",
             CC.Completions.empty() ? "<empty>" : CC.Completions[0].Name);
      }
    }
  }
};

} // namespace

bool check(llvm::StringRef File, const ThreadsafeFS &TFS,
           const ClangdLSPServer::Options &Opts) {
  std::optional<Range> LineRange;
  if (!CheckFileLines.empty()) {
    uint32_t Begin = 0, End = std::numeric_limits<uint32_t>::max();
    StringRef RangeStr(CheckFileLines);
    bool ParseError = RangeStr.consumeInteger(0, Begin);
    if (RangeStr.empty()) {
      End = Begin;
    } else {
      ParseError |= !RangeStr.consume_front("-");
      ParseError |= RangeStr.consumeInteger(0, End);
    }
    if (ParseError || !RangeStr.empty() || Begin <= 0 || End < Begin) {
      elog("Invalid --check-lines specified. Use Begin-End format, e.g. 3-17");
      return false;
    }
    LineRange = Range{Position{static_cast<int>(Begin - 1), 0},
                      Position{static_cast<int>(End), 0}};
  }

  llvm::SmallString<0> FakeFile;
  std::optional<std::string> Contents;
  if (File.empty()) {
    llvm::sys::path::system_temp_directory(false, FakeFile);
    llvm::sys::path::append(FakeFile, "test.cc");
    File = FakeFile;
    Contents = R"cpp(
      #include <stddef.h>
      #include <string>

      size_t N = 50;
      auto xxx = std::string(N, 'x');
    )cpp";
  }
  log("Testing on source file {0}", File);

  class OverrideConfigProvider : public config::Provider {
    std::vector<config::CompiledFragment>
    getFragments(const config::Params &,
                 config::DiagnosticCallback Diag) const override {
      config::Fragment F;
      // If we're timing clang-tidy checks, implicitly disabling the slow ones
      // is counterproductive! 
      if (CheckTidyTime.getNumOccurrences())
        F.Diagnostics.ClangTidy.FastCheckFilter.emplace("None");
      return {std::move(F).compile(Diag)};
    }
  } OverrideConfig;
  auto ConfigProvider =
      config::Provider::combine({Opts.ConfigProvider, &OverrideConfig});

  auto ContextProvider = ClangdServer::createConfiguredContextProvider(
      ConfigProvider.get(), nullptr);
  WithContext Ctx(ContextProvider(
      FakeFile.empty()
          ? File
          : /*Don't turn on local configs for an arbitrary temp path.*/ ""));
  Checker C(File, Opts);
  if (!C.buildCommand(TFS) || !C.buildInvocation(TFS, Contents) ||
      !C.buildAST())
    return false;
  C.buildInlayHints(LineRange);
  C.buildSemanticHighlighting(LineRange);
  if (CheckLocations)
    C.testLocationFeatures(LineRange);

  log("All checks completed, {0} errors", C.ErrCount);
  return C.ErrCount == 0;
}

} // namespace clangd
} // namespace clang
