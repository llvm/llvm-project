//===--- ClangRefactor.cpp - Clang-based refactoring tool -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a clang-refactor tool that performs various
/// source transformations.
///
//===----------------------------------------------------------------------===//

#include "TestSupport.h"
#include "clang/Frontend/CommandLineSourceLoc.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/RefactoringAction.h"
#include "clang/Tooling/Refactoring/RefactoringOptions.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace clang;
using namespace tooling;
using namespace refactor;
namespace cl = llvm::cl;

namespace opts {

static cl::OptionCategory CommonRefactorOptions("Refactoring options");

static cl::opt<bool> Verbose("v", cl::desc("Use verbose output"),
                             cl::cat(cl::GeneralCategory),
                             cl::sub(*cl::AllSubCommands));
} // end namespace opts

namespace {

/// Stores the parsed `-selection` argument.
class SourceSelectionArgument {
public:
  virtual ~SourceSelectionArgument() {}

  /// Parse the `-selection` argument.
  ///
  /// \returns A valid argument when the parse succedeed, null otherwise.
  static std::unique_ptr<SourceSelectionArgument> fromString(StringRef Value);

  /// Prints any additional state associated with the selection argument to
  /// the given output stream.
  virtual void print(raw_ostream &OS) {}

  /// Returns a replacement refactoring result consumer (if any) that should
  /// consume the results of a refactoring operation.
  ///
  /// The replacement refactoring result consumer is used by \c
  /// TestSourceSelectionArgument to inject a test-specific result handling
  /// logic into the refactoring operation. The test-specific consumer
  /// ensures that the individual results in a particular test group are
  /// identical.
  virtual std::unique_ptr<ClangRefactorToolConsumerInterface>
  createCustomConsumer() {
    return nullptr;
  }

  /// Runs the give refactoring function for each specified selection.
  ///
  /// \returns true if an error occurred, false otherwise.
  virtual bool
  forAllRanges(const SourceManager &SM,
               llvm::function_ref<void(SourceRange R)> Callback) = 0;
};

/// Stores the parsed -selection=test:<filename> option.
class TestSourceSelectionArgument final : public SourceSelectionArgument {
public:
  TestSourceSelectionArgument(TestSelectionRangesInFile TestSelections)
      : TestSelections(std::move(TestSelections)) {}

  void print(raw_ostream &OS) override { TestSelections.dump(OS); }

  std::unique_ptr<ClangRefactorToolConsumerInterface>
  createCustomConsumer() override {
    return TestSelections.createConsumer();
  }

  /// Testing support: invokes the selection action for each selection range in
  /// the test file.
  bool forAllRanges(const SourceManager &SM,
                    llvm::function_ref<void(SourceRange R)> Callback) override {
    return TestSelections.foreachRange(SM, Callback);
  }

private:
  TestSelectionRangesInFile TestSelections;
};

/// Stores the parsed -selection=filename:line:column[-line:column] option.
class SourceRangeSelectionArgument final : public SourceSelectionArgument {
public:
  SourceRangeSelectionArgument(ParsedSourceRange Range)
      : Range(std::move(Range)) {}

  bool forAllRanges(const SourceManager &SM,
                    llvm::function_ref<void(SourceRange R)> Callback) override {
    const FileEntry *FE = SM.getFileManager().getFile(Range.FileName);
    FileID FID = FE ? SM.translateFile(FE) : FileID();
    if (!FE || FID.isInvalid()) {
      llvm::errs() << "error: -selection=" << Range.FileName
                   << ":... : given file is not in the target TU\n";
      return true;
    }

    SourceLocation Start = SM.getMacroArgExpandedLocation(
        SM.translateLineCol(FID, Range.Begin.first, Range.Begin.second));
    SourceLocation End = SM.getMacroArgExpandedLocation(
        SM.translateLineCol(FID, Range.End.first, Range.End.second));
    if (Start.isInvalid() || End.isInvalid()) {
      llvm::errs() << "error: -selection=" << Range.FileName << ':'
                   << Range.Begin.first << ':' << Range.Begin.second << '-'
                   << Range.End.first << ':' << Range.End.second
                   << " : invalid source location\n";
      return true;
    }
    Callback(SourceRange(Start, End));
    return false;
  }

private:
  ParsedSourceRange Range;
};

std::unique_ptr<SourceSelectionArgument>
SourceSelectionArgument::fromString(StringRef Value) {
  if (Value.startswith("test:")) {
    StringRef Filename = Value.drop_front(strlen("test:"));
    Optional<TestSelectionRangesInFile> ParsedTestSelection =
        findTestSelectionRanges(Filename);
    if (!ParsedTestSelection)
      return nullptr; // A parsing error was already reported.
    return llvm::make_unique<TestSourceSelectionArgument>(
        std::move(*ParsedTestSelection));
  }
  Optional<ParsedSourceRange> Range = ParsedSourceRange::fromString(Value);
  if (Range)
    return llvm::make_unique<SourceRangeSelectionArgument>(std::move(*Range));
  llvm::errs() << "error: '-selection' option must be specified using "
                  "<file>:<line>:<column> or "
                  "<file>:<line>:<column>-<line>:<column> format\n";
  return nullptr;
}

/// A container that stores the command-line options used by a single
/// refactoring option.
class RefactoringActionCommandLineOptions {
public:
  void addStringOption(const RefactoringOption &Option,
                       std::unique_ptr<cl::opt<std::string>> CLOption) {
    StringOptions[&Option] = std::move(CLOption);
  }

  const cl::opt<std::string> &
  getStringOption(const RefactoringOption &Opt) const {
    auto It = StringOptions.find(&Opt);
    return *It->second;
  }

private:
  llvm::DenseMap<const RefactoringOption *,
                 std::unique_ptr<cl::opt<std::string>>>
      StringOptions;
};

/// Passes the command-line option values to the options used by a single
/// refactoring action rule.
class CommandLineRefactoringOptionVisitor final
    : public RefactoringOptionVisitor {
public:
  CommandLineRefactoringOptionVisitor(
      const RefactoringActionCommandLineOptions &Options)
      : Options(Options) {}

  void visit(const RefactoringOption &Opt,
             Optional<std::string> &Value) override {
    const cl::opt<std::string> &CLOpt = Options.getStringOption(Opt);
    if (!CLOpt.getValue().empty()) {
      Value = CLOpt.getValue();
      return;
    }
    Value = None;
    if (Opt.isRequired())
      MissingRequiredOptions.push_back(&Opt);
  }

  ArrayRef<const RefactoringOption *> getMissingRequiredOptions() const {
    return MissingRequiredOptions;
  }

private:
  llvm::SmallVector<const RefactoringOption *, 4> MissingRequiredOptions;
  const RefactoringActionCommandLineOptions &Options;
};

/// Creates the refactoring options used by all the rules in a single
/// refactoring action.
class CommandLineRefactoringOptionCreator final
    : public RefactoringOptionVisitor {
public:
  CommandLineRefactoringOptionCreator(
      cl::OptionCategory &Category, cl::SubCommand &Subcommand,
      RefactoringActionCommandLineOptions &Options)
      : Category(Category), Subcommand(Subcommand), Options(Options) {}

  void visit(const RefactoringOption &Opt, Optional<std::string> &) override {
    if (Visited.insert(&Opt).second)
      Options.addStringOption(Opt, create<std::string>(Opt));
  }

private:
  template <typename T>
  std::unique_ptr<cl::opt<T>> create(const RefactoringOption &Opt) {
    if (!OptionNames.insert(Opt.getName()).second)
      llvm::report_fatal_error("Multiple identical refactoring options "
                               "specified for one refactoring action");
    // FIXME: cl::Required can be specified when this option is present
    // in all rules in an action.
    return llvm::make_unique<cl::opt<T>>(
        Opt.getName(), cl::desc(Opt.getDescription()), cl::Optional,
        cl::cat(Category), cl::sub(Subcommand));
  }

  llvm::SmallPtrSet<const RefactoringOption *, 8> Visited;
  llvm::StringSet<> OptionNames;
  cl::OptionCategory &Category;
  cl::SubCommand &Subcommand;
  RefactoringActionCommandLineOptions &Options;
};

/// A subcommand that corresponds to individual refactoring action.
class RefactoringActionSubcommand : public cl::SubCommand {
public:
  RefactoringActionSubcommand(std::unique_ptr<RefactoringAction> Action,
                              RefactoringActionRules ActionRules,
                              cl::OptionCategory &Category)
      : SubCommand(Action->getCommand(), Action->getDescription()),
        Action(std::move(Action)), ActionRules(std::move(ActionRules)) {
    // Check if the selection option is supported.
    bool HasSelection = false;
    for (const auto &Rule : this->ActionRules) {
      if ((HasSelection = Rule->hasSelectionRequirement()))
        break;
    }
    if (HasSelection) {
      Selection = llvm::make_unique<cl::opt<std::string>>(
          "selection",
          cl::desc("The selected source range in which the refactoring should "
                   "be initiated (<file>:<line>:<column>-<line>:<column> or "
                   "<file>:<line>:<column>)"),
          cl::cat(Category), cl::sub(*this));
    }
    // Create the refactoring options.
    for (const auto &Rule : this->ActionRules) {
      CommandLineRefactoringOptionCreator OptionCreator(Category, *this,
                                                        Options);
      Rule->visitRefactoringOptions(OptionCreator);
    }
  }

  ~RefactoringActionSubcommand() { unregisterSubCommand(); }

  const RefactoringActionRules &getActionRules() const { return ActionRules; }

  /// Parses the command-line arguments that are specific to this rule.
  ///
  /// \returns true on error, false otherwise.
  bool parseArguments() {
    if (Selection) {
      ParsedSelection = SourceSelectionArgument::fromString(*Selection);
      if (!ParsedSelection)
        return true;
    }
    return false;
  }

  SourceSelectionArgument *getSelection() const {
    assert(Selection && "selection not supported!");
    return ParsedSelection.get();
  }

  const RefactoringActionCommandLineOptions &getOptions() const {
    return Options;
  }

private:
  std::unique_ptr<RefactoringAction> Action;
  RefactoringActionRules ActionRules;
  std::unique_ptr<cl::opt<std::string>> Selection;
  std::unique_ptr<SourceSelectionArgument> ParsedSelection;
  RefactoringActionCommandLineOptions Options;
};

class ClangRefactorConsumer final : public ClangRefactorToolConsumerInterface {
public:
  ClangRefactorConsumer() {}

  void handleError(llvm::Error Err) override {
    Optional<PartialDiagnosticAt> Diag = DiagnosticError::take(Err);
    if (!Diag) {
      llvm::errs() << llvm::toString(std::move(Err)) << "\n";
      return;
    }
    llvm::cantFail(std::move(Err)); // This is a success.
    DiagnosticBuilder DB(
        getDiags().Report(Diag->first, Diag->second.getDiagID()));
    Diag->second.Emit(DB);
  }

  void handle(AtomicChanges Changes) override {
    SourceChanges.insert(SourceChanges.begin(), Changes.begin(), Changes.end());
  }

  void handle(SymbolOccurrences Occurrences) override {
    llvm_unreachable("symbol occurrence results are not handled yet");
  }

  const AtomicChanges &getSourceChanges() const { return SourceChanges; }

private:
  AtomicChanges SourceChanges;
};

class ClangRefactorTool {
public:
  std::vector<std::unique_ptr<RefactoringActionSubcommand>> SubCommands;

  ClangRefactorTool() {
    std::vector<std::unique_ptr<RefactoringAction>> Actions =
        createRefactoringActions();

    // Actions must have unique command names so that we can map them to one
    // subcommand.
    llvm::StringSet<> CommandNames;
    for (const auto &Action : Actions) {
      if (!CommandNames.insert(Action->getCommand()).second) {
        llvm::errs() << "duplicate refactoring action command '"
                     << Action->getCommand() << "'!";
        exit(1);
      }
    }

    // Create subcommands and command-line options.
    for (auto &Action : Actions) {
      SubCommands.push_back(llvm::make_unique<RefactoringActionSubcommand>(
          std::move(Action), Action->createActiveActionRules(),
          opts::CommonRefactorOptions));
    }
  }

  using TUCallbackType = llvm::function_ref<void(ASTContext &)>;

  /// Parses the translation units that were given to the subcommand using
  /// the 'sources' option and invokes the callback for each parsed
  /// translation unit.
  bool foreachTranslationUnit(const CompilationDatabase &DB,
                              ArrayRef<std::string> Sources,
                              TUCallbackType Callback) {
    class ToolASTConsumer : public ASTConsumer {
    public:
      TUCallbackType Callback;
      ToolASTConsumer(TUCallbackType Callback) : Callback(Callback) {}

      void HandleTranslationUnit(ASTContext &Context) override {
        Callback(Context);
      }
    };
    class ActionWrapper {
    public:
      TUCallbackType Callback;
      ActionWrapper(TUCallbackType Callback) : Callback(Callback) {}

      std::unique_ptr<ASTConsumer> newASTConsumer() {
        return llvm::make_unique<ToolASTConsumer>(Callback);
      }
    };

    ClangTool Tool(DB, Sources);
    ActionWrapper ToolAction(Callback);
    std::unique_ptr<tooling::FrontendActionFactory> Factory =
        tooling::newFrontendActionFactory(&ToolAction);
    return Tool.run(Factory.get());
  }

  /// Logs an individual refactoring action invocation to STDOUT.
  void logInvocation(RefactoringActionSubcommand &Subcommand,
                     const RefactoringRuleContext &Context) {
    if (!opts::Verbose)
      return;
    llvm::outs() << "invoking action '" << Subcommand.getName() << "':\n";
    if (Context.getSelectionRange().isValid()) {
      SourceRange R = Context.getSelectionRange();
      llvm::outs() << "  -selection=";
      R.getBegin().print(llvm::outs(), Context.getSources());
      llvm::outs() << " -> ";
      R.getEnd().print(llvm::outs(), Context.getSources());
      llvm::outs() << "\n";
    }
  }

  bool applySourceChanges(const AtomicChanges &Replacements) {
    std::set<std::string> Files;
    for (const auto &Change : Replacements)
      Files.insert(Change.getFilePath());
    // FIXME: Add automatic formatting support as well.
    tooling::ApplyChangesSpec Spec;
    // FIXME: We should probably cleanup the result by default as well.
    Spec.Cleanup = false;
    for (const auto &File : Files) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferErr =
          llvm::MemoryBuffer::getFile(File);
      if (!BufferErr) {
        llvm::errs() << "error: failed to open " << File << " for rewriting\n";
        return true;
      }
      auto Result = tooling::applyAtomicChanges(File, (*BufferErr)->getBuffer(),
                                                Replacements, Spec);
      if (!Result) {
        llvm::errs() << toString(Result.takeError());
        return true;
      }

      std::error_code EC;
      llvm::raw_fd_ostream OS(File, EC, llvm::sys::fs::F_Text);
      if (EC) {
        llvm::errs() << EC.message() << "\n";
        return true;
      }
      OS << *Result;
    }
    return false;
  }

  bool invokeAction(RefactoringActionSubcommand &Subcommand,
                    const CompilationDatabase &DB,
                    ArrayRef<std::string> Sources) {
    // Find a set of matching rules.
    SmallVector<RefactoringActionRule *, 4> MatchingRules;
    llvm::StringSet<> MissingOptions;

    bool HasSelection = false;
    for (const auto &Rule : Subcommand.getActionRules()) {
      bool SelectionMatches = true;
      if (Rule->hasSelectionRequirement()) {
        HasSelection = true;
        if (!Subcommand.getSelection()) {
          MissingOptions.insert("selection");
          SelectionMatches = false;
        }
      }
      CommandLineRefactoringOptionVisitor Visitor(Subcommand.getOptions());
      Rule->visitRefactoringOptions(Visitor);
      if (SelectionMatches && Visitor.getMissingRequiredOptions().empty()) {
        MatchingRules.push_back(Rule.get());
        continue;
      }
      for (const RefactoringOption *Opt : Visitor.getMissingRequiredOptions())
        MissingOptions.insert(Opt->getName());
    }
    if (MatchingRules.empty()) {
      llvm::errs() << "error: '" << Subcommand.getName()
                   << "' can't be invoked with the given arguments:\n";
      for (const auto &Opt : MissingOptions)
        llvm::errs() << "  missing '-" << Opt.getKey() << "' option\n";
      return true;
    }

    ClangRefactorConsumer Consumer;
    bool HasFailed = false;
    if (foreachTranslationUnit(DB, Sources, [&](ASTContext &AST) {
          RefactoringRuleContext Context(AST.getSourceManager());
          Context.setASTContext(AST);

          auto InvokeRule = [&](RefactoringResultConsumer &Consumer) {
            logInvocation(Subcommand, Context);
            for (RefactoringActionRule *Rule : MatchingRules) {
              if (!Rule->hasSelectionRequirement())
                continue;
              Rule->invoke(Consumer, Context);
              return;
            }
            // FIXME (Alex L): If more than one initiation succeeded, then the
            // rules are ambiguous.
            llvm_unreachable(
                "The action must have at least one selection rule");
          };

          std::unique_ptr<ClangRefactorToolConsumerInterface> CustomConsumer;
          if (HasSelection)
            CustomConsumer = Subcommand.getSelection()->createCustomConsumer();
          ClangRefactorToolConsumerInterface &ActiveConsumer =
              CustomConsumer ? *CustomConsumer : Consumer;
          ActiveConsumer.beginTU(AST);
          if (HasSelection) {
            assert(Subcommand.getSelection() && "Missing selection argument?");
            if (opts::Verbose)
              Subcommand.getSelection()->print(llvm::outs());
            if (Subcommand.getSelection()->forAllRanges(
                    Context.getSources(), [&](SourceRange R) {
                      Context.setSelectionRange(R);
                      InvokeRule(ActiveConsumer);
                    }))
              HasFailed = true;
            ActiveConsumer.endTU();
            return;
          }
          // FIXME (Alex L): Implement non-selection based invocation path.
          ActiveConsumer.endTU();
        }))
      return true;
    return HasFailed || applySourceChanges(Consumer.getSourceChanges());
  }
};

} // end anonymous namespace

int main(int argc, const char **argv) {
  ClangRefactorTool Tool;

  CommonOptionsParser Options(
      argc, argv, cl::GeneralCategory, cl::ZeroOrMore,
      "Clang-based refactoring tool for C, C++ and Objective-C");

  // Figure out which action is specified by the user. The user must specify
  // the action using a command-line subcommand, e.g. the invocation
  // `clang-refactor local-rename` corresponds to the `LocalRename` refactoring
  // action. All subcommands must have a unique names. This allows us to figure
  // out which refactoring action should be invoked by looking at the first
  // subcommand that's enabled by LLVM's command-line parser.
  auto It = llvm::find_if(
      Tool.SubCommands,
      [](const std::unique_ptr<RefactoringActionSubcommand> &SubCommand) {
        return !!(*SubCommand);
      });
  if (It == Tool.SubCommands.end()) {
    llvm::errs() << "error: no refactoring action given\n";
    llvm::errs() << "note: the following actions are supported:\n";
    for (const auto &Subcommand : Tool.SubCommands)
      llvm::errs().indent(2) << Subcommand->getName() << "\n";
    return 1;
  }
  RefactoringActionSubcommand &ActionCommand = **It;

  if (ActionCommand.parseArguments())
    return 1;
  if (Tool.invokeAction(ActionCommand, Options.getCompilations(),
                        Options.getSourcePathList()))
    return 1;

  return 0;
}
