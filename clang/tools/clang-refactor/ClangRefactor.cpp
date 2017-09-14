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
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/RefactoringAction.h"
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

static cl::OptionCategory CommonRefactorOptions("Common refactoring options");

static cl::opt<bool>
    NoDatabases("no-dbs",
                cl::desc("Ignore external databases including Clang's "
                         "compilation database and indexer stores"),
                cl::cat(CommonRefactorOptions), cl::sub(*cl::AllSubCommands));

static cl::opt<bool> Verbose("v", cl::desc("Use verbose output"),
                             cl::cat(CommonRefactorOptions),
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
  virtual void print(raw_ostream &OS) = 0;

  /// Returns a replacement refactoring result consumer (if any) that should
  /// consume the results of a refactoring operation.
  ///
  /// The replacement refactoring result consumer is used by \c
  /// TestSourceSelectionArgument to inject a test-specific result handling
  /// logic into the refactoring operation. The test-specific consumer
  /// ensures that the individual results in a particular test group are
  /// identical.
  virtual std::unique_ptr<RefactoringResultConsumer> createCustomConsumer() {
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

  std::unique_ptr<RefactoringResultConsumer> createCustomConsumer() override {
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
  // FIXME: Support true selection ranges.
  llvm::errs() << "error: '-selection' option must be specified using "
                  "<file>:<line>:<column> or "
                  "<file>:<line>:<column>-<line>:<column> format";
  return nullptr;
}

/// A subcommand that corresponds to individual refactoring action.
class RefactoringActionSubcommand : public cl::SubCommand {
public:
  RefactoringActionSubcommand(std::unique_ptr<RefactoringAction> Action,
                              RefactoringActionRules ActionRules,
                              cl::OptionCategory &Category)
      : SubCommand(Action->getCommand(), Action->getDescription()),
        Action(std::move(Action)), ActionRules(std::move(ActionRules)) {
    Sources = llvm::make_unique<cl::list<std::string>>(
        cl::Positional, cl::ZeroOrMore, cl::desc("<source0> [... <sourceN>]"),
        cl::cat(Category), cl::sub(*this));

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

  ArrayRef<std::string> getSources() const { return *Sources; }

private:
  std::unique_ptr<RefactoringAction> Action;
  RefactoringActionRules ActionRules;
  std::unique_ptr<cl::list<std::string>> Sources;
  std::unique_ptr<cl::opt<std::string>> Selection;
  std::unique_ptr<SourceSelectionArgument> ParsedSelection;
};

class ClangRefactorConsumer : public RefactoringResultConsumer {
public:
  void handleError(llvm::Error Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
  }

  // FIXME: Consume atomic changes and apply them to files.
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
  bool foreachTranslationUnit(RefactoringActionSubcommand &Subcommand,
                              TUCallbackType Callback) {
    std::unique_ptr<CompilationDatabase> Compilations;
    if (opts::NoDatabases) {
      // FIXME (Alex L): Support compilation options.
      Compilations =
          llvm::make_unique<clang::tooling::FixedCompilationDatabase>(
              ".", std::vector<std::string>());
    } else {
      // FIXME (Alex L): Support compilation database.
      llvm::errs() << "compilation databases are not supported yet!\n";
      return true;
    }

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
        return llvm::make_unique<ToolASTConsumer>(std::move(Callback));
      }
    };

    ClangTool Tool(*Compilations, Subcommand.getSources());
    ActionWrapper ToolAction(std::move(Callback));
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

  bool invokeAction(RefactoringActionSubcommand &Subcommand) {
    // Find a set of matching rules.
    SmallVector<RefactoringActionRule *, 4> MatchingRules;
    llvm::StringSet<> MissingOptions;

    bool HasSelection = false;
    for (const auto &Rule : Subcommand.getActionRules()) {
      if (Rule->hasSelectionRequirement()) {
        HasSelection = true;
        if (Subcommand.getSelection())
          MatchingRules.push_back(Rule.get());
        else
          MissingOptions.insert("selection");
      }
      // FIXME (Alex L): Support custom options.
    }
    if (MatchingRules.empty()) {
      llvm::errs() << "error: '" << Subcommand.getName()
                   << "' can't be invoked with the given arguments:\n";
      for (const auto &Opt : MissingOptions)
        llvm::errs() << "  missing '-" << Opt.getKey() << "' option\n";
      return true;
    }

    bool HasFailed = false;
    ClangRefactorConsumer Consumer;
    if (foreachTranslationUnit(Subcommand, [&](ASTContext &AST) {
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

          if (HasSelection) {
            assert(Subcommand.getSelection() && "Missing selection argument?");
            if (opts::Verbose)
              Subcommand.getSelection()->print(llvm::outs());
            auto CustomConsumer =
                Subcommand.getSelection()->createCustomConsumer();
            if (Subcommand.getSelection()->forAllRanges(
                    Context.getSources(), [&](SourceRange R) {
                      Context.setSelectionRange(R);
                      InvokeRule(CustomConsumer ? *CustomConsumer : Consumer);
                    }))
              HasFailed = true;
            return;
          }
          // FIXME (Alex L): Implement non-selection based invocation path.
        }))
      return true;
    return HasFailed;
  }
};

} // end anonymous namespace

int main(int argc, const char **argv) {
  ClangRefactorTool Tool;

  // FIXME: Use LibTooling's CommonOptions parser when subcommands are supported
  // by it.
  cl::HideUnrelatedOptions(opts::CommonRefactorOptions);
  cl::ParseCommandLineOptions(
      argc, argv, "Clang-based refactoring tool for C, C++ and Objective-C");
  cl::PrintOptionValues();

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

  ArrayRef<std::string> Sources = ActionCommand.getSources();
  // When -no-dbs is used, at least one file (TU) must be given to any
  // subcommand.
  if (opts::NoDatabases && Sources.empty()) {
    llvm::errs() << "error: must provide paths to the source files when "
                    "'-no-dbs' is used\n";
    return 1;
  }
  if (ActionCommand.parseArguments())
    return 1;
  if (Tool.invokeAction(ActionCommand))
    return 1;

  return 0;
}
