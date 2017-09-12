//===- unittest/Tooling/RefactoringTestActionRulesTest.cpp ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ReplacementTest.h"
#include "RewriterTestContext.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/RefactoringActionRules.h"
#include "clang/Tooling/Refactoring/Rename/SymbolName.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Errc.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace refactoring_action_rules;

namespace {

class RefactoringActionRulesTest : public ::testing::Test {
protected:
  void SetUp() override {
    Context.Sources.setMainFileID(
        Context.createInMemoryFile("input.cpp", DefaultCode));
  }

  RewriterTestContext Context;
  std::string DefaultCode = std::string(100, 'a');
};

Expected<AtomicChanges>
createReplacements(const std::unique_ptr<RefactoringActionRule> &Rule,
                   RefactoringRuleContext &Context) {
  class Consumer final : public RefactoringResultConsumer {
    void handleError(llvm::Error Err) override { Result = std::move(Err); }

    void handle(AtomicChanges SourceReplacements) override {
      Result = std::move(SourceReplacements);
    }
    void handle(SymbolOccurrences Occurrences) override {
      RefactoringResultConsumer::handle(std::move(Occurrences));
    }

  public:
    Optional<Expected<AtomicChanges>> Result;
  };

  Consumer C;
  Rule->invoke(C, Context);
  return std::move(*C.Result);
}

TEST_F(RefactoringActionRulesTest, MyFirstRefactoringRule) {
  auto ReplaceAWithB =
      [](std::pair<selection::SourceSelectionRange, int> Selection)
      -> Expected<AtomicChanges> {
    const SourceManager &SM = Selection.first.getSources();
    SourceLocation Loc = Selection.first.getRange().getBegin().getLocWithOffset(
        Selection.second);
    AtomicChange Change(SM, Loc);
    llvm::Error E = Change.replace(SM, Loc, 1, "b");
    if (E)
      return std::move(E);
    return AtomicChanges{Change};
  };
  class SelectionRequirement : public selection::Requirement {
  public:
    std::pair<selection::SourceSelectionRange, int>
    evaluateSelection(selection::SourceSelectionRange Selection) const {
      return std::make_pair(Selection, 20);
    }
  };
  auto Rule = createRefactoringRule(ReplaceAWithB,
                                    requiredSelection(SelectionRequirement()));

  // When the requirements are satisifed, the rule's function must be invoked.
  {
    RefactoringRuleContext RefContext(Context.Sources);
    SourceLocation Cursor =
        Context.Sources.getLocForStartOfFile(Context.Sources.getMainFileID())
            .getLocWithOffset(10);
    RefContext.setSelectionRange({Cursor, Cursor});

    Expected<AtomicChanges> ErrorOrResult =
        createReplacements(Rule, RefContext);
    ASSERT_FALSE(!ErrorOrResult);
    AtomicChanges Result = std::move(*ErrorOrResult);
    ASSERT_EQ(Result.size(), 1u);
    std::string YAMLString =
        const_cast<AtomicChange &>(Result[0]).toYAMLString();

    ASSERT_STREQ("---\n"
                 "Key:             'input.cpp:30'\n"
                 "FilePath:        input.cpp\n"
                 "Error:           ''\n"
                 "InsertedHeaders: \n"
                 "RemovedHeaders:  \n"
                 "Replacements:    \n" // Extra whitespace here!
                 "  - FilePath:        input.cpp\n"
                 "    Offset:          30\n"
                 "    Length:          1\n"
                 "    ReplacementText: b\n"
                 "...\n",
                 YAMLString.c_str());
  }

  // When one of the requirements is not satisfied, invoke should return a
  // valid error.
  {
    RefactoringRuleContext RefContext(Context.Sources);
    Expected<AtomicChanges> ErrorOrResult =
        createReplacements(Rule, RefContext);

    ASSERT_TRUE(!ErrorOrResult);
    std::string Message;
    llvm::handleAllErrors(
        ErrorOrResult.takeError(),
        [&](llvm::StringError &Error) { Message = Error.getMessage(); });
    EXPECT_EQ(Message, "refactoring action can't be initiated with the "
                       "specified selection range");
  }
}

TEST_F(RefactoringActionRulesTest, ReturnError) {
  Expected<AtomicChanges> (*Func)(selection::SourceSelectionRange) =
      [](selection::SourceSelectionRange) -> Expected<AtomicChanges> {
    return llvm::make_error<llvm::StringError>(
        "Error", llvm::make_error_code(llvm::errc::invalid_argument));
  };
  auto Rule = createRefactoringRule(
      Func, requiredSelection(
                selection::identity<selection::SourceSelectionRange>()));

  RefactoringRuleContext RefContext(Context.Sources);
  SourceLocation Cursor =
      Context.Sources.getLocForStartOfFile(Context.Sources.getMainFileID());
  RefContext.setSelectionRange({Cursor, Cursor});
  Expected<AtomicChanges> Result = createReplacements(Rule, RefContext);

  ASSERT_TRUE(!Result);
  std::string Message;
  llvm::handleAllErrors(Result.takeError(), [&](llvm::StringError &Error) {
    Message = Error.getMessage();
  });
  EXPECT_EQ(Message, "Error");
}

TEST_F(RefactoringActionRulesTest, ReturnInitiationDiagnostic) {
  RefactoringRuleContext RefContext(Context.Sources);
  class SelectionRequirement : public selection::Requirement {
  public:
    Expected<Optional<int>>
    evaluateSelection(selection::SourceSelectionRange Selection) const {
      return llvm::make_error<llvm::StringError>(
          "bad selection", llvm::make_error_code(llvm::errc::invalid_argument));
    }
  };
  auto Rule = createRefactoringRule(
      [](int) -> Expected<AtomicChanges> {
        llvm::report_fatal_error("Should not run!");
      },
      requiredSelection(SelectionRequirement()));

  SourceLocation Cursor =
      Context.Sources.getLocForStartOfFile(Context.Sources.getMainFileID());
  RefContext.setSelectionRange({Cursor, Cursor});
  Expected<AtomicChanges> Result = createReplacements(Rule, RefContext);

  ASSERT_TRUE(!Result);
  std::string Message;
  llvm::handleAllErrors(Result.takeError(), [&](llvm::StringError &Error) {
    Message = Error.getMessage();
  });
  EXPECT_EQ(Message, "bad selection");
}

Optional<SymbolOccurrences> findOccurrences(RefactoringActionRule &Rule,
                                            RefactoringRuleContext &Context) {
  class Consumer final : public RefactoringResultConsumer {
    void handleError(llvm::Error) override {}
    void handle(SymbolOccurrences Occurrences) override {
      Result = std::move(Occurrences);
    }
    void handle(AtomicChanges Changes) override {
      RefactoringResultConsumer::handle(std::move(Changes));
    }

  public:
    Optional<SymbolOccurrences> Result;
  };

  Consumer C;
  Rule.invoke(C, Context);
  return std::move(C.Result);
}

TEST_F(RefactoringActionRulesTest, ReturnSymbolOccurrences) {
  auto Rule = createRefactoringRule(
      [](selection::SourceSelectionRange Selection)
          -> Expected<SymbolOccurrences> {
        SymbolOccurrences Occurrences;
        Occurrences.push_back(SymbolOccurrence(
            SymbolName("test"), SymbolOccurrence::MatchingSymbol,
            Selection.getRange().getBegin()));
        return std::move(Occurrences);
      },
      requiredSelection(
          selection::identity<selection::SourceSelectionRange>()));

  RefactoringRuleContext RefContext(Context.Sources);
  SourceLocation Cursor =
      Context.Sources.getLocForStartOfFile(Context.Sources.getMainFileID());
  RefContext.setSelectionRange({Cursor, Cursor});
  Optional<SymbolOccurrences> Result = findOccurrences(*Rule, RefContext);

  ASSERT_FALSE(!Result);
  SymbolOccurrences Occurrences = std::move(*Result);
  EXPECT_EQ(Occurrences.size(), 1u);
  EXPECT_EQ(Occurrences[0].getKind(), SymbolOccurrence::MatchingSymbol);
  EXPECT_EQ(Occurrences[0].getNameRanges().size(), 1u);
  EXPECT_EQ(Occurrences[0].getNameRanges()[0],
            SourceRange(Cursor, Cursor.getLocWithOffset(strlen("test"))));
}

} // end anonymous namespace
