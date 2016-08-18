//===--- TestVisitor.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines utility templates for RecursiveASTVisitor related tests.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_TOOLING_TESTVISITOR_H
#define LLVM_CLANG_UNITTESTS_TOOLING_TESTVISITOR_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {

/// \brief Base class for simple RecursiveASTVisitor based tests.
///
/// This is a drop-in replacement for RecursiveASTVisitor itself, with the
/// additional capability of running it over a snippet of code.
///
/// Visits template instantiations and implicit code by default.
template <typename T>
class TestVisitor : public RecursiveASTVisitor<T> {
public:
  TestVisitor() { }

  virtual ~TestVisitor() { }

  enum Language {
    Lang_C,
    Lang_CXX98,
    Lang_CXX11,
    Lang_CXX14,
    Lang_OBJC,
    Lang_OBJCXX11,
    Lang_CXX = Lang_CXX98
  };

  /// \brief Runs the current AST visitor over the given code.
  bool runOver(StringRef Code, Language L = Lang_CXX) {
    std::vector<std::string> Args;
    switch (L) {
      case Lang_C: Args.push_back("-std=c99"); break;
      case Lang_CXX98: Args.push_back("-std=c++98"); break;
      case Lang_CXX11: Args.push_back("-std=c++11"); break;
      case Lang_CXX14: Args.push_back("-std=c++14"); break;
      case Lang_OBJC: Args.push_back("-ObjC"); break;
      case Lang_OBJCXX11:
        Args.push_back("-ObjC++");
        Args.push_back("-std=c++11");
        Args.push_back("-fblocks");
        break;
    }
    return tooling::runToolOnCodeWithArgs(CreateTestAction(), Code, Args);
  }

  bool shouldVisitTemplateInstantiations() const {
    return true;
  }

  bool shouldVisitImplicitCode() const {
    return true;
  }

protected:
  virtual ASTFrontendAction* CreateTestAction() {
    return new TestAction(this);
  }

  class FindConsumer : public ASTConsumer {
  public:
    FindConsumer(TestVisitor *Visitor) : Visitor(Visitor) {}

    void HandleTranslationUnit(clang::ASTContext &Context) override {
      Visitor->Context = &Context;
      Visitor->TraverseDecl(Context.getTranslationUnitDecl());
    }

  private:
    TestVisitor *Visitor;
  };

  class TestAction : public ASTFrontendAction {
  public:
    TestAction(TestVisitor *Visitor) : Visitor(Visitor) {}

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(CompilerInstance &, llvm::StringRef dummy) override {
      /// TestConsumer will be deleted by the framework calling us.
      return llvm::make_unique<FindConsumer>(Visitor);
    }

  protected:
    TestVisitor *Visitor;
  };

  ASTContext *Context;
};

/// \brief A RecursiveASTVisitor to check that certain matches are (or are
/// not) observed during visitation.
///
/// This is a RecursiveASTVisitor for testing the RecursiveASTVisitor itself,
/// and allows simple creation of test visitors running matches on only a small
/// subset of the Visit* methods.
template <typename T, template <typename> class Visitor = TestVisitor>
class ExpectedLocationVisitor : public Visitor<T> {
public:
  /// \brief Expect 'Match' *not* to occur at the given 'Line' and 'Column'.
  ///
  /// Any number of matches can be disallowed.
  void DisallowMatch(Twine Match, unsigned Line, unsigned Column) {
    DisallowedMatches.push_back(MatchCandidate(Match, Line, Column));
  }

  /// \brief Expect 'Match' to occur at the given 'Line' and 'Column'.
  ///
  /// Any number of expected matches can be set by calling this repeatedly.
  /// Each is expected to be matched 'Times' number of times. (This is useful in
  /// cases in which different AST nodes can match at the same source code
  /// location.)
  void ExpectMatch(Twine Match, unsigned Line, unsigned Column,
                   unsigned Times = 1) {
    ExpectedMatches.push_back(ExpectedMatch(Match, Line, Column, Times));
  }

  /// \brief Checks that all expected matches have been found.
  ~ExpectedLocationVisitor() override {
    for (typename std::vector<ExpectedMatch>::const_iterator
             It = ExpectedMatches.begin(), End = ExpectedMatches.end();
         It != End; ++It) {
      It->ExpectFound();
    }
  }

protected:
  /// \brief Checks an actual match against expected and disallowed matches.
  ///
  /// Implementations are required to call this with appropriate values
  /// for 'Name' during visitation.
  void Match(StringRef Name, SourceLocation Location) {
    const FullSourceLoc FullLocation = this->Context->getFullLoc(Location);

    for (typename std::vector<MatchCandidate>::const_iterator
             It = DisallowedMatches.begin(), End = DisallowedMatches.end();
         It != End; ++It) {
      EXPECT_FALSE(It->Matches(Name, FullLocation))
          << "Matched disallowed " << *It;
    }

    for (typename std::vector<ExpectedMatch>::iterator
             It = ExpectedMatches.begin(), End = ExpectedMatches.end();
         It != End; ++It) {
      It->UpdateFor(Name, FullLocation, this->Context->getSourceManager());
    }
  }

 private:
  struct MatchCandidate {
    std::string ExpectedName;
    unsigned LineNumber;
    unsigned ColumnNumber;

    MatchCandidate(Twine Name, unsigned LineNumber, unsigned ColumnNumber)
      : ExpectedName(Name.str()), LineNumber(LineNumber),
        ColumnNumber(ColumnNumber) {
    }

    bool Matches(StringRef Name, FullSourceLoc const &Location) const {
      return MatchesName(Name) && MatchesLocation(Location);
    }

    bool PartiallyMatches(StringRef Name, FullSourceLoc const &Location) const {
      return MatchesName(Name) || MatchesLocation(Location);
    }

    bool MatchesName(StringRef Name) const {
      return Name == ExpectedName;
    }

    bool MatchesLocation(FullSourceLoc const &Location) const {
      return Location.isValid() &&
          Location.getSpellingLineNumber() == LineNumber &&
          Location.getSpellingColumnNumber() == ColumnNumber;
    }

    friend std::ostream &operator<<(std::ostream &Stream,
                                    MatchCandidate const &Match) {
      return Stream << Match.ExpectedName
                    << " at " << Match.LineNumber << ":" << Match.ColumnNumber;
    }
  };

  struct ExpectedMatch {
    ExpectedMatch(Twine Name, unsigned LineNumber, unsigned ColumnNumber,
                  unsigned Times)
        : Candidate(Name, LineNumber, ColumnNumber), TimesExpected(Times),
          TimesSeen(0) {}

    void UpdateFor(StringRef Name, FullSourceLoc Location, SourceManager &SM) {
      if (Candidate.Matches(Name, Location)) {
        EXPECT_LT(TimesSeen, TimesExpected);
        ++TimesSeen;
      } else if (TimesSeen < TimesExpected &&
                 Candidate.PartiallyMatches(Name, Location)) {
        llvm::raw_string_ostream Stream(PartialMatches);
        Stream << ", partial match: \"" << Name << "\" at ";
        Location.print(Stream, SM);
      }
    }

    void ExpectFound() const {
      EXPECT_EQ(TimesExpected, TimesSeen)
          << "Expected \"" << Candidate.ExpectedName
          << "\" at " << Candidate.LineNumber
          << ":" << Candidate.ColumnNumber << PartialMatches;
    }

    MatchCandidate Candidate;
    std::string PartialMatches;
    unsigned TimesExpected;
    unsigned TimesSeen;
  };

  std::vector<MatchCandidate> DisallowedMatches;
  std::vector<ExpectedMatch> ExpectedMatches;
};
}

#endif
