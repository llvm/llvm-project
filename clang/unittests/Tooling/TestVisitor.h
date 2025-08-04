//===--- TestVisitor.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace detail {
// Use 'TestVisitor' or include 'CRTPTestVisitor.h' and use 'CRTPTestVisitor'
// instead of using this directly.
class TestVisitorHelper {
public:
  enum Language {
    Lang_C,
    Lang_CXX98,
    Lang_CXX11,
    Lang_CXX14,
    Lang_CXX17,
    Lang_CXX2a,
    Lang_OBJC,
    Lang_OBJCXX11,
    Lang_CXX = Lang_CXX98
  };

  /// \brief Runs the current AST visitor over the given code.
  bool runOver(StringRef Code, Language L = Lang_CXX) {
    std::vector<std::string> Args;
    switch (L) {
    case Lang_C:
      Args.push_back("-x");
      Args.push_back("c");
      break;
    case Lang_CXX98:
      Args.push_back("-std=c++98");
      break;
    case Lang_CXX11:
      Args.push_back("-std=c++11");
      break;
    case Lang_CXX14:
      Args.push_back("-std=c++14");
      break;
    case Lang_CXX17:
      Args.push_back("-std=c++17");
      break;
    case Lang_CXX2a:
      Args.push_back("-std=c++2a");
      break;
    case Lang_OBJC:
      Args.push_back("-ObjC");
      Args.push_back("-fobjc-runtime=macosx-10.12.0");
      break;
    case Lang_OBJCXX11:
      Args.push_back("-ObjC++");
      Args.push_back("-std=c++11");
      Args.push_back("-fblocks");
      break;
    }
    return tooling::runToolOnCodeWithArgs(CreateTestAction(), Code, Args);
  }

protected:
  TestVisitorHelper() = default;
  virtual ~TestVisitorHelper() = default;
  virtual void InvokeTraverseDecl(TranslationUnitDecl *D) = 0;

  virtual std::unique_ptr<ASTFrontendAction> CreateTestAction() {
    return std::make_unique<TestAction>(this);
  }

  class FindConsumer : public ASTConsumer {
  public:
    FindConsumer(TestVisitorHelper *Visitor) : Visitor(Visitor) {}

    void HandleTranslationUnit(clang::ASTContext &Context) override {
      Visitor->Context = &Context;
      Visitor->InvokeTraverseDecl(Context.getTranslationUnitDecl());
    }

  private:
    TestVisitorHelper *Visitor;
  };

  class TestAction : public ASTFrontendAction {
  public:
    TestAction(TestVisitorHelper *Visitor) : Visitor(Visitor) {}

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(CompilerInstance &, llvm::StringRef dummy) override {
      /// TestConsumer will be deleted by the framework calling us.
      return std::make_unique<FindConsumer>(Visitor);
    }

  protected:
    TestVisitorHelper *Visitor;
  };

  ASTContext *Context;
};

class ExpectedLocationVisitorHelper {
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
  virtual ~ExpectedLocationVisitorHelper() {
    // FIXME: Range-based for loop.
    for (std::vector<ExpectedMatch>::const_iterator
             It = ExpectedMatches.begin(),
             End = ExpectedMatches.end();
         It != End; ++It) {
      It->ExpectFound();
    }
  }

protected:
  virtual ASTContext *getASTContext() = 0;

  /// \brief Checks an actual match against expected and disallowed matches.
  ///
  /// Implementations are required to call this with appropriate values
  /// for 'Name' during visitation.
  void Match(StringRef Name, SourceLocation Location) {
    const FullSourceLoc FullLocation = getASTContext()->getFullLoc(Location);

    // FIXME: Range-based for loop.
    for (std::vector<MatchCandidate>::const_iterator
             It = DisallowedMatches.begin(),
             End = DisallowedMatches.end();
         It != End; ++It) {
      EXPECT_FALSE(It->Matches(Name, FullLocation))
          << "Matched disallowed " << *It;
    }

    // FIXME: Range-based for loop.
    for (std::vector<ExpectedMatch>::iterator It = ExpectedMatches.begin(),
                                              End = ExpectedMatches.end();
         It != End; ++It) {
      It->UpdateFor(Name, FullLocation, getASTContext()->getSourceManager());
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
} // namespace detail

/// \brief Base class for simple (Dynamic)RecursiveASTVisitor based tests.
///
/// This is a drop-in replacement for DynamicRecursiveASTVisitor itself, with
/// the additional capability of running it over a snippet of code.
///
/// Visits template instantiations and implicit code by default.
///
/// For post-order traversal etc. use CTRPTestVisitor from
/// CTRPTestVisitor.h instead.
class TestVisitor : public DynamicRecursiveASTVisitor,
                    public detail::TestVisitorHelper {
public:
  TestVisitor() {
    ShouldVisitTemplateInstantiations = true;
    ShouldVisitImplicitCode = true;
  }

  void InvokeTraverseDecl(TranslationUnitDecl *D) override { TraverseDecl(D); }
};

/// \brief A RecursiveASTVisitor to check that certain matches are (or are
/// not) observed during visitation.
///
/// This is a RecursiveASTVisitor for testing the RecursiveASTVisitor itself,
/// and allows simple creation of test visitors running matches on only a small
/// subset of the Visit* methods.
///
/// For post-order traversal etc. use CTRPExpectedLocationVisitor from
/// CTRPTestVisitor.h instead.
class ExpectedLocationVisitor : public TestVisitor,
                                public detail::ExpectedLocationVisitorHelper {
  ASTContext *getASTContext() override { return Context; }
};
} // namespace clang

#endif
