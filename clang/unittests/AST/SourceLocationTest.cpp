//===- unittest/AST/SourceLocationTest.cpp - AST source loc unit tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for SourceLocation and SourceRange fields
// in AST nodes.
//
// FIXME: In the long-term, when we test more than source locations, we may
// want to have a unit test file for an AST node (or group of related nodes),
// rather than a unit test file for source locations for all AST nodes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

using clang::tooling::newFrontendActionFactory;
using clang::tooling::runToolOnCodeWithArgs;
using clang::tooling::FrontendActionFactory;

enum Language { Lang_C, Lang_C89, Lang_CXX };

/// \brief Base class for verifying some property of nodes found by a matcher.
///
/// FIXME: This class should be shared with other AST tests.
template <typename NodeType>
class MatchVerifier : public MatchFinder::MatchCallback {
public:
  template <typename MatcherType>
  testing::AssertionResult match(const std::string &Code,
                                 const MatcherType &AMatcher) {
    return match(Code, AMatcher, Lang_CXX);
  }

  template <typename MatcherType>
  testing::AssertionResult match(const std::string &Code,
                                 const MatcherType &AMatcher, Language L);

protected:
  virtual void run(const MatchFinder::MatchResult &Result);
  virtual void verify(const MatchFinder::MatchResult &Result,
                      const NodeType &Node) = 0;

  void setFailure(const Twine &Result) {
    Verified = false;
    VerifyResult = Result.str();
  }

private:
  bool Verified;
  std::string VerifyResult;
};

/// \brief Runs a matcher over some code, and returns the result of the
/// verifier for the matched node.
template <typename NodeType> template <typename MatcherType>
testing::AssertionResult MatchVerifier<NodeType>::match(
    const std::string &Code, const MatcherType &AMatcher, Language L) {
  MatchFinder Finder;
  Finder.addMatcher(AMatcher.bind(""), this);
  OwningPtr<FrontendActionFactory> Factory(newFrontendActionFactory(&Finder));

  std::vector<std::string> Args;
  StringRef FileName;
  switch (L) {
  case Lang_C:
    Args.push_back("-std=c99");
    FileName = "input.c";
    break;
  case Lang_C89:
    Args.push_back("-std=c89");
    FileName = "input.c";
    break;
  case Lang_CXX:
    Args.push_back("-std=c++98");
    FileName = "input.cc";
    break;
  }

  // Default to failure in case callback is never called
  setFailure("Could not find match");
  if (!runToolOnCodeWithArgs(Factory->create(), Code, Args, FileName))
    return testing::AssertionFailure() << "Parsing error";
  if (!Verified)
    return testing::AssertionFailure() << VerifyResult;
  return testing::AssertionSuccess();
}

template <typename NodeType>
void MatchVerifier<NodeType>::run(const MatchFinder::MatchResult &Result) {
  const NodeType *Node = Result.Nodes.getNodeAs<NodeType>("");
  if (!Node) {
    setFailure("Matched node has wrong type");
  } else {
    // Callback has been called, default to success
    Verified = true;
    verify(Result, *Node);
  }
}

/// \brief Verify whether a node has the correct source location.
///
/// By default, Node.getSourceLocation() is checked. This can be changed
/// by overriding getLocation().
template <typename NodeType>
class LocationVerifier : public MatchVerifier<NodeType> {
public:
  void expectLocation(unsigned Line, unsigned Column) {
    ExpectLine = Line;
    ExpectColumn = Column;
  }

protected:
  void verify(const MatchFinder::MatchResult &Result, const NodeType &Node) {
    SourceLocation Loc = getLocation(Node);
    unsigned Line = Result.SourceManager->getSpellingLineNumber(Loc);
    unsigned Column = Result.SourceManager->getSpellingColumnNumber(Loc);
    if (Line != ExpectLine || Column != ExpectColumn) {
      std::string MsgStr;
      llvm::raw_string_ostream Msg(MsgStr);
      Msg << "Expected location <" << ExpectLine << ":" << ExpectColumn
          << ">, found <";
      Loc.print(Msg, *Result.SourceManager);
      Msg << '>';
      this->setFailure(Msg.str());
    }
  }

  virtual SourceLocation getLocation(const NodeType &Node) {
    return Node.getLocation();
  }

private:
  unsigned ExpectLine, ExpectColumn;
};

/// \brief Verify whether a node has the correct source range.
///
/// By default, Node.getSourceRange() is checked. This can be changed
/// by overriding getRange().
template <typename NodeType>
class RangeVerifier : public MatchVerifier<NodeType> {
public:
  void expectRange(unsigned BeginLine, unsigned BeginColumn,
                   unsigned EndLine, unsigned EndColumn) {
    ExpectBeginLine = BeginLine;
    ExpectBeginColumn = BeginColumn;
    ExpectEndLine = EndLine;
    ExpectEndColumn = EndColumn;
  }

protected:
  void verify(const MatchFinder::MatchResult &Result, const NodeType &Node) {
    SourceRange R = getRange(Node);
    SourceLocation Begin = R.getBegin();
    SourceLocation End = R.getEnd();
    unsigned BeginLine = Result.SourceManager->getSpellingLineNumber(Begin);
    unsigned BeginColumn = Result.SourceManager->getSpellingColumnNumber(Begin);
    unsigned EndLine = Result.SourceManager->getSpellingLineNumber(End);
    unsigned EndColumn = Result.SourceManager->getSpellingColumnNumber(End);
    if (BeginLine != ExpectBeginLine || BeginColumn != ExpectBeginColumn ||
        EndLine != ExpectEndLine || EndColumn != ExpectEndColumn) {
      std::string MsgStr;
      llvm::raw_string_ostream Msg(MsgStr);
      Msg << "Expected range <" << ExpectBeginLine << ":" << ExpectBeginColumn
          << '-' << ExpectEndLine << ":" << ExpectEndColumn << ">, found <";
      Begin.print(Msg, *Result.SourceManager);
      Msg << '-';
      End.print(Msg, *Result.SourceManager);
      Msg << '>';
      this->setFailure(Msg.str());
    }
  }

  virtual SourceRange getRange(const NodeType &Node) {
    return Node.getSourceRange();
  }

private:
  unsigned ExpectBeginLine, ExpectBeginColumn, ExpectEndLine, ExpectEndColumn;
};

TEST(MatchVerifier, ParseError) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i", varDecl()));
}

TEST(MatchVerifier, NoMatch) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", recordDecl()));
}

TEST(MatchVerifier, WrongType) {
  LocationVerifier<RecordDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

TEST(LocationVerifier, WrongLocation) {
  LocationVerifier<VarDecl> Verifier;
  Verifier.expectLocation(1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

TEST(RangeVerifier, WrongRange) {
  RangeVerifier<VarDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 1);
  EXPECT_FALSE(Verifier.match("int i;", varDecl()));
}

class LabelDeclRangeVerifier : public RangeVerifier<LabelStmt> {
protected:
  virtual SourceRange getRange(const LabelStmt &Node) {
    return Node.getDecl()->getSourceRange();
  }
};

TEST(LabelDecl, Range) {
  LabelDeclRangeVerifier Verifier;
  Verifier.expectRange(1, 12, 1, 12);
  EXPECT_TRUE(Verifier.match("void f() { l: return; }", labelStmt()));
}

TEST(LabelStmt, Range) {
  RangeVerifier<LabelStmt> Verifier;
  Verifier.expectRange(1, 12, 1, 15);
  EXPECT_TRUE(Verifier.match("void f() { l: return; }", labelStmt()));
}

TEST(ParmVarDecl, KNRLocation) {
  LocationVerifier<ParmVarDecl> Verifier;
  Verifier.expectLocation(1, 8);
  EXPECT_TRUE(Verifier.match("void f(i) {}", varDecl(), Lang_C));
}

TEST(ParmVarDecl, KNRRange) {
  RangeVerifier<ParmVarDecl> Verifier;
  Verifier.expectRange(1, 8, 1, 8);
  EXPECT_TRUE(Verifier.match("void f(i) {}", varDecl(), Lang_C));
}

TEST(CXXNewExpr, ArrayRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 12, 1, 22);
  EXPECT_TRUE(Verifier.match("void f() { new int[10]; }", newExpr()));
}

TEST(CXXNewExpr, ParenRange) {
  RangeVerifier<CXXNewExpr> Verifier;
  Verifier.expectRange(1, 12, 1, 20);
  EXPECT_TRUE(Verifier.match("void f() { new int(); }", newExpr()));
}

TEST(MemberExpr, ImplicitMemberRange) {
  RangeVerifier<MemberExpr> Verifier;
  Verifier.expectRange(2, 30, 2, 30);
  EXPECT_TRUE(Verifier.match("struct S { operator int() const; };\n"
                             "int foo(const S& s) { return s; }",
                             memberExpr()));
}

TEST(VarDecl, VMTypeFixedVarDeclRange) {
  RangeVerifier<VarDecl> Verifier;
  Verifier.expectRange(1, 1, 1, 23);
  EXPECT_TRUE(Verifier.match("int a[(int)(void*)1234];",
                             varDecl(), Lang_C89));
}

TEST(CXXConstructorDecl, NoRetFunTypeLocRange) {
  RangeVerifier<CXXConstructorDecl> Verifier;
  Verifier.expectRange(1, 11, 1, 13);
  EXPECT_TRUE(Verifier.match("class C { C(); };", functionDecl()));
}

} // end namespace ast_matchers
} // end namespace clang
