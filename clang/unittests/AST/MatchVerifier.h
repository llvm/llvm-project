//===- unittest/AST/MatchVerifier.h - AST unit test support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Provides MatchVerifier, a base class to implement gtest matchers that
//  verify things that can be matched on the AST.
//
//  Also implements matchers based on MatchVerifier:
//  LocationVerifier and RangeVerifier to verify whether a matched node has
//  the expected source location or source range.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_AST_MATCHVERIFIER_H
#define LLVM_CLANG_UNITTESTS_AST_MATCHVERIFIER_H

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Testing/CommandLineArgs.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {

/// \brief Base class for verifying some property of nodes found by a matcher.
template <typename NodeType>
class MatchVerifier : public MatchFinder::MatchCallback {
public:
  template <typename MatcherType>
  testing::AssertionResult match(const std::string &Code,
                                 const MatcherType &AMatcher) {
    std::vector<std::string> Args;
    return match(Code, AMatcher, Args, Lang_CXX03);
  }

  template <typename MatcherType>
  testing::AssertionResult match(const std::string &Code,
                                 const MatcherType &AMatcher, TestLanguage L) {
    std::vector<std::string> Args;
    return match(Code, AMatcher, Args, L);
  }

  template <typename MatcherType>
  testing::AssertionResult
  match(const std::string &Code, const MatcherType &AMatcher,
        std::vector<std::string> &Args, TestLanguage L);

  template <typename MatcherType>
  testing::AssertionResult match(const Decl *D, const MatcherType &AMatcher);

protected:
  void run(const MatchFinder::MatchResult &Result) override;
  virtual void verify(const MatchFinder::MatchResult &Result,
                      const NodeType &Node) {}

  void setFailure(const Twine &Result) {
    Verified = false;
    VerifyResult = Result.str();
  }

  void setSuccess() {
    Verified = true;
  }

private:
  bool Verified;
  std::string VerifyResult;
};

/// \brief Runs a matcher over some code, and returns the result of the
/// verifier for the matched node.
template <typename NodeType>
template <typename MatcherType>
testing::AssertionResult
MatchVerifier<NodeType>::match(const std::string &Code,
                               const MatcherType &AMatcher,
                               std::vector<std::string> &Args, TestLanguage L) {
  MatchFinder Finder;
  Finder.addMatcher(AMatcher.bind(""), this);
  std::unique_ptr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));

  StringRef FileName;
  switch (L) {
  case Lang_C89:
    Args.push_back("-std=c89");
    FileName = "input.c";
    break;
  case Lang_C99:
    Args.push_back("-std=c99");
    FileName = "input.c";
    break;
  case Lang_CXX03:
    Args.push_back("-std=c++03");
    FileName = "input.cc";
    break;
  case Lang_CXX11:
    Args.push_back("-std=c++11");
    FileName = "input.cc";
    break;
  case Lang_CXX14:
    Args.push_back("-std=c++14");
    FileName = "input.cc";
    break;
  case Lang_CXX17:
    Args.push_back("-std=c++17");
    FileName = "input.cc";
    break;
  case Lang_CXX20:
    Args.push_back("-std=c++20");
    FileName = "input.cc";
    break;
  case Lang_OpenCL:
    FileName = "input.cl";
    break;
  case Lang_OBJCXX:
    FileName = "input.mm";
    break;
  }

  // Default to failure in case callback is never called
  setFailure("Could not find match");
  if (!tooling::runToolOnCodeWithArgs(Factory->create(), Code, Args, FileName))
    return testing::AssertionFailure() << "Parsing error";
  if (!Verified)
    return testing::AssertionFailure() << VerifyResult;
  return testing::AssertionSuccess();
}

/// \brief Runs a matcher over some AST, and returns the result of the
/// verifier for the matched node.
template <typename NodeType> template <typename MatcherType>
testing::AssertionResult MatchVerifier<NodeType>::match(
    const Decl *D, const MatcherType &AMatcher) {
  MatchFinder Finder;
  Finder.addMatcher(AMatcher.bind(""), this);

  setFailure("Could not find match");
  Finder.match(*D, D->getASTContext());

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
    // Callback has been called, default to success.
    setSuccess();
    verify(Result, *Node);
  }
}

template <>
inline void
MatchVerifier<DynTypedNode>::run(const MatchFinder::MatchResult &Result) {
  BoundNodes::IDToNodeMap M = Result.Nodes.getMap();
  BoundNodes::IDToNodeMap::const_iterator I = M.find("");
  if (I == M.end()) {
    setFailure("Node was not bound");
  } else {
    // Callback has been called, default to success.
    setSuccess();
    verify(Result, I->second);
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
  void verify(const MatchFinder::MatchResult &Result,
              const NodeType &Node) override {
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
  void verify(const MatchFinder::MatchResult &Result,
              const NodeType &Node) override {
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

/// \brief Verify whether a node's dump contains a given substring.
class DumpVerifier : public MatchVerifier<DynTypedNode> {
public:
  void expectSubstring(const std::string &Str) {
    ExpectSubstring = Str;
  }

protected:
  void verify(const MatchFinder::MatchResult &Result,
              const DynTypedNode &Node) override {
    std::string DumpStr;
    llvm::raw_string_ostream Dump(DumpStr);
    Node.dump(Dump, *Result.SourceManager);

    if (Dump.str().find(ExpectSubstring) == std::string::npos) {
      std::string MsgStr;
      llvm::raw_string_ostream Msg(MsgStr);
      Msg << "Expected dump substring <" << ExpectSubstring << ">, found <"
          << Dump.str() << '>';
      this->setFailure(Msg.str());
    }
  }

private:
  std::string ExpectSubstring;
};

/// \brief Verify whether a node's pretty print matches a given string.
class PrintVerifier : public MatchVerifier<DynTypedNode> {
public:
  void expectString(const std::string &Str) {
    ExpectString = Str;
  }

protected:
  void verify(const MatchFinder::MatchResult &Result,
              const DynTypedNode &Node) override {
    std::string PrintStr;
    llvm::raw_string_ostream Print(PrintStr);
    Node.print(Print, Result.Context->getPrintingPolicy());

    if (Print.str() != ExpectString) {
      std::string MsgStr;
      llvm::raw_string_ostream Msg(MsgStr);
      Msg << "Expected pretty print <" << ExpectString << ">, found <"
          << Print.str() << '>';
      this->setFailure(Msg.str());
    }
  }

private:
  std::string ExpectString;
};

} // end namespace ast_matchers
} // end namespace clang

#endif
