//===--- UnwrappedLineParser.h - Format C++ code ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declaration of the UnwrappedLineParser,
/// which turns a stream of tokens into UnwrappedLines.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEPARSER_H
#define LLVM_CLANG_LIB_FORMAT_UNWRAPPEDLINEPARSER_H

#include "FormatToken.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Format/Format.h"
#include <list>
#include <stack>

namespace clang {
namespace format {

struct UnwrappedLineNode;

/// \brief An unwrapped line is a sequence of \c Token, that we would like to
/// put on a single line if there was no column limit.
///
/// This is used as a main interface between the \c UnwrappedLineParser and the
/// \c UnwrappedLineFormatter. The key property is that changing the formatting
/// within an unwrapped line does not affect any other unwrapped lines.
struct UnwrappedLine {
  UnwrappedLine();

  // FIXME: Don't use std::list here.
  /// \brief The \c Tokens comprising this \c UnwrappedLine.
  std::list<UnwrappedLineNode> Tokens;

  /// \brief The indent level of the \c UnwrappedLine.
  unsigned Level;

  /// \brief Whether this \c UnwrappedLine is part of a preprocessor directive.
  bool InPPDirective;

  bool MustBeDeclaration;
};

class UnwrappedLineConsumer {
public:
  virtual ~UnwrappedLineConsumer() {}
  virtual void consumeUnwrappedLine(const UnwrappedLine &Line) = 0;
  virtual void finishRun() = 0;
};

class FormatTokenSource;

class UnwrappedLineParser {
public:
  UnwrappedLineParser(const FormatStyle &Style,
                      const AdditionalKeywords &Keywords,
                      ArrayRef<FormatToken *> Tokens,
                      UnwrappedLineConsumer &Callback);

  void parse();

private:
  void reset();
  void parseFile();
  void parseLevel(bool HasOpeningBrace);
  void parseBlock(bool MustBeDeclaration, bool AddLevel = true,
                  bool MunchSemi = true);
  void parseChildBlock();
  void parsePPDirective();
  void parsePPDefine();
  void parsePPIf(bool IfDef);
  void parsePPElIf();
  void parsePPElse();
  void parsePPEndIf();
  void parsePPUnknown();
  void parseStructuralElement();
  bool tryToParseBracedList();
  bool parseBracedList(bool ContinueOnSemicolons = false);
  void parseParens();
  void parseSquare();
  void parseIfThenElse();
  void parseTryCatch();
  void parseForOrWhileLoop();
  void parseDoWhile();
  void parseLabel();
  void parseCaseLabel();
  void parseSwitch();
  void parseNamespace();
  void parseNew();
  void parseAccessSpecifier();
  bool parseEnum();
  void parseJavaEnumBody();
  void parseRecord();
  void parseObjCProtocolList();
  void parseObjCUntilAtEnd();
  void parseObjCInterfaceOrImplementation();
  void parseObjCProtocol();
  void parseJavaScriptEs6ImportExport();
  bool tryToParseLambda();
  bool tryToParseLambdaIntroducer();
  void tryToParseJSFunction();
  void addUnwrappedLine();
  bool eof() const;
  void nextToken();
  void readToken();
  void flushComments(bool NewlineBeforeNext);
  void pushToken(FormatToken *Tok);
  void calculateBraceTypes(bool ExpectClassBody = false);

  // Marks a conditional compilation edge (for example, an '#if', '#ifdef',
  // '#else' or merge conflict marker). If 'Unreachable' is true, assumes
  // this branch either cannot be taken (for example '#if false'), or should
  // not be taken in this round.
  void conditionalCompilationCondition(bool Unreachable);
  void conditionalCompilationStart(bool Unreachable);
  void conditionalCompilationAlternative();
  void conditionalCompilationEnd();

  bool isOnNewLine(const FormatToken &FormatTok);

  // FIXME: We are constantly running into bugs where Line.Level is incorrectly
  // subtracted from beyond 0. Introduce a method to subtract from Line.Level
  // and use that everywhere in the Parser.
  std::unique_ptr<UnwrappedLine> Line;

  // Comments are sorted into unwrapped lines by whether they are in the same
  // line as the previous token, or not. If not, they belong to the next token.
  // Since the next token might already be in a new unwrapped line, we need to
  // store the comments belonging to that token.
  SmallVector<FormatToken *, 1> CommentsBeforeNextToken;
  FormatToken *FormatTok;
  bool MustBreakBeforeNextToken;

  // The parsed lines. Only added to through \c CurrentLines.
  SmallVector<UnwrappedLine, 8> Lines;

  // Preprocessor directives are parsed out-of-order from other unwrapped lines.
  // Thus, we need to keep a list of preprocessor directives to be reported
  // after an unwarpped line that has been started was finished.
  SmallVector<UnwrappedLine, 4> PreprocessorDirectives;

  // New unwrapped lines are added via CurrentLines.
  // Usually points to \c &Lines. While parsing a preprocessor directive when
  // there is an unfinished previous unwrapped line, will point to
  // \c &PreprocessorDirectives.
  SmallVectorImpl<UnwrappedLine> *CurrentLines;

  // We store for each line whether it must be a declaration depending on
  // whether we are in a compound statement or not.
  std::vector<bool> DeclarationScopeStack;

  const FormatStyle &Style;
  const AdditionalKeywords &Keywords;

  FormatTokenSource *Tokens;
  UnwrappedLineConsumer &Callback;

  // FIXME: This is a temporary measure until we have reworked the ownership
  // of the format tokens. The goal is to have the actual tokens created and
  // owned outside of and handed into the UnwrappedLineParser.
  ArrayRef<FormatToken *> AllTokens;

  // Represents preprocessor branch type, so we can find matching
  // #if/#else/#endif directives.
  enum PPBranchKind {
    PP_Conditional, // Any #if, #ifdef, #ifndef, #elif, block outside #if 0
    PP_Unreachable  // #if 0 or a conditional preprocessor block inside #if 0
  };

  // Keeps a stack of currently active preprocessor branching directives.
  SmallVector<PPBranchKind, 16> PPStack;

  // The \c UnwrappedLineParser re-parses the code for each combination
  // of preprocessor branches that can be taken.
  // To that end, we take the same branch (#if, #else, or one of the #elif
  // branches) for each nesting level of preprocessor branches.
  // \c PPBranchLevel stores the current nesting level of preprocessor
  // branches during one pass over the code.
  int PPBranchLevel;

  // Contains the current branch (#if, #else or one of the #elif branches)
  // for each nesting level.
  SmallVector<int, 8> PPLevelBranchIndex;

  // Contains the maximum number of branches at each nesting level.
  SmallVector<int, 8> PPLevelBranchCount;

  // Contains the number of branches per nesting level we are currently
  // in while parsing a preprocessor branch sequence.
  // This is used to update PPLevelBranchCount at the end of a branch
  // sequence.
  std::stack<int> PPChainBranchIndex;

  friend class ScopedLineState;
  friend class CompoundStatementIndenter;
};

struct UnwrappedLineNode {
  UnwrappedLineNode() : Tok(nullptr) {}
  UnwrappedLineNode(FormatToken *Tok) : Tok(Tok) {}

  FormatToken *Tok;
  SmallVector<UnwrappedLine, 0> Children;
};

inline UnwrappedLine::UnwrappedLine()
    : Level(0), InPPDirective(false), MustBeDeclaration(false) {}

} // end namespace format
} // end namespace clang

#endif
