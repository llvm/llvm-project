#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MD_PARSER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MD_PARSER_H
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include <list>

using namespace llvm;

namespace clang {
namespace doc {
using llvm::SmallString;
enum class MDState { Emphasis, Strong, None };

enum class MDType {
  Paragraph,
  Emphasis,
  Strong,
  Text,
  Softbreak,
};

enum class MDTokenType { LeftDelimiterRun, RightDelimiterRun, Text };

struct Node {
  SmallVector<Node*> Children;
  MDType Type;
  Node *Parent;
  std::string Content;
};

struct DelimiterContext {
  bool RightFlanking;
  bool LeftFlanking;
  bool CanOpen;
  bool CanClose;
  char DelimChar;
  // Since Content is a StringRef, we separately track the length so that we can
  // decrement when necessary without modifying the string.
  size_t Length;
};

/// A LineNode might be a valid delimiter run, text, or a delimiter run that
/// will later be merged with a text if there is no matching run e.g. ***foo.
/// @brief A preprocessing structure for tracking text in a line.
struct LineNode {
  StringRef Content;
  // Instantiated if the line is a delimiter run.
  std::optional<DelimiterContext> DelimiterContext;
};

class MarkdownParser {
  // MDState State;
  BumpPtrAllocator Arena;
  StringSaver Saver;

  /// If a delimiter is found, determine if it is a delimiter run, what type of
  /// run it is, and whether it can be an opener or closer.
  ///
  /// The CommonMark specification defines delimiter runs as:
  /// A delimiter run is either a sequence of one or more * or _ characters that
  /// is not preceded or followed by a non-backslash-escaped * or _ character
  ///
  /// A left-flanking delimiter run is a delimiter run that is (1) not followed
  /// by Unicode whitespace, and either (2a) not followed by a Unicode
  /// punctuation character, or (2b) followed by a Unicode punctuation character
  /// and preceded by Unicode whitespace or a Unicode punctuation character.
  ///
  /// A right-flanking delimiter run is a delimiter run that is (1) not preceded
  /// by Unicode whitespace, and either (2a) not preceded by a Unicode
  /// punctuation character, or (2b) preceded by a Unicode punctuation character
  /// and followed by Unicode whitespace or a Unicode punctuation character.
  ///
  /// @param IdxOrigin the index of * or _ that might start a delimiter run.
  /// @return A pair denoting the type of run and the index where the run stops
  std::pair<std::optional<DelimiterContext>, size_t>
  processDelimiters(SmallString<64> &Line, const size_t &Origin = 0);

  void parseLine(SmallString<64> &Line, Node *Current);
  std::list<Node *> processEmphasis(std::list<LineNode *> &Stack);
  void convertToNode(LineNode LN, Node *Parent);

  Node *reverseIterateLine(std::list<LineNode *> &Stack,
                           std::list<LineNode *>::iterator &It);

  Node *createTextNode(const std::list<LineNode *> &Text);

  std::string traverse(Node *Current);

  /// @param Lines An entire Document that resides in a comment.
  /// @return the root of a Markdown document.
  Node* parse(std::vector<SmallString<64>> &Lines);
public:
  MarkdownParser() : Arena(BumpPtrAllocator()), Saver(Arena) {}
  std::string render(std::vector<SmallString<64>> &Lines);
};
} // namespace doc
} // namespace clang
#endif
