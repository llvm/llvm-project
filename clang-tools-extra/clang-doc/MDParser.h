#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MD_PARSER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MD_PARSER_H
#include "llvm/ADT/AllocatorList.h"
#include "llvm/ADT/simple_ilist.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace doc {
namespace md {

enum class MDType { Paragraph, Document, None };

enum class InlineType { Emphasis, Strong, Text };

enum class MDTokenType { LeftDelimiterRun, RightDelimiterRun, Text };

template <class T>
using List = llvm::simple_ilist<T, llvm::ilist_sentinel_tracking<true>>;

struct Block : llvm::ilist_node<Block, llvm::ilist_sentinel_tracking<true>> {
  Block() = default;
  Block(MDType Ty) : Type(Ty) {}
  Block(MDType Ty, Block *Parent) : Parent(Parent), Type(Ty) {}

  Block *Parent;
  MDType Type;
};

struct ContainerBlock : Block {
  List<Block> Children;
};

struct Inline;

/// A block that can contain inline elements.
struct InlineContainerBlock : Block {
  InlineContainerBlock() = default;
  InlineContainerBlock(MDType Type) : Block(Type) {}
  InlineContainerBlock(MDType Type, llvm::StringRef Line)
      : Block(Type), UnresolvedInlineText(Line) {}
  List<Inline> Children;

  std::optional<llvm::StringRef> UnresolvedInlineText;
};

struct ParagraphBlock : InlineContainerBlock {
  ParagraphBlock() : InlineContainerBlock(MDType::Paragraph) {}
  ParagraphBlock(llvm::StringRef Line)
      : InlineContainerBlock(MDType::Paragraph, Line) {}
};

struct Inline : llvm::ilist_node<Inline, llvm::ilist_sentinel_tracking<true>> {
  Inline() = default;
  Inline(InlineType Type) : Type(Type) {}
  InlineType Type;
};

/// An inline element that can contain other inline elements.
struct InlineContainer : Inline {
  InlineContainer() = default;
  InlineContainer(InlineType Type) : Inline(Type) {}
  List<Inline> Children;
};

struct TextInline : Inline {
  TextInline(llvm::StringRef Text) : Inline(InlineType::Text), Text(Text) {}
  llvm::StringRef Text;
};

struct EmphasisInline : InlineContainer {
  EmphasisInline() : InlineContainer(InlineType::Emphasis) {}
};

struct StrongInline : InlineContainer {
  StrongInline() : InlineContainer(InlineType::Strong) {}
};

class MarkdownParser;

class ASTContext {
  llvm::BumpPtrAllocator Arena;
  llvm::StringSaver SSaver;
  ContainerBlock *Root;

  friend MarkdownParser;

public:
  ASTContext()
      : Arena(llvm::BumpPtrAllocator()), SSaver(Arena), Root(nullptr) {}
  ContainerBlock *getRoot();
  Block *allocate();
  Block *allocate(MDType Type);
  llvm::StringRef intern(llvm::Twine String);
  llvm::StringRef intern(std::string &String);
  llvm::StringRef intern(llvm::StringRef String);
};

struct DelimiterContext {
  // Since Content is a StringRef, we separately track the length so that we can
  // decrement when necessary without modifying the string.
  size_t Length;
  char DelimChar;
  bool RightFlanking;
  bool LeftFlanking;
  bool CanOpen = false;
  bool CanClose = false;
};

/// \brief A temporary structure for tracking text in a line.
///
/// A LineNode might be a valid delimiter run, text, or a delimiter run that
/// will later be merged with a text if there is no matching run e.g. ***foo.
///
/// Line nodes live in a BumpPtrList, so they will be destroyed once a line is
/// parsed.
struct LineNode
    : llvm::ilist_node<LineNode, llvm::ilist_sentinel_tracking<true>> {
  llvm::Twine Content;
  // Instantiated if the line is a delimiter run.
  std::optional<DelimiterContext> DelimiterCtx;

  LineNode() : DelimiterCtx(std::nullopt) {}
  LineNode(llvm::StringRef Content)
      : Content(Content), DelimiterCtx(std::nullopt) {}
  LineNode(llvm::StringRef Content, DelimiterContext Ctx)
      : Content(Content), DelimiterCtx(Ctx) {
    if (DelimiterCtx)
      DelimiterCtx->Length = Content.size();
  }
  LineNode(llvm::StringRef Content, std::optional<DelimiterContext> Ctx)
      : Content(Content), DelimiterCtx(Ctx) {
    if (DelimiterCtx)
      DelimiterCtx->Length = Content.size();
  }
};

/// \brief A Markdown parser based on the CommonMark specification
///
/// The parser constructs an AST from a collection of strings.
class MarkdownParser {
  ASTContext &Ctx;
  MDType State = MDType::None;

  using LineNodeList = llvm::BumpPtrList<LineNode>;

  /// \brief Determine whether a substring is a delimiter run, what type of
  /// run it is, and if it can be an opener or closer.
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
  std::optional<std::pair<DelimiterContext, size_t>>
  processDelimiters(llvm::StringRef Line, const size_t &Origin = 0);

  void appendText(List<Inline> &Out, llvm::StringRef Text);

  void processEmphasis(LineNodeList &Stack, List<Inline> &Out);

  /// \brief Combine text within a range of nodes
  template <class Iterator, class ReverseIterator>
  void gatherTextNodes(Block *Parent, Iterator Start, ReverseIterator End);
  InlineContainer *determineEmphasisNode(DelimiterContext &Opener,
                                         DelimiterContext &Closer);

  void parse(InlineContainerBlock *Inline);
  void parse(llvm::StringRef Line, List<Inline> &Out);

public:
  MarkdownParser(ASTContext &Ctx) : Ctx(Ctx) {}

  /// \brief Parses a collection of strings to construct a Document.
  void parse(std::vector<llvm::StringRef> &Lines);
};
} // namespace md
} // namespace doc
} // namespace clang
#endif
