#include "MDParser.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/AllocatorList.h"
#include <cassert>
#include <iterator>
#include <stack>

namespace clang {
namespace doc {
namespace md {

using namespace llvm;
// FIXME: These functions need to account for special characters, ASCII codes,
// etc.
static bool isEmphasisDelimiter(char Token) {
  if (Token == '*' || Token == '_')
    return true;
  return false;
}

static bool isEmpty(StringRef Line) {
  if (Line.size() == 0)
    return true;
  if (Line.size() == 1 && Line[0] == ' ')
    return true;
  for (auto &Char : Line) {
    if (Char != ' ')
      return false;
  }
  return true;
}

static MDType determineBlockType(StringRef Line) {
  StringRef TrimmedLine = Line.ltrim(' ');
  if (TrimmedLine.empty())
    return MDType::None;

  switch (TrimmedLine[0]) {
  case '*':
  case '-':
    return MDType::Paragraph;
  case '>':
    return MDType::Paragraph;
  default:
    return MDType::Paragraph;
  }
}

ContainerBlock *ASTContext::getRoot() { return Root; }

Block *ASTContext::allocate() {
  if (!Root) {
    Root = new (Arena) ContainerBlock();
    return Root;
  }

  return new (Arena) Block();
}

Block *ASTContext::allocate(MDType Type) {
  auto *NewNode = allocate();
  NewNode->Type = Type;
  return NewNode;
}

StringRef ASTContext::intern(Twine String) { return SSaver.save(String); }
StringRef ASTContext::intern(std::string &String) {
  return SSaver.save(String);
}
StringRef ASTContext::intern(StringRef String) { return SSaver.save(String); }

std::optional<std::pair<DelimiterContext, size_t>>
MarkdownParser::processDelimiters(StringRef Line, const size_t &Start) {
  size_t Idx = Start;
  while (Idx < Line.size() && Line[Idx] == Line[Start]) {
    ++Idx;
  }
  size_t DelimiterRunLength = Idx - Start;

  char Preceeding = (Start == 0) ? ' ' : Line[Start - 1];
  char Proceeding = (Idx >= Line.size()) ? ' ' : Line[Idx];

  bool LeftFlanking = !isWhitespace(Proceeding) &&
                      (!isPunctuation(Proceeding) || isWhitespace(Preceeding) ||
                       isPunctuation(Preceeding));
  bool RightFlanking = !isWhitespace(Preceeding) &&
                       (!isPunctuation(Preceeding) ||
                        isWhitespace(Proceeding) || isPunctuation(Proceeding));

  DelimiterContext Ctx;
  Ctx.Length = DelimiterRunLength;
  Ctx.DelimChar = Line[Start];
  Ctx.LeftFlanking = LeftFlanking;
  Ctx.RightFlanking = RightFlanking;
  Ctx.CanOpen = LeftFlanking;
  Ctx.CanClose = RightFlanking;

  if (LeftFlanking || RightFlanking)
    return std::make_pair(Ctx, Idx);
  return std::nullopt;
}

InlineContainer *
MarkdownParser::determineEmphasisNode(DelimiterContext &Opener,
                                      DelimiterContext &Closer) {
  auto &OpenerLength = Opener.Length;
  auto &CloserLength = Closer.Length;

  InlineContainer *PossibleNode;
  if (OpenerLength >= 2 && CloserLength >= 2) {
    PossibleNode = new (Ctx.Arena) StrongInline();
    OpenerLength -= 2;
    CloserLength -= 2;
    return PossibleNode;
  }

  if (OpenerLength == 1 && CloserLength == 1) {
    PossibleNode = new (Ctx.Arena) EmphasisInline();
    OpenerLength -= 1;
    CloserLength -= 1;
    return PossibleNode;
  }

  return nullptr;
}

void MarkdownParser::processEmphasis(LineNodeList &Stack, List<Inline> &Out) {
  auto It = Stack.begin();
  while (It != Stack.end()) {
    if (It->DelimiterCtx && It->DelimiterCtx->CanOpen) {
      auto CloseIt = std::next(It);
      while (CloseIt != Stack.end()) {
        if (CloseIt->DelimiterCtx && CloseIt->DelimiterCtx->CanClose &&
            CloseIt->DelimiterCtx->DelimChar == It->DelimiterCtx->DelimChar) {
          break;
        }
        ++CloseIt;
      }

      if (CloseIt != Stack.end() && It->DelimiterCtx && CloseIt->DelimiterCtx) {
        auto *Node =
            determineEmphasisNode(*It->DelimiterCtx, *CloseIt->DelimiterCtx);
        if (Node) {
          std::string InnerText;
          for (auto InnerIt = std::next(It); InnerIt != CloseIt; ++InnerIt)
            InnerText += InnerIt->Content.str();

          if (Node->Type == InlineType::Strong) {
            parse(StringRef(InnerText), Node->Children);
            Out.push_back(*Node);
          } else if (Node->Type == InlineType::Emphasis) {
            parse(StringRef(InnerText), Node->Children);
            Out.push_back(*Node);
          }
          It = std::next(CloseIt);
          continue;
        }
      }
    }

    appendText(Out, It->Content.str());
    ++It;
  }
}

void MarkdownParser::appendText(List<Inline> &Out, StringRef Text) {
  if (Text.empty())
    return;
  auto *Node = new (Ctx.Arena) TextInline(Ctx.intern(Text));
  Out.push_back(*Node);
}

void MarkdownParser::parse(InlineContainerBlock *Inline) {
  if (!Inline)
    return;
  if (!Inline->UnresolvedInlineText)
    return;
  parse(Inline->UnresolvedInlineText.value(), Inline->Children);
  Inline->UnresolvedInlineText.reset();
}

void MarkdownParser::parse(llvm::StringRef Line, List<Inline> &Out) {
  LineNodeList Nodes;
  size_t Idx = 0;
  while (Idx < Line.size()) {
    if (isEmphasisDelimiter(Line[Idx])) {
      if (auto Run = processDelimiters(Line, Idx)) {
        size_t End = Run->second;
        Nodes.emplace_back(Line.substr(Idx, End - Idx), Run->first);
        Idx = End;
        continue;
      }
    }

    // FIXME: support '_'
    size_t Next = Line.find_first_of("*", Idx);
    if (Next == StringRef::npos)
      Next = Line.size();
    Nodes.emplace_back(Line.substr(Idx, Next - Idx));
    Idx = Next;
  }

  processEmphasis(Nodes, Out);
}

void MarkdownParser::parse(std::vector<StringRef> &Lines) {
  assert(!Ctx.getRoot() && "ASTContext is single-use; create a new context");
  if (Ctx.getRoot())
    return;

  auto *Current = new (Ctx.Arena) ContainerBlock();
  Ctx.Root = Current;
  Current->Type = MDType::Document;
  std::stack<ContainerBlock *> OpenContainers;
  OpenContainers.push(Current);

  std::vector<InlineContainerBlock *> Inlines;
  for (auto &Line : Lines) {
    if (isEmpty(Line)) {
      State = MDType::None;
      continue;
    }

    auto CurrentBlockType = determineBlockType(Line);
    auto &Children = OpenContainers.top()->Children;

    if (State != CurrentBlockType) {
      State = CurrentBlockType;
      switch (CurrentBlockType) {
      case MDType::Paragraph: {
        auto *NewParagraph = new (Ctx.Arena) ParagraphBlock(Line);
        Children.push_back(*NewParagraph);
        Inlines.push_back(NewParagraph);
        break;
      }
      case MDType::None:
        break;
      }
      continue;
    }
  }
  // The only container left should be the root document node
  if (!(OpenContainers.size() == 1))
    return;
  OpenContainers.pop();

  for (auto &CurrentInline : Inlines) {
    if (!CurrentInline->UnresolvedInlineText.has_value())
      continue;
    parse(CurrentInline);
  }
}
} // namespace md
} // namespace doc
} // namespace clang
