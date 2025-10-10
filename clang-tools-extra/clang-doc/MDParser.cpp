#include "MDParser.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/AllocatorList.h"

namespace clang {
namespace doc {
namespace {
bool isEmphasisDelimiter(char &Token) {
  // TODO: support '_'
  if (Token == '*')
    return true;
  return false;
}
} // namespace

std::pair<std::optional<DelimiterContext>, size_t>
MarkdownParser::processDelimiters(SmallString<64> &Line, const size_t &Origin) {
  size_t Idx = Origin;
  while (Idx < Line.size() && Line[Idx] == Line[Origin]) {
    ++Idx;
  }

  char Preceeding = (Origin == 0) ? ' ' : Line[Origin - 1];
  char Proceeding = (Idx >= Line.size()) ? ' ' : Line[Idx];

  bool LeftFlanking = !isWhitespace(Proceeding) &&
                      (!isPunctuation(Proceeding) || isWhitespace(Preceeding) ||
                       isPunctuation(Preceeding));
  bool RightFlanking = !isWhitespace(Preceeding) &&
                       (!isPunctuation(Preceeding) || isWhitespace(Proceeding) ||
                        isPunctuation(Proceeding));

  if (LeftFlanking && RightFlanking)
    return {DelimiterContext{LeftFlanking, RightFlanking, true, true}, Idx};
  if (LeftFlanking)
    return {DelimiterContext{LeftFlanking, RightFlanking, true, false}, Idx};
  if (RightFlanking)
    return {DelimiterContext{LeftFlanking, RightFlanking, false, true}, Idx};
  return {std::nullopt, 0};
}

Node *MarkdownParser::createTextNode(const std::list<LineNode *> &Text) {
  Node *TextNode = new (Arena) Node();
  for (const auto *Node : Text) {
    TextNode->Content.append(Node->Content);
  }
  TextNode->Type = MDType::Text;
  return TextNode;
}

Node *MarkdownParser::reverseIterateLine(std::list<LineNode *> &Stack,
                                         std::list<LineNode *>::iterator &It) {
  auto ReverseIt = std::make_reverse_iterator(It);
  std::list<LineNode *> Text;
  while (ReverseIt != Stack.rend()) {
    auto *ReverseCurrent = *ReverseIt;
    if (!ReverseCurrent->DelimiterContext && !ReverseCurrent->Content.empty()) {
      Text.push_back(ReverseCurrent);
      ReverseIt++;
      continue;
    }

    if (ReverseCurrent->DelimiterContext &&
        ReverseCurrent->DelimiterContext->CanOpen) {
      if (Text.empty()) {
        // If there is no text between the runs, there is no emphasis, so both
        // delimiter runs are literal text.
        auto *DelimiterTextNode = new (Arena) Node();
        DelimiterTextNode->Content =
            Saver.save((*It)->Content + ReverseCurrent->Content);
        DelimiterTextNode->Type = MDType::Text;
        return DelimiterTextNode;
      }
      Node *Emphasis = nullptr;

      auto &Closer = (*It)->DelimiterContext;
      auto &Opener = ReverseCurrent->DelimiterContext;

      if (Closer->Length >= 2 && Opener->Length >= 2) {
        // We have at least one strong node.
        Closer->Length -= 2;
        Opener->Length -= 2;
        Emphasis = new (Arena) Node();
        Emphasis->Type = MDType::Strong;
        auto *Child = createTextNode(Text);
        Child->Parent = Emphasis;
        Emphasis->Children.push_back(Child);
      } else if (Closer->Length == 1 && Opener->Length == 1) {
        Closer->Length -= 1;
        Opener->Length -= 1;
        Emphasis = new (Arena) Node();
        Emphasis->Type = MDType::Emphasis;
        auto *Child = createTextNode(Text);
        Child->Parent = Emphasis;
        Emphasis->Children.push_back(Child);
      }

      if (Closer->Length == 0)
        It = Stack.erase(It);
      if (Opener->Length == 0)
        ReverseIt = std::make_reverse_iterator(Stack.erase(ReverseIt.base()));
      if (!Text.empty())
        for (auto *Node : Text)
          Stack.remove(Node);
      return Emphasis;
    }
    ReverseIt++;
  }
  return nullptr;
}

std::list<Node *>
MarkdownParser::processEmphasis(std::list<LineNode *> &Stack) {
  std::list<Node *> Result;
  auto It = Stack.begin();
  while (It != Stack.end()) {
    LineNode *Current = *It;
    if (Current->DelimiterContext && Current->DelimiterContext->CanClose) {
      auto *NewNode = reverseIterateLine(Stack, It);
      if (NewNode) {
        Result.push_back(NewNode);
        It = Stack.begin();
        continue;
      }
    }
    ++It;
  }

  return Result;
}

void MarkdownParser::parseLine(SmallString<64> &Line, Node *Current) {
  std::list<LineNode *> Stack;
  BumpPtrAllocator LineArena;
  size_t StrCount = 0;
  size_t Idx = 0;
  for (; Idx < Line.size(); ++Idx) {
    if (isEmphasisDelimiter(Line[Idx])) {
      auto DelimiterResult = processDelimiters(Line, Idx);
      if (DelimiterResult.first != std::nullopt) {
        if (StrCount > 0) {
          auto *TextNode = new (LineArena) LineNode();
          TextNode->Content = Line.substr(Idx - StrCount, StrCount);
          Stack.push_back(TextNode);
          StrCount = 0;
        }
        auto *NewNode = new (LineArena) LineNode();
        NewNode->Content = Line.substr(Idx, DelimiterResult.second - Idx);
        NewNode->DelimiterContext = std::move(DelimiterResult.first);
        NewNode->DelimiterContext->Length = NewNode->Content.size();
        Stack.push_back(NewNode);
        Idx = DelimiterResult.second - 1;
        continue;
      }
    }
    // Not any emphasis delimiter, so it will be appended as a string later
    StrCount += 1;
  }

  if (StrCount > 0) {
    auto *TextNode = new (LineArena) LineNode();
    TextNode->Content = Line.substr(Line.size() - StrCount, StrCount);
    Stack.push_back(TextNode);
  }

  auto Resolved = processEmphasis(Stack);
  for (auto *Node : Resolved) {
    Node->Parent = Current;
    Current->Children.push_back(Node);
  }
}

Node *MarkdownParser::parse(std::vector<SmallString<64>> &Lines) {
  auto *Root = new (Arena) Node();
  Node *Current = Root;
  for (auto &Line : Lines) {
    if (Line.empty()) {
      auto *Paragraph = new (Arena) Node();
      Paragraph->Type = MDType::Paragraph;
      Paragraph->Parent = Current;
      Current->Children.push_back(Paragraph);
      Current = Paragraph;
      continue;
    }
    parseLine(Line, Current);
  }
  return Root;
}

std::string MarkdownParser::traverse(Node *Current) {
  std::string Result;
  switch (Current->Type) {
  case MDType::Strong:
    Result.append("<strong>");
    for (auto *Child : Current->Children)
      Result.append(traverse(Child));
    Result.append("</strong>");
    break;
  case MDType::Text:
    Result.append(Current->Content);
    break;
  case MDType::Softbreak:
    Result.append("\n");
    break;
  case MDType::Paragraph:
    Result.append("<p>");
    for (auto *Child : Current->Children)
      Result.append(traverse(Child));
    Result.append("</p>");
    break;
  case MDType::Emphasis:
    Result.append("<em>");
    for (auto *Child : Current->Children)
      Result.append(traverse(Child));
    Result.append("</em>");
    break;
  }
  return Result;
}

std::string MarkdownParser::render(std::vector<SmallString<64>> &Lines) {
  auto *Document = parse(Lines);
  std::string Result;
  for (auto *Child : Document->Children)
    Result.append(traverse(Child));
  return Result;
}
} // namespace doc
} // namespace clang
