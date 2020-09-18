//===- Tree.cpp -----------------------------------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/Syntax/Tree.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cassert>

using namespace clang;

namespace {
static void traverse(const syntax::Node *N,
                     llvm::function_ref<void(const syntax::Node *)> Visit) {
  if (auto *T = dyn_cast<syntax::Tree>(N)) {
    for (const auto *C = T->getFirstChild(); C; C = C->getNextSibling())
      traverse(C, Visit);
  }
  Visit(N);
}
static void traverse(syntax::Node *N,
                     llvm::function_ref<void(syntax::Node *)> Visit) {
  traverse(static_cast<const syntax::Node *>(N), [&](const syntax::Node *N) {
    Visit(const_cast<syntax::Node *>(N));
  });
}
} // namespace

syntax::Arena::Arena(SourceManager &SourceMgr, const LangOptions &LangOpts,
                     const TokenBuffer &Tokens)
    : SourceMgr(SourceMgr), LangOpts(LangOpts), Tokens(Tokens) {}

const syntax::TokenBuffer &syntax::Arena::getTokenBuffer() const {
  return Tokens;
}

std::pair<FileID, ArrayRef<syntax::Token>>
syntax::Arena::lexBuffer(std::unique_ptr<llvm::MemoryBuffer> Input) {
  auto FID = SourceMgr.createFileID(std::move(Input));
  auto It = ExtraTokens.try_emplace(FID, tokenize(FID, SourceMgr, LangOpts));
  assert(It.second && "duplicate FileID");
  return {FID, It.first->second};
}

syntax::Leaf::Leaf(const syntax::Token *Tok) : Node(NodeKind::Leaf), Tok(Tok) {
  assert(Tok != nullptr);
}

bool syntax::Leaf::classof(const Node *N) {
  return N->getKind() == NodeKind::Leaf;
}

syntax::Node::Node(NodeKind Kind)
    : Parent(nullptr), NextSibling(nullptr), Kind(static_cast<unsigned>(Kind)),
      Role(0), Original(false), CanModify(false) {
  this->setRole(NodeRole::Detached);
}

bool syntax::Node::isDetached() const {
  return getRole() == NodeRole::Detached;
}

void syntax::Node::setRole(NodeRole NR) {
  this->Role = static_cast<unsigned>(NR);
}

bool syntax::Tree::classof(const Node *N) {
  return N->getKind() > NodeKind::Leaf;
}

void syntax::Tree::prependChildLowLevel(Node *Child, NodeRole Role) {
  assert(Child->getRole() == NodeRole::Detached);
  assert(Role != NodeRole::Detached);

  Child->setRole(Role);
  prependChildLowLevel(Child);
}

void syntax::Tree::prependChildLowLevel(Node *Child) {
  assert(Child->Parent == nullptr);
  assert(Child->NextSibling == nullptr);
  assert(Child->getRole() != NodeRole::Detached);

  Child->Parent = this;
  Child->NextSibling = this->FirstChild;
  this->FirstChild = Child;
}

void syntax::Tree::replaceChildRangeLowLevel(Node *BeforeBegin, Node *End,
                                             Node *New) {
  assert(!BeforeBegin || BeforeBegin->Parent == this);

#ifndef NDEBUG
  for (auto *N = New; N; N = N->getNextSibling()) {
    assert(N->Parent == nullptr);
    assert(N->getRole() != NodeRole::Detached && "Roles must be set");
    // FIXME: sanity-check the role.
  }
#endif

  // Detach old nodes.
  for (auto *N = !BeforeBegin ? FirstChild : BeforeBegin->getNextSibling();
       N != End;) {
    auto *Next = N->NextSibling;

    N->setRole(NodeRole::Detached);
    N->Parent = nullptr;
    N->NextSibling = nullptr;
    if (N->Original)
      traverse(N, [&](Node *C) { C->Original = false; });

    N = Next;
  }

  // Attach new nodes.
  if (BeforeBegin)
    BeforeBegin->NextSibling = New ? New : End;
  else
    FirstChild = New ? New : End;

  if (New) {
    auto *Last = New;
    for (auto *N = New; N != nullptr; N = N->getNextSibling()) {
      Last = N;
      N->Parent = this;
    }
    Last->NextSibling = End;
  }

  // Mark the node as modified.
  for (auto *T = this; T && T->Original; T = T->Parent)
    T->Original = false;
}

namespace {
static void dumpLeaf(raw_ostream &OS, const syntax::Leaf *L,
                     const SourceManager &SM) {
  assert(L);
  const auto *Token = L->getToken();
  assert(Token);
  // Handle 'eof' separately, calling text() on it produces an empty string.
  if (Token->kind() == tok::eof)
    OS << "<eof>";
  else
    OS << Token->text(SM);
}

static void dumpNode(raw_ostream &OS, const syntax::Node *N,
                     const SourceManager &SM, std::vector<bool> IndentMask) {
  auto dumpExtraInfo = [&OS](const syntax::Node *N) {
    if (N->getRole() != syntax::NodeRole::Unknown)
      OS << " " << N->getRole();
    if (!N->isOriginal())
      OS << " synthesized";
    if (!N->canModify())
      OS << " unmodifiable";
  };

  assert(N);
  if (const auto *L = dyn_cast<syntax::Leaf>(N)) {
    OS << "'";
    dumpLeaf(OS, L, SM);
    OS << "'";
    dumpExtraInfo(N);
    OS << "\n";
    return;
  }

  const auto *T = cast<syntax::Tree>(N);
  OS << T->getKind();
  dumpExtraInfo(N);
  OS << "\n";

  for (const auto *It = T->getFirstChild(); It; It = It->getNextSibling()) {
    for (bool Filled : IndentMask) {
      if (Filled)
        OS << "| ";
      else
        OS << "  ";
    }
    if (!It->getNextSibling()) {
      OS << "`-";
      IndentMask.push_back(false);
    } else {
      OS << "|-";
      IndentMask.push_back(true);
    }
    dumpNode(OS, It, SM, IndentMask);
    IndentMask.pop_back();
  }
}
} // namespace

std::string syntax::Node::dump(const SourceManager &SM) const {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  dumpNode(OS, this, SM, /*IndentMask=*/{});
  return std::move(OS.str());
}

std::string syntax::Node::dumpTokens(const SourceManager &SM) const {
  std::string Storage;
  llvm::raw_string_ostream OS(Storage);
  traverse(this, [&](const syntax::Node *N) {
    if (const auto *L = dyn_cast<syntax::Leaf>(N)) {
      dumpLeaf(OS, L, SM);
      OS << " ";
    }
  });
  return OS.str();
}

void syntax::Node::assertInvariants() const {
#ifndef NDEBUG
  if (isDetached())
    assert(getParent() == nullptr);
  else
    assert(getParent() != nullptr);

  const auto *T = dyn_cast<Tree>(this);
  if (!T)
    return;
  for (const auto *C = T->getFirstChild(); C; C = C->getNextSibling()) {
    if (T->isOriginal())
      assert(C->isOriginal());
    assert(!C->isDetached());
    assert(C->getParent() == T);
  }

  const auto *L = dyn_cast<List>(T);
  if (!L)
    return;
  for (const auto *C = T->getFirstChild(); C; C = C->getNextSibling()) {
    assert(C->getRole() == NodeRole::ListElement ||
           C->getRole() == NodeRole::ListDelimiter);
    if (C->getRole() == NodeRole::ListDelimiter) {
      assert(isa<Leaf>(C));
      assert(cast<Leaf>(C)->getToken()->kind() == L->getDelimiterTokenKind());
    }
  }

#endif
}

void syntax::Node::assertInvariantsRecursive() const {
#ifndef NDEBUG
  traverse(this, [&](const syntax::Node *N) { N->assertInvariants(); });
#endif
}

syntax::Leaf *syntax::Tree::findFirstLeaf() {
  auto *T = this;
  while (auto *C = T->getFirstChild()) {
    if (auto *L = dyn_cast<syntax::Leaf>(C))
      return L;
    T = cast<syntax::Tree>(C);
  }
  return nullptr;
}

syntax::Leaf *syntax::Tree::findLastLeaf() {
  auto *T = this;
  while (auto *C = T->getFirstChild()) {
    // Find the last child.
    while (auto *Next = C->getNextSibling())
      C = Next;

    if (auto *L = dyn_cast<syntax::Leaf>(C))
      return L;
    T = cast<syntax::Tree>(C);
  }
  return nullptr;
}

syntax::Node *syntax::Tree::findChild(NodeRole R) {
  for (auto *C = FirstChild; C; C = C->getNextSibling()) {
    if (C->getRole() == R)
      return C;
  }
  return nullptr;
}

bool syntax::List::classof(const syntax::Node *N) {
  switch (N->getKind()) {
  case syntax::NodeKind::NestedNameSpecifier:
  case syntax::NodeKind::CallArguments:
  case syntax::NodeKind::ParameterDeclarationList:
    return true;
  default:
    return false;
  }
}

std::vector<syntax::List::ElementAndDelimiter<syntax::Node>>
syntax::List::getElementsAsNodesAndDelimiters() {
  if (!getFirstChild())
    return {};

  auto children = std::vector<syntax::List::ElementAndDelimiter<Node>>();
  syntax::Node *elementWithoutDelimiter = nullptr;
  for (auto *C = getFirstChild(); C; C = C->getNextSibling()) {
    switch (C->getRole()) {
    case syntax::NodeRole::ListElement: {
      if (elementWithoutDelimiter) {
        children.push_back({elementWithoutDelimiter, nullptr});
      }
      elementWithoutDelimiter = C;
      break;
    }
    case syntax::NodeRole::ListDelimiter: {
      children.push_back({elementWithoutDelimiter, cast<syntax::Leaf>(C)});
      elementWithoutDelimiter = nullptr;
      break;
    }
    default:
      llvm_unreachable(
          "A list can have only elements and delimiters as children.");
    }
  }

  switch (getTerminationKind()) {
  case syntax::List::TerminationKind::Separated: {
    children.push_back({elementWithoutDelimiter, nullptr});
    break;
  }
  case syntax::List::TerminationKind::Terminated:
  case syntax::List::TerminationKind::MaybeTerminated: {
    if (elementWithoutDelimiter) {
      children.push_back({elementWithoutDelimiter, nullptr});
    }
    break;
  }
  }

  return children;
}

// Almost the same implementation of `getElementsAsNodesAndDelimiters` but
// ignoring delimiters
std::vector<syntax::Node *> syntax::List::getElementsAsNodes() {
  if (!getFirstChild())
    return {};

  auto children = std::vector<syntax::Node *>();
  syntax::Node *elementWithoutDelimiter = nullptr;
  for (auto *C = getFirstChild(); C; C = C->getNextSibling()) {
    switch (C->getRole()) {
    case syntax::NodeRole::ListElement: {
      if (elementWithoutDelimiter) {
        children.push_back(elementWithoutDelimiter);
      }
      elementWithoutDelimiter = C;
      break;
    }
    case syntax::NodeRole::ListDelimiter: {
      children.push_back(elementWithoutDelimiter);
      elementWithoutDelimiter = nullptr;
      break;
    }
    default:
      llvm_unreachable("A list has only elements or delimiters.");
    }
  }

  switch (getTerminationKind()) {
  case syntax::List::TerminationKind::Separated: {
    children.push_back(elementWithoutDelimiter);
    break;
  }
  case syntax::List::TerminationKind::Terminated:
  case syntax::List::TerminationKind::MaybeTerminated: {
    if (elementWithoutDelimiter) {
      children.push_back(elementWithoutDelimiter);
    }
    break;
  }
  }

  return children;
}

clang::tok::TokenKind syntax::List::getDelimiterTokenKind() const {
  switch (this->getKind()) {
  case NodeKind::NestedNameSpecifier:
    return clang::tok::coloncolon;
  case NodeKind::CallArguments:
  case NodeKind::ParameterDeclarationList:
    return clang::tok::comma;
  default:
    llvm_unreachable("This is not a subclass of List, thus "
                     "getDelimiterTokenKind() cannot be called");
  }
}

syntax::List::TerminationKind syntax::List::getTerminationKind() const {
  switch (this->getKind()) {
  case NodeKind::NestedNameSpecifier:
    return TerminationKind::Terminated;
  case NodeKind::CallArguments:
  case NodeKind::ParameterDeclarationList:
    return TerminationKind::Separated;
  default:
    llvm_unreachable("This is not a subclass of List, thus "
                     "getTerminationKind() cannot be called");
  }
}

bool syntax::List::canBeEmpty() const {
  switch (this->getKind()) {
  case NodeKind::NestedNameSpecifier:
    return false;
  case NodeKind::CallArguments:
    return true;
  case NodeKind::ParameterDeclarationList:
    return true;
  default:
    llvm_unreachable("This is not a subclass of List, thus canBeEmpty() "
                     "cannot be called");
  }
}
