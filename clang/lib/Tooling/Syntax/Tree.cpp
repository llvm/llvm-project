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

using namespace clang;

syntax::Arena::Arena(SourceManager &SourceMgr, const LangOptions &LangOpts,
                     TokenBuffer Tokens)
    : SourceMgr(SourceMgr), LangOpts(LangOpts), Tokens(std::move(Tokens)) {}

const clang::syntax::TokenBuffer &syntax::Arena::tokenBuffer() const {
  return Tokens;
}

std::pair<FileID, llvm::ArrayRef<syntax::Token>>
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
  return N->kind() == NodeKind::Leaf;
}

syntax::Node::Node(NodeKind Kind)
    : Parent(nullptr), NextSibling(nullptr), Kind(static_cast<unsigned>(Kind)),
      Role(static_cast<unsigned>(NodeRole::Detached)) {}

bool syntax::Tree::classof(const Node *N) { return N->kind() > NodeKind::Leaf; }

void syntax::Tree::prependChildLowLevel(Node *Child, NodeRole Role) {
  assert(Child->Parent == nullptr);
  assert(Child->NextSibling == nullptr);
  assert(Child->role() == NodeRole::Detached);
  assert(Role != NodeRole::Detached);

  Child->Parent = this;
  Child->NextSibling = this->FirstChild;
  Child->Role = static_cast<unsigned>(Role);
  this->FirstChild = Child;
}

namespace {
static void traverse(const syntax::Node *N,
                     llvm::function_ref<void(const syntax::Node *)> Visit) {
  if (auto *T = dyn_cast<syntax::Tree>(N)) {
    for (auto *C = T->firstChild(); C; C = C->nextSibling())
      traverse(C, Visit);
  }
  Visit(N);
}
static void dumpTokens(llvm::raw_ostream &OS, ArrayRef<syntax::Token> Tokens,
                       const SourceManager &SM) {
  assert(!Tokens.empty());
  bool First = true;
  for (const auto &T : Tokens) {
    if (!First)
      OS << " ";
    else
      First = false;
    // Handle 'eof' separately, calling text() on it produces an empty string.
    if (T.kind() == tok::eof) {
      OS << "<eof>";
      continue;
    }
    OS << T.text(SM);
  }
}

static void dumpTree(llvm::raw_ostream &OS, const syntax::Node *N,
                     const syntax::Arena &A, std::vector<bool> IndentMask) {
  if (N->role() != syntax::NodeRole::Unknown) {
    // FIXME: print the symbolic name of a role.
    if (N->role() == syntax::NodeRole::Detached)
      OS << "*: ";
    else
      OS << static_cast<int>(N->role()) << ": ";
  }
  if (auto *L = llvm::dyn_cast<syntax::Leaf>(N)) {
    dumpTokens(OS, *L->token(), A.sourceManager());
    OS << "\n";
    return;
  }

  auto *T = llvm::cast<syntax::Tree>(N);
  OS << T->kind() << "\n";

  for (auto It = T->firstChild(); It != nullptr; It = It->nextSibling()) {
    for (bool Filled : IndentMask) {
      if (Filled)
        OS << "| ";
      else
        OS << "  ";
    }
    if (!It->nextSibling()) {
      OS << "`-";
      IndentMask.push_back(false);
    } else {
      OS << "|-";
      IndentMask.push_back(true);
    }
    dumpTree(OS, It, A, IndentMask);
    IndentMask.pop_back();
  }
}
} // namespace

std::string syntax::Node::dump(const Arena &A) const {
  std::string Str;
  llvm::raw_string_ostream OS(Str);
  dumpTree(OS, this, A, /*IndentMask=*/{});
  return std::move(OS.str());
}

std::string syntax::Node::dumpTokens(const Arena &A) const {
  std::string Storage;
  llvm::raw_string_ostream OS(Storage);
  traverse(this, [&](const syntax::Node *N) {
    auto *L = llvm::dyn_cast<syntax::Leaf>(N);
    if (!L)
      return;
    ::dumpTokens(OS, *L->token(), A.sourceManager());
  });
  return OS.str();
}

syntax::Node *syntax::Tree::findChild(NodeRole R) {
  for (auto *C = FirstChild; C; C = C->nextSibling()) {
    if (C->role() == R)
      return C;
  }
  return nullptr;
}
