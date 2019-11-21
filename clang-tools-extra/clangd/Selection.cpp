//===--- Selection.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Selection.h"
#include "Logger.h"
#include "SourceCode.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>

namespace clang {
namespace clangd {
namespace {
using Node = SelectionTree::Node;
using ast_type_traits::DynTypedNode;

// Identifies which tokens are selected, and evaluates claims of source ranges
// by AST nodes. Tokens may be claimed only once: first-come, first-served.
class SelectedTokens {
public:
  SelectedTokens(llvm::ArrayRef<syntax::Token> Spelled, const SourceManager &SM,
                 unsigned SelBegin, unsigned SelEnd)
      : SelBegin(SelBegin), SelEnd(SelEnd) {
    // Extract bounds and selected-ness for all tokens spelled in the file.
    Tokens.reserve(Spelled.size());
    for (const auto& Tok : Spelled) {
      // As well as comments, don't count semicolons as real tokens.
      // They're not properly claimed as expr-statement is missing from the AST.
      if (Tok.kind() == tok::comment || Tok.kind() == tok::semi)
        continue;

      Tokens.emplace_back();
      TokInfo &S = Tokens.back();
      S.StartOffset = SM.getFileOffset(Tok.location());
      S.EndOffset = S.StartOffset + Tok.length();
      if (S.StartOffset >= SelBegin && S.EndOffset <= SelEnd)
        S.Selected = SelectionTree::Complete;
      else if (S.EndOffset > SelBegin && S.StartOffset < SelEnd)
        S.Selected = SelectionTree::Partial;
      else
        S.Selected = SelectionTree::Unselected;
      S.Claimed = false;
    }
  }

  // Associates any tokens overlapping [Begin, End) with an AST node.
  // Tokens that were already claimed by another AST node are not claimed again.
  // Updates Result if the node is selected in the sense of SelectionTree.
  void claim(unsigned Begin, unsigned End, SelectionTree::Selection &Result) {
    assert(Begin <= End);

    // Fast-path for missing the selection entirely.
    if (Begin >= SelEnd || End <= SelBegin)
      return;

    // We will consider the range (at least partially) selected if it hit any
    // selected and previously unclaimed token.
    bool ClaimedAnyToken = false;
    // The selection is (at most) partial if:
    // - any claimed token is partially selected
    // - any token in the range is unselected
    bool PartialSelection = false;

    // Find the first token that (maybe) overlaps the claimed range.
    auto Start = llvm::partition_point(Tokens, [&](const TokInfo &Tok) {
      return Tok.EndOffset <= Begin;
    });
    // Iterate over every token that overlaps the range.
    // Claim selected tokens, and update the two result flags.
    for (auto It = Start; It != Tokens.end() && It->StartOffset < End; ++It) {
      if (It->Selected) {
        if (!It->Claimed) {
          // Token is selected, in the node's range, and unclaimed; claim it.
          It->Claimed = true;
          ClaimedAnyToken = true;
          // If the token was only partially selected, so is the node.
          PartialSelection |= (It->Selected == SelectionTree::Partial);
        }
      } else {
        // If the node covers an unselected token, it's not completely selected.
        PartialSelection = true;
      }
    }

    // If some tokens were previously claimed (Result != Unselected), we may
    // upgrade from Partial->Complete, even if no new tokens were claimed.
    // Important for [[int a]].
    if (ClaimedAnyToken || Result) {
      Result = std::max(Result, PartialSelection ? SelectionTree::Partial
                                                 : SelectionTree::Complete);
    }
  }

private:
  struct TokInfo {
    unsigned StartOffset;
    unsigned EndOffset;
    SelectionTree::Selection Selected;
    bool Claimed;
    bool operator<(const TokInfo &Other) const {
      return StartOffset < Other.StartOffset;
    }
  };
  std::vector<TokInfo> Tokens;
  unsigned SelBegin, SelEnd;
};

// Show the type of a node for debugging.
void printNodeKind(llvm::raw_ostream &OS, const DynTypedNode &N) {
  if (const TypeLoc *TL = N.get<TypeLoc>()) {
    // TypeLoc is a hierarchy, but has only a single ASTNodeKind.
    // Synthesize the name from the Type subclass (except for QualifiedTypeLoc).
    if (TL->getTypeLocClass() == TypeLoc::Qualified)
      OS << "QualifiedTypeLoc";
    else
      OS << TL->getType()->getTypeClassName() << "TypeLoc";
  } else {
    OS << N.getNodeKind().asStringRef();
  }
}

#ifndef NDEBUG
std::string printNodeToString(const DynTypedNode &N, const PrintingPolicy &PP) {
  std::string S;
  llvm::raw_string_ostream OS(S);
  printNodeKind(OS, N);
  OS << " ";
  return std::move(OS.str());
}
#endif

bool isImplicit(const Stmt* S) {
  // Some Stmts are implicit and shouldn't be traversed, but there's no
  // "implicit" attribute on Stmt/Expr.
  // Unwrap implicit casts first if present (other nodes too?).
  if (auto *ICE = llvm::dyn_cast<ImplicitCastExpr>(S))
    S = ICE->getSubExprAsWritten();
  // Implicit this in a MemberExpr is not filtered out by RecursiveASTVisitor.
  // It would be nice if RAV handled this (!shouldTraverseImplicitCode()).
  if (auto *CTI = llvm::dyn_cast<CXXThisExpr>(S))
    if (CTI->isImplicit())
      return true;
  // Refs to operator() and [] are (almost?) always implicit as part of calls.
  if (auto *DRE = llvm::dyn_cast<DeclRefExpr>(S)) {
    if (auto *FD = llvm::dyn_cast<FunctionDecl>(DRE->getDecl())) {
      switch (FD->getOverloadedOperator()) {
      case OO_Call:
      case OO_Subscript:
        return true;
      default:
        break;
      }
    }
  }
  return false;
}

// We find the selection by visiting written nodes in the AST, looking for nodes
// that intersect with the selected character range.
//
// While traversing, we maintain a parent stack. As nodes pop off the stack,
// we decide whether to keep them or not. To be kept, they must either be
// selected or contain some nodes that are.
//
// For simple cases (not inside macros) we prune subtrees that don't intersect.
class SelectionVisitor : public RecursiveASTVisitor<SelectionVisitor> {
public:
  // Runs the visitor to gather selected nodes and their ancestors.
  // If there is any selection, the root (TUDecl) is the first node.
  static std::deque<Node> collect(ASTContext &AST,
                                  const syntax::TokenBuffer &Tokens,
                                  const PrintingPolicy &PP, unsigned Begin,
                                  unsigned End, FileID File) {
    SelectionVisitor V(AST, Tokens, PP, Begin, End, File);
    V.TraverseAST(AST);
    assert(V.Stack.size() == 1 && "Unpaired push/pop?");
    assert(V.Stack.top() == &V.Nodes.front());
    // We selected TUDecl if tokens were unclaimed (or the file is empty).
    SelectionTree::Selection UnclaimedTokens = SelectionTree::Unselected;
    V.Claimed.claim(Begin, End, UnclaimedTokens);
    if (UnclaimedTokens || V.Nodes.size() == 1) {
      StringRef FileContent = AST.getSourceManager().getBufferData(File);
      // Don't require the trailing newlines to be selected.
      bool SelectedAll = Begin == 0 && End >= FileContent.rtrim().size();
      V.Stack.top()->Selected =
          SelectedAll ? SelectionTree::Complete : SelectionTree::Partial;
    }
    return std::move(V.Nodes);
  }

  // We traverse all "well-behaved" nodes the same way:
  //  - push the node onto the stack
  //  - traverse its children recursively
  //  - pop it from the stack
  //  - hit testing: is intersection(node, selection) - union(children) empty?
  //  - attach it to the tree if it or any children hit the selection
  //
  // Two categories of nodes are not "well-behaved":
  //  - those without source range information, we don't record those
  //  - those that can't be stored in DynTypedNode.
  // We're missing some interesting things like Attr due to the latter.
  bool TraverseDecl(Decl *X) {
    if (X && isa<TranslationUnitDecl>(X))
      return Base::TraverseDecl(X); // Already pushed by constructor.
    // Base::TraverseDecl will suppress children, but not this node itself.
    if (X && X->isImplicit())
      return true;
    return traverseNode(X, [&] { return Base::TraverseDecl(X); });
  }
  bool TraverseTypeLoc(TypeLoc X) {
    return traverseNode(&X, [&] { return Base::TraverseTypeLoc(X); });
  }
  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc X) {
    return traverseNode(
        &X, [&] { return Base::TraverseNestedNameSpecifierLoc(X); });
  }
  bool TraverseConstructorInitializer(CXXCtorInitializer *X) {
    return traverseNode(
        X, [&] { return Base::TraverseConstructorInitializer(X); });
  }
  // Stmt is the same, but this form allows the data recursion optimization.
  bool dataTraverseStmtPre(Stmt *X) {
    if (!X || isImplicit(X))
      return false;
    auto N = DynTypedNode::create(*X);
    if (canSafelySkipNode(N))
      return false;
    push(std::move(N));
    if (shouldSkipChildren(X)) {
      pop();
      return false;
    }
    return true;
  }
  bool dataTraverseStmtPost(Stmt *X) {
    pop();
    return true;
  }
  // QualifiedTypeLoc is handled strangely in RecursiveASTVisitor: the derived
  // TraverseTypeLoc is not called for the inner UnqualTypeLoc.
  // This means we'd never see 'int' in 'const int'! Work around that here.
  // (The reason for the behavior is to avoid traversing the nested Type twice,
  // but we ignore TraverseType anyway).
  bool TraverseQualifiedTypeLoc(QualifiedTypeLoc QX) {
    return traverseNode<TypeLoc>(
        &QX, [&] { return TraverseTypeLoc(QX.getUnqualifiedLoc()); });
  }
  // Uninteresting parts of the AST that don't have locations within them.
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *) { return true; }
  bool TraverseType(QualType) { return true; }

  // The DeclStmt for the loop variable claims to cover the whole range
  // inside the parens, this causes the range-init expression to not be hit.
  // Traverse the loop VarDecl instead, which has the right source range.
  bool TraverseCXXForRangeStmt(CXXForRangeStmt *S) {
    return traverseNode(S, [&] {
      return TraverseStmt(S->getInit()) && TraverseDecl(S->getLoopVariable()) &&
             TraverseStmt(S->getRangeInit()) && TraverseStmt(S->getBody());
    });
  }

private:
  using Base = RecursiveASTVisitor<SelectionVisitor>;

  SelectionVisitor(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                   const PrintingPolicy &PP, unsigned SelBegin, unsigned SelEnd,
                   FileID SelFile)
      : SM(AST.getSourceManager()), LangOpts(AST.getLangOpts()),
#ifndef NDEBUG
        PrintPolicy(PP),
#endif
        Claimed(Tokens.spelledTokens(SelFile), SM, SelBegin, SelEnd),
        SelFile(SelFile),
        SelBeginTokenStart(SM.getFileOffset(Lexer::GetBeginningOfToken(
            SM.getComposedLoc(SelFile, SelBegin), SM, LangOpts))),
        SelEnd(SelEnd) {
    // Ensure we have a node for the TU decl, regardless of traversal scope.
    Nodes.emplace_back();
    Nodes.back().ASTNode = DynTypedNode::create(*AST.getTranslationUnitDecl());
    Nodes.back().Parent = nullptr;
    Nodes.back().Selected = SelectionTree::Unselected;
    Stack.push(&Nodes.back());
  }

  // Generic case of TraverseFoo. Func should be the call to Base::TraverseFoo.
  // Node is always a pointer so the generic code can handle any null checks.
  template <typename T, typename Func>
  bool traverseNode(T *Node, const Func &Body) {
    if (Node == nullptr)
      return true;
    auto N = DynTypedNode::create(*Node);
    if (canSafelySkipNode(N))
      return true;
    push(DynTypedNode::create(*Node));
    bool Ret = Body();
    pop();
    return Ret;
  }

  // HIT TESTING
  //
  // We do rough hit testing on the way down the tree to avoid traversing
  // subtrees that don't touch the selection (canSafelySkipNode), but
  // fine-grained hit-testing is mostly done on the way back up (in pop()).
  // This means children get to claim parts of the selection first, and parents
  // are only selected if they own tokens that no child owned.
  //
  // Nodes *usually* nest nicely: a child's getSourceRange() lies within the
  // parent's, and a node (transitively) owns all tokens in its range.
  //
  // Exception 1: child range claims tokens that should be owned by the parent.
  //              e.g. in `void foo(int);`, the FunctionTypeLoc should own
  //              `void (int)` but the parent FunctionDecl should own `foo`.
  // To handle this case, certain nodes claim small token ranges *before*
  // their children are traversed. (see earlySourceRange).
  //
  // Exception 2: siblings both claim the same node.
  //              e.g. `int x, y;` produces two sibling VarDecls.
  //                    ~~~~~ x
  //                    ~~~~~~~~ y
  // Here the first ("leftmost") sibling claims the tokens it wants, and the
  // other sibling gets what's left. So selecting "int" only includes the left
  // VarDecl in the selection tree.

  // An optimization for a common case: nodes outside macro expansions that
  // don't intersect the selection may be recursively skipped.
  bool canSafelySkipNode(const DynTypedNode &N) {
    SourceRange S = N.getSourceRange();
    auto B = SM.getDecomposedLoc(S.getBegin());
    auto E = SM.getDecomposedLoc(S.getEnd());
    // Node lies in a macro expansion?
    if (B.first != SelFile || E.first != SelFile)
      return false;
    // Node intersects selection tokens?
    if (B.second < SelEnd && E.second >= SelBeginTokenStart)
      return false;
    // Otherwise, allow skipping over the node.
    dlog("{1}skip: {0}", printNodeToString(N, PrintPolicy), indent());
    dlog("{1}skipped range = {0}", S.printToString(SM), indent(1));
    return true;
  }

  // There are certain nodes we want to treat as leaves in the SelectionTree,
  // although they do have children.
  bool shouldSkipChildren(const Stmt *X) const {
    // UserDefinedLiteral (e.g. 12_i) has two children (12 and _i).
    // Unfortunately TokenBuffer sees 12_i as one token and can't split it.
    // So we treat UserDefinedLiteral as a leaf node, owning the token.
    return llvm::isa<UserDefinedLiteral>(X);
  }

  // Pushes a node onto the ancestor stack. Pairs with pop().
  // Performs early hit detection for some nodes (on the earlySourceRange).
  void push(DynTypedNode Node) {
    SourceRange Early = earlySourceRange(Node);
    dlog("{1}push: {0}", printNodeToString(Node, PrintPolicy), indent());
    Nodes.emplace_back();
    Nodes.back().ASTNode = std::move(Node);
    Nodes.back().Parent = Stack.top();
    Stack.push(&Nodes.back());
    claimRange(Early, Nodes.back().Selected);
    // Early hit detection never selects the whole node.
    if (Nodes.back().Selected)
      Nodes.back().Selected = SelectionTree::Partial;
  }

  // Pops a node off the ancestor stack, and finalizes it. Pairs with push().
  // Performs primary hit detection.
  void pop() {
    Node &N = *Stack.top();
    dlog("{1}pop: {0}", printNodeToString(N.ASTNode, PrintPolicy), indent(-1));
    claimRange(N.ASTNode.getSourceRange(), N.Selected);
    if (N.Selected || !N.Children.empty()) {
      // Attach to the tree.
      N.Parent->Children.push_back(&N);
    } else {
      // Neither N any children are selected, it doesn't belong in the tree.
      assert(&N == &Nodes.back());
      Nodes.pop_back();
    }
    Stack.pop();
  }

  // Returns the range of tokens that this node will claim directly, and
  // is not available to the node's children.
  // Usually empty, but sometimes children cover tokens but shouldn't own them.
  SourceRange earlySourceRange(const DynTypedNode &N) {
    if (const Decl *D = N.get<Decl>()) {
      // void [[foo]]();
      if (auto *FD = llvm::dyn_cast<FunctionDecl>(D))
        return FD->getNameInfo().getSourceRange();
      // int (*[[s]])();
      else if (auto *VD = llvm::dyn_cast<VarDecl>(D))
        return VD->getLocation();
    } else if (const auto* CCI = N.get<CXXCtorInitializer>()) {
      // : [[b_]](42)
      return CCI->getMemberLocation();
    }
    return SourceRange();
  }

  // Perform hit-testing of a complete Node against the selection.
  // This runs for every node in the AST, and must be fast in common cases.
  // This is usually called from pop(), so we can take children into account.
  // The existing state of Result is relevant (early/late claims can interact).
  void claimRange(SourceRange S, SelectionTree::Selection &Result) {
    if (!S.isValid())
      return;
    // toHalfOpenFileRange() allows selection of constructs in macro args. e.g:
    //   #define LOOP_FOREVER(Body) for(;;) { Body }
    //   void IncrementLots(int &x) {
    //     LOOP_FOREVER( ++x; )
    //   }
    // Selecting "++x" or "x" will do the right thing.
    auto Range = toHalfOpenFileRange(SM, LangOpts, S);
    assert(Range && "We should be able to get the File Range");
    dlog("{1}claimRange: {0}", Range->printToString(SM), indent());
    auto B = SM.getDecomposedLoc(Range->getBegin());
    auto E = SM.getDecomposedLoc(Range->getEnd());
    // Otherwise, nodes in macro expansions can't be selected.
    if (B.first != SelFile || E.first != SelFile)
      return;
    // Attempt to claim the remaining range. If there's nothing to claim, only
    // children were selected.
    Claimed.claim(B.second, E.second, Result);
    if (Result)
      dlog("{1}hit selection: {0}",
           SourceRange(SM.getComposedLoc(B.first, B.second),
                       SM.getComposedLoc(E.first, E.second))
               .printToString(SM),
           indent());
  }

  std::string indent(int Offset = 0) {
    // Cast for signed arithmetic.
    int Amount = int(Stack.size()) + Offset;
    assert(Amount >= 0);
    return std::string(Amount, ' ');
  }

  SourceManager &SM;
  const LangOptions &LangOpts;
#ifndef NDEBUG
  const PrintingPolicy &PrintPolicy;
#endif
  std::stack<Node *> Stack;
  SelectedTokens Claimed;
  std::deque<Node> Nodes; // Stable pointers as we add more nodes.
  FileID SelFile;
  // If the selection start slices a token in half, the beginning of that token.
  // This is useful for checking whether the end of a token range overlaps
  // the selection: range.end < SelBeginTokenStart is equivalent to
  // range.end + measureToken(range.end) < SelBegin (assuming range.end points
  // to a token), and it saves a lex every time.
  unsigned SelBeginTokenStart;
  unsigned SelEnd;
};

} // namespace

void SelectionTree::print(llvm::raw_ostream &OS, const SelectionTree::Node &N,
                          int Indent) const {
  if (N.Selected)
    OS.indent(Indent - 1) << (N.Selected == SelectionTree::Complete ? '*'
                                                                    : '.');
  else
    OS.indent(Indent);
  printNodeKind(OS, N.ASTNode);
  OS << ' ';
  N.ASTNode.print(OS, PrintPolicy);
  OS << "\n";
  for (const Node *Child : N.Children)
    print(OS, *Child, Indent + 2);
}

std::string SelectionTree::Node::kind() const {
  std::string S;
  llvm::raw_string_ostream OS(S);
  printNodeKind(OS, ASTNode);
  return std::move(OS.str());
}

// Decide which selection emulates a "point" query in between characters.
static std::pair<unsigned, unsigned> pointBounds(unsigned Offset, FileID FID,
                                                 ASTContext &AST) {
  StringRef Buf = AST.getSourceManager().getBufferData(FID);
  // Edge-cases where the choice is forced.
  if (Buf.size() == 0)
    return {0, 0};
  if (Offset == 0)
    return {0, 1};
  if (Offset == Buf.size())
    return {Offset - 1, Offset};
  // We could choose either this byte or the previous. Usually we prefer the
  // character on the right of the cursor (or under a block cursor).
  // But if that's whitespace, we likely want the token on the left.
  if (isWhitespace(Buf[Offset]) && !isWhitespace(Buf[Offset - 1]))
    return {Offset - 1, Offset};
  return {Offset, Offset + 1};
}

SelectionTree::SelectionTree(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                             unsigned Begin, unsigned End)
    : PrintPolicy(AST.getLangOpts()) {
  // No fundamental reason the selection needs to be in the main file,
  // but that's all clangd has needed so far.
  const SourceManager &SM = AST.getSourceManager();
  FileID FID = SM.getMainFileID();
  if (Begin == End)
    std::tie(Begin, End) = pointBounds(Begin, FID, AST);
  PrintPolicy.TerseOutput = true;
  PrintPolicy.IncludeNewlines = false;

  dlog("Computing selection for {0}",
       SourceRange(SM.getComposedLoc(FID, Begin), SM.getComposedLoc(FID, End))
           .printToString(SM));
  Nodes = SelectionVisitor::collect(AST, Tokens, PrintPolicy, Begin, End, FID);
  Root = Nodes.empty() ? nullptr : &Nodes.front();
  dlog("Built selection tree\n{0}", *this);
}

SelectionTree::SelectionTree(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                             unsigned Offset)
    : SelectionTree(AST, Tokens, Offset, Offset) {}

const Node *SelectionTree::commonAncestor() const {
  const Node *Ancestor = Root;
  while (Ancestor->Children.size() == 1 && !Ancestor->Selected)
    Ancestor = Ancestor->Children.front();
  // Returning nullptr here is a bit unprincipled, but it makes the API safer:
  // the TranslationUnitDecl contains all of the preamble, so traversing it is a
  // performance cliff. Callers can check for null and use root() if they want.
  return Ancestor != Root ? Ancestor : nullptr;
}

const DeclContext& SelectionTree::Node::getDeclContext() const {
  for (const Node* CurrentNode = this; CurrentNode != nullptr;
       CurrentNode = CurrentNode->Parent) {
    if (const Decl* Current = CurrentNode->ASTNode.get<Decl>()) {
      if (CurrentNode != this)
        if (auto *DC = dyn_cast<DeclContext>(Current))
          return *DC;
      return *Current->getDeclContext();
    }
  }
  llvm_unreachable("A tree must always be rooted at TranslationUnitDecl.");
}

const SelectionTree::Node &SelectionTree::Node::ignoreImplicit() const {
  if (Children.size() == 1 &&
      Children.front()->ASTNode.getSourceRange() == ASTNode.getSourceRange())
    return Children.front()->ignoreImplicit();
  return *this;
}

const SelectionTree::Node &SelectionTree::Node::outerImplicit() const {
  if (Parent && Parent->ASTNode.getSourceRange() == ASTNode.getSourceRange())
    return Parent->outerImplicit();
  return *this;
}

} // namespace clangd
} // namespace clang
