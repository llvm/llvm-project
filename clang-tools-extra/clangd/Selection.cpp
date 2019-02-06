//===--- Selection.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Selection.h"
#include "ClangdUnit.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
namespace clangd {
namespace {
using Node = SelectionTree::Node;
using ast_type_traits::DynTypedNode;

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
  static std::deque<Node> collect(ASTContext &AST, unsigned Begin,
                                  unsigned End, FileID File) {
    SelectionVisitor V(AST, Begin, End, File);
    V.TraverseAST(AST);
    assert(V.Stack.size() == 1 && "Unpaired push/pop?");
    assert(V.Stack.top() == &V.Nodes.front());
    if (V.Nodes.size() == 1) // TUDecl, but no nodes under it.
      V.Nodes.clear();
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
    if (isa<TranslationUnitDecl>(X))
      return Base::TraverseDecl(X); // Already pushed by constructor.
    return traverseNode(X, [&] { return Base::TraverseDecl(X); });
  }
  bool TraverseTypeLoc(TypeLoc X) {
    return traverseNode(&X, [&] { return Base::TraverseTypeLoc(X); });
  }
  bool TraverseTypeNestedNameSpecifierLoc(NestedNameSpecifierLoc X) {
    return traverseNode(
        &X, [&] { return Base::TraverseNestedNameSpecifierLoc(X); });
  }
  bool TraverseConstructorInitializer(CXXCtorInitializer *X) {
    return traverseNode(
        X, [&] { return Base::TraverseConstructorInitializer(X); });
  }
  // Stmt is the same, but this form allows the data recursion optimization.
  bool dataTraverseStmtPre(Stmt *X) {
    if (!X || canSafelySkipNode(X->getSourceRange()))
      return false;
    push(DynTypedNode::create(*X));
    return true;
  }
  bool dataTraverseStmtPost(Stmt *X) {
    pop();
    return true;
  }
  // Uninteresting parts of the AST that don't have locations within them.
  bool TraverseNestedNameSpecifier(NestedNameSpecifier *) { return true; }
  bool TraverseType(QualType) { return true; }

private:
  using Base = RecursiveASTVisitor<SelectionVisitor>;
  SelectionVisitor(ASTContext &AST, unsigned SelBegin, unsigned SelEnd,
                   FileID SelFile)
      : SM(AST.getSourceManager()), LangOpts(AST.getLangOpts()),
        SelBegin(SelBegin), SelEnd(SelEnd), SelFile(SelFile),
        SelBeginTokenStart(SM.getFileOffset(Lexer::GetBeginningOfToken(
            SM.getComposedLoc(SelFile, SelBegin), SM, LangOpts))) {
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
    if (Node == nullptr || canSafelySkipNode(Node->getSourceRange()))
      return true;
    push(DynTypedNode::create(*Node));
    bool Ret = Body();
    pop();
    return Ret;
  }

  // An optimization for a common case: nodes outside macro expansions that
  // don't intersect the selection may be recursively skipped.
  bool canSafelySkipNode(SourceRange S) {
    auto B = SM.getDecomposedLoc(S.getBegin());
    auto E = SM.getDecomposedLoc(S.getEnd());
    if (B.first != SelFile || E.first != SelFile)
      return false;
    return B.second >= SelEnd || E.second < SelBeginTokenStart;
  }

  // Pushes a node onto the ancestor stack. Pairs with pop().
  void push(DynTypedNode Node) {
    Nodes.emplace_back();
    Nodes.back().ASTNode = std::move(Node);
    Nodes.back().Parent = Stack.top();
    Nodes.back().Selected = SelectionTree::Unselected;
    Stack.push(&Nodes.back());
  }

  // Pops a node off the ancestor stack, and finalizes it. Pairs with push().
  void pop() {
    Node &N = *Stack.top();
    N.Selected = computeSelection(N);
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

  // Perform hit-testing of a complete Node against the selection.
  // This runs for every node in the AST, and must be fast in common cases.
  // This is called from pop(), so we can take children into account.
  SelectionTree::Selection computeSelection(const Node &N) {
    SourceRange S = N.ASTNode.getSourceRange();
    if (!S.isValid())
      return SelectionTree::Unselected;
    // getTopMacroCallerLoc() allows selection of constructs in macro args. e.g:
    //   #define LOOP_FOREVER(Body) for(;;) { Body }
    //   void IncrementLots(int &x) {
    //     LOOP_FOREVER( ++x; )
    //   }
    // Selecting "++x" or "x" will do the right thing.
    auto B = SM.getDecomposedLoc(SM.getTopMacroCallerLoc(S.getBegin()));
    auto E = SM.getDecomposedLoc(SM.getTopMacroCallerLoc(S.getEnd()));
    // Otherwise, nodes in macro expansions can't be selected.
    if (B.first != SelFile || E.first != SelFile)
      return SelectionTree::Unselected;
    // Cheap test: is there any overlap at all between the selection and range?
    // Note that E.second is the *start* of the last token, which is why we
    // compare against the "rounded-down" SelBegin.
    if (B.second >= SelEnd || E.second < SelBeginTokenStart)
      return SelectionTree::Unselected;

    // We hit something, need some more precise checks.
    // Adjust [B, E) to be a half-open character range.
    E.second += Lexer::MeasureTokenLength(S.getEnd(), SM, LangOpts);
    // This node's own selected text is (this range ^ selection) - child ranges.
    // If that's empty, then we've only collided with children.
    if (nodesCoverRange(N.Children, std::max(SelBegin, B.second),
                        std::min(SelEnd, E.second)))
      return SelectionTree::Unselected; // Hit children only.
    // Some of our own characters are covered, this is a true hit.
    return (B.second >= SelBegin && E.second <= SelEnd)
               ? SelectionTree::Complete
               : SelectionTree::Partial;
  }

  // Is the range [Begin, End) entirely covered by the union of the Nodes?
  // (The range is a parent node's extent, and the covering nodes are children).
  bool nodesCoverRange(llvm::ArrayRef<const Node *> Nodes, unsigned Begin,
                       unsigned End) {
    if (Begin >= End)
      return true;
    if (Nodes.empty())
      return false;

    // Collect all the expansion ranges, as offsets.
    SmallVector<std::pair<unsigned, unsigned>, 8> ChildRanges;
    for (const Node *N : Nodes) {
      CharSourceRange R = SM.getExpansionRange(N->ASTNode.getSourceRange());
      auto B = SM.getDecomposedLoc(R.getBegin());
      auto E = SM.getDecomposedLoc(R.getEnd());
      if (B.first != SelFile || E.first != SelFile)
        continue;
      assert(R.isTokenRange());
      // Try to cover up to the next token, spaces between children don't count.
      if (auto Tok = Lexer::findNextToken(R.getEnd(), SM, LangOpts))
        E.second = SM.getFileOffset(Tok->getLocation());
      else
        E.second += Lexer::MeasureTokenLength(R.getEnd(), SM, LangOpts);
      ChildRanges.push_back({B.second, E.second});
    }
    llvm::sort(ChildRanges);

    // Scan through the child ranges, removing as we go.
    for (const auto R : ChildRanges) {
      if (R.first > Begin)
        return false;   // [Begin, R.first) is not covered.
      Begin = R.second; // Eliminate [R.first, R.second).
      if (Begin >= End)
        return true; // Remaining range is empty.
    }
    return false; // Went through all children, trailing characters remain.
  }

  SourceManager &SM;
  const LangOptions &LangOpts;
  std::stack<Node *> Stack;
  std::deque<Node> Nodes; // Stable pointers as we add more nodes.
  // Half-open selection range.
  unsigned SelBegin;
  unsigned SelEnd;
  FileID SelFile;
  // If the selection start slices a token in half, the beginning of that token.
  // This is useful for checking whether the end of a token range overlaps
  // the selection: range.end < SelBeginTokenStart is equivalent to
  // range.end + measureToken(range.end) < SelBegin (assuming range.end points
  // to a token), and it saves a lex every time.
  unsigned SelBeginTokenStart;
};

} // namespace

void SelectionTree::print(llvm::raw_ostream &OS, const SelectionTree::Node &N,
                          int Indent) const {
  if (N.Selected)
    OS.indent(Indent - 1) << (N.Selected == SelectionTree::Complete ? '*'
                                                                    : '.');
  else
    OS.indent(Indent);
  OS << N.ASTNode.getNodeKind().asStringRef() << " ";
  N.ASTNode.print(OS, PrintPolicy);
  OS << "\n";
  for (const Node *Child : N.Children)
    print(OS, *Child, Indent + 2);
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

SelectionTree::SelectionTree(ASTContext &AST, unsigned Begin, unsigned End)
    : PrintPolicy(AST.getLangOpts()) {
  // No fundamental reason the selection needs to be in the main file,
  // but that's all clangd has needed so far.
  FileID FID = AST.getSourceManager().getMainFileID();
  if (Begin == End)
    std::tie(Begin, End) = pointBounds(Begin, FID, AST);
  PrintPolicy.TerseOutput = true;

  Nodes = SelectionVisitor::collect(AST, Begin, End, FID);
  Root = Nodes.empty() ? nullptr : &Nodes.front();
}

SelectionTree::SelectionTree(ASTContext &AST, unsigned Offset)
    : SelectionTree(AST, Offset, Offset) {}

const Node *SelectionTree::commonAncestor() const {
  if (!Root)
    return nullptr;
  for (const Node *Ancestor = Root;; Ancestor = Ancestor->Children.front()) {
    if (Ancestor->Selected || Ancestor->Children.size() > 1)
      return Ancestor;
    // The tree only contains ancestors of the interesting nodes.
    assert(!Ancestor->Children.empty() && "bad node in selection tree");
  }
}

} // namespace clangd
} // namespace clang
