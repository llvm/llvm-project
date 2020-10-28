//===--- Selection.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Selection.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>

namespace clang {
namespace clangd {
namespace {
using Node = SelectionTree::Node;
using ast_type_traits::DynTypedNode;

// Measure the fraction of selections that were enabled by recovery AST.
void recordMetrics(const SelectionTree &S) {
  static constexpr trace::Metric SelectionUsedRecovery(
      "selection_recovery", trace::Metric::Distribution);
  static constexpr trace::Metric RecoveryType("selection_recovery_type",
                                              trace::Metric::Distribution);
  const auto *Common = S.commonAncestor();
  for (const auto *N = Common; N; N = N->Parent) {
    if (const auto *RE = N->ASTNode.get<RecoveryExpr>()) {
      SelectionUsedRecovery.record(1); // used recovery ast.
      RecoveryType.record(RE->isTypeDependent() ? 0 : 1);
      return;
    }
  }
  if (Common)
    SelectionUsedRecovery.record(0); // unused.
}

// An IntervalSet maintains a set of disjoint subranges of an array.
//
// Initially, it contains the entire array.
//           [-----------------------------------------------------------]
//
// When a range is erased(), it will typically split the array in two.
//  Claim:                     [--------------------]
//  after:   [----------------]                      [-------------------]
//
// erase() returns the segments actually erased. Given the state above:
//  Claim:          [---------------------------------------]
//  Out:            [---------]                      [------]
//  After:   [-----]                                         [-----------]
//
// It is used to track (expanded) tokens not yet associated with an AST node.
// On traversing an AST node, its token range is erased from the unclaimed set.
// The tokens actually removed are associated with that node, and hit-tested
// against the selection to determine whether the node is selected.
template <typename T> class IntervalSet {
public:
  IntervalSet(llvm::ArrayRef<T> Range) { UnclaimedRanges.insert(Range); }

  // Removes the elements of Claim from the set, modifying or removing ranges
  // that overlap it.
  // Returns the continuous subranges of Claim that were actually removed.
  llvm::SmallVector<llvm::ArrayRef<T>, 4> erase(llvm::ArrayRef<T> Claim) {
    llvm::SmallVector<llvm::ArrayRef<T>, 4> Out;
    if (Claim.empty())
      return Out;

    // General case:
    // Claim:                   [-----------------]
    // UnclaimedRanges: [-A-] [-B-] [-C-] [-D-] [-E-] [-F-] [-G-]
    // Overlap:               ^first                  ^second
    // Ranges C and D are fully included. Ranges B and E must be trimmed.
    auto Overlap = std::make_pair(
        UnclaimedRanges.lower_bound({Claim.begin(), Claim.begin()}), // C
        UnclaimedRanges.lower_bound({Claim.end(), Claim.end()}));    // F
    // Rewind to cover B.
    if (Overlap.first != UnclaimedRanges.begin()) {
      --Overlap.first;
      // ...unless B isn't selected at all.
      if (Overlap.first->end() <= Claim.begin())
        ++Overlap.first;
    }
    if (Overlap.first == Overlap.second)
      return Out;

    // First, copy all overlapping ranges into the output.
    auto OutFirst = Out.insert(Out.end(), Overlap.first, Overlap.second);
    // If any of the overlapping ranges were sliced by the claim, split them:
    //  - restrict the returned range to the claimed part
    //  - save the unclaimed part so it can be reinserted
    llvm::ArrayRef<T> RemainingHead, RemainingTail;
    if (Claim.begin() > OutFirst->begin()) {
      RemainingHead = {OutFirst->begin(), Claim.begin()};
      *OutFirst = {Claim.begin(), OutFirst->end()};
    }
    if (Claim.end() < Out.back().end()) {
      RemainingTail = {Claim.end(), Out.back().end()};
      Out.back() = {Out.back().begin(), Claim.end()};
    }

    // Erase all the overlapping ranges (invalidating all iterators).
    UnclaimedRanges.erase(Overlap.first, Overlap.second);
    // Reinsert ranges that were merely trimmed.
    if (!RemainingHead.empty())
      UnclaimedRanges.insert(RemainingHead);
    if (!RemainingTail.empty())
      UnclaimedRanges.insert(RemainingTail);

    return Out;
  }

private:
  using TokenRange = llvm::ArrayRef<T>;
  struct RangeLess {
    bool operator()(llvm::ArrayRef<T> L, llvm::ArrayRef<T> R) const {
      return L.begin() < R.begin();
    }
  };

  // Disjoint sorted unclaimed ranges of expanded tokens.
  std::set<llvm::ArrayRef<T>, RangeLess> UnclaimedRanges;
};

// Sentinel value for the selectedness of a node where we've seen no tokens yet.
// This resolves to Unselected if no tokens are ever seen.
// But Unselected + Complete -> Partial, while NoTokens + Complete --> Complete.
// This value is never exposed publicly.
constexpr SelectionTree::Selection NoTokens =
    static_cast<SelectionTree::Selection>(
        static_cast<unsigned char>(SelectionTree::Complete + 1));

// Nodes start with NoTokens, and then use this function to aggregate the
// selectedness as more tokens are found.
void update(SelectionTree::Selection &Result, SelectionTree::Selection New) {
  if (New == NoTokens)
    return;
  if (Result == NoTokens)
    Result = New;
  else if (Result != New)
    // Can only be completely selected (or unselected) if all tokens are.
    Result = SelectionTree::Partial;
}

// As well as comments, don't count semicolons as real tokens.
// They're not properly claimed as expr-statement is missing from the AST.
bool shouldIgnore(const syntax::Token &Tok) {
  return Tok.kind() == tok::comment || Tok.kind() == tok::semi;
}

// Determine whether 'Target' is the first expansion of the macro
// argument whose top-level spelling location is 'SpellingLoc'.
bool isFirstExpansion(FileID Target, SourceLocation SpellingLoc,
                      const SourceManager &SM) {
  SourceLocation Prev = SpellingLoc;
  while (true) {
    // If the arg is expanded multiple times, getMacroArgExpandedLocation()
    // returns the first expansion.
    SourceLocation Next = SM.getMacroArgExpandedLocation(Prev);
    // So if we reach the target, target is the first-expansion of the
    // first-expansion ...
    if (SM.getFileID(Next) == Target)
      return true;

    // Otherwise, if the FileID stops changing, we've reached the innermost
    // macro expansion, and Target was on a different branch.
    if (SM.getFileID(Next) == SM.getFileID(Prev))
      return false;

    Prev = Next;
  }
  return false;
}

// SelectionTester can determine whether a range of tokens from the PP-expanded
// stream (corresponding to an AST node) is considered selected.
//
// When the tokens result from macro expansions, the appropriate tokens in the
// main file are examined (macro invocation or args). Similarly for #includes.
// However, only the first expansion of a given spelled token is considered
// selected.
//
// It tests each token in the range (not just the endpoints) as contiguous
// expanded tokens may not have contiguous spellings (with macros).
//
// Non-token text, and tokens not modeled in the AST (comments, semicolons)
// are ignored when determining selectedness.
class SelectionTester {
public:
  // The selection is offsets [SelBegin, SelEnd) in SelFile.
  SelectionTester(const syntax::TokenBuffer &Buf, FileID SelFile,
                  unsigned SelBegin, unsigned SelEnd, const SourceManager &SM)
      : SelFile(SelFile), SM(SM) {
    // Find all tokens (partially) selected in the file.
    auto AllSpelledTokens = Buf.spelledTokens(SelFile);
    const syntax::Token *SelFirst =
        llvm::partition_point(AllSpelledTokens, [&](const syntax::Token &Tok) {
          return SM.getFileOffset(Tok.endLocation()) <= SelBegin;
        });
    const syntax::Token *SelLimit = std::partition_point(
        SelFirst, AllSpelledTokens.end(), [&](const syntax::Token &Tok) {
          return SM.getFileOffset(Tok.location()) < SelEnd;
        });
    auto Sel = llvm::makeArrayRef(SelFirst, SelLimit);
    // Find which of these are preprocessed to nothing and should be ignored.
    std::vector<bool> PPIgnored(Sel.size(), false);
    for (const syntax::TokenBuffer::Expansion &X :
         Buf.expansionsOverlapping(Sel)) {
      if (X.Expanded.empty()) {
        for (const syntax::Token &Tok : X.Spelled) {
          if (&Tok >= SelFirst && &Tok < SelLimit)
            PPIgnored[&Tok - SelFirst] = true;
        }
      }
    }
    // Precompute selectedness and offset for selected spelled tokens.
    for (unsigned I = 0; I < Sel.size(); ++I) {
      if (shouldIgnore(Sel[I]) || PPIgnored[I])
        continue;
      SpelledTokens.emplace_back();
      Tok &S = SpelledTokens.back();
      S.Offset = SM.getFileOffset(Sel[I].location());
      if (S.Offset >= SelBegin && S.Offset + Sel[I].length() <= SelEnd)
        S.Selected = SelectionTree::Complete;
      else
        S.Selected = SelectionTree::Partial;
    }
  }

  // Test whether a consecutive range of tokens is selected.
  // The tokens are taken from the expanded token stream.
  SelectionTree::Selection
  test(llvm::ArrayRef<syntax::Token> ExpandedTokens) const {
    if (SpelledTokens.empty())
      return NoTokens;
    SelectionTree::Selection Result = NoTokens;
    while (!ExpandedTokens.empty()) {
      // Take consecutive tokens from the same context together for efficiency.
      FileID FID = SM.getFileID(ExpandedTokens.front().location());
      auto Batch = ExpandedTokens.take_while([&](const syntax::Token &T) {
        return SM.getFileID(T.location()) == FID;
      });
      assert(!Batch.empty());
      ExpandedTokens = ExpandedTokens.drop_front(Batch.size());

      update(Result, testChunk(FID, Batch));
    }
    return Result;
  }

  // Cheap check whether any of the tokens in R might be selected.
  // If it returns false, test() will return NoTokens or Unselected.
  // If it returns true, test() may return any value.
  bool mayHit(SourceRange R) const {
    if (SpelledTokens.empty())
      return false;
    auto B = SM.getDecomposedLoc(R.getBegin());
    auto E = SM.getDecomposedLoc(R.getEnd());
    if (B.first == SelFile && E.first == SelFile)
      if (E.second < SpelledTokens.front().Offset ||
          B.second > SpelledTokens.back().Offset)
        return false;
    return true;
  }

private:
  // Hit-test a consecutive range of tokens from a single file ID.
  SelectionTree::Selection
  testChunk(FileID FID, llvm::ArrayRef<syntax::Token> Batch) const {
    assert(!Batch.empty());
    SourceLocation StartLoc = Batch.front().location();
    // There are several possible categories of FileID depending on how the
    // preprocessor was used to generate these tokens:
    //   main file, #included file, macro args, macro bodies.
    // We need to identify the main-file tokens that represent Batch, and
    // determine whether we want to exclusively claim them. Regular tokens
    // represent one AST construct, but a macro invocation can represent many.

    // Handle tokens written directly in the main file.
    if (FID == SelFile) {
      return testTokenRange(SM.getFileOffset(Batch.front().location()),
                            SM.getFileOffset(Batch.back().location()));
    }

    // Handle tokens in another file #included into the main file.
    // Check if the #include is selected, but don't claim it exclusively.
    if (StartLoc.isFileID()) {
      for (SourceLocation Loc = Batch.front().location(); Loc.isValid();
           Loc = SM.getIncludeLoc(SM.getFileID(Loc))) {
        if (SM.getFileID(Loc) == SelFile)
          // FIXME: use whole #include directive, not just the filename string.
          return testToken(SM.getFileOffset(Loc));
      }
      return NoTokens;
    }

    assert(StartLoc.isMacroID());
    // Handle tokens that were passed as a macro argument.
    SourceLocation ArgStart = SM.getTopMacroCallerLoc(StartLoc);
    if (SM.getFileID(ArgStart) == SelFile) {
      if (isFirstExpansion(FID, ArgStart, SM)) {
        SourceLocation ArgEnd =
            SM.getTopMacroCallerLoc(Batch.back().location());
        return testTokenRange(SM.getFileOffset(ArgStart),
                              SM.getFileOffset(ArgEnd));
      } else {
        /* fall through and treat as part of the macro body */
      }
    }

    // Handle tokens produced by non-argument macro expansion.
    // Check if the macro name is selected, don't claim it exclusively.
    auto Expansion = SM.getDecomposedExpansionLoc(StartLoc);
    if (Expansion.first == SelFile)
      // FIXME: also check ( and ) for function-like macros?
      return testToken(Expansion.second);
    else
      return NoTokens;
  }

  // Is the closed token range [Begin, End] selected?
  SelectionTree::Selection testTokenRange(unsigned Begin, unsigned End) const {
    assert(Begin <= End);
    // Outside the selection entirely?
    if (End < SpelledTokens.front().Offset ||
        Begin > SpelledTokens.back().Offset)
      return SelectionTree::Unselected;

    // Compute range of tokens.
    auto B = llvm::partition_point(
        SpelledTokens, [&](const Tok &T) { return T.Offset < Begin; });
    auto E = std::partition_point(
        B, SpelledTokens.end(), [&](const Tok &T) { return T.Offset <= End; });

    // Aggregate selectedness of tokens in range.
    bool ExtendsOutsideSelection = Begin < SpelledTokens.front().Offset ||
                                   End > SpelledTokens.back().Offset;
    SelectionTree::Selection Result =
        ExtendsOutsideSelection ? SelectionTree::Unselected : NoTokens;
    for (auto It = B; It != E; ++It)
      update(Result, It->Selected);
    return Result;
  }

  // Is the token at `Offset` selected?
  SelectionTree::Selection testToken(unsigned Offset) const {
    // Outside the selection entirely?
    if (Offset < SpelledTokens.front().Offset ||
        Offset > SpelledTokens.back().Offset)
      return SelectionTree::Unselected;
    // Find the token, if it exists.
    auto It = llvm::partition_point(
        SpelledTokens, [&](const Tok &T) { return T.Offset < Offset; });
    if (It != SpelledTokens.end() && It->Offset == Offset)
      return It->Selected;
    return NoTokens;
  }

  struct Tok {
    unsigned Offset;
    SelectionTree::Selection Selected;
  };
  std::vector<Tok> SpelledTokens;
  FileID SelFile;
  const SourceManager &SM;
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

bool isImplicit(const Stmt *S) {
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
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &X) {
    return traverseNode(&X,
                        [&] { return Base::TraverseTemplateArgumentLoc(X); });
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
  // OpaqueValueExpr blocks traversal, we must explicitly traverse it.
  bool TraverseOpaqueValueExpr(OpaqueValueExpr *E) {
    return traverseNode(E, [&] { return TraverseStmt(E->getSourceExpr()); });
  }
  // We only want to traverse the *syntactic form* to understand the selection.
  bool TraversePseudoObjectExpr(PseudoObjectExpr *E) {
    return traverseNode(E, [&] { return TraverseStmt(E->getSyntacticForm()); });
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
        TokenBuf(Tokens), SelChecker(Tokens, SelFile, SelBegin, SelEnd, SM),
        UnclaimedExpandedTokens(Tokens.expandedTokens()) {
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
    if (auto *TL = N.get<TypeLoc>()) {
      // FIXME: TypeLoc::getBeginLoc()/getEndLoc() are pretty fragile
      // heuristics. We should consider only pruning critical TypeLoc nodes, to
      // be more robust.

      // DeclTypeTypeLoc::getSourceRange() is incomplete, which would lead to
      // failing
      // to descend into the child expression.
      // decltype(2+2);
      // ~~~~~~~~~~~~~ <-- correct range
      // ~~~~~~~~      <-- range reported by getSourceRange()
      // ~~~~~~~~~~~~  <-- range with this hack(i.e, missing closing paren)
      // FIXME: Alter DecltypeTypeLoc to contain parentheses locations and get
      // rid of this patch.
      if (auto DT = TL->getAs<DecltypeTypeLoc>())
        S.setEnd(DT.getUnderlyingExpr()->getEndLoc());
      // AttributedTypeLoc may point to the attribute's range, NOT the modified
      // type's range.
      if (auto AT = TL->getAs<AttributedTypeLoc>())
        S = AT.getModifiedLoc().getSourceRange();
    }
    if (!SelChecker.mayHit(S)) {
      dlog("{1}skip: {0}", printNodeToString(N, PrintPolicy), indent());
      dlog("{1}skipped range = {0}", S.printToString(SM), indent(1));
      return true;
    }
    return false;
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
    Nodes.back().Selected = NoTokens;
    Stack.push(&Nodes.back());
    claimRange(Early, Nodes.back().Selected);
  }

  // Pops a node off the ancestor stack, and finalizes it. Pairs with push().
  // Performs primary hit detection.
  void pop() {
    Node &N = *Stack.top();
    dlog("{1}pop: {0}", printNodeToString(N.ASTNode, PrintPolicy), indent(-1));
    claimRange(N.ASTNode.getSourceRange(), N.Selected);
    if (N.Selected == NoTokens)
      N.Selected = SelectionTree::Unselected;
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
      // We want constructor name to be claimed by TypeLoc not the constructor
      // itself. Similar for deduction guides, we rather want to select the
      // underlying TypeLoc.
      // FIXME: Unfortunately this doesn't work, even though RecursiveASTVisitor
      // traverses the underlying TypeLoc inside DeclarationName, it is null for
      // constructors.
      if (isa<CXXConstructorDecl>(D) || isa<CXXDeductionGuideDecl>(D))
        return SourceRange();
      // This will capture Field, Function, MSProperty, NonTypeTemplateParm and
      // VarDecls. We want the name in the declarator to be claimed by the decl
      // and not by any children. For example:
      // void [[foo]]();
      // int (*[[s]])();
      // struct X { int [[hash]] [32]; [[operator]] int();}
      if (const auto *DD = llvm::dyn_cast<DeclaratorDecl>(D))
        return DD->getLocation();
    } else if (const auto *CCI = N.get<CXXCtorInitializer>()) {
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
    for (const auto &ClaimedRange :
         UnclaimedExpandedTokens.erase(TokenBuf.expandedTokens(S)))
      update(Result, SelChecker.test(ClaimedRange));

    if (Result && Result != NoTokens)
      dlog("{1}hit selection: {0}", S.printToString(SM), indent());
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
  const syntax::TokenBuffer &TokenBuf;
  std::stack<Node *> Stack;
  SelectionTester SelChecker;
  IntervalSet<syntax::Token> UnclaimedExpandedTokens;
  std::deque<Node> Nodes; // Stable pointers as we add more nodes.
};

} // namespace

llvm::SmallString<256> abbreviatedString(DynTypedNode N,
                                         const PrintingPolicy &PP) {
  llvm::SmallString<256> Result;
  {
    llvm::raw_svector_ostream OS(Result);
    N.print(OS, PP);
  }
  auto Pos = Result.find('\n');
  if (Pos != llvm::StringRef::npos) {
    bool MoreText =
        !llvm::all_of(llvm::StringRef(Result).drop_front(Pos), llvm::isSpace);
    Result.resize(Pos);
    if (MoreText)
      Result.append(" …");
  }
  return Result;
}

void SelectionTree::print(llvm::raw_ostream &OS, const SelectionTree::Node &N,
                          int Indent) const {
  if (N.Selected)
    OS.indent(Indent - 1) << (N.Selected == SelectionTree::Complete ? '*'
                                                                    : '.');
  else
    OS.indent(Indent);
  printNodeKind(OS, N.ASTNode);
  OS << ' ' << abbreviatedString(N.ASTNode, PrintPolicy) << "\n";
  for (const Node *Child : N.Children)
    print(OS, *Child, Indent + 2);
}

std::string SelectionTree::Node::kind() const {
  std::string S;
  llvm::raw_string_ostream OS(S);
  printNodeKind(OS, ASTNode);
  return std::move(OS.str());
}

// Decide which selections emulate a "point" query in between characters.
// If it's ambiguous (the neighboring characters are selectable tokens), returns
// both possibilities in preference order.
// Always returns at least one range - if no tokens touched, and empty range.
static llvm::SmallVector<std::pair<unsigned, unsigned>, 2>
pointBounds(unsigned Offset, const syntax::TokenBuffer &Tokens) {
  const auto &SM = Tokens.sourceManager();
  SourceLocation Loc = SM.getComposedLoc(SM.getMainFileID(), Offset);
  llvm::SmallVector<std::pair<unsigned, unsigned>, 2> Result;
  // Prefer right token over left.
  for (const syntax::Token &Tok :
       llvm::reverse(spelledTokensTouching(Loc, Tokens))) {
    if (shouldIgnore(Tok))
      continue;
    unsigned Offset = Tokens.sourceManager().getFileOffset(Tok.location());
    Result.emplace_back(Offset, Offset + Tok.length());
  }
  if (Result.empty())
    Result.emplace_back(Offset, Offset);
  return Result;
}

bool SelectionTree::createEach(ASTContext &AST,
                               const syntax::TokenBuffer &Tokens,
                               unsigned Begin, unsigned End,
                               llvm::function_ref<bool(SelectionTree)> Func) {
  if (Begin != End)
    return Func(SelectionTree(AST, Tokens, Begin, End));
  for (std::pair<unsigned, unsigned> Bounds : pointBounds(Begin, Tokens))
    if (Func(SelectionTree(AST, Tokens, Bounds.first, Bounds.second)))
      return true;
  return false;
}

SelectionTree SelectionTree::createRight(ASTContext &AST,
                                         const syntax::TokenBuffer &Tokens,
                                         unsigned int Begin, unsigned int End) {
  llvm::Optional<SelectionTree> Result;
  createEach(AST, Tokens, Begin, End, [&](SelectionTree T) {
    Result = std::move(T);
    return true;
  });
  return std::move(*Result);
}

SelectionTree::SelectionTree(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                             unsigned Begin, unsigned End)
    : PrintPolicy(AST.getLangOpts()) {
  // No fundamental reason the selection needs to be in the main file,
  // but that's all clangd has needed so far.
  const SourceManager &SM = AST.getSourceManager();
  FileID FID = SM.getMainFileID();
  PrintPolicy.TerseOutput = true;
  PrintPolicy.IncludeNewlines = false;

  dlog("Computing selection for {0}",
       SourceRange(SM.getComposedLoc(FID, Begin), SM.getComposedLoc(FID, End))
           .printToString(SM));
  Nodes = SelectionVisitor::collect(AST, Tokens, PrintPolicy, Begin, End, FID);
  Root = Nodes.empty() ? nullptr : &Nodes.front();
  recordMetrics(*this);
  dlog("Built selection tree\n{0}", *this);
}

const Node *SelectionTree::commonAncestor() const {
  const Node *Ancestor = Root;
  while (Ancestor->Children.size() == 1 && !Ancestor->Selected)
    Ancestor = Ancestor->Children.front();
  // Returning nullptr here is a bit unprincipled, but it makes the API safer:
  // the TranslationUnitDecl contains all of the preamble, so traversing it is a
  // performance cliff. Callers can check for null and use root() if they want.
  return Ancestor != Root ? Ancestor : nullptr;
}

const DeclContext &SelectionTree::Node::getDeclContext() const {
  for (const Node *CurrentNode = this; CurrentNode != nullptr;
       CurrentNode = CurrentNode->Parent) {
    if (const Decl *Current = CurrentNode->ASTNode.get<Decl>()) {
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
