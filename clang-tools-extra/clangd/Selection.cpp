//===--- Selection.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Selection.h"
#include "AST.h"
#include "support/Logger.h"
#include "support/Trace.h"
#include "clang/AST/ASTConcept.h"
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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
#include <string>

namespace clang {
namespace clangd {
namespace {
using Node = SelectionTree::Node;

// Measure the fraction of selections that were enabled by recovery AST.
void recordMetrics(const SelectionTree &S, const LangOptions &Lang) {
  if (!trace::enabled())
    return;
  const char *LanguageLabel = Lang.CPlusPlus ? "C++" : Lang.ObjC ? "ObjC" : "C";
  static constexpr trace::Metric SelectionUsedRecovery(
      "selection_recovery", trace::Metric::Distribution, "language");
  static constexpr trace::Metric RecoveryType(
      "selection_recovery_type", trace::Metric::Distribution, "language");
  const auto *Common = S.commonAncestor();
  for (const auto *N = Common; N; N = N->Parent) {
    if (const auto *RE = N->ASTNode.get<RecoveryExpr>()) {
      SelectionUsedRecovery.record(1, LanguageLabel); // used recovery ast.
      RecoveryType.record(RE->isTypeDependent() ? 0 : 1, LanguageLabel);
      return;
    }
  }
  if (Common)
    SelectionUsedRecovery.record(0, LanguageLabel); // unused.
}

// Return the range covering a node and all its children.
SourceRange getSourceRange(const DynTypedNode &N) {
  // MemberExprs to implicitly access anonymous fields should not claim any
  // tokens for themselves. Given:
  //   struct A { struct { int b; }; };
  // The clang AST reports the following nodes for an access to b:
  //   A().b;
  //   [----] MemberExpr, base = A().<anonymous>, member = b
  //   [----] MemberExpr: base = A(), member = <anonymous>
  //   [-]    CXXConstructExpr
  // For our purposes, we don't want the second MemberExpr to own any tokens,
  // so we reduce its range to match the CXXConstructExpr.
  // (It's not clear that changing the clang AST would be correct in general).
  if (const auto *ME = N.get<MemberExpr>()) {
    if (!ME->getMemberDecl()->getDeclName())
      return ME->getBase()
                 ? getSourceRange(DynTypedNode::create(*ME->getBase()))
                 : SourceRange();
  }
  return N.getSourceRange();
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
  llvm::SmallVector<llvm::ArrayRef<T>> erase(llvm::ArrayRef<T> Claim) {
    llvm::SmallVector<llvm::ArrayRef<T>> Out;
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
  switch (Tok.kind()) {
    // Even "attached" comments are not considered part of a node's range.
    case tok::comment:
    // The AST doesn't directly store locations for terminating semicolons.
    case tok::semi:
    // We don't have locations for cvr-qualifiers: see QualifiedTypeLoc.
    case tok::kw_const:
    case tok::kw_volatile:
    case tok::kw_restrict:
      return true;
    default:
      return false;
  }
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
      : SelFile(SelFile), SelFileBounds(SM.getLocForStartOfFile(SelFile),
                                        SM.getLocForEndOfFile(SelFile)),
        SM(SM) {
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
    llvm::BitVector PPIgnored(Sel.size(), false);
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
      SelectedSpelled.emplace_back();
      Tok &S = SelectedSpelled.back();
      S.Offset = SM.getFileOffset(Sel[I].location());
      if (S.Offset >= SelBegin && S.Offset + Sel[I].length() <= SelEnd)
        S.Selected = SelectionTree::Complete;
      else
        S.Selected = SelectionTree::Partial;
    }
    MaybeSelectedExpanded = computeMaybeSelectedExpandedTokens(Buf);
  }

  // Test whether a consecutive range of tokens is selected.
  // The tokens are taken from the expanded token stream.
  SelectionTree::Selection
  test(llvm::ArrayRef<syntax::Token> ExpandedTokens) const {
    if (ExpandedTokens.empty())
      return NoTokens;
    if (SelectedSpelled.empty())
      return SelectionTree::Unselected;
    // Cheap (pointer) check whether any of the tokens could touch selection.
    // In most cases, the node's overall source range touches ExpandedTokens,
    // or we would have failed mayHit(). However now we're only considering
    // the *unclaimed* spans of expanded tokens.
    // This is a significant performance improvement when a lot of nodes
    // surround the selection, including when generated by macros.
    if (MaybeSelectedExpanded.empty() ||
        &ExpandedTokens.front() > &MaybeSelectedExpanded.back() ||
        &ExpandedTokens.back() < &MaybeSelectedExpanded.front()) {
      return SelectionTree::Unselected;
    }

    // The eof token is used as a sentinel.
    // In general, source range from an AST node should not claim the eof token,
    // but it could occur for unmatched-bracket cases.
    // FIXME: fix it in TokenBuffer, expandedTokens(SourceRange) should not
    // return the eof token.
    if (ExpandedTokens.back().kind() == tok::eof)
      ExpandedTokens = ExpandedTokens.drop_back();

    SelectionTree::Selection Result = NoTokens;
    while (!ExpandedTokens.empty()) {
      // Take consecutive tokens from the same context together for efficiency.
      SourceLocation Start = ExpandedTokens.front().location();
      FileID FID = SM.getFileID(Start);
      // Comparing SourceLocations against bounds is cheaper than getFileID().
      SourceLocation Limit = SM.getComposedLoc(FID, SM.getFileIDSize(FID));
      auto Batch = ExpandedTokens.take_while([&](const syntax::Token &T) {
        return T.location() >= Start && T.location() < Limit;
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
    if (SelectedSpelled.empty() || MaybeSelectedExpanded.empty())
      return false;
    // If the node starts after the selection ends, it is not selected.
    // Tokens a macro location might claim are >= its expansion start.
    // So if the expansion start > last selected token, we can prune it.
    // (This is particularly helpful for GTest's TEST macro).
    if (auto B = offsetInSelFile(getExpansionStart(R.getBegin())))
      if (*B > SelectedSpelled.back().Offset)
        return false;
    // If the node ends before the selection begins, it is not selected.
    SourceLocation EndLoc = R.getEnd();
    while (EndLoc.isMacroID())
      EndLoc = SM.getImmediateExpansionRange(EndLoc).getEnd();
    // In the rare case that the expansion range is a char range, EndLoc is
    // ~one token too far to the right. We may fail to prune, that's OK.
    if (auto E = offsetInSelFile(EndLoc))
      if (*E < SelectedSpelled.front().Offset)
        return false;
    return true;
  }

private:
  // Plausible expanded tokens that might be affected by the selection.
  // This is an overestimate, it may contain tokens that are not selected.
  // The point is to allow cheap pruning in test()
  llvm::ArrayRef<syntax::Token>
  computeMaybeSelectedExpandedTokens(const syntax::TokenBuffer &Toks) {
    if (SelectedSpelled.empty())
      return {};

    auto LastAffectedToken = [&](SourceLocation Loc) {
      auto Offset = offsetInSelFile(Loc);
      while (Loc.isValid() && !Offset) {
        Loc = Loc.isMacroID() ? SM.getImmediateExpansionRange(Loc).getEnd()
                              : SM.getIncludeLoc(SM.getFileID(Loc));
        Offset = offsetInSelFile(Loc);
      }
      return Offset;
    };
    auto FirstAffectedToken = [&](SourceLocation Loc) {
      auto Offset = offsetInSelFile(Loc);
      while (Loc.isValid() && !Offset) {
        Loc = Loc.isMacroID() ? SM.getImmediateExpansionRange(Loc).getBegin()
                              : SM.getIncludeLoc(SM.getFileID(Loc));
        Offset = offsetInSelFile(Loc);
      }
      return Offset;
    };

    const syntax::Token *Start = llvm::partition_point(
        Toks.expandedTokens(),
        [&, First = SelectedSpelled.front().Offset](const syntax::Token &Tok) {
          if (Tok.kind() == tok::eof)
            return false;
          // Implausible if upperbound(Tok) < First.
          if (auto Offset = LastAffectedToken(Tok.location()))
            return *Offset < First;
          // A prefix of the expanded tokens may be from an an implicit
          // inclusion (e.g. preamble patch, or command-line -include).
          return true;
        });

    bool EndInvalid = false;
    const syntax::Token *End = std::partition_point(
        Start, Toks.expandedTokens().end(),
        [&, Last = SelectedSpelled.back().Offset](const syntax::Token &Tok) {
          if (Tok.kind() == tok::eof)
            return false;
          // Plausible if lowerbound(Tok) <= Last.
          if (auto Offset = FirstAffectedToken(Tok.location()))
            return *Offset <= Last;
          // Shouldn't happen: once we've seen tokens traceable to the main
          // file, there shouldn't be any more implicit inclusions.
          assert(false && "Expanded token could not be resolved to main file!");
          EndInvalid = true;
          return true; // conservatively assume this token can overlap
        });
    if (EndInvalid)
      End = Toks.expandedTokens().end();

    return llvm::makeArrayRef(Start, End);
  }

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
      return testTokenRange(*offsetInSelFile(Batch.front().location()),
                            *offsetInSelFile(Batch.back().location()));
    }

    // Handle tokens in another file #included into the main file.
    // Check if the #include is selected, but don't claim it exclusively.
    if (StartLoc.isFileID()) {
      for (SourceLocation Loc = Batch.front().location(); Loc.isValid();
           Loc = SM.getIncludeLoc(SM.getFileID(Loc))) {
        if (auto Offset = offsetInSelFile(Loc))
          // FIXME: use whole #include directive, not just the filename string.
          return testToken(*Offset);
      }
      return NoTokens;
    }

    assert(StartLoc.isMacroID());
    // Handle tokens that were passed as a macro argument.
    SourceLocation ArgStart = SM.getTopMacroCallerLoc(StartLoc);
    if (auto ArgOffset = offsetInSelFile(ArgStart)) {
      if (isFirstExpansion(FID, ArgStart, SM)) {
        SourceLocation ArgEnd =
            SM.getTopMacroCallerLoc(Batch.back().location());
        return testTokenRange(*ArgOffset, *offsetInSelFile(ArgEnd));
      } else { // NOLINT(llvm-else-after-return)
        /* fall through and treat as part of the macro body */
      }
    }

    // Handle tokens produced by non-argument macro expansion.
    // Check if the macro name is selected, don't claim it exclusively.
    if (auto ExpansionOffset = offsetInSelFile(getExpansionStart(StartLoc)))
      // FIXME: also check ( and ) for function-like macros?
      return testToken(*ExpansionOffset);
    return NoTokens;
  }

  // Is the closed token range [Begin, End] selected?
  SelectionTree::Selection testTokenRange(unsigned Begin, unsigned End) const {
    assert(Begin <= End);
    // Outside the selection entirely?
    if (End < SelectedSpelled.front().Offset ||
        Begin > SelectedSpelled.back().Offset)
      return SelectionTree::Unselected;

    // Compute range of tokens.
    auto B = llvm::partition_point(
        SelectedSpelled, [&](const Tok &T) { return T.Offset < Begin; });
    auto E = std::partition_point(B, SelectedSpelled.end(), [&](const Tok &T) {
      return T.Offset <= End;
    });

    // Aggregate selectedness of tokens in range.
    bool ExtendsOutsideSelection = Begin < SelectedSpelled.front().Offset ||
                                   End > SelectedSpelled.back().Offset;
    SelectionTree::Selection Result =
        ExtendsOutsideSelection ? SelectionTree::Unselected : NoTokens;
    for (auto It = B; It != E; ++It)
      update(Result, It->Selected);
    return Result;
  }

  // Is the token at `Offset` selected?
  SelectionTree::Selection testToken(unsigned Offset) const {
    // Outside the selection entirely?
    if (Offset < SelectedSpelled.front().Offset ||
        Offset > SelectedSpelled.back().Offset)
      return SelectionTree::Unselected;
    // Find the token, if it exists.
    auto It = llvm::partition_point(
        SelectedSpelled, [&](const Tok &T) { return T.Offset < Offset; });
    if (It != SelectedSpelled.end() && It->Offset == Offset)
      return It->Selected;
    return NoTokens;
  }

  // Decomposes Loc and returns the offset if the file ID is SelFile.
  llvm::Optional<unsigned> offsetInSelFile(SourceLocation Loc) const {
    // Decoding Loc with SM.getDecomposedLoc is relatively expensive.
    // But SourceLocations for a file are numerically contiguous, so we
    // can use cheap integer operations instead.
    if (Loc < SelFileBounds.getBegin() || Loc >= SelFileBounds.getEnd())
      return llvm::None;
    // FIXME: subtracting getRawEncoding() is dubious, move this logic into SM.
    return Loc.getRawEncoding() - SelFileBounds.getBegin().getRawEncoding();
  }

  SourceLocation getExpansionStart(SourceLocation Loc) const {
    while (Loc.isMacroID())
      Loc = SM.getImmediateExpansionRange(Loc).getBegin();
    return Loc;
  }

  struct Tok {
    unsigned Offset;
    SelectionTree::Selection Selected;
  };
  std::vector<Tok> SelectedSpelled;
  llvm::ArrayRef<syntax::Token> MaybeSelectedExpanded;
  FileID SelFile;
  SourceRange SelFileBounds;
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
  // Make sure implicit access of anonymous structs don't end up owning tokens.
  if (auto *ME = llvm::dyn_cast<MemberExpr>(S)) {
    if (auto *FD = llvm::dyn_cast<FieldDecl>(ME->getMemberDecl()))
      if (FD->isAnonymousStructOrUnion())
        // If Base is an implicit CXXThis, then the whole MemberExpr has no
        // tokens. If it's a normal e.g. DeclRef, we treat the MemberExpr like
        // an implicit cast.
        return isImplicit(ME->getBase());
  }
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
  bool TraverseDecl(Decl *X) {
    if (llvm::isa_and_nonnull<TranslationUnitDecl>(X))
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
  bool TraverseCXXBaseSpecifier(const CXXBaseSpecifier &X) {
    return traverseNode(&X, [&] { return Base::TraverseCXXBaseSpecifier(X); });
  }
  bool TraverseAttr(Attr *X) {
    return traverseNode(X, [&] { return Base::TraverseAttr(X); });
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
  bool TraverseObjCProtocolLoc(ObjCProtocolLoc PL) {
    return traverseNode(&PL, [&] { return Base::TraverseObjCProtocolLoc(PL); });
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
  bool TraverseTypeConstraint(const TypeConstraint *C) {
    if (auto *E = C->getImmediatelyDeclaredConstraint()) {
      // Technically this expression is 'implicit' and not traversed by the RAV.
      // However, the range is correct, so we visit expression to avoid adding
      // an extra kind to 'DynTypeNode' that hold 'TypeConstraint'.
      return TraverseStmt(E);
    }
    return Base::TraverseTypeConstraint(C);
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
  // Exception 1: when declarators nest, *inner* declarator is the *outer* type.
  //              e.g. void foo[5](int) is an array of functions.
  // To handle this case, declarators are careful to only claim the tokens they
  // own, rather than claim a range and rely on claim ordering.
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
    SourceRange S = getSourceRange(N);
    if (auto *TL = N.get<TypeLoc>()) {
      // FIXME: TypeLoc::getBeginLoc()/getEndLoc() are pretty fragile
      // heuristics. We should consider only pruning critical TypeLoc nodes, to
      // be more robust.

      // AttributedTypeLoc may point to the attribute's range, NOT the modified
      // type's range.
      if (auto AT = TL->getAs<AttributedTypeLoc>())
        S = AT.getModifiedLoc().getSourceRange();
    }
    // SourceRange often doesn't manage to accurately cover attributes.
    // Fortunately, attributes are rare.
    if (llvm::any_of(getAttributes(N),
                     [](const Attr *A) { return !A->isImplicit(); }))
      return false;
    if (!SelChecker.mayHit(S)) {
      dlog("{2}skip: {0} {1}", printNodeToString(N, PrintPolicy),
           S.printToString(SM), indent());
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
    dlog("{2}push: {0} {1}", printNodeToString(Node, PrintPolicy),
         Node.getSourceRange().printToString(SM), indent());
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
    claimTokensFor(N.ASTNode, N.Selected);
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
      // We want the name in the var-decl to be claimed by the decl itself and
      // not by any children. Ususally, we don't need this, because source
      // ranges of children are not overlapped with their parent's.
      // An exception is lambda captured var decl, where AutoTypeLoc is
      // overlapped with the name loc.
      //    auto fun = [bar = foo]() { ... }
      //                ~~~~~~~~~   VarDecl
      //                ~~~         |- AutoTypeLoc
      if (const auto *DD = llvm::dyn_cast<VarDecl>(D))
        return DD->getLocation();
    }

    return SourceRange();
  }

  // Claim tokens for N, after processing its children.
  // By default this claims all unclaimed tokens in getSourceRange().
  // We override this if we want to claim fewer tokens (e.g. there are gaps).
  void claimTokensFor(const DynTypedNode &N, SelectionTree::Selection &Result) {
    // CXXConstructExpr often shows implicit construction, like `string s;`.
    // Don't associate any tokens with it unless there's some syntax like {}.
    // This prevents it from claiming 's', its primary location.
    if (const auto *CCE = N.get<CXXConstructExpr>()) {
      claimRange(CCE->getParenOrBraceRange(), Result);
      return;
    }
    // ExprWithCleanups is always implicit. It often wraps CXXConstructExpr.
    // Prevent it claiming 's' in the case above.
    if (N.get<ExprWithCleanups>())
      return;

    // Declarators nest "inside out", with parent types inside child ones.
    // Instead of claiming the whole range (clobbering parent tokens), carefully
    // claim the tokens owned by this node and non-declarator children.
    // (We could manipulate traversal order instead, but this is easier).
    //
    // Non-declarator types nest normally, and are handled like other nodes.
    //
    // Example:
    //   Vec<R<int>(*[2])(A<char>)> is a Vec of arrays of pointers to functions,
    //                              which accept A<char> and return R<int>.
    // The TypeLoc hierarchy:
    //   Vec<R<int>(*[2])(A<char>)> m;
    //   Vec<#####################>      TemplateSpecialization Vec
    //       --------[2]----------       `-Array
    //       -------*-------------         `-Pointer
    //       ------(----)---------           `-Paren
    //       ------------(#######)             `-Function
    //       R<###>                              |-TemplateSpecialization R
    //         int                               | `-Builtin int
    //                    A<####>                `-TemplateSpecialization A
    //                      char                   `-Builtin char
    //
    // In each row
    //   --- represents unclaimed parts of the SourceRange.
    //   ### represents parts that children already claimed.
    if (const auto *TL = N.get<TypeLoc>()) {
      if (auto PTL = TL->getAs<ParenTypeLoc>()) {
        claimRange(PTL.getLParenLoc(), Result);
        claimRange(PTL.getRParenLoc(), Result);
        return;
      }
      if (auto ATL = TL->getAs<ArrayTypeLoc>()) {
        claimRange(ATL.getBracketsRange(), Result);
        return;
      }
      if (auto PTL = TL->getAs<PointerTypeLoc>()) {
        claimRange(PTL.getStarLoc(), Result);
        return;
      }
      if (auto FTL = TL->getAs<FunctionTypeLoc>()) {
        claimRange(SourceRange(FTL.getLParenLoc(), FTL.getEndLoc()), Result);
        return;
      }
    }
    claimRange(getSourceRange(N), Result);
  }

  // Perform hit-testing of a complete Node against the selection.
  // This runs for every node in the AST, and must be fast in common cases.
  // This is usually called from pop(), so we can take children into account.
  // The existing state of Result is relevant.
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
    bool MoreText = !llvm::all_of(Result.str().drop_front(Pos), llvm::isSpace);
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
  recordMetrics(*this, AST.getLangOpts());
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
      return *Current->getLexicalDeclContext();
    }
  }
  llvm_unreachable("A tree must always be rooted at TranslationUnitDecl.");
}

const SelectionTree::Node &SelectionTree::Node::ignoreImplicit() const {
  if (Children.size() == 1 &&
      getSourceRange(Children.front()->ASTNode) == getSourceRange(ASTNode))
    return Children.front()->ignoreImplicit();
  return *this;
}

const SelectionTree::Node &SelectionTree::Node::outerImplicit() const {
  if (Parent && getSourceRange(Parent->ASTNode) == getSourceRange(ASTNode))
    return Parent->outerImplicit();
  return *this;
}

} // namespace clangd
} // namespace clang
