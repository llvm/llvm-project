//===--- ASTSlice.cpp - Represents a portion of the AST -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ASTSlice.h"
#include "SourceLocationUtilities.h"
#include "StmtUtils.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include <algorithm>

using namespace clang;
using namespace clang::tooling;

namespace {

/// Searches for AST nodes around the given source location and range that can
/// be used to initiate a refactoring operation.
class ASTSliceFinder : public clang::RecursiveASTVisitor<ASTSliceFinder> {
public:
  explicit ASTSliceFinder(SourceLocation Location, SourceRange SelectionRange,
                          const ASTContext &Context)
      : Location(Location), SelectionRange(SelectionRange), Context(Context) {}

  bool TraverseDecl(Decl *D) {
    if (!D)
      return true;
    if (isa<DeclContext>(D) && !D->isImplicit())
      collectDeclIfInRange(D);
    // TODO: Handle Lambda/Blocks.
    if (!isa<FunctionDecl>(D) && !isa<ObjCMethodDecl>(D)) {
      RecursiveASTVisitor::TraverseDecl(D);
      return true;
    }
    const Decl *PreviousDecl = CurrentDecl;
    CurrentDecl = D;
    RecursiveASTVisitor::TraverseDecl(D);
    CurrentDecl = PreviousDecl;
    return true;
  }

  bool TraverseStmt(Stmt *S) {
    if (!S)
      return true;
    // PseudoObjectExpressions don't have to be parents.
    if (isa<PseudoObjectExpr>(S))
      return RecursiveASTVisitor::TraverseStmt(S);
    llvm::SaveAndRestore<const Stmt *> Parent(ParentStmt, CurrentStmt);
    llvm::SaveAndRestore<const Stmt *> Current(CurrentStmt, S);
    RecursiveASTVisitor::TraverseStmt(S);
    return true;
  }

  bool TraversePseudoObjectExpr(PseudoObjectExpr *E) {
    // Avoid traversing the getter/setter message sends for property
    // expressions.
    TraverseStmt(E->getSyntacticForm());
    return true;
  }

  bool TraverseObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
    RecursiveASTVisitor::TraverseObjCPropertyRefExpr(E);
    // Visit the opaque base manually as it won't be traversed by the
    // PseudoObjectExpr.
    if (E->isObjectReceiver()) {
      if (const auto *Opaque = dyn_cast<OpaqueValueExpr>(E->getBase()))
        TraverseStmt(Opaque->getSourceExpr());
    }
    return true;
  }

  // Statement visitors:

  bool VisitStmt(Stmt *S) {
    collectStmtIfInRange(S, S->getSourceRange());
    return true;
  }

  // Ignore some implicit expressions.

  bool WalkUpFromMaterializeTemporaryExpr(MaterializeTemporaryExpr *E) {
    return true;
  }

  bool WalkUpFromCXXThisExpr(CXXThisExpr *E) {
    if (E->isImplicit())
      return true;
    return RecursiveASTVisitor::WalkUpFromCXXThisExpr(E);
  }

  /// Checks if the given statement and its source range has the location
  /// of interest or overlaps with the selection range, and adds this node to
  /// the set of statements for the slice that's being constructed.
  void collectStmtIfInRange(const Stmt *S, SourceRange Range) {
    SourceLocation Start = Range.getBegin();
    const auto &SM = Context.getSourceManager();
    bool IsStartMacroArg = false;
    if (Start.isMacroID()) {
      if (SM.isMacroArgExpansion(Start)) {
        Start = SM.getSpellingLoc(Start);
        IsStartMacroArg = true;
      } else {
        Start = SM.getExpansionLoc(Start);
      }
    }
    SourceLocation End = Range.getEnd();
    if (End.isMacroID() && SM.isMacroArgExpansion(End)) {
      // Ignore the node that's span across normal code and a macro argument.
      if (IsStartMacroArg)
        End = SM.getSpellingLoc(End);
    }
    End = getPreciseTokenLocEnd(End, SM, Context.getLangOpts());
    if (!isPairOfFileLocations(Start, End))
      return;
    if (SelectionRange.isValid()) {
      if (!areRangesOverlapping(SelectionRange, SourceRange(Start, End),
                                Context.getSourceManager()))
        return;
    } else if (!isPointWithin(Location, Start, End, Context.getSourceManager()))
      return;
    Matches.emplace_back(S, ParentStmt, CurrentDecl, SourceRange(Start, End));
  }

  void collectDeclIfInRange(const Decl *D) {
    SourceLocation Start = D->getSourceRange().getBegin();
    SourceLocation End = getPreciseTokenLocEnd(
        getLexicalEndLocForDecl(D, Context.getSourceManager(),
                                Context.getLangOpts()),
        Context.getSourceManager(), Context.getLangOpts());
    if (!isPairOfFileLocations(Start, End))
      return;
    if (SelectionRange.isValid()) {
      if (!areRangesOverlapping(SelectionRange, SourceRange(Start, End),
                                Context.getSourceManager()))
        return;
    } else if (!isPointWithin(Location, Start, End, Context.getSourceManager()))
      return;
    Matches.emplace_back(D, CurrentDecl, SourceRange(Start, End));
  }

  SmallVector<ASTSlice::Node, 16> Matches;
  /// The point of interest.
  ///
  /// Represents a location at which refactoring should be initiated.
  const SourceLocation Location;
  const SourceRange SelectionRange;
  const ASTContext &Context;
  const Decl *CurrentDecl = nullptr;
  const Stmt *ParentStmt = nullptr, *CurrentStmt = nullptr;
};

} // end anonymous namespace

ASTSlice::SelectedStmt::SelectedStmt(ASTSlice &Slice, const Stmt *S,
                                     unsigned Index)
    : Slice(Slice), S(S), Index(Index) {
  assert(S && "No statement given!");
}

ASTSlice::SelectedDecl::SelectedDecl(const Decl *D) : D(D) {
  assert(D && "No decl given!");
}

const Decl *ASTSlice::SelectedStmt::getParentDecl() {
  return Slice.parentDeclForIndex(Index);
}

ASTSlice::ASTSlice(SourceLocation Location, SourceRange SelectionRange,
                   ASTContext &Context)
    : Context(Context), SelectionLocation(Location),
      SelectionRange(SelectionRange) {
  FileID SearchFile = Context.getSourceManager().getFileID(Location);
  ASTSliceFinder Visitor(Location, SelectionRange, Context);
  SourceLocation EndLoc;
  for (auto *CurrDecl : Context.getTranslationUnitDecl()->decls()) {
    if (EndLoc.isValid() &&
        !Context.getSourceManager().isBeforeInTranslationUnit(
            CurrDecl->getBeginLoc(), EndLoc))
      break;
    const SourceLocation FileLoc =
        Context.getSourceManager().getSpellingLoc(CurrDecl->getBeginLoc());
    if (Context.getSourceManager().getFileID(FileLoc) == SearchFile)
      Visitor.TraverseDecl(CurrDecl);
    // We are only interested in looking at a single top level declaration
    // even if our selection range spans across multiple top level declarations.
    if (!Visitor.Matches.empty()) {
      // Objective-C @implementation declarations might have trailing functions
      // that are declared outside of the @implementation, so continue looking
      // through them.
      if (isa<ObjCImplDecl>(CurrDecl)) {
        EndLoc = CurrDecl->getEndLoc();
        continue;
      }
      break;
    }
  }

  for (auto I = Visitor.Matches.rbegin(), E = Visitor.Matches.rend(); I != E;
       ++I)
    NodeTree.push_back(*I);
}

bool ASTSlice::isSourceRangeSelected(CharSourceRange Range) const {
  SourceRange R = Range.getAsRange();
  if (Range.isTokenRange())
    R.setEnd(getPreciseTokenLocEnd(R.getEnd(), Context.getSourceManager(),
                                   Context.getLangOpts()));
  if (SelectionRange.isInvalid())
    return isPointWithin(SelectionLocation, R.getBegin(), R.getEnd(),
                         Context.getSourceManager());
  return areRangesOverlapping(SelectionRange, R, Context.getSourceManager());
}

/// Find the 'if' statement that acts as the start of the
/// 'if'/'else if'/'else' construct.
static std::pair<const IfStmt *, unsigned>
findIfStmtStart(const IfStmt *If, unsigned Index,
                ArrayRef<ASTSlice::Node> NodeTree) {
  if (Index >= NodeTree.size())
    return {If, Index}; // We've reached the top of the tree, return.
  const auto *ParentIf =
      dyn_cast_or_null<IfStmt>(NodeTree[Index + 1].getStmtOrNull());
  // The current 'if' is actually an 'else if' when the next 'if' has an else
  // statement that points to the current 'if'.
  if (!ParentIf || ParentIf->getElse() != If)
    return {If, Index};
  return findIfStmtStart(ParentIf, Index + 1, NodeTree);
}

/// Find an expression that best represents the given selected expression.
static std::pair<const Stmt *, unsigned>
canonicalizeSelectedExpr(const Stmt *S, unsigned Index,
                         ArrayRef<ASTSlice::Node> NodeTree) {
  const auto Same = std::make_pair(S, Index);
  if (Index + 1 >= NodeTree.size())
    return Same;
  const Stmt *Parent = NodeTree[Index + 1].getStmtOrNull();
  if (!Parent)
    return Same;
  auto Next = std::make_pair(Parent, Index + 1);
  // The entire pseudo expression is selected when just its syntactic
  // form is selected.
  if (isa<Expr>(S)) {
    if (const auto *POE = dyn_cast_or_null<PseudoObjectExpr>(Parent)) {
      if (POE->getSyntacticForm() == S)
        return Next;
    }
  }

  // Look through the implicit casts in the parents.
  unsigned ParentIndex = Index + 1;
  for (; ParentIndex <= NodeTree.size() && isa<ImplicitCastExpr>(Parent);
       ++ParentIndex) {
    const Stmt *NewParent = NodeTree[ParentIndex + 1].getStmtOrNull();
    if (!NewParent)
      break;
    Parent = NewParent;
  }
  Next = std::make_pair(Parent, ParentIndex);

  // The entire ObjC string literal is selected when just its string
  // literal is selected.
  if (isa<StringLiteral>(S) && isa<ObjCStringLiteral>(Parent))
    return Next;
  // The entire call should be selected when just the member expression
  // that refers to the method is selected.
  // FIXME: Check if this can be one of the call arguments.
  if (isa<MemberExpr>(S) && isa<CXXMemberCallExpr>(Parent))
    return Next;
  // The entire call should be selected when just the callee is selected.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(S)) {
    if (const auto *Call = dyn_cast<CallExpr>(Parent)) {
      if (Call->getCalleeDecl() == DRE->getDecl())
        return Next;
    }
    }
  return Same;
}

Optional<ASTSlice::SelectedStmt> ASTSlice::nearestSelectedStmt(
    llvm::function_ref<bool(const Stmt *)> Predicate) {
  for (const auto &Node : llvm::enumerate(NodeTree)) {
    const Stmt *S = Node.value().getStmtOrNull();
    if (!S || !Predicate(S))
      continue;

    // Found the match. Perform any additional adjustments.
    if (isa<Expr>(S)) {
      auto CanonicalExpr = canonicalizeSelectedExpr(S, Node.index(), NodeTree);
      return SelectedStmt(*this, CanonicalExpr.first, CanonicalExpr.second);
    }
    switch (S->getStmtClass()) {
    case Stmt::IfStmtClass: {
      // TODO: Fix findIfStmtStart bug with Index where it will return the
      // index of the last statement.
      auto If = findIfStmtStart(cast<IfStmt>(S), Node.index(), NodeTree);
      return SelectedStmt(*this, If.first, If.second);
    }
    default:
      break;
    }

    return SelectedStmt(*this, S, Node.index());
  }
  return None;
}

Optional<ASTSlice::SelectedStmt>
ASTSlice::nearestSelectedStmt(Stmt::StmtClass Class) {
  return nearestSelectedStmt(
      [Class](const Stmt *S) -> bool { return S->getStmtClass() == Class; });
}

const Stmt *ASTSlice::nearestStmt(Stmt::StmtClass Class) {
  auto Result = nearestSelectedStmt(Class);
  return Result ? Result->getStmt() : nullptr;
}

Optional<ASTSlice::SelectedDecl> ASTSlice::innermostSelectedDecl(
    llvm::function_ref<bool(const Decl *)> Predicate, unsigned Options) {
  if (SelectionRange.isValid()) {
    if (Options & ASTSlice::InnermostDeclOnly) {
      auto Result = getInnermostCompletelySelectedDecl();
      if (!Result)
        return None;
      if (Predicate(Result->getDecl()))
        return Result;
      return None;
    }
    // Traverse down through all of the selected node checking the predicate.
    // TODO: Cache the SelectionRangeOverlap kinds properly instead of relying
    // on getInnermostCompletelySelectedDecl.
    getInnermostCompletelySelectedDecl();
    for (const auto &N : NodeTree) {
      const Decl *D = N.getDeclOrNull();
      if (!D)
        continue;
      if (N.SelectionRangeOverlap != Node::ContainsSelectionRange)
        continue;
      if (Predicate(D))
        return SelectedDecl(D);
    }
    return None;
  }
  for (const auto &Node : llvm::enumerate(NodeTree)) {
    const Decl *D = Node.value().getDeclOrNull();
    if (!D)
      continue;
    if (Predicate(D))
      return SelectedDecl(D);
    if (Options & ASTSlice::InnermostDeclOnly)
      return None;
  }
  return None;
}

Optional<ASTSlice::SelectedDecl>
ASTSlice::innermostSelectedDecl(ArrayRef<Decl::Kind> Classes,
                                unsigned Options) {
  assert(!Classes.empty() && "Expected at least one decl kind");
  return innermostSelectedDecl(
      [&](const Decl *D) {
        for (Decl::Kind Class : Classes) {
          if (D->getKind() == Class)
            return true;
        }
        return false;
      },
      Options);
}

/// Compute the SelectionRangeOverlap kinds for matched AST nodes.
///
/// The overlap kinds are computed only upto the first node that contains the
/// entire selection range.
static void
computeSelectionRangeOverlapKinds(MutableArrayRef<ASTSlice::Node> NodeTree,
                                  SourceRange SelectionRange,
                                  const SourceManager &SM) {
  for (ASTSlice::Node &Node : NodeTree) {
    bool HasStart =
        isPointWithin(SelectionRange.getBegin(), Node.Range.getBegin(),
                      Node.Range.getEnd(), SM);
    bool HasEnd = isPointWithin(SelectionRange.getEnd(), Node.Range.getBegin(),
                                Node.Range.getEnd(), SM);
    if (HasStart && HasEnd)
      Node.SelectionRangeOverlap = ASTSlice::Node::ContainsSelectionRange;
    else if (HasStart)
      Node.SelectionRangeOverlap = ASTSlice::Node::ContainsSelectionRangeStart;
    else if (HasEnd)
      Node.SelectionRangeOverlap = ASTSlice::Node::ContainsSelectionRangeEnd;
  }
}

const Stmt *findFirstStatementAfter(const CompoundStmt *CS, SourceLocation Loc,
                                    const SourceManager &SM) {
  for (const Stmt *S : CS->body()) {
    if (!SM.isBeforeInTranslationUnit(S->getBeginLoc(), Loc))
      return S;
  }
  return nullptr;
}

const Stmt *findLastStatementBefore(const CompoundStmt *CS, SourceLocation Loc,
                                    const Stmt *StartAt,
                                    const SourceManager &SM) {
  auto It = std::find(CS->body_begin(), CS->body_end(), StartAt);
  assert(It != CS->body_end());
  const Stmt *Last = StartAt;
  for (auto E = CS->body_end(); It != E; ++It) {
    const Stmt *S = *It;
    if (!SM.isBeforeInTranslationUnit(S->getBeginLoc(), Loc))
      return Last;
    Last = S;
  }
  return Last;
}

/// Return the source construct that contains the given compound statement.
///
/// This is useful to find the source construct to which the given compound
/// statement belongs to lexically. For example, if we've selected just the
/// body of an if statement, we ideally want to select the entire if statement.
static std::pair<const Stmt *, unsigned>
findCompoundStatementSourceConstruct(const CompoundStmt *CS,
                                     ArrayRef<ASTSlice::Node> NodeTree) {
  for (const auto &Node : llvm::enumerate(NodeTree)) {
    const Stmt *S = Node.value().getStmtOrNull();
    if (!S)
      continue;
    for (const Stmt *Child : S->children()) {
      if (Child == CS) {
        if (isa<CompoundStmt>(S))
          return {CS, 0};
        if (const auto *If = dyn_cast<IfStmt>(S))
          return findIfStmtStart(If, Node.index(), NodeTree);
        return {S, Node.index()};
      }
    }
  }
  // This is the outer compound statement.
  return {CS, 0};
}

/// Return the source construct that contains the given switch case.
static std::pair<const Stmt *, unsigned>
findSwitchSourceConstruct(const SwitchCase *Case,
                          ArrayRef<ASTSlice::Node> NodeTree) {
  for (const auto &Node : llvm::enumerate(NodeTree)) {
    const Stmt *S = Node.value().getStmtOrNull();
    if (!S)
      continue;
    if (isa<SwitchStmt>(S))
      return {S, Node.index()};
  }
  return {Case, 0};
}

SelectedStmtSet SelectedStmtSet::createFromEntirelySelected(const Stmt *S,
                                                            unsigned Index) {
  SelectedStmtSet Result;
  Result.containsSelectionRange = S;
  Result.containsSelectionRangeIndex = Index;
  return Result;
}

Optional<ASTSlice::SelectedDecl>
ASTSlice::getInnermostCompletelySelectedDecl() {
  assert(SelectionRange.isValid() && "No selection range!");
  if (CachedSelectedInnermostDecl)
    return *CachedSelectedInnermostDecl;
  computeSelectionRangeOverlapKinds(NodeTree, SelectionRange,
                                    Context.getSourceManager());
  Optional<SelectedDecl> Result;
  for (const auto &N : llvm::enumerate(NodeTree)) {
    const Decl *D = N.value().getDeclOrNull();
    if (!D)
      continue;
    if (N.value().SelectionRangeOverlap != Node::ContainsSelectionRange)
      continue;
    Result = SelectedDecl(D);
    break;
  }
  CachedSelectedInnermostDecl = Result;
  return Result;
}

static bool isCaseSelected(const SwitchStmt *S, SourceRange SelectionRange,
                           const SourceManager &SM) {
  for (const SwitchCase *Case = S->getSwitchCaseList(); Case;
       Case = Case->getNextSwitchCase()) {
    SourceRange Range(Case->getBeginLoc(), Case->getColonLoc());
    if (areRangesOverlapping(Range, SelectionRange, SM))
      return true;
  }
  return false;
}

Optional<SelectedStmtSet> ASTSlice::computeSelectedStmtSet() {
  if (SelectionRange.isInvalid())
    return None;
  computeSelectionRangeOverlapKinds(NodeTree, SelectionRange,
                                    Context.getSourceManager());

  SelectedStmtSet Result;
  for (const auto &N : llvm::enumerate(NodeTree)) {
    const auto *S = N.value().getStmtOrNull();
    if (!S)
      continue;
    switch (N.value().SelectionRangeOverlap) {
    case Node::ContainsSelectionRange: {
      Result.containsSelectionRange = S;
      Result.containsSelectionRangeIndex = N.index();

      const auto *CS = dyn_cast<CompoundStmt>(Result.containsSelectionRange);
      if (!CS) {
        // The entire if should be selected when just the 'else if' overlaps
        // with the selection range.
        if (const auto *If = dyn_cast<IfStmt>(Result.containsSelectionRange)) {
          auto IfConstruct = findIfStmtStart(If, N.index(), NodeTree);
          return SelectedStmtSet::createFromEntirelySelected(
              IfConstruct.first, IfConstruct.second);
        }
        // The entire switch should be selected when just a 'case' overlaps
        // with the selection range.
        if (const auto *Case =
                dyn_cast<SwitchCase>(Result.containsSelectionRange)) {
          auto Switch = findSwitchSourceConstruct(
              Case, makeArrayRef(NodeTree).drop_front(N.index() + 1));
          return SelectedStmtSet::createFromEntirelySelected(
              Switch.first, N.index() + Switch.second);
        }

        auto CanonicalExpr = canonicalizeSelectedExpr(S, N.index(), NodeTree);
        Result.containsSelectionRange = CanonicalExpr.first;
        Result.containsSelectionRangeIndex = CanonicalExpr.second;
        return Result;
      }

      bool IsLBraceSelected =
          !Context.getSourceManager().isBeforeInTranslationUnit(
              CS->getLBracLoc(), SelectionRange.getBegin());
      bool IsRBraceSelected =
          Context.getSourceManager().isBeforeInTranslationUnit(
              CS->getRBracLoc(), SelectionRange.getEnd());

      // Return the entire source construct that has the compound statement
      // when one of the braces is selected, or when an actual `case` of the
      // switch is selected.
      auto Construct = findCompoundStatementSourceConstruct(
          CS, makeArrayRef(NodeTree).drop_front(N.index() + 1));
      if (Construct.first != CS &&
          ((IsLBraceSelected || IsRBraceSelected) ||
           (isa<SwitchStmt>(Construct.first) &&
            isCaseSelected(cast<SwitchStmt>(Construct.first), SelectionRange,
                           Context.getSourceManager()))))
        return SelectedStmtSet::createFromEntirelySelected(
            Construct.first, N.index() + Construct.second);

      // When both braces are selected the entire compound statement is
      // considered to be selected.
      if (IsLBraceSelected && IsRBraceSelected)
        return Result;
      if (IsLBraceSelected)
        Result.containsSelectionRangeStart = CS->body_front();
      else if (IsRBraceSelected)
        Result.containsSelectionRangeEnd = CS->body_back();

      if (!Result.containsSelectionRangeStart)
        Result.containsSelectionRangeStart = findFirstStatementAfter(
            CS, SelectionRange.getBegin(), Context.getSourceManager());

      // Return an empty set when the compound statements os empty or the
      // selection range starts after the last statement or the selection range
      // doesn't overlap with any actual statements.
      if (!Result.containsSelectionRangeStart ||
          !Context.getSourceManager().isBeforeInTranslationUnit(
              Result.containsSelectionRangeStart->getBeginLoc(),
              SelectionRange.getEnd()))
        return None;

      if (!Result.containsSelectionRangeEnd)
        Result.containsSelectionRangeEnd = findLastStatementBefore(
            CS, SelectionRange.getEnd(), Result.containsSelectionRangeStart,
            Context.getSourceManager());

      return Result;
    }
    case Node::ContainsSelectionRangeStart:
      Result.containsSelectionRangeStart = S;
      break;
    case Node::ContainsSelectionRangeEnd:
      Result.containsSelectionRangeEnd = S;
      break;
    case Node::UnknownOverlap:
      break;
    }
  }
  return Result;
}

Optional<SelectedStmtSet> ASTSlice::getSelectedStmtSet() {
  if (CachedSelectedStmtSet)
    return *CachedSelectedStmtSet;
  CachedSelectedStmtSet = computeSelectedStmtSet();
  return *CachedSelectedStmtSet;
}

bool ASTSlice::isContainedInCompoundStmt(unsigned Index) {
  assert(Index < NodeTree.size() && "Invalid node index");
  for (unsigned I = Index + 1, E = NodeTree.size(); I != E; ++I) {
    const Stmt *S = NodeTree[I].getStmtOrNull();
    if (!S)
      continue;
    if (isa<CompoundStmt>(S))
      return true;
  }
  return false;
}

const Decl *ASTSlice::parentDeclForIndex(unsigned Index) {
  return NodeTree[Index].ParentDecl;
}

const Stmt *ASTSlice::parentStmtForIndex(unsigned Index) {
  return NodeTree[Index].ParentStmt;
}
