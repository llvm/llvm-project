//===--- ASTSelection.cpp - Clang refactoring library ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/ASTSelection.h"
#include "clang/AST/LexicallyOrderedRecursiveASTVisitor.h"
#include "clang/Lex/Lexer.h"

using namespace clang;
using namespace tooling;
using ast_type_traits::DynTypedNode;

namespace {

CharSourceRange getLexicalDeclRange(Decl *D, const SourceManager &SM,
                                    const LangOptions &LangOpts) {
  if (!isa<ObjCImplDecl>(D))
    return CharSourceRange::getTokenRange(D->getSourceRange());
  // Objective-C implementation declarations end at the '@' instead of the 'end'
  // keyword. Use the lexer to find the location right after 'end'.
  SourceRange R = D->getSourceRange();
  SourceLocation LocAfterEnd = Lexer::findLocationAfterToken(
      R.getEnd(), tok::raw_identifier, SM, LangOpts,
      /*SkipTrailingWhitespaceAndNewLine=*/false);
  return LocAfterEnd.isValid()
             ? CharSourceRange::getCharRange(R.getBegin(), LocAfterEnd)
             : CharSourceRange::getTokenRange(R);
}

/// Constructs the tree of selected AST nodes that either contain the location
/// of the cursor or overlap with the selection range.
class ASTSelectionFinder
    : public LexicallyOrderedRecursiveASTVisitor<ASTSelectionFinder> {
public:
  ASTSelectionFinder(SourceRange Selection, FileID TargetFile,
                     const ASTContext &Context)
      : LexicallyOrderedRecursiveASTVisitor(Context.getSourceManager()),
        SelectionBegin(Selection.getBegin()),
        SelectionEnd(Selection.getBegin() == Selection.getEnd()
                         ? SourceLocation()
                         : Selection.getEnd()),
        TargetFile(TargetFile), Context(Context) {
    // The TU decl is the root of the selected node tree.
    SelectionStack.push_back(
        SelectedASTNode(DynTypedNode::create(*Context.getTranslationUnitDecl()),
                        SourceSelectionKind::None));
  }

  Optional<SelectedASTNode> getSelectedASTNode() {
    assert(SelectionStack.size() == 1 && "stack was not popped");
    SelectedASTNode Result = std::move(SelectionStack.back());
    SelectionStack.pop_back();
    if (Result.Children.empty())
      return None;
    return std::move(Result);
  }

  bool TraverseDecl(Decl *D) {
    if (isa<TranslationUnitDecl>(D))
      return LexicallyOrderedRecursiveASTVisitor::TraverseDecl(D);
    if (D->isImplicit())
      return true;

    // Check if this declaration is written in the file of interest.
    const SourceRange DeclRange = D->getSourceRange();
    const SourceManager &SM = Context.getSourceManager();
    SourceLocation FileLoc;
    if (DeclRange.getBegin().isMacroID() && !DeclRange.getEnd().isMacroID())
      FileLoc = DeclRange.getEnd();
    else
      FileLoc = SM.getSpellingLoc(DeclRange.getBegin());
    if (SM.getFileID(FileLoc) != TargetFile)
      return true;

    SourceSelectionKind SelectionKind =
        selectionKindFor(getLexicalDeclRange(D, SM, Context.getLangOpts()));
    SelectionStack.push_back(
        SelectedASTNode(DynTypedNode::create(*D), SelectionKind));
    LexicallyOrderedRecursiveASTVisitor::TraverseDecl(D);
    popAndAddToSelectionIfSelected(SelectionKind);

    if (DeclRange.getEnd().isValid() &&
        SM.isBeforeInTranslationUnit(SelectionEnd.isValid() ? SelectionEnd
                                                            : SelectionBegin,
                                     DeclRange.getEnd())) {
      // Stop early when we've reached a declaration after the selection.
      return false;
    }
    return true;
  }

  bool TraverseStmt(Stmt *S) {
    if (!S)
      return true;
    // FIXME (Alex Lorenz): Improve handling for macro locations.
    SourceSelectionKind SelectionKind =
        selectionKindFor(CharSourceRange::getTokenRange(S->getSourceRange()));
    SelectionStack.push_back(
        SelectedASTNode(DynTypedNode::create(*S), SelectionKind));
    LexicallyOrderedRecursiveASTVisitor::TraverseStmt(S);
    popAndAddToSelectionIfSelected(SelectionKind);
    return true;
  }

private:
  void popAndAddToSelectionIfSelected(SourceSelectionKind SelectionKind) {
    SelectedASTNode Node = std::move(SelectionStack.back());
    SelectionStack.pop_back();
    if (SelectionKind != SourceSelectionKind::None || !Node.Children.empty())
      SelectionStack.back().Children.push_back(std::move(Node));
  }

  SourceSelectionKind selectionKindFor(CharSourceRange Range) {
    SourceLocation End = Range.getEnd();
    const SourceManager &SM = Context.getSourceManager();
    if (Range.isTokenRange())
      End = Lexer::getLocForEndOfToken(End, 0, SM, Context.getLangOpts());
    if (!SourceLocation::isPairOfFileLocations(Range.getBegin(), End))
      return SourceSelectionKind::None;
    if (!SelectionEnd.isValid()) {
      // Do a quick check when the selection is of length 0.
      if (SM.isPointWithin(SelectionBegin, Range.getBegin(), End))
        return SourceSelectionKind::ContainsSelection;
      return SourceSelectionKind::None;
    }
    bool HasStart = SM.isPointWithin(SelectionBegin, Range.getBegin(), End);
    bool HasEnd = SM.isPointWithin(SelectionEnd, Range.getBegin(), End);
    if (HasStart && HasEnd)
      return SourceSelectionKind::ContainsSelection;
    if (SM.isPointWithin(Range.getBegin(), SelectionBegin, SelectionEnd) &&
        SM.isPointWithin(End, SelectionBegin, SelectionEnd))
      return SourceSelectionKind::InsideSelection;
    // Ensure there's at least some overlap with the 'start'/'end' selection
    // types.
    if (HasStart && SelectionBegin != End)
      return SourceSelectionKind::ContainsSelectionStart;
    if (HasEnd && SelectionEnd != Range.getBegin())
      return SourceSelectionKind::ContainsSelectionEnd;

    return SourceSelectionKind::None;
  }

  const SourceLocation SelectionBegin, SelectionEnd;
  FileID TargetFile;
  const ASTContext &Context;
  std::vector<SelectedASTNode> SelectionStack;
};

} // end anonymous namespace

Optional<SelectedASTNode>
clang::tooling::findSelectedASTNodes(const ASTContext &Context,
                                     SourceRange SelectionRange) {
  assert(SelectionRange.isValid() &&
         SourceLocation::isPairOfFileLocations(SelectionRange.getBegin(),
                                               SelectionRange.getEnd()) &&
         "Expected a file range");
  FileID TargetFile =
      Context.getSourceManager().getFileID(SelectionRange.getBegin());
  assert(Context.getSourceManager().getFileID(SelectionRange.getEnd()) ==
             TargetFile &&
         "selection range must span one file");

  ASTSelectionFinder Visitor(SelectionRange, TargetFile, Context);
  Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  return Visitor.getSelectedASTNode();
}

static const char *selectionKindToString(SourceSelectionKind Kind) {
  switch (Kind) {
  case SourceSelectionKind::None:
    return "none";
  case SourceSelectionKind::ContainsSelection:
    return "contains-selection";
  case SourceSelectionKind::ContainsSelectionStart:
    return "contains-selection-start";
  case SourceSelectionKind::ContainsSelectionEnd:
    return "contains-selection-end";
  case SourceSelectionKind::InsideSelection:
    return "inside";
  }
  llvm_unreachable("invalid selection kind");
}

static void dump(const SelectedASTNode &Node, llvm::raw_ostream &OS,
                 unsigned Indent = 0) {
  OS.indent(Indent * 2);
  if (const Decl *D = Node.Node.get<Decl>()) {
    OS << D->getDeclKindName() << "Decl";
    if (const auto *ND = dyn_cast<NamedDecl>(D))
      OS << " \"" << ND->getNameAsString() << '"';
  } else if (const Stmt *S = Node.Node.get<Stmt>()) {
    OS << S->getStmtClassName();
  }
  OS << ' ' << selectionKindToString(Node.SelectionKind) << "\n";
  for (const auto &Child : Node.Children)
    dump(Child, OS, Indent + 1);
}

void SelectedASTNode::dump(llvm::raw_ostream &OS) const { ::dump(*this, OS); }
