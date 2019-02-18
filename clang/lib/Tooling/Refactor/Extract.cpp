//===--- Extract.cpp -  ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the "extract" refactoring operation.
//
//===----------------------------------------------------------------------===//

#include "ExtractionUtils.h"
#include "RefactoringOperations.h"
#include "SourceLocationUtilities.h"
#include "StmtUtils.h"
#include "TypeUtils.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactor/RefactoringOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Path.h"
#include <algorithm>

using namespace clang;
using namespace clang::tooling;

namespace {

struct CompoundStatementRange {
  CompoundStmt::const_body_iterator First, Last;

  const Stmt *getFirst() const {
    // We must have selected just the child of the case, since a selection that
    // includes the case is treated like a selection of the entire switch.
    if (const auto *Case = dyn_cast<SwitchCase>(*First)) {
      if (const Stmt *S = Case->getSubStmt())
        return S;
    }
    return *First;
  }

  const Stmt *getLast() const { return *Last; }

  // TODO: We might not want to iterate over the switch case if we've just
  // selected its child. We should switch over to an array of nodes instead of
  // an iterator pair instead.
  CompoundStmt::const_body_iterator begin() const { return First; }
  CompoundStmt::const_body_iterator end() const { return Last + 1; }
};

enum class ExtractionKind { Function, Method, Expression };

class ExtractOperation : public RefactoringOperation {
public:
  struct CandidateInfo {
    CandidateInfo(SourceRange Range, StringRef PreInsertedText = "",
                  const Stmt *AnalyzedStatement = nullptr)
        : Range(Range), PreInsertedText(PreInsertedText),
          AnalyzedStatement(AnalyzedStatement) {}

    /// The candidate token range, i.e. the end location is the starting
    /// location of the last token.
    SourceRange Range;
    /// The text that should be inserted before the call to the extracted
    /// function.
    StringRef PreInsertedText;
    /// The expression that should be analyzed for captured variables and the
    /// return value.
    const Stmt *AnalyzedStatement;
  };

  ExtractOperation(const Stmt *S, const Stmt *ParentStmt,
                   const Decl *FunctionLikeParentDecl,
                   std::vector<std::string> Candidates,
                   Optional<CompoundStatementRange> ExtractedStmtRange,
                   Optional<CandidateInfo> FirstCandidateInfo,
                   ExtractionKind Kind)
      : S(S), ParentStmt(ParentStmt),
        FunctionLikeParentDecl(FunctionLikeParentDecl),
        Candidates(std::move(Candidates)),
        ExtractedStmtRange(ExtractedStmtRange), Kind(Kind) {
    if (FirstCandidateInfo)
      CandidateExtractionInfo.push_back(*FirstCandidateInfo);
  }

  const Stmt *getTransformedStmt() const override {
    if (ExtractedStmtRange)
      return ExtractedStmtRange->getFirst();
    return S;
  }

  const Stmt *getLastTransformedStmt() const override {
    if (ExtractedStmtRange)
      return ExtractedStmtRange->getLast();
    return nullptr;
  }

  std::vector<std::string> getRefactoringCandidates() override {
    return Candidates;
  }

  std::vector<RefactoringActionType> getAvailableSubActions() override {
    std::vector<RefactoringActionType> SubActions;
    if (isa<CXXMethodDecl>(FunctionLikeParentDecl) ||
        isa<ObjCMethodDecl>(FunctionLikeParentDecl))
      SubActions.push_back(RefactoringActionType::Extract_Method);
    if (isLexicalExpression(S, ParentStmt))
      SubActions.push_back(RefactoringActionType::Extract_Expression);
    return SubActions;
  }

  bool isMethodExtraction() const { return Kind == ExtractionKind::Method; }

  bool isExpressionExtraction() const {
    return Kind == ExtractionKind::Expression;
  }

  llvm::Expected<RefactoringResult> perform(ASTContext &Context, const Preprocessor &ThePreprocessor,
          const RefactoringOptionSet &Options,
          unsigned SelectedCandidateIndex) override;

  llvm::Expected<RefactoringResult>
  performExpressionExtraction(ASTContext &Context, PrintingPolicy &PP);

  const Stmt *S, *ParentStmt;
  const Decl *FunctionLikeParentDecl;
  std::vector<std::string> Candidates;
  /// A set of extraction candidates that correspond to the extracted code.
  SmallVector<CandidateInfo, 2> CandidateExtractionInfo;
  Optional<CompoundStatementRange> ExtractedStmtRange;
  ExtractionKind Kind;
};

} // end anonymous namespace

bool isSimpleExpression(const Expr *E) {
  switch (E->IgnoreParenCasts()->getStmtClass()) {
  case Stmt::DeclRefExprClass:
  case Stmt::PredefinedExprClass:
  case Stmt::IntegerLiteralClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::ImaginaryLiteralClass:
  case Stmt::CharacterLiteralClass:
  case Stmt::StringLiteralClass:
    return true;
  default:
    return false;
  }
}

static bool isMultipleCandidateBinOp(BinaryOperatorKind Op) {
  return Op == BO_Add || Op == BO_Sub;
}

/// Searches for the selected statement in the given CompoundStatement, looking
/// through things like PseudoObjectExpressions.
static CompoundStmt::const_body_iterator
findSelectedStmt(CompoundStmt::body_const_range Statements,
                 const Stmt *Target) {
  return llvm::find_if(Statements, [=](const Stmt *S) {
    if (S == Target)
      return true;
    if (const auto *POE = dyn_cast<PseudoObjectExpr>(S)) {
      if (POE->getSyntacticForm() == Target)
        return true;
    }
    return false;
  });
}

/// Returns the first and the last statements that should be extracted from a
/// compound statement.
Optional<CompoundStatementRange> getExtractedStatements(const CompoundStmt *CS,
                                                        const Stmt *Begin,
                                                        const Stmt *End) {
  if (CS->body_empty())
    return None;
  assert(Begin && End);
  CompoundStatementRange Result;
  Result.First = findSelectedStmt(CS->body(), Begin);
  if (Result.First == CS->body_end())
    return None;
  Result.Last = findSelectedStmt(
      CompoundStmt::body_const_range(Result.First, CS->body_end()), End);
  if (Result.Last == CS->body_end())
    return None;
  return Result;
}

static RefactoringOperationResult
initiateAnyExtractOperation(ASTSlice &Slice, ASTContext &Context,
                            SourceLocation Location, SourceRange SelectionRange,
                            bool CreateOperation,
                            ExtractionKind Kind = ExtractionKind::Function) {
  auto SelectedStmtsOpt = Slice.getSelectedStmtSet();
  if (!SelectedStmtsOpt)
    return None;
  SelectedStmtSet Stmts = *SelectedStmtsOpt;
  // The selection range is contained entirely within this statement (without
  // taking leading/trailing comments and whitespace into account).
  const Stmt *Selected = Stmts.containsSelectionRange;

  // We only want to perform the extraction if the selection range is entirely
  // within a body of a function or method.
  if (!Selected)
    return None;
  const Decl *ParentDecl =
      Slice.parentDeclForIndex(*Stmts.containsSelectionRangeIndex);

  if (!ParentDecl ||
      (!Stmts.isCompoundStatementPartiallySelected() &&
       !Slice.isContainedInCompoundStmt(*Stmts.containsSelectionRangeIndex)))
    return RefactoringOperationResult(
        "the selected expression is not in a function");

  if (isa<Expr>(Selected) && isSimpleExpression(cast<Expr>(Selected)))
    return RefactoringOperationResult("the selected expression is too simple");
  if (const auto *PRE = dyn_cast<ObjCPropertyRefExpr>(Selected)) {
    if (!PRE->isMessagingGetter())
      return RefactoringOperationResult("property setter can't be extracted");
  }

  const Stmt *ParentStmt =
      Slice.parentStmtForIndex(*Stmts.containsSelectionRangeIndex);
  if (Kind == ExtractionKind::Expression &&
      !isLexicalExpression(Selected, ParentStmt))
    return None;

  RefactoringOperationResult Result;
  Result.Initiated = true;
  if (!CreateOperation)
    return Result;

  Optional<CompoundStatementRange> ExtractedStmtRange;

  // Check if there are multiple candidates that can be extracted.
  std::vector<std::string> Candidates;
  Optional<ExtractOperation::CandidateInfo> FirstCandidateInfo;
  if (const auto *BinOp = dyn_cast<BinaryOperator>(Selected)) {
    // Binary '+' and '-' operators allow multiple candidates when the
    // selection range starts after the LHS expression but still overlaps
    // with the RHS.
    if (isMultipleCandidateBinOp(BinOp->getOpcode()) &&
        (!Stmts.containsSelectionRangeStart ||
         getPreciseTokenLocEnd(
             BinOp->getLHS()->getEndLoc(), Context.getSourceManager(),
             Context.getLangOpts()) == SelectionRange.getBegin()) &&
        Stmts.containsSelectionRangeEnd) {
      SourceRange FirstCandidateRange =
          SourceRange(SelectionRange.getBegin(), BinOp->getEndLoc());
      if (FirstCandidateRange.getEnd().isMacroID())
        FirstCandidateRange.setEnd(Context.getSourceManager().getExpansionLoc(
            FirstCandidateRange.getEnd()));
      FirstCandidateInfo = ExtractOperation::CandidateInfo(
          FirstCandidateRange, "+ ",
          /*AnalyzedStatement=*/BinOp->getRHS());
      Candidates.push_back(
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(FirstCandidateRange),
              Context.getSourceManager(), Context.getLangOpts())
              .trim());
      Candidates.push_back(Lexer::getSourceText(
          CharSourceRange::getTokenRange(BinOp->getSourceRange()),
          Context.getSourceManager(), Context.getLangOpts()));
    }
  } else if (const auto *CS = dyn_cast<CompoundStmt>(Selected)) {
    // We want to extract some child statements from a compound statement unless
    // we've selected the entire compound statement including the opening and
    // closing brace.
    if (Stmts.containsSelectionRangeStart)
      ExtractedStmtRange =
          getExtractedStatements(CS, Stmts.containsSelectionRangeStart,
                                 Stmts.containsSelectionRangeEnd);
  }

  auto Operation = llvm::make_unique<ExtractOperation>(
      Selected, ParentStmt, ParentDecl, std::move(Candidates),
      ExtractedStmtRange, FirstCandidateInfo, Kind);
  auto &CandidateExtractionInfo = Operation->CandidateExtractionInfo;
  SourceRange Range;
  if (ExtractedStmtRange)
    Range = SourceRange(ExtractedStmtRange->getFirst()->getBeginLoc(),
                        ExtractedStmtRange->getLast()->getEndLoc());
  else
    Range = Selected->getSourceRange();
  bool IsBeginMacroArgument = false;
  if (Range.getBegin().isMacroID()) {
    if (Context.getSourceManager().isMacroArgExpansion(Range.getBegin())) {
      Range.setBegin(
          Context.getSourceManager().getSpellingLoc(Range.getBegin()));
      IsBeginMacroArgument = true;
    } else {
      Range.setBegin(
          Context.getSourceManager().getExpansionLoc(Range.getBegin()));
    }
  }
  if (Range.getEnd().isMacroID()) {
    if (IsBeginMacroArgument &&
        Context.getSourceManager().isMacroArgExpansion(Range.getEnd()))
      Range.setEnd(Context.getSourceManager().getSpellingLoc(Range.getEnd()));
    else
      Range.setEnd(Context.getSourceManager()
                       .getExpansionRange(Range.getEnd())
                       .getEnd());
  }
  CandidateExtractionInfo.push_back(ExtractOperation::CandidateInfo(Range));
  Result.RefactoringOp = std::move(Operation);
  return Result;
}

RefactoringOperationResult clang::tooling::initiateExtractOperation(
    ASTSlice &Slice, ASTContext &Context, SourceLocation Location,
    SourceRange SelectionRange, bool CreateOperation) {
  return initiateAnyExtractOperation(Slice, Context, Location, SelectionRange,
                                     CreateOperation);
}

RefactoringOperationResult clang::tooling::initiateExtractMethodOperation(
    ASTSlice &Slice, ASTContext &Context, SourceLocation Location,
    SourceRange SelectionRange, bool CreateOperation) {
  // TODO: Verify that method extraction is actually possible.
  return initiateAnyExtractOperation(Slice, Context, Location, SelectionRange,
                                     CreateOperation, ExtractionKind::Method);
}

RefactoringOperationResult clang::tooling::initiateExtractExpressionOperation(
    ASTSlice &Slice, ASTContext &Context, SourceLocation Location,
    SourceRange SelectionRange, bool CreateOperation) {
  RefactoringOperationResult R =
      initiateAnyExtractOperation(Slice, Context, Location, SelectionRange,
                                  CreateOperation, ExtractionKind::Expression);
  return R;
}

using ReferencedEntity =
    llvm::PointerUnion<const DeclRefExpr *, const FieldDecl *>;

/// Iterate over the entities (variables/instance variables) that are directly
/// referenced by the given expression \p E.
///
/// Note: Objective-C ivars are always captured via 'self'.
static void findEntitiesDirectlyReferencedInExpr(
    const Expr *E,
    llvm::function_ref<void(const ReferencedEntity &Entity)> Handler) {
  E = E->IgnoreParenCasts();
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    return Handler(DRE);

  if (const auto *ME = dyn_cast<MemberExpr>(E)) {
    if (isa<CXXThisExpr>(ME->getBase()->IgnoreParenCasts())) {
      if (const auto *FD = dyn_cast_or_null<FieldDecl>(ME->getMemberDecl()))
        Handler(FD);
      return;
    }
    if (const auto *MD = ME->getMemberDecl()) {
      if (isa<FieldDecl>(MD) || isa<IndirectFieldDecl>(MD))
        findEntitiesDirectlyReferencedInExpr(ME->getBase(), Handler);
    }
    return;
  }

  if (const auto *CO = dyn_cast<ConditionalOperator>(E)) {
    findEntitiesDirectlyReferencedInExpr(CO->getTrueExpr(), Handler);
    findEntitiesDirectlyReferencedInExpr(CO->getFalseExpr(), Handler);
    return;
  }

  if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
    if (BO->getOpcode() == BO_Comma)
      return findEntitiesDirectlyReferencedInExpr(BO->getRHS(), Handler);
  }
}

template <typename T, typename Matcher>
static void
findMatchingParameters(Matcher &ParameterMatcher, const Stmt *S,
                       ASTContext &Context, StringRef Node,
                       llvm::function_ref<void(const T *E)> Handler) {
  using namespace clang::ast_matchers;
  auto Matches = match(findAll(callExpr(ParameterMatcher)), *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.template getNodeAs<T>(Node));
  Matches = match(findAll(cxxConstructExpr(ParameterMatcher)), *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.template getNodeAs<T>(Node));
}

static void
findUseOfConstThis(const Stmt *S, ASTContext &Context,
                   llvm::function_ref<void(const CXXThisExpr *E)> Handler) {
  using namespace clang::ast_matchers;
  // Check the receiver in method call and member operator calls.
  auto This = cxxThisExpr().bind("this");
  auto ThisReceiver = ignoringParenCasts(
      anyOf(This, unaryOperator(hasOperatorName("*"),
                                hasUnaryOperand(ignoringParenCasts(This)))));
  auto ConstMethodCallee = callee(cxxMethodDecl(isConst()));
  auto Matches = match(
      findAll(expr(anyOf(cxxMemberCallExpr(ConstMethodCallee, on(ThisReceiver)),
                         cxxOperatorCallExpr(ConstMethodCallee,
                                             hasArgument(0, ThisReceiver))))),
      *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.getNodeAs<CXXThisExpr>("this"));
  // Check parameters in calls.
  auto ConstPointee = pointee(qualType(isConstQualified()));
  auto RefParameter = forEachArgumentWithParam(
      ThisReceiver,
      parmVarDecl(hasType(qualType(referenceType(ConstPointee)))));
  findMatchingParameters(RefParameter, S, Context, "this", Handler);
  auto PtrParameter = forEachArgumentWithParam(
      ignoringParenCasts(This),
      parmVarDecl(hasType(qualType(pointerType(ConstPointee)))));
  findMatchingParameters(PtrParameter, S, Context, "this", Handler);
}

static void findArgumentsPassedByNonConstReference(
    const Stmt *S, ASTContext &Context,
    llvm::function_ref<void(const Expr *E)> Handler) {
  using namespace clang::ast_matchers;
  // Check the receiver in method call and member operator calls.
  auto NonPointerReceiver =
      expr(unless(hasType(qualType(pointerType())))).bind("arg");
  auto NonConstMethodCallee = callee(cxxMethodDecl(unless(isConst())));
  auto Matches =
      match(findAll(expr(anyOf(
                cxxMemberCallExpr(NonConstMethodCallee, on(NonPointerReceiver)),
                cxxOperatorCallExpr(NonConstMethodCallee,
                                    hasArgument(0, NonPointerReceiver))))),
            *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.getNodeAs<Expr>("arg"));
  // Check parameters in calls.
  auto RefParameter = forEachArgumentWithParam(
      expr().bind("arg"), parmVarDecl(hasType(qualType(referenceType(unless(
                              pointee(qualType(isConstQualified()))))))));
  Matches = match(findAll(callExpr(RefParameter)), *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.getNodeAs<Expr>("arg"));
  Matches = match(findAll(cxxConstructExpr(RefParameter)), *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.getNodeAs<Expr>("arg"));
}

static void findAddressExpressionsPassedByConstPointer(
    const Stmt *S, ASTContext &Context,
    llvm::function_ref<void(const UnaryOperator *E)> Handler) {
  using namespace clang::ast_matchers;
  auto ConstPtrParameter = forEachArgumentWithParam(
      ignoringParenImpCasts(unaryOperator(hasOperatorName("&")).bind("arg")),
      parmVarDecl(hasType(
          qualType(pointerType(pointee(qualType(isConstQualified())))))));
  auto Matches = match(findAll(callExpr(ConstPtrParameter)), *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.getNodeAs<UnaryOperator>("arg"));
  Matches = match(findAll(cxxConstructExpr(ConstPtrParameter)), *S, Context);
  for (const auto &Match : Matches)
    Handler(Match.getNodeAs<UnaryOperator>("arg"));
}

static bool isImplicitInitializer(const VarDecl *VD) {
  assert(VD->hasInit());
  const auto *E = VD->getInit();
  if (isa<ExprWithCleanups>(E))
    return false;
  const auto *Construct = dyn_cast<CXXConstructExpr>(E);
  if (!Construct)
    return E->getBeginLoc() == VD->getLocation();
  return Construct->getParenOrBraceRange().isInvalid();
}

static const Expr *getInitializerExprWithLexicalRange(const Expr *E) {
  if (const auto *EWC = dyn_cast<ExprWithCleanups>(E)) {
    if (const auto *Construct = dyn_cast<CXXConstructExpr>(EWC->getSubExpr())) {
      if (Construct->getNumArgs() == 1) {
        if (const auto *ME =
                dyn_cast<MaterializeTemporaryExpr>(Construct->getArg(0)))
          return ME;
      }
    }
  }
  return E;
}

namespace {

class ExtractedCodeVisitor : public RecursiveASTVisitor<ExtractedCodeVisitor> {
  int DefineOrdering = 0;

public:
  struct CaptureInfo {
    bool IsMutated = false;
    bool IsDefined = false;
    bool IsAddressTaken = false;
    bool IsConstAddressTaken = false;
    bool IsFieldCapturedWithThis = false;
    bool IsUsed = false;
    int DefineOrderingPriority = 0;

    bool isPassedByRefOrPtr() const {
      return IsMutated || IsAddressTaken || IsConstAddressTaken;
    }
    bool isRefOrPtrConst() const {
      return IsConstAddressTaken && !IsMutated && !IsAddressTaken;
    }
  };

  const ImplicitParamDecl *SelfDecl;

  ExtractedCodeVisitor(const ImplicitParamDecl *SelfDecl)
      : SelfDecl(SelfDecl) {}

  bool HasReturnInExtracted = false;

  CaptureInfo &captureVariable(const VarDecl *VD) {
    CaptureInfo &Result = CapturedVariables[VD];
    Result.IsUsed = true;
    return Result;
  }

  CaptureInfo &captureField(const FieldDecl *FD) { return CapturedFields[FD]; }

  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());
    if (!VD)
      return true;
    if (VD == SelfDecl) {
      CaptureSelf = true;
      SelfType = VD->getType();
      return true;
    }
    if (!VD->isLocalVarDeclOrParm())
      return true;
    captureVariable(VD);
    return true;
  }

  void captureThisWithoutConstConcerns(const CXXThisExpr *E) {
    CaptureThis = true;
    ThisRecordType = E->getType()->getPointeeType();
  }

  bool VisitCXXThisExpr(const CXXThisExpr *E) {
    captureThisWithoutConstConcerns(E);
    ThisUsesWithUnknownConstness.insert(E);
    return true;
  }

  bool TraverseMemberExpr(MemberExpr *E) {
    const auto *Base = dyn_cast<CXXThisExpr>(E->getBase()->IgnoreParenCasts());
    if (!Base)
      return RecursiveASTVisitor::TraverseMemberExpr(E);
    const FieldDecl *FD = dyn_cast_or_null<FieldDecl>(E->getMemberDecl());
    if (!FD)
      return RecursiveASTVisitor::TraverseMemberExpr(E);
    CaptureInfo &Info = captureField(FD);
    // Don't capture the implicit 'this' for private fields as we don't want to
    // capture this if we only use the private fields.
    if (FD->getAccess() == AS_public || !Base->isImplicit()) {
      Info.IsFieldCapturedWithThis = true;
      // The member might have an effect on the constness of the captured 'this'
      // but this is checked via mutation/const tracking for the field itself,
      // so we just capture 'this' without worrying about checking if it's used
      // in a 'const' manner here.
      captureThisWithoutConstConcerns(Base);
    }
    return true;
  }

  void captureSuper(QualType T) {
    if (CaptureSuper)
      return;
    SuperType = T;
    CaptureSuper = true;
  }

  bool TraverseObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
    if (E->isSuperReceiver())
      captureSuper(E->getSuperReceiverType());
    // Base might be an opaque expression, so we have to visit it manually as
    // we don't necessarily visit the setter/getter message sends if just the
    // property was selected.
    if (E->isObjectReceiver()) {
      if (const auto *OVE = dyn_cast<OpaqueValueExpr>(E->getBase()))
        TraverseStmt(OVE->getSourceExpr());
    }
    return RecursiveASTVisitor::TraverseObjCPropertyRefExpr(E);
  }

  bool TraverseBinAssign(BinaryOperator *S) {
    // RHS might be an opaque expression, if this is a property assignment. We
    // have to visit it manually as we don't necessarily visit the setter/getter
    // message sends if just the property was selected.
    if (const auto *OVE = dyn_cast<OpaqueValueExpr>(S->getRHS()))
      TraverseStmt(OVE->getSourceExpr());
    return RecursiveASTVisitor::TraverseBinAssign(S);
  }

  void findCapturedVariableOrFieldsInExpression(
      const Expr *E, llvm::function_ref<void(CaptureInfo &)> Handler) {
    findEntitiesDirectlyReferencedInExpr(
        E, [&Handler, this](const ReferencedEntity &Entity) {
          if (const auto *DRE = Entity.dyn_cast<const DeclRefExpr *>()) {
            const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
            if (!VD || !VD->isLocalVarDeclOrParm() || VD->isImplicit())
              return;
            return Handler(captureVariable(VD));
          }
          return Handler(captureField(Entity.get<const FieldDecl *>()));
        });
  }

  void
  markDirectlyReferencedVariableOrFieldInExpressionAsMutated(const Expr *E) {
    findCapturedVariableOrFieldsInExpression(
        E, [](CaptureInfo &Capture) { Capture.IsMutated = true; });
  }

  bool VisitBinaryOperator(const BinaryOperator *E) {
    if (E->isAssignmentOp())
      markDirectlyReferencedVariableOrFieldInExpressionAsMutated(E->getLHS());
    return true;
  }

  bool VisitUnaryPreInc(const UnaryOperator *E) {
    markDirectlyReferencedVariableOrFieldInExpressionAsMutated(E->getSubExpr());
    return true;
  }

  bool VisitUnaryPostInc(const UnaryOperator *E) {
    markDirectlyReferencedVariableOrFieldInExpressionAsMutated(E->getSubExpr());
    return true;
  }

  bool VisitUnaryPreDec(const UnaryOperator *E) {
    markDirectlyReferencedVariableOrFieldInExpressionAsMutated(E->getSubExpr());
    return true;
  }

  bool VisitUnaryPostDec(const UnaryOperator *E) {
    markDirectlyReferencedVariableOrFieldInExpressionAsMutated(E->getSubExpr());
    return true;
  }

  /// If the given expression refers to a local/instance variable or a
  /// a member of such variable that variable is marked as captured by
  /// reference.
  void captureVariableOrFieldInExpressionByReference(const Expr *E) {
    findCapturedVariableOrFieldsInExpression(
        E, [](CaptureInfo &Capture) { Capture.IsAddressTaken = true; });
  }

  bool VisitUnaryAddrOf(const UnaryOperator *E) {
    // Capture the entity with 'const' reference/pointer when its address is
    // passed into a function that takes a 'const' pointer and no other
    // mutations or non-const address/reference acquisitions occur.
    if (AddressExpressionsPassedToConstPointerParameter.count(E))
      findCapturedVariableOrFieldsInExpression(
          E->getSubExpr(),
          [](CaptureInfo &Capture) { Capture.IsConstAddressTaken = true; });
    else
      captureVariableOrFieldInExpressionByReference(E->getSubExpr());
    return true;
  }

  bool VisitObjCMessageExpr(const ObjCMessageExpr *E) {
    if (E->getSuperLoc().isValid())
      captureSuper(E->getSuperType());
    const ObjCMethodDecl *MD = E->getMethodDecl();
    if (!MD)
      return true;
    for (const auto &Param : llvm::enumerate(MD->parameters())) {
      QualType T = Param.value()->getType();
      if (Param.index() >= E->getNumArgs())
        break;
      if (T->isReferenceType() && !T->getPointeeType().isConstQualified())
        captureVariableOrFieldInExpressionByReference(E->getArg(Param.index()));
      if (T->isPointerType() && T->getPointeeType().isConstQualified()) {
        // Check if this is an '&' passed into a const pointer parameter.
        const Expr *Arg = E->getArg(Param.index());
        if (const auto *Op =
                dyn_cast<UnaryOperator>(Arg->IgnoreParenImpCasts())) {
          if (Op->getOpcode() == UO_AddrOf)
            AddressExpressionsPassedToConstPointerParameter.insert(Op);
        }
      }
    }
    return true;
  }

  bool VisitVarDecl(const VarDecl *VD) {
    // Don't capture using the captureVariable method as we don't want to mark
    // the declaration as a 'use'. This allows us to avoid passing in variables
    // that are defined in extracted code, used afterwards, but never actually
    // used in the extracted code.
    CaptureInfo &Capture = CapturedVariables[VD];
    Capture.IsDefined = true;
    Capture.DefineOrderingPriority = ++DefineOrdering;
    // Ensure the capture is marked as 'used' when the variable declaration has
    // an explicit initialization expression. This allows us to pass it by
    // reference when it's defined in extracted code, used afterwards, but never
    // actually used in the extracted code. The main reason why we want to try
    // to keep this initialization in the extracted code is to preserve
    // semantics as the initialization expression might have side-effects.
    if (!Capture.IsUsed && VD->hasInit() && !isImplicitInitializer(VD))
      Capture.IsUsed = true;
    QualType T = VD->getType();
    if (T->isReferenceType() && !T->getPointeeType().isConstQualified() &&
        VD->hasInit())
      captureVariableOrFieldInExpressionByReference(VD->getInit());
    return true;
  }

  bool VisitReturnStmt(const ReturnStmt *S) {
    HasReturnInExtracted = true;
    return true;
  }

  void InspectExtractedStmt(Stmt *S, ASTContext &Context) {
    findAddressExpressionsPassedByConstPointer(
        S, Context, [this](const UnaryOperator *Arg) {
          AddressExpressionsPassedToConstPointerParameter.insert(Arg);
        });
    TraverseStmt(S);
    findArgumentsPassedByNonConstReference(S, Context, [this](const Expr *Arg) {
      captureVariableOrFieldInExpressionByReference(Arg);
    });
    if (CaptureThis && !ThisUsesWithUnknownConstness.empty()) {
      // Compare the definite 'const' uses of 'this' to all the seen uses
      // (except for the known field uses).
      findUseOfConstThis(S, Context, [this](const CXXThisExpr *Arg) {
        ThisUsesWithUnknownConstness.erase(Arg);
      });
      IsThisConstForNonCapturedFieldUses = ThisUsesWithUnknownConstness.empty();
    }
  }

  llvm::DenseMap<const VarDecl *, CaptureInfo> CapturedVariables;
  llvm::DenseMap<const FieldDecl *, CaptureInfo> CapturedFields;
  llvm::SmallPtrSet<const UnaryOperator *, 8>
      AddressExpressionsPassedToConstPointerParameter;
  llvm::SmallPtrSet<const CXXThisExpr *, 16> ThisUsesWithUnknownConstness;
  bool CaptureThis = false;
  bool IsThisConstForNonCapturedFieldUses = true;
  QualType ThisRecordType;
  bool CaptureSelf = false, CaptureSuper = false;
  QualType SelfType, SuperType;
};

/// Traverses the extracted code and finds the uses of captured variables
/// that are passed into the extracted function using a pointer.
class VariableDefinedInExtractedCodeUseAfterExtractionFinder
    : public RecursiveASTVisitor<
          VariableDefinedInExtractedCodeUseAfterExtractionFinder> {
  bool IsAfterExtracted = false;

public:
  const Stmt *LastExtractedStmt;
  const llvm::SmallPtrSetImpl<const VarDecl *> &VariablesDefinedInExtractedCode;
  llvm::SmallPtrSet<const VarDecl *, 4> VariablesUsedAfterExtraction;

  VariableDefinedInExtractedCodeUseAfterExtractionFinder(
      const Stmt *LastExtractedStmt,
      const llvm::SmallPtrSetImpl<const VarDecl *>
          &VariablesDefinedInExtractedCode)
      : LastExtractedStmt(LastExtractedStmt),
        VariablesDefinedInExtractedCode(VariablesDefinedInExtractedCode) {}

  bool TraverseStmt(Stmt *S) {
    RecursiveASTVisitor::TraverseStmt(S);
    if (S == LastExtractedStmt)
      IsAfterExtracted = true;
    return true;
  }

  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    if (!IsAfterExtracted)
      return true;
    const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());
    if (!VD)
      return true;
    if (VariablesDefinedInExtractedCode.count(VD))
      VariablesUsedAfterExtraction.insert(VD);
    return true;
  }
};

class PossibleShadowingVariableFinder
    : public RecursiveASTVisitor<PossibleShadowingVariableFinder> {
  const VarDecl *TargetVD;

  PossibleShadowingVariableFinder(const VarDecl *TargetVD)
      : TargetVD(TargetVD) {}

public:
  bool VisitVarDecl(const VarDecl *VD) {
    if (VD == TargetVD || VD->getName() != TargetVD->getName())
      return true;
    return false;
  }

  /// Returns true if the given statement \p S has a variable declaration whose
  /// name is identical to the given variable declaration \p VD.
  static bool hasShadowingVar(const VarDecl *VD, const Stmt *S) {
    return !PossibleShadowingVariableFinder(VD).TraverseStmt(
        const_cast<Stmt *>(S));
  }
};

/// Traverses the extracted code and rewrites the 'return' statements to ensure
/// that they now return some value.
class ReturnRewriter : public RecursiveASTVisitor<ReturnRewriter> {
  Rewriter &SourceRewriter;
  std::string Text;

public:
  ReturnRewriter(Rewriter &SourceRewriter, StringRef Text)
      : SourceRewriter(SourceRewriter), Text(std::string(" ") + Text.str()) {}

  bool VisitReturnStmt(const ReturnStmt *S) {
    SourceRewriter.InsertText(
        getPreciseTokenLocEnd(S->getEndLoc(), SourceRewriter.getSourceMgr(),
                              SourceRewriter.getLangOpts()),
        Text);
    return true;
  }
};

/// Prints the given initializer expression using the original source code if
/// possible.
static void printInitializerExpressionUsingOriginalSyntax(
    const VarDecl *VD, const Expr *E, bool IsDeclaration, const ASTContext &Ctx,
    llvm::raw_ostream &OS, const PrintingPolicy &PP) {
  E = getInitializerExprWithLexicalRange(E);
  SourceRange Range = E->getSourceRange();
  bool UseEquals = true;
  bool UseTypeName = false;
  if (const auto *Construct = dyn_cast<CXXConstructExpr>(E)) {
    SourceRange SubRange = Construct->getParenOrBraceRange();
    if (SubRange.isValid()) {
      UseEquals = false;
      UseTypeName = true;
      Range = SubRange;
    }
  }
  if (Range.getBegin().isMacroID())
    Range.setBegin(Ctx.getSourceManager().getExpansionLoc(Range.getBegin()));
  if (Range.getEnd().isMacroID())
    Range.setEnd(Ctx.getSourceManager().getExpansionLoc(Range.getEnd()));
  bool IsInvalid = false;
  StringRef Text = Lexer::getSourceText(CharSourceRange::getTokenRange(Range),
                                        Ctx.getSourceManager(),
                                        Ctx.getLangOpts(), &IsInvalid);
  if (IsDeclaration && UseEquals)
    OS << " = ";
  else if (!IsDeclaration && UseTypeName)
    VD->getType().print(OS, PP);
  if (IsInvalid)
    E->printPretty(OS, nullptr, PP);
  else
    OS << Text;
};

/// Traverses the extracted code and rewrites the declaration statements that
/// declare variables that are used after the extracted code.
class DefinedInExtractedCodeDeclStmtRewriter
    : public RecursiveASTVisitor<DefinedInExtractedCodeDeclStmtRewriter> {
public:
  Rewriter &SourceRewriter;
  const llvm::SmallPtrSetImpl<const VarDecl *> &VariablesUsedAfterExtraction;
  const PrintingPolicy &PP;

  DefinedInExtractedCodeDeclStmtRewriter(
      Rewriter &SourceRewriter, const llvm::SmallPtrSetImpl<const VarDecl *>
                                    &VariablesUsedAfterExtraction,
      const PrintingPolicy &PP)
      : SourceRewriter(SourceRewriter),
        VariablesUsedAfterExtraction(VariablesUsedAfterExtraction), PP(PP) {}

  /// When a declaration statement declares variables that are all used
  /// after extraction, we can rewrite it completely into a set of assignments
  /// while still preserving the original initializer expressions when we
  /// can.
  void rewriteAllVariableDeclarationsToAssignments(const DeclStmt *S) {
    SourceLocation StartLoc = S->getBeginLoc();
    for (const Decl *D : S->decls()) {
      const auto *VD = dyn_cast<VarDecl>(D);
      if (!VD || !VariablesUsedAfterExtraction.count(VD))
        continue;
      if (!VD->hasInit() || isImplicitInitializer(VD)) {
        // Remove the variable declarations without explicit initializers.
        // This can affect the semantics of the program if the implicit
        // initialization expression has side effects.
        SourceRange Range = SourceRange(
            StartLoc, S->isSingleDecl() ? S->getEndLoc() : VD->getLocation());
        SourceRewriter.RemoveText(Range);
        continue;
      }
      std::string Str;
      llvm::raw_string_ostream OS(Str);
      if (StartLoc != S->getBeginLoc())
        OS << "; ";
      const ASTContext &Ctx = D->getASTContext();
      // Dereference the variable unless the source uses C++.
      if (!Ctx.getLangOpts().CPlusPlus)
        OS << '*';
      OS << VD->getName() << " = ";
      const Expr *Init = getInitializerExprWithLexicalRange(VD->getInit());
      SourceLocation End = Init->getBeginLoc();
      if (const auto *Construct = dyn_cast<CXXConstructExpr>(Init)) {
        SourceRange SubRange = Construct->getParenOrBraceRange();
        if (SubRange.isValid()) {
          End = SubRange.getBegin();
          VD->getType().print(OS, PP);
        }
      }
      if (End.isMacroID())
        End = Ctx.getSourceManager().getExpansionLoc(End);
      auto Range = CharSourceRange::getCharRange(StartLoc, End);
      SourceRewriter.ReplaceText(StartLoc, SourceRewriter.getRangeSize(Range),
                                 OS.str());
      StartLoc = getPreciseTokenLocEnd(D->getEndLoc(), Ctx.getSourceManager(),
                                       Ctx.getLangOpts());
    }
  }

  /// When a declaration statement has variables that are both used after
  /// extraction and not used after extraction, we create new declaration
  /// statements that declare the unused variables, while creating assignment
  /// statements that "initialize" the variables that are used after the
  /// extraction. This way we can preserve the order of
  /// initialization/assignment from the original declaration statement.
  void rewriteMixedDeclarations(const DeclStmt *S) {
    // Completely rewrite the declaration statement.
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    for (const Decl *D : S->decls()) {
      const ASTContext &Ctx = D->getASTContext();
      const VarDecl *VD = dyn_cast<VarDecl>(D);
      bool IsLast = D == S->decl_end()[-1];
      if (!VD) {
        OS << "<<unsupported declaration>>;";
        continue;
      }

      auto PrintInit = [&](bool IsDeclaration) {
        printInitializerExpressionUsingOriginalSyntax(
            VD, VD->getInit(), IsDeclaration, Ctx, OS, PP);
      };
      if (!VariablesUsedAfterExtraction.count(VD)) {
        VD->getType().print(OS, PP);
        OS << " " << VD->getName();
        if (VD->hasInit() && !isImplicitInitializer(VD))
          PrintInit(/*IsDeclaration=*/true);
        OS << ";";
        if (!IsLast)
          OS << ' ';
        continue;
      }
      if (VD->hasInit() && !isImplicitInitializer(VD)) {
        // Dereference the variable unless the source uses C++.
        if (!Ctx.getLangOpts().CPlusPlus)
          OS << '*';
        OS << VD->getName() << " = ";
        PrintInit(/*IsDeclaration=*/false);
        OS << ";";
        if (!IsLast)
          OS << ' ';
      }
    }
    SourceRewriter.ReplaceText(S->getSourceRange(), OS.str());
  }

  bool VisitDeclStmt(const DeclStmt *S) {
    bool AreAllUsed = true;
    bool AreNoneUsed = true;
    for (const Decl *D : S->decls()) {
      const auto *VD = dyn_cast<VarDecl>(D);
      if (!VD || !VariablesUsedAfterExtraction.count(VD)) {
        AreAllUsed = false;
        continue;
      }
      AreNoneUsed = false;
      // Exit early when both flags were set in the loop.
      if (!AreAllUsed)
        break;
    }
    if (AreNoneUsed)
      return true;

    if (AreAllUsed)
      rewriteAllVariableDeclarationsToAssignments(S);
    else
      rewriteMixedDeclarations(S);
    return true;
  }
};

/// Takes care of pseudo object expressions and Objective-C properties to avoid
/// duplicate rewrites and missing rewrites.
template <typename T>
class PseudoObjectRewriter : public RecursiveASTVisitor<T> {
  typedef RecursiveASTVisitor<T> Base;

public:
  bool TraversePseudoObjectExpr(PseudoObjectExpr *E) {
    return Base::TraverseStmt(E->getSyntacticForm());
  }

  bool TraverseObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
    // Base might be an opaque expression, so we have to visit it manually as
    // we don't necessarily visit the setter/getter message sends if just the
    // property was selected.
    if (E->isObjectReceiver()) {
      if (const auto *OVE = dyn_cast<OpaqueValueExpr>(E->getBase()))
        Base::TraverseStmt(OVE->getSourceExpr());
    }
    return Base::TraverseObjCPropertyRefExpr(E);
  }

  bool TraverseBinAssign(BinaryOperator *S) {
    // RHS might be an opaque expression, if this is a property assignment. We
    // have to visit it manually as we don't necessarily visit the setter/getter
    // message sends if just the property was selected.
    if (const auto *OVE = dyn_cast<OpaqueValueExpr>(S->getRHS()))
      Base::TraverseStmt(OVE->getSourceExpr());
    return Base::TraverseBinAssign(S);
  }
};

/// Traverses the extracted code and rewrites the uses of captured variables
/// that are passed into the extracted function using a pointer.
class CapturedVariableCaptureByPointerRewriter
    : public PseudoObjectRewriter<CapturedVariableCaptureByPointerRewriter> {
public:
  const VarDecl *TargetVD;
  Rewriter &SourceRewriter;

  CapturedVariableCaptureByPointerRewriter(const VarDecl *VD,
                                           Rewriter &SourceRewriter)
      : TargetVD(VD), SourceRewriter(SourceRewriter) {}

  bool isTargetDeclRefExpr(const Expr *E) {
    const auto *DRE = dyn_cast<DeclRefExpr>(E);
    if (!DRE)
      return false;
    return dyn_cast<VarDecl>(DRE->getDecl()) == TargetVD;
  }

  void dereferenceTargetVar(const Expr *E, bool WrapInParens = false) {
    SourceRewriter.InsertTextBefore(E->getBeginLoc(),
                                    WrapInParens ? "(*" : "*");
    if (WrapInParens)
      SourceRewriter.InsertTextAfterToken(E->getEndLoc(), ")");
  }

  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());
    if (VD != TargetVD)
      return true;
    dereferenceTargetVar(E);
    return true;
  }

  bool TraverseUnaryAddrOf(UnaryOperator *E) {
    if (const auto *DRE =
            dyn_cast<DeclRefExpr>(E->getSubExpr()->IgnoreParenCasts())) {
      const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
      if (VD == TargetVD) {
        // Remove the '&' as the variable is now a pointer.
        SourceRewriter.RemoveText(
            CharSourceRange::getTokenRange(E->getBeginLoc(), E->getBeginLoc()));
        return true;
      }
    }
    return RecursiveASTVisitor::TraverseUnaryAddrOf(E);
  }

  bool TraverseMemberExpr(MemberExpr *E) {
    if (!E->isArrow()) {
      if (const auto *DRE =
              dyn_cast<DeclRefExpr>(E->getBase()->IgnoreParenCasts())) {
        const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
        if (VD == TargetVD) {
          // Replace '.' with '->'.
          SourceRewriter.ReplaceText(E->getOperatorLoc(), 1, "->");
          return true;
        }
      }
    } else if (isTargetDeclRefExpr(E->getBase()->IgnoreImpCasts())) {
      // Ensure the variable is wrapped in parenthesis when it's the base of
      // '->' operator.
      dereferenceTargetVar(E->getBase(), /*WrapInParens=*/true);
      return true;
    }
    return RecursiveASTVisitor::TraverseMemberExpr(E);
  }
};

/// Traverses the extracted code and rewrites the uses of 'this' that can be
/// rewritten as references.
class CapturedThisReferenceRewriter
    : public PseudoObjectRewriter<CapturedThisReferenceRewriter> {
public:
  Rewriter &SourceRewriter;
  llvm::SmallPtrSet<const CXXThisExpr *, 8> RewrittenExpressions;

  CapturedThisReferenceRewriter(Rewriter &SourceRewriter)
      : SourceRewriter(SourceRewriter) {}

  void rewriteThis(const CXXThisExpr *E) {
    RewrittenExpressions.insert(E);
    if (!E->isImplicit())
      SourceRewriter.ReplaceText(E->getBeginLoc(), 4, "object");
    else
      SourceRewriter.InsertText(E->getBeginLoc(), "object");
  }

  bool VisitMemberExpr(const MemberExpr *E) {
    const auto *This =
        dyn_cast<CXXThisExpr>(E->getBase()->IgnoreParenImpCasts());
    if (This) {
      rewriteThis(This);
      if (!This->isImplicit() && E->isArrow())
        SourceRewriter.ReplaceText(E->getOperatorLoc(), 2, ".");
      else
        SourceRewriter.InsertText(E->getBase()->getEndLoc(), ".");
    }
    return true;
  }
};

/// Traverses the extracted code and rewrites the uses of 'this' into '&object'.
class CapturedThisPointerRewriter
    : public PseudoObjectRewriter<CapturedThisPointerRewriter> {
public:
  Rewriter &SourceRewriter;
  const llvm::SmallPtrSetImpl<const CXXThisExpr *> &RewrittenExpressions;

  CapturedThisPointerRewriter(
      Rewriter &SourceRewriter,
      const llvm::SmallPtrSetImpl<const CXXThisExpr *> &RewrittenExpressions)
      : SourceRewriter(SourceRewriter),
        RewrittenExpressions(RewrittenExpressions) {}

  void replace(const CXXThisExpr *E, StringRef Text) {
    SourceRewriter.ReplaceText(E->getBeginLoc(), 4, Text);
  }

  bool VisitCXXThisExpr(const CXXThisExpr *E) {
    if (RewrittenExpressions.count(E))
      return true;
    if (!E->isImplicit())
      replace(E, "&object");
    return true;
  }

  bool TraverseUnaryDeref(UnaryOperator *E) {
    if (const auto *This =
            dyn_cast<CXXThisExpr>(E->getSubExpr()->IgnoreParenImpCasts())) {
      if (!This->isImplicit()) {
        // Remove the '*' as the variable is now a reference.
        SourceRewriter.RemoveText(
            CharSourceRange::getTokenRange(E->getBeginLoc(), E->getBeginLoc()));
        replace(This, "object");
        return true;
      }
    }
    return RecursiveASTVisitor::TraverseUnaryAddrOf(E);
  }
};

/// Traverses the extracted code and rewrites the uses of 'self' into 'object'.
class CapturedSelfRewriter : public PseudoObjectRewriter<CapturedSelfRewriter> {
public:
  Rewriter &SourceRewriter;
  const ImplicitParamDecl *SelfDecl;

  CapturedSelfRewriter(Rewriter &SourceRewriter,
                       const ImplicitParamDecl *SelfDecl)
      : SourceRewriter(SourceRewriter), SelfDecl(SelfDecl) {
    assert(SelfDecl);
  }

  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());
    if (!VD || VD != SelfDecl)
      return true;
    if (E->getBeginLoc().isInvalid())
      return true;
    SourceRewriter.ReplaceText(E->getBeginLoc(), 4, "object");
    return true;
  }

  void insertObjectForImplicitSelf(const Expr *E, SourceLocation Loc,
                                   StringRef Text) {
    const auto *DRE = dyn_cast<DeclRefExpr>(E);
    if (!DRE)
      return;
    const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (!VD || VD != SelfDecl || DRE->getBeginLoc().isValid())
      return;
    SourceRewriter.InsertText(Loc, Text);
  }

  bool VisitObjCIvarRefExpr(const ObjCIvarRefExpr *E) {
    insertObjectForImplicitSelf(E->getBase()->IgnoreImpCasts(),
                                E->getBeginLoc(), "object->");
    return true;
  }
};

/// Traverses the extracted code and rewrites the uses of 'self' into the name
/// of the class.
class CapturedClassSelfRewriter
    : public PseudoObjectRewriter<CapturedClassSelfRewriter> {
public:
  Rewriter &SourceRewriter;
  StringRef ClassName;
  const ImplicitParamDecl *SelfDecl;

  CapturedClassSelfRewriter(Rewriter &SourceRewriter, StringRef ClassName,
                            const ImplicitParamDecl *SelfDecl)
      : SourceRewriter(SourceRewriter), ClassName(ClassName),
        SelfDecl(SelfDecl) {

    assert(SelfDecl);
  }

  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());
    if (!VD || VD != SelfDecl || E->getBeginLoc().isInvalid())
      return true;
    SourceRewriter.ReplaceText(E->getBeginLoc(), 4, ClassName);
    return true;
  }
};

/// Traverses the extracted code and rewrites the uses of 'super' into
/// 'superObject' or the name of the super class.
class CapturedSuperRewriter
    : public PseudoObjectRewriter<CapturedSuperRewriter> {
public:
  Rewriter &SourceRewriter;
  StringRef ReplacementString;

  CapturedSuperRewriter(Rewriter &SourceRewriter, StringRef ReplacementString)
      : SourceRewriter(SourceRewriter), ReplacementString(ReplacementString) {}

  void rewriteSuper(SourceLocation Loc) {
    SourceRewriter.ReplaceText(Loc, strlen("super"), ReplacementString);
  }

  bool VisitObjCPropertyRefExpr(const ObjCPropertyRefExpr *E) {
    if (E->isSuperReceiver())
      rewriteSuper(E->getReceiverLocation());
    return true;
  }

  bool VisitObjCMessageExpr(const ObjCMessageExpr *E) {
    if (E->getSuperLoc().isValid())
      rewriteSuper(E->getSuperLoc());
    return true;
  }
};

struct ExtractionSemicolonPolicy {
  bool IsNeededInExtractedFunction;
  bool IsNeededInOriginalFunction;

  static ExtractionSemicolonPolicy neededInExtractedFunction() {
    return {true, false};
  }
  static ExtractionSemicolonPolicy neededInOriginalFunction() {
    return {false, true};
  }
  static ExtractionSemicolonPolicy neededInBoth() { return {true, true}; }
};

} // end anonymous namespace

ExtractionSemicolonPolicy
computeSemicolonExtractionPolicy(const Stmt *S, SourceRange &ExtractedRange,
                                 const SourceManager &SM,
                                 const LangOptions &LangOpts) {
  if (isa<Expr>(S))
    return ExtractionSemicolonPolicy::neededInExtractedFunction();
  bool NeedsSemi = isSemicolonRequiredAfter(S);
  if (!NeedsSemi)
    return ExtractionSemicolonPolicy::neededInOriginalFunction();
  SourceLocation End = ExtractedRange.getEnd();
  if (isSemicolonAtLocation(End, SM, LangOpts))
    return ExtractionSemicolonPolicy::neededInOriginalFunction();
  SourceLocation NextTokenLoc =
      Lexer::findNextTokenLocationAfterTokenAt(End, SM, LangOpts);
  if (NextTokenLoc.isValid() &&
      isSemicolonAtLocation(NextTokenLoc, SM, LangOpts) &&
      areOnSameLine(NextTokenLoc, End, SM)) {
    ExtractedRange.setEnd(NextTokenLoc);
    return ExtractionSemicolonPolicy::neededInOriginalFunction();
  }
  return ExtractionSemicolonPolicy::neededInBoth();
}

PrintingPolicy getPrintingPolicy(const ASTContext &Context,
                                 const Preprocessor &PP) {
  PrintingPolicy Policy = Context.getPrintingPolicy();
  // Our printing policy is copied over the ASTContext printing policy whenever
  // a diagnostic is emitted, so recompute it.
  Policy.Bool = Context.getLangOpts().Bool;
  // FIXME: This is duplicated with Sema.cpp. When upstreaming this should be
  // cleaned up.
  if (!Policy.Bool) {
    if (const MacroInfo *BoolMacro = PP.getMacroInfo(Context.getBoolName())) {
      Policy.Bool = BoolMacro->isObjectLike() &&
                    BoolMacro->getNumTokens() == 1 &&
                    BoolMacro->getReplacementToken(0).is(tok::kw__Bool);
    }
  }
  return Policy;
}

static QualType getFunctionLikeParentDeclReturnType(const Decl *D) {
  // FIXME: might need to handle ObjC blocks in the future.
  if (const auto *M = dyn_cast<ObjCMethodDecl>(D))
    return M->getReturnType();
  return cast<FunctionDecl>(D)->getReturnType();
}

static const Stmt *getEnclosingDeclBody(const Decl *D) {
  // FIXME: might need to handle ObjC blocks in the future.
  if (const auto *M = dyn_cast<ObjCMethodDecl>(D))
    return M->getBody();
  return cast<FunctionDecl>(D)->getBody();
}

static bool isEnclosingMethodConst(const Decl *D) {
  if (const auto *MD = dyn_cast<CXXMethodDecl>(D))
    return MD->isConst();
  return false;
}

static bool isEnclosingMethodStatic(const Decl *D) {
  if (const auto *MD = dyn_cast<CXXMethodDecl>(D))
    return MD->isStatic();
  return false;
}

static bool isEnclosingMethodOutOfLine(const Decl *D) {
  const auto *MD = dyn_cast<CXXMethodDecl>(D);
  if (!MD)
    return false;
  return MD->isOutOfLine();
}

static void printEnclosingMethodScope(const Decl *D, llvm::raw_ostream &OS,
                                      const PrintingPolicy &PP) {
  const auto *MD = dyn_cast<CXXMethodDecl>(D);
  if (!MD)
    return;
  if (!MD->isOutOfLine() || !MD->getQualifier())
    return;
  MD->getQualifier()->print(OS, PP);
}

static SourceLocation
computeFunctionExtractionLocation(const Decl *D, bool IsMethodExtraction) {
  if (!IsMethodExtraction && isa<CXXMethodDecl>(D)) {
    // Code from methods that defined in class bodies should be extracted to a
    // function defined just before the class.
    while (const auto *RD = dyn_cast<CXXRecordDecl>(D->getLexicalDeclContext()))
      D = RD;
  }
  return D->getBeginLoc();
}

namespace {
enum class MethodDeclarationPlacement { After, Before };

/// \brief Represents an entity captured from the original function that's
/// passed into the new function/method.
struct CapturedVariable {
  const VarDecl *VD;
  const FieldDecl *FD;
  QualType ThisType;
  bool PassByRefOrPtr;
  bool IsRefOrPtrConst;
  bool IsThisSelf = false;
  bool IsThisSuper = false;
  bool TakeAddress = false;
  QualType ParameterType;

  CapturedVariable(const VarDecl *VD, bool PassByRefOrPtr, bool IsRefOrPtrConst)
      : VD(VD), FD(nullptr), PassByRefOrPtr(PassByRefOrPtr),
        IsRefOrPtrConst(IsRefOrPtrConst) {}
  CapturedVariable(const FieldDecl *FD, bool PassByRefOrPtr,
                   bool IsRefOrPtrConst)
      : VD(nullptr), FD(FD), PassByRefOrPtr(PassByRefOrPtr),
        IsRefOrPtrConst(IsRefOrPtrConst) {}
  CapturedVariable(QualType ThisType, bool PassByRefOrPtr, bool IsConst)
      : VD(nullptr), FD(nullptr), ThisType(ThisType),
        PassByRefOrPtr(PassByRefOrPtr), IsRefOrPtrConst(IsConst) {}

  static CapturedVariable getThis(QualType T, bool IsConst) {
    return CapturedVariable(T, /*PassByRefOrPtr=*/true, /*IsConst*/ IsConst);
  }

  static CapturedVariable getSelf(QualType T) {
    auto Result =
        CapturedVariable(T, /*PassByRefOrPtr=*/false, /*IsConst*/ false);
    Result.IsThisSelf = true;
    return Result;
  }

  static CapturedVariable getSuper(QualType T) {
    auto Result =
        CapturedVariable(T, /*PassByRefOrPtr=*/false, /*IsConst*/ false);
    Result.IsThisSuper = true;
    return Result;
  }

  StringRef getName() const {
    return VD ? VD->getName()
              : FD ? FD->getName() : IsThisSuper ? "superObject" : "object";
  }
  StringRef getExpr() const {
    return ThisType.isNull()
               ? getName()
               : IsThisSelf ? "self" : IsThisSuper ? "super.self" : "*this";
  }
  QualType getType() const {
    return VD ? VD->getType() : FD ? FD->getType() : ThisType;
  }
};
} // end anonymous namespace

static std::pair<SourceLocation, MethodDeclarationPlacement>
computeAppropriateExtractionLocationForMethodDeclaration(
    const CXXMethodDecl *D) {
  const CXXRecordDecl *RD = D->getParent();
  // Try to put the new declaration after the last method, or just before the
  // end of the class.
  SourceLocation Loc;
  for (const CXXMethodDecl *M : RD->methods()) {
    if (M->isImplicit())
      continue;
    Loc = M->getEndLoc();
  }
  return Loc.isValid() ? std::make_pair(Loc, MethodDeclarationPlacement::After)
                       : std::make_pair(RD->getEndLoc(),
                                        MethodDeclarationPlacement::Before);
}

static bool isInHeader(SourceLocation Loc, const SourceManager &SM) {
  // Base the header decision on the filename.
  StringRef Extension = llvm::sys::path::extension(SM.getFilename(Loc));
  if (Extension.empty())
    return false;
  return llvm::StringSwitch<bool>(Extension.drop_front())
      .Case("h", true)
      .Case("hpp", true)
      .Case("hh", true)
      .Case("h++", true)
      .Case("hxx", true)
      .Case("inl", true)
      .Case("def", true)
      .Default(false);
}

llvm::Expected<RefactoringResult>
ExtractOperation::performExpressionExtraction(ASTContext &Context,
                                              PrintingPolicy &PP) {
  assert(isExpressionExtraction() && "Not an expression extraction");
  std::vector<RefactoringReplacement> Replacements;
  const Expr *E = cast<Expr>(S);
  QualType VarType = findExpressionLexicalType(FunctionLikeParentDecl, E,
                                               E->getType(), PP, Context);
  StringRef VarName = "extractedExpr";
  auto CreatedSymbol = llvm::make_unique<RefactoringResultAssociatedSymbol>(
      OldSymbolName(VarName));

  SourceRange ExtractedTokenRange = CandidateExtractionInfo[0].Range;
  SourceRange ExtractedCharRange = SourceRange(
      ExtractedTokenRange.getBegin(),
      getPreciseTokenLocEnd(ExtractedTokenRange.getEnd(),
                            Context.getSourceManager(), Context.getLangOpts()));

  // Create the variable that will hold the value of the duplicate expression.
  std::string VariableDeclarationString;
  llvm::raw_string_ostream OS(VariableDeclarationString);
  VarType.print(OS, PP, /*PlaceHolder*/ VarName);
  // FIXME: We should hook into the TypePrinter when moving over to llvm.org
  // instead and get the offset from it.
  unsigned NameOffset = StringRef(OS.str()).find(VarName);
  OS << " = ";
  OS << Lexer::getSourceText(CharSourceRange::getCharRange(ExtractedCharRange),
                             Context.getSourceManager(), Context.getLangOpts());
  OS << ";\n";

  // Variable declaration.
  SourceLocation InsertionLoc =
      extract::locationForExtractedVariableDeclaration(
          E, FunctionLikeParentDecl, Context.getSourceManager());
  Replacements.push_back(RefactoringReplacement(
      SourceRange(InsertionLoc, InsertionLoc), OS.str(), CreatedSymbol.get(),
      RefactoringReplacement::AssociatedSymbolLocation(
          llvm::makeArrayRef(NameOffset), /*IsDeclaration=*/true)));
  // Replace the expression with the variable.
  Replacements.push_back(
      RefactoringReplacement(ExtractedCharRange, VarName, CreatedSymbol.get(),
                             /*NameOffset=*/llvm::makeArrayRef(unsigned(0))));

  RefactoringResult Result(std::move(Replacements));
  Result.AssociatedSymbols.push_back(std::move(CreatedSymbol));
  return std::move(Result);
}

llvm::Expected<RefactoringResult> ExtractOperation::perform(
    ASTContext &Context, const Preprocessor &ThePreprocessor,
    const RefactoringOptionSet &Options, unsigned SelectedCandidateIndex) {
  std::vector<RefactoringReplacement> Replacements;
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();
  Rewriter SourceRewriter(SM, LangOpts);
  PrintingPolicy PP = getPrintingPolicy(Context, ThePreprocessor);
  PP.UseStdFunctionForLambda = true;
  PP.SuppressStrongLifetime = true;
  PP.SuppressLifetimeQualifiers = true;
  PP.SuppressUnwrittenScope = true;

  if (isExpressionExtraction())
    return performExpressionExtraction(Context, PP);

  const Stmt *S =
      CandidateExtractionInfo[SelectedCandidateIndex].AnalyzedStatement
          ? CandidateExtractionInfo[SelectedCandidateIndex].AnalyzedStatement
          : this->S;

  const auto *EnclosingObjCMethod =
      dyn_cast<ObjCMethodDecl>(FunctionLikeParentDecl);

  // Find the variables that are captured by the extracted code.
  ExtractedCodeVisitor Visitor(/*SelfDecl=*/EnclosingObjCMethod
                                   ? EnclosingObjCMethod->getSelfDecl()
                                   : nullptr);
  if (ExtractedStmtRange) {
    for (const Stmt *S : *ExtractedStmtRange)
      Visitor.InspectExtractedStmt(const_cast<Stmt *>(S), Context);
  } else
    Visitor.InspectExtractedStmt(const_cast<Stmt *>(S), Context);
  // Compute the return type.
  bool IsExpr = isLexicalExpression(S, ParentStmt);
  QualType ReturnType;
  if (IsExpr || Visitor.HasReturnInExtracted) {
    if (const auto *E = dyn_cast<Expr>(S)) {
      assert(!ExtractedStmtRange);
      ReturnType = findExpressionLexicalType(FunctionLikeParentDecl, E,
                                             E->getType(), PP, Context);
    } else
      ReturnType = getFunctionLikeParentDeclReturnType(FunctionLikeParentDecl);
  } else
    ReturnType = Context.VoidTy;
  // Sort the captured variables.
  std::vector<CapturedVariable> CapturedVariables;
  llvm::SmallPtrSet<const VarDecl *, 4> VariablesDefinedInExtractedCode;
  CapturedVariables.reserve(Visitor.CapturedVariables.size() +
                            Visitor.CapturedFields.size());
  for (const auto &I : Visitor.CapturedVariables) {
    if (I.getSecond().IsDefined) {
      VariablesDefinedInExtractedCode.insert(I.getFirst());
      continue;
    }
    CapturedVariables.push_back(
        CapturedVariable(I.getFirst(), I.getSecond().isPassedByRefOrPtr(),
                         I.getSecond().isRefOrPtrConst()));
  }
  // Take a look at the variables that are defined in the extracted code.
  VariableDefinedInExtractedCodeUseAfterExtractionFinder
      UsedAfterExtractionFinder(ExtractedStmtRange ? *ExtractedStmtRange->Last
                                                   : S,
                                VariablesDefinedInExtractedCode);
  UsedAfterExtractionFinder.TraverseStmt(
      const_cast<Stmt *>(getEnclosingDeclBody(FunctionLikeParentDecl)));
  struct RedeclaredVariable {
    const VarDecl *VD;
    int OrderingPriority;
  };
  llvm::SmallVector<RedeclaredVariable, 4> RedeclaredVariables;
  bool CanUseReturnForVariablesUsedAfterwards =
      !isa<Expr>(S) && ReturnType->isVoidType() &&
      UsedAfterExtractionFinder.VariablesUsedAfterExtraction.size() == 1;
  if (CanUseReturnForVariablesUsedAfterwards) {
    // Avoid using the return value for the variable that's used afterwards as
    // another variable might shadow it at the point of a 'return' that we
    // have to rewrite to 'return var'.
    const VarDecl *VD =
        *UsedAfterExtractionFinder.VariablesUsedAfterExtraction.begin();
    if (ExtractedStmtRange) {
      for (const Stmt *S : *ExtractedStmtRange) {
        if (PossibleShadowingVariableFinder::hasShadowingVar(VD, S)) {
          CanUseReturnForVariablesUsedAfterwards = false;
          break;
        }
      }
    } else
      CanUseReturnForVariablesUsedAfterwards =
          !PossibleShadowingVariableFinder::hasShadowingVar(VD, S);
  }
  if (CanUseReturnForVariablesUsedAfterwards) {
    for (const auto &I : Visitor.CapturedVariables) {
      if (!I.getSecond().IsDefined ||
          !UsedAfterExtractionFinder.VariablesUsedAfterExtraction.count(
              I.getFirst()))
        continue;
      RedeclaredVariables.push_back(
          {I.getFirst(), I.getSecond().DefineOrderingPriority});
      ReturnType = I.getFirst()->getType();
      // Const qualifier can be dropped as we don't want to declare the return
      // type as 'const'.
      if (ReturnType.isConstQualified())
        ReturnType.removeLocalConst();
      break;
    }
    if (Visitor.HasReturnInExtracted) {
      ReturnRewriter ReturnsRewriter(SourceRewriter,
                                     RedeclaredVariables.front().VD->getName());
      if (ExtractedStmtRange) {
        for (const Stmt *S : *ExtractedStmtRange)
          ReturnsRewriter.TraverseStmt(const_cast<Stmt *>(S));
      } else
        ReturnsRewriter.TraverseStmt(const_cast<Stmt *>(S));
    }
  } else {
    for (const auto &I : Visitor.CapturedVariables) {
      if (!I.getSecond().IsDefined ||
          !UsedAfterExtractionFinder.VariablesUsedAfterExtraction.count(
              I.getFirst()))
        continue;
      RedeclaredVariables.push_back(
          {I.getFirst(), I.getSecond().DefineOrderingPriority});
      if (!I.getSecond().IsUsed)
        continue;
      // Pass the variable that's defined in the extracted code but used
      // afterwards as a parameter only when it's actually used in the extracted
      // code.
      CapturedVariables.push_back(CapturedVariable(I.getFirst(),
                                                   /*PassByRefOrPtr=*/true,
                                                   /*IsRefOrPtrConst=*/false));
    }
    std::sort(RedeclaredVariables.begin(), RedeclaredVariables.end(),
              [](const RedeclaredVariable &X, const RedeclaredVariable &Y) {
                return X.OrderingPriority < Y.OrderingPriority;
              });
    DefinedInExtractedCodeDeclStmtRewriter DeclRewriter(
        SourceRewriter, UsedAfterExtractionFinder.VariablesUsedAfterExtraction,
        PP);
    if (ExtractedStmtRange) {
      for (const Stmt *S : *ExtractedStmtRange)
        DeclRewriter.TraverseStmt(const_cast<Stmt *>(S));
    } else
      DeclRewriter.TraverseStmt(const_cast<Stmt *>(S));
  }
  // Capture any fields if necessary.
  bool IsThisConstInCapturedFieldUses = true;
  if (!isMethodExtraction()) {
    for (const auto &I : Visitor.CapturedFields) {
      if (I.getSecond().isPassedByRefOrPtr() &&
          !I.getSecond().isRefOrPtrConst())
        IsThisConstInCapturedFieldUses = false;
      // Private fields that use explicit 'this' should be captured using 'this'
      // even if they might end up being inaccessible in the extracted function.
      if (I.getSecond().IsFieldCapturedWithThis)
        continue;
      CapturedVariables.push_back(
          CapturedVariable(I.getFirst(), I.getSecond().isPassedByRefOrPtr(),
                           I.getSecond().isRefOrPtrConst()));
    }
  }
  std::sort(CapturedVariables.begin(), CapturedVariables.end(),
            [](const CapturedVariable &X, const CapturedVariable &Y) {
              return X.getName() < Y.getName();
            });
  // 'This'/'self' should be passed-in first.
  if (!isMethodExtraction() && Visitor.CaptureThis) {
    CapturedVariables.insert(
        CapturedVariables.begin(),
        CapturedVariable::getThis(
            Visitor.ThisRecordType,
            IsThisConstInCapturedFieldUses &&
                Visitor.IsThisConstForNonCapturedFieldUses));
    CapturedThisReferenceRewriter ThisRewriter(SourceRewriter);
    if (ExtractedStmtRange) {
      for (const Stmt *S : *ExtractedStmtRange)
        ThisRewriter.TraverseStmt(const_cast<Stmt *>(S));
    } else
      ThisRewriter.TraverseStmt(const_cast<Stmt *>(S));
    CapturedThisPointerRewriter PtrThisRewriter(
        SourceRewriter, ThisRewriter.RewrittenExpressions);
    if (ExtractedStmtRange) {
      for (const Stmt *S : *ExtractedStmtRange)
        PtrThisRewriter.TraverseStmt(const_cast<Stmt *>(S));
    } else
      PtrThisRewriter.TraverseStmt(const_cast<Stmt *>(S));
  } else if (!isMethodExtraction() && Visitor.CaptureSelf &&
             EnclosingObjCMethod) {
    if (EnclosingObjCMethod->isInstanceMethod()) {
      // Instance methods rewrite 'self' into an 'object' parameter.
      CapturedVariables.insert(CapturedVariables.begin(),
                               CapturedVariable::getSelf(Visitor.SelfType));
      CapturedSelfRewriter SelfRewriter(SourceRewriter,
                                        EnclosingObjCMethod->getSelfDecl());
      if (ExtractedStmtRange) {
        for (const Stmt *S : *ExtractedStmtRange)
          SelfRewriter.TraverseStmt(const_cast<Stmt *>(S));
      } else
        SelfRewriter.TraverseStmt(const_cast<Stmt *>(S));
    } else {
      // Class methods rewrite 'self' into the class name and don't pass 'self'
      // as a parameter.
      CapturedClassSelfRewriter SelfRewriter(
          SourceRewriter, EnclosingObjCMethod->getClassInterface()->getName(),
          EnclosingObjCMethod->getSelfDecl());
      if (ExtractedStmtRange) {
        for (const Stmt *S : *ExtractedStmtRange)
          SelfRewriter.TraverseStmt(const_cast<Stmt *>(S));
      } else
        SelfRewriter.TraverseStmt(const_cast<Stmt *>(S));
    }
  }
  if (!isMethodExtraction() && Visitor.CaptureSuper && EnclosingObjCMethod) {
    if (EnclosingObjCMethod->isInstanceMethod())
      // Instance methods rewrite 'super' into an 'superObject' parameter.
      CapturedVariables.insert(Visitor.CaptureSelf
                                   ? CapturedVariables.begin() + 1
                                   : CapturedVariables.begin(),
                               CapturedVariable::getSuper(Visitor.SuperType));
    CapturedSuperRewriter SuperRewriter(
        SourceRewriter, EnclosingObjCMethod->isInstanceMethod()
                            ? "superObject"
                            : EnclosingObjCMethod->getClassInterface()
                                  ->getSuperClass()
                                  ->getName());
    if (ExtractedStmtRange) {
      for (const Stmt *S : *ExtractedStmtRange)
        SuperRewriter.TraverseStmt(const_cast<Stmt *>(S));
    } else
      SuperRewriter.TraverseStmt(const_cast<Stmt *>(S));
  }

  // Compute the parameter types.
  for (auto &Var : CapturedVariables) {
    QualType T = Var.getType();

    // Array types are passed into the extracted function using a pointer.
    if (const auto *AT = Context.getAsArrayType(T))
      T = Context.getPointerType(AT->getElementType());

    // Captured records and other mutated variables are passed into the
    // extracted function either using a reference (C++) or a pointer.
    if ((T->isRecordType() || Var.PassByRefOrPtr) && !T->isReferenceType()) {
      // Add a 'const' qualifier to the record when it's not mutated in the
      // extracted code or when we are taking the address of the captured
      // variable for just a 'const' use.
      if (!Var.PassByRefOrPtr || Var.IsRefOrPtrConst)
        T.addConst();

      if (LangOpts.CPlusPlus)
        T = Context.getLValueReferenceType(T);
      else {
        T = Context.getPointerType(T);
        CapturedVariableCaptureByPointerRewriter UseRewriter(Var.VD,
                                                             SourceRewriter);
        if (ExtractedStmtRange) {
          for (const Stmt *S : *ExtractedStmtRange)
            UseRewriter.TraverseStmt(const_cast<Stmt *>(S));
        } else
          UseRewriter.TraverseStmt(const_cast<Stmt *>(S));
        Var.TakeAddress = true;
      }
    }
    // Const qualifier can be dropped as we don't want to declare the parameter
    // as 'const'.
    else if (T.isLocalConstQualified())
      T.removeLocalConst();

    Var.ParameterType = T;
  }

  // TODO: Choose a better name if there are collisions.
  StringRef ExtractedName = "extracted";
  llvm::SmallVector<StringRef, 4> ExtractedNamePieces;
  ExtractedNamePieces.push_back(ExtractedName);
  if (isMethodExtraction() && EnclosingObjCMethod &&
      !CapturedVariables.empty()) {
    for (const auto &Var : llvm::makeArrayRef(CapturedVariables).drop_front())
      ExtractedNamePieces.push_back(Var.getName());
  }
  std::unique_ptr<RefactoringResultAssociatedSymbol> CreatedSymbol =
      llvm::make_unique<RefactoringResultAssociatedSymbol>(
          OldSymbolName(ExtractedNamePieces));

  SourceLocation FunctionExtractionLoc = computeFunctionExtractionLocation(
      FunctionLikeParentDecl, isMethodExtraction());
  FunctionExtractionLoc =
      getLocationOfPrecedingComment(FunctionExtractionLoc, SM, LangOpts);

  // Create the replacement that contains the new function.
  auto PrintFunctionHeader =
      [&](llvm::raw_string_ostream &OS,
          bool IsDefinition =
              true) -> RefactoringReplacement::AssociatedSymbolLocation {
    if (isMethodExtraction() && EnclosingObjCMethod) {
      OS << (EnclosingObjCMethod->isClassMethod() ? '+' : '-') << " (";
      ReturnType.print(OS, PP);
      OS << ')';
      llvm::SmallVector<unsigned, 4> NameOffsets;
      NameOffsets.push_back(OS.str().size());
      OS << ExtractedName;
      bool IsFirst = true;
      for (const auto &Var : CapturedVariables) {
        if (!IsFirst) {
          OS << ' ';
          NameOffsets.push_back(OS.str().size());
          OS << Var.getName();
        }
        IsFirst = false;
        OS << ":(";
        Var.ParameterType.print(OS, PP);
        OS << ')' << Var.getName();
      }
      return RefactoringReplacement::AssociatedSymbolLocation(
          NameOffsets, /*IsDeclaration=*/true);
    }
    auto *FD = dyn_cast<FunctionDecl>(FunctionLikeParentDecl);
    if (isMethodExtraction() && IsDefinition &&
        !FD->getDescribedFunctionTemplate()) {
      // Print the class template parameter lists for an out-of-line method.
      for (unsigned I = 0,
                    NumTemplateParams = FD->getNumTemplateParameterLists();
           I < NumTemplateParams; ++I) {
        FD->getTemplateParameterList(I)->print(OS, PP, Context);
        OS << "\n";
      }
    }
    if (isMethodExtraction() && isEnclosingMethodStatic(FunctionLikeParentDecl))
      OS << "static ";
    else if (!isMethodExtraction())
      OS << (isInHeader(FunctionExtractionLoc, SM) ? "inline " : "static ");
    std::string QualifiedName;
    llvm::raw_string_ostream NameOS(QualifiedName);
    if (isMethodExtraction() && IsDefinition)
      printEnclosingMethodScope(FunctionLikeParentDecl, NameOS, PP);
    NameOS << ExtractedName;
    NameOS << '(';
    bool IsFirst = true;
    for (const auto &Var : CapturedVariables) {
      if (!IsFirst)
        NameOS << ", ";
      IsFirst = false;
      Var.ParameterType.print(NameOS, PP, /*PlaceHolder=*/Var.getName());
    }
    NameOS << ')';
    ReturnType.print(OS, PP, NameOS.str());
    unsigned NameOffset = OS.str().find(ExtractedName);
    if (isMethodExtraction() && isEnclosingMethodConst(FunctionLikeParentDecl))
      OS << " const";
    return RefactoringReplacement::AssociatedSymbolLocation(
        NameOffset, /*IsDeclaration=*/true);
    ;
  };

  if (isMethodExtraction() &&
      isEnclosingMethodOutOfLine(FunctionLikeParentDecl)) {
    // The location of the declaration should be either before the original
    // declararation, or, if this method has not declaration, somewhere
    // appropriate in the class.
    MethodDeclarationPlacement Placement;
    SourceLocation DeclarationLoc;
    if (FunctionLikeParentDecl->getCanonicalDecl() != FunctionLikeParentDecl) {
      DeclarationLoc = computeFunctionExtractionLocation(
          FunctionLikeParentDecl->getCanonicalDecl(), isMethodExtraction());
      Placement = MethodDeclarationPlacement::Before;
    } else {
      auto LocAndPlacement =
          computeAppropriateExtractionLocationForMethodDeclaration(
              cast<CXXMethodDecl>(FunctionLikeParentDecl));
      DeclarationLoc = LocAndPlacement.first;
      Placement = LocAndPlacement.second;
    }
    if (Placement == MethodDeclarationPlacement::Before)
      DeclarationLoc =
          getLocationOfPrecedingComment(DeclarationLoc, SM, LangOpts);
    else
      DeclarationLoc = getLastLineLocationUnlessItHasOtherTokens(
          getPreciseTokenLocEnd(DeclarationLoc, SM, LangOpts), SM, LangOpts);
    // Add a replacement for the method declaration if necessary.
    std::string DeclarationString;
    llvm::raw_string_ostream OS(DeclarationString);
    if (Placement == MethodDeclarationPlacement::After)
      OS << "\n\n";
    RefactoringReplacement::AssociatedSymbolLocation SymbolLoc =
        PrintFunctionHeader(OS, /*IsDefinition=*/false);
    OS << ";\n";
    if (Placement == MethodDeclarationPlacement::Before)
      OS << "\n";
    Replacements.push_back(RefactoringReplacement(
        SourceRange(DeclarationLoc, DeclarationLoc), std::move(OS.str()),
        CreatedSymbol.get(), SymbolLoc));
  }
  std::string ExtractedCode;
  llvm::raw_string_ostream ExtractedOS(ExtractedCode);
  RefactoringReplacement::AssociatedSymbolLocation SymbolLoc =
      PrintFunctionHeader(ExtractedOS);
  ExtractedOS << " {\n";
  if (IsExpr && !ReturnType->isVoidType())
    ExtractedOS << "return ";
  SourceRange ExtractedTokenRange =
      CandidateExtractionInfo[SelectedCandidateIndex].Range;
  auto Semicolons = computeSemicolonExtractionPolicy(
      ExtractedStmtRange ? *(ExtractedStmtRange->Last) : S, ExtractedTokenRange,
      SM, LangOpts);
  bool ShouldCopyBlock = false;
  if (IsExpr && !LangOpts.ObjCAutoRefCount &&
      ReturnType->isBlockPointerType()) {
    // We can't return local blocks directly without ARC; they should be copied.
    // FIXME: This is overly pessimistic, as we only need the copy for local
    // blocks.
    ExtractedOS << "[(";
    ShouldCopyBlock = true;
  }
  ExtractedOS << SourceRewriter.getRewrittenText(ExtractedTokenRange);
  if (ShouldCopyBlock)
    ExtractedOS << ") copy]";
  if (Semicolons.IsNeededInExtractedFunction)
    ExtractedOS << ';';
  if (CanUseReturnForVariablesUsedAfterwards)
    ExtractedOS << "\nreturn " << RedeclaredVariables.front().VD->getName()
                << ";";
  ExtractedOS << "\n}\n\n";
  Replacements.push_back(RefactoringReplacement(
      SourceRange(FunctionExtractionLoc, FunctionExtractionLoc),
      std::move(ExtractedOS.str()), CreatedSymbol.get(), SymbolLoc));

  // Create a replacements that removes the extracted code in favor of the
  // function call.
  std::string InsertedCode;
  llvm::raw_string_ostream InsertedOS(InsertedCode);
  // We might have to declare variables that were declared in the extracted code
  // but still used afterwards.
  if (CanUseReturnForVariablesUsedAfterwards) {
    const auto &Var = RedeclaredVariables.front();
    Var.VD->getType().print(InsertedOS, PP);
    InsertedOS << ' ' << Var.VD->getName() << " = ";
  } else {
    for (const auto &Var : RedeclaredVariables) {
      Var.VD->getType().print(InsertedOS, PP);
      InsertedOS << ' ' << Var.VD->getName() << ";\n";
    }
  }
  InsertedOS << CandidateExtractionInfo[SelectedCandidateIndex].PreInsertedText;
  llvm::SmallVector<unsigned, 4> NameOffsets;
  if (isMethodExtraction() && EnclosingObjCMethod) {
    InsertedOS << "[self ";
    NameOffsets.push_back(InsertedOS.str().size());
    InsertedOS << ExtractedName;
    bool IsFirst = true;
    for (const auto &Var : CapturedVariables) {
      if (!IsFirst) {
        InsertedOS << ' ';
        NameOffsets.push_back(InsertedOS.str().size());
        InsertedOS << Var.getName();
      }
      IsFirst = false;
      InsertedOS << ':';
      if (Var.TakeAddress)
        InsertedOS << '&';
      InsertedOS << Var.getExpr();
    }
    InsertedOS << ']';
  } else {
    NameOffsets.push_back(InsertedOS.str().size());
    InsertedOS << ExtractedName << '(';
    bool IsFirst = true;
    for (const auto &Var : CapturedVariables) {
      if (!IsFirst)
        InsertedOS << ", ";
      IsFirst = false;
      if (Var.TakeAddress)
        InsertedOS << '&';
      InsertedOS << Var.getExpr();
    }
    InsertedOS << ')';
  }
  if (Semicolons.IsNeededInOriginalFunction)
    InsertedOS << ';';
  SourceRange ExtractedCharRange = SourceRange(
      ExtractedTokenRange.getBegin(),
      getPreciseTokenLocEnd(ExtractedTokenRange.getEnd(), SM, LangOpts));
  Replacements.push_back(RefactoringReplacement(
      ExtractedCharRange, std::move(InsertedOS.str()), CreatedSymbol.get(),
      llvm::makeArrayRef(NameOffsets)));

  RefactoringResult Result(std::move(Replacements));
  Result.AssociatedSymbols.push_back(std::move(CreatedSymbol));
  return std::move(Result);
}
