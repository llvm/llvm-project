//===- CppBoundedBuffers.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformations/CppBoundedBuffers.h"
#include "../../Analyses/SSAFAnalysesCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowPairs.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationRegistry.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::ssaf;

static constexpr llvm::StringLiteral SkippedRuleId =
    "cpp-bounded-buffers-skipped";

namespace {

/// A declarator whose type can carry pointer levels.
bool isCandidateType(QualType T) {
  QualType U = T.getNonReferenceType();
  return U->isPointerType() || U->isArrayType();
}

std::string spell(QualType T, const ASTContext &Ctx) {
  return T.getAsString(Ctx.getPrintingPolicy());
}

/// Whether \p T is a type with a name that can be used in template arguments.
bool isNamable(QualType T) {
  if (!T->isTypedefNameType())
    if (const auto *RT = T->getAs<RecordType>()) {
      const RecordDecl *RD = RT->getDecl();
      return RD->getIdentifier() || RD->getTypedefNameForAnonDecl();
    }
  return true;
}

std::string renderNewType(const ClassifyResult &R, QualType T,
                          const ASTContext &Ctx) {
  assert(!R.Skip);
  if (R.NewType == BoundedType::Ptr)
    return "bounded_ptr<" + R.InnerSpelling + "> ";
  const auto *CAT = Ctx.getAsConstantArrayType(T);
  std::string N = std::to_string(CAT->getSize().getZExtValue());
  return "bounded_array<" + R.InnerSpelling + ", " + N + ">";
}

/// Whether another declarator in \p D's lexical context shares its type
/// specifier, i.e. \p D is one declarator of a multi-declarator group.
bool sharesTypeSpecifier(const DeclaratorDecl *D) {
  const TypeSourceInfo *TSI = D->getTypeSourceInfo();
  const DeclContext *DC = D->getLexicalDeclContext();
  if (!TSI || !DC)
    return false;
  SourceLocation Begin = TSI->getTypeLoc().getBeginLoc();
  for (const Decl *Sibling : DC->decls()) {
    if (Sibling == D)
      continue;
    const auto *Other = dyn_cast<DeclaratorDecl>(Sibling);
    if (Other && Other->getTypeSourceInfo() &&
        Other->getTypeSourceInfo()->getTypeLoc().getBeginLoc() == Begin)
      return true;
  }
  return false;
}

bool hasTrailingReturnType(const FunctionDecl *FD) {
  const auto *FPT = FD->getType()->getAs<FunctionProtoType>();
  return FPT && FPT->hasTrailingReturn();
}

CharSourceRange declTypeRange(const DeclaratorDecl *D) {
  if (const TypeSourceInfo *TSI = D->getTypeSourceInfo())
    return CharSourceRange::getTokenRange(TSI->getTypeLoc().getSourceRange());
  return CharSourceRange::getTokenRange(D->getSourceRange());
}

/// \return the pointee or element types TypeLoc if TL is a (qualified) pointer
/// or array type.
TypeLoc getInnerTypeLoc(TypeLoc TL) {
  TL = TL.getUnqualifiedLoc();
  if (auto PTL = TL.getAs<PointerTypeLoc>())
    return PTL.getPointeeLoc();
  if (auto ATL = TL.getAs<ArrayTypeLoc>())
    return ATL.getElementLoc();
  return {};
}

/// Whether \p T spells a cv-qualifier keyword.
bool isCVQualifier(const Token &T) {
  return T.is(tok::raw_identifier) && (T.getRawIdentifier() == "const" ||
                                       T.getRawIdentifier() == "volatile");
}

/// Probe leading qualifiers for a type 'T'. The probe is bounded in the range
/// [ \p DeclBegin, \p TypeBegin ), where the lower bound is the begin location
/// of the declaration where 'T' is spelled and the upper bound is the begin of
/// the  spell of 'T'.
///
/// The function updates \p TypeBegin if it finds cv-qualifiers preceding the
/// original \p TypeBegin without any other token intervening in between. \p
/// TypeBegin is not updated if there is no leading cv-qualifier. Otherwise,
/// returns the probe failed reason.
///
/// \p TypeBegin is always token location.
std::optional<ReportReason> extendLeadingQualifiers(SourceLocation DeclBegin,
                                                    SourceLocation &TypeBegin,
                                                    const ASTContext &Ctx) {
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();

  std::optional<SourceLocation> FirstCVBegin;
  std::optional<Token> Tok = Token();

  if (Lexer::getRawToken(DeclBegin, *Tok, SM, LangOpts,
                         /*IgnoreWhiteSpace=*/true))
    return ReportReason::EmissionFailed;
  while (SM.isBeforeInTranslationUnit(Tok->getLocation(), TypeBegin)) {
    if (isCVQualifier(*Tok)) {
      if (!FirstCVBegin) {
        // Found first cv-qualifier, set `FirstCVBegin`.
        FirstCVBegin = Tok->getLocation();
      }
    } else if (FirstCVBegin)
      // Bail when there is unexpected token between cv-qualifiers and the
      // original TypeBegin:
      return ReportReason::UnexpectedLeadingQualifier;
    Tok = Lexer::findNextToken(Tok->getEndLoc(), SM, LangOpts,
                               /*IncludeComments=*/true);
    if (!Tok)
      return ReportReason::EmissionFailed;
  }
  if (FirstCVBegin)
    TypeBegin = *FirstCVBegin; // set the real TypeBegin after propagation
  return std::nullopt;
}

/// Probe trailing qualifiers for a type 'T'. The probe is bounded in the range
/// ( \p TypeEnd, \p UpperBound ), where the lower bound is the end location
/// of 'T' and the upper bound should be a location within the declaration where
/// 'T' is spelled.
///
/// The function updates \p TypeEnd if it finds cv-qualifiers following the
/// original \p TypeEnd without any other token intervening in between.
/// \p TypeEnd is not updated if there is no following cv-qualifier. Otherwise,
/// returns the probe failed reason.
///
/// \p TypeBegin is always token location.
std::optional<ReportReason> extendTrailingQualifiers(SourceLocation &TypeEnd,
                                                     SourceLocation UpperBound,
                                                     const ASTContext &Ctx) {
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();

  std::optional<SourceLocation> LastCVBegin;
  bool RunEnded = false;

  std::optional<Token> Tok = Lexer::findNextToken(TypeEnd, SM, LangOpts,
                                                  /*IncludeComments=*/true);
  if (!Tok)
    return ReportReason::EmissionFailed;
  while (SM.isBeforeInTranslationUnit(Tok->getLocation(), UpperBound)) {
    if (isCVQualifier(*Tok)) {
      // Bail if there is anything unexpected between TypeEnd and a
      // cv-qualifier.
      if (RunEnded)
        return ReportReason::UnexpectedTrailingQualifier;
      LastCVBegin = Tok->getLocation();
    } else
      RunEnded = true;
    Tok = Lexer::findNextToken(Tok->getEndLoc(), SM, LangOpts,
                               /*IncludeComments=*/true);
    if (!Tok)
      return ReportReason::EmissionFailed;
  }
  if (LastCVBegin)
    TypeEnd = *LastCVBegin; // set the real TypeEnd after propagation
  return std::nullopt;
}

using Levels = llvm::SmallSet<unsigned, 4>;
using DeclLevels = std::map<const Decl *, Levels>;
using ReturnLevels = std::map<const FunctionDecl *, Levels>;

/// Reverse index from the whole-program reachability result onto entity names,
/// so a declaration in this TU can look up its reachable pointer levels.
class ReachabilityMap {
  const EntityPointerLevelSet &Reachables;
  std::map<EntityName, EntityId> NameToId;

public:
  ReachabilityMap(const WPASuite &Suite,
                  const EntityPointerLevelSet &Reachables)
      : Reachables(Reachables) {
    Suite.getIdTable().forEach([this](const EntityName &Name, EntityId Id) {
      NameToId.emplace(Name, Id);
    });
  }

  llvm::SmallSet<unsigned, 4> levelsFor(std::optional<EntityName> Name) const {
    llvm::SmallSet<unsigned, 4> Levels;
    if (!Name)
      return Levels;
    auto NameIt = NameToId.find(*Name);
    if (NameIt == NameToId.end())
      return Levels;
    auto [Begin, End] = Reachables.equal_range(NameIt->second);
    for (const EntityPointerLevel &EPL : llvm::make_range(Begin, End))
      Levels.insert(EPL.getPointerLevel());
    return Levels;
  }
};

/// Collects the reachable pointer/array declarators and function returns
/// declared in this TU.
class CollectVisitor : public DynamicRecursiveASTVisitor {
public:
  CollectVisitor(const ReachabilityMap &Reach,
                 const NestedBuildNamespace &TUNamespace,
                 const NestedBuildNamespace &LUNamespace, DeclLevels &Decls,
                 ReturnLevels &Returns)
      : Reach(Reach), TUNamespace(TUNamespace), LUNamespace(LUNamespace),
        Decls(Decls), Returns(Returns) {}

  bool VisitVarDecl(VarDecl *D) override {
    collect(D, D->getType(),
            getQualifiedEntityName(D, TUNamespace, LUNamespace));
    return true;
  }

  bool VisitFieldDecl(FieldDecl *D) override {
    collect(D, D->getType(),
            getQualifiedEntityName(D, TUNamespace, LUNamespace));
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) override {
    if (!FD->isTemplated() && isCandidateType(FD->getReturnType())) {
      llvm::SmallSet<unsigned, 4> Levels = Reach.levelsFor(
          getQualifiedEntityNameForReturn(FD, TUNamespace, LUNamespace));
      if (!Levels.empty())
        Returns[FD] = std::move(Levels);
    }
    return true;
  }

private:
  void collect(const Decl *D, QualType T, std::optional<EntityName> Name) {
    if (D->isTemplated() || !isCandidateType(T))
      return;
    llvm::SmallSet<unsigned, 4> Levels = Reach.levelsFor(Name);
    if (!Levels.empty())
      Decls[D] = std::move(Levels);
  }

  const ReachabilityMap &Reach;
  NestedBuildNamespace TUNamespace;
  NestedBuildNamespace LUNamespace;
  DeclLevels &Decls;
  ReturnLevels &Returns;
};

/// Rewrites or reports every collected declarator and function return.
class RewriteVisitor : public DynamicRecursiveASTVisitor {
public:
  // Decls and their ClassifyResults for all that are successfully
  // transformed by `emit`:
  llvm::DenseMap<const Decl *, ClassifyResult> TransformedDecls;
  llvm::DenseMap<const FunctionDecl *, ClassifyResult> TransformedReturns;

  RewriteVisitor(ASTContext &Ctx, DeclLevels &Decls, ReturnLevels &Returns,
                 SourceEditEmitter &Edits, TransformationReportEmitter &Report)
      : Ctx(Ctx), Decls(Decls), Returns(Returns), Edits(Edits), Report(Report) {
  }

  bool VisitVarDecl(VarDecl *D) override {
    processDecl(D, D->getType());
    return true;
  }

  bool VisitFieldDecl(FieldDecl *D) override {
    processDecl(D, D->getType());
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) override {
    auto It = Returns.find(FD);
    if (It == Returns.end())
      return true;
    const Levels &ReachableLevels = It->second;
    if (hasTrailingReturnType(FD))
      return report(FD, ReportReason::TrailingReturnType);

    SourceLocation NameLoc = FD->getLocation();

    ClassifyResult R =
        classifyDeclType(FD->getReturnType(), ReachableLevels, Ctx);
    if (R.Skip)
      return report(FD, *R.Skip);

    FunctionTypeLoc FunTypeLoc = FD->getFunctionTypeLoc();

    if (!FunTypeLoc)
      return report(FD, ReportReason::EmissionFailed);

    auto Reason = emit(FD->getBeginLoc(), NameLoc, FunTypeLoc.getReturnLoc(),
                       FD->getReturnType(), R);
    if (!Reason)
      TransformedReturns[FD] = R;
    return report(FD, Reason);
  }

private:
  void processDecl(DeclaratorDecl *D, QualType T) {
    auto It = Decls.find(D);
    if (It == Decls.end())
      return;
    const Levels &ReachableLevels = It->second;
    if (sharesTypeSpecifier(D))
      return (void)report(D, ReportReason::DeclarationGroup);

    const TypeSourceInfo *TSI = D->getTypeSourceInfo();

    if (!TSI)
      return (void)report(D, ReportReason::EmissionFailed);

    SourceLocation NameLoc = D->getLocation();
    ClassifyResult R = classifyDeclType(T, ReachableLevels, Ctx);

    if (R.Skip)
      return (void)report(D, *R.Skip);

    auto Reason = emit(D->getBeginLoc(), NameLoc, TSI->getTypeLoc(), T, R);

    if (!Reason)
      TransformedDecls[D] = R;
    report(D, Reason);
  }

  /// Compute the precise source range for rewriting.  The produced range is
  /// token range.
  ///
  /// For pointer types, the rewrite range is from the leading cv-qualifier of
  /// the pointee type to the '*' token of the pointer type.
  ///
  /// For array types, the rewrite range is from the leading cv-qualifier to the
  /// trailing cv-qualifier around the element type. It stops short of the
  /// declarator name, leaving the name and the extent that follows it to be
  /// handled separately.
  ///
  /// \param DeclBegin the begin location of the declaration, the lower bound of
  /// the source range before narrowing down to the precise one.
  /// \param NameLoc the location of the name of the declaration, the upper
  /// bound of the source range before narrowing down to the precise one.
  /// \param TLoc the TypeLoc of the type of the declaration
  /// \param BoundedType indicates whether it is a pointer or an array
  /// \return ReportReason if it cannot narrow down the rewrite range to the
  /// aforementioned range. std::nullopt and updated \p Result otherwise.
  std::optional<ReportReason>
  computeRewriteRange(SourceLocation DeclBegin, SourceLocation NameLoc,
                      TypeLoc TLoc, BoundedType BoundedType,
                      const ASTContext &Ctx, SourceRange &RewriteRange) {
    TypeLoc InnerTypeLoc = getInnerTypeLoc(TLoc);

    if (!InnerTypeLoc)
      return ReportReason::NoInnerTypeLoc;

    SourceLocation RewriteRangeBegin = InnerTypeLoc.getBeginLoc();
    SourceRange Result;

    if (BoundedType == BoundedType::Ptr) {
      auto PTL = TLoc.getUnqualifiedLoc().getAs<PointerTypeLoc>();

      if (!PTL || TLoc.getEndLoc() != PTL.getStarLoc())
        return ReportReason::NotPointerTypeEndWithStar;
      if (auto Reason =
              extendLeadingQualifiers(DeclBegin, RewriteRangeBegin, Ctx))
        return Reason;
      Result = {RewriteRangeBegin, PTL.getStarLoc()};
    } else {
      SourceLocation RewriteRangeEnd = InnerTypeLoc.getEndLoc();

      if (auto Reason =
              extendLeadingQualifiers(DeclBegin, RewriteRangeBegin, Ctx))
        return Reason;
      if (auto Reason = extendTrailingQualifiers(RewriteRangeEnd, NameLoc, Ctx))
        return Reason;
      Result = {RewriteRangeBegin, RewriteRangeEnd};
    }

    if (Result.getBegin().isMacroID() || Result.getEnd().isMacroID())
      return ReportReason::MacroExpansion;
    if (Result.getBegin().isInvalid() || Result.getEnd().isInvalid())
      return ReportReason::EmissionFailed;

    const SourceManager &SM = Ctx.getSourceManager();
    if (SM.getFileID(Result.getBegin()) != SM.getFileID(Result.getEnd()))
      return ReportReason::EmissionFailed;
    RewriteRange = Result;
    return std::nullopt;
  }

  /// Emits the type-token replacement (and, for arrays, deletes the trailing
  /// extent). Returns false without emitting anything if a valid,
  /// self-contained edit cannot be formed.
  std::optional<ReportReason> emit(SourceLocation DeclBegin,
                                   SourceLocation NameLoc, TypeLoc TLoc,
                                   QualType T, const ClassifyResult &R) {
    const SourceManager &SM = Ctx.getSourceManager();
    SourceRange TypeRewriteRange;

    if (auto Reason = computeRewriteRange(DeclBegin, NameLoc, TLoc, R.NewType,
                                          Ctx, TypeRewriteRange))
      return Reason;

    // TypeRewriteRange is bounded by the tokens (begin location) of the two
    // ends.  Now convert it to char range for source edit, which requires the
    // bounds to be the characters of the two ends.
    CharSourceRange TypeRewriteCharRange =
        Lexer::getAsCharRange(TypeRewriteRange, SM, Ctx.getLangOpts());
    llvm::SmallVector<tooling::Replacement, 2> Edited;

    Edited.emplace_back(SM, TypeRewriteCharRange, renderNewType(R, T, Ctx),
                        Ctx.getLangOpts());

    if (R.NewType == BoundedType::Array) {
      ArrayTypeLoc ATL = TLoc.getUnqualifiedLoc().getAs<ArrayTypeLoc>();

      if (!ATL)
        return ReportReason::EmissionFailed;

      SourceLocation LBracket = ATL.getLBracketLoc();
      SourceLocation RBracket = ATL.getRBracketLoc();
      // A clean array declarator ends at its closing bracket; otherwise the
      // element spelling wraps the name (e.g. an array of function pointers)
      // and cannot be rewritten by stripping a trailing extent.
      if (ATL.getEndLoc() != RBracket)
        return ReportReason::ArrayNotEndInBracket;
      if (LBracket.isInvalid() || RBracket.isInvalid())
        return ReportReason::EmissionFailed;
      Edited.emplace_back(SM,
                          CharSourceRange::getTokenRange(LBracket, RBracket),
                          "", Ctx.getLangOpts());
    }

    if (!llvm::all_of(Edited, std::mem_fn(&tooling::Replacement::isApplicable)))
      return ReportReason::EmissionFailed;
    for (tooling::Replacement &Repl : Edited)
      Edits.addReplacement(std::move(Repl));
    return std::nullopt;
  }

  /// Reports \p Reason for \p D, if one is given. Always returns true so that
  /// visitors can tail-call it.
  bool report(const DeclaratorDecl *D, std::optional<ReportReason> Reason) {
    if (Reason) {
      CharSourceRange Range = Lexer::getAsCharRange(
          declTypeRange(D), Ctx.getSourceManager(), Ctx.getLangOpts());
      Report.addResult(SkippedRuleId, SarifResultLevel::Note, Range,
                       messageFor(*Reason));
    }
    return true;
  }

  ASTContext &Ctx;
  DeclLevels &Decls;
  ReturnLevels &Returns;
  SourceEditEmitter &Edits;
  TransformationReportEmitter &Report;
};

// FIXME: adding report for any unsuccessful edits
// FIXME: we need clusters to group edits atomically

/// Traverses the whole TU and create edits for expressions in order to adapt to
/// transformed Decls.
class ExpressionRewriter {
public:
  ExpressionRewriter(
      ASTContext &Ctx, const SSAFOptions &Opts, SourceEditEmitter &Edits,
      const llvm::DenseMap<const Decl *, ClassifyResult> &TransformedDecls,
      const llvm::DenseMap<const FunctionDecl *, ClassifyResult>
          &TransformedReturns)
      : Ctx(Ctx), Opts(Opts), Edits(Edits), TransformedDecls(TransformedDecls),
        TransformedReturns(TransformedReturns) {}

  /// Traverses the whole \c TU and create edits expressions in order to adapt
  /// to transformed Decls.
  void rewriteExprInTU(const TranslationUnitDecl *TU);

  /// Create edits in \c AC to update expression \c E, if it is
  /// \c isPtrExprBaseTransformed, so that \c E has proper bounded type after
  /// transformation.
  ///
  /// Note that this step is self-contained.  It does not rely on the context of
  /// where \c E is.  If \c E's base is transformed, it needs retrofit.
  ///
  /// \return true iff \c E needs edits.
  bool rewriteExpression(const Expr *E, tooling::AtomicChange &AC) const;

  /// Create edits for a \c PointerFlowPair that
  /// 1) \c rewriteExpression for involved expressions; and
  /// 2) further update expressions in order to adapt the flow to the
  ///    transformation.
  std::optional<tooling::AtomicChange>
  adaptPointerFlow(const PointerFlowPair &Pair) const;

private:
  /// Associate edit operations to \c CharSourceRange always, since they carry
  /// token/char range info while single SourceLocation doesn't.  This way, the
  /// edit kind (replacement or insertion) cannot be inferred from the input
  /// (i.e., SourceRange vs. SourceLocation). Use this enum to explicitly
  /// express the kinds.
  enum EditKind { Replace, InsertAtBegin, InsertAtEnd };

  /// \param Range The source locations associated with the edit. For any \c
  /// EditKind, source locations are given by a \c CharSourceRange, which
  /// carries the information of whether it is a token range or a char range.
  /// \param NewText The text of the edit that will replace a source range or
  /// be inserted at a location.
  /// \param EditKind The kind of edit: replacement or insertion.
  /// \param AC IN/OUT parameter. The new edit will be added to \c AC
  /// \return true if edit is successfully added to \c AC
  bool addEditToAtomicChange(CharSourceRange Range, StringRef NewText,
                             EditKind EditKind,
                             tooling::AtomicChange &AC) const;

  /// \return true iff the base(s) of the pointer/array expression \c E  are all
  /// transformed to have bounded types.
  bool isExprBaseTransformed(const Expr *E) const;

  /// \return a non-null pointer to a \c ClassifyResult, if `D` is transformed
  /// to have bounded types.
  const ClassifyResult *getDeclClassifyResultsIfTransformed(const Decl *D,
                                                            bool IsRet) const;

  /// \return a non-empty vector of \c ClassifyResult, the base(s) of the
  /// pointer/array expression \c E  are all transformed to have bounded
  /// types.
  std::vector<const ClassifyResult *>
  getPtrExprClassifyResultsIfTransformed(const Expr *E) const;

  //==----------------  Expression rewrite rules  -------------------==//

  /// - Append '.data()' to RHS, if LHS is a parameter and NOT transformed, and
  /// RHS is transformed; or
  /// - Append '.as_bounded<T>()' to RHS, if LHS is a parameter and both sides
  /// are transformed but element types are not identical.
  /// \return true iff the pattern is matched and edits were attempted to be
  /// added into \c AC.
  bool adaptCallArgumentInFlow(const PointerFlowPair &Pair,
                               tooling::AtomicChange &AC) const;

  /// Edit '&e[i]' to '(e + i)' or '&*e' to 'ptr', if 'e' is transformed.
  /// \return true iff the pattern is matched and edits were attempted to be
  /// added into \c AC.
  bool retrofitAddrofElementAccess(const Expr *E,
                                   tooling::AtomicChange &AC) const;

  /// Edit '(T*)e/static_cast<T>(e)/reinterpret_cast<T>(e)' to
  /// 'e.as_bounded<T>()' , if 'e' is transformed.
  /// \return true iff the pattern is matched and edits were attempted to be
  /// added into \c AC.
  bool retrofitPointerCast(const Expr *E, tooling::AtomicChange &AC) const;

  ASTContext &Ctx;
  const SSAFOptions &Opts;
  SourceEditEmitter &Edits;
  const llvm::DenseMap<const Decl *, ClassifyResult> &TransformedDecls;
  const llvm::DenseMap<const FunctionDecl *, ClassifyResult>
      &TransformedReturns;
};

} // namespace

namespace clang::ssaf {

llvm::StringRef messageFor(ReportReason Reason) {
  switch (Reason) {
  case ReportReason::ArrayNotEndInBracket:
    return "the array type does not end in a closing bracket";
  case ReportReason::DeclarationGroup:
    return "declarator of a multi-declarator group is not yet rewritten";
  case ReportReason::EmissionFailed:
    return "no source edit could be formed for this declarator";
  case ReportReason::IncompleteArray:
    return "array of unknown bound is not yet rewritten";
  case ReportReason::MacroExpansion:
    return "declarator spelled through a macro is not yet rewritten";
  case ReportReason::MultiDimensionalArray:
    return "multi-dimensional array is not yet rewritten";
  case ReportReason::MultiLevelPointer:
    return "multi-level pointer indirection is not yet rewritten";
  case ReportReason::NoInnerTypeLoc:
    return "no TypeLoc for the pointee or array element type";
  case ReportReason::NotPointerTypeEndWithStar:
    return "pointer declarator does not end at its '*'";
  case ReportReason::NotTransformed:
    return "this declaration was not transformed";
  case ReportReason::PointerToArray:
    return "pointer to array is not yet rewritten";
  case ReportReason::ReferenceToPointer:
    return "reference to pointer is not yet rewritten";
  case ReportReason::TrailingReturnType:
    return "trailing return type is not yet rewritten";
  case ReportReason::UnexpectedLeadingQualifier:
    return "unexpected token between a leading cv-qualifier and the type";
  case ReportReason::UnexpectedTrailingQualifier:
    return "unexpected token between the type and a trailing cv-qualifier";
  case ReportReason::UnnamableType:
    return "the pointee or array element type has no name that can be written "
           "as a template argument";
  }
  llvm_unreachable("unhandled ReportReason");
}

ClassifyResult
classifyDeclType(QualType T, const llvm::SmallSet<unsigned, 4> &ReachableLevels,
                 const ASTContext &Ctx) {
  ClassifyResult R;
  if (!ReachableLevels.count(1))
    return R;

  // A deeper indirection level is reachable too; that is a multi-level rewrite,
  // which is not yet supported.
  if (llvm::any_of(ReachableLevels, [](unsigned L) { return L > 1; })) {
    R.Skip = ReportReason::MultiLevelPointer;
    return R;
  }

  if (T->isReferenceType()) {
    QualType Pointee = T.getNonReferenceType();
    if (Pointee->isPointerType() || Pointee->isArrayType())
      R.Skip = ReportReason::ReferenceToPointer;
    return R;
  }

  if (const auto *PT = T->getAs<PointerType>()) {
    QualType Pointee = PT->getPointeeType();
    if (Pointee->isFunctionType()) {
      assert(false &&
             "function pointer entities are not expected to be reachable");
      return R;
    }
    if (Pointee->isPointerType()) {
      R.Skip = ReportReason::MultiLevelPointer;
      return R;
    }
    if (Pointee->isArrayType()) {
      R.Skip = ReportReason::PointerToArray;
      return R;
    }
    if (!isNamable(Pointee)) {
      R.Skip = ReportReason::UnnamableType;
      return R;
    }
    R.NewType = BoundedType::Ptr;
    R.InnerSpelling = Pointee->isVoidType() ? "char" : spell(Pointee, Ctx);
    R.Skip = std::nullopt;
    return R;
  }

  if (const auto *CAT = Ctx.getAsConstantArrayType(T)) {
    QualType Element = CAT->getElementType();
    if (Element->isArrayType()) {
      R.Skip = ReportReason::MultiDimensionalArray;
      return R;
    }
    if (!isNamable(Element)) {
      R.Skip = ReportReason::UnnamableType;
      return R;
    }
    R.NewType = BoundedType::Array;
    R.InnerSpelling = spell(Element, Ctx);
    R.Skip = std::nullopt;
    return R;
  }

  if (T->isArrayType())
    R.Skip = ReportReason::IncompleteArray;
  return R;
}

void CppBoundedBuffers::HandleTranslationUnit(ASTContext &Ctx) {
  auto Reachable = Suite.get<UnsafeBufferReachableAnalysisResult>();
  if (!Reachable) {
    llvm::consumeError(Reachable.takeError());
    return;
  }

  ReachabilityMap Reach(Suite, Reachable->Reachables);
  NestedBuildNamespace TUNamespace =
      NestedBuildNamespace::makeCompilationUnit(Opts.CompilationUnitId);
  NestedBuildNamespace LUNamespace =
      NestedBuildNamespace::makeLinkUnit(Opts.LinkUnitId);
  DeclLevels Decls;
  ReturnLevels Returns;

  auto *TU = Ctx.getTranslationUnitDecl();
  CollectVisitor(Reach, Decls, Returns).TraverseDecl(TU);
  auto RV = RewriteVisitor(Ctx, Decls, Returns, Edits, Report);

  RV.TraverseDecl(TU);

  ExpressionRewriter ExprRewriter(Ctx, Opts, Edits, RV.TransformedDecls,
                                  RV.TransformedReturns);

  ExprRewriter.rewriteExprInTU(TU);
}

} // namespace clang::ssaf

namespace {

//===------------ ExpressionRewriter implementation --------------===//

std::optional<tooling::AtomicChange>
ExpressionRewriter::adaptPointerFlow(const PointerFlowPair &Pair) const {
  tooling::AtomicChange AC("", "");

  rewriteExpression(Pair.RHS, AC);
  adaptCallArgumentInFlow(Pair, AC); // FIXME: more flow patterns
  return AC;
}

bool ExpressionRewriter::rewriteExpression(const Expr *E,
                                           tooling::AtomicChange &AC) const {
  if (isExprBaseTransformed(E))
    return retrofitAddrofElementAccess(E, AC) || retrofitPointerCast(E, AC);
  return false;
}

bool ExpressionRewriter::adaptCallArgumentInFlow(
    const PointerFlowPair &Pair, tooling::AtomicChange &AC) const {
  const auto *PVD =
      dyn_cast_or_null<ParmVarDecl>(Pair.LHS.dyn_cast<const ValueDecl *>());

  if (!PVD)
    return false;

  // '(RHS).data()' is a prvalue that can't bind to a reference. So bail.
  if (PVD->getType()->isReferenceType())
    return false;

  QualType RTypeBeforeImpCast = Pair.RHS->IgnoreImpCasts()->getType();
  QualType LType = PVD->getType().getNonReferenceType();

  // If RHS has `void*` type, it will have `char*` after transformation and
  // being appened '.data()'.  This type change may cause the callee to be
  // silently swapped to a different overload. So bail.
  if (RTypeBeforeImpCast->isVoidPointerType() && LType->isVoidPointerType())
    return false;

  auto It = TransformedDecls.find(PVD);
  bool IsLHSTransformed = It != TransformedDecls.end();
  bool IsRHSTransformed = isExprBaseTransformed(Pair.RHS);

  CharSourceRange RHSRange = Lexer::getAsCharRange(
      Pair.RHS->getSourceRange(), Ctx.getSourceManager(), Ctx.getLangOpts());

  if (!IsLHSTransformed && IsRHSTransformed) {
    addEditToAtomicChange(RHSRange, "(", EditKind::InsertAtBegin, AC);
    addEditToAtomicChange(RHSRange, ").data()", EditKind::InsertAtEnd, AC);
    return true;
  }
  if (IsLHSTransformed && IsRHSTransformed) {
    QualType RPteTy = RTypeBeforeImpCast->getPointeeType();
    QualType LPteTy = LType->getPointeeType();

    if (LPteTy.isNull() || RPteTy.isNull())
      return false;

    if (Ctx.hasSameType(LPteTy, RPteTy))
      return false;

    StringRef InnerSpelling = It->getSecond().InnerSpelling;

    addEditToAtomicChange(RHSRange, "(", EditKind::InsertAtBegin, AC);
    addEditToAtomicChange(
        RHSRange,
        (").as_bounded<" + InnerSpelling + ">()").getSingleStringRef(),
        EditKind::InsertAtEnd, AC);
    return true;
  }
  return false;
}

bool ExpressionRewriter::retrofitAddrofElementAccess(
    const Expr *E, tooling::AtomicChange &AC) const {
  const auto *UO = dyn_cast<UnaryOperator>(E->IgnoreParenImpCasts());
  if (!UO || UO->getOpcode() != UO_AddrOf)
    return false;

  const Expr *SubExpr = UO->getSubExpr()->IgnoreParenImpCasts();
  const Expr *Ptr, *Offset = nullptr;

  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(SubExpr)) {
    Ptr = ASE->getBase();
    Offset = ASE->getIdx();
  } else if (const auto *Deref = dyn_cast<UnaryOperator>(SubExpr);
             Deref && Deref->getOpcode() == UO_Deref) {
    Ptr = Deref->getSubExpr();
  } else
    return false;

  if (!isExprBaseTransformed(Ptr))
    return false;

  rewriteExpression(Ptr, AC);
  // Ptr may have been recursively edited, so its source range should stay
  // intact.

  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LO = Ctx.getLangOpts();
  auto PtrCR = Lexer::getAsCharRange(Ptr->getSourceRange(), SM, LO);
  auto FullExprCR = Lexer::getAsCharRange(UO->getSourceRange(), SM, LO);
  // Source range before `Ptr`:
  auto PrePtrCR =
      CharSourceRange::getCharRange(FullExprCR.getBegin(), PtrCR.getBegin());

  if (!Offset) {
    // Source range after `Ptr`:
    auto PostPtrCR =
        CharSourceRange::getCharRange(PtrCR.getEnd(), FullExprCR.getEnd());
    // For '&*ptr' or '&(*ptr)', drop contents in PrePtrCR and PostPtrCR:
    addEditToAtomicChange(PrePtrCR, "", EditKind::Replace, AC);
    addEditToAtomicChange(PostPtrCR, "", EditKind::Replace, AC);
    return true;
  }

  auto OffsetCR = Lexer::getAsCharRange(Offset->getSourceRange(), SM, LO);
  auto PostPtrPreOffsetCR =
      CharSourceRange::getCharRange(PtrCR.getEnd(), OffsetCR.getBegin());
  auto PostOffsetCR =
      CharSourceRange::getCharRange(OffsetCR.getEnd(), FullExprCR.getEnd());

  // For '&ptr[offset]' or '&(ptr[offset])',
  // 1. replace contents in PrePtrCR with "(", and
  // 2. replace contents in PostPtrPreOffsetCR with " + ", and
  // 3. replace contents in postOffsetCR  with ") ",
  // results in '(ptr + offset)':
  addEditToAtomicChange(PrePtrCR, "(", EditKind::Replace, AC);
  addEditToAtomicChange(PostPtrPreOffsetCR, " + ", EditKind::Replace, AC);
  addEditToAtomicChange(PostOffsetCR, ")", EditKind::Replace, AC);
  return true;
}

bool ExpressionRewriter::retrofitPointerCast(const Expr *E,
                                             tooling::AtomicChange &AC) const {
  const auto *CE = dyn_cast<ExplicitCastExpr>(E->IgnoreParenImpCasts());

  if (!CE ||
      !isa<CStyleCastExpr, CXXStaticCastExpr, CXXReinterpretCastExpr>(CE))
    return false;
  if (!isExprBaseTransformed(CE))
    return false;

  QualType DestTy = CE->getTypeAsWritten();
  QualType DestPteTy = DestTy->getPointeeType();

  if (!DestTy->isPointerType())
    return false;

  const Expr *Ptr = CE->getSubExpr();
  rewriteExpression(Ptr, AC);
  // Ptr may have been recursively edited, so its source range should stay
  // intact.

  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LO = Ctx.getLangOpts();
  CharSourceRange PtrCR =
      Lexer::getAsCharRange(CE->getSubExpr()->getSourceRange(), SM, LO);
  CharSourceRange FullCastExprCR =
      Lexer::getAsCharRange(CE->getSourceRange(), SM, LO);
  CharSourceRange PrePtrCR = CharSourceRange::getCharRange(
      FullCastExprCR.getBegin(), PtrCR.getBegin());
  CharSourceRange PostPtrCR =
      CharSourceRange::getCharRange(PtrCR.getEnd(), FullCastExprCR.getEnd());
  std::string T = spell(DestPteTy, Ctx);

  // For `(T*)ptr` or *_cast<T>(ptr),
  // 1. replace contents in PrePtrCR with "(", and
  // 2. replace contents in PostPtrCR with ").as_bounded<T>()",
  // results in '(ptr).as_bounded<T>()'.
  addEditToAtomicChange(PrePtrCR, "(", EditKind::Replace, AC);
  addEditToAtomicChange(PostPtrCR, ").as_bounded<" + T + ">()",
                        EditKind::Replace, AC);
  return true;
}

void ExpressionRewriter::rewriteExprInTU(const TranslationUnitDecl *TU) {
  llvm::DenseMap<const NamedDecl *, std::vector<const NamedDecl *>>
      ContributorGroups;

  findContributors(Ctx, Opts, ContributorGroups,
                   /*ExtractFromSystemHeaders=*/false);

  llvm::SmallVector<PointerFlowPair> Pairs;
  PointerFlowPairMatcher Matcher{Ctx};

  for (auto &[GrpCano, ContriGrp] : ContributorGroups)
    for (auto *ContriDecl : ContriGrp) {
      auto PairsCollector = [&Pairs, &Matcher,
                             &ContriDecl](const DynTypedNode &Node) {
        Matcher.matches(Node, ContriDecl, Pairs);
      };
      findMatchesIn(ContriDecl, PairsCollector);
    }

  for (const PointerFlowPair &Pair : Pairs) {
    if (auto AC = adaptPointerFlow(Pair);
        AC && llvm::all_of(AC->getReplacements(),
                           std::mem_fn(&tooling::Replacement::isApplicable)))
      for (const tooling::Replacement &R : AC->getReplacements())
        Edits.addReplacement(R);
  }
}

bool ExpressionRewriter::isExprBaseTransformed(const Expr *E) const {
  return !getPtrExprClassifyResultsIfTransformed(E).empty();
}

const ClassifyResult *
ExpressionRewriter::getDeclClassifyResultsIfTransformed(const Decl *D,
                                                        bool IsRet) const {
  auto Lookup = [](const auto &Map, const auto *Key) {
    auto It = Map.find(Key);
    return It == Map.end() ? nullptr : &It->second;
  };
  return IsRet ? Lookup(TransformedReturns, cast<FunctionDecl>(D))
               : Lookup(TransformedDecls, D);
}

std::vector<const ClassifyResult *>
ExpressionRewriter::getPtrExprClassifyResultsIfTransformed(
    const Expr *E) const {
  auto DPLs = translateDeclPointerLevel(E, Ctx);

  if (!DPLs) {
    // Errors indicate no transformation for E. No further action.
    llvm::consumeError(DPLs.takeError());
    return {};
  }

  std::vector<const ClassifyResult *> Result;

  for (auto &DPL : *DPLs) {
    const auto *ClassifyResult =
        getDeclClassifyResultsIfTransformed(DPL.Decl, DPL.IsReturn);

    if (!ClassifyResult)
      return {};
    Result.push_back(ClassifyResult);
  }
  return Result;
}

bool ExpressionRewriter::addEditToAtomicChange(
    CharSourceRange Range, StringRef NewText, EditKind EditKind,
    tooling::AtomicChange &AC) const {
  assert(Range.isCharRange());

  if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID())
    // FIXME: report...
    return false;

  const SourceManager &SM = Ctx.getSourceManager();
  llvm::Error Err = [&]() -> llvm::Error {
    switch (EditKind) {
    case Replace:
      return AC.replace(SM, Range, NewText);
    case InsertAtBegin:
      return AC.insert(SM, Range.getBegin(), NewText, /*InsertAfter=*/false);
    case InsertAtEnd:
      return AC.insert(SM, Range.getEnd(), NewText, /*InsertAfter=*/true);
    }
    llvm_unreachable("unhandled EditKind");
  }();

  if (Err) {
    llvm::consumeError(std::move(Err));
    // FIXME: generate a Report
    // If AtomicChange has an error,the whole should be discard
    return false;
  }
  return true;
}

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int CppBoundedBuffersAnchorSource = 0;
} // namespace clang::ssaf

static clang::ssaf::TransformationRegistry::Add<CppBoundedBuffers>
    RegisterCppBoundedBuffers("cpp-bounded-buffers",
                              "Rewrites buffers into bounded types");
