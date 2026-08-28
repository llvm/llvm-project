//===- CppBoundedBuffers.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformations/CppBoundedBuffers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationRegistry.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <map>
#include <optional>
#include <string>

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
  const std::map<EntityId, EntityPointerLevelSet> &Reachables;
  std::map<EntityName, EntityId> NameToId;

public:
  ReachabilityMap(const WPASuite &Suite,
                  const std::map<EntityId, EntityPointerLevelSet> &Reachables)
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
    auto ReachIt = Reachables.find(NameIt->second);
    if (ReachIt == Reachables.end())
      return Levels;
    for (const EntityPointerLevel &EPL : ReachIt->second)
      Levels.insert(EPL.getPointerLevel());
    return Levels;
  }
};

/// Collects the reachable pointer/array declarators and function returns
/// declared in this TU.
class CollectVisitor : public DynamicRecursiveASTVisitor {
public:
  CollectVisitor(const ReachabilityMap &Reach, DeclLevels &Decls,
                 ReturnLevels &Returns)
      : Reach(Reach), Decls(Decls), Returns(Returns) {}

  bool VisitVarDecl(VarDecl *D) override {
    collect(D, D->getType(), getEntityName(D));
    return true;
  }

  bool VisitFieldDecl(FieldDecl *D) override {
    collect(D, D->getType(), getEntityName(D));
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) override {
    if (!FD->isTemplated() && isCandidateType(FD->getReturnType())) {
      llvm::SmallSet<unsigned, 4> Levels =
          Reach.levelsFor(getEntityNameForReturn(FD));
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
  DeclLevels &Decls;
  ReturnLevels &Returns;
};

/// Rewrites or reports every collected declarator and function return.
class RewriteVisitor : public DynamicRecursiveASTVisitor {
public:
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
    return report(FD, emit(FD->getBeginLoc(), NameLoc,
                           FunTypeLoc.getReturnLoc(), FD->getReturnType(), R));
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
    report(D, emit(D->getBeginLoc(), NameLoc, TSI->getTypeLoc(), T, R));
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
    if (Reason)
      Report.addResult(SkippedRuleId, SarifResultLevel::Note, declTypeRange(D),
                       messageFor(*Reason));
    return true;
  }

  ASTContext &Ctx;
  DeclLevels &Decls;
  ReturnLevels &Returns;
  SourceEditEmitter &Edits;
  TransformationReportEmitter &Report;
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
  DeclLevels Decls;
  ReturnLevels Returns;

  Decl *TU = Ctx.getTranslationUnitDecl();
  CollectVisitor(Reach, Decls, Returns).TraverseDecl(TU);
  RewriteVisitor(Ctx, Decls, Returns, Edits, Report).TraverseDecl(TU);
}

} // namespace clang::ssaf

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int CppBoundedBuffersAnchorSource = 0;
} // namespace clang::ssaf

static clang::ssaf::TransformationRegistry::Add<CppBoundedBuffers>
    RegisterCppBoundedBuffers("cpp-bounded-buffers",
                              "Rewrites buffers into bounded types");
