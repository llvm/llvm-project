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

/// Whether \p T can be re-emitted as written. Anonymous records and lambdas
/// have no usable spelling.
bool isReproducible(QualType T) {
  const auto *RT = T->getAs<RecordType>();
  if (!RT)
    return true;
  const RecordDecl *RD = RT->getDecl();
  if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    if (CXXRD->isLambda())
      return false;
  return RD->getIdentifier() || RD->getTypedefNameForAnonDecl();
}

std::string cvPrefix(QualType T) {
  std::string Prefix;
  if (T.isLocalConstQualified())
    Prefix += "const ";
  if (T.isLocalVolatileQualified())
    Prefix += "volatile ";
  return Prefix;
}

std::string renderNewType(const ClassifyResult &R, QualType T,
                          const ASTContext &Ctx) {
  if (*R.NewType == BoundedType::Ptr)
    return cvPrefix(T) + "bounded_ptr<" + R.InnerSpelling + "> ";
  const auto *CAT = Ctx.getAsConstantArrayType(T);
  std::string N = std::to_string(CAT->getSize().getZExtValue());
  return "bounded_array<" + R.InnerSpelling + ", " + N + "> ";
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

/// A leading cv-qualifier keyword (e.g. the `const` in `const char *`) is not
/// covered by the type-loc's begin location; extend \p TypeBegin left over it.
SourceLocation extendOverLeadingQualifiers(SourceLocation TypeBegin,
                                           const ASTContext &Ctx) {
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();
  while (std::optional<Token> Prev = Lexer::findPreviousToken(
             TypeBegin, SM, LangOpts, /*IncludeComments=*/false)) {
    // findPreviousToken lexes raw tokens, so keywords arrive as identifiers.
    if (!Prev->is(tok::raw_identifier))
      break;
    StringRef Text = Prev->getRawIdentifier();
    if (Text != "const" && Text != "volatile")
      break;
    TypeBegin = Prev->getLocation();
  }
  return TypeBegin;
}

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

struct Candidate {
  llvm::SmallSet<unsigned, 4> Levels;
  bool AccountedFor = false;
};

using DeclLevels = std::map<const Decl *, Candidate>;
using ReturnLevels = std::map<const FunctionDecl *, Candidate>;

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
        Returns[FD].Levels = std::move(Levels);
    }
    return true;
  }

private:
  void collect(const Decl *D, QualType T, std::optional<EntityName> Name) {
    if (D->isTemplated() || !isCandidateType(T))
      return;
    llvm::SmallSet<unsigned, 4> Levels = Reach.levelsFor(Name);
    if (!Levels.empty())
      Decls[D].Levels = std::move(Levels);
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
    Candidate &Cand = It->second;
    if (hasTrailingReturnType(FD))
      return account(Cand, FD, ReportReason::TrailingReturnType);

    SourceLocation TypeBegin = FD->getReturnTypeSourceRange().getBegin();
    SourceLocation NameLoc = FD->getLocation();
    if (TypeBegin.isMacroID() || NameLoc.isMacroID())
      return account(Cand, FD, ReportReason::MacroExpansion);

    ClassifyResult R = classifyDeclType(FD->getReturnType(), Cand.Levels, Ctx);
    if (R.Skip)
      return account(Cand, FD, *R.Skip);
    if (R.NewType) {
      bool Ok = emit(TypeBegin, NameLoc, FD->getReturnType(), R,
                     /*ArrayTypeLoc=*/std::nullopt);
      return account(Cand, FD,
                     Ok ? std::nullopt
                        : std::optional(ReportReason::EmissionFailed));
    }
    return true;
  }

private:
  void processDecl(DeclaratorDecl *D, QualType T) {
    auto It = Decls.find(D);
    if (It == Decls.end())
      return;
    Candidate &Cand = It->second;
    if (sharesTypeSpecifier(D))
      return (void)account(Cand, D, ReportReason::DeclarationGroup);

    const TypeSourceInfo *TSI = D->getTypeSourceInfo();
    SourceLocation TypeBegin =
        TSI ? TSI->getTypeLoc().getBeginLoc() : SourceLocation();
    SourceLocation NameLoc = D->getLocation();
    if (TypeBegin.isMacroID() || NameLoc.isMacroID())
      return (void)account(Cand, D, ReportReason::MacroExpansion);

    ClassifyResult R = classifyDeclType(T, Cand.Levels, Ctx);
    if (R.Skip)
      return (void)account(Cand, D, *R.Skip);
    if (R.NewType) {
      std::optional<TypeLoc> ArrayTypeLoc;
      if (*R.NewType == BoundedType::Array && TSI)
        ArrayTypeLoc = TSI->getTypeLoc();
      bool Ok = emit(TypeBegin, NameLoc, T, R, ArrayTypeLoc);
      account(Cand, D,
              Ok ? std::nullopt : std::optional(ReportReason::EmissionFailed));
    }
  }

  /// Emits the type-token replacement (and, for arrays, deletes the trailing
  /// extent). Returns false without emitting anything if a valid,
  /// self-contained edit cannot be formed.
  bool emit(SourceLocation TypeBegin, SourceLocation NameLoc, QualType T,
            const ClassifyResult &R, std::optional<TypeLoc> ForArray) {
    const SourceManager &SM = Ctx.getSourceManager();
    if (TypeBegin.isValid() && !TypeBegin.isMacroID())
      TypeBegin = extendOverLeadingQualifiers(TypeBegin, Ctx);
    if (TypeBegin.isInvalid() || NameLoc.isInvalid() || TypeBegin.isMacroID() ||
        NameLoc.isMacroID() ||
        SM.getFileID(TypeBegin) != SM.getFileID(NameLoc) ||
        SM.getFileOffset(NameLoc) <= SM.getFileOffset(TypeBegin))
      return false;

    llvm::SmallVector<tooling::Replacement, 2> Edited;
    Edited.emplace_back(SM, CharSourceRange::getCharRange(TypeBegin, NameLoc),
                        renderNewType(R, T, Ctx), Ctx.getLangOpts());

    if (ForArray) {
      ArrayTypeLoc ATL = ForArray->getAs<ArrayTypeLoc>();
      if (!ATL)
        return false;
      SourceLocation LBracket = ATL.getLBracketLoc();
      SourceLocation RBracket = ATL.getRBracketLoc();
      // A clean array declarator ends at its closing bracket; otherwise the
      // element spelling wraps the name (e.g. an array of function pointers)
      // and cannot be rewritten by stripping a trailing extent.
      if (LBracket.isInvalid() || RBracket.isInvalid() ||
          ForArray->getEndLoc() != RBracket)
        return false;
      Edited.emplace_back(SM,
                          CharSourceRange::getTokenRange(LBracket, RBracket),
                          "", Ctx.getLangOpts());
    }

    for (const tooling::Replacement &Repl : Edited)
      if (!Repl.isApplicable())
        return false;
    for (tooling::Replacement &Repl : Edited)
      Edits.addReplacement(std::move(Repl));
    return true;
  }

  /// Marks \p Cand accounted for, reporting \p Reason if one is given.
  bool account(Candidate &Cand, const DeclaratorDecl *D,
               std::optional<ReportReason> Reason) {
    Cand.AccountedFor = true;
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
  case ReportReason::MultiLevelPointer:
    return "multi-level pointer indirection is not yet rewritten";
  case ReportReason::PointerToArray:
    return "pointer to array is not yet rewritten";
  case ReportReason::ReferenceToPointer:
    return "reference to pointer is not yet rewritten";
  case ReportReason::MultiDimensionalArray:
    return "multi-dimensional array is not yet rewritten";
  case ReportReason::IncompleteArray:
    return "array of unknown bound is not yet rewritten";
  case ReportReason::UnreproducibleType:
    return "type spelling cannot be reproduced";
  case ReportReason::DeclarationGroup:
    return "declarator of a multi-declarator group is not yet rewritten";
  case ReportReason::MacroExpansion:
    return "declarator spelled through a macro is not yet rewritten";
  case ReportReason::TrailingReturnType:
    return "trailing return type is not yet rewritten";
  case ReportReason::EmissionFailed:
    return "no source edit could be formed for this declarator";
  case ReportReason::NotTransformed:
    return "reachable buffer was not transformed";
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
    if (!isReproducible(Pointee)) {
      R.Skip = ReportReason::UnreproducibleType;
      return R;
    }
    R.NewType = BoundedType::Ptr;
    R.InnerSpelling = Pointee->isVoidType() ? "char" : spell(Pointee, Ctx);
    return R;
  }

  if (const auto *CAT = Ctx.getAsConstantArrayType(T)) {
    QualType Element = CAT->getElementType();
    if (Element->isArrayType()) {
      R.Skip = ReportReason::MultiDimensionalArray;
      return R;
    }
    if (!isReproducible(Element)) {
      R.Skip = ReportReason::UnreproducibleType;
      return R;
    }
    R.NewType = BoundedType::Array;
    R.InnerSpelling = spell(Element, Ctx);
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

  // Every reachable buffer in this TU is either rewritten or reported; a
  // leftover means it was neither, which must still be surfaced.
  for (const auto &[D, Cand] : Decls)
    if (!Cand.AccountedFor)
      Report.addResult(SkippedRuleId, SarifResultLevel::Note,
                       declTypeRange(cast<DeclaratorDecl>(D)),
                       messageFor(ReportReason::NotTransformed));
  for (const auto &[FD, Cand] : Returns)
    if (!Cand.AccountedFor)
      Report.addResult(
          SkippedRuleId, SarifResultLevel::Note,
          CharSourceRange::getTokenRange(FD->getReturnTypeSourceRange()),
          messageFor(ReportReason::NotTransformed));
}

} // namespace clang::ssaf

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int CppBoundedBuffersAnchorSource = 0;
} // namespace clang::ssaf

static clang::ssaf::TransformationRegistry::Add<CppBoundedBuffers>
    RegisterCppBoundedBuffers("cpp-bounded-buffers",
                              "Rewrites buffers into bounded types");
