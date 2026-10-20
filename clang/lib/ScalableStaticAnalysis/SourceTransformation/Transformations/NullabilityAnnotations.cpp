//===- NullabilityAnnotations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/SourceTransformation/Transformations/NullabilityAnnotations.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Analysis/Analyses/NullabilitySafety.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/NullabilitySafety/NullabilitySafetyAnalysis.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/TransformationRegistry.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include <map>
#include <optional>
#include <string>

using namespace clang;
using namespace clang::ssaf;

static constexpr llvm::StringLiteral SkippedRuleId =
    "nullability-annotation-skipped";
static constexpr llvm::StringLiteral ReachableRuleId =
    "nullability-nullable-reachable";
static constexpr llvm::StringLiteral NullStoreRuleId =
    "nullability-null-store-to-field";
static constexpr llvm::StringLiteral NullableSuggestionRuleId =
    "nullability-nullable-suggestion";

namespace {

struct EntityFacts {
  bool Nonnull = false;
  bool Nullable = false;
  bool Reachable = false;
  bool InnerLevel = false;
};

class InferenceMap {
  const NullabilityInferenceAnalysisResult &Result;
  std::map<EntityName, EntityId> NameToId;

  static void scan(const EntityPointerLevelSet &Set, EntityId Id, bool &Outer,
                   bool *Inner) {
    auto [Begin, End] = Set.equal_range(Id);
    for (const EntityPointerLevel &EPL : llvm::make_range(Begin, End)) {
      if (EPL.getPointerLevel() == 1)
        Outer = true;
      else if (Inner)
        *Inner = true;
    }
  }

public:
  InferenceMap(const WPASuite &Suite,
               const NullabilityInferenceAnalysisResult &Result)
      : Result(Result) {
    Suite.getIdTable().forEach([this](const EntityName &Name, EntityId Id) {
      NameToId.emplace(Name, Id);
    });
  }

  EntityFacts factsFor(const std::optional<EntityName> &Name) const {
    EntityFacts F;
    if (!Name)
      return F;
    auto It = NameToId.find(*Name);
    if (It == NameToId.end())
      return F;
    scan(Result.Nonnull, It->second, F.Nonnull, &F.InnerLevel);
    scan(Result.Nullable, It->second, F.Nullable, &F.InnerLevel);
    scan(Result.NullableReachable, It->second, F.Reachable, nullptr);
    return F;
  }
};

bool isNullabilityKeyword(const Token &Tok) {
  if (!Tok.is(tok::raw_identifier))
    return false;
  return llvm::StringSwitch<bool>(Tok.getRawIdentifier())
      .Cases({"_Nonnull", "_Nullable", "_Nullable_result", "_Null_unspecified"},
             true)
      .Cases({"__nonnull", "__nullable", "__null_unspecified"}, true)
      .Default(false);
}

TypeLoc stripQualifiersAndAttributes(TypeLoc TL) {
  TL = TL.getUnqualifiedLoc();
  while (auto ATL = TL.getAs<AttributedTypeLoc>())
    TL = ATL.getModifiedLoc().getUnqualifiedLoc();
  return TL;
}

class NullStoreCollector final : public NullabilitySafetyHandler {
public:
  llvm::SmallVector<std::pair<const Expr *, const FieldDecl *>, 16> Stores;

  void handleNullableDereference(const Expr *, QualType) override {}

  void handleMemberAssignEvidence(const Expr *AssignExpr,
                                  const FieldDecl *Member,
                                  NullabilityEvidence Kind) override {
    if (Kind == NullabilityEvidence::Nullable &&
        Seen.insert({AssignExpr, Member}).second)
      Stores.push_back({AssignExpr, Member});
  }

private:
  llvm::DenseSet<std::pair<const Expr *, const FieldDecl *>> Seen;
};

class AnnotateVisitor : public DynamicRecursiveASTVisitor {
public:
  AnnotateVisitor(ASTContext &Ctx, const InferenceMap &Inference,
                  const NestedBuildNamespace &TUNamespace,
                  const NestedBuildNamespace &LUNamespace,
                  SourceEditEmitter &Edits, TransformationReportEmitter &Report)
      : Ctx(Ctx), SM(Ctx.getSourceManager()), Inference(Inference),
        TUNamespace(TUNamespace), LUNamespace(LUNamespace), Edits(Edits),
        Report(Report) {}

  bool VisitParmVarDecl(ParmVarDecl *D) override {
    if (const TypeSourceInfo *TSI = D->getTypeSourceInfo())
      process(D, TSI->getTypeLoc(), D->getType(),
              getQualifiedEntityName(D, TUNamespace, LUNamespace));
    return true;
  }

  bool VisitFieldDecl(FieldDecl *D) override {
    if (const TypeSourceInfo *TSI = D->getTypeSourceInfo())
      process(D, TSI->getTypeLoc(), D->getType(),
              getQualifiedEntityName(D, TUNamespace, LUNamespace));
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) override {
    if (FunctionTypeLoc FTL = FD->getFunctionTypeLoc())
      process(FD, FTL.getReturnLoc(), FD->getReturnType(),
              getQualifiedEntityNameForReturn(FD, TUNamespace, LUNamespace));
    return true;
  }

private:
  void process(const Decl *D, TypeLoc TL, QualType T,
               const std::optional<EntityName> &Name) {
    if (D->isImplicit() || D->isTemplated() ||
        SM.isInSystemHeader(D->getLocation()))
      return;
    EntityFacts F = Inference.factsFor(Name);
    if (F.Nullable) {
      if (!isa<FieldDecl>(D))
        report(TL, NullableSuggestionRuleId,
               "a null value reaches this pointer on every path from some "
               "caller or store; consider _Nullable (not written)");
      return;
    }
    const char *Keyword = F.Nonnull ? "_Nonnull" : nullptr;
    if (!Keyword) {
      if (F.InnerLevel)
        report(TL, SkippedRuleId,
               messageFor(NullabilitySkipReason::InnerPointerLevel));
      else if (F.Reachable)
        report(TL, ReachableRuleId,
               "a nullable value may reach this pointer through pointer flow; "
               "left unannotated");
      return;
    }
    if (auto K = T->getNullability(); K && *K != NullabilityKind::Unspecified)
      return;
    if (std::optional<NullabilitySkipReason> Reason = annotate(TL, Keyword))
      report(TL, SkippedRuleId, messageFor(*Reason));
  }

  std::optional<NullabilitySkipReason> annotate(TypeLoc TL,
                                                llvm::StringRef Keyword) {
    TypeLoc Outer = stripQualifiersAndAttributes(TL);
    SourceLocation After;
    bool LeadingSpace = false;
    if (auto PTL = Outer.getAs<PointerTypeLoc>()) {
      After = PTL.getStarLoc();
    } else if (Outer.getAs<TypedefTypeLoc>() || Outer.getAs<UsingTypeLoc>()) {
      After = Outer.getEndLoc();
      LeadingSpace = true;
    } else {
      return NullabilitySkipReason::NoPointerSpelling;
    }
    if (After.isInvalid())
      return NullabilitySkipReason::EmissionFailed;
    if (After.isMacroID())
      return NullabilitySkipReason::MacroExpansion;

    const LangOptions &LO = Ctx.getLangOpts();
    std::optional<Token> Next =
        Lexer::findNextToken(After, SM, LO, /*IncludeComments=*/false);
    if (Next && isNullabilityKeyword(*Next))
      return std::nullopt;

    SourceLocation InsertLoc = Lexer::getLocForEndOfToken(After, 0, SM, LO);
    if (InsertLoc.isInvalid())
      return NullabilitySkipReason::EmissionFailed;
    bool Invalid = false;
    const char *NextChar = SM.getCharacterData(InsertLoc, &Invalid);
    if (Invalid)
      return NullabilitySkipReason::EmissionFailed;

    std::string Text = LeadingSpace ? " " : "";
    Text += Keyword;
    if (isAsciiIdentifierContinue(*NextChar))
      Text += " ";
    tooling::Replacement Repl(SM, InsertLoc, 0, Text);
    if (!Repl.isApplicable())
      return NullabilitySkipReason::EmissionFailed;
    Edits.addReplacement(std::move(Repl));
    return std::nullopt;
  }

  void report(TypeLoc TL, llvm::StringRef RuleId, llvm::StringRef Message) {
    CharSourceRange Range = Lexer::getAsCharRange(
        CharSourceRange::getTokenRange(TL.getSourceRange()), SM,
        Ctx.getLangOpts());
    Report.addResult(RuleId, SarifResultLevel::Note, Range, Message);
  }

  ASTContext &Ctx;
  const SourceManager &SM;
  const InferenceMap &Inference;
  NestedBuildNamespace TUNamespace;
  NestedBuildNamespace LUNamespace;
  SourceEditEmitter &Edits;
  TransformationReportEmitter &Report;
};

} // namespace

namespace clang::ssaf {

llvm::StringRef messageFor(NullabilitySkipReason Reason) {
  switch (Reason) {
  case NullabilitySkipReason::EmissionFailed:
    return "no source edit could be formed for this declarator";
  case NullabilitySkipReason::InnerPointerLevel:
    return "nullability was inferred for an inner pointer level, which is not "
           "yet annotated";
  case NullabilitySkipReason::MacroExpansion:
    return "pointer spelled through a macro is not annotated";
  case NullabilitySkipReason::NoPointerSpelling:
    return "the declared type is not spelled with a '*' or a typedef name";
  }
  llvm_unreachable("unhandled NullabilitySkipReason");
}

void NullabilityAnnotations::HandleTranslationUnit(ASTContext &Ctx) {
  auto Inferred = Suite.get<NullabilityInferenceAnalysisResult>();
  if (!Inferred) {
    llvm::consumeError(Inferred.takeError());
    return;
  }
  InferenceMap Inference(Suite, *Inferred);
  NestedBuildNamespace TUNamespace =
      NestedBuildNamespace::makeCompilationUnit(Opts.CompilationUnitId);
  NestedBuildNamespace LUNamespace =
      NestedBuildNamespace::makeLinkUnit(Opts.LinkUnitId);
  AnnotateVisitor(Ctx, Inference, TUNamespace, LUNamespace, Edits, Report)
      .TraverseDecl(Ctx.getTranslationUnitDecl());

  const SourceManager &SM = Ctx.getSourceManager();
  NullStoreCollector Collector;
  runNullabilitySafetyOnTU(
      Ctx.getTranslationUnitDecl(), Collector,
      NullabilitySafetyOptions::fromLangOptions(Ctx.getLangOpts()),
      [&](const Decl *Def) {
        return !SM.isInSystemHeader(Def->getLocation());
      });
  for (const auto &[Store, Field] : Collector.Stores) {
    if (SM.isInSystemHeader(Field->getLocation()) ||
        !Inference
             .factsFor(getQualifiedEntityName(Field, TUNamespace, LUNamespace))
             .Nullable)
      continue;
    CharSourceRange Range = Lexer::getAsCharRange(
        CharSourceRange::getTokenRange(Store->getSourceRange()), SM,
        Ctx.getLangOpts());
    Report.addResult(NullStoreRuleId, SarifResultLevel::Note, Range,
                     ("stores null into field '" + Field->getName() +
                      "', which is left unannotated and treated as non-null; "
                      "check that it is not read while null")
                         .str());
  }
}

} // namespace clang::ssaf

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int NullabilityAnnotationsAnchorSource = 0;
} // namespace clang::ssaf

static clang::ssaf::TransformationRegistry::Add<NullabilityAnnotations>
    RegisterNullabilityAnnotations(
        "nullability-annotations",
        "Inserts inferred _Nonnull annotations (experimental)");
