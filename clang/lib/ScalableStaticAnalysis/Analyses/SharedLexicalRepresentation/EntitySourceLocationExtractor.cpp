//===- EntitySourceLocationExtractor.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/ScalableStaticAnalysis/Analyses/SharedLexicalRepresentation/SharedLexicalRepresentation.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/Path.h"

#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace clang::ssaf {
extern EntitySourceLocationsSummary
buildEntitySourceLocationsSummary(std::vector<SourceLocationRecord> Locs);
} // namespace clang::ssaf

namespace {
using namespace clang;
using namespace ssaf;

/// Per-EntityId aggregation buffer used during one TU walk. Each visited
/// declaration contributes one record; the buffer is committed once per
/// EntityId at the end of the walk.
using LocationMap = std::map<EntityId, std::vector<SourceLocationRecord>>;

/// Compute a SourceLocationRecord from \p Loc. Returns std::nullopt for
/// invalid / system-header / unreachable-on-disk paths; these are silently
/// dropped per the spec.
std::optional<SourceLocationRecord> makeRecord(SourceLocation Loc,
                                               const SourceManager &SM) {
  if (!Loc.isValid() || SM.isInSystemHeader(Loc))
    return std::nullopt;

  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  if (PLoc.isInvalid())
    return std::nullopt;

  llvm::StringRef Filename(PLoc.getFilename());
  if (Filename.empty())
    return std::nullopt;

  // Path canonicalization touches the filesystem, which is denied under the
  // sandboxed compile drivers (e.g. xcodebuild). Reading source files the
  // compiler is already reading is benign, so disable the sandbox for the
  // duration of these calls.
  llvm::SmallString<256> Abs(Filename);
  llvm::SmallString<256> Real;
  {
    auto SandboxOff = llvm::sys::sandbox::scopedDisable();
    if (!llvm::sys::path::is_absolute(Abs))
      if (auto EC = llvm::sys::fs::make_absolute(Abs))
        return std::nullopt;
    if (auto EC = llvm::sys::fs::real_path(Abs, Real))
      return std::nullopt;
  }

  SourceLocationRecord R;
  R.FilePath = Real.str().str();
  R.Line = PLoc.getLine();
  R.Column = PLoc.getColumn();
  return R;
}

void handleNamedDecl(TUSummaryExtractor &Extractor, const NamedDecl *D,
                     SourceLocation Loc, const SourceManager &SM,
                     LocationMap &Records) {
  auto Rec = makeRecord(Loc, SM);
  if (!Rec)
    return;
  std::optional<EntityId> Id = Extractor.addEntity(D);
  if (!Id)
    return;
  Records[*Id].push_back(std::move(*Rec));
}

void handleFunction(TUSummaryExtractor &Extractor, const FunctionDecl *FD,
                    const SourceManager &SM, LocationMap &Records) {
  handleNamedDecl(Extractor, FD, FD->getLocation(), SM, Records);

  if (auto RetRec = makeRecord(FD->getReturnTypeSourceRange().getBegin(), SM))
    if (auto RetId = Extractor.addEntityForReturn(FD))
      Records[*RetId].push_back(std::move(*RetRec));

  // The variadic '...' terminator is not represented by a ParmVarDecl in the
  // AST, so this loop naturally skips it.
  for (const ParmVarDecl *P : FD->parameters()) {
    if (!P)
      continue;
    handleNamedDecl(Extractor, P, P->getTypeSpecStartLoc(), SM, Records);
  }
}

/// AST visitor that records per-entity declaration source-locations as it
/// walks. ParmVarDecls are visited as part of their parent FunctionDecl so
/// the parameter-type-spec-start anchor is recorded once per redeclaration of
/// the parent function.
class EntityVisitor : public DynamicRecursiveASTVisitor {
  TUSummaryExtractor &Extractor;
  const SourceManager &SM;
  LocationMap &Records;

public:
  EntityVisitor(TUSummaryExtractor &Extractor, const SourceManager &SM,
                LocationMap &Records)
      : Extractor(Extractor), SM(SM), Records(Records) {
    ShouldVisitTemplateInstantiations = true;
    ShouldVisitImplicitCode = false;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) override {
    if (FD)
      handleFunction(Extractor, FD, SM, Records);
    return true;
  }

  bool VisitVarDecl(VarDecl *VD) override {
    if (VD && !isa<ParmVarDecl>(VD))
      handleNamedDecl(Extractor, VD, VD->getLocation(), SM, Records);
    return true;
  }

  bool VisitFieldDecl(FieldDecl *FD) override {
    if (FD)
      handleNamedDecl(Extractor, FD, FD->getLocation(), SM, Records);
    return true;
  }

  bool VisitRecordDecl(RecordDecl *RD) override {
    if (RD)
      handleNamedDecl(Extractor, RD, RD->getLocation(), SM, Records);
    return true;
  }
};

class EntitySourceLocationExtractor final : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;

private:
  void HandleTranslationUnit(ASTContext &Ctx) override;
};

void EntitySourceLocationExtractor::HandleTranslationUnit(ASTContext &Ctx) {
  const SourceManager &SM = Ctx.getSourceManager();
  LocationMap Records;
  EntityVisitor(*this, SM, Records).TraverseAST(Ctx);

  for (auto &[Id, Locs] : Records) {
    auto Summary = std::make_unique<EntitySourceLocationsSummary>(
        buildEntitySourceLocationsSummary(std::move(Locs)));
    [[maybe_unused]] auto [Ignored, Inserted] =
        SummaryBuilder.addSummary(Id, std::move(Summary));
    assert(Inserted &&
           "EntitySourceLocations summary inserted twice for same EntityId");
  }
}

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int EntitySourceLocationExtractorAnchorSource = 0;
} // namespace clang::ssaf

static clang::ssaf::TUSummaryExtractorRegistry::Add<
    EntitySourceLocationExtractor>
    RegisterExtractor(EntitySourceLocationsSummary::Name,
                      "Extract per-entity declaration source-locations");
