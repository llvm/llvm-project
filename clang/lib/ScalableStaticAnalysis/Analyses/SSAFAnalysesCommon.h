//===- SSAFAnalysesCommon.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Common code in SSAF analyses implementations
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_SSAFANALYSESCOMMON_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_SSAFANALYSESCOMMON_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace clang::ssaf {
///\return a short descriptions of a json::Value
std::string describeJSONValue(const llvm::json::Value &V);
///\return a short descriptions of a json::Array
std::string describeJSONValue(const llvm::json::Array &A);
///\return a short descriptions of a json::Object
std::string describeJSONValue(const llvm::json::Object &O);

template <typename NodeTy, typename... Ts>
llvm::Error makeErrAtNode(clang::ASTContext &Ctx, const NodeTy *N,
                          llvm::StringRef Fmt, const Ts &...Args) {
  std::string LocStr = N->getBeginLoc().printToString(Ctx.getSourceManager());
  return llvm::createStringError((Fmt + " at %s").str().c_str(), Args...,
                                 LocStr.c_str());
}

template <typename JSONTy, typename... Ts>
llvm::Error makeSawButExpectedError(const JSONTy &Saw, llvm::StringRef Expected,
                                    const Ts &...ExpectedArgs) {
  std::string Fmt = ("saw %s but expected " + Expected).str();
  std::string SawStr = describeJSONValue(Saw);

  return llvm::createStringError(Fmt.c_str(), SawStr.c_str(), ExpectedArgs...);
}

///\return true iff expression `E` has pointer or array type.
inline bool hasPtrOrArrType(const Expr *E) {
  return llvm::isa<clang::PointerType, clang::ArrayType>(
      E->getType().getCanonicalType());
}

///\return true iff Decl `D` has (reference-to) pointer or array type.
inline bool hasPtrOrArrType(const ValueDecl *D) {
  return llvm::isa<clang::PointerType, clang::ArrayType>(
      D->getType().getNonReferenceType().getCanonicalType());
}

llvm::Error makeEntityNameErr(clang::ASTContext &Ctx,
                              const clang::NamedDecl *D);

/// Log a warning from an llvm::Error
inline void logWarningFromError(llvm::Error Err) {
  DEBUG_WITH_TYPE("ssaf-analyses", llvm::errs() << Err);
  llvm::consumeError(std::move(Err));
}

/// Find all contributors in an AST. The found contributors are organized as a
/// map from the canonical declaration of each entity to all of its
/// declarations.
void findContributors(
    ASTContext &Ctx,
    llvm::DenseMap<const NamedDecl *, std::vector<const NamedDecl *>>
        &Contributors);

/// Perform "MatchAction" on each Stmt and Decl belonging to the `Contributor`.
/// \param Contributor
/// \param MatchActionRef a reference (view) to a "MatchAction"
void findMatchesIn(
    const NamedDecl *Contributor,
    llvm::function_ref<void(const DynTypedNode &)> MatchActionRef);

/// The standard contributor-summary extraction procedure:
///   1. Find and group all contributor decls by their canonical decls.
///   2. Use \p Extract to get an EntitySummary of a contributor from all of its
///   decls.
///   3. Insert the EntitySummary into the \p Builder.
///
/// \param ExtractorFnT the template parameter that should be a function type
/// 'std::unique_ptr<SummaryT>(std::vector<const NamedDecl *>)' for different
/// entity summary type `SummaryT`s
/// \param ExtractFn The function that extracts summaries of a contributor from
/// its decls.
/// \param ExtractorName The optional information inserted into the warning
/// message when duplicate contributor names (EntityNames) are seen.
template <typename ExtractorFnT>
void extractAndAddSummaries(TUSummaryExtractor &Extractor,
                            TUSummaryBuilder &Builder, ASTContext &Ctx,
                            ExtractorFnT ExtractFn,
                            llvm::StringRef ExtractorName = "") {
  llvm::DenseMap<const NamedDecl *, std::vector<const NamedDecl *>>
      Contributors;
  findContributors(Ctx, Contributors);
  for (const auto &[Cano, Decls] : Contributors) {
    assert(!Decls.empty() &&
           "'findContributors' guarantess that 'Decls' are non-empty");
    assert(!Decls[0]->isImplicit() &&
           "'findContributors' guarantess that 'Decls' are non-implicit");

    // Templates are skipped, but their instantiations are handled. The idea
    // is that we can conclude facts about a template through all of its
    // instantiations.
    if (Decls[0]->isTemplated())
      continue;

    auto Summary = ExtractFn(Decls);
    assert(Summary);
    if (Summary->empty())
      continue;

    if (auto Id = Extractor.addEntity(Decls[0])) {
      if (!Builder.addSummary(*Id, std::move(Summary)).second)
        logWarningFromError(makeErrAtNode(
            Ctx, Decls[0], "dropping duplicate %s summary for entity %s",
            ExtractorName.str().c_str(), Decls[0]->getNameAsString().c_str()));
    } else
      logWarningFromError(makeEntityNameErr(Ctx, Decls[0]));
  }
}

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_SSAFANALYSESCOMMON_H
