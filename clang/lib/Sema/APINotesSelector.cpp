//===--- APINotesSelector.cpp - API Notes selector helpers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/APINotesSelector.h"
#include "clang/APINotes/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"

using namespace clang;

namespace {

PrintingPolicy
getAPINotesParameterSelectorPrintingPolicy(const ASTContext &Context) {
  PrintingPolicy Policy(Context.getLangOpts());
  Policy.PrintAsCanonical = false;
  Policy.FullyQualifiedName = false;
  Policy.SuppressScope = false;
  Policy.UsePreferredNames = false;
  Policy.MSVCFormatting = false;
  Policy.SplitTemplateClosers = false;
  Policy.IncludeNewlines = false;
  return Policy;
}

// Print the APINotes selector spelling for one parameter. The source-spelled
// selector is tried first. The desugared spelling is only a permissive
// fallback.
std::string getAPINotesParameterSelectorSpelling(QualType ParamType,
                                                 const ASTContext &Context,
                                                 const PrintingPolicy &Policy,
                                                 bool Desugar) {
  if (Desugar)
    ParamType = ParamType.getDesugaredType(Context);

  ParamType.removeLocalConst();
  ParamType.removeLocalVolatile();
  ParamType = ParamType.stripNullability(Context);

  return api_notes::normalizeAPINotesParameterSelector(
      ParamType.getAsString(Policy));
}

} // namespace

std::optional<APINotesParameterSelectorCandidates>
clang::getAPINotesParameterSelectorCandidates(const ASTContext &Context,
                                              const FunctionDecl *FD) {
  const auto *FPT = FD->getType()->getAs<FunctionProtoType>();
  if (!FPT)
    return std::nullopt;

  APINotesParameterSelectorCandidates Candidates;
  APINotesParameterSelector Desugared;
  Candidates.Source.reserve(FPT->getNumParams());
  Desugared.reserve(FPT->getNumParams());

  const PrintingPolicy Policy =
      getAPINotesParameterSelectorPrintingPolicy(Context);
  for (QualType ParamType : FPT->param_types()) {
    Candidates.Source.push_back(
        getAPINotesParameterSelectorSpelling(ParamType, Context, Policy,
                                             /*Desugar=*/false));
    Desugared.push_back(getAPINotesParameterSelectorSpelling(
        ParamType, Context, Policy, /*Desugar=*/true));
  }

  if (Candidates.Source != Desugared)
    Candidates.Desugared = std::move(Desugared);

  return Candidates;
}
