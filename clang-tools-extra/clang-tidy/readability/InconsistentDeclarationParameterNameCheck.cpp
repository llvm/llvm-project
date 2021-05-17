//===--- InconsistentDeclarationParameterNameCheck.cpp - clang-tidy-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InconsistentDeclarationParameterNameCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include <algorithm>
#include <functional>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

AST_MATCHER(FunctionDecl, hasOtherDeclarations) {
  auto It = Node.redecls_begin();
  auto EndIt = Node.redecls_end();

  if (It == EndIt)
    return false;

  ++It;
  return It != EndIt;
}

struct DifferingParamInfo {
  DifferingParamInfo(StringRef SourceName, StringRef OtherName,
                     SourceRange OtherNameRange, bool GenerateFixItHint)
      : SourceName(SourceName), OtherName(OtherName),
        OtherNameRange(OtherNameRange), GenerateFixItHint(GenerateFixItHint) {}

  StringRef SourceName;
  StringRef OtherName;
  SourceRange OtherNameRange;
  bool GenerateFixItHint;
};

using DifferingParamsContainer = llvm::SmallVector<DifferingParamInfo, 10>;

struct InconsistentDeclarationInfo {
  InconsistentDeclarationInfo(SourceLocation DeclarationLocation,
                              DifferingParamsContainer &&DifferingParams)
      : DeclarationLocation(DeclarationLocation),
        DifferingParams(std::move(DifferingParams)) {}

  SourceLocation DeclarationLocation;
  DifferingParamsContainer DifferingParams;
};

using InconsistentDeclarationsContainer =
    llvm::SmallVector<InconsistentDeclarationInfo, 2>;

bool checkIfFixItHintIsApplicable(
    const FunctionDecl *ParameterSourceDeclaration,
    const ParmVarDecl *SourceParam, const FunctionDecl *OriginalDeclaration) {
  // Assumptions with regard to function declarations/definition:
  //  * If both function declaration and definition are seen, assume that
  //    definition is most up-to-date, and use it to generate replacements.
  //  * If only function declarations are seen, there is no easy way to tell
  //    which is up-to-date and which is not, so don't do anything.
  // TODO: This may be changed later, but for now it seems the reasonable
  // solution.
  if (!ParameterSourceDeclaration->isThisDeclarationADefinition())
    return false;

  // Assumption: if parameter is not referenced in function definition body, it
  // may indicate that it's outdated, so don't touch it.
  if (!SourceParam->isReferenced())
    return false;

  // In case there is the primary template definition and (possibly several)
  // template specializations (and each with possibly several redeclarations),
  // it is not at all clear what to change.
  if (OriginalDeclaration->getTemplatedKind() ==
      FunctionDecl::TK_FunctionTemplateSpecialization)
    return false;

  // Other cases seem OK to allow replacements.
  return true;
}

bool nameMatch(StringRef L, StringRef R, bool Strict) {
  if (Strict)
    return L.empty() || R.empty() || L == R;
  // We allow two names if one is a prefix/suffix of the other, ignoring case.
  // Important special case: this is true if either parameter has no name!
  return L.startswith_lower(R) || R.startswith_lower(L) ||
         L.endswith_lower(R) || R.endswith_lower(L);
}

DifferingParamsContainer
findDifferingParamsInDeclaration(const FunctionDecl *ParameterSourceDeclaration,
                                 const FunctionDecl *OtherDeclaration,
                                 const FunctionDecl *OriginalDeclaration,
                                 bool Strict) {
  DifferingParamsContainer DifferingParams;

  auto SourceParamIt = ParameterSourceDeclaration->param_begin();
  auto OtherParamIt = OtherDeclaration->param_begin();

  while (SourceParamIt != ParameterSourceDeclaration->param_end() &&
         OtherParamIt != OtherDeclaration->param_end()) {
    auto SourceParamName = (*SourceParamIt)->getName();
    auto OtherParamName = (*OtherParamIt)->getName();

    // FIXME: Provide a way to extract commented out parameter name from comment
    // next to it.
    if (!nameMatch(SourceParamName, OtherParamName, Strict)) {
      SourceRange OtherParamNameRange =
          DeclarationNameInfo((*OtherParamIt)->getDeclName(),
                              (*OtherParamIt)->getLocation())
              .getSourceRange();

      bool GenerateFixItHint = checkIfFixItHintIsApplicable(
          ParameterSourceDeclaration, *SourceParamIt, OriginalDeclaration);

      DifferingParams.emplace_back(SourceParamName, OtherParamName,
                                   OtherParamNameRange, GenerateFixItHint);
    }

    ++SourceParamIt;
    ++OtherParamIt;
  }

  return DifferingParams;
}

InconsistentDeclarationsContainer
findInconsistentDeclarations(const FunctionDecl *OriginalDeclaration,
                            const FunctionDecl *ParameterSourceDeclaration,
                            SourceManager &SM, bool Strict) {
  InconsistentDeclarationsContainer InconsistentDeclarations;
  SourceLocation ParameterSourceLocation =
      ParameterSourceDeclaration->getLocation();

  for (const FunctionDecl *OtherDeclaration : OriginalDeclaration->redecls()) {
    SourceLocation OtherLocation = OtherDeclaration->getLocation();
    if (OtherLocation != ParameterSourceLocation) { // Skip self.
      DifferingParamsContainer DifferingParams =
          findDifferingParamsInDeclaration(ParameterSourceDeclaration,
                                           OtherDeclaration,
                                           OriginalDeclaration, Strict);
      if (!DifferingParams.empty()) {
        InconsistentDeclarations.emplace_back(OtherDeclaration->getLocation(),
                                              std::move(DifferingParams));
      }
    }
  }

  // Sort in order of appearance in translation unit to generate clear
  // diagnostics.
  std::sort(InconsistentDeclarations.begin(), InconsistentDeclarations.end(),
            [&SM](const InconsistentDeclarationInfo &Info1,
                  const InconsistentDeclarationInfo &Info2) {
              return SM.isBeforeInTranslationUnit(Info1.DeclarationLocation,
                                                  Info2.DeclarationLocation);
            });
  return InconsistentDeclarations;
}

const FunctionDecl *
getParameterSourceDeclaration(const FunctionDecl *OriginalDeclaration) {
  const FunctionTemplateDecl *PrimaryTemplate =
      OriginalDeclaration->getPrimaryTemplate();
  if (PrimaryTemplate != nullptr) {
    // In case of template specializations, use primary template declaration as
    // the source of parameter names.
    return PrimaryTemplate->getTemplatedDecl();
  }

  // In other cases, try to change to function definition, if available.

  if (OriginalDeclaration->isThisDeclarationADefinition())
    return OriginalDeclaration;

  for (const FunctionDecl *OtherDeclaration : OriginalDeclaration->redecls()) {
    if (OtherDeclaration->isThisDeclarationADefinition()) {
      return OtherDeclaration;
    }
  }

  // No definition found, so return original declaration.
  return OriginalDeclaration;
}

std::string joinParameterNames(
    const DifferingParamsContainer &DifferingParams,
    llvm::function_ref<StringRef(const DifferingParamInfo &)> ChooseParamName) {
  llvm::SmallString<40> Str;
  bool First = true;
  for (const DifferingParamInfo &ParamInfo : DifferingParams) {
    if (First)
      First = false;
    else
      Str += ", ";
    Str.append({"'", ChooseParamName(ParamInfo), "'"});
  }
  return std::string(Str);
}

void formatDifferingParamsDiagnostic(
    InconsistentDeclarationParameterNameCheck *Check, SourceLocation Location,
    StringRef OtherDeclarationDescription,
    const DifferingParamsContainer &DifferingParams) {
  auto ChooseOtherName = [](const DifferingParamInfo &ParamInfo) {
    return ParamInfo.OtherName;
  };
  auto ChooseSourceName = [](const DifferingParamInfo &ParamInfo) {
    return ParamInfo.SourceName;
  };

  auto ParamDiag =
      Check->diag(Location,
                  "differing parameters are named here: (%0), in %1: (%2)",
                  DiagnosticIDs::Level::Note)
      << joinParameterNames(DifferingParams, ChooseOtherName)
      << OtherDeclarationDescription
      << joinParameterNames(DifferingParams, ChooseSourceName);

  for (const DifferingParamInfo &ParamInfo : DifferingParams) {
    if (ParamInfo.GenerateFixItHint) {
      ParamDiag << FixItHint::CreateReplacement(
          CharSourceRange::getTokenRange(ParamInfo.OtherNameRange),
          ParamInfo.SourceName);
    }
  }
}

void formatDiagnosticsForDeclarations(
    InconsistentDeclarationParameterNameCheck *Check,
    const FunctionDecl *ParameterSourceDeclaration,
    const FunctionDecl *OriginalDeclaration,
    const InconsistentDeclarationsContainer &InconsistentDeclarations) {
  Check->diag(
      OriginalDeclaration->getLocation(),
      "function %q0 has %1 other declaration%s1 with different parameter names")
      << OriginalDeclaration
      << static_cast<int>(InconsistentDeclarations.size());
  int Count = 1;
  for (const InconsistentDeclarationInfo &InconsistentDeclaration :
       InconsistentDeclarations) {
    Check->diag(InconsistentDeclaration.DeclarationLocation,
                "the %ordinal0 inconsistent declaration seen here",
                DiagnosticIDs::Level::Note)
        << Count;

    formatDifferingParamsDiagnostic(
        Check, InconsistentDeclaration.DeclarationLocation,
        "the other declaration", InconsistentDeclaration.DifferingParams);

    ++Count;
  }
}

void formatDiagnostics(
    InconsistentDeclarationParameterNameCheck *Check,
    const FunctionDecl *ParameterSourceDeclaration,
    const FunctionDecl *OriginalDeclaration,
    const InconsistentDeclarationsContainer &InconsistentDeclarations,
    StringRef FunctionDescription, StringRef ParameterSourceDescription) {
  for (const InconsistentDeclarationInfo &InconsistentDeclaration :
       InconsistentDeclarations) {
    Check->diag(InconsistentDeclaration.DeclarationLocation,
                "%0 %q1 has a %2 with different parameter names")
        << FunctionDescription << OriginalDeclaration
        << ParameterSourceDescription;

    Check->diag(ParameterSourceDeclaration->getLocation(), "the %0 seen here",
                DiagnosticIDs::Level::Note)
        << ParameterSourceDescription;

    formatDifferingParamsDiagnostic(
        Check, InconsistentDeclaration.DeclarationLocation,
        ParameterSourceDescription, InconsistentDeclaration.DifferingParams);
  }
}

} // anonymous namespace

void InconsistentDeclarationParameterNameCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
  Options.store(Opts, "Strict", Strict);
}

void InconsistentDeclarationParameterNameCheck::registerMatchers(
    MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(hasOtherDeclarations()).bind("functionDecl"),
                     this);
}

void InconsistentDeclarationParameterNameCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *OriginalDeclaration =
      Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");

  if (VisitedDeclarations.count(OriginalDeclaration) > 0)
    return; // Avoid multiple warnings.

  const FunctionDecl *ParameterSourceDeclaration =
      getParameterSourceDeclaration(OriginalDeclaration);

  InconsistentDeclarationsContainer InconsistentDeclarations =
      findInconsistentDeclarations(OriginalDeclaration,
                                   ParameterSourceDeclaration,
                                   *Result.SourceManager, Strict);
  if (InconsistentDeclarations.empty()) {
    // Avoid unnecessary further visits.
    markRedeclarationsAsVisited(OriginalDeclaration);
    return;
  }

  SourceLocation StartLoc = OriginalDeclaration->getBeginLoc();
  if (StartLoc.isMacroID() && IgnoreMacros) {
    markRedeclarationsAsVisited(OriginalDeclaration);
    return;
  }

  if (OriginalDeclaration->getTemplatedKind() ==
      FunctionDecl::TK_FunctionTemplateSpecialization) {
    formatDiagnostics(this, ParameterSourceDeclaration, OriginalDeclaration,
                      InconsistentDeclarations,
                      "function template specialization",
                      "primary template declaration");
  } else if (ParameterSourceDeclaration->isThisDeclarationADefinition()) {
    formatDiagnostics(this, ParameterSourceDeclaration, OriginalDeclaration,
                      InconsistentDeclarations, "function", "definition");
  } else {
    formatDiagnosticsForDeclarations(this, ParameterSourceDeclaration,
                                     OriginalDeclaration,
                                     InconsistentDeclarations);
  }

  markRedeclarationsAsVisited(OriginalDeclaration);
}

void InconsistentDeclarationParameterNameCheck::markRedeclarationsAsVisited(
    const FunctionDecl *OriginalDeclaration) {
  for (const FunctionDecl *Redecl : OriginalDeclaration->redecls()) {
    VisitedDeclarations.insert(Redecl);
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
