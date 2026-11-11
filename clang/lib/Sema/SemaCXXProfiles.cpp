//===------ SemaCXXProfiles.cpp --- C++ profiles (P3589) ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for C++ profiles (P3589).
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaCXXProfiles.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include <algorithm>

using namespace clang;

SemaCXXProfiles::SemaCXXProfiles(Sema &S) : SemaBase(S) {}

/// Returns the name of the diagnostic group that contains the rules of
/// \p Profile, e.g. "cxx-profile-std-casting" for "std::casting".
static std::string getProfileDiagGroupName(CXXProfileID Profile) {
  std::string Group = "cxx-profile-";
  StringRef Name = getCXXProfileName(Profile);
  for (size_t I = 0, E = Name.size(); I != E; ++I) {
    if (Name.substr(I).starts_with("::")) {
      Group += '-';
      ++I;
    } else if (Name[I] == '_') {
      Group += '-';
    } else {
      Group += Name[I];
    }
  }
  return Group;
}

/// Returns the template (or class template member) that \p D was instantiated
/// from, or \p D itself.
static const Decl *getPatternDecl(const Decl *D) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    if (const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern())
      return Pattern;
  if (const auto *RD = dyn_cast<CXXRecordDecl>(D))
    if (const CXXRecordDecl *Pattern = RD->getTemplateInstantiationPattern())
      return Pattern;
  if (const auto *VD = dyn_cast<VarDecl>(D))
    if (const VarDecl *Pattern = VD->getTemplateInstantiationPattern())
      return Pattern;
  return D;
}

/// Returns true if \p Range (a declaration or statement range, possibly
/// spanning macro expansions) contains \p Loc. A range without an end is
/// that of a declaration that is still being parsed (e.g. a namespace), and
/// contains every location after its beginning.
static bool rangeContains(const SourceManager &SM, SourceRange Range,
                          SourceLocation Loc) {
  if (Range.getBegin().isInvalid() || Loc.isInvalid())
    return false;
  SourceLocation Begin = SM.getExpansionLoc(Range.getBegin());
  SourceLocation Point = SM.getExpansionLoc(Loc);
  if (SM.isBeforeInTranslationUnit(Point, Begin))
    return false;
  if (Range.getEnd().isInvalid())
    return true;
  SourceLocation End = SM.getExpansionRange(Range.getEnd()).getEnd();
  return !SM.isBeforeInTranslationUnit(End, Point);
}

/// Returns true if the file \p Path is the header named by \p Header, i.e.
/// \p Header matches the trailing path components of \p Path.
static bool pathMatchesHeader(StringRef Path, StringRef Header) {
  std::string Normalized = Path.str();
  std::replace(Normalized.begin(), Normalized.end(), '\\', '/');
  StringRef P = Normalized;
  if (Header.empty() || !P.ends_with(Header))
    return false;
  return P.size() == Header.size() || P[P.size() - Header.size() - 1] == '/';
}

/// Returns the spelling of argument \p I of a profiles attribute: the
/// spelling of an identifier argument, or the contents of a string literal.
static StringRef getArgSpelling(const ParsedAttr &AL, unsigned I) {
  if (AL.isArgIdent(I))
    return AL.getArgAsIdent(I)->getIdentifierInfo()->getName();
  if (const auto *SL = dyn_cast<StringLiteral>(AL.getArgAsExpr(I)))
    return SL->getString();
  return StringRef();
}

static SourceLocation getArgLoc(const ParsedAttr &AL, unsigned I) {
  if (AL.isArgIdent(I))
    return AL.getArgAsIdent(I)->getLoc();
  return AL.getArgAsExpr(I)->getBeginLoc();
}

namespace {
/// The arguments of [[profiles::suppress]] and [[profiles::exempt]].
struct NamedProfileArgs {
  StringRef Profile;
  SourceLocation ProfileLoc;
  StringRef Rule;
  SourceLocation RuleLoc;
  StringRef Justification;
  StringRef Header;
  bool Angled = false;
  bool HasHeader = false;
};
} // namespace

/// Parse the arguments of [[profiles::suppress]] (if \p IsExempt is false) or
/// [[profiles::exempt]]. Returns std::nullopt if the attribute is invalid.
static std::optional<NamedProfileArgs>
parseNamedProfileArgs(Sema &S, const ParsedAttr &AL, bool IsExempt) {
  if (AL.getNumArgs() == 0 || !AL.isArgIdent(0)) {
    S.Diag(AL.getLoc(), diag::err_cxx_profiles_attr_requires_profile)
        << AL << /*a profile name*/ 1;
    return std::nullopt;
  }

  NamedProfileArgs Args;
  Args.Profile = getArgSpelling(AL, 0);
  Args.ProfileLoc = getArgLoc(AL, 0);

  // The remaining arguments are (name, value) pairs; see
  // Parser::ParseCXXProfilesAttributeArgs.
  for (unsigned I = 1; I + 1 < AL.getNumArgs(); I += 2) {
    if (!AL.isArgIdent(I))
      continue;
    StringRef Key = getArgSpelling(AL, I);
    SourceLocation KeyLoc = getArgLoc(AL, I);
    StringRef Value = getArgSpelling(AL, I + 1);
    bool IsString = AL.isArgExpr(I + 1);

    if (Key == "justification") {
      if (!IsString) {
        S.Diag(KeyLoc, diag::err_cxx_profiles_argument_requires_string)
            << Key << AL;
        return std::nullopt;
      }
      Args.Justification = Value;
    } else if (!IsExempt && Key == "rule") {
      Args.Rule = Value;
      Args.RuleLoc = KeyLoc;
    } else if (IsExempt && (Key == "angle_header" || Key == "quote_header")) {
      if (!IsString) {
        S.Diag(KeyLoc, diag::err_cxx_profiles_argument_requires_string)
            << Key << AL;
        return std::nullopt;
      }
      Args.Header = Value;
      Args.Angled = Key == "angle_header";
      Args.HasHeader = true;
    } else {
      S.Diag(KeyLoc, diag::warn_cxx_profiles_unknown_argument) << Key << AL;
    }
  }
  return Args;
}

void SemaCXXProfiles::ActOnStartOfTranslationUnit() {
  if (Initialized)
    return;
  Initialized = true;

  const LangOptions &LangOpts = getLangOpts();
  if (!LangOpts.CXXProfiles) {
    if (!LangOpts.CXXProfilesEnforced.empty())
      Diag(SourceLocation(), diag::err_cxx_profiles_not_enabled) << 0;
    if (!LangOpts.CXXProfilesPreview.empty())
      Diag(SourceLocation(), diag::err_cxx_profiles_not_enabled) << 1;
    return;
  }

  for (const std::string &Designator : LangOpts.CXXProfilesEnforced)
    addEnforcedProfile(Designator, SourceLocation(), UnitProfiles);

  for (const std::string &Designator : LangOpts.CXXProfilesPreview) {
    StringRef Name = getCXXProfileDesignatorName(Designator);
    std::optional<CXXProfileID> ID = lookupCXXProfile(Name);
    if (!ID) {
      Diag(SourceLocation(), diag::warn_cxx_profiles_unknown_profile)
          << Name << /*will not be enforced*/ 0;
      continue;
    }
    if (llvm::is_contained(PreviewProfiles, *ID))
      continue;
    PreviewProfiles.push_back(*ID);
    // Preview the rules of a profile that is not enforced as warnings.
    if (llvm::none_of(UnitProfiles,
                      [&](const EnforcedProfile &E) { return E.ID == *ID; }))
      getDiagnostics().setSeverityForGroup(diag::Flavor::WarningOrError,
                                           getProfileDiagGroupName(*ID),
                                           diag::Severity::Warning);
  }
}

void SemaCXXProfiles::ActOnEndOfTranslationUnit() {
  flushPendingViolations();

  // A header unit has no module declaration; its module is the current
  // module for the whole translation unit.
  if (!UnitModule)
    if (Module *M = SemaRef.getCurrentModule(); M && M->isHeaderUnit())
      UnitModule = M;
  updateModuleRecord();
}

void SemaCXXProfiles::updateModuleRecord() {
  if (!UnitModule)
    return;
  UnitModule->CXXProfilesUnitEnforced.clear();
  for (const EnforcedProfile &E : UnitProfiles)
    UnitModule->CXXProfilesUnitEnforced.push_back(E.Designator);
}

bool SemaCXXProfiles::addEnforcedProfile(StringRef Designator,
                                         SourceLocation Loc,
                                         SmallVectorImpl<EnforcedProfile> &List,
                                         bool Diagnose) {
  StringRef Name = getCXXProfileDesignatorName(Designator);
  std::optional<CXXProfileID> ID = lookupCXXProfile(Name);
  if (!ID && Diagnose)
    Diag(Loc, diag::warn_cxx_profiles_unknown_profile)
        << Name << /*will not be enforced*/ 0;

  // Enforcing the same designator again is a no-op; enforcing the same
  // profile with a different designator is an error.
  for (const SmallVectorImpl<EnforcedProfile> *Existing :
       {&UnitProfiles, &ModuleProfiles}) {
    for (const EnforcedProfile &E : *Existing) {
      if (getCXXProfileDesignatorName(E.Designator) != Name)
        continue;
      if (E.Designator == Designator)
        return false;
      if (Diagnose) {
        Diag(Loc, diag::err_cxx_profiles_enforce_mismatch)
            << Name << E.Designator;
        if (E.Loc.isValid())
          Diag(E.Loc, diag::note_cxx_profiles_previous_enforce) << Name;
        else
          Diag(Loc, diag::note_cxx_profiles_enforced_on_command_line) << Name;
      }
      return false;
    }
  }

  List.push_back({ID, Designator.str(), Loc});
  return true;
}

const Decl *SemaCXXProfiles::findPrecedingNonEmptyDecl(const Decl *D) const {
  for (const Decl *Other : getASTContext().getTranslationUnitDecl()->decls()) {
    if (Other == D)
      break;
    if (isa<EmptyDecl>(Other) || Other->isImplicit())
      continue;
    // The implicit import of the primary module interface by a module
    // implementation unit is not written in the source.
    if (const auto *Import = dyn_cast<ImportDecl>(Other);
        Import && PrimaryInterface &&
        Import->getImportedModule() == PrimaryInterface)
      continue;
    return Other;
  }
  return nullptr;
}

void SemaCXXProfiles::handleEnforceAttr(Decl *D, const ParsedAttr &AL) {
  if (AL.getNumArgs() == 0) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_attr_requires_profile)
        << AL << /*at least one profile designator*/ 0;
    return;
  }
  if (!D->getDeclContext()->isTranslationUnit()) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_enforce_not_at_tu_scope) << AL;
    return;
  }
  if (const Decl *First = findPrecedingNonEmptyDecl(D)) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_enforce_after_decl) << AL;
    Diag(First->getLocation(), diag::note_cxx_profiles_first_decl);
    return;
  }

  SmallVector<StringRef, 2> Designators;
  for (unsigned I = 0, E = AL.getNumArgs(); I != E; ++I) {
    if (!AL.isArgIdent(I))
      continue;
    StringRef Designator = getArgSpelling(AL, I);
    Designators.push_back(Designator);
    addEnforcedProfile(Designator, getArgLoc(AL, I), UnitProfiles);
  }
  updateModuleRecord();

  ASTContext &Ctx = getASTContext();
  D->addAttr(::new (Ctx) ProfilesEnforceAttr(Ctx, AL, Designators.data(),
                                             Designators.size()));
}

std::optional<SemaCXXProfiles::SuppressionSite>
SemaCXXProfiles::makeSuppressionSite(
    const ProfilesSuppressAttr *A,
    llvm::PointerUnion<const Decl *, const Stmt *> Node) const {
  std::optional<CXXProfileID> Profile = lookupCXXProfile(A->getProfileName());
  if (!Profile)
    return std::nullopt;
  std::optional<CXXProfileRuleID> Rule;
  if (!A->getRule().empty()) {
    Rule = lookupCXXProfileRule(*Profile, A->getRule());
    if (!Rule)
      return std::nullopt;
  }
  return SuppressionSite{*Profile, Rule, Node};
}

/// Diagnose an unknown profile or rule named by a suppression.
static void diagnoseSuppressionNames(Sema &S, const NamedProfileArgs &Args) {
  std::optional<CXXProfileID> Profile = lookupCXXProfile(Args.Profile);
  if (!Profile) {
    S.Diag(Args.ProfileLoc, diag::warn_cxx_profiles_unknown_profile)
        << Args.Profile << /*no suffix*/ 1;
    return;
  }
  if (!Args.Rule.empty() && !lookupCXXProfileRule(*Profile, Args.Rule))
    S.Diag(Args.RuleLoc, diag::warn_cxx_profiles_unknown_rule)
        << Args.Rule << Args.Profile;
}

void SemaCXXProfiles::handleSuppressAttr(Decl *D, const ParsedAttr &AL) {
  std::optional<NamedProfileArgs> Args =
      parseNamedProfileArgs(SemaRef, AL, /*IsExempt=*/false);
  if (!Args)
    return;
  diagnoseSuppressionNames(SemaRef, *Args);

  ASTContext &Ctx = getASTContext();
  auto *A = ::new (Ctx) ProfilesSuppressAttr(Ctx, AL, Args->Profile, Args->Rule,
                                             Args->Justification);
  D->addAttr(A);
  if (std::optional<SuppressionSite> Site = makeSuppressionSite(A, D))
    SuppressionSites.push_back(*Site);
}

Attr *SemaCXXProfiles::handleSuppressStmtAttr(Stmt *St, const ParsedAttr &AL) {
  std::optional<NamedProfileArgs> Args =
      parseNamedProfileArgs(SemaRef, AL, /*IsExempt=*/false);
  if (!Args)
    return nullptr;
  diagnoseSuppressionNames(SemaRef, *Args);

  ASTContext &Ctx = getASTContext();
  auto *A = ::new (Ctx) ProfilesSuppressAttr(Ctx, AL, Args->Profile, Args->Rule,
                                             Args->Justification);
  if (std::optional<SuppressionSite> Site = makeSuppressionSite(A, St))
    SuppressionSites.push_back(*Site);
  return A;
}

void SemaCXXProfiles::handleExemptAttr(Decl *D, const ParsedAttr &AL) {
  std::optional<NamedProfileArgs> Args =
      parseNamedProfileArgs(SemaRef, AL, /*IsExempt=*/true);
  if (!Args)
    return;
  if (!D->getDeclContext()->isTranslationUnit()) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_enforce_not_at_tu_scope) << AL;
    return;
  }
  if (!Args->HasHeader) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_exempt_requires_header) << AL;
    return;
  }

  std::optional<CXXProfileID> Profile = lookupCXXProfile(Args->Profile);
  if (!Profile)
    Diag(Args->ProfileLoc, diag::warn_cxx_profiles_unknown_profile)
        << Args->Profile << /*no suffix*/ 1;
  else
    ExemptHeaders.push_back({*Profile, Args->Header.str()});

  ASTContext &Ctx = getASTContext();
  D->addAttr(::new (Ctx) ProfilesExemptAttr(
      Ctx, AL, Args->Profile, Args->Header, Args->Angled, Args->Justification));
}

void SemaCXXProfiles::handleMisplacedRequireAttr(const ParsedAttr &AL) {
  Diag(AL.getLoc(), diag::err_cxx_profiles_require_not_on_import) << AL;
}

void SemaCXXProfiles::ActOnModuleDecl(Module *Mod, Module *Interface,
                                      SourceLocation ModuleLoc) {
  UnitModule = Mod;
  ModuleDeclLoc = ModuleLoc;
  PrimaryInterface = Interface;

  // [[profiles::enforce]] on the primary module interface's module
  // declaration also applies to every implementation unit of the module.
  if (Interface)
    for (const std::string &Designator : Interface->CXXProfilesModuleEnforced)
      addEnforcedProfile(Designator, SourceLocation(), ModuleProfiles,
                         /*Diagnose=*/false);
  updateModuleRecord();
}

void SemaCXXProfiles::ActOnModuleDeclEnforceAttr(const ParsedAttr &AL) {
  if (AL.getNumArgs() == 0) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_attr_requires_profile)
        << AL << /*at least one profile designator*/ 0;
    return;
  }
  // Errors in the module declaration have already been diagnosed.
  Module *Mod = SemaRef.getCurrentModule();
  if (!Mod || !Mod->isNamedModule())
    return;

  for (unsigned I = 0, E = AL.getNumArgs(); I != E; ++I) {
    if (!AL.isArgIdent(I))
      continue;
    StringRef Designator = getArgSpelling(AL, I);
    if (addEnforcedProfile(Designator, getArgLoc(AL, I), ModuleProfiles))
      Mod->CXXProfilesModuleEnforced.push_back(Designator.str());
  }
}

void SemaCXXProfiles::ActOnModuleImportRequireAttr(ImportDecl *Import,
                                                   const ParsedAttr &AL) {
  if (AL.getNumArgs() == 0) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_attr_requires_profile)
        << AL << /*at least one profile designator*/ 0;
    return;
  }
  Module *M = Import->getImportedModule();
  if (!M)
    return;
  if (!M->isNamedModule() && !M->isHeaderUnit()) {
    Diag(AL.getLoc(), diag::err_cxx_profiles_require_not_named_module) << AL;
    return;
  }

  // A named module advertises the profiles enforced on its module
  // declaration; a header unit advertises the profiles it enforces.
  Module *Top = M->getTopLevelModule();
  const std::vector<std::string> &Advertised =
      M->isHeaderUnit() ? Top->CXXProfilesUnitEnforced
                        : Top->CXXProfilesModuleEnforced;
  for (unsigned I = 0, E = AL.getNumArgs(); I != E; ++I) {
    if (!AL.isArgIdent(I))
      continue;
    StringRef Designator = getArgSpelling(AL, I);
    if (llvm::none_of(Advertised, [&](const std::string &Other) {
          return Other == Designator;
        }))
      Diag(getArgLoc(AL, I), diag::err_cxx_profiles_require_not_enforced)
          << M->getFullModuleName() << Designator;
  }
}

bool SemaCXXProfiles::isEnforcedInCurrentUnit(CXXProfileID Profile,
                                              SourceLocation Loc) const {
  for (const EnforcedProfile &E : UnitProfiles)
    if (E.ID == Profile)
      return true;
  if (llvm::is_contained(PreviewProfiles, Profile))
    return true;
  // A profile enforced on the module declaration applies from the module
  // declaration onwards; it does not apply to the global module fragment.
  for (const EnforcedProfile &E : ModuleProfiles)
    if (E.ID == Profile)
      return ModuleDeclLoc.isInvalid() || Loc.isInvalid() ||
             !SemaRef.getSourceManager().isBeforeInTranslationUnit(
                 Loc, ModuleDeclLoc);
  return false;
}

bool SemaCXXProfiles::isEnforced(CXXProfileID Profile, SourceLocation Loc,
                                 const Decl *Owner) const {
  // Code is checked according to the profiles enforced where it is written:
  // the instantiation of a template is checked according to the profiles
  // enforced by the unit that defines the template.
  const Decl *Pattern = Owner ? getPatternDecl(Owner) : nullptr;
  if (!Pattern || !Pattern->isFromASTFile())
    return isEnforcedInCurrentUnit(Profile, Loc);

  Module *M = Pattern->getOwningModule();
  // Declarations from a precompiled header are part of this translation
  // unit.
  if (!M)
    return isEnforcedInCurrentUnit(Profile, Loc);
  // Clang modules and the global module fragments of named modules do not
  // record enforced profiles.
  if (!M->isNamedModule() && !M->isHeaderUnit())
    return false;

  Module *Top = M->getTopLevelModule();
  auto Enforces = [&](const std::vector<std::string> &Designators) {
    return llvm::any_of(Designators, [&](const std::string &Designator) {
      return lookupCXXProfile(getCXXProfileDesignatorName(Designator)) ==
             Profile;
    });
  };
  return Enforces(Top->CXXProfilesUnitEnforced) ||
         (M->isNamedModule() && Enforces(Top->CXXProfilesModuleEnforced));
}

bool SemaCXXProfiles::isExempt(CXXProfileID Profile, SourceLocation Loc) const {
  const SourceManager &SM = SemaRef.getSourceManager();
  SourceLocation ExpansionLoc = SM.getExpansionLoc(Loc);
  if (SM.isInSystemHeader(ExpansionLoc))
    return true;
  if (ExemptHeaders.empty())
    return false;

  // A [[profiles::exempt]] header exempts its own code and the code of every
  // header it includes.
  FileID FID = SM.getFileID(ExpansionLoc);
  while (FID.isValid()) {
    if (OptionalFileEntryRef File = SM.getFileEntryRefForID(FID))
      for (const ExemptHeader &Exempt : ExemptHeaders)
        if (Exempt.Profile == Profile &&
            pathMatchesHeader(File->getName(), Exempt.Header))
          return true;
    SourceLocation IncludeLoc = SM.getIncludeLoc(FID);
    if (IncludeLoc.isInvalid())
      break;
    FID = SM.getFileID(IncludeLoc);
  }
  return false;
}

bool SemaCXXProfiles::shouldCheck(CXXProfileID Profile,
                                  SourceLocation Loc) const {
  if (!getLangOpts().CXXProfiles || Loc.isInvalid())
    return false;
  // Profiles restrict what is accepted; they never affect which overloads or
  // specializations are viable, and code that is not evaluated has no runtime
  // effect.
  if (SemaRef.isUnevaluatedContext() || SemaRef.isConstantEvaluatedContext() ||
      SemaRef.isSFINAEContext())
    return false;
  const Decl *Owner =
      SemaRef.CurContext ? cast<Decl>(SemaRef.CurContext) : nullptr;
  return isEnforced(Profile, Loc, Owner) && !isExempt(Profile, Loc);
}

void SemaCXXProfiles::addViolation(CXXProfileRuleID Rule,
                                   PartialDiagnosticAt Diag,
                                   SmallVector<PartialDiagnosticAt, 1> Notes) {
  const Decl *Owner =
      SemaRef.CurContext ? cast<Decl>(SemaRef.CurContext) : nullptr;
  PendingViolations.push_back({Rule, Owner, std::move(Diag), std::move(Notes)});
}

namespace {
/// Collects the [[profiles::suppress]] attributes within a declaration.
class SuppressionCollector : public ConstDynamicRecursiveASTVisitor {
public:
  explicit SuppressionCollector(
      llvm::function_ref<void(const ProfilesSuppressAttr *,
                              llvm::PointerUnion<const Decl *, const Stmt *>)>
          Callback)
      : Callback(Callback) {
    ShouldVisitTemplateInstantiations = false;
    ShouldVisitImplicitCode = false;
  }

  bool VisitDecl(const Decl *D) override {
    for (const auto *A : D->specific_attrs<ProfilesSuppressAttr>())
      if (!A->isInherited())
        Callback(A, D);
    return true;
  }

  bool VisitAttributedStmt(const AttributedStmt *S) override {
    for (const Attr *A : S->getAttrs())
      if (const auto *Suppress = dyn_cast<ProfilesSuppressAttr>(A))
        Callback(Suppress, S->getSubStmt());
    return true;
  }

private:
  llvm::function_ref<void(const ProfilesSuppressAttr *,
                          llvm::PointerUnion<const Decl *, const Stmt *>)>
      Callback;
};
} // namespace

ArrayRef<SemaCXXProfiles::SuppressionSite>
SemaCXXProfiles::getImportedSuppressionSites(const Decl *Pattern) {
  auto It = ImportedSuppressionSites.find(Pattern);
  if (It != ImportedSuppressionSites.end())
    return It->second;

  SmallVector<SuppressionSite, 2> &Sites = ImportedSuppressionSites[Pattern];
  auto Add = [&](const ProfilesSuppressAttr *A,
                 llvm::PointerUnion<const Decl *, const Stmt *> Node) {
    if (std::optional<SuppressionSite> Site = makeSuppressionSite(A, Node))
      Sites.push_back(*Site);
  };

  // Collect the suppressions in the outermost enclosing function (so that
  // suppressions around a lambda are found) and on its enclosing
  // declarations.
  const Decl *Root = Pattern;
  for (const DeclContext *DC = Pattern->getLexicalDeclContext(); DC;
       DC = DC->getLexicalParent())
    if (const auto *FD = dyn_cast<FunctionDecl>(DC))
      Root = FD;
  SuppressionCollector(Add).TraverseDecl(Root);
  for (const DeclContext *DC = Root->getLexicalDeclContext();
       DC && !isa<TranslationUnitDecl>(DC); DC = DC->getLexicalParent())
    if (const auto *D = dyn_cast<Decl>(DC))
      for (const auto *A : D->specific_attrs<ProfilesSuppressAttr>())
        Add(A, D);
  return Sites;
}

bool SemaCXXProfiles::isSuppressed(CXXProfileRuleID Rule, SourceLocation Loc,
                                   const Decl *Owner) {
  CXXProfileID Profile = getCXXProfileOfRule(Rule);
  const SourceManager &SM = SemaRef.getSourceManager();
  auto Matches = [&](const SuppressionSite &Site) {
    if (Site.Profile != Profile || (Site.Rule && *Site.Rule != Rule))
      return false;
    SourceRange Range = isa<const Decl *>(Site.Node)
                            ? cast<const Decl *>(Site.Node)->getSourceRange()
                            : cast<const Stmt *>(Site.Node)->getSourceRange();
    return rangeContains(SM, Range, Loc);
  };

  if (llvm::any_of(SuppressionSites, Matches))
    return true;

  // The suppressions of a template from another module are in the AST file.
  const Decl *Pattern = Owner ? getPatternDecl(Owner) : nullptr;
  if (Pattern && Pattern->isFromASTFile())
    return llvm::any_of(getImportedSuppressionSites(Pattern), Matches);
  return false;
}

void SemaCXXProfiles::flushPendingViolations() {
  if (PendingViolations.empty())
    return;

  SmallVector<PendingViolation, 4> Violations;
  std::swap(Violations, PendingViolations);
  for (PendingViolation &V : Violations) {
    SourceLocation Loc = V.Diag.first;
    if (!ReportedViolations
             .insert({static_cast<unsigned>(V.Rule), Loc.getRawEncoding()})
             .second)
      continue;
    if (isSuppressed(V.Rule, Loc, V.Owner))
      continue;
    Diag(Loc, V.Diag.second);
    for (const PartialDiagnosticAt &Note : V.Notes)
      Diag(Note.first, Note.second);
  }
}

//===----------------------------------------------------------------------===//
// std::casting
//===----------------------------------------------------------------------===//

/// Returns the type of the operand of \p E as written, after the
/// array-to-pointer and function-to-pointer conversions that apply to the
/// operand of a cast to a non-reference type.
static QualType getCastOperandType(ASTContext &Ctx, const ExplicitCastExpr *E) {
  QualType T = E->getSubExprAsWritten()->getType();
  if (E->getTypeAsWritten()->isReferenceType())
    return T;
  if (T->isArrayType())
    return Ctx.getArrayDecayedType(T);
  if (T->isFunctionType())
    return Ctx.getPointerType(T);
  return T;
}

/// Returns the type a pointer or reference type \p T refers to, or a null
/// type.
static QualType getReferencedType(QualType T) {
  if (T->isPointerType() || T->isReferenceType())
    return T->getPointeeType();
  return QualType();
}

static bool isCharacterType(QualType T) {
  const auto *BT = T.isNull() ? nullptr : T->getAs<BuiltinType>();
  return BT && (BT->getKind() == BuiltinType::Char_S ||
                BT->getKind() == BuiltinType::Char_U ||
                BT->getKind() == BuiltinType::UChar);
}

/// Returns true if the reinterpret_cast \p E is permitted by std::casting:
/// a cast to a pointer or reference to std::byte, a pointer to integer
/// conversion to an integer type wide enough to hold any pointer, or a cast
/// that does not change the type beyond adding qualifiers.
static bool isPermittedReinterpretCast(ASTContext &Ctx,
                                       const ExplicitCastExpr *E) {
  if (E->getCastKind() == CK_NoOp)
    return true;
  QualType DestType = E->getTypeAsWritten();
  if (E->getCastKind() == CK_PointerToIntegral)
    return Ctx.getTypeSize(DestType) >= Ctx.getTypeSize(Ctx.VoidPtrTy);
  QualType Referenced = getReferencedType(DestType);
  return !Referenced.isNull() && Referenced->isStdByteType();
}

/// Returns a hint that replaces the C-style or functional cast \p E with the
/// named cast \p CastName, or false if no hint can be built.
static bool buildNamedCastFixIts(Sema &S, const ExplicitCastExpr *E,
                                 StringRef CastName,
                                 SmallVectorImpl<FixItHint> &Hints) {
  const SourceManager &SM = S.getSourceManager();
  const LangOptions &LangOpts = S.getLangOpts();

  if (const auto *CE = dyn_cast<CStyleCastExpr>(E)) {
    SourceLocation LParen = CE->getLParenLoc();
    SourceLocation RParen = CE->getRParenLoc();
    const Expr *Operand = CE->getSubExprAsWritten();
    SourceLocation OperandEnd = Operand->getEndLoc();
    if (LParen.isMacroID() || RParen.isMacroID() || OperandEnd.isMacroID())
      return false;
    StringRef TypeText =
        Lexer::getSourceText(
            CharSourceRange::getCharRange(LParen.getLocWithOffset(1), RParen),
            SM, LangOpts)
            .trim();
    if (TypeText.empty())
      return false;
    bool NeedsParens = !isa<ParenExpr>(Operand);
    std::string Replacement = (CastName + "<" + TypeText + ">").str();
    if (NeedsParens)
      Replacement += "(";
    Hints.push_back(FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(LParen, RParen), Replacement));
    if (NeedsParens)
      Hints.push_back(FixItHint::CreateInsertion(
          Lexer::getLocForEndOfToken(OperandEnd, 0, SM, LangOpts), ")"));
    return true;
  }

  const auto *FE = cast<CXXFunctionalCastExpr>(E);
  if (FE->isListInitialization())
    return false;
  SourceLocation Begin = FE->getBeginLoc();
  SourceLocation LParen = FE->getLParenLoc();
  if (Begin.isMacroID() || LParen.isMacroID())
    return false;
  StringRef TypeText =
      Lexer::getSourceText(CharSourceRange::getCharRange(Begin, LParen), SM,
                           LangOpts)
          .trim();
  if (TypeText.empty())
    return false;
  Hints.push_back(
      FixItHint::CreateReplacement(CharSourceRange::getCharRange(Begin, LParen),
                                   (CastName + "<" + TypeText + ">").str()));
  return true;
}

void SemaCXXProfiles::checkExplicitCast(
    const ExplicitCastExpr *E, std::optional<Qualifiers> CastAwayQualifiers) {
  if (E->getCastKind() == CK_Dependent || E->isTypeDependent())
    return;
  SourceLocation Loc = E->getBeginLoc();
  if (!shouldCheck(CXXProfileID::StdCasting, Loc))
    return;

  // How the cast is spelled, for the diagnostics: 0 for a named cast, 1 for
  // a C-style cast, 2 for a functional-style cast.
  unsigned Spelling = isa<CStyleCastExpr>(E)          ? 1
                      : isa<CXXFunctionalCastExpr>(E) ? 2
                                                      : 0;
  // The named cast (or sequence of named casts) the cast performs.
  CStyleCastInterpretationKind Interpretation;
  if (isa<CXXReinterpretCastExpr>(E))
    Interpretation = CStyleCastInterpretationKind::ReinterpretCast;
  else if (isa<CXXConstCastExpr>(E))
    Interpretation = CStyleCastInterpretationKind::ConstCast;
  else if (isa<CXXStaticCastExpr>(E))
    Interpretation = CStyleCastInterpretationKind::StaticCast;
  else if (Spelling != 0)
    Interpretation = E->getCStyleCastInterpretationKind();
  else
    return;
  if (Interpretation == CStyleCastInterpretationKind::Unknown)
    return;
  bool IsReinterpretCast =
      Interpretation == CStyleCastInterpretationKind::ReinterpretCast ||
      Interpretation == CStyleCastInterpretationKind::ReinterpretThenConstCast;
  bool IsStaticCast =
      Interpretation == CStyleCastInterpretationKind::StaticCast ||
      Interpretation == CStyleCastInterpretationKind::StaticThenConstCast;
  bool HasTrailingConstCast =
      Interpretation == CStyleCastInterpretationKind::StaticThenConstCast ||
      Interpretation == CStyleCastInterpretationKind::ReinterpretThenConstCast;

  ASTContext &Ctx = getASTContext();
  StringRef ProfileName = getCXXProfileName(CXXProfileID::StdCasting);
  QualType SourceType = getCastOperandType(Ctx, E);
  QualType DestType = E->getTypeAsWritten();
  SourceRange Range = E->getSourceRange();

  // For a C-style or functional cast, offer to spell the cast as the named
  // cast it performs, unless that takes two named casts. The hint is attached
  // to the first violation only.
  bool OfferedNamedCast = Spelling == 0;
  auto MakeNotes = [&] {
    SmallVector<PartialDiagnosticAt, 1> Notes;
    if (OfferedNamedCast)
      return Notes;
    OfferedNamedCast = true;
    if (HasTrailingConstCast)
      return Notes;
    StringRef CastName =
        ExplicitCastExpr::getCStyleCastInterpretationKindName(Interpretation);
    SmallVector<FixItHint, 2> Hints;
    if (!buildNamedCastFixIts(SemaRef, E, CastName, Hints))
      return Notes;
    PartialDiagnostic Note = PDiag(diag::note_cxx_profile_casting_named_cast)
                             << CastName;
    for (const FixItHint &Hint : Hints)
      Note << Hint;
    Notes.push_back({Loc, Note});
    return Notes;
  };

  if (IsReinterpretCast && !isPermittedReinterpretCast(Ctx, E)) {
    bool ToCharacter = isCharacterType(getReferencedType(DestType));
    SmallVector<PartialDiagnosticAt, 1> Notes = MakeNotes();
    QualType SourceReferenced = getReferencedType(SourceType);
    QualType DestReferenced = getReferencedType(DestType);
    if (!SourceReferenced.isNull() && !DestReferenced.isNull() &&
        SourceReferenced->isObjectType() && DestReferenced->isObjectType() &&
        !SourceReferenced->isIncompleteType() &&
        !DestReferenced->isIncompleteType() &&
        SourceReferenced.isTriviallyCopyableType(Ctx) &&
        DestReferenced.isTriviallyCopyableType(Ctx) &&
        Ctx.getTypeSizeInChars(SourceReferenced) ==
            Ctx.getTypeSizeInChars(DestReferenced))
      Notes.push_back(
          {Loc, PDiag(diag::note_cxx_profile_casting_use_bit_cast)});
    addViolation(
        ToCharacter ? CXXProfileRuleID::StdCastingReinterpretCastChar
                    : CXXProfileRuleID::StdCastingReinterpretCast,
        {Loc, PDiag(ToCharacter
                        ? diag::warn_cxx_profile_casting_reinterpret_cast_char
                        : diag::warn_cxx_profile_casting_reinterpret_cast)
                  << ProfileName << Spelling << SourceType << DestType
                  << Range},
        std::move(Notes));
  }

  // Removing 'volatile' (or 'restrict') alone is not casting away constness
  // for the purposes of this rule; removing qualifiers at an inner level of
  // a multi-level pointer is.
  if (CastAwayQualifiers &&
      (CastAwayQualifiers->hasConst() || (!CastAwayQualifiers->hasVolatile() &&
                                          !CastAwayQualifiers->hasRestrict())))
    addViolation(CXXProfileRuleID::StdCastingConstCast,
                 {Loc, PDiag(diag::warn_cxx_profile_casting_const_cast)
                           << ProfileName << Spelling << SourceType << DestType
                           << Range},
                 MakeNotes());

  if (!IsStaticCast)
    return;

  if (E->getCastKind() == CK_BaseToDerived) {
    SmallVector<PartialDiagnosticAt, 1> Notes = MakeNotes();
    QualType SourceClass =
        SourceType->isPointerType() ? SourceType->getPointeeType() : SourceType;
    QualType DestClass =
        DestType->isPointerType() || DestType->isReferenceType()
            ? DestType->getPointeeType()
            : DestType;
    if (const auto *RD = SourceClass->getAsCXXRecordDecl();
        RD && RD->hasDefinition() && RD->isPolymorphic())
      Notes.push_back(
          {Loc, PDiag(diag::note_cxx_profile_casting_use_dynamic_cast)});
    addViolation(CXXProfileRuleID::StdCastingStaticDowncast,
                 {Loc, PDiag(diag::warn_cxx_profile_casting_static_downcast)
                           << ProfileName << Spelling << SourceClass
                           << DestClass << Range},
                 std::move(Notes));
    return;
  }

  if (SourceType->isVoidPointerType() && DestType->isPointerType() &&
      !DestType->isVoidPointerType() &&
      !DestType->getPointeeType()->isStdByteType())
    addViolation(CXXProfileRuleID::StdCastingStaticCastVoid,
                 {Loc, PDiag(diag::warn_cxx_profile_casting_static_cast_void)
                           << ProfileName << Spelling << SourceType << DestType
                           << Range},
                 MakeNotes());
}
