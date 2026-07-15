//===----- SemaProfiles.cpp --- C++ profiles framework --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for the C++ profiles framework
/// (P3589R2) and the built-in std::init initialization profile (P4222R1.1):
/// profile enforcement and suppression state, the shared violation gate, and
/// the parse-time std::init rule checks. The CFG-based std::init checks live
/// in AnalysisBasedWarnings.cpp.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaProfiles.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMap.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Attr.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

using namespace clang;

SemaProfiles::SemaProfiles(Sema &S) : SemaBase(S) {}


bool SemaProfiles::isProfileEnforced(StringRef ProfileName) const {
  if (!getLangOpts().Profiles)
    return false;
  // The built-in test:: profiles only exercise the framework; keep them inert
  // unless the test suite opts in via -fprofiles-test-profiles.
  if (!getLangOpts().ProfilesTestProfiles && ProfileName.starts_with("test::"))
    return false;
  return getProfileEnforcement(ProfileName) != nullptr;
}

const SemaProfiles::ProfileEnforcement *
SemaProfiles::getProfileEnforcement(StringRef ProfileName) const {
  for (const auto &E : EnforcedProfiles)
    if (E.ProfileName == ProfileName)
      return &E;
  return nullptr;
}

bool SemaProfiles::addProfileEnforcement(StringRef Name, StringRef Designator,
                                 SourceLocation Loc) {
  if (const auto *Existing = getProfileEnforcement(Name)) {
    if (Existing->Designator != Designator) {
      Diag(Loc, diag::err_profiles_enforce_mismatch) << Name;
      Diag(Existing->EnforceLoc, diag::note_previous_attribute);
      return false;
    }
    return true;
  }
  EnforcedProfiles.push_back({{Name.str(), Designator.str()}, Loc});
  return true;
}

// Unzip profile arguments into the parallel key/value/kind arrays that the
// semantic attributes store (Attr.td cannot hold structured arguments).
static void unzipProfileArguments(ArrayRef<profiles::ProfileArgument> Arguments,
                                  SmallVectorImpl<StringRef> &Keys,
                                  SmallVectorImpl<StringRef> &Values,
                                  SmallVectorImpl<unsigned> &Kinds) {
  for (const auto &Arg : Arguments) {
    Keys.push_back(Arg.Key);
    Values.push_back(Arg.Value);
    Kinds.push_back(static_cast<unsigned>(Arg.Kind));
  }
}

static void appendProfileArgumentData(
    ArrayRef<profiles::ProfileArgument> Arguments,
    SmallVectorImpl<unsigned> *ArgumentCounts,
    SmallVectorImpl<StringRef> *ArgumentKeys,
    SmallVectorImpl<StringRef> *ArgumentValues,
    SmallVectorImpl<unsigned> *ArgumentKinds) {
  if (!ArgumentCounts)
    return;

  assert(ArgumentKeys && ArgumentValues && ArgumentKinds);
  ArgumentCounts->push_back(Arguments.size());
  unzipProfileArguments(Arguments, *ArgumentKeys, *ArgumentValues,
                        *ArgumentKinds);
}

bool SemaProfiles::processProfilesEnforceAttr(
    const ParsedAttr &AL, Module *Mod, SmallVectorImpl<StringRef> *NewNames,
    SmallVectorImpl<StringRef> *NewDesignators,
    SmallVectorImpl<unsigned> *NewArgumentCounts,
    SmallVectorImpl<StringRef> *NewArgumentKeys,
    SmallVectorImpl<StringRef> *NewArgumentValues,
    SmallVectorImpl<unsigned> *NewArgumentKinds) {
  const auto &Args = AL.getProfileEnforceArgs();
  if (Args.Designators.empty()) {
    Diag(AL.getLoc(), diag::err_attribute_too_few_arguments) << AL << 1;
    return false;
  }

  for (const auto &D : Args.Designators) {
    StringRef Name = D.Name;
    StringRef Spelling = D.Spelling;

    // "Already recorded?" must use the ungated lookup: isProfileEnforced
    // filters gated-off test:: names (and -fprofiles off), which would make
    // every repetition of such a profile look new and re-append its
    // designator to the attribute's argument arrays.
    bool IsNew = !getProfileEnforcement(Name);
    if (!addProfileEnforcement(Name, Spelling, AL.getLoc()))
      continue;

    if (Mod && !llvm::any_of(Mod->EnforcedProfileDesignators,
                             [&](const Module::EnforcedProfile &EP) {
                               return EP.ProfileName == Name;
                             }))
      Mod->EnforcedProfileDesignators.push_back({Name.str(), Spelling.str()});

    if (IsNew) {
      if (NewNames)
        NewNames->push_back(Name);
      if (NewDesignators)
        NewDesignators->push_back(Spelling);
      appendProfileArgumentData(D.Arguments, NewArgumentCounts,
                                NewArgumentKeys, NewArgumentValues,
                                NewArgumentKinds);
    }
  }
  return true;
}

// P3589R2 [decl.attr.enforce]p5: profiles are compatible if they are the same
// -- by name; arguments configure a profile without changing its identity --
// or proclaimed compatible by the implementation. "All standard profiles are
// compatible with each other" is the one proclamation modeled here.
static bool areProfilesCompatible(StringRef A, StringRef B) {
  return A == B || (A.starts_with("std::") && B.starts_with("std::"));
}

void SemaProfiles::checkRedeclarationProfileCompatibility(
    const NamedDecl *New, const NamedDecl *Old) {
  if (!getLangOpts().Profiles)
    return;
  // Only a previous declaration from another module unit (a named module or a
  // header unit) can carry a different profile dominion. A textual or PCH
  // previous declaration shares this TU's dominion: the placement rule makes
  // a TU's dominion uniform over its declarations, and a PCH's enforcements
  // are restored into this TU (ENFORCED_PROFILES).
  if (!Old->isFromASTFile())
    return;
  Module *M = Old->getOwningModule();
  if (!M)
    return;
  // A declaration in an explicit global-module-fragment precedes the module
  // declaration, so the module's exported enforcements do not cover it, and
  // its TU's empty-declaration enforcements are not serialized into the BMI.
  // Its dominion is unknown: skip rather than guess (a missed diagnostic,
  // never a wrong one). An *implicit* global-module-fragment declaration (a
  // purview extern "C"/"C++" declaration, the common redeclarable case) sits
  // inside the module declaration's dominion and is checked.
  if (M->isExplicitGlobalModule())
    return;
  Module *Top = M->getTopLevelModule();
  // Within the same module family (an implementation or partition unit seeing
  // its interface) the exported set under-approximates the interface TU's
  // full dominion, and the interface's enforcements are inherited into this
  // unit anyway; skip rather than false-positive on locally added profiles.
  if (Module *Current = SemaRef.getCurrentModule())
    if (Current->getTopLevelModule()->getPrimaryModuleInterfaceName() ==
        Top->getPrimaryModuleInterfaceName())
      return;

  // A gated-off test:: profile is inert in this compilation, on either side.
  auto IsActive = [&](StringRef Name) {
    return getLangOpts().ProfilesTestProfiles || !Name.starts_with("test::");
  };
  // First active profile in Enforced with no compatible counterpart in
  // Covering; empty if fully covered.
  auto FindUncovered = [&](const auto &Enforced,
                           const auto &Covering) -> StringRef {
    for (const auto &EP : Enforced) {
      StringRef Name = EP.ProfileName;
      if (!IsActive(Name))
        continue;
      if (llvm::none_of(Covering, [&](const auto &Other) {
            return areProfilesCompatible(Name, Other.ProfileName);
          }))
        return Name;
    }
    return {};
  };

  // The rule is symmetric: every profile whose dominion covers one
  // declaration must have a compatible counterpart covering the other.
  // Report the first violation in each direction.
  StringRef MissingHere =
      FindUncovered(Top->EnforcedProfileDesignators, EnforcedProfiles);
  StringRef MissingThere =
      FindUncovered(EnforcedProfiles, Top->EnforcedProfileDesignators);
  if (MissingHere.empty() && MissingThere.empty())
    return;
  if (!MissingHere.empty())
    Diag(New->getLocation(), diag::err_profiles_redecl_incompatible)
        << /*PreviouslyEnforced=*/0 << New << MissingHere << Top->Name;
  if (!MissingThere.empty())
    Diag(New->getLocation(), diag::err_profiles_redecl_incompatible)
        << /*PreviouslyEnforced=*/1 << New << MissingThere << Top->Name;
  Diag(Old->getLocation(), diag::note_previous_declaration);
}

ProfilesSuppressAttr *
SemaProfiles::makeProfilesSuppressAttr(const ParsedAttr &AL) {
  const auto &Args = AL.getProfileSuppressArgs();
  if (Args.Name.empty())
    return nullptr;

  SmallVector<StringRef, 4> RawArgs;
  for (const auto &Arg : Args.RawArguments)
    RawArgs.push_back(Arg);
  SmallVector<StringRef, 4> RawArgumentKeys;
  SmallVector<StringRef, 4> RawArgumentValues;
  SmallVector<unsigned, 4> RawArgumentKinds;
  unzipProfileArguments(Args.Arguments, RawArgumentKeys, RawArgumentValues,
                        RawArgumentKinds);

  return ::new (getASTContext()) ProfilesSuppressAttr(
      getASTContext(), AL, Args.Name, Args.Justification, Args.Rule,
      RawArgs.data(), RawArgs.size(), RawArgumentKeys.data(),
      RawArgumentKeys.size(), RawArgumentValues.data(),
      RawArgumentValues.size(), RawArgumentKinds.data(),
      RawArgumentKinds.size());
}

ProfilesSuppressAttr *
SemaProfiles::makeImplicitProfilesSuppressAttr(StringRef ProfileName,
                                       StringRef RuleName) {
  return ProfilesSuppressAttr::CreateImplicit(
      getASTContext(), ProfileName, /*Justification=*/"", RuleName,
      /*RawArguments=*/nullptr, /*RawArgumentsSize=*/0,
      /*RawArgumentKeys=*/nullptr, /*RawArgumentKeysSize=*/0,
      /*RawArgumentValues=*/nullptr, /*RawArgumentValuesSize=*/0,
      /*RawArgumentKinds=*/nullptr, /*RawArgumentKindsSize=*/0);
}

static bool profileSuppressMatches(StringRef EntryProfile, StringRef EntryRule,
                                   StringRef Profile, StringRef Rule) {
  return EntryProfile == Profile &&
         (EntryRule.empty() || EntryRule == Rule);
}

bool SemaProfiles::isProfileSuppressed(StringRef ProfileName,
                                       StringRef RuleName,
                                       SourceLocation Loc) const {
  const SourceManager &SM = getASTContext().getSourceManager();
  for (const auto &E : ProfileSuppressStack) {
    if (!profileSuppressMatches(E.ProfileName, E.RuleName, ProfileName,
                                RuleName))
      continue;
    // The entry's dominion is its construct's token range (P3589R2 s2.4p3):
    // a violation before the recorded begin -- e.g. in a template pattern
    // instantiated synchronously while the scope is live -- is outside it,
    // as is one past the recorded end -- e.g. in a pattern first declared
    // *after* the suppressed construct. The end is recorded only for a
    // construct fully parsed at push time; when it is invalid the scope's
    // lifetime bounds the dominion, which is exact mid-parse (later tokens
    // are unparsed and instantiation of undefined templates is deferred).
    // Fail open on an invalid location on either side
    // (isBeforeInTranslationUnit rejects invalid locations), preserving
    // plain-liveness behavior for synthesized code. Locations are compared
    // in raw TU token order: expansion-loc normalization would collapse all
    // tokens of one macro expansion onto the invocation and over-suppress.
    if (Loc.isInvalid() || E.Begin.isInvalid() ||
        (!SM.isBeforeInTranslationUnit(Loc, E.Begin) &&
         (E.End.isInvalid() || !SM.isBeforeInTranslationUnit(E.End, Loc))))
      return true;
  }
  return false;
}

bool SemaProfiles::isProfileSuppressed(StringRef ProfileName,
                                       StringRef RuleName,
                                       const Decl *D) const {
  for (; D;) {
    for (const auto *PSA : D->specific_attrs<ProfilesSuppressAttr>())
      if (profileSuppressMatches(PSA->getProfileName(), PSA->getRule(),
                                 ProfileName, RuleName))
        return true;
    const DeclContext *DC = D->getLexicalDeclContext();
    D = DC ? dyn_cast<Decl>(DC) : nullptr;
  }
  return false;
}

bool SemaProfiles::isProfileSuppressed(StringRef ProfileName,
                                       StringRef RuleName, const Stmt *S,
                                       AnalysisDeclContext &AC) const {
  ParentMap &PM = AC.getParentMap();
  for (const Stmt *Cur = S; Cur; Cur = PM.getParent(Cur)) {
    if (const auto *AS = dyn_cast<AttributedStmt>(Cur))
      for (const Attr *A : AS->getAttrs())
        if (const auto *PSA = dyn_cast<ProfilesSuppressAttr>(A))
          if (profileSuppressMatches(PSA->getProfileName(), PSA->getRule(),
                                     ProfileName, RuleName))
            return true;
    // [[profiles::suppress]] on a local variable attaches to the VarDecl,
    // not the enclosing DeclStmt. Walk the declared decls so the post-parse
    // walker matches the parse-time ProfileSuppressForInit RAII behavior.
    if (const auto *DS = dyn_cast<DeclStmt>(Cur))
      for (const Decl *D : DS->decls())
        for (const auto *PSA : D->specific_attrs<ProfilesSuppressAttr>())
          if (profileSuppressMatches(PSA->getProfileName(), PSA->getRule(),
                                     ProfileName, RuleName))
            return true;
  }
  return isProfileSuppressed(ProfileName, RuleName, AC.getDecl());
}

bool SemaProfiles::shouldEmitProfileViolation(StringRef ProfileName,
                                              StringRef RuleName,
                                              SourceLocation Loc) {
  return shouldEmitProfileViolation(ProfileName, RuleName, Loc, /*D=*/nullptr);
}

bool SemaProfiles::shouldEmitProfileViolation(StringRef ProfileName,
                                              StringRef RuleName,
                                              SourceLocation Loc,
                                              const Decl *D) {
  if (!isProfileEnforced(ProfileName))
    return false;
  // Honor [[profiles::suppress]] from the parse-time stack and, when a Decl is
  // available, from the declaration and its lexical parents. The latter does
  // not depend on a parse-time scope still being active, so finalization checks
  // that run after the parse scope is torn down still respect suppression.
  //
  // The stack consult is dominion-checked against Loc: a check that fires
  // while an unrelated construct's ProfileSuppressScope is live -- a
  // synchronously instantiated pattern, or a class finalized as a side effect
  // of one -- matches only entries whose construct's tokens cover Loc
  // (P3589R2 s2.4p3), so no explicit finalization or instantiation guard is
  // needed here.
  if (isProfileSuppressed(ProfileName, RuleName, Loc) ||
      isProfileSuppressed(ProfileName, RuleName, D))
    return false;
  // P3589R2 Section 1.1: "its static semantic effects are as-if applied only
  // after translation phase 7. It is not possible for a profile to change the
  // outcome of overload resolution or template instantiation, nor is it
  // possible to 'SFINAE out' failure of a program to satisfy a profile
  // requirement."
  //
  // A templated entity is not yet a phase-7 entity, so a Decl-carrying rule
  // fires only on its instantiation -- where D is the instantiated,
  // non-templated declaration -- not on the template pattern (the [[uninit]]
  // marker checks and the binding checks that pass a Decl are re-run on the
  // instantiated field / variable, so they defer here). Checking the pattern
  // too would double-fire, once at parse and again per instantiation.
  //
  // The Decl-less expression check sites (D == nullptr here) instead defer
  // from their own entry points, and only when their check-relevant operands
  // are instantiation-dependent -- exactly the constructs TreeTransform
  // always rebuilds, so the hosting Build* routine re-runs the deferred check
  // at instantiation. A fully non-dependent construct may be returned
  // unchanged by TreeTransform (its Build* never re-runs), so it is checked
  // at definition time instead; when such a construct is rebuilt at
  // instantiation anyway (a local operand, a call argument, a return), the
  // definition-time diagnostic repeats there -- accepted for now. This
  // deliberately trades strict phase-7 purity for reuse-proof diagnostics: a
  // non-dependent violation in a never-instantiated template, or in an
  // if-constexpr branch not yet known to be discarded, diagnoses at
  // definition time -- the same model the test::type_cast profile and the
  // reinterpret_cast check follow.
  if (D && D->isTemplated())
    return false;
  if (SemaRef.isUnevaluatedContext())
    return false;
  if (SemaRef.currentEvaluationContext().isDiscardedStatementContext())
    return false;
  return true;
}

bool SemaProfiles::shouldEmitProfileViolation(StringRef ProfileName,
                                              StringRef RuleName,
                                              const Stmt *UseStmt,
                                              AnalysisDeclContext &AC) const {
  if (!isProfileEnforced(ProfileName))
    return false;
  if (isProfileSuppressed(ProfileName, RuleName, UseStmt, AC))
    return false;
  return true;
}

bool SemaProfiles::checkProfileViolation(StringRef ProfileName,
                                         StringRef RuleName, SourceLocation Loc,
                                         unsigned DiagID) {
  if (!shouldEmitProfileViolation(ProfileName, RuleName, Loc))
    return false;
  Diag(Loc, DiagID) << ProfileName;
  return true;
}

void SemaProfiles::ProfileSuppressScope::push(StringRef ProfileName,
                                      StringRef RuleName,
                                      SourceLocation Begin,
                                      SourceLocation End) {
  S.Profiles().ProfileSuppressStack.push_back(
      {ProfileName, RuleName, Begin, End});
  ++Count;
}

SemaProfiles::ProfileSuppressScope::ProfileSuppressScope(
    Sema &S, const ParsedAttributesView &Attrs)
    : S(S) {
  if (!S.getLangOpts().Profiles)
    return;
  for (const auto &AL : Attrs) {
    if (AL.getKind() != ParsedAttr::AT_ProfilesSuppress)
      continue;
    const auto &Args = AL.getProfileSuppressArgs();
    // These are the prefix attributes of a statement or declaration about to
    // be parsed, so the attribute's own location is the construct's begin and
    // no end is known yet (the scope's lifetime bounds it).
    if (!Args.Name.empty())
      push(Args.Name, Args.Rule, AL.getLoc(), SourceLocation());
  }
}

/// The end location of \p D's construct if it is fully parsed, invalid
/// otherwise. A partially parsed construct's end location is usually *valid
/// but early* -- a mid-parse class collapses to its name token (the brace
/// range is set only by ActOnTagFinishDefinition, after even the late-parsed
/// members), a body-pending function ends at its declarator, an
/// uninitialized variable at its declarator -- so each arm gates on the
/// marker that the construct's real end has been seen. Returning invalid
/// falls back to scope-lifetime bounding, which is exact mid-parse.
static SourceLocation getCompletedConstructEnd(const Decl *D) {
  if (const auto *TD = dyn_cast<TagDecl>(D))
    return TD->getBraceRange().getEnd();
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    // isLateTemplateParsed makes doesThisDeclarationHaveABody true while the
    // body is merely token-cached and the range still ends at the declarator.
    if (FD->doesThisDeclarationHaveABody() && !FD->isLateTemplateParsed())
      return FD->getSourceRange().getEnd();
    return SourceLocation();
  }
  if (const auto *VD = dyn_cast<VarDecl>(D)) {
    if (VD->hasInit())
      return VD->getSourceRange().getEnd();
    return SourceLocation();
  }
  if (const auto *FD = dyn_cast<FieldDecl>(D)) {
    // The in-class initializer expression is null while its late parse is
    // still pending.
    if (FD->hasNonNullInClassInitializer())
      return FD->getInClassInitializer()->getEndLoc();
    return SourceLocation();
  }
  if (const auto *ND = dyn_cast<NamespaceDecl>(D))
    return ND->getRBraceLoc();
  return SourceLocation();
}

void SemaProfiles::ProfileSuppressScope::addFromDecl(const Decl *D) {
  SourceLocation Begin = D->getBeginLoc();
  if (Begin.isInvalid())
    Begin = D->getLocation();
  SourceLocation End = getCompletedConstructEnd(D);
  for (const auto *A : D->specific_attrs<ProfilesSuppressAttr>())
    push(A->getProfileName(), A->getRule(), Begin, End);
}

SemaProfiles::ProfileSuppressScope::ProfileSuppressScope(Sema &S, const Decl *D,
                                                  bool WalkLexicalParents)
    : S(S) {
  if (!S.getLangOpts().Profiles || !D)
    return;
  addFromDecl(D);
  if (WalkLexicalParents) {
    for (const DeclContext *DC = D->getLexicalDeclContext(); DC;
         DC = DC->getLexicalParent())
      if (const auto *Parent = dyn_cast<Decl>(DC))
        addFromDecl(Parent);
  }
}

SemaProfiles::ProfileSuppressScope::ProfileSuppressScope(Sema &S,
                                                  ArrayRef<const Attr *> Attrs,
                                                  SourceLocation Begin,
                                                  SourceLocation End)
    : S(S) {
  if (!S.getLangOpts().Profiles)
    return;
  for (const auto *A : Attrs)
    if (const auto *PSA = dyn_cast<ProfilesSuppressAttr>(A))
      push(PSA->getProfileName(), PSA->getRule(), Begin, End);
}

SemaProfiles::ProfileSuppressScope::~ProfileSuppressScope() {
  assert(S.Profiles().ProfileSuppressStack.size() >= Count);
  S.Profiles().ProfileSuppressStack.pop_back_n(Count);
}

static bool defaultInitLeavesScalarIndeterminateImpl(
    ASTContext &Ctx, QualType T, bool HonorUninitMarkers,
    llvm::SmallPtrSetImpl<const CXXRecordDecl *> &Visited) {
  if (T->isDependentType() || T->isIncompleteType())
    return false;
  if (const ArrayType *AT = Ctx.getAsArrayType(T))
    return defaultInitLeavesScalarIndeterminateImpl(
        Ctx, AT->getElementType(), HonorUninitMarkers, Visited);
  if (T->isReferenceType())
    return false;
  const auto *RD = T->getAsCXXRecordDecl();
  if (!RD)
    // Scalars, pointers, and enums are left indeterminate by default-init,
    // except std::byte, which the profile permits to be uninitialized
    // (paper §4), so a std::byte subobject does not make a record
    // indeterminate.
    return T->isScalarType() && !T->isStdByteType();
  if (RD->isInvalidDecl())
    return false;
  // A union's members are mutually exclusive, so the per-member walk below does
  // not apply. Default-initialization leaves it without an initialized member
  // (paper §6.5) unless it has no members, has a user-provided default
  // constructor (trusted), or a default member initializer initializes one.
  if (RD->isUnion()) {
    if (RD->field_empty() || RD->hasUserProvidedDefaultConstructor())
      return false;
    for (const FieldDecl *F : RD->fields())
      if (F->hasInClassInitializer())
        return false;
    return true;
  }
  // Break cycles from ill-formed self-containing types (e.g. struct S {S x;}).
  if (!Visited.insert(RD->getCanonicalDecl()).second)
    return false;
  // Trust a user-provided default constructor: ctor_uninit_member checks at its
  // definition.
  if (RD->hasUserProvidedDefaultConstructor())
    return false;
  for (const CXXBaseSpecifier &Base : RD->bases())
    if (defaultInitLeavesScalarIndeterminateImpl(Ctx, Base.getType(),
                                                 HonorUninitMarkers, Visited))
      return true;
  for (const FieldDecl *F : RD->fields()) {
    if (F->isUnnamedBitField() || F->hasInClassInitializer())
      continue;
    // A member the type's author marked [[uninit]] is acknowledged as
    // intentionally uninitialized, so it does not leave an unacknowledged
    // scalar indeterminate (paper §6.2).
    if (HonorUninitMarkers && F->hasAttr<UninitAttr>())
      continue;
    if (defaultInitLeavesScalarIndeterminateImpl(Ctx, F->getType(),
                                                 HonorUninitMarkers, Visited))
      return true;
  }
  return false;
}

bool SemaProfiles::defaultInitLeavesScalarIndeterminate(QualType T,
                                                bool HonorUninitMarkers) {
  llvm::SmallPtrSet<const CXXRecordDecl *, 8> Visited;
  return defaultInitLeavesScalarIndeterminateImpl(getASTContext(), T,
                                                  HonorUninitMarkers, Visited);
}

bool SemaProfiles::defaultInitIsVacuous(QualType T) {
  QualType BaseTy = getASTContext().getBaseElementType(T);
  if (const auto *RD = BaseTy->getAsCXXRecordDecl()) {
    // A non-trivial default constructor (user-provided anywhere in the
    // subtree, a default member initializer, a virtual table pointer)
    // initializes something, contradicting an [[uninit]] marker (paper §4.2
    // rule 2, §5.3). hasTrivialDefaultConstructor asserts without a
    // definition.
    if (!RD->hasDefinition() || !RD->hasTrivialDefaultConstructor())
      return false;
    // A deleted default constructor keeps the triviality bit, but makes
    // default-initialization ill-formed rather than a no-op: the entity can
    // never be left default-initialized, so the marker is unsatisfiable. Only
    // a declared deleted constructor is visible here; a lazily *implicitly*
    // deleted one of an otherwise trivial type escapes this scan -- a missed
    // diagnostic (never a false positive), like the framework's other
    // conservative omissions.
    for (const CXXConstructorDecl *Ctor : RD->getDefinition()->ctors())
      if (Ctor->isDefaultConstructor() && Ctor->isDeleted())
        return false;
  }
  // The factual (HonorUninitMarkers=false) walk: an all-scalars-determinate
  // type (e.g. an empty struct) has nothing uninitialized, so the marker
  // contradicts it too, while a type whose only indeterminate scalars are
  // themselves marked members really is left uninitialized.
  return defaultInitLeavesScalarIndeterminate(T, /*HonorUninitMarkers=*/false);
}

// Whether the declaration initializer \p Init is a vacuous
// default-initialization of \p T: one that runs no code and leaves the object
// factually uninitialized, hence consistent with an [[uninit]] marker. No
// initializer at all (a scalar default-init synthesizes none) is vacuous; a
// synthesized trivial default-constructor call is vacuous iff the type's
// default-initialization is (defaultInitIsVacuous); anything else -- a
// user-written initializer, a `= P()` value-initialization (a
// CXXTemporaryObjectExpr, which zeroes), any zero-initializing construction
// -- initializes the object and contradicts the marker. The shared guard of
// static_marker and uninit_with_initializer keeps the pair complementary by
// construction: exactly one of the two fires for a marked static.
static bool isVacuousDefaultInit(SemaProfiles &SP, const Expr *Init,
                                 QualType T) {
  if (!Init)
    return true;
  if (const auto *CCE = dyn_cast<CXXConstructExpr>(Init->IgnoreImplicit()))
    return CCE->getConstructor()->isDefaultConstructor() &&
           !isa<CXXTemporaryObjectExpr>(CCE) &&
           !CCE->requiresZeroInitialization() && SP.defaultInitIsVacuous(T);
  return false;
}

void SemaProfiles::checkInitProfileUninitDecl(const VarDecl *Var) {
  // std::init / uninit_decl: a definition without any initializer (after
  // attempted default-initialization) must either carry [[uninit]] or
  // be initialized by a language rule. Static / thread storage duration is
  // excluded -- those are zero-initialized; runtime-init concerns are R3's.
  static constexpr StringRef Profile = "std::init";
  static constexpr StringRef Rule = "uninit_decl";
  // The enforcement check gates the (possibly recursive) type walk below so
  // it runs only under the profile, not on every default-initialized
  // variable.
  QualType BaseTy = getASTContext().getBaseElementType(Var->getType());
  if (!Var->isInvalidDecl() && Var->getStorageDuration() == SD_Automatic &&
      !Var->hasAttr<UninitAttr>() &&
      // std::byte may be left uninitialized (paper §4), so it -- and arrays
      // of it -- are exempt from this rule.
      !BaseTy->isStdByteType() &&
      shouldEmitProfileViolation(Profile, Rule, Var->getLocation(), Var) &&
      // A definition with no initializer (scalar / pointer / enum, or an
      // array of them), or a class/aggregate type -- possibly the element
      // type of an array -- whose default-init leaves a scalar subobject
      // indeterminate (its synthesized constructor call provides an
      // initializer, so the !getInit() test alone misses it).
      (!Var->getInit() ||
       (BaseTy->isRecordType() &&
        defaultInitLeavesScalarIndeterminate(Var->getType(),
                                             /*HonorUninitMarkers=*/true)))) {
    // A union variable cannot carry [[uninit]] (union_marker bans it),
    // so it must be initialized; use a message that does not suggest the
    // marker as a remedy.
    bool IsUnion = BaseTy->isUnionType();
    Diag(Var->getLocation(),
         IsUnion ? diag::err_init_uninit_union : diag::err_init_uninit_decl)
        << Profile << Var->getDeclName();
  }
}

void SemaProfiles::checkInitProfileStaticMarker(const VarDecl *Var) {
  // std::init / static_marker: a variable with static or thread storage
  // duration is zero-initialized by language rule (paper §3), so it is an
  // initialized object; marking it [[uninit]] contradicts paper §4.2 ("an
  // initialized object marked [[uninit]] is an error"). The case with a real
  // initializer -- explicit, or a default-initialization that is not a no-op
  // -- is already caught by uninit_with_initializer (R4, in
  // CheckCompleteVariableDeclaration); this covers the vacuous-initialization
  // case R4 treats as consistent. The guard is the shared
  // isVacuousDefaultInit, so the pair stays complementary by construction and
  // exactly one of static_marker / uninit_with_initializer fires (this one
  // runs first, from ActOnUninitializedDecl, after the synthesized
  // default-initialization is attached).
  static constexpr StringRef Profile = "std::init";
  QualType BaseTy = getASTContext().getBaseElementType(Var->getType());
  if (!Var->isInvalidDecl() &&
      (Var->getStorageDuration() == SD_Static ||
       Var->getStorageDuration() == SD_Thread) &&
      Var->hasAttr<UninitAttr>() &&
      // A union or pointer object -- or an array of them -- marked [[uninit]]
      // is already rejected by union_marker / pointer_marker (regardless of
      // storage duration, and keyed on the same base element type), and they
      // retain the marker; do not pile a second diagnostic on top.
      !BaseTy->isUnionType() && !BaseTy->isPointerType() &&
      shouldEmitProfileViolation(Profile, "static_marker", Var->getLocation(),
                                 Var) &&
      isVacuousDefaultInit(*this, Var->getInit(), Var->getType())) {
    bool IsThread = Var->getStorageDuration() == SD_Thread;
    Diag(Var->getLocation(), diag::err_init_uninit_static_marker)
        << Profile << Var->getDeclName() << IsThread;
  }
}

bool SemaProfiles::checkInitProfileStaticRuntimeInit(
    const VarDecl *Var, llvm::function_ref<bool()> CheckConstInit) {
  // Thread-locals have thread (not static) storage duration; paper §3 scopes
  // this rule to non-local *static* objects (uninit_decl likewise excludes
  // thread storage).
  if (Var->getTLSKind() != VarDecl::TLS_None)
    return false;
  // std::init / static_runtime_init: paper says non-local statics must be
  // initialized at compile or link time. CheckConstInit() permits trivial
  // default initialization (not a constant initializer but needs no global
  // constructor), so a zero-initialized aggregate such as
  // `struct S { int x; }; S g;` is not a violation. Runs before
  // -Wglobal-constructors so the profile error (when enforced) takes
  // precedence over the standalone warning.
  static constexpr StringRef Profile = "std::init";
  static constexpr StringRef Rule = "static_runtime_init";
  // Gate on enforcement before evaluating the initializer: this call site
  // sits ahead of -Wglobal-constructors' isIgnored guard, so evaluating
  // first would charge every global with a non-constant initializer for the
  // constant-initializer evaluation even with profiles disabled.
  if (!shouldEmitProfileViolation(Profile, Rule, Var->getLocation(), Var))
    return false;
  if (CheckConstInit())
    return false;
  Diag(Var->getLocation(), diag::err_init_static_runtime_init)
      << Profile << Var->getDeclName();
  return true;
}

void SemaProfiles::checkInitProfileUninitWithInitializer(const ValueDecl *D,
                                                 const Expr *Init) {
  // [[uninit]] documents that the entity is intentionally left
  // uninitialized, so it contradicts an explicit initializer. A RecoveryExpr
  // is a placeholder for an initialization that already failed (e.g.
  // default-init of a const scalar), not an initializer the user wrote, so it
  // must not trigger this rule.
  if (!D->hasAttr<UninitAttr>() || !Init ||
      isa<RecoveryExpr>(Init->IgnoreParens()))
    return;
  SourceLocation Loc = D->getLocation();
  static constexpr StringRef Profile = "std::init";
  static constexpr StringRef Rule = "uninit_with_initializer";
  // Gate the (possibly recursive) type walk below on enforcement.
  if (!shouldEmitProfileViolation(Profile, Rule, Loc, D))
    return;
  // A vacuous default-initialization -- a synthesized trivial
  // default-constructor call that runs no code and leaves the object
  // indeterminate -- is consistent with the marker: the object really is left
  // uninitialized, to be initialized later (e.g. via construct_at), mirroring
  // the scalar case. Anything else -- an explicit initializer, a `= P()`
  // value-initialization, or a default-initialization that initializes
  // something (a non-trivial default constructor, paper §4.2 rule 2, §5.3) --
  // contradicts it.
  if (isVacuousDefaultInit(*this, Init, D->getType()))
    return;
  Diag(Loc, diag::err_init_uninit_with_initializer)
      << Profile << D->getDeclName() << isa<FieldDecl>(D);
}

void SemaProfiles::checkInitProfileMarkerPlacement(const Decl *D) {
  const auto *UA = D->getAttr<UninitAttr>();
  if (!UA)
    return;
  SourceLocation Loc = UA->getLocation();

  // std::init / union_marker (paper §5.6): the marker is banned on a union
  // object or a union member, because delayed initialization by assigning a
  // member would be an erroneous assignment when compiled without the profile.
  // std::init / pointer_marker (paper §4.1): "a reference cannot be
  // uninitialized. The initialization profile requires the same for pointers."
  // A pointer must instead be initialized (e.g. to nullptr). Both are profile
  // policy (not a meaningless subject), so they are gated on enforcement; the
  // marker is left in place so uninit_decl / ctor_uninit_member treat the
  // entity as acknowledged rather than re-diagnosing it.
  //
  // Passing \p D makes shouldEmitProfileViolation defer on a templated pattern
  // (paper / P3589R2: a rule fires on the instantiation, not the template),
  // so the parse-time handler skips template members and the rule is re-run on
  // the instantiated entity (VisitFieldDecl / VisitVarDecl), once the
  // substituted type is known to be a pointer or union.
  //
  // Both rules key on the base element type: an array of unions or pointers
  // leaves the same uninitialized elements as a single one, and the marker
  // would otherwise slip past uninit_decl (which trusts marked declarations)
  // entirely. The union rule also covers a union-typed data member of a
  // non-union class -- delayed initialization by assigning its member is just
  // as erroneous there (paper §5.6).
  QualType BaseTy =
      getASTContext().getBaseElementType(cast<ValueDecl>(D)->getType());
  bool UnionMember =
      isa<FieldDecl>(D) && cast<FieldDecl>(D)->getParent()->isUnion();
  if ((BaseTy->isUnionType() || UnionMember) &&
      shouldEmitProfileViolation("std::init", "union_marker", Loc, D))
    Diag(Loc, diag::err_init_union_marker)
        << "std::init" << (UnionMember ? 1 : isa<FieldDecl>(D) ? 2 : 0);
  else if (BaseTy->isPointerType() &&
           shouldEmitProfileViolation("std::init", "pointer_marker", Loc, D))
    Diag(Loc, diag::err_init_uninit_pointer_marker) << "std::init";
}

bool SemaProfiles::diagnoseInvalidUninitMarker(const Decl *D,
                                               SourceLocation AttrLoc,
                                               bool Diagnose) {
  const auto *VD = dyn_cast<ValueDecl>(D);
  if (!VD)
    return false;
  QualType T = VD->getType();

  // A dependent subject is validated at instantiation instead, once the
  // substituted type is known (Sema::InstantiateAttrs).
  if (T->isDependentType())
    return false;

  if (T->isReferenceType()) {
    if (Diagnose)
      Diag(AttrLoc, diag::err_uninit_attr_invalid_subject) << /*Reference=*/0u;
    return true;
  }
  return false;
}

bool SemaProfiles::diagnoseInvalidRefToUninitMarker(const Decl *D,
                                                    SourceLocation AttrLoc,
                                                    bool Diagnose) {
  QualType T;
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    T = FD->getReturnType();
  else
    T = cast<ValueDecl>(D)->getType();

  // A dependent subject is validated at instantiation instead, once the
  // substituted type is known (Sema::InstantiateAttrs).
  if (T->isDependentType())
    return false;

  if ((!T->isPointerType() && !T->isReferenceType()) ||
      T->isFunctionPointerType() || T->isFunctionReferenceType()) {
    if (Diagnose)
      Diag(AttrLoc, diag::err_ref_to_uninit_attr_invalid_type);
    return true;
  }
  return false;
}

// std::init / ref_to_uninit (paper §5). Two mutually-recursive local
// recognizers over the syntactic form of a source expression -- no flow
// analysis and no type-system tracking. Uninitialized storage is only ever
// introduced by an explicit [[uninit]] / [[ref_to_uninit]] marker.
//
// The classification is tri-state: a recognized form is Initialized or
// Uninitialized, while an unrecognized one (pointer arithmetic, an
// integer-to-pointer cast, a call through a function pointer) is Unknown rather
// than assumed Initialized. Callers wanting a plain "is it uninitialized?"
// answer (SemaProfiles::refersToUninitializedMemory, the read-through check)
// treat
// Unknown as not uninitialized.
//
// How the classified expression is being accessed is carried by
// UninitAccessOpts below.
enum class UninitStorage { Initialized, Uninitialized, Unknown };

// How an expression is being used, for the uninit recognizers.
//
// DropTopLevelUninit: a *directly named* [[uninit]] entity does not count as
// uninitialized. A value access of such an entity is owned elsewhere: a named
// [[uninit]] object by the CFG uninit_read pass, a current-object member by
// the ctor-body pass, an [[uninit]] member of a constructor-less aggregate
// local by the local-aggregate pass (all three credit assignments), and a
// marked member of an object with a user-provided constructor reached through
// any other object is deliberately trusted (paper §5.1: its constructor body
// may have assigned it, which local analysis cannot see). So the read-through
// check must not second-guess them -- and for a store, writing the whole
// named entity IS its initialization (paper §4.5: for a built-in type, a
// write is its initialization). The flag is cleared at the first *deeper*
// subobject step (a member's member, an array element), where no flow pass
// tracks the storage and only whole-object construct_at could re-initialize
// (paper §5.4).
//
// TrustRefToUninit: [[ref_to_uninit]] markers are ignored -- the storage
// reached through a marked pointer/reference (or returned by a marked
// function) classifies as Unknown rather than Uninitialized. Stores use this:
// a scalar write through the marker is the pointee's initialization (paper
// §4.5), and verifying class-type writes (construct_at) is a deferred slice,
// so a store through the marker must be neither banned nor endorsed.
// SubscriptBase: the classification runs below an element access (p[i]),
// where pointee store credit must not apply: element-wise state is
// untrackable by design (paper §5.4/§5.5 ban random access through the
// marker), so only the whole-`*p` form is ever credited. Purely syntactic:
// p[0] is not credited even though it denotes the same storage as *p.
//
// Credit: when non-null, the recognizers consult the parse-order store
// credit recorded by SemaProfiles::recordInitProfileStore -- a credited
// entity classifies as Initialized. Null in the constexpr presets; the
// checking entry points attach it via withCredit.
struct UninitAccessOpts {
  bool DropTopLevelUninit = false;
  bool TrustRefToUninit = false;
  bool SubscriptBase = false;
  const SemaProfiles *Credit = nullptr;

  UninitAccessOpts withoutTopLevelDrop() const {
    return {false, TrustRefToUninit, SubscriptBase, Credit};
  }
  UninitAccessOpts withSubscriptBase() const {
    return {DropTopLevelUninit, TrustRefToUninit, true, Credit};
  }
  UninitAccessOpts withCredit(const SemaProfiles *SP) const {
    return {DropTopLevelUninit, TrustRefToUninit, SubscriptBase, SP};
  }
};

// Presets: a binding source (markers count everywhere), a value read (the
// top-level drop applies), and a scalar store (additionally, storage reached
// through [[ref_to_uninit]] is trusted).
constexpr UninitAccessOpts UninitBindAccess{};
constexpr UninitAccessOpts UninitReadAccess{/*DropTopLevelUninit=*/true};
constexpr UninitAccessOpts UninitWriteAccess{/*DropTopLevelUninit=*/true,
                                             /*TrustRefToUninit=*/true};

// Combine the arms of a conditional: Uninitialized dominates (either arm may be
// taken), then Unknown, else Initialized.
static UninitStorage combineArms(UninitStorage A, UninitStorage B) {
  if (A == UninitStorage::Uninitialized || B == UninitStorage::Uninitialized)
    return UninitStorage::Uninitialized;
  if (A == UninitStorage::Unknown || B == UninitStorage::Unknown)
    return UninitStorage::Unknown;
  return UninitStorage::Initialized;
}

static UninitStorage
glvalueDenotesUninitStorage(ASTContext &Ctx, const Expr *E,
                            UninitAccessOpts Opts = UninitBindAccess);

const ValueDecl *SemaProfiles::getDirectlyNamedDecl(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl();
  if (const auto *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl();
  return nullptr;
}

// True if E denotes the current object: `this` (the implicit/explicit pointer
// of an arrow access) or `*this` (the object lvalue of a dot access). A local
// twin of AnalysisBasedWarnings.cpp's isCurrentObjectBase (the CFG passes'
// helper); each file keeps its recognizer vocabulary self-contained.
static bool isCurrentObjectExpr(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  if (isa<CXXThisExpr>(E))
    return true;
  const auto *UO = dyn_cast<UnaryOperator>(E);
  return UO && UO->getOpcode() == UO_Deref &&
         isa<CXXThisExpr>(UO->getSubExpr()->IgnoreParenImpCasts());
}

const Decl *SemaProfiles::resolveMemberStoreBase(const MemberExpr *ME) const {
  const Expr *Base = ME->getBase()->IgnoreParenImpCasts();
  // The current object: this->m, the implicit m, or (*this).m. Keyed on the
  // enclosing function declaration (`this` cannot be reseated, so the key is
  // stable for the whole body); AllowLambda gives a lambda body inside a
  // member function its own key, so its stores and the enclosing function's
  // never share credit. No current function (e.g. an NSDMI parse) is
  // untrackable.
  if (isCurrentObjectExpr(Base))
    return SemaRef.getCurFunctionDecl(/*AllowLambda=*/true);
  // A directly named local object: a dot access on a local-storage,
  // non-reference VarDecl (a by-value parameter is its own object and
  // qualifies). A reference base is an alias to an object also reachable
  // under other names, and an arrow base reaches the object through an
  // arbitrary (reseatable) pointer value -- both untrackable per object, the
  // same aliasing boundary that keeps fields of parameter-reached objects
  // uncredited. A deeper base (a.b.m) is §5.4's rejected deep
  // delayed-initialization tracking.
  if (!ME->isArrow())
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Base))
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
          VD && VD->hasLocalStorage() && !VD->getType()->isReferenceType())
        return VD;
  return nullptr;
}

// Pass-through forms shared by the pointer and glvalue recognizers, which are
// transparent to their operand: a single-element braced initializer { e }
// binds from e (modeling
// MismatchingNewDeleteDetector::getNewExprFromInitListOrExpr); a conditional
// is uninit if either arm is, so a value that may be uninit forces a marked
// target; a comma yields its right operand. \p EmptyListState classifies an
// empty braced list: {} value-initializes a pointer to null, which the
// pointer recognizer classifies Unknown (the null policy at its null arm),
// while a glvalue has no empty-list form (Unknown); a multi-element list is
// Unknown for both. Returns std::nullopt when E is not a pass-through form.
template <typename RecurseFn>
static std::optional<UninitStorage>
classifyUninitPassThrough(const Expr *E, UninitStorage EmptyListState,
                          RecurseFn Recurse) {
  if (const auto *ILE = dyn_cast<InitListExpr>(E)) {
    if (ILE->getNumInits() == 1)
      return Recurse(ILE->getInit(0));
    return ILE->getNumInits() == 0 ? EmptyListState : UninitStorage::Unknown;
  }
  if (const auto *CO = dyn_cast<ConditionalOperator>(E))
    return combineArms(Recurse(CO->getTrueExpr()),
                       Recurse(CO->getFalseExpr()));
  if (const auto *BO = dyn_cast<BinaryOperator>(E); BO && BO->isCommaOp())
    return Recurse(BO->getRHS());
  return std::nullopt;
}

// A call to a [[ref_to_uninit]]-returning function yields uninitialized
// storage (the pointed-to memory, or the returned referent) -- deferred to
// Unknown when the marker is trusted (a store). An unmarked direct callee is
// trusted Initialized (paper §4.3); a call with no direct callee (through a
// function pointer) is Unknown. Shared by both recognizers.
static UninitStorage classifyRefToUninitCallee(const CallExpr *CE,
                                               UninitAccessOpts Opts) {
  if (const FunctionDecl *FD = CE->getDirectCallee()) {
    if (!FD->hasAttr<RefToUninitAttr>())
      return UninitStorage::Initialized;
    return Opts.TrustRefToUninit ? UninitStorage::Unknown
                                 : UninitStorage::Uninitialized;
  }
  return UninitStorage::Unknown;
}

// \p E is a pointer prvalue. Classifies whether it points to uninitialized
// storage.
static UninitStorage
pointerRefersToUninitStorage(ASTContext &Ctx, const Expr *E,
                             UninitAccessOpts Opts = UninitBindAccess) {
  if (!E)
    return UninitStorage::Unknown;
  E = E->IgnoreParenImpCasts();

  // An empty braced list value-initializes a pointer to null, so it takes the
  // null classification below (Unknown), keeping `= {}` and `= nullptr`
  // consistent.
  if (auto PassThrough = classifyUninitPassThrough(
          E, /*EmptyListState=*/UninitStorage::Unknown,
          [&](const Expr *Sub) {
            return pointerRefersToUninitStorage(Ctx, Sub, Opts);
          }))
    return *PassThrough;

  // A null pointer refers to no object, so it is consistent with both a
  // marked target (the marker means "zero or more uninitialized objects",
  // paper §8) and an unmarked one (paper §4.3's f1(p2) example): Unknown,
  // which neither direction diagnoses. A dedicated UninitStorage::Null state
  // was considered and deferred -- behaviorally identical today; add it only
  // when a future rule (e.g. construct_at on null) needs to distinguish null
  // from unclassifiable.
  if (E->isNullPointerConstant(Ctx, Expr::NPC_ValueDependentIsNotNull) !=
      Expr::NPCK_NotNull)
    return UninitStorage::Unknown;

  // Array-to-pointer decay has been stripped above, leaving the array glvalue.
  // Clear the top-level drop here, like the member-access arm: neither the CFG
  // uninit_read pass nor the ctor-body pass tracks array elements, and
  // element-wise delayed initialization of an [[uninit]] array is itself
  // banned (paper §5.5), so below an element access the marker counts even
  // for a value access.
  if (E->getType()->isArrayType())
    return glvalueDenotesUninitStorage(Ctx, E, Opts.withoutTopLevelDrop());

  // &G, where G denotes uninitialized storage.
  if (const auto *UO = dyn_cast<UnaryOperator>(E))
    if (UO->getOpcode() == UO_AddrOf)
      return glvalueDenotesUninitStorage(Ctx, UO->getSubExpr(), Opts);

  // A value of a [[ref_to_uninit]] pointer is Uninitialized (Unknown when the
  // marker is trusted); an unmarked named pointer is a trusted Initialized
  // pointer (paper §4.3).
  if (const ValueDecl *VD = SemaProfiles::getDirectlyNamedDecl(E)) {
    if (!VD->hasAttr<RefToUninitAttr>()) {
      // An unmarked *local* whose declaration initializer is null -- a null
      // pointer constant, or an empty braced list, which value-initializes to
      // null -- is a null source like the literal (paper §4.3's f1(p2)
      // example): Unknown. Reassignment after the null init is a documented,
      // accepted missed diagnostic (parse-order leniency). Deliberately
      // excluded: globals/extern (an extern pointer may be initialized
      // elsewhere, and keeping them Initialized preserves the
      // marked-direction diagnostics), null-NSDMI fields, and parameters --
      // a ParmVarDecl's getInit() is its *default argument*, which is not
      // the parameter's value on most calls. A *marked* decl keeps its
      // marker classification below (respect the explicit marker).
      if (const auto *Var = dyn_cast<VarDecl>(VD);
          Var && Var->hasLocalStorage() && !isa<ParmVarDecl>(Var)) {
        if (const Expr *Init = Var->getInit()) {
          const Expr *InnerInit = Init->IgnoreParenImpCasts();
          const auto *ILE = dyn_cast<InitListExpr>(InnerInit);
          if ((ILE && ILE->getNumInits() == 0) ||
              InnerInit->isNullPointerConstant(
                  Ctx, Expr::NPC_ValueDependentIsNotNull) != Expr::NPCK_NotNull)
            return UninitStorage::Unknown;
        }
      }
      return UninitStorage::Initialized;
    }
    // Parse-order store credit: after a whole-`*p` store, the marked
    // pointer's pointee counts as initialized (paper §4.3: "p no longer
    // refers to uninitialized memory") for further whole-`*p` accesses --
    // until the pointer is reseated, which clears the credit. The consult
    // sits before the TrustRefToUninit branch (under the write preset the
    // outcome merely changes Unknown to Initialized, both "not
    // Uninitialized": no preset regression) and is skipped below an element
    // access (SubscriptBase), preserving §5.4's random-access ban.
    if (Opts.Credit && !Opts.SubscriptBase &&
        Opts.Credit->hasPointeeStoreCredit(VD))
      return UninitStorage::Initialized;
    return Opts.TrustRefToUninit ? UninitStorage::Unknown
                                 : UninitStorage::Uninitialized;
  }
  if (const auto *CE = dyn_cast<CallExpr>(E))
    return classifyRefToUninitCallee(CE, Opts);

  // A default-initialized new-expression (none init style: no initializer
  // written) whose allocated type's default-initialization leaves a scalar
  // subobject indeterminate produces uninitialized free-store memory (paper
  // §1.2/§4.3), like a [[ref_to_uninit]] allocator. The style gates this
  // rather
  // than hasInitializer(), which is also true for new Agg -- default-
  // initializing a class synthesizes a (possibly trivial) constructor call.
  // new T(...) / new T{} are value- or list-initialized; a user-provided
  // default constructor is trusted by defaultInitLeavesScalarIndeterminate.
  if (const auto *NE = dyn_cast<CXXNewExpr>(E)) {
    if (NE->getInitializationStyle() != CXXNewInitializationStyle::None)
      return UninitStorage::Initialized;
    llvm::SmallPtrSet<const CXXRecordDecl *, 8> Visited;
    return defaultInitLeavesScalarIndeterminateImpl(Ctx, NE->getAllocatedType(),
                                                    /*HonorUninitMarkers=*/true,
                                                    Visited)
               ? UninitStorage::Uninitialized
               : UninitStorage::Initialized;
  }

  // Paper §4.3: a [[ref_to_uninit]] pointer cast to another pointer type is
  // itself [[ref_to_uninit]]. Implicit casts were already stripped above, so
  // this only looks through an explicit pointer-to-pointer cast; a pointer
  // manufactured from an integer (operand not a pointer) is Unknown.
  if (const auto *CE = dyn_cast<ExplicitCastExpr>(E))
    if (CE->getSubExpr()->getType()->isPointerType())
      return pointerRefersToUninitStorage(Ctx, CE->getSubExpr(), Opts);

  return UninitStorage::Unknown;
}

// \p E is a glvalue. Classifies whether it denotes uninitialized storage.
static UninitStorage glvalueDenotesUninitStorage(ASTContext &Ctx, const Expr *E,
                                                 UninitAccessOpts Opts) {
  if (!E)
    return UninitStorage::Unknown;
  E = E->IgnoreParenImpCasts();

  if (auto PassThrough = classifyUninitPassThrough(
          E, /*EmptyListState=*/UninitStorage::Unknown,
          [&](const Expr *Sub) {
            return glvalueDenotesUninitStorage(Ctx, Sub, Opts);
          }))
    return *PassThrough;

  // A named entity denotes uninitialized storage if it is [[uninit]], or
  // if it is a reference marked [[ref_to_uninit]] (the glvalue is its referent,
  // which is uninitialized). A [[ref_to_uninit]] *pointer* named here denotes
  // the pointer object itself -- which is initialized -- so it does not count.
  // Under the top-level drop the [[uninit]] arm is skipped: a value access of
  // a directly named [[uninit]] object is the flow-based passes' territory, so
  // only a [[ref_to_uninit]] reference (or indirection, handled below) still
  // counts. Parse-order store credit clears both arms: a whole-entity store
  // is the [[uninit]] entity's initialization (paper §4.2/§4.5), and a store
  // through a marked reference initializes its referent (§4.3; references
  // cannot be reseated, so that credit never lapses).
  auto DeclDenotesUninit = [&](const ValueDecl *VD) {
    return (!Opts.DropTopLevelUninit && VD->hasAttr<UninitAttr>() &&
            !(Opts.Credit && Opts.Credit->hasWholeObjectStoreCredit(VD))) ||
           (!Opts.TrustRefToUninit && VD->getType()->isReferenceType() &&
            VD->hasAttr<RefToUninitAttr>() &&
            !(Opts.Credit && Opts.Credit->hasPointeeStoreCredit(VD)));
  };
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    return DeclDenotesUninit(DRE->getDecl()) ? UninitStorage::Uninitialized
                                             : UninitStorage::Initialized;
  if (const auto *ME = dyn_cast<MemberExpr>(E)) {
    // a->m reaches m through the pointer a (object *a); a.m through the
    // glvalue a. When m does not itself denote uninit storage, the subobject is
    // uninit exactly when its base is. The base recursion clears the top-level
    // drop: the drop exists because a directly named [[uninit]] entity's value
    // accesses are owned by the flow passes or deliberately trusted (see the
    // UninitAccessOpts comment above), but nothing tracks a subobject reached
    // through a *further* member access -- and member-wise delayed
    // initialization of an [[uninit]] object is itself banned (paper §5.4;
    // only whole-object construct_at re-initializes, which is uniformly
    // unmodeled) -- so below the top level the marker counts for every access.
    //
    // Parse-order member store credit: after `a.m = 5` / `this->m = 5`, the
    // marked member of that *specific* base object counts as initialized
    // (paper §4.2: "After initialization, the object is no longer
    // [[uninit]]"; §6: assignment initializes a built-in) -- covering both
    // `a.m` (a reference binding lands in this arm directly) and `&a.m` (the
    // UO_AddrOf arm recurses here). The consult keys on the same base
    // identity the recording resolved, so the same member observed through
    // any other object -- including a copy (§5.2: a copy does not inherit
    // credit) -- stays uncredited. It applies at any chain depth: the map
    // only ever holds whole-member stores (which initialize the entire
    // member), and in practice only scalar members (see
    // recordInitProfileStore), which have no subobjects to chain through.
    const ValueDecl *MD = ME->getMemberDecl();
    if (const auto *F = dyn_cast<FieldDecl>(MD);
        F && Opts.Credit && F->hasAttr<UninitAttr>() &&
        Opts.Credit->hasMemberStoreCredit(
            Opts.Credit->resolveMemberStoreBase(ME), F))
      return UninitStorage::Initialized;
    if (DeclDenotesUninit(MD))
      return UninitStorage::Uninitialized;
    return ME->isArrow() ? pointerRefersToUninitStorage(
                               Ctx, ME->getBase(), Opts.withoutTopLevelDrop())
                         : glvalueDenotesUninitStorage(
                               Ctx, ME->getBase(), Opts.withoutTopLevelDrop());
  }
  // A call to a [[ref_to_uninit]]-returning reference function: the referent
  // it returns is uninitialized.
  if (const auto *CE = dyn_cast<CallExpr>(E))
    return classifyRefToUninitCallee(CE, Opts);
  // An element access classifies like its base, but pointee store credit
  // must not apply below it (SubscriptBase): `*p = 5;` never legalizes
  // `p[1]` -- the pointee may be an array with only element 0 written, and
  // element-wise state is untrackable by design (paper §5.4/§5.5).
  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E))
    return pointerRefersToUninitStorage(Ctx, ASE->getBase(),
                                        Opts.withSubscriptBase());

  // *p, where p points to uninitialized storage.
  if (const auto *UO = dyn_cast<UnaryOperator>(E))
    if (UO->getOpcode() == UO_Deref)
      return pointerRefersToUninitStorage(Ctx, UO->getSubExpr(), Opts);

  // A reference cast (an explicit cast yielding a glvalue) denotes the same
  // storage as its operand; propagate. Symmetric to the pointer-cast arm.
  if (const auto *CE = dyn_cast<ExplicitCastExpr>(E))
    if (CE->getSubExpr()->isGLValue())
      return glvalueDenotesUninitStorage(Ctx, CE->getSubExpr(), Opts);

  return UninitStorage::Unknown;
}

// Dispatches a binding source to the pointer or glvalue recognizer.
static UninitStorage
classifyUninitSource(ASTContext &Ctx, const Expr *E, bool IsReference,
                     UninitAccessOpts Opts = UninitBindAccess) {
  return IsReference ? glvalueDenotesUninitStorage(Ctx, E, Opts)
                     : pointerRefersToUninitStorage(Ctx, E, Opts);
}

bool SemaProfiles::refersToUninitializedMemory(const Expr *E,
                                               bool IsReference) const {
  return classifyUninitSource(getASTContext(), E, IsReference,
                              UninitBindAccess.withCredit(this)) ==
         UninitStorage::Uninitialized;
}

void SemaProfiles::checkInitProfileRefToUninit(SourceLocation Loc,
                                        bool TargetIsRefToUninit,
                                        bool IsReference, const Expr *Src,
                                        const Decl *D) {
  // A RecoveryExpr is a placeholder for an initialization that already failed,
  // not a source the user wrote, so it must not drive this rule.
  if (!Src || isa<RecoveryExpr>(Src->IgnoreParens()))
    return;
  // An instantiation-dependent source cannot be classified yet; its construct
  // is always rebuilt at instantiation, re-running this funnel with the
  // substituted source. A non-dependent source is checked here, at definition
  // time; if the construct is rebuilt at instantiation anyway (a local
  // operand, a call argument, a return), the same diagnostic repeats there --
  // accepted for now. Decl-carrying callers are exempt: they defer via the
  // D->isTemplated() check in shouldEmitProfileViolation and fire on the
  // instantiated declaration.
  if (!D && Src->isInstantiationDependent())
    return;
  static constexpr StringRef Profile = "std::init";
  static constexpr StringRef Rule = "ref_to_uninit";
  if (!shouldEmitProfileViolation(Profile, Rule, Loc, D))
    return;
  UninitStorage SrcState = classifyUninitSource(
      getASTContext(), Src, IsReference, UninitBindAccess.withCredit(this));
  unsigned IsRef = IsReference ? 1 : 0;
  // A marked target is a violation only against an affirmatively Initialized
  // source: an Unknown one (pointer arithmetic, an integer-to-pointer cast, a
  // call through a function pointer) cannot be proven initialized, so rejecting
  // it would be a false positive. An unmarked target is diagnosed only against
  // an affirmatively Uninitialized source (Unknown stays a missed diagnostic).
  if (TargetIsRefToUninit && SrcState == UninitStorage::Initialized)
    Diag(Loc, diag::err_init_ref_to_uninit_requires_uninit) << Profile << IsRef;
  else if (!TargetIsRefToUninit && SrcState == UninitStorage::Uninitialized)
    Diag(Loc, diag::err_init_uninit_requires_ref_to_uninit) << Profile << IsRef;
}

void SemaProfiles::checkInitProfileRefToUninitBinding(SourceLocation Loc,
                                           const ValueDecl *Target, QualType T,
                                           const Expr *Src, const Decl *D) {
  if (!getLangOpts().Profiles || T.isNull() || T->isDependentType() ||
      (!T->isPointerType() && !T->isReferenceType()))
    return;
  // A null Target is a binding site with no declaration to carry the marker
  // (a parameter of a call through a function pointer): always unmarked.
  checkInitProfileRefToUninit(Loc,
                              Target && Target->hasAttr<RefToUninitAttr>(),
                              T->isReferenceType(), Src, D);
}

void SemaProfiles::checkInitProfileVariadicArgument(const Expr *Arg) {
  // std::init / ref_to_uninit (paper §5): a variadic argument never reaches
  // parameter copy-initialization, and a `...` parameter cannot carry
  // [[ref_to_uninit]], so a pointer passed through it is checked as an
  // unmarked target (paper §7.2: passing uninitialized memory needs an
  // appropriately declared callee). Value reads of the promoted argument
  // already funnel through the lvalue-to-rvalue chokepoint; the pointer
  // binding is the only direction added here. Called from the two C++
  // variadic promotion loops -- Sema::GatherArgumentsForCall and
  // Sema::BuildCallToObjectOfClassType (functors, variadic lambdas) -- not
  // from Sema::DefaultVariadicArgumentPromotion itself, whose other callers
  // re-promote already-promoted arguments (the os_log builtin check) or
  // promote during ObjC method matching, where a check would double-fire.
  if (!getLangOpts().Profiles || !Arg || !Arg->getType()->isPointerType())
    return;
  checkInitProfileRefToUninit(Arg->getExprLoc(), /*TargetIsRefToUninit=*/false,
                              /*IsReference=*/false, Arg);
}

void SemaProfiles::checkInitProfileRefCapture(SourceLocation Loc,
                                              const ValueDecl *Var) {
  if (!getLangOpts().Profiles)
    return;
  // Mirrors the glvalue recognizer's named-entity arm: the captured variable
  // denotes uninitialized storage if it is [[uninit]], or if it is a
  // [[ref_to_uninit]] reference (the capture binds to its referent) -- in
  // both cases unless parse-order store credit says it has been initialized
  // (u = 5; then a by-ref capture of u is accepted, symmetric with &u). A
  // copy capture is not this check's: it reads the variable in the enclosing
  // function's CFG, which is the flow-based uninit_read pass's territory.
  bool UninitNoCredit =
      Var->hasAttr<UninitAttr>() && !hasWholeObjectStoreCredit(Var);
  bool RefNoCredit = Var->getType()->isReferenceType() &&
                     Var->hasAttr<RefToUninitAttr>() &&
                     !hasPointeeStoreCredit(Var);
  if (!UninitNoCredit && !RefNoCredit)
    return;
  // The only Expr-less deferral here: an instantiation-dependent captured
  // type defers to instantiation, where TreeTransform's unconditional lambda
  // rebuild re-processes the capture. A concrete capture fires at definition
  // time and repeats on that same rebuild -- accepted for now.
  if (Var->getType()->isInstantiationDependentType())
    return;
  if (!shouldEmitProfileViolation("std::init", "ref_to_uninit", Loc))
    return;
  Diag(Loc, diag::err_init_uninit_ref_capture) << "std::init" << Var;
}

void SemaProfiles::checkInitProfileObjectArgument(const Expr *Object,
                                                  const CXXMethodDecl *Method) {
  // A RecoveryExpr is a placeholder for an expression that already failed, not
  // an object argument the user wrote, so it must not drive this rule.
  if (!getLangOpts().Profiles || !Object ||
      isa<RecoveryExpr>(Object->IgnoreParens()))
    return;
  // Destroying uninitialized storage is the deferred destroy_at slice (the
  // paper models destruction, like construct_at, as a lifetime operation);
  // implicit scope-exit destructions never reach this funnel, so diagnosing
  // only the explicit s.~S() spelling would be an inconsistent sliver.
  if (isa<CXXDestructorDecl>(Method))
    return;
  // A static call operator (C++23) has no implicit object parameter: its
  // object argument is evaluated but its value is never used, exactly like a
  // static member function named through an object -- whose call path never
  // reaches this funnel. BuildCallToObjectOfClassType converts the object
  // argument for static call operators all the same, so skip them here.
  if (Method->isStatic())
    return;
  // An instantiation-dependent object argument cannot be classified yet; its
  // call is always rebuilt at instantiation, re-running this funnel with the
  // substituted object. A non-dependent call fires at definition time and
  // repeats if the call is rebuilt at instantiation anyway -- accepted.
  if (Object->isInstantiationDependent())
    return;
  if (!shouldEmitProfileViolation("std::init", "ref_to_uninit",
                                  Object->getExprLoc()))
    return;
  // An arrow call's object argument arrives as the pointer expression, a dot
  // call's as the object glvalue; dispatch the recognizer accordingly.
  bool IsPointer = Object->getType()->isPointerType();
  if (!refersToUninitializedMemory(Object, /*IsReference=*/!IsPointer))
    return;
  Diag(Object->getExprLoc(), diag::err_init_member_call_on_uninit)
      << "std::init" << Method;
}

// The read-through diagnostic distinguishes indirection through a
// [[ref_to_uninit]] pointer/reference from a subobject read of a named
// [[uninit]] object, which involves no [[ref_to_uninit]] entity. Approximate
// but sufficient for phrasing: walk up the dot member / array-element chain;
// the read is the latter form iff the chain reaches an [[uninit]]-marked
// member (e.g. the class-type member in this->agg.f) or bottoms out at a
// named [[uninit]] declaration. An arrow access or a subscript on a pointer
// reaches its object through a pointer, so the pointer wording applies from
// there on.
static bool isMemberChainOfUninitObject(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  while (true) {
    if (const auto *ME = dyn_cast<MemberExpr>(E)) {
      if (ME->getMemberDecl()->hasAttr<UninitAttr>())
        return true;
      if (ME->isArrow())
        return false;
      E = ME->getBase()->IgnoreParenImpCasts();
      continue;
    }
    // a[i] and *a on an array glvalue (decay stripped below) are subobject
    // accesses like a dot member access; on a pointer base they reach the
    // object through the pointer.
    if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
      const Expr *Base = ASE->getBase()->IgnoreParenImpCasts();
      if (!Base->getType()->isArrayType())
        return false;
      E = Base;
      continue;
    }
    if (const auto *UO = dyn_cast<UnaryOperator>(E);
        UO && UO->getOpcode() == UO_Deref) {
      const Expr *Sub = UO->getSubExpr()->IgnoreParenImpCasts();
      if (!Sub->getType()->isArrayType())
        return false;
      E = Sub;
      continue;
    }
    break;
  }
  const auto *DRE = dyn_cast<DeclRefExpr>(E);
  return DRE && DRE->getDecl()->hasAttr<UninitAttr>();
}

void SemaProfiles::checkInitProfileReadThrough(SourceLocation Loc,
                                               const Expr *Glvalue,
                                               QualType ValueType) {
  // A RecoveryExpr is a placeholder for an expression that already failed, not
  // a read the user wrote, so it must not drive this rule.
  if (!Glvalue || isa<RecoveryExpr>(Glvalue->IgnoreParens()))
    return;
  // An instantiation-dependent glvalue cannot be classified yet; its read is
  // always rebuilt at instantiation, where this check re-runs with the
  // substituted operand. A non-dependent read fires at definition time and
  // repeats if the read is rebuilt at instantiation anyway -- accepted.
  if (Glvalue->isInstantiationDependent())
    return;
  // Paper §4.5: reading an uninitialized std::byte is permitted.
  if (getASTContext().getBaseElementType(ValueType)->isStdByteType())
    return;
  if (!shouldEmitProfileViolation("std::init", "uninit_read", Loc))
    return;
  if (glvalueDenotesUninitStorage(getASTContext(), Glvalue,
                                  UninitReadAccess.withCredit(this)) !=
      UninitStorage::Uninitialized)
    return;
  Diag(Loc, diag::err_init_uninit_read_through)
      << "std::init" << (isMemberChainOfUninitObject(Glvalue) ? 1 : 0);
}

void SemaProfiles::checkInitProfileSubobjectWrite(SourceLocation Loc,
                                                  const Expr *LHS) {
  // A RecoveryExpr is a placeholder for an expression that already failed, not
  // a store the user wrote, so it must not drive this rule.
  if (!LHS || isa<RecoveryExpr>(LHS->IgnoreParens()))
    return;
  // An instantiation-dependent store target cannot be classified yet; its
  // assignment is always rebuilt at instantiation, where this check re-runs
  // with the substituted LHS. A non-dependent store fires at definition time
  // and repeats if the assignment is rebuilt at instantiation anyway --
  // accepted.
  if (LHS->isInstantiationDependent())
    return;
  // Paper §4.5: an uninitialized std::byte may be manipulated freely.
  if (getASTContext().getBaseElementType(LHS->getType())->isStdByteType())
    return;
  if (!shouldEmitProfileViolation("std::init", "uninit_write", Loc))
    return;
  if (glvalueDenotesUninitStorage(getASTContext(), LHS,
                                  UninitWriteAccess.withCredit(this)) !=
      UninitStorage::Uninitialized)
    return;
  Diag(Loc, diag::err_init_uninit_subobject_write)
      << "std::init" << !isa<MemberExpr>(LHS->IgnoreParenImpCasts());
}

void SemaProfiles::checkInitProfilePointerAssignment(Expr *LHS, Expr *RHS,
                                                     SourceLocation OpLoc) {
  // References cannot be reseated, so only pointer assignment applies. The
  // marker is read when the LHS directly names a pointer entity; any other
  // lvalue (e.g. *pp, arr[i]) cannot carry a local marker, so it is the
  // default unmarked pointer (paper §4.3) and must not be bound to
  // uninitialized memory.
  if (!LHS->getType()->isPointerType())
    return;
  // An instantiation-dependent LHS (e.g. an unresolved member access) has no
  // readable marker yet -- getDirectlyNamedDecl would report it unmarked, a
  // false positive when the instantiated entity is [[ref_to_uninit]]. The
  // assignment is rebuilt at instantiation, where the marker is concrete.
  // The source's dependence is the funnel's to defer on.
  if (LHS->isInstantiationDependent())
    return;
  const ValueDecl *VD = getDirectlyNamedDecl(LHS);
  checkInitProfileRefToUninit(OpLoc, VD && VD->hasAttr<RefToUninitAttr>(),
                              /*IsReference=*/false, RHS);
}

void SemaProfiles::checkInitProfileAssignmentOperands(BinaryOperatorKind Opc,
                                                      Expr *LHSExpr,
                                                      bool IsCompound,
                                                      SourceLocation OpLoc) {
  // A compound assignment reads the old value but builds no lvalue-to-rvalue
  // node for it, so the DefaultLvalueConversion read-through chokepoint never
  // sees the load; check it here. The shift forms are the exception:
  // CheckShiftOperands promotes their LHS through DefaultLvalueConversion,
  // which has already fired for them.
  if (IsCompound && Opc != BO_ShlAssign && Opc != BO_ShrAssign)
    checkInitProfileReadThrough(LHSExpr->getExprLoc(), LHSExpr,
                                LHSExpr->getType());
  checkInitProfileSubobjectWrite(OpLoc, LHSExpr);
}

void SemaProfiles::checkInitProfileIncDec(Expr *Operand, SourceLocation OpLoc) {
  // ++/-- reads the old value with no lvalue-to-rvalue node (unlike -x or
  // !x), then stores to its operand like an assignment does to its LHS.
  checkInitProfileReadThrough(Operand->getExprLoc(), Operand,
                              Operand->getType());
  checkInitProfileSubobjectWrite(OpLoc, Operand);
  // Record last: the pre-store checks above must see pre-store state (there
  // is no RHS). ++u credits the whole entity; ++p reseats a marked pointer.
  recordInitProfileStore(Operand);
}

void SemaProfiles::recordInitProfileStore(const Expr *LHS) {
  if (!getLangOpts().Profiles || !LHS)
    return;
  // A store in an unevaluated or discarded-statement context never executes
  // (mirroring shouldEmitProfileViolation's context checks), so it earns no
  // credit. There is deliberately no enforcement or [[profiles::suppress]]
  // gate -- a suppressed store still initializes; failing to credit it would
  // turn suppression into later false positives -- and no in-template gate:
  // non-dependent code in a template is checked at definition time and must
  // find pattern-time credit (instantiations rebuild their DeclRefExprs
  // against fresh decls, so they re-record independently).
  if (SemaRef.isUnevaluatedContext())
    return;
  if (SemaRef.currentEvaluationContext().isDiscardedStatementContext())
    return;
  const Expr *E = LHS->IgnoreParenImpCasts();
  // *p = e: a store through the exact whole-`*p` lvalue of a marked
  // local/parameter pointer is the pointee's initialization (paper
  // §4.3/§4.5: for a built-in type, a write is its initialization).
  // Class-typed pointees never get here: `*sp = S{...}` resolves to a member
  // operator= (already rejected as a call on uninitialized storage), so
  // PointeeStored is only ever set for built-in-typed pointee stores.
  // Subscript stores (p[i] = e) are deliberately neither credited nor
  // invalidating: the paper bans element-wise tracking (§5.4/§5.5).
  if (const auto *UO = dyn_cast<UnaryOperator>(E);
      UO && UO->getOpcode() == UO_Deref) {
    if (const auto *VD =
            dyn_cast_or_null<VarDecl>(getDirectlyNamedDecl(UO->getSubExpr()));
        VD && VD->hasLocalStorage() && VD->getType()->isPointerType() &&
        VD->hasAttr<RefToUninitAttr>())
      InitStoreCredit[VD] |= PointeeStored;
    return;
  }
  // a.m = e / this->m = e / m = e (also `@=` and `++`, via the shared
  // hosts): a whole-member store to an [[uninit]] field of a trackable base
  // object is that member's initialization (paper §4.2: "After
  // initialization, the object is no longer [[uninit]]"; §6: ordinary
  // assignment initializes a built-in), keyed per (base, field) so unrelated
  // objects and other function bodies never share credit. Only single-level
  // bases earn credit (x.agg.m = e resolves no base -- and is itself an
  // uninit_write violation; §5.4 rejects deep delayed-initialization
  // tracking), element stores (a.m[i] = e) present a subscript, not a
  // MemberExpr, and stay uncredited, and a class-typed x.agg = e is a member
  // operator= call (rejected as a call on uninitialized storage) that never
  // reaches this built-in-assignment funnel -- so only scalar members are
  // ever credited. Member *pointee* stores (*a.p = e) took the deref arm
  // above, which keys on local pointers only: the pinned per-object aliasing
  // boundary (copies share pointees).
  if (const auto *ME = dyn_cast<MemberExpr>(E)) {
    if (const auto *F = dyn_cast<FieldDecl>(ME->getMemberDecl());
        F && F->hasAttr<UninitAttr>())
      if (const Decl *Base = resolveMemberStoreBase(ME))
        MemberStoreCredit[{Base, F}] |= WholeStored;
    return;
  }
  // Only a directly named local-storage variable can be credited beyond
  // this point: statics fail hasLocalStorage.
  const auto *VD = dyn_cast_or_null<VarDecl>(getDirectlyNamedDecl(E));
  if (!VD || !VD->hasLocalStorage())
    return;
  // u = e (also u @= e and ++u, via the inc-dec host): assigning the whole
  // [[uninit]] entity is its initialization (paper §4.2/§4.5).
  if (VD->hasAttr<UninitAttr>()) {
    InitStoreCredit[VD] |= WholeStored;
    return;
  }
  if (!VD->hasAttr<RefToUninitAttr>())
    return;
  if (VD->getType()->isReferenceType()) {
    // r = e stores through the marked reference to its referent; a reference
    // cannot be reseated, so the credit is never cleared.
    InitStoreCredit[VD] |= PointeeStored;
  } else if (VD->getType()->isPointerType()) {
    // p = q / p += n / ++p reseats the marked pointer: any pointee credit no
    // longer describes the new pointee. The clear lives here in the tail
    // funnel -- not in checkInitProfilePointerAssignment, which runs only
    // for plain assignment and would miss compound reseats.
    InitStoreCredit[VD] &= ~unsigned(PointeeStored);
  }
}

bool SemaProfiles::hasWholeObjectStoreCredit(const ValueDecl *VD) const {
  const auto *Var = dyn_cast<VarDecl>(VD);
  if (!Var)
    return false;
  auto It = InitStoreCredit.find(Var);
  return It != InitStoreCredit.end() && (It->second & WholeStored);
}

bool SemaProfiles::hasPointeeStoreCredit(const ValueDecl *VD) const {
  const auto *Var = dyn_cast<VarDecl>(VD);
  if (!Var)
    return false;
  auto It = InitStoreCredit.find(Var);
  return It != InitStoreCredit.end() && (It->second & PointeeStored);
}

bool SemaProfiles::hasMemberStoreCredit(const Decl *Base,
                                        const FieldDecl *F) const {
  if (!Base || !F)
    return false;
  auto It = MemberStoreCredit.find({Base, F});
  return It != MemberStoreCredit.end() && (It->second & WholeStored);
}

void SemaProfiles::checkInitProfileThrowOperand(const Expr *Operand) {
  // A thrown pointer copy-initializes the exception object, which cannot
  // carry [[ref_to_uninit]], so throwing a pointer to uninitialized memory is
  // always the unmarked-direction violation. (Reads like `throw *p` funnel
  // through the read-through check instead.)
  QualType ExceptionObjectTy =
      getASTContext().getExceptionObjectType(Operand->getType());
  if (!ExceptionObjectTy->isPointerType())
    return;
  checkInitProfileRefToUninit(Operand->getExprLoc(),
                              /*TargetIsRefToUninit=*/false,
                              /*IsReference=*/false, Operand);
}

void SemaProfiles::checkInitProfileNewInitializer(QualType AllocType,
                                                  Expr *Init) {
  // A written initializer for an allocated pointer binds it like a variable
  // initialization -- but a heap pointer object cannot carry
  // [[ref_to_uninit]], so binding it to uninitialized memory is always the
  // unmarked-direction violation. A braced `new T*{&x}` presents the
  // InitListExpr, which the recognizer's single-element pass-through looks
  // through. An instantiation-dependent allocated type (note that a
  // dependent-pointee `T*` still passes isPointerType) defers to the
  // instantiation rebuild, which re-runs this check with the concrete type.
  if (!AllocType->isPointerType() ||
      AllocType->isInstantiationDependentType() || !Init)
    return;
  checkInitProfileRefToUninit(Init->getExprLoc(),
                              /*TargetIsRefToUninit=*/false,
                              /*IsReference=*/false, Init);
}

namespace {
// Row for the unified finalization dispatch shared by class-finalization
// (pattern 3) and constructor-finalization (pattern 4): a profile name plus a
// callback invoked once per finalized, non-dependent, non-invalid Node (a
// CXXRecordDecl or a CXXConstructorDecl). Adding a new profile is a single row
// in the matching table below plus a ProfileRuleError diagnostic in
// DiagnosticSemaKinds.td and a callback that consults
// SemaProfiles::shouldEmitProfileViolation before emitting.
template <class Node> struct FinalizationProfile {
  StringRef Name;
  void (*Callback)(Sema &, Node *);
};

void runTestClassFinalCallback(Sema &S, CXXRecordDecl *RD) {
  if (!S.Profiles().shouldEmitProfileViolation("test::class_final", /*Rule=*/"",
                                    RD->getLocation(), RD))
    return;
  S.Diag(RD->getLocation(), diag::err_profile_class_final_test)
      << "test::class_final" << RD;
}

void runTestCtorFinalCallback(Sema &S, CXXConstructorDecl *Ctor) {
  if (!S.Profiles().shouldEmitProfileViolation("test::ctor_final", /*Rule=*/"",
                                    Ctor->getLocation(), Ctor))
    return;
  S.Diag(Ctor->getLocation(), diag::err_profile_ctor_final_test)
      << "test::ctor_final" << Ctor->getParent();
}

void runStdInitCtorUninitMemberCallback(Sema &S, CXXConstructorDecl *Ctor) {
  // Paper §6.1: a user-provided constructor must initialize every member via
  // its member-initializer list or an NSDMI, unless the member is marked
  // [[uninit]] (whose body initialization is the deferred R7 check).
  // A plain assignment in the constructor body does not count.
  if (!Ctor->isUserProvided())
    return;

  // A union's members are mutually exclusive; a constructor initializes at most
  // one, so the "every member" rule does not apply (paper §6.5). Whether the
  // active member is set is a constructor-body flow question, deferred.
  if (Ctor->getParent()->isUnion())
    return;

  // Members and direct bases given a written initializer by this constructor.
  llvm::SmallPtrSet<const FieldDecl *, 8> Written;
  llvm::SmallPtrSet<const Type *, 4> WrittenBases;
  for (const CXXCtorInitializer *Init : Ctor->inits()) {
    if (!Init->isWritten())
      continue;
    if (Init->isAnyMemberInitializer()) {
      if (const FieldDecl *F = Init->getAnyMember())
        Written.insert(F);
    } else if (Init->isBaseInitializer()) {
      if (const Type *T = Init->getBaseClass())
        WrittenBases.insert(
            S.Context.getCanonicalType(QualType(T, 0)).getTypePtr());
    }
  }

  for (const FieldDecl *F : Ctor->getParent()->fields()) {
    // Anonymous aggregate members and unnamed bit-fields are skipped; a named
    // bit-field is checked like any other member. Reference and const members
    // already have dedicated diagnostics when left uninitialized.
    if (F->isUnnamedBitField() || !F->getDeclName() ||
        F->getType()->isReferenceType() || F->getType().isConstQualified())
      continue;
    if (F->hasAttr<UninitAttr>() || F->hasInClassInitializer() ||
        Written.count(F))
      continue;
    if (!S.Profiles().defaultInitLeavesScalarIndeterminate(F->getType(),
                                                /*HonorUninitMarkers=*/true))
      continue;
    if (!S.Profiles().shouldEmitProfileViolation(
            "std::init", "ctor_uninit_member", Ctor->getLocation(), Ctor))
      continue;
    S.Diag(Ctor->getLocation(), diag::err_init_ctor_uninit_member)
        << "std::init" << F->getDeclName();
    S.Diag(F->getLocation(), diag::note_init_uninit_member_here)
        << F->getDeclName();
  }

  // The guarantee is over the complete object (paper §5.1, §7.1), so a
  // direct base-class subobject left indeterminate is as much a violation as a
  // member. A base cannot carry an [[uninit]] marker (the attribute's subjects
  // are Var/Field), so an indeterminate base must always be initialized --
  // there
  // is no marker escape. Virtual bases are the most-derived constructor's
  // responsibility, not a local property of this constructor, so they are
  // deferred. A written base-initializer initializes the base; an implicit
  // (non-written) one is default-init, handled by the indeterminate check.
  for (const CXXBaseSpecifier &Base : Ctor->getParent()->bases()) {
    if (Base.isVirtual())
      continue;
    if (WrittenBases.count(
            S.Context.getCanonicalType(Base.getType()).getTypePtr()))
      continue;
    if (!S.Profiles().defaultInitLeavesScalarIndeterminate(Base.getType(),
                                                /*HonorUninitMarkers=*/true))
      continue;
    if (!S.Profiles().shouldEmitProfileViolation(
            "std::init", "ctor_uninit_member", Ctor->getLocation(), Ctor))
      continue;
    S.Diag(Ctor->getLocation(), diag::err_init_ctor_uninit_base)
        << "std::init" << Base.getType();
    S.Diag(Base.getBeginLoc(), diag::note_init_uninit_base_here)
        << Base.getType();
  }
}

void runStdInitUninitFieldMarkerCallback(Sema &S, CXXRecordDecl *RD) {
  // std::init / uninit_with_initializer, field flavor (paper §4.2 rule 2,
  // §5.3): [[uninit]] on a data member claims default-initialization leaves
  // the member uninitialized. When the member type's default-initialization
  // is not a no-op (a non-trivial default constructor initializes something)
  // or leaves nothing indeterminate (nothing to acknowledge), the marker is a
  // contradiction, just like a variable's explicit initializer. The NSDMI
  // case is the ActOnFinishCXXInClassMemberInitializer flavor's to diagnose;
  // this covers the initializer-less member, which only the class walk sees.
  //
  // A union's members are union_marker's territory (the marker is banned on
  // them wholesale, paper §5.6).
  if (RD->isUnion())
    return;
  for (const FieldDecl *F : RD->fields()) {
    const auto *UA = F->getAttr<UninitAttr>();
    // hasInClassInitializer is style-based, so it is true even while a
    // late-parsed NSDMI is still pending.
    if (!UA || F->isInvalidDecl() || F->hasInClassInitializer())
      continue;
    QualType BaseTy = S.Context.getBaseElementType(F->getType());
    // A union- or pointer-typed member (keyed on the same base element type)
    // already draws union_marker / pointer_marker and keeps the marker; do
    // not pile a second diagnostic on top. Load-bearing for union members: a
    // union with a non-trivial member has a deleted -- hence non-trivial --
    // default constructor and would otherwise draw both.
    if (BaseTy->isUnionType() || BaseTy->isPointerType())
      continue;
    // std::byte may be left uninitialized (paper §4), mirroring
    // checkInitProfileUninitDecl.
    if (BaseTy->isStdByteType())
      continue;
    if (S.Profiles().defaultInitIsVacuous(F->getType()))
      continue;
    // Decl-aware gate: defers on templated patterns (instantiations re-fire
    // through CheckCompletedCXXClass) and honors [[profiles::suppress]] on
    // the field or the enclosing class. Diagnose at the attribute -- the
    // marker is the thing to delete -- like union_marker / pointer_marker.
    if (!S.Profiles().shouldEmitProfileViolation(
            "std::init", "uninit_with_initializer", UA->getLocation(), F))
      continue;
    S.Diag(UA->getLocation(), diag::err_init_uninit_member_initialized)
        << "std::init" << F->getDeclName() << F->getType();
    // Distinguish the non-vacuity reasons for the note: a non-trivial default
    // constructor runs code (0); a trivial one that leaves no subobject
    // uninitialized has nothing to acknowledge (1); a deleted or absent one
    // makes the marker unsatisfiable (2).
    unsigned Reason = 1;
    if (const auto *MemberRD = BaseTy->getAsCXXRecordDecl();
        MemberRD && MemberRD->hasDefinition()) {
      const CXXRecordDecl *Def = MemberRD->getDefinition();
      bool Deleted = !Def->hasDefaultConstructor();
      for (const CXXConstructorDecl *Ctor : Def->ctors())
        if (Ctor->isDefaultConstructor() && Ctor->isDeleted())
          Deleted = true;
      if (Deleted)
        Reason = 2;
      else if (!Def->hasTrivialDefaultConstructor())
        Reason = 0;
    }
    S.Diag(F->getTypeSpecStartLoc(), diag::note_init_uninit_member_type)
        << BaseTy << Reason;
  }
}

// Class-finalization opt-in table (pattern 3).
constexpr FinalizationProfile<CXXRecordDecl> ClassFinalizationProfiles[] = {
    {"test::class_final", &runTestClassFinalCallback},
    {"std::init", &runStdInitUninitFieldMarkerCallback},
};

// Constructor-finalization opt-in table (pattern 4).
constexpr FinalizationProfile<CXXConstructorDecl>
    ConstructorFinalizationProfiles[] = {
        {"test::ctor_final", &runTestCtorFinalCallback},
        {"std::init", &runStdInitCtorUninitMemberCallback},
};

// Run the enforced finalization-profile callbacks in Table for D. Merges the
// former per-node dispatchers; the per-node filter (dependent, lambda,
// delegating, ...) stays at each call site. Each callback passes D to the
// Decl-aware SemaProfiles::shouldEmitProfileViolation, which honors
// [[profiles::suppress]]
// on D or a lexical parent, so the dispatcher needs no suppress scope of its
// own. The table is taken by reference-to-array, not ArrayRef: deducing Node
// from a C array against an ArrayRef<FinalizationProfile<Node>> parameter is
// not
// possible (no array-to-ArrayRef conversion happens during template argument
// deduction).
template <class Node, std::size_t N>
void dispatchFinalizationProfiles(Sema &S, Node *D,
                                  const FinalizationProfile<Node> (&Table)[N]) {
  if (!S.Profiles().anyProfileEnforced(Table))
    return;
  // Finalization can run nested in an unrelated instantiation whose
  // [[profiles::suppress]] scope is still on the parse-time stack. No guard
  // is needed: the stack consult is dominion-checked against the entry's
  // recorded construct range, so such an entry matches only if its
  // construct's tokens cover the finalized declaration -- including when the
  // finalized pattern is first declared after the suppressed construct.
  for (const auto &E : Table)
    if (S.Profiles().isProfileEnforced(E.Name))
      E.Callback(S, D);
}
} // namespace

void SemaProfiles::checkProfileViolationsAtClassFinalization(
    CXXRecordDecl *RD) {
  if (!getLangOpts().Profiles || !RD)
    return;
  if (RD->isInvalidDecl() || RD->isDependentType() || RD->isLambda())
    return;
  dispatchFinalizationProfiles(SemaRef, RD, ClassFinalizationProfiles);
}

void SemaProfiles::checkProfileViolationsAtConstructorFinalization(
    CXXConstructorDecl *Ctor) {
  if (!getLangOpts().Profiles || !Ctor)
    return;
  // A dependent constructor pattern re-fires on instantiation; a delegating
  // constructor leaves member initialization to its target.
  if (Ctor->isInvalidDecl() || Ctor->isDependentContext() ||
      Ctor->isDelegatingConstructor())
    return;
  dispatchFinalizationProfiles(SemaRef, Ctor, ConstructorFinalizationProfiles);
}

