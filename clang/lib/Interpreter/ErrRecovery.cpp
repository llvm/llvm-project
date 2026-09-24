//===--------- ErrorRecovery.cpp - Declaration State Recovery -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//
//
// This file implements declaration state recovery for failed
// partial translation units (PTUs), restoring declaration state to
// the state of the previous PTU.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/Interpreter/ErrorRecovery.h"
#include "clang/Sema/Sema.h"

namespace clang {

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::Helpers ///////////////////////
/// -----------------------------------------------------------------------

DefinitionDataFootprint *
DeclStateReverter::createDefinitionDataFootprint(const ASTContext &Ctx,
                                                 const CXXRecordDecl &RD) {
  auto *FP =
      new (Ctx, alignof(DefinitionDataFootprint)) DefinitionDataFootprint();
  const auto &Live = RD.data();
#define FIELD(Name, Width, Merge) FP->Name = Live.Name;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
  return FP;
}

bool DeclStateReverter::compareDefinitionDataFootprint(
    const DefinitionDataFootprint &FP, const CXXRecordDecl &RD) {
  const auto &Live = RD.data();
#define FIELD(Name, Width, Merge)                                              \
  if (FP.Name != Live.Name)                                                    \
    return false;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
  return true;
}

void DeclStateReverter::restoreDefinitionDataFootprint(
    const DefinitionDataFootprint &FP, CXXRecordDecl &RD) {
  auto &Live = RD.data();
#define FIELD(Name, Width, Merge) Live.Name = FP.Name;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
}

// NEEDS: friend class DeclStateReverter; in CXXRecordDecl (DefinitionData
// is PRIVATE there -- already granted) and nothing further for the
// DeclContext/TagDecl parts below, which are routed through the
// protected accessors above instead.
void DeclStateReverter::restoreDefinitionAndRevertDC(CXXRecordDecl &RD) {
  // 1. DefinitionData pointer -- back to null.
  RD.DefinitionData = nullptr;

  // 2. DeclContext member list -- back to empty. RD's own FirstDecl/
  // LastDecl were populated by InstantiateClassImpl substituting RD's
  // members for the first time (SemaTemplateInstantiate.cpp:3658-3729,
  // and for the static_assert/friend-decl case specifically,
  // Sema::BuildStaticAssertDeclaration's CurContext->addDecl(), which
  // never notifies since neither is a NamedDecl) -- an incomplete,
  // never-defined class's DeclContext has neither, so this is RD's
  // true prior state, not a guess.
  clearDeclContextChain(RD);

  // 3. TagDecl completion bits -- back to "never started". Flipped by
  // startDefinition()/completeDefinition().
  clearBeingDefined(RD);
  RD.setCompleteDefinition(false);
}

void DeclStateReverter::revertDefinitionArrival(Decl &D) {
  if (auto *FD = dyn_cast<FunctionDecl>(&D))
    FD->setBody(nullptr);
  else if (auto *VD = dyn_cast<VarDecl>(&D))
    VD->setInit(nullptr);
  else if (auto *Field = dyn_cast<FieldDecl>(&D))
    Field->setInClassInitializer(nullptr);
}

void DeclStateReverter::removeSpecializations(
    PTUCheckpointLedger &Ledger, const RedeclarableTemplateDecl *TD, PTUID ID) {
  if (const auto *CTD = dyn_cast<ClassTemplateDecl>(TD)) {
    auto &Specs =
        static_cast<const ClassTemplateSpecAccess &>(*CTD).getSpecializations();
    while (!Specs.empty()) {
      auto It = Specs.end();
      --It;
      if (!Ledger.isFromThisPTU(&*It, ID))
        break;
      Specs.pop_back();
    }
  } else if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(TD)) {
    auto &Specs = static_cast<const FunctionTemplateSpecAccess &>(*FTD)
                      .getSpecializations();
    while (!Specs.empty()) {
      auto It = Specs.end();
      --It;
      if (!Ledger.isFromThisPTU(It->getFunction(), ID))
        break;
      Specs.pop_back();
    }
  } else if (const auto *VTD = dyn_cast<VarTemplateDecl>(TD)) {
    auto &Specs =
        static_cast<const VarTemplateSpecAccess &>(*VTD).getSpecializations();
    while (!Specs.empty()) {
      auto It = Specs.end();
      --It;
      if (!Ledger.isFromThisPTU(&*It, ID))
        break;
      Specs.pop_back();
    }
  }
}

void DeclStateReverter::detachDefData(const Decl *D) {
  // TODO: handle Template's own separate redecl chain the same way.
  const auto *RD = dyn_cast<CXXRecordDecl>(D);
  if (!RD)
    return;
  for (const auto *I : RD->redecls())
    const_cast<CXXRecordDecl *>(cast<CXXRecordDecl>(I))->DefinitionData =
        nullptr;
}

void DeclStateReverter::detachCommonBase(const RedeclarableTemplateDecl *RT) {
  // Common is shared across the WHOLE redecl chain once path-compression
  // (RedeclarableTemplateDecl::getCommonPtr()'s backward walk-then-
  // backfill) has run -- resetting only the one decl this PTU created
  // would leave some other redecl holding a stale, dangling copy of the
  // same pointer. Walk every redecl, not just a backward chain from RT.
  for (const auto *R : RT->redecls())
    clearCommonPtr(*R);
}

/// Remove D from every lookup map it became visible in, reinstating the
/// previous declaration where D had replaced one in-place (which is what
/// StoredDeclsList::HandleRedeclaration does on a redeclaration --
/// erasing the slot outright would lose the older decl entirely; that is
/// the ReopenNs failure).
void DeclStateReverter::detachFromDCLookup(Decl *D, PTUID ID) {
  NamedDecl *ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return; // nothing nameable -- never entered a lookup map

  // Chain must still be intact to find the replacement -- that is why
  // this runs before detachFromRedeclChain.
  NamedDecl *Survivor = findSurvivor(ND, ID);
  DeclarationName Name = ND->getDeclName();

  // A decl in a transparent context (an unnamed namespace, an unscoped
  // enum, a linkage-spec block) is visible in the enclosing context's map
  // as well, so every level it was inserted into needs the same repair --
  // not just its own primary context.
  DeclContext *DC = ND->getDeclContext();
  do {
    DeclContext *Primary = DC->getPrimaryContext();
    if (StoredDeclsMap *Map = Primary->getLookupPtr()) {
      auto Pos = Map->find(Name);
      // Not an error: the map is built lazily, so a context that was
      // never looked up in has no entry for this name at all.
      if (Pos != Map->end()) {
        StoredDeclsList &List = Pos->second;
        List.remove(ND);
        if (Survivor)
          List.addOrReplaceDecl(Survivor);
        if (List.isNull())
          Map->erase(Pos);
      }
    }
  } while (DC->isTransparentContext() && (DC = DC->getParent()));

  //   DeclContext *Lexical = const_cast<DeclContext
  //   *>(ND->getLexicalDeclContext()); if
  //   (RepairedLexicalContexts.insert(Lexical).second)
  //     repairLexicalChain(*Lexical);
}

// Walk D's redecl chain looking for the newest decl that predates this
// PTU. Returns nullptr if the entire chain was created this PTU.
template <typename DeclT>
DeclT *DeclStateReverter::findSurvivor(DeclT *D, PTUID ID) const {
  for (Decl *It = D->getMostRecentDecl(); It; It = It->getPreviousDecl()) {
    if (!PTUSlabCheckpoints.isFromThisPTU(D, ID))
      return dyn_cast<DeclT>(It);
  }
  return nullptr;
}

/// Point the canonical decl's "most recent" link back at the newest
/// redeclaration that predates this PTU.
void DeclStateReverter::detachFromRedeclChain(const Decl *D, PTUID ID) {
  (void)tryDetachRedeclChain(const_cast<Decl *>(D), ID);
}

void DeclStateReverter::repairLexicalChain(DeclContext &DC, PTUID ID) {
  auto &Access = static_cast<DeclContextLinkAccess &>(DC);
  Decl *Prev = nullptr;
  for (Decl *Cur = Access.FirstDecl; Cur; Cur = Cur->getNextDeclInContext()) {
    if (PTUSlabCheckpoints.isFromThisPTU(Cur, ID)) {
      if (Prev) {
        Access.LastDecl = Prev;
        clearNextInContext(*Prev);
      } else {
        Access.FirstDecl = Access.LastDecl = nullptr; // nothing survives
      }
      return;
    }
    Prev = Cur;
  }
}

NamedDecl *DeclStateReverter::tryDetachRedeclChain(Decl *D, PTUID ID) {
  if (hasNoPreviousDecl(D))
    return nullptr; // nothing to do

  NamedDecl *Survivor = nullptr;

  auto unlinkRedeclChain = [&](auto *RD) {
    auto *SurvivorDecl = findSurvivor(RD, ID);
    if (SurvivorDecl)
      patchRedeclLink(RD, SurvivorDecl);

    Survivor = SurvivorDecl;
  };

  switch (D->getKind()) {
  case Decl::Function:
    unlinkRedeclChain(cast<FunctionDecl>(D));
    break;

  case Decl::Var:
    unlinkRedeclChain(cast<VarDecl>(D));
    break;

  case Decl::Enum:
  case Decl::Record:
  case Decl::CXXRecord:
    unlinkRedeclChain(cast<TagDecl>(D));
    break;

  case Decl::ClassTemplate: {
    // NamedDecl *ND = cast<RedeclarableTemplateDecl>(D)->getTemplatedDecl();
    unlinkRedeclChain(cast<RedeclarableTemplateDecl>(D));
    // tryDetachRedeclChain(ND);
  } break;

  case Decl::FunctionTemplate:
    unlinkRedeclChain(cast<RedeclarableTemplateDecl>(D));
    break;

  case Decl::TypeAliasTemplate:
    unlinkRedeclChain(cast<RedeclarableTemplateDecl>(D));
    break;

  case Decl::VarTemplate:
    unlinkRedeclChain(cast<RedeclarableTemplateDecl>(D));
    break;

  case Decl::Namespace:
    unlinkRedeclChain(cast<NamespaceDecl>(D));
    break;

  default:
    break;
  }

  return Survivor;
}

static DeclShape classifyShape(const Decl *D) {
  // Base shape -- most-derived first.
  if (isa<RedeclarableTemplateDecl>(D))
    return DeclShape::Template;
  else if (isa<EnumDecl>(D))
    return DeclShape::Enum;
  else if (isa<FunctionDecl>(D))
    return DeclShape::Function;
  else if (isa<VarDecl>(D))
    return DeclShape::Var;
  else if (isa<CXXRecordDecl>(D))
    return DeclShape::Class;
  else if (isa<TypedefDecl, TypeAliasDecl>(D))
    return DeclShape::Typedef;
  else
    return DeclShape::None;
}

/// Called once when a decl is first seen. This only sets up the possible
/// MutationRecord kinds; it does not detect the mutation itself.
///
/// Add a kind here only if we need to do something before the mutation:
/// - save the old value because it will be overwritten before the listener
///   runs, or
/// - start hidden tracking because there is no listener for that change.
///
/// If the listener itself is enough to detect the change, don't add it here.
/// noteMutated() will create the record when the change actually happens.
///
/// Explicit specializations are terminal, so their SpecInfo/MemberSpecInfo
/// does not need to be tracked.
uint32_t PTUMutationActions::kindsNeedingTracking(DeclShape S, const Decl *D) {
  switch (S) {
  case DeclShape::Class: {
    const auto *RD = cast<CXXRecordDecl>(D);
    uint32_t Kinds = 0;
    // TypeForDecl is written twice for TagDecls. Keep tracking until both
    // writes are done.
    // if (const Type *T = DeclStateReverter::getRawTypeForDecl(RD);
    //     !T || T->isCanonicalUnqualified())
    //   Kinds |= uint32_t(MutationType::TypeForDecl);
    if (RD->hasDefinition() && RD == RD->getDefinition()) {
      // Do we really need to create a DefData chain here? Usually, an implicit
      // decl can be mutated later after the definition. If it is already wired
      // correctly, there is nothing to mutate or track here.
      //
      // Save the current DefinitionData because it may change later.
      if (RD->needsImplicitDefaultConstructor() ||
          RD->needsImplicitCopyConstructor() ||
          RD->needsImplicitCopyAssignment() ||
          RD->needsImplicitMoveConstructor() ||
          RD->needsImplicitMoveAssignment() || RD->needsImplicitDestructor())
        Kinds |= uint32_t(MutationType::DefinitionData);
    }
    if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
      if (Spec->getSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationType::SpecInfo);
    } else if (const auto *MSI = RD->getMemberSpecializationInfo()) {
      // Same terminal state, same enum, reused for MSI's own Kind/POI.
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationType::MemberSpecInfo);
    }
    return Kinds;
  }
  case DeclShape::Function: {
    const auto *FD = cast<FunctionDecl>(D);
    uint32_t Kinds = 0;
    const auto *FPT = FD->getType()->getAs<FunctionProtoType>();
    // Exception-spec resolution changes the type before the listener runs,
    // so we need the old type saved beforehand.
    if (FPT && (FPT->getExceptionSpecType() == EST_Unevaluated ||
                FPT->getExceptionSpecType() == EST_Uninstantiated ||
                FPT->getExceptionSpecType() == EST_DependentNoexcept ||
                FPT->getExceptionSpecType() == EST_Unparsed))
      Kinds |= uint32_t(MutationType::ExceptionSpec);
    if (FD->getReturnType()->isUndeducedType())
      // Deduced return type works the same way: the type is changed before
      // the listener runs, so save it beforehand.
      Kinds |= uint32_t(MutationType::DeducedReturnType);
    if (FD->getTemplateSpecializationInfo() &&
        FD->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
      // A specialization’s Kind/point-of-instantiation is overwritten in place,
      // with no separate record of the old value. The footprint chain is the
      // only place where the previous value is preserved.
      Kinds |= uint32_t(MutationType::SpecInfo);
    if (const auto *MSI = FD->getMemberSpecializationInfo()) {
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationType::MemberSpecInfo);
    }

    return Kinds;
  }
  case DeclShape::Var: {
    const auto *VD = cast<VarDecl>(D);
    uint32_t Kinds = 0;
    if (const auto *Spec = dyn_cast<VarTemplateSpecializationDecl>(VD)) {
      // Same terminal-state exclusion as the Class/Function cases above.
      if (Spec->getSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationType::SpecInfo);
    } else if (const auto *MSI = VD->getMemberSpecializationInfo()) {
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationType::MemberSpecInfo);
    }
    // EvaluatedStmt::WasEvaluated is set the first time this variable is
    // constant-evaluated, which can happen in any PTU long after the VarDecl
    // was created. There is no ASTMutationListener event for this
    // (VarDecl::evaluateValueImpl doesn’t notify anything), so this is a
    // hidden kind that we have to check live here, just like
    // Shape::Template’s CommonCreated.
    //
    // getEvaluatedStmt() is a pure read – unlike evaluateValue(), it doesn’t
    // trigger evaluation itself, so it’s safe to check on every classify call.
    if (const EvaluatedStmt *Eval = VD->getEvaluatedStmt();
        !Eval || !Eval->WasEvaluated)
      Kinds |= uint32_t(MutationType::EvaluatedValue);

    return Kinds;
  }
  case DeclShape::Enum: {
    const auto *ED = cast<EnumDecl>(D);
    uint32_t Kinds = 0;
    // TypeForDecl: EnumDecl is also a TagDecl, so the same two-write pattern
    // and “still open” condition as the Class case above apply here – see
    // its comment for the details.
    //
    // if (const Type *T = DeclStateReverter::getRawTypeForDecl(ED);
    //     !T || T->isCanonicalUnqualified())
    //   Kinds |= uint32_t(MutationType::TypeForDecl);
    //
    // MSI-backed part: for an ordinary enum, its own completion is the
    // redeclaration-creates-a-new-object case. EnumDecl is Redeclarable just
    // like CXXRecordDecl, so the same reasoning as Function/Var applies here.
    if (const auto *MSI = ED->getMemberSpecializationInfo()) {
      // Same terminal-state exclusion as the other MSI-backed cases above.
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationType::MemberSpecInfo);
    }
    return Kinds;
  }
  case DeclShape::Template: {
    uint32_t Kinds = 0;
    // TODO: track only canon decl; and also add check
    // HiddenMutationTracker.isTrackedFor(RT->getCanonicalDecl()) is not being
    // already tracked
    const RedeclarableTemplateDecl *RT =
        cast<RedeclarableTemplateDecl>(D->getCanonicalDecl());
    if (!DeclStateReverter::isCommonPtrValid(*RT) &&
        !HiddenMutationTracker.isTrackedFor(
            RT, uint32_t(MutationType::TemplateCommon))) {
      // No listener reports these Common changes, so use hidden tracking.
      Kinds |= uint32_t(MutationType::TemplateCommon);
    } else if (const auto *CTD = dyn_cast<ClassTemplateDecl>(RT)) {
      // Common exists, so calling getCommonPtr() here is safe (won't
      // allocate) -- only ClassTemplateDecl has CanonInjectedTST in its
      // own Common, not the Function/Var template siblings.
      if (!DeclStateReverter::isTemplateCanonInjectedTSTValid(CTD))
        Kinds |= uint32_t(MutationType::CanonInjectedTST);
    }
    return Kinds;
  }
  case DeclShape::Typedef: {
    // TypedefDecl/TypeAliasDecl only -- ASTContext::getTypedefType() has its
    // own guard (if (Decl->TypeForDecl) // return), so this is effectively
    // write-once: once non-null, TypeForDecl
    // never changes again for this decl.
    // const auto *TD = cast<TypedefNameDecl>(D);
    // if (!DeclStateReverter::getRawTypeForDecl(TD))
    //   return uint32_t(MutationType::TypeForDecl);
    return 0;
  }
  case DeclShape::None:
    return 0;
  }
  return 0;
}

/// Given a Decl known to be DeclShape S with FlaggedKinds set, return the
/// subset of kinds that are actually confirmed.
///
/// Everything reaching this function through verifyMutations() belongs to
/// a decl from an earlier PTU. Same-PTU mutations are ignored by the
/// reactive listeners, so these mutations were already found during an
/// earlier commit().
///
/// Some mutation kinds are fully confirmed by their listener, so there is
/// no need to walk the chain for them:
/// - DefinitionInstantiate, SpecializationAdded, DefinitionData
/// - ExceptionSpec, DeducedReturnType
///
/// For these, the listener itself is the source of truth for the change.
/// In particular, don’t use the FunctionTypeMutations chain for
/// ExceptionSpec/DeducedReturnType here. That chain is populated later by
/// commitFunctionFamily(), so checking it here would miss the first
/// resolution.
///
/// The remaining kinds don’t have a reliable listener, so they need to be
/// checked against the stored state:
/// - SpecInfo, MemberSpecInfo
/// - Template Common/CanonInjectedTST
/// - Var EvaluatedValue
/// - Typedef TypeForDecl
///
/// For footprint-based checks, the baseline should already be there since
/// it is seeded when the decl is first discovered. If it’s missing here,
/// assert in debug builds, but still confirm the mutation. commitDecl()
/// will create the missing chain while committing, which fixes up the
/// baseline for future checks.
uint32_t PTUMutationActions::verifyMutationFor(const Decl *D, DeclShape S,
                                               uint32_t FlaggedKinds) {

  uint32_t Verified = 0;
  Verified |= FlaggedKinds & uint32_t(MutationType::DefinitionInstantiate);
  switch (S) {
  case DeclShape::Class: {
    const auto *RD = cast<CXXRecordDecl>(D);
    // AddedCXXImplicitMember firing is itself the confirmation -- see
    // this function's own top comment for why no chain lookup belongs
    // here.
    Verified |= FlaggedKinds & uint32_t(MutationType::DefinitionData);
    if (FlaggedKinds & uint32_t(MutationType::TypeForDecl)) {
      //   const Type *Now = DeclStateReverter::getRawTypeForDecl(RD);
      //   if (Tracker.getTagDeclTypeInfo().mostRecent(RD).value_or(nullptr) !=
      //   Now)
      //     Verified |= uint32_t(MutationType::TypeForDecl);
    }
    if (FlaggedKinds & uint32_t(MutationType::SpecInfo)) {
      if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
        auto *Chain = Tracker.getChainFor(Spec);
        assert(Chain && "SpecInfo baseline missing at verify time");
        const SpecializationFootprint *Last =
            Chain ? Chain->mostRecent() : nullptr;
        if (!Last ||
            !DeclStateReverter::compareSpecializationFootprint(*Last, *Spec))
          Verified |= uint32_t(MutationType::SpecInfo);
      }
    } else if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      auto *Chain = Tracker.getMemberSpecChainFor(RD);
      assert(Chain && "MemberSpecInfo baseline missing at verify time");
      const MemberSpecializationFootprint *Last =
          Chain ? Chain->mostRecent() : nullptr;
      if (!Last ||
          !DeclStateReverter::compareMemberSpecializationFootprint(*Last, *RD))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Function: {
    const auto *FD = cast<FunctionDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::SpecInfo)) {
      auto *Chain = Tracker.getChainFor(FD);
      assert(Chain && "SpecInfo baseline missing at verify time");
      const FunctionSpecializationFootprint *Last =
          Chain ? Chain->mostRecent() : nullptr;
      if (!Last || !DeclStateReverter::compareFunctionSpecializationFootprint(
                       *Last, *FD))
        Verified |= uint32_t(MutationType::SpecInfo);
    } else if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      auto *Chain = Tracker.getMemberSpecChainFor(FD);
      assert(Chain && "MemberSpecInfo baseline missing at verify time");
      const MemberSpecializationFootprint *Last =
          Chain ? Chain->mostRecent() : nullptr;
      if (!Last ||
          !DeclStateReverter::compareMemberSpecializationFootprint(*Last, *FD))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    // ResolvedExceptionSpec/DeducedReturnType firing is itself the
    // confirmation.
    Verified |= FlaggedKinds & (uint32_t(MutationType::ExceptionSpec) |
                                uint32_t(MutationType::DeducedReturnType));
    break;
  }
  case DeclShape::Var: {
    const auto *VD = cast<VarDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::SpecInfo)) {
      if (const auto *Spec = dyn_cast<VarTemplateSpecializationDecl>(VD)) {
        auto *Chain = Tracker.getChainFor(Spec);
        assert(Chain && "SpecInfo baseline missing at verify time");
        const VarSpecializationFootprint *Last =
            Chain ? Chain->mostRecent() : nullptr;
        if (!Last ||
            !DeclStateReverter::compareVarSpecializationFootprint(*Last, *Spec))
          Verified |= uint32_t(MutationType::SpecInfo);
      }
    } else if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      auto *Chain = Tracker.getMemberSpecChainFor(VD);
      assert(Chain && "MemberSpecInfo baseline missing at verify time");
      const MemberSpecializationFootprint *Last =
          Chain ? Chain->mostRecent() : nullptr;
      if (!Last ||
          !DeclStateReverter::compareMemberSpecializationFootprint(*Last, *VD))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    if (FlaggedKinds & uint32_t(MutationType::EvaluatedValue) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            VD, uint32_t(MutationType::EvaluatedValue))) {
      // No chain to compare against -- write-once field, so presence of
      // WasEvaluated=true right now IS the confirmation, same reasoning
      // as Shape::Template's CommonCreated/CanonInjectedTSTCached above.
      const EvaluatedStmt *Eval = VD->getEvaluatedStmt();
      if (Eval && Eval->WasEvaluated)
        Verified |= uint32_t(MutationType::EvaluatedValue);
    }
    break;
  }
  case DeclShape::Enum: {
    const auto *ED = cast<EnumDecl>(D);
    // if (FlaggedKinds & uint32_t(MutationType::TypeForDecl)) {
    //   const Type *Now = DeclStateReverter::getRawTypeForDecl(ED);
    //   if (Tracker.getTagDeclTypeInfo().mostRecent(ED).value_or(nullptr) !=
    //   Now)
    //     Verified |= uint32_t(MutationType::TypeForDecl);
    // }
    if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      auto *Chain = Tracker.getMemberSpecChainFor(ED);
      assert(Chain && "MemberSpecInfo baseline missing at verify time");
      const MemberSpecializationFootprint *Last =
          Chain ? Chain->mostRecent() : nullptr;
      if (!Last ||
          !DeclStateReverter::compareMemberSpecializationFootprint(*Last, *ED))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Template: {
    // SpecializationAdded is known directly from
    // AddedCXXTemplateSpecialization firing, so
    // always confirmed when flagged.
    Verified |= FlaggedKinds & uint32_t(MutationType::SpecializationAdded);
    const auto *RT = cast<RedeclarableTemplateDecl>(D)->getCanonicalDecl();
    if (FlaggedKinds & uint32_t(MutationType::TemplateCommon) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            RT, uint32_t(MutationType::TemplateCommon))) {
      // No chain to compare against -- write-once, so Common's own
      // existence right now IS the confirmation.
      if (DeclStateReverter::isCommonPtrValid(*RT))
        Verified |= uint32_t(MutationType::TemplateCommon);
    }
    if (FlaggedKinds & uint32_t(MutationType::CanonInjectedTST) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            RT, uint32_t(MutationType::CanonInjectedTST))) {
      if (const auto *CTD = dyn_cast<ClassTemplateDecl>(RT))
        if (DeclStateReverter::isTemplateCanonInjectedTSTValid(CTD))
          Verified |= uint32_t(MutationType::CanonInjectedTST);
    }
  } break;
  case DeclShape::Typedef: {
    const auto *TD = cast<TypedefNameDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::TypeForDecl) &&
        // Defensive, on top of SweepTracker::sweep()'s own bookkeeping --
        // same reasoning as the Var/Template cases above.
        Tracker.getHiddenMutationTracker().isTrackedFor(
            TD, uint32_t(MutationType::TypeForDecl))) {
      // No chain to compare against -- write-once field (unlike
      // TagDecl's TypeForDecl, this one is guarded by ASTContext's own
      // `if (Decl->TypeForDecl) return`, never overwritten a second
      // time), so non-null right now IS the confirmation.
      //   if (DeclStateReverter::getRawTypeForDecl(TD))
      //     Verified |= uint32_t(MutationType::TypeForDecl);
    }
  } break;
  case DeclShape::None:
    break; // never flagged in the first place.
  }
  return Verified;
}

template <typename OnConfirmedFn>
void SweepTracker::sweep(PTUMutationActions &Act, OnConfirmedFn OnConfirmed) {
  llvm::SmallVector<std::pair<const Decl *, uint32_t>, 8> Confirmed;
  for (auto &Entry : Active) {
    const Decl *D = Entry.getFirst();
    uint32_t StillOpen = Entry.getSecond();
    DeclShape S = classifyShape(D);
    if (uint32_t Newly = Act.verifyMutationFor(D, S, StillOpen)) {
      OnConfirmed(D, S, Newly);
      Confirmed.emplace_back(D, Newly);
    }
  }
  for (auto &[D, Newly] : Confirmed)
    settle(D, Newly);
}

/// Verify recorded mutations and remove any spurious ones. If none of a
/// decl’s mutation kinds are confirmed, remove the record entirely so
/// restore() does not pop chain entries that were never written.
void PTUStateInfo::verifyMutations(PTUMutationActions &Actions) {
  Mutations.remove_if([&](auto &Entry) -> bool {
    uint32_t Confirmed = Actions.verifyMutationFor(Entry.first, Entry.second.S,
                                                   Entry.second.MutationType);
    Entry.second.MutationType = Confirmed;
    return Confirmed == 0;
  });
}

class DeclStateCommitPolicy {
  PTUID ID;
  PTUMutationActions &Action;

public:
  DeclStateCommitPolicy(PTUID ID, PTUMutationActions &Action)
      : ID(ID), Action(Action) {}
  /// Everything shouldAct resolved, carried to the caller instead of
  /// stashed. Empty means "skip this decl" -- there's no separate bool.
  struct CommitAction {
    DeclShape S;
    uint32_t Kinds;
    explicit operator bool() const { return Kinds != 0; }
  };

  bool shouldRecurse(const Decl *D) const {
    if (isa<NamespaceDecl>(D))
      return true;
    if (const auto *RD = dyn_cast<CXXRecordDecl>(D))
      return RD->isCompleteDefinition();
    return false;
  }

  CommitAction actionFor(const Decl *D) const {
    if (!D->isDefinedOutsideFunctionOrMethod())
      return {};

    DeclShape S = classifyShape(D);
    if (S == DeclShape::None)
      return {};

    // we have to handle every case that if a decl need to create baseline
    // info or if any decl has done all mutations already in this for exmpla
    // membe spec already properly done like this kind of case so we don't
    // create spec info blindly
    uint32_t Kinds = Action.kindsNeedingTracking(S, D);
    if (!Kinds)
      return {}; // structurally immutable -- no baseline needed

    return CommitAction{S, Kinds};
  }

  void runAction(const Decl *D, const CommitAction &A) {
    MutationRecord Rec = MutationRecord{A.S, A.Kinds};
    Action.commitDecl(ID, D, Rec, /*IsNew=*/true);
  }
};

class DeclStateUnlinkPolicy {
  PTUID ID;
  IncrementalStateTracker &Tracker;
  DeclStateReverter &Reverter;
  llvm::SmallPtrSet<const Decl *, 16> RepairedChains;
  llvm::SmallPtrSet<const DeclContext *, 8> RepairedLexicalContexts;
  llvm::SmallPtrSet<const DeclContext *, 8> TouchedDC;

public:
  DeclStateUnlinkPolicy(PTUID ID, IncrementalStateTracker &Tracker,
                        DeclStateReverter &Reverter)
      : ID(ID), Tracker(Tracker), Reverter(Reverter) {}

  struct RestoreAction {
    bool DetachFromLexicalChain = false;
    bool DetachFromDC = false;
    bool DetachFromRedecl = false;
    bool DetachDefData = false;
    // bool DetachCommonBase = false;
    bool HasTemplated = false;
    bool TemplatedDetachDefData = false;
    explicit operator bool() const {
      return DetachFromLexicalChain || DetachFromDC || DetachFromRedecl ||
             DetachDefData || /*DetachCommonBase ||*/ HasTemplated ||
             TemplatedDetachDefData;
    }
  };

  bool shouldRecurse(const Decl *D) const {
    // we don't care about the decl which complete chain is part belong to
    // current PTU.
    if (Tracker.isFromThisPTU(D->getCanonicalDecl(), ID))
      return false;
    return isa<NamespaceDecl>(D);
  }

  RestoreAction actionFor(const Decl *D) const {
    RestoreAction A;
    const Decl *Canon = D->getCanonicalDecl();

    /// Keyed on the canonical decl: every redeclaration shares one
    /// RedeclLink, so truncating twice would step past the intended survivor.
    if (!RepairedChains.count(Canon) && D->getPreviousDecl() &&
        !Tracker.isFromThisPTU(Canon, ID))
      A.DetachFromRedecl = true;

    // Keyed on the primary context, a different granularity: a reopened
    // namespace's redeclarations all resolve to the same primary.
    const DeclContext *DC = D->getDeclContext()->getPrimaryContext();
    if (TouchedDC.contains(DC))
      A.DetachFromDC = true;

    if (D->isImplicit() &&
        !RepairedLexicalContexts.count(D->getLexicalDeclContext()))
      A.DetachFromLexicalChain = true;

    if (const auto *RT = dyn_cast<RedeclarableTemplateDecl>(D)) {
      // if (Tracker.isFromThisPTU(::needToDetachCommonPtr(CheckPoint, RT),
      // ID))
      //   A.DetachCommonBase = true;

      const NamedDecl *T = RT->getTemplatedDecl();
      A.HasTemplated = true;
      if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(T)) {
        if (const CXXRecordDecl *Def = RD->getDefinition();
            Def && Tracker.isFromThisPTU(Def, ID))
          A.TemplatedDetachDefData = true;
      }
    } else if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
      if (const CXXRecordDecl *Def = RD->getDefinition();
          Def && Tracker.isFromThisPTU(Def, ID))
        A.DetachDefData = true;
    }

    return A;
  }

  void runAction(const Decl *D, const RestoreAction &A) {
    if (A.DetachDefData)
      Reverter.detachDefData(D);
    // if (A.DetachCommonBase)
    //   Detacher.detachCommonBase(D);
    if (A.DetachFromDC)
      Reverter.detachFromDCLookup(const_cast<Decl *>(D), ID);
    if (A.DetachFromRedecl) {
      RepairedChains.insert(D->getCanonicalDecl());
      Reverter.detachFromRedeclChain(D, ID);
    }

    if (A.HasTemplated) {
      const Decl *T = cast<RedeclarableTemplateDecl>(D)->getTemplatedDecl();
      if (A.TemplatedDetachDefData)
        Reverter.detachDefData(T);
      if (A.DetachFromRedecl)
        Reverter.detachFromRedeclChain(T, ID);
    }

    if (A.DetachFromLexicalChain) {
      Reverter.repairLexicalChain(
          *const_cast<DeclContext *>(D->getLexicalDeclContext()), ID);
      RepairedLexicalContexts.insert(D->getLexicalDeclContext());
    }
  }
};

template <typename DeclStateProxyT>
void PTUMutationActions::walkDecls(const DeclContext *DC,
                                   DeclStateProxyT &Proxy) {
  for (const Decl *D : DC->decls()) {
    if (Proxy.shouldRecurse(D))
      walkDecls(cast<DeclContext>(D), Proxy);
    if (auto Act = Proxy.actionFor(D))
      Proxy.runAction(D, Act);
  }
}

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::restore //////////////////////
/// -----------------------------------------------------------------------

void PTUMutationActions::restoreClass(PTUID ID, const CXXRecordDecl *RD,
                                      MutationRecord &Rec) {

  // DefinitionArrived and DefinitionDataChanged commit identically -- both
  // mean "this record's DefinitionData needs a fresh footprint." They stay
  // distinct kinds because rollback and dependency-edge attribution treat
  // them differently (arrival is this PTU completing a record that
  // predates it; a data change may be either).
  if (Rec.has(MutationType::DefinitionInstantiate) ||
      Rec.has(MutationType::DefinitionData)) {
    // CXXRecordDecl *RD = cast<CXXRecordDecl>(RD);
    // New: no prior entry exists, so write unconditionally.
    // Existing: only write a real diff.
    /// SpecializationDecl can have lazy implicit generated body;
    if (Rec.has(MutationType::DefinitionInstantiate)) {
      DeclStateReverter::revertDefinitionArrival(
          *const_cast<CXXRecordDecl *>(RD));
    } else {
      auto *Chain = Tracker.getChainFor(RD);
      if (Chain) {
        if (const DefinitionDataFootprint *Restored = Chain->mostRecent())
          DeclStateReverter::restoreDefinitionDataFootprint(
              *Restored, *const_cast<CXXRecordDecl *>(RD));
      }
    }
  }

  // TypeForDecl is shared across the whole redeclaration chain -- revert all.
  if (Rec.has(MutationType::TypeForDecl)) {
    // CXXRecordDecl *RD = cast<CXXRecordDecl>(D);
    for (const TagDecl *Redecl : RD->redecls()) {
      const Type *LastKnown =
          Tracker.getTagDeclTypeInfo().mostRecent(Redecl).value_or(nullptr);
      const TypeDecl *TD = cast<TypeDecl>(Redecl);
      if (LastKnown != TD->getTypeForDecl())
        const_cast<TypeDecl *>(TD)->setTypeForDecl(LastKnown);
    }
  }

  if (Rec.has(MutationType::SpecInfo)) {
    // Only a specialization can carry this kind; a plain CXXRecordDecl
    // reaching here means a note*() wrapper passed the wrong kind.
    const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD);
    assert(Spec && "SpecializationAdded noted on a non-specialization");
    if (auto *Chain = Tracker.getChainFor(Spec)) {
      if (const SpecializationFootprint *Restored = Chain->mostRecent())
        DeclStateReverter::restoreSpecializationFootprint(
            *Restored, *const_cast<ClassTemplateSpecializationDecl *>(Spec));
    }
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    // A nested member of a class template – its MSI kind or
    // point-of-instantiation changed.
    assert(RD->getMemberSpecializationInfo());
    if (auto *Chain = Tracker.getMemberSpecChainFor(RD)) {
      if (const MemberSpecializationFootprint *Restored = Chain->mostRecent())
        DeclStateReverter::restoreMemberSpecializationFootprint(
            *Restored, *const_cast<CXXRecordDecl *>(RD));
    }
  }
}

void PTUMutationActions::restoreFunction(PTUID ID, const FunctionDecl *FD,
                                         MutationRecord &Rec) {

  // FunctionType is shared across the whole redeclaration chain.
  // Resolving a deferred noexcept or deducing an auto return updates the
  // type for every redecl, including ones owned by earlier PTUs, so each
  // redecl needs to be recorded rather than just FD’s own.
  if (Rec.has(MutationType::ExceptionSpec) ||
      Rec.has(MutationType::DeducedReturnType)) {
    for (const FunctionDecl *Redecl : FD->redecls()) {
      QualType LastKnown =
          Tracker.getFunctionTypeMutations().mostRecent(Redecl).value_or(
              QualType());
      if (LastKnown != Redecl->getType())
        const_cast<FunctionDecl *>(Redecl)->setType(LastKnown);
    }
  }

  if (Rec.has(MutationType::DefinitionInstantiate))
    DeclStateReverter::revertDefinitionArrival(*const_cast<FunctionDecl *>(FD));

  if (Rec.has(MutationType::SpecInfo)) {
    // A function template specialization is a plain FunctionDecl carrying
    // FunctionTemplateSpecializationInfo.
    assert(FD->getTemplateSpecializationKind() != TSK_Undeclared &&
           "SpecInfoChanged on a function with no specialization info");
    if (auto *Chain = Tracker.getChainFor(FD)) {
      if (const FunctionSpecializationFootprint *Restored = Chain->mostRecent())
        DeclStateReverter::restoreFunctionSpecializationFootprint(
            *Restored, *const_cast<FunctionDecl *>(FD));
    }
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    assert(FD->getMemberSpecializationInfo());
    if (auto *Chain = Tracker.getMemberSpecChainFor(FD)) {
      if (const MemberSpecializationFootprint *Restored = Chain->mostRecent())
        DeclStateReverter::restoreMemberSpecializationFootprint(
            *Restored, *const_cast<FunctionDecl *>(FD));
    }
  }
}

void PTUMutationActions::restoreTemplate(PTUID ID,
                                         const RedeclarableTemplateDecl *TD,
                                         MutationRecord &Rec) {

  if (Rec.has(MutationType::TemplateCommon)) {
    // TODO: call DeclStateReverter to detech all decl's common ptr if canon
    // decl of template ie being tracked;
    // DeclStateReverter::resetTemplateCommonBase(
    //     *const_cast<RedeclarableTemplateDecl *>(TD));
    DeclStateReverter::detachCommonBase(TD);
    // Common’s creation was done by this PTU.
    // Re-register both bits, since a later PTU may trigger either one again,
    // and CanonInjectedTST cannot exist until Common exists again first.
    //
    // TODO: Track only the canonical decl.
    HiddenMutationTracker.track(TD,
                                uint32_t(MutationType::TemplateCommon) |
                                    uint32_t(MutationType::CanonInjectedTST));
  } else if (Rec.has(MutationType::CanonInjectedTST)) {
    if (const auto *CTD = dyn_cast<ClassTemplateDecl>(TD)) {
      DeclStateReverter::resetCanonInjectedTST(
          *const_cast<ClassTemplateDecl *>(CTD));
      HiddenMutationTracker.track(TD, uint32_t(MutationType::CanonInjectedTST));
    }
  }

  if (Rec.has(MutationType::SpecializationAdded))
    DeclStateReverter::removeSpecializations(Tracker.getPTUSlabCheckpoints(),
                                             TD, ID);
}

void PTUMutationActions::restoreTypedef(PTUID ID, const TypedefNameDecl *TD,
                                        MutationRecord &Rec) {
  if (Rec.has(MutationType::TypeForDecl)) {
    // TD survives this rollback (only the cached Type was this PTU's
    // doing) -- reset the live field and re-register: some later PTU
    // could re-trigger ASTContext::getTypedefType's first-call write.
    // DeclStateReverter::resetTypeForDecl(const_cast<TypedefNameDecl *>(TD));
    // HiddenMutationTracker.track(TD, uint32_t(MutationType::TypeForDecl));
  }
}

void PTUMutationActions::restoreVar(PTUID ID, const VarDecl *VD,
                                    MutationRecord &Rec) {

  using VarMutation = MutationType;

  // EvaluatedStmt::WasEvaluated is write-once, so its previous state is always
  // null here. If it made it into Rec, this PTU had confirmed it as evaluated.
  // Reverting it is therefore just a reset, not a snapshot restore – the
  // original initializer expression in Eval->Value stays untouched; we only
  // clear the cached result.
  if (Rec.has(VarMutation::EvaluatedValue)) {
    // VD itself survives this rollback (only its cached value was this
    // PTU's doing) -- reset the live field, and re-register with
    // SweepTracker since kindsNeedingTracking will flag this as open
    // again and some later PTU could re-trigger evaluation.
    if (EvaluatedStmt *Eval = VD->getEvaluatedStmt()) {
      Eval->WasEvaluated = false;
      Eval->Evaluated = APValue();
    }
    HiddenMutationTracker.track(VD, uint32_t(VarMutation::EvaluatedValue));
  }

  if (Rec.has(VarMutation::DefinitionInstantiate)) {
    /// TODO;
    DeclStateReverter::revertDefinitionArrival(*const_cast<VarDecl *>(VD));
  }

  if (Rec.has(VarMutation::SpecInfo)) {
    // Unlike functions, a variable specialization IS a distinct type.
    const auto *Spec = cast<VarTemplateSpecializationDecl>(VD);
    if (auto *Chain = Tracker.getChainFor(Spec)) {
      if (const VarSpecializationFootprint *Restored = Chain->mostRecent())
        DeclStateReverter::restoreVarSpecializationFootprint(
            *Restored, *const_cast<VarTemplateSpecializationDecl *>(Spec));
    }
  }

  if (Rec.has(VarMutation::MemberSpecInfo)) {
    assert(VD->getMemberSpecializationInfo());
    if (auto *Chain = Tracker.getMemberSpecChainFor(VD)) {
      if (const MemberSpecializationFootprint *Restored = Chain->mostRecent())
        DeclStateReverter::restoreMemberSpecializationFootprint(
            *Restored, *const_cast<VarDecl *>(VD));
    }
  }
}

void PTUMutationActions::restoreEnum(PTUID ID, const EnumDecl *ED,
                                     MutationRecord &Rec) {

  // Like a class, an EnumDecl's TypeForDecl is shared across its whole
  // redeclaration chain (`enum class E : int;` forward-declared, then
  // defined later), so every redecl's cached pointer needs checking.
  if (Rec.has(MutationType::TypeForDecl)) {
    for (const TagDecl *Redecl : ED->redecls()) {
      const Type *LastKnown =
          Tracker.getTagDeclTypeInfo().mostRecent(Redecl).value_or(nullptr);
      const TypeDecl *TD = cast<TypeDecl>(Redecl);
      if (LastKnown != TD->getTypeForDecl())
        const_cast<TypeDecl *>(TD)->setTypeForDecl(LastKnown);
    }
  }

  if (Rec.has(MutationType::DefinitionInstantiate))
    DeclStateReverter::revertDefinitionArrival(*const_cast<EnumDecl *>(ED));

  if (Rec.has(MutationType::MemberSpecInfo)) {
    // A scoped member enumeration of a class template -- instantiated with
    // the enclosing specialization, carrying MSI back to the pattern enum.
    assert(ED->getMemberSpecializationInfo());
    if (auto *Chain = Tracker.getMemberSpecChainFor(ED)) {
      if (const MemberSpecializationFootprint *Restored = Chain->mostRecent())
        DeclStateReverter::restoreMemberSpecializationFootprint(
            *Restored, *const_cast<EnumDecl *>(ED));
    }
  }
}

void PTUMutationActions::restoreDecl(PTUID ID, const Decl *D,
                                     MutationRecord &Rec) {
  switch (Rec.S) {
  case DeclShape::Class:
    restoreClass(ID, cast<CXXRecordDecl>(D), Rec);
    break;
  case DeclShape::Function:
    restoreFunction(ID, cast<FunctionDecl>(D), Rec);
    break;
  case DeclShape::Var:
    restoreVar(ID, cast<VarDecl>(D), Rec);
    break;
  case DeclShape::Enum:
    restoreEnum(ID, cast<EnumDecl>(D), Rec);
    break;
  case DeclShape::Template:
    restoreTemplate(ID, cast<RedeclarableTemplateDecl>(D), Rec);
    break;
  case DeclShape::Typedef:
    restoreTypedef(ID, cast<TypedefNameDecl>(D), Rec);
    break;
  case DeclShape::None:
    break;
  }
}

void PTUMutationActions::restore(TranslationUnitDecl *ThisTU) {
  PTUStateInfo &Cur = Tracker.current();
  assert(ThisTU == Cur.ThisPTU);
  //   TranslationUnitDecl *ThisTU = Cur.ThisPTU;
  PTUID ID = Cur.ID;
  if (Cur.Commited)
    Tracker.undoLastEntries();
  else {
    // TODO: verify Mutations;
    Cur.verifyMutations(*this);
    Tracker.getHiddenMutationTracker().sweep(
        *this, [&](const Decl *D, DeclShape S, uint32_t K) {
          Cur.noteMutated(D, S, K);
        });
  }

  for (auto &[D, Rec] : Cur.Mutations)
    restoreDecl(ID, D, Rec);

  DeclStateReverter Detacher(Tracker.getPTUSlabCheckpoints());
  DeclStateUnlinkPolicy Proxy(ID, Tracker, Detacher);

  for (const Decl *D : Cur.ImplicitDecls)
    if (auto A = Proxy.actionFor(D))
      Proxy.runAction(D, A);

  walkDecls(ThisTU, Proxy);

  // Sema::SpecialMemberCache (public) caches, per (RD, kind+qualifiers), the
  // CXXMethodDecl* a prior LookupSpecialMember() call resolved to -- and on
  // a cache hit it returns that decl directly, with no re-validation at all
  // (SemaLookup.cpp). Any entry whose cached method was created by the PTU
  // being rolled back is now stale, so purge it here rather than trying to
  // reconstruct which of the (RD, kind) keys it could occupy.
  {
    Sema &SemaRef = Tracker.getSema();
    llvm::SmallVector<Sema::SpecialMemberCacheKey, 8> Stale;
    for (auto &Entry : SemaRef.SpecialMemberCache) {
      CXXMethodDecl *MD = Entry.second.getMethod();
      if (MD &&
          (Tracker.isFromThisPTU(MD, ID) || Cur.ImplicitDecls.contains(MD)))
        Stale.push_back(Entry.first);
    }
    for (const auto &Key : Stale)
      SemaRef.SpecialMemberCache.erase(Key);
  }

  // Must be last: Cur/ID/ThisTU are all references into (or derived from)
  // the PTU this call retires, and popCurrentPTU() destroys that entry.
  // Runs for BOTH branches above (committed or not) -- a PTU that never
  // committed still needs its checkpoint-ledger slot and PTUStack
  // entry reclaimed, otherwise every failed input would permanently grow
  // both, and NextID would desync from PTUSlabCheckpoints on top of that.
  Tracker.popCurrentPTU();
}

/// -----------------------------------------------------------------------
////////////////////////// MutationListener ///////////////////////////////
/// -----------------------------------------------------------------------

void PTUMutationRecorder::CompletedTagDefinition(const TagDecl *D) {
  if (!D->isDefinedOutsideFunctionOrMethod())
    return; // ordinary local class -- structurally unreachable by any other
            // PTU

  PTUStateInfo &Cur = Tracker.current();
  if (Tracker.isFromThisPTU(D, Tracker.currentID())) {
    if (D->isImplicit())
      Cur.ImplicitDecls.insert(D);
    return;
  }
  // Predates this PTU: a genuine cross-PTU mutation -- needs the full
  // definition-data footprint captured in commit().
  // Cur.TouchedTagDecls.insert(D);
  DeclShape S = classifyShape(D);

  /// doing this for memspec because we are not sure that if this cause member
  /// spec mutation or not we can't put compare check here as well since
  /// notifer never guarranty that will called before mutation or later. so
  /// this check about mutation will be handled in verification layer
  Cur.noteMutated(D, S, MutationType::DefinitionInstantiate);
  if (S == DeclShape::Enum) {
    if (const EnumDecl *ED = dyn_cast<EnumDecl>(D);
        ED && ED->getMemberSpecializationInfo())
      Cur.noteMutated(D, S, MutationType::MemberSpecInfo);
  } else if (S == DeclShape::Class) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
        RD && RD->getMemberSpecializationInfo())
      Cur.noteMutated(D, S, MutationType::MemberSpecInfo);
    if (isa<ClassTemplateSpecializationDecl>(D))
      Cur.noteMutated(D, S, MutationType::SpecInfo);
  }
}

void PTUMutationRecorder::AddedVisibleDecl(const DeclContext *DC,
                                           const Decl *D) {
  PTUStateInfo &Cur = Tracker.current();
  if (D->isImplicit())
    Cur.ImplicitDecls.insert(D);

  if (Tracker.isFromThisPTU(DC, Cur.ID) || Cur.TouchedDC.contains(DC))
    return;
  //   auto Origin =
  //       Tracker.getPTUSlabCheckpoints().attribute(Tracker.Ctx, DC);
  //   assert(Origin && "must have existing PTUID");
  //   assert(*Origin != Cur.ID);
  Cur.TouchedDC.insert(DC);
}

void PTUMutationRecorder::AddedCXXImplicitMember(const CXXRecordDecl *RD,
                                                 const Decl *D) {
  if (!RD->isDefinedOutsideFunctionOrMethod())
    return; // ordinary local class -- structurally unreachable by any other
            // PTU

  PTUStateInfo &Cur = Tracker.current();

  if (D->isImplicit())
    Cur.ImplicitDecls.insert(D);
  // cover this part as well
  // if (RD->MemberInfo and D is implcit plus has MSIInfo)

  // cover only not from this ptu.
  if (Tracker.isFromThisPTU(RD, Cur.ID))
    return;

  Cur.noteMutated(RD, DeclShape::Class, MutationType::DefinitionData);
}

template <typename TemplateT, typename SpecT>
void PTUMutationRecorder::noteTemplateDeclMutation(const TemplateT *TD,
                                                   const SpecT *Spec,
                                                   DeclShape S) {
  PTUStateInfo &Cur = Tracker.current();

  if (Spec->isImplicit())
    Cur.ImplicitDecls.insert(Spec);

  if (Tracker.isFromThisPTU(TD, Cur.ID))
    return;

  Cur.noteMutated(TD, DeclShape::Template, MutationType::SpecializationAdded);
}

void PTUMutationRecorder::AddedCXXTemplateSpecialization(
    const ClassTemplateDecl *TD, const ClassTemplateSpecializationDecl *D) {
  noteTemplateDeclMutation(TD, TD, DeclShape::Class);
}

void PTUMutationRecorder::AddedCXXTemplateSpecialization(
    const VarTemplateDecl *TD, const VarTemplateSpecializationDecl *D) {
  noteTemplateDeclMutation(TD, TD, DeclShape::Var);
}

void PTUMutationRecorder::AddedCXXTemplateSpecialization(
    const FunctionTemplateDecl *TD, const FunctionDecl *D) {
  noteTemplateDeclMutation(TD, TD, DeclShape::Function);
}

void PTUMutationRecorder::noteExceptionSpecMutation(const FunctionDecl *FD,
                                                    MutationType K) {
  PTUStateInfo &Cur = Tracker.current();
  if (Tracker.isFromThisPTU(FD, Cur.ID))
    return;
  Cur.noteMutated(FD, DeclShape::Function, K);
}

void PTUMutationRecorder::ResolvedExceptionSpec(const FunctionDecl *FD) {
  noteExceptionSpecMutation(FD);
}

void PTUMutationRecorder::DeducedReturnType(const FunctionDecl *FD,
                                            QualType ReturnType) {
  noteExceptionSpecMutation(FD, MutationType::DeducedReturnType);
}

void PTUMutationRecorder::noteDefinitionInstantiated(const Decl *D) {
  PTUStateInfo &Cur = Tracker.current();
  if (Tracker.isFromThisPTU(D, Cur.ID)) {
    return;
  }
  Cur.noteMutated(D, classifyShape(D), MutationType::DefinitionInstantiate);
}

/// InstantiationRequested fires when Sema decides a template entity needs
/// instantiating -- before the instantiation actually happens.
void PTUMutationRecorder::InstantiationRequested(const ValueDecl *D) {
  PTUStateInfo &Cur = Tracker.current();

  if (Tracker.isFromThisPTU(D, Cur.ID)) {
    // Created this PTU -- the creation path covers it, no mutation to note.
    // But an implicitly-created decl may never be appended to any lexical
    // chain, so walkTU would never see it. Record it here as the
    // only path that will.
    if (D->isImplicit())
      Cur.ImplicitDecls.insert(D);
    return;
  }

  DeclShape S;
  MutationType K = MutationType::None;

  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    S = DeclShape::Function;
    if (FD->getMemberSpecializationInfo())
      K = MutationType::MemberSpecInfo;
    else if (FD->getTemplateSpecializationKind() != TSK_Undeclared)
      // A function template specialization is a plain FunctionDecl
      // carrying FunctionTemplateSpecializationInfo -- there is no
      // distinct decl type to dyn_cast to, unlike the variable case.
      K = MutationType::SpecInfo;
    else
      return; // neither -- nothing this notifier can be about
  } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
    S = DeclShape::Var;
    if (VD->getMemberSpecializationInfo())
      K = MutationType::MemberSpecInfo;
    else if (isa<VarTemplateSpecializationDecl>(VD))
      K = MutationType::SpecInfo;
    else
      return;
  } else {
    return; // no other ValueDecl kind participates in instantiation
  }

  Cur.noteMutated(D, S, K);
}

} // end namespace clang