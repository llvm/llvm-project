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

#define DECL_SHAPES                                                            \
  DECL_SHAPE(Class, CXXRecordDecl)                                             \
  DECL_SHAPE(Function, FunctionDecl)                                           \
  DECL_SHAPE(Var, VarDecl)                                                     \
  DECL_SHAPE(Enum, EnumDecl)                                                   \
  DECL_SHAPE(Template, RedeclarableTemplateDecl)                               \
  DECL_SHAPE(Typedef, TypedefNameDecl)

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::Helpers ///////////////////////
/// -----------------------------------------------------------------------

DefinitionDataFootprint
DeclStateReverter::createDefinitionDataFootprint(const CXXRecordDecl &RD) {
  DefinitionDataFootprint FP;
  const auto &Live = RD.data();
#define FIELD(Name, Width, Merge) FP.Name = Live.Name;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
  return FP;
}

// bool DeclStateReverter::compareDefinitionDataFootprint(
//     const DefinitionDataFootprint &FP, const CXXRecordDecl &RD) {
//   const auto &Live = RD.data();
// #define FIELD(Name, Width, Merge) \
//   if (FP.Name != Live.Name) \
//     return false;
// #include "clang/AST/CXXRecordDeclDefinitionBits.def"
//   return true;
// }

void DeclStateReverter::restoreDefinitionDataFootprint(
    const DefinitionDataFootprint &FP, CXXRecordDecl &RD) {
  auto &Live = RD.data();
#define FIELD(Name, Width, Merge) Live.Name = FP.Name;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
}

void DeclStateReverter::restoreDefinitionAndRevertDC(CXXRecordDecl &RD) {
  // 1. DefinitionData pointer -- back to null.
  RD.DefinitionData = nullptr;

  // 2. DeclContext member list -- back to empty.
  clearDeclContextChain(RD);
  DeclContext *Primary = RD.getDeclContext()->getPrimaryContext();
  if (StoredDeclsMap *Map = Primary->getLookupPtr())
    Map->clear();

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
  const auto *RD = dyn_cast<CXXRecordDecl>(D);
  if (!RD)
    return;
  for (const auto *I : RD->redecls())
    const_cast<CXXRecordDecl *>(cast<CXXRecordDecl>(I))->DefinitionData =
        nullptr;
}

void DeclStateReverter::detachCommonBase(const RedeclarableTemplateDecl *RT) {
  for (const auto *R : RT->redecls())
    clearCommonPtr(*R);
}

void DeclStateReverter::removeFromLookupMap(NamedDecl *ND, DeclContext *DC,
                                            PTUID ID) {}

void DeclStateReverter::detachFromExternCLookup(Decl *D,
                                                DeclContext *ExternCCtx,
                                                PTUID ID) {
  auto *ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return;

  bool IsExternC = false;
  if (const auto *FD = dyn_cast<FunctionDecl>(ND))
    IsExternC = FD->isExternC();
  else if (const auto *VD = dyn_cast<VarDecl>(ND))
    IsExternC = VD->isExternC();

  if (IsExternC)
    removeFromLookupMap(ND, ExternCCtx, ID);
}

void DeclStateReverter::removeFromIdResolver(Sema &S, NamedDecl *D) {
  if (D->getDeclName().getFETokenInfo())
    S.IdResolver.RemoveDecl(D);
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

  if (auto *FD = dyn_cast<FunctionDecl>(D))
    unlinkRedeclChain(FD);
  else if (auto *VD = dyn_cast<VarDecl>(D))
    unlinkRedeclChain(VD);
  else if (auto *TD = dyn_cast<TagDecl>(D))
    unlinkRedeclChain(TD);
  else if (auto *RT = dyn_cast<RedeclarableTemplateDecl>(D))
    unlinkRedeclChain(RT);
  else if (auto *NS = dyn_cast<NamespaceDecl>(D))
    unlinkRedeclChain(NS);

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
uint32_t PTUMutationActions::DeclNeedingTracking(DeclShape S, const Decl *D) {
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

template <typename T>
static bool differsFromLast(IncrementalStateTracker &Tracker, const Decl *Owner,
                            MutationType K, const T &LiveValue) {
  const auto *Last = Tracker.getFootprints().mostRecent<T>(Owner, K);
  assert(Last && "baseline missing at verify time");
  return !Last || *Last != LiveValue;
}

template <typename FootprintT, typename OwnerT>
static FootprintT Footprint(const OwnerT &Owner) {
  FootprintT FP;
  FP.update(Owner);
  return FP;
}

/// The single place to check whether a kind actually changed.
///
/// It handles both listener-less kinds and SpecInfo/MemberSpecInfo. The
/// latter can be checked from both the sweep and the listener re-check.
///
/// Keeping everything in one function makes sure SpecInfo/MemberSpecInfo
/// changes are caught even when the listener never fires.
uint32_t PTUMutationActions::confirmMutation(const Decl *D, DeclShape S,
                                             uint32_t FlaggedKinds) {
  uint32_t Verified = 0;
  switch (S) {
  case DeclShape::Class: {
    const auto *RD = cast<CXXRecordDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::TypeForDecl)) {
      // No listener at all for this mutation.
      // const Type *Now = DeclStateReverter::getRawTypeForDecl(RD);
      // if (differsFromLast(Tracker, RD, MutationType::TypeForDecl, Now))
      //   Verified |= uint32_t(MutationType::TypeForDecl);
    }
    if (FlaggedKinds & uint32_t(MutationType::SpecInfo)) {
      if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD))
        if (differsFromLast(Tracker, Spec, MutationType::SpecInfo,
                            Footprint<SpecializationFootprint>(*Spec)))
          Verified |= uint32_t(MutationType::SpecInfo);
    } else if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      if (differsFromLast(Tracker, RD, MutationType::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*RD)))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Function: {
    const auto *FD = cast<FunctionDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::SpecInfo)) {
      if (differsFromLast(Tracker, FD, MutationType::SpecInfo,
                          Footprint<FunctionSpecializationFootprint>(*FD)))
        Verified |= uint32_t(MutationType::SpecInfo);
    } else if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      if (differsFromLast(Tracker, FD, MutationType::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*FD)))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    break; // Function carries no listener-less kind at all.
  }
  case DeclShape::Var: {
    const auto *VD = cast<VarDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::EvaluatedValue) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            VD, uint32_t(MutationType::EvaluatedValue))) {
      // No chain to compare against -- write-once field.
      const EvaluatedStmt *Eval = VD->getEvaluatedStmt();
      if (Eval && Eval->WasEvaluated)
        Verified |= uint32_t(MutationType::EvaluatedValue);
    }
    if (FlaggedKinds & uint32_t(MutationType::SpecInfo)) {
      if (const auto *Spec = dyn_cast<VarTemplateSpecializationDecl>(VD))
        if (differsFromLast(Tracker, Spec, MutationType::SpecInfo,
                            Footprint<VarSpecializationFootprint>(*Spec)))
          Verified |= uint32_t(MutationType::SpecInfo);
    } else if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      if (differsFromLast(Tracker, VD, MutationType::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*VD)))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Enum: {
    const auto *ED = cast<EnumDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::TypeForDecl)) {
      // const Type *Now = DeclStateReverter::getRawTypeForDecl(ED);
      // if (differsFromLast(Tracker, ED, MutationType::TypeForDecl, Now))
      //   Verified |= uint32_t(MutationType::TypeForDecl);
    }
    if (FlaggedKinds & uint32_t(MutationType::MemberSpecInfo)) {
      if (differsFromLast(Tracker, ED, MutationType::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*ED)))
        Verified |= uint32_t(MutationType::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Template: {
    // Use the canonical key. commitTemplate() tracks and settles CommonBase
    // modifications, and CommonPtr is shared across the redeclaration chain.
    // Any modification to CommonPtr is reflected in the canonical declaration.
    const auto *RT = cast<RedeclarableTemplateDecl>(D)->getCanonicalDecl();
    if (FlaggedKinds & uint32_t(MutationType::TemplateCommon) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            RT, uint32_t(MutationType::TemplateCommon))) {
      // No chain to compare against -- write-once.
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
    break;
  }
  case DeclShape::Typedef: {
    const auto *TD = cast<TypedefNameDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::TypeForDecl) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            TD, uint32_t(MutationType::TypeForDecl))) {
      // No chain to compare against -- write-once field.
      // if (DeclStateReverter::getRawTypeForDecl(TD))
      //   Verified |= uint32_t(MutationType::TypeForDecl);
    }
    break;
  }
  case DeclShape::None:
    break;
  }
  return Verified;
}

template <typename OnConfirmedFn>
void SweepTracker::sweep(PTUMutationActions &Act, OnConfirmedFn OnConfirmed) {
  for (auto &Entry : Active) {
    const Decl *D = Entry.getFirst();
    uint32_t StillOpen = Entry.getSecond();
    DeclShape S = classifyShape(D);
    if (uint32_t Newly = Act.confirmMutation(D, S, StillOpen)) {
      OnConfirmed(D, S, Newly);
    }
  }
}

// These kinds have listeners, but the listener alone can’t reliably confirm
// a real change. So, compare the current value with the last known value,
// just like we do for kinds without listeners.
//
// These kinds are also checked by the sweep. This is needed because the
// listener can report a change that didn’t happen, or fail to fire when a
// real change did happen. The sweep catches the second case.
static constexpr uint32_t KindsNeedingVerification =
    uint32_t(MutationType::SpecInfo) | uint32_t(MutationType::MemberSpecInfo);

/// Start tracking D for the mutation kinds that need live verification.
/// Called when a new mutation is committed or an undone mutation is restored.
/// The callers have already established which kinds need tracking, so no
/// additional DeclNeedingTracking check is needed.
void PTUMutationActions::registerLiveVerification(const Decl *D,
                                                  uint32_t Kinds) {
  if (uint32_t Live = Kinds & KindsNeedingVerification)
    HiddenMutationTracker.track(D, Live);
}

/// Stop tracking Decl D when its mutation no longer needs live verification.
/// Called from verifyMutations() and the !IsNew SpecInfo/MemberSpecInfo
/// paths in commitDecl. Re-check DeclNeedingTracking to match the same
/// "is this still open?" condition used when D is first discovered:
/// keep tracking if a later PTU can still update it; otherwise settle it.
void PTUMutationActions::settleIfClosed(const Decl *D, DeclShape S,
                                        uint32_t Confirmed) {
  uint32_t Live = Confirmed & KindsNeedingVerification;
  if (!Live)
    return;
  uint32_t StillOpen = DeclNeedingTracking(S, D) & Live;
  uint32_t Closed = Live & ~StillOpen;
  if (Closed && HiddenMutationTracker.isTrackedFor(D, Closed))
    HiddenMutationTracker.settle(D, Closed);
}

/// Verify recorded mutations and remove any spurious ones. If none of a
/// decl’s mutation kinds are confirmed, remove the record entirely so
/// restore() does not pop chain entries that were never written.
void PTUStateInfo::verifyMutations(PTUMutationActions &Actions) {
  Mutations.remove_if([&](auto &Entry) -> bool {
    const Decl *D = Entry.first;
    MutationRecord &Rec = Entry.second;
    uint32_t Flagged = Rec.MutationType & KindsNeedingVerification;
    if (!Flagged)
      return false; // nothing here needs live verification.
    uint32_t Confirmed = Actions.confirmMutation(D, Rec.S, Flagged);
    Actions.settleIfClosed(D, Rec.S, Confirmed);
    Rec.MutationType = (Rec.MutationType & ~Flagged) | Confirmed;
    return Rec.MutationType == 0;
  });
}

class DeclStateCommitPolicy {
  PTUID ID;
  PTUMutationActions &Action;

public:
  DeclStateCommitPolicy(PTUID ID, PTUMutationActions &Action)
      : ID(ID), Action(Action) {}

  bool shouldRecurse(const Decl *D) const {
    if (isa<NamespaceDecl>(D))
      return true;
    if (const auto *RD = dyn_cast<CXXRecordDecl>(D))
      return RD->isCompleteDefinition();
    return false;
  }

  void process(const Decl *D) const {
    if (!D->isDefinedOutsideFunctionOrMethod())
      return;

    DeclShape S = classifyShape(D);
    if (S == DeclShape::None)
      return;

    // we have to handle every case that if a decl need to create baseline
    // info or if any decl has done all mutations already in this for exmpla
    // membe spec already properly done like this kind of case so we don't
    // create spec info blindly
    uint32_t Kinds = Action.DeclNeedingTracking(S, D);
    if (!Kinds)
      return; // structurally immutable -- no baseline needed

    MutationRecord Rec{S, Kinds};
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

  bool shouldRecurse(const Decl *D) const {
    // we don't care about the decl which complete chain is part belong to
    // current PTU.
    if (Tracker.isFromThisPTU(D->getCanonicalDecl(), ID))
      return false;
    return isa<NamespaceDecl>(D);
  }

  void process(const Decl *D) {
    Reverter.detachFromExternCLookup(
        const_cast<Decl *>(D), Tracker.getASTContext().getExternCContextDecl(),
        ID);
    if (auto *ND = dyn_cast<NamedDecl>(const_cast<Decl *>(D)))
      DeclStateReverter::removeFromIdResolver(Tracker.getSema(), ND);

    const Decl *Canon = D->getCanonicalDecl();

    // Whether Candidate's own DefinitionData belongs to this PTU and needs
    // detaching.
    auto needsDefDataDetach = [&](const Decl *Candidate) {
      const auto *RD = dyn_cast<CXXRecordDecl>(Candidate);
      const CXXRecordDecl *Def = RD ? RD->getDefinition() : nullptr;
      return Def && Tracker.isFromThisPTU(Def, ID);
    };

    /// Keyed by the canonical declaration: all redeclarations share a single
    /// RedeclLink, so truncating more than once would advance past the intended
    /// surviving declaration.
    bool DetachFromRedecl = !RepairedChains.count(Canon) &&
                            D->getPreviousDecl() &&
                            !Tracker.isFromThisPTU(Canon, ID);

    if (needsDefDataDetach(D))
      Reverter.detachDefData(D);

    // Keyed on the primary context, a different granularity: a reopened
    // namespace's redeclarations all resolve to the same primary.
    if (const DeclContext *DC = D->getDeclContext()->getPrimaryContext();
        TouchedDC.contains(DC))
      Reverter.detachFromDCLookup(const_cast<Decl *>(D), ID);

    if (DetachFromRedecl) {
      RepairedChains.insert(Canon);
      Reverter.detachFromRedeclChain(D, ID);
    }

    if (const auto *RT = dyn_cast<RedeclarableTemplateDecl>(D)) {
      const NamedDecl *Templated = RT->getTemplatedDecl();
      if (needsDefDataDetach(Templated))
        Reverter.detachDefData(Templated);
      if (DetachFromRedecl)
        Reverter.detachFromRedeclChain(Templated, ID);
    }

    if (const DeclContext *LexicalDC = D->getLexicalDeclContext();
        D->isImplicit() && !RepairedLexicalContexts.count(LexicalDC)) {
      Reverter.repairLexicalChain(*const_cast<DeclContext *>(LexicalDC), ID);
      RepairedLexicalContexts.insert(LexicalDC);
    }
  }
};

template <typename DeclStatePolicyT>
void PTUMutationActions::walkDecls(const DeclContext *DC,
                                   DeclStatePolicyT &Policy) {
  for (const Decl *D : DC->decls()) {
    if (Policy.shouldRecurse(D))
      walkDecls(cast<DeclContext>(D), Policy);
    Policy.process(D);
  }
}

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::commit ///////////////////////
/// -----------------------------------------------------------------------

void PTUMutationActions::commitMembers(PTUID ID, const DeclContext *Members) {
  DeclStateCommitPolicy Policy(ID, *this);
  walkDecls(Members, Policy);
}

void PTUMutationActions::commitClass(PTUID ID, const CXXRecordDecl *RD,
                                     MutationRecord &Rec, bool IsNew) {

  using ClassMutation = MutationType;

  if (Rec.has(ClassMutation::TypeForDecl)) {
    // for (const TagDecl *Redecl : RD->redecls()) {
    //   const auto *Last = Tracker.getFootprints().mostRecent<const Type *>(
    //       Redecl, MutationType::TypeForDecl);
    //   const Type *LastKnown = Last ? *Last : nullptr;
    //   if (LastKnown != cast<TypeDecl>(Redecl)->getTypeForDecl())
    //     Tracker.commitFootprint(Redecl, MutationType::TypeForDecl, ID,
    //                             LastKnown);
    // }
  }

  if (Rec.has(ClassMutation::DefinitionInstantiate) ||
      Rec.has(ClassMutation::DefinitionData)) {
    Tracker.commitFootprint(
        RD, MutationType::DefinitionData, ID,
        DeclStateReverter::createDefinitionDataFootprint(*RD));

    /// SpecializationDecl can have lazy implicit generated body;
    if (Rec.has(ClassMutation::DefinitionInstantiate)) {
      if (RD->getMemberSpecializationInfo())
        commitMembers(ID, cast<DeclContext>(RD));
    }
  }

  if (Rec.has(ClassMutation::SpecInfo)) {
    const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD);
    if (Spec) {
      Tracker.commitFootprint(
          Spec, MutationType::SpecInfo, ID,
          DeclStateReverter::createFootprint<SpecializationFootprint>(*Spec));
      // Member declarations are instantiated eagerly with the class
      // definition regardless of TSK; only their definitions defer. Walk
      // them now so each member's MemberSpecializationInfo is tracked from
      // the moment it exists.
      if (IsNew && Spec->getSpecializationKind() == TSK_ImplicitInstantiation)
        commitMembers(ID, cast<DeclContext>(Spec));
      if (IsNew)
        registerLiveVerification(Spec, uint32_t(MutationType::SpecInfo));
      else
        settleIfClosed(Spec, DeclShape::Class,
                       uint32_t(MutationType::SpecInfo));
    }
  }

  if (Rec.has(ClassMutation::MemberSpecInfo)) {
    // A nested member of a class template -- its MSI's kind or
    // point-of-instantiation changed.
    if (RD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          RD, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *RD));
      if (IsNew)
        registerLiveVerification(RD, uint32_t(MutationType::MemberSpecInfo));
      else
        settleIfClosed(RD, DeclShape::Class,
                       uint32_t(MutationType::MemberSpecInfo));
    }
  }
}

void PTUMutationActions::commitFunction(PTUID ID, const FunctionDecl *FD,
                                        MutationRecord &Rec, bool IsNew) {

  if (Rec.has(MutationType::ExceptionSpec) ||
      Rec.has(MutationType::DeducedReturnType)) {
    for (const FunctionDecl *Redecl : FD->redecls()) {
      if (IsNew) {
        Tracker.commitFootprint(Redecl, MutationType::ExceptionSpec, ID,
                                Redecl->getType());
      } else {
        const auto *Last = Tracker.Footprints.mostRecent<QualType>(
            Redecl, MutationType::ExceptionSpec);
        QualType LastKnown = Last ? *Last : QualType();
        if (LastKnown != Redecl->getType())
          Tracker.commitFootprint(Redecl, MutationType::ExceptionSpec, ID,
                                  LastKnown);
      }
    }
  }

  //   if (!IsNew && Rec.has(MutationType::DefinitionInstantiate)) {
  //   }

  if (Rec.has(MutationType::SpecInfo)) {
    assert(FD->getTemplateSpecializationKind() != TSK_Undeclared &&
           "SpecInfoChanged on a function with no specialization info");
    Tracker.commitFootprint(
        FD, MutationType::SpecInfo, ID,
        DeclStateReverter::createFootprint<FunctionSpecializationFootprint>(
            *FD));
    if (IsNew)
      registerLiveVerification(FD, uint32_t(MutationType::SpecInfo));
    else
      settleIfClosed(FD, DeclShape::Function, uint32_t(MutationType::SpecInfo));
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    if (FD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          FD, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *FD));
      if (IsNew)
        registerLiveVerification(FD, uint32_t(MutationType::MemberSpecInfo));
      else
        settleIfClosed(FD, DeclShape::Function,
                       uint32_t(MutationType::MemberSpecInfo));
    }
  }
}

void PTUMutationActions::commitVar(PTUID ID, const VarDecl *VD,
                                   MutationRecord &Rec, bool IsNew) {

  // EvaluatedStmt / APValue: populated on the FIRST constant-evaluation of
  // this variable, which can land in any PTU long after the VarDecl was
  // created. This is a mutation of an earlier PTU's decl, not a property
  // settled at declaration time.
  if (Rec.has(MutationType::EvaluatedValue)) {
    // No footprint chain -- write-once field with a fixed null prior
    // state (see restoreVar), so nothing to seed a baseline for.
    if (IsNew) {
      // Just created: register VD so a later commit's
      // SweepTracker::sweep() call notices if/when WasEvaluated actually
      // flips (no listener exists to tell us directly -- see
      // hiddenKindsMask).
      HiddenMutationTracker.track(VD, uint32_t(MutationType::EvaluatedValue));
    } else {
      // This IS the confirmation, arriving here via SweepTracker::sweep()'s
      // OnConfirmed callback (there is no other path to this bit -- no
      // listener exists for it). Write-once means terminal: once
      // WasEvaluated is true it can never become false again on its own,
      // so settle the bit now rather than waiting for the next sweep to
      // notice kindsNeedingTracking no longer flags it. restoreVar
      // is what re-track()s it, only if a rollback actually reverts this.
      HiddenMutationTracker.settle(VD, uint32_t(MutationType::EvaluatedValue));
    }
  }

  //   if (!IsNew && Rec.has(MutationType::DefinitionInstantiate)) {
  //   }

  if (Rec.has(MutationType::SpecInfo)) {
    // Unlike functions, a variable specialization IS a distinct type.
    const auto *Spec = cast<VarTemplateSpecializationDecl>(VD);
    Tracker.commitFootprint(
        Spec, MutationType::SpecInfo, ID,
        DeclStateReverter::createFootprint<VarSpecializationFootprint>(*Spec));
    if (IsNew)
      registerLiveVerification(Spec, uint32_t(MutationType::SpecInfo));
    else
      settleIfClosed(Spec, DeclShape::Var, uint32_t(MutationType::SpecInfo));
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    if (VD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          VD, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *VD));
      if (IsNew)
        registerLiveVerification(VD, uint32_t(MutationType::MemberSpecInfo));
      else
        settleIfClosed(VD, DeclShape::Var,
                       uint32_t(MutationType::MemberSpecInfo));
    }
  }
}

void PTUMutationActions::commitEnum(PTUID ID, const EnumDecl *ED,
                                    MutationRecord &Rec, bool IsNew) {

  if (Rec.has(MutationType::TypeForDecl)) {
    // for (const TagDecl *Redecl : ED->redecls()) {
    //   const Type *Now = Redecl->getTypeForDecl();
    //   const Type *LastKnown = nullptr;
    //   if (!IsNew) {
    //     const auto *Last = Tracker.getFootprints().mostRecent<const Type *>(
    //         Redecl, MutationType::TypeForDecl);
    //     LastKnown = Last ? *Last : nullptr;
    //   }
    //   if (IsNew || (LastKnown != Now))
    //     Tracker.commitFootprint(Redecl, MutationType::TypeForDecl, ID,
    //                             LastKnown);
    // }
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    // A scoped member enumeration of a class template -- instantiated with
    // the enclosing specialization, carrying MSI back to the pattern enum.
    if (ED->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          ED, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *ED));
      if (IsNew)
        registerLiveVerification(ED, uint32_t(MutationType::MemberSpecInfo));
      else
        settleIfClosed(ED, DeclShape::Enum,
                       uint32_t(MutationType::MemberSpecInfo));
    }
  }
}

void PTUMutationActions::commitTemplate(PTUID ID,
                                        const RedeclarableTemplateDecl *RT,
                                        MutationRecord &Rec, bool IsNew) {
  const RedeclarableTemplateDecl *RTCanon = RT->getCanonicalDecl();
  if (Rec.has(MutationType::TemplateCommon)) {
    if (IsNew)
      HiddenMutationTracker.track(RTCanon,
                                  uint32_t(MutationType::TemplateCommon));
    else
      HiddenMutationTracker.settle(RTCanon,
                                   uint32_t(MutationType::TemplateCommon));
  }

  if (Rec.has(MutationType::CanonInjectedTST)) {
    if (IsNew)
      HiddenMutationTracker.track(RTCanon,
                                  uint32_t(MutationType::CanonInjectedTST));
    else
      HiddenMutationTracker.settle(RTCanon,
                                   uint32_t(MutationType::CanonInjectedTST));
  }
}

void PTUMutationActions::commitTypedef(PTUID ID, const TypedefNameDecl *TD,
                                       MutationRecord &Rec, bool IsNew) {
  if (Rec.has(MutationType::TypeForDecl)) {
    if (IsNew)
      HiddenMutationTracker.track(TD, uint32_t(MutationType::TypeForDecl));
    else
      HiddenMutationTracker.settle(TD, uint32_t(MutationType::TypeForDecl));
  }
}

void PTUMutationActions::commitDecl(PTUID ID, const Decl *D,
                                    MutationRecord &Rec, bool IsNew) {
  switch (Rec.S) {
#define DECL_SHAPE(NAME, TYPE)                                                 \
  case DeclShape::NAME:                                                        \
    commit##NAME(ID, cast<TYPE>(D), Rec, IsNew);                               \
    break;
    DECL_SHAPES
#undef DECL_SHAPE
  case DeclShape::None:
    break;
  }
}

void PTUMutationActions::commit(TranslationUnitDecl *ThisTU) {
  PTUStateInfo &Cur = Tracker.current();
  assert(!Cur.Commited);
  PTUID ID = Cur.ID;

  Cur.verifyMutations(*this);
  Tracker.getHiddenMutationTracker().sweep(
      *this, [&](const Decl *D, DeclShape S, uint32_t K) {
        Cur.noteMutated(D, S, K);
      });

  for (auto &[D, Rec] : Cur.Mutations)
    commitDecl(ID, D, Rec, /*IsNew=*/false);

  DeclStateCommitPolicy Proxy(ID, *this);
  for (const Decl *D : Cur.ImplicitDecls)
    Proxy.process(D);

  walkDecls(ThisTU, Proxy);

  Cur.Commited = true;
}

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::restore //////////////////////
/// -----------------------------------------------------------------------

void PTUMutationActions::restoreClass(PTUID ID, const CXXRecordDecl *RD,
                                      MutationRecord &Rec) {

  if (Rec.has(MutationType::DefinitionInstantiate) ||
      Rec.has(MutationType::DefinitionData)) {
    if (Rec.has(MutationType::DefinitionInstantiate)) {
      DeclStateReverter::revertDefinitionArrival(
          *const_cast<CXXRecordDecl *>(RD));
    } else if (const auto *Restored =
                   Tracker.getFootprints().mostRecent<DefinitionDataFootprint>(
                       RD, MutationType::DefinitionData)) {
      DeclStateReverter::restoreDefinitionDataFootprint(
          *Restored, *const_cast<CXXRecordDecl *>(RD));
    }
  }

  // TypeForDecl is shared across the whole redeclaration chain -- revert all.
  if (Rec.has(MutationType::TypeForDecl)) {
    // for (const TagDecl *Redecl : RD->redecls()) {
    //   const auto *Last = Tracker.Footprints.mostRecent<const Type *>(
    //       Redecl, MutationType::TypeForDecl);
    //   const Type *LastKnown = Last ? *Last : nullptr;
    //   TODO: need to handle?
    // }
  }

  if (Rec.has(MutationType::SpecInfo)) {
    // Only a specialization can carry this kind; a plain CXXRecordDecl
    // reaching here means a note*() wrapper passed the wrong kind.
    const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD);
    assert(Spec && "SpecializationAdded noted on a non-specialization");
    if (const SpecializationFootprint *Restored =
            Tracker.getFootprints().mostRecent<SpecializationFootprint>(
                Spec, MutationType::SpecInfo))
      DeclStateReverter::restoreFootprint<SpecializationFootprint>(
          *Restored, *const_cast<ClassTemplateSpecializationDecl *>(Spec));
    registerLiveVerification(Spec, uint32_t(MutationType::SpecInfo));
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    // A nested member of a class template – its MSI kind or
    // point-of-instantiation changed.
    assert(RD->getMemberSpecializationInfo());
    if (const MemberSpecializationFootprint *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                RD, MutationType::MemberSpecInfo))
      DeclStateReverter::restoreFootprint<MemberSpecializationFootprint>(
          *Restored, *const_cast<CXXRecordDecl *>(RD));
    registerLiveVerification(RD, uint32_t(MutationType::MemberSpecInfo));
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
      const auto *Last = Tracker.getFootprints().mostRecent<QualType>(
          Redecl, MutationType::ExceptionSpec);
      QualType LastKnown = Last ? *Last : QualType();
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
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<FunctionSpecializationFootprint>(
                FD, MutationType::SpecInfo))
      DeclStateReverter::restoreFootprint<FunctionSpecializationFootprint>(
          *Restored, *const_cast<FunctionDecl *>(FD));
    registerLiveVerification(FD, uint32_t(MutationType::SpecInfo));
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    assert(FD->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                FD, MutationType::MemberSpecInfo))
      DeclStateReverter::restoreFootprint<MemberSpecializationFootprint>(
          *Restored, *const_cast<FunctionDecl *>(FD));
    registerLiveVerification(FD, uint32_t(MutationType::MemberSpecInfo));
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
  // EvaluatedStmt::WasEvaluated is write-once, so a recorded entry means this
  // PTU confirmed the evaluation.
  if (Rec.has(MutationType::EvaluatedValue)) {
    // VD cached evaluation came from this PTU.
    // Reset it and track it again so a later PTU can re-trigger evaluation.
    if (EvaluatedStmt *Eval = VD->getEvaluatedStmt()) {
      Eval->WasEvaluated = false;
      Eval->Evaluated = APValue();
    }
    HiddenMutationTracker.track(VD, uint32_t(MutationType::EvaluatedValue));
  }

  if (Rec.has(MutationType::DefinitionInstantiate)) {
    DeclStateReverter::revertDefinitionArrival(*const_cast<VarDecl *>(VD));
  }

  if (Rec.has(MutationType::SpecInfo)) {
    const auto *Spec = cast<VarTemplateSpecializationDecl>(VD);
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<VarSpecializationFootprint>(
                Spec, MutationType::SpecInfo))
      DeclStateReverter::restoreFootprint<VarSpecializationFootprint>(
          *Restored, *const_cast<VarTemplateSpecializationDecl *>(Spec));
    registerLiveVerification(VD, uint32_t(MutationType::SpecInfo));
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    assert(VD->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.Footprints.mostRecent<MemberSpecializationFootprint>(
                VD, MutationType::MemberSpecInfo))
      DeclStateReverter::restoreFootprint<MemberSpecializationFootprint>(
          *Restored, *const_cast<VarDecl *>(VD));
    registerLiveVerification(VD, uint32_t(MutationType::MemberSpecInfo));
  }
}

void PTUMutationActions::restoreEnum(PTUID ID, const EnumDecl *ED,
                                     MutationRecord &Rec) {

  if (Rec.has(MutationType::TypeForDecl)) {
    // for (const TagDecl *Redecl : ED->redecls()) {
    //   const auto *Last = Tracker.Footprints.mostRecent<const Type *>(
    //       Redecl, MutationType::TypeForDecl);
    //   const Type *LastKnown = Last ? *Last : nullptr;
    //   TODO: need to handle?
    // }
  }

  //   if (Rec.has(MutationType::DefinitionInstantiate))
  //     DeclStateReverter::revertDefinitionArrival(*const_cast<EnumDecl
  //     *>(ED));

  if (Rec.has(MutationType::MemberSpecInfo)) {
    assert(ED->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                ED, MutationType::MemberSpecInfo))
      DeclStateReverter::restoreFootprint<MemberSpecializationFootprint>(
          *Restored, *const_cast<EnumDecl *>(ED));

    registerLiveVerification(ED, uint32_t(MutationType::MemberSpecInfo));
  }
}

void PTUMutationActions::restoreDecl(PTUID ID, const Decl *D,
                                     MutationRecord &Rec) {
  switch (Rec.S) {
#define DECL_SHAPE(NAME, TYPE)                                                 \
  case DeclShape::NAME:                                                        \
    restore##NAME(ID, cast<TYPE>(D), Rec);                                     \
    break;
    DECL_SHAPES
#undef DECL_SHAPE
  case DeclShape::None:
    break;
  }
}
#undef DECL_SHAPES

void PTUMutationActions::restoreSpecialMemberCache(PTUID ID) {
  PTUStateInfo &Cur = Tracker.current();
  if (!Cur.HadImplicitCXXMember)
    return;

  Sema &SemaRef = Tracker.getSema();
  llvm::SmallVector<Sema::SpecialMemberCacheKey, 8> Stale;
  for (auto &Entry : SemaRef.SpecialMemberCache) {
    CXXMethodDecl *MD = Entry.second.getMethod();
    if (MD && (Tracker.isFromThisPTU(MD, ID) || Cur.ImplicitDecls.contains(MD)))
      Stale.push_back(Entry.first);
  }
  for (const auto &Key : Stale)
    SemaRef.SpecialMemberCache.erase(Key);
}

void PTUMutationActions::restore(TranslationUnitDecl *ThisTU) {
  PTUStateInfo &Cur = Tracker.current();
  assert((!Cur.Commited || ThisTU == Cur.ThisPTU) &&
         "MostRecentTU doesn't match the TU recorded at commit time");
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
  DeclStateUnlinkPolicy Policy(ID, Tracker, Detacher);

  for (const Decl *D : Cur.ImplicitDecls)
    Policy.process(D);

  walkDecls(ThisTU, Policy);

  // SpecialMemberCache stores the CXXMethodDecl* resolved by a previous
  // LookupSpecialMember() call and returns it directly on a cache hit.
  // If that method was created by a PTU that is being rolled back, the entry
  // is stale, so remove it here.
  restoreSpecialMemberCache(ID);

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

  DeclShape S = classifyShape(D);

  Cur.noteMutated(D, S, MutationType::DefinitionInstantiate);

  /// Handle this for MemberSpec because the notifier does not guarantee whether
  /// it fires before or after the mutation. We cannot reliably compare the
  /// state here, so mutation detection is deferred to the verification layer.
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

// Remove all entries for each (Owner, Kind) key written by this PTU.
void IncrementalStateTracker::undoLastEntries() {
  PTUStateInfo &Cur = current();
  PTUID ID = Cur.ID;

  for (auto &[Owner, K] : Cur.TouchedFootprints)
    Footprints.removeFrom(Owner, K, ID);
}

} // end namespace clang