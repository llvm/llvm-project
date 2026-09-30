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

void DeclStateReverter::restoreDefinitionDataFootprint(
    const DefinitionDataFootprint &FP, CXXRecordDecl &RD) {
  auto &Live = RD.data();
#define FIELD(Name, Width, Merge) Live.Name = FP.Name;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
}

void DeclStateReverter::revertDefinitionArrival(Decl *D) {
  if (auto *FD = dyn_cast<FunctionDecl>(D)) {
    FD->setBody(nullptr);
    // Reset the body state so a later PTU sees this as not instantiated.
    FD->setWillHaveBody(false);

    // The reverted body may have caused a substitution failure to mark
    // the function invalid. Clear that state as well.
    FD->setInvalidDecl(false);
  } else if (auto *VD = dyn_cast<VarDecl>(D)) {
    VD->setInit(nullptr);
    VD->setInvalidDecl(false);
  } else if (auto *Field = dyn_cast<FieldDecl>(D))
    Field->setInClassInitializer(nullptr);
  else if (auto *ED = dyn_cast<EnumDecl>(D)) {
    // Undo EnumDecl::completeDefinition() writes. PromotionType and both
    // bit-count fields are always overwritten during completion, so they can
    // be reset unconditionally.
    ED->setPromotionType(QualType());
    // resetEnumBits(*ED);
    // EnumDecl::setNumPositiveBits/setNumNegativeBits -- private.
    // ED->setNumPositiveBits(0);
    // ED->setNumNegativeBits(0);

    // completeDefinition() only sets IntegerType for non-fixed enums. A fixed
    // enum may already have an IntegerType from its forward declaration, which
    // must be preserved.
    if (!ED->isFixed())
      ED->setIntegerType(QualType());
    ED->setCompleteDefinition(false);
    clearBeingDefined(*ED);
  } else if (auto *RD = dyn_cast<CXXRecordDecl>(D)) {
    // 1. DefinitionData pointer -- back to null.
    RD->DefinitionData = nullptr;

    // 2. DeclContext member list -- back to empty.
    clearDeclContextChain(*RD);
    DeclContext *Primary = RD->getDeclContext()->getPrimaryContext();
    if (StoredDeclsMap *Map = Primary->getLookupPtr())
      Map->clear();

    // 3. TagDecl completion bits -- back to "never started". Flipped by
    // startDefinition()/completeDefinition().
    clearBeingDefined(*RD);
    RD->setCompleteDefinition(false);
  }
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

void DeclStateReverter::detachFromExternCLookup(Decl *D,
                                                DeclContext *ExternCCtx,
                                                PTUID ID) {
  auto *ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return;

  // bool IsExternC = false;
  // if (const auto *FD = dyn_cast<FunctionDecl>(ND))
  //   IsExternC = FD->isExternC();
  // else if (const auto *VD = dyn_cast<VarDecl>(ND))
  //   IsExternC = VD->isExternC();

  // if (IsExternC)
  //   removeFromLookupMap(ND, ExternCCtx, ID);

  if (StoredDeclsMap *Map = ExternCCtx->getPrimaryContext()->getLookupPtr()) {
    auto It = Map->find(ND->getDeclName());
    if (It != Map->end())
      It->second.remove(ND);
  }
}

void DeclStateReverter::removeFromIdResolver(Sema &S, NamedDecl *D) {
  if (D->getDeclName().isEmpty())
    return;
  if (D->getDeclName().getFETokenInfo())
    S.IdResolver.RemoveDecl(D);
}

/// Remove D from its DeclContext lookup map and restore the previous
/// declaration if it was replaced.
void DeclStateReverter::detachFromDCLookup(Decl *D, PTUID ID) {
  NamedDecl *ND = dyn_cast<NamedDecl>(D);
  if (!ND)
    return;

  // Find any declaration from the previous PTU in the redecl chain that may
  // have been replaced by this declaration.
  NamedDecl *Survivor = findSurvivor(ND, ID);
  DeclarationName Name = ND->getDeclName();

  DeclContext *DC = ND->getDeclContext();
  do {
    DeclContext *Primary = DC->getPrimaryContext();
    if (StoredDeclsMap *Map = Primary->getLookupPtr()) {
      auto Pos = Map->find(Name);
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

// Walk D's redecl chain looking for the newest decl that is from previous
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

/// This is called when a decl is created or seen for the first time. It also
/// checks whether the decl is still in a mutation-sensitive state, meaning that
/// a later PTU can modify it. Because of this, we need to track its state. This
/// function provides a single place to decide whether a decl needs to be
/// committed or not. It is called from DeclStateCommitPolicy::process to
/// determine whether a new decl needs to be tracked.
///
/// This requires DeclShape S and Decl to determine which mutation types
/// apply to the shape and whether they need to be tracked.
/// In the future, if a new mutation site is found for a shape, it should be
/// added here along with the logic to determine whether the decl is still in an
/// open, modifiable state.
uint32_t PTUMutationActions::DeclNeedingTracking(DeclShape S, const Decl *D) {
  switch (S) {
  case DeclShape::Class: {
    const auto *RD = cast<CXXRecordDecl>(D);
    uint32_t Kinds = 0;
    // TypeForDecl is written twice for TagDecls. Keep tracking until both
    // writes are done.
    // Requires friend access to TypeDecl and was added for
    // memory restoration, so we can keep it commented out for now.
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
      // If SpecializationKind == TSK_ExplicitSpecialization, there is less
      // chance that the info will be modified later, so we are all set and
      // don't need to track it.
      // A specialization's Kind/point-of-instantiation is overwritten in place,
      // with no separate record of the previous value. The footprint chain is
      // the only place where the previous value is stored.
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
    // Exception-spec resolution changes the type before the listener notify,
    // so we need the old type saved beforehand.
    if (FPT && (FPT->getExceptionSpecType() == EST_Unevaluated ||
                FPT->getExceptionSpecType() == EST_Uninstantiated ||
                FPT->getExceptionSpecType() == EST_DependentNoexcept ||
                FPT->getExceptionSpecType() == EST_Unparsed))
      Kinds |= uint32_t(MutationType::ExceptionSpec);
    if (FD->getReturnType()->isUndeducedType())
      // Deduced return type is changed before
      // the listener notify, so save it beforehand.
      Kinds |= uint32_t(MutationType::DeducedReturnType);
    if (FD->getTemplateSpecializationInfo() &&
        FD->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
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
    // hidden kind that we have to check here.
    if (const EvaluatedStmt *Eval = VD->getEvaluatedStmt();
        !Eval || !Eval->WasEvaluated)
      Kinds |= uint32_t(MutationType::EvaluatedValue);

    return Kinds;
  }
  case DeclShape::Enum: {
    const auto *ED = cast<EnumDecl>(D);
    uint32_t Kinds = 0;
    // TypeForDecl: EnumDecl is also a TagDecl, so the same two-write pattern
    // and "still open" condition as in the Class case above apply here. See
    // its comment for details.
    // if (const Type *T = DeclStateReverter::getRawTypeForDecl(ED);
    //     !T || T->isCanonicalUnqualified())
    //   Kinds |= uint32_t(MutationType::TypeForDecl);
    //
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
    if (!DeclStateReverter::isCommonPtrValid(RT) &&
        !HiddenMutationTracker.isTrackedFor(
            RT, uint32_t(MutationType::TemplateCommon))) {
      // No listener reports these Common changes, so use hidden tracking.
      Kinds |= uint32_t(MutationType::TemplateCommon);
    } else if (const auto *CTD = dyn_cast<ClassTemplateDecl>(RT)) {
      if (!DeclStateReverter::isTemplateCanonInjectedTSTValid(CTD))
        Kinds |= uint32_t(MutationType::CanonInjectedTST);
    }
    return Kinds;
  }
  case DeclShape::Typedef: {
    // TypedefDecl/TypeAliasDecl only -- ASTContext::getTypedefType() has its
    // own guard (if (Decl->TypeForDecl) // return), so this is
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

/// The single place to check whether a kind has actually changed.
///
/// This handles both listener-less kinds and listener-unconfirmed mutations
/// (currently SpecInfo and MemberSpecInfo). Mutations can be detected by
/// both SweepTracker and the listener re-check. SweepTracker tracks hidden
/// and listener-unconfirmed mutations in case the listener misses them. If the
/// listener catches a mutation, it is removed from SweepTracker; see
/// PTUStateInfo::verifyMutations. both use this functionality to verify
/// mutation kind.
///
/// This check uses the decl shape and mutation kind to determine whether the
/// decl has changed from its previous state.
///
/// Similarly, if a new hidden or listener-unconfirmed mutation is introduced
/// in DeclNeedingTracking, and we need to determine whether it changed the
/// previous state, the corresponding logic should be added here too
/// when the kind is categorized as a hidden mutation from the listener.
uint32_t PTUMutationActions::confirmMutation(const Decl *D, DeclShape S,
                                             uint32_t FlaggedKinds) {
  uint32_t Verified = 0;
  switch (S) {
  case DeclShape::Class: {
    const auto *RD = cast<CXXRecordDecl>(D);
    if (FlaggedKinds & uint32_t(MutationType::TypeForDecl)) {
      // No listener at all for this mutation. why it is commented out:
      // Requires friend access to TypeDecl and was added for
      // memory restoration, so we can keep it commented out for now.
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
    break;
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
    // Use the canonical key. commitTemplate() tracks and untrack CommonBase
    // modifications, and CommonPtr is shared across the redeclaration chain.
    // Any modification to CommonPtr is reflected in the canonical declaration.
    const auto *RT = cast<RedeclarableTemplateDecl>(D)->getCanonicalDecl();
    if (FlaggedKinds & uint32_t(MutationType::TemplateCommon) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            RT, uint32_t(MutationType::TemplateCommon))) {
      // No chain to compare against -- write-once.
      if (DeclStateReverter::isCommonPtrValid(RT))
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

/// A single function called during commit and restore (when the PTU is
/// uncommitted).
///
/// This is where we check for any hidden mutations previously registered with
/// SweepTracker. If a mutation is confirmed, we record it in the
/// PTUStateInfo mutation set via OnConfirmed.
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

/// These kinds have listeners, but listeners alone cannot reliably detect
/// real changes. Compare the current value with the last known value, as we
/// do for kinds without listeners.
///
/// These kinds are also checked by the sweep to catch changes that listeners
/// miss or report incorrectly.
///
/// Currently, SpecInfo/MemberSpecInfo are tracked from the listener side.
/// This applies when the listener reports that a mutation may have happened
/// but it has not been confirmed yet.
///
/// Not all hidden mutations are tracked here; they are tracked in
/// SweepTracker. This logic applies only to listener-unconfirmed mutations.
static constexpr uint32_t KindsNeedingVerification =
    uint32_t(MutationType::SpecInfo) | uint32_t(MutationType::MemberSpecInfo);

/// Start tracking D for mutation kinds that need verification.
/// Called when a new mutation is committed or an undone mutation is restored.
/// we apply this for listener unconfirmed cases.
void PTUMutationActions::trackForSweep(const Decl *D, uint32_t Kinds) {
  if (uint32_t Flag = Kinds & KindsNeedingVerification)
    HiddenMutationTracker.track(D, Flag);
}

/// Stop tracking Decl D when it no longer needs verification.
/// Re-check DeclNeedingTracking to see if a later PTU can still update D.
void PTUMutationActions::untrackIfClosed(const Decl *D, DeclShape S,
                                         uint32_t Confirmed) {
  uint32_t Flag = Confirmed & KindsNeedingVerification;
  if (!Flag || !HiddenMutationTracker.isTrackedFor(D, Flag))
    return;
  uint32_t StillOpen = DeclNeedingTracking(S, D) & Flag;
  uint32_t Closed = Flag & ~StillOpen;
  if (Closed)
    HiddenMutationTracker.untrack(D, Closed);
}

// Shared sweep-tracking logic(track/untrack) for
// commitClass/commitFunction/commitVar/commitEnum.
void PTUMutationActions::syncSweepTracking(const Decl *D, DeclShape S,
                                           uint32_t Kinds, bool IsNew) {
  if (IsNew)
    trackForSweep(D, Kinds);
  else
    untrackIfClosed(D, S, Kinds);
}

/// Verify recorded mutations and remove any false mutation ones. If none of a
/// decl’s mutation kinds are confirmed, remove the record entirely so
/// restore() does not pop chain entries that were never written.
void PTUStateInfo::verifyMutations(PTUMutationActions &Actions) {
  Mutations.remove_if([&](auto &Entry) -> bool {
    const Decl *D = Entry.first;
    MutationRecord &Rec = Entry.second;
    uint32_t Flagged = Rec.MutationType & KindsNeedingVerification;
    if (!Flagged)
      return false;
    uint32_t Confirmed = Actions.confirmMutation(D, Rec.S, Flagged);
    /// untrack kind from SweepTracker if confirmed.
    Actions.untrackIfClosed(D, Rec.S, Confirmed);
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

    // Handle cases where baseline info needs to be created, as well as cases
    // where a declaration has already completed all required mutations. For
    // example, member specs may already be fully populated, so we should avoid
    // creating spec info blindly.
    uint32_t Kinds = Action.DeclNeedingTracking(S, D);
    if (!Kinds)
      return;

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

    /// Use the canonical decl because redecl unlinking should happen once per
    /// redecl chain, not once per individual decl. A PTU can contain multiple
    /// redecls, so we perform the unlinking only once.
    bool DetachFromRedecl = !RepairedChains.count(Canon) &&
                            D->getPreviousDecl() &&
                            !Tracker.isFromThisPTU(Canon, ID);

    if (needsDefDataDetach(D))
      Reverter.detachDefData(D);

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
        !Tracker.isFromThisPTU(LexicalDC, ID) &&
        !RepairedLexicalContexts.count(LexicalDC)) {
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

/// This are the places to commit or update the incremental state of a decl
/// based on its DeclShape.
///
/// The idea is to narrow each decl shape down to a small set of known
/// mutation sites that require their state to be registered with
/// IncrementalStateTracker. The state is committed when the decl is newly
/// created by this PTU, and updated when an existing decl is mutated.
///
/// Currently, known mutation sites include commitClass, commitFunction,
/// commitEnum, commitTemplate, etc. This follows a DeclShape ->
/// mutation type relationship.
///
/// When adding a new mutation site that requires committing or updating
/// state, it should be added based on the corresponding DeclShape ->
/// mutation type relationship.

void PTUMutationActions::commitMembers(PTUID ID, const DeclContext *Members) {
  DeclStateCommitPolicy Policy(ID, *this);
  walkDecls(Members, Policy);
}

void PTUMutationActions::commitClass(PTUID ID, const CXXRecordDecl *RD,
                                     MutationRecord &Rec, bool IsNew) {

  using ClassMutation = MutationType;

  if (Rec.has(ClassMutation::TypeForDecl)) {
    // Require friend grant to TypeDecl.
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
      /// Member declarations are instantiated eagerly with the class
      /// definition, egardless of TSK; only their definitions are deferred.
      ///
      /// Walk them here so each member’s MemberSpecializationInfo is tracked as
      /// soon as it is created.
      if (IsNew && Spec->getSpecializationKind() == TSK_ImplicitInstantiation)
        commitMembers(ID, cast<DeclContext>(Spec));
      // FIXME: Remove syncSweepTracking from all places
      // (SpecInfo/MemberSpecInfo) once the missing notifier
      // (ASTMutationListener) is added for (SpecInfo/MemberSpecInfo).
      syncSweepTracking(Spec, DeclShape::Class,
                        uint32_t(MutationType::SpecInfo), IsNew);
    }
  }

  if (Rec.has(ClassMutation::MemberSpecInfo)) {
    if (RD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          RD, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *RD));
      syncSweepTracking(RD, DeclShape::Class,
                        uint32_t(MutationType::MemberSpecInfo), IsNew);
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
    syncSweepTracking(FD, DeclShape::Function, uint32_t(MutationType::SpecInfo),
                      IsNew);
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    if (FD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          FD, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *FD));
      syncSweepTracking(FD, DeclShape::Function,
                        uint32_t(MutationType::MemberSpecInfo), IsNew);
    }
  }
}

void PTUMutationActions::commitVar(PTUID ID, const VarDecl *VD,
                                   MutationRecord &Rec, bool IsNew) {

  // EvaluatedStmt / APValue: set on the first constant evaluation of the
  // variable, which can happen in a later PTU.
  if (Rec.has(MutationType::EvaluatedValue)) {
    // No footprint: this is a write-once field with a fixed null
    // initial state, so there is no baseline to track.
    if (IsNew) {
      // Track the VarDecl so a later sweep can detect when WasEvaluated flips.
      // There is no listener for this mutation.
      HiddenMutationTracker.track(VD, uint32_t(MutationType::EvaluatedValue));
    } else {
      // This is the confirmation from SweepTracker::sweep(). Since this is a
      // write-once field, it is now settled and no longer needs tracking.
      // restoreVar() will re-track it if a rollback reverts the mutation.
      HiddenMutationTracker.untrack(VD, uint32_t(MutationType::EvaluatedValue));
    }
  }

  //   if (!IsNew && Rec.has(MutationType::DefinitionInstantiate)) {
  //   }

  if (Rec.has(MutationType::SpecInfo)) {
    const auto *Spec = cast<VarTemplateSpecializationDecl>(VD);
    Tracker.commitFootprint(
        Spec, MutationType::SpecInfo, ID,
        DeclStateReverter::createFootprint<VarSpecializationFootprint>(*Spec));
    syncSweepTracking(Spec, DeclShape::Var, uint32_t(MutationType::SpecInfo),
                      IsNew);
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    if (VD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          VD, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *VD));
      syncSweepTracking(VD, DeclShape::Var,
                        uint32_t(MutationType::MemberSpecInfo), IsNew);
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
    if (ED->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          ED, MutationType::MemberSpecInfo, ID,
          DeclStateReverter::createFootprint<MemberSpecializationFootprint>(
              *ED));
      syncSweepTracking(ED, DeclShape::Enum,
                        uint32_t(MutationType::MemberSpecInfo), IsNew);
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
      HiddenMutationTracker.untrack(RTCanon,
                                    uint32_t(MutationType::TemplateCommon));
  }

  if (Rec.has(MutationType::CanonInjectedTST)) {
    if (IsNew)
      HiddenMutationTracker.track(RTCanon,
                                  uint32_t(MutationType::CanonInjectedTST));
    else
      HiddenMutationTracker.untrack(RTCanon,
                                    uint32_t(MutationType::CanonInjectedTST));
  }
}

void PTUMutationActions::commitTypedef(PTUID ID, const TypedefNameDecl *TD,
                                       MutationRecord &Rec, bool IsNew) {
  if (Rec.has(MutationType::TypeForDecl)) {
    if (IsNew)
      HiddenMutationTracker.track(TD, uint32_t(MutationType::TypeForDecl));
    else
      HiddenMutationTracker.untrack(TD, uint32_t(MutationType::TypeForDecl));
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

  // 1. Verify any unconfirmed mutation (see : KindNeedingVerification) reported
  // by the listener, whether true or false.
  Cur.verifyMutations(*this);
  // 2. Catch any mutation-sensitive state of Decl from previous PTUs that
  //    ASTMutationListener cannot catch if it was registered during a
  //    previous PTU commit.
  Tracker.getHiddenMutationTracker().sweep(
      *this, [&](const Decl *D, DeclShape S, uint32_t K) {
        Cur.noteMutated(D, S, K);
      });

  // 3. Update the tracking state of declarations from the previous PTU using
  //    Cur.Mutations reported by ASTMutationListener and HiddenMutationTracker.
  for (auto &[D, Rec] : Cur.Mutations)
    commitDecl(ID, D, Rec, /*IsNew=*/false);

  // 4. Cur.ImplicitDecls: Declarations created by this PTU. Register them for
  //    tracking if they have mutation-sensitive state that can be modified by
  //    later PTUs.
  //
  // DeclStateCommitPolicy: Decides what to do with declarations created by the
  //    current PTU, such as whether a declaration needs to be tracked or not.
  DeclStateCommitPolicy Policy(ID, *this);
  for (const Decl *D : Cur.ImplicitDecls)
    Policy.process(D);

  // Iterate over all declarations from this TU and apply DeclStateCommitPolicy.
  walkDecls(ThisTU, Policy);

  Cur.Commited = true;
}

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::restore //////////////////////
/// -----------------------------------------------------------------------

/// These are the places where we restore or undo/unlink the incremental state
/// of a decl based on its DeclShape.
///
/// The same DeclShape -> mutation type relationship used during commit is
/// followed here, but in the opposite direction. Restore handles unlinking
/// newly created decls and undoing the committed or updated state of existing
/// decls, restoring them to their previous state.
///
/// As with commit, when adding a new mutation site that requires restoring or
/// unlinking state, it should be added based on the corresponding
/// DeclShape -> mutation type relationship.

void PTUMutationActions::restoreClass(PTUID ID, const CXXRecordDecl *RD,
                                      MutationRecord &Rec) {

  if (Rec.has(MutationType::DefinitionInstantiate) ||
      Rec.has(MutationType::DefinitionData)) {
    if (Rec.has(MutationType::DefinitionInstantiate)) {
      DeclStateReverter::revertDefinitionArrival(
          const_cast<CXXRecordDecl *>(RD));
    } else if (const auto *Restored =
                   Tracker.getFootprints().mostRecent<DefinitionDataFootprint>(
                       RD, MutationType::DefinitionData)) {
      DeclStateReverter::restoreDefinitionDataFootprint(
          *Restored, *const_cast<CXXRecordDecl *>(RD));
    }
  }

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
    trackForSweep(Spec, uint32_t(MutationType::SpecInfo));
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
    trackForSweep(RD, uint32_t(MutationType::MemberSpecInfo));
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
    DeclStateReverter::revertDefinitionArrival(const_cast<FunctionDecl *>(FD));

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
    const_cast<FunctionDecl *>(FD)->setInstantiationIsPending(false);
    trackForSweep(FD, uint32_t(MutationType::SpecInfo));
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    assert(FD->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                FD, MutationType::MemberSpecInfo))
      DeclStateReverter::restoreFootprint<MemberSpecializationFootprint>(
          *Restored, *const_cast<FunctionDecl *>(FD));
    const_cast<FunctionDecl *>(FD)->setInstantiationIsPending(false);
    trackForSweep(FD, uint32_t(MutationType::MemberSpecInfo));
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
    DeclStateReverter::revertDefinitionArrival(const_cast<VarDecl *>(VD));
  }

  if (Rec.has(MutationType::SpecInfo)) {
    const auto *Spec = cast<VarTemplateSpecializationDecl>(VD);
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<VarSpecializationFootprint>(
                Spec, MutationType::SpecInfo))
      DeclStateReverter::restoreFootprint<VarSpecializationFootprint>(
          *Restored, *const_cast<VarTemplateSpecializationDecl *>(Spec));
    trackForSweep(VD, uint32_t(MutationType::SpecInfo));
  }

  if (Rec.has(MutationType::MemberSpecInfo)) {
    assert(VD->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.Footprints.mostRecent<MemberSpecializationFootprint>(
                VD, MutationType::MemberSpecInfo))
      DeclStateReverter::restoreFootprint<MemberSpecializationFootprint>(
          *Restored, *const_cast<VarDecl *>(VD));
    trackForSweep(VD, uint32_t(MutationType::MemberSpecInfo));
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

  if (Rec.has(MutationType::DefinitionInstantiate))
    DeclStateReverter::revertDefinitionArrival(const_cast<EnumDecl *>(ED));

  if (Rec.has(MutationType::MemberSpecInfo)) {
    assert(ED->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                ED, MutationType::MemberSpecInfo))
      DeclStateReverter::restoreFootprint<MemberSpecializationFootprint>(
          *Restored, *const_cast<EnumDecl *>(ED));

    trackForSweep(ED, uint32_t(MutationType::MemberSpecInfo));
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
  llvm::SmallVector<Sema::SpecialMemberCacheKey, 8> ToRemove;
  for (auto &Entry : SemaRef.SpecialMemberCache) {
    CXXMethodDecl *MD = Entry.second.getMethod();
    if (MD && (Tracker.isFromThisPTU(MD, ID) || Cur.ImplicitDecls.contains(MD)))
      ToRemove.push_back(Entry.first);
  }
  for (const auto &Key : ToRemove)
    SemaRef.SpecialMemberCache.erase(Key);
}

void PTUMutationActions::restore(TranslationUnitDecl *ThisTU) {
  PTUStateInfo &Cur = Tracker.current();
  assert((!Cur.Commited || ThisTU == Cur.ThisPTU) &&
         "MostRecentTU doesn't match the TU recorded at commit time");
  PTUID ID = Cur.ID;
  // 1. Check whether this PTU was committed. If the PTU failed, it was not
  //    committed and therefore did not go through the commit policy or register
  //    any declaration state, so there is nothing to undo from
  //    IncrementalStateTracker.
  //
  //    If the PTU was successfully committed, the commit policy must have
  //    registered the declaration state and updated the state of declarations
  //    from the previous PTU in IncrementalStateTracker. First undo all state
  //    recorded by IncrementalStateTracker, then perform the regular undo so
  //    that mutated declarations are restored to the PTU(N-1) state rather
  //    than the current PTU state.
  if (Cur.Commited) {
    // remove sweep-tracked decls owned by this PTU,
    // which would otherwise become stale references with no cleanup path.
    Tracker.getHiddenMutationTracker().untrackIf(
        [&](const Decl *D) { return Tracker.isFromThisPTU(D, ID); });
    Tracker.undoLastEntries();
  } else {
    // Same as the commit() step, but performed here because this PTU failed and
    // was never committed. Since the commit step was never executed, we need to
    // perform it here for an uncommitted PTU.
    Cur.verifyMutations(*this);
    Tracker.getHiddenMutationTracker().sweep(
        *this, [&](const Decl *D, DeclShape S, uint32_t K) {
          Cur.noteMutated(D, S, K);
        });
  }

  // 2. Same as the commit() step, but in reverse: commit() updates the
  //    IncrementalStateTracker with the mutated state of declarations from the
  //    previous state. Here, we restore each mutated declaration using its most
  //    recent state in IncrementalStateTracker (i.e., the PTU(N-1) state).
  for (auto &[D, Rec] : Cur.Mutations)
    restoreDecl(ID, D, Rec);

  DeclStateReverter Detacher(Tracker.getPTUSlabCheckpoints());
  // 3. Cur.ImplicitDecls: Declarations created by this PTU. Unlink them from
  //    the compiler's shared state.
  //
  //    This is the counterpart to the commit DeclStateCommitPolicy, but
  //    performs the opposite operation by unlinking the declarations.
  //
  // DeclStateUnlinkPolicy: Decides what to do with declarations created by the
  //    current PTU, such as whether a declaration needs to be unlinked from
  //    redeclarations, DeclContext lookup, etc.
  DeclStateUnlinkPolicy Policy(ID, Tracker, Detacher);
  for (const Decl *D : Cur.ImplicitDecls)
    Policy.process(D);

  walkDecls(ThisTU, Policy);

  // SpecialMemberCache stores the CXXMethodDecl* from a previous
  // LookupSpecialMember() call. If it was created by a PTU being rolled back,
  // the cache entry is stale, so remove it here.
  restoreSpecialMemberCache(ID);

  // Must be last: Cur/ID/ThisTU refer to data owned by this PTU, and
  // popCurrentPTU() destroys that entry. This runs for both committed and
  // rolled-back PTUs to reclaim the PTUStack and checkpoint entry.
  Tracker.popCurrentPTU();
}

/// -----------------------------------------------------------------------
////////////////////////// MutationListener ///////////////////////////////
/// -----------------------------------------------------------------------

/// This is the place for listener-related functionality. When Clang notifies
/// us about an AST mutation, we record it in PTUStateInfo.
///
/// We only capture mutations here. For example, if a decl from a previous PTU
/// is mutated, we record its Decl -> DeclShape -> mutation type relationship.
/// We do not process the mutation here. Processing is deferred until commit
/// if the PTU succeeds; if the PTU fails, the flow goes through restore
/// instead.
///
/// We also track implicit decls. During commit, we walk the current PTU and
/// can only commit decls that are visible from that TU. However, implicit decls
/// can be created and added to another PTU, so we need to track them as well
/// so they can be committed or unlinked during restore.
///
/// We do not cover every mutation case here yet. Any new case added should
/// follow the same rule: if a mutation affects a decl from a previous PTU,
/// record it in PTUStateInfo.
///
/// There are two separate restore paths:
/// 1. Restoring a mutated decl to its previous state.
/// 2. Unlinking a decl from shared state, such as redecl chains or DC lookup.
///
/// For example, see noteDefinitionInstantiated, where a decl is created by
/// Clang and added to another PTU even though it is not marked as implicit.
/// Such a decl needs to be tracked for unlinking rather than restoring its
/// previous state.

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

  if (Tracker.isFromThisPTU(TD, Cur.ID)) {
    return;
  }

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
    // Normally, walkDecls covers declarations created in this PTU.
    // But an out-of-line static data member is a separate VarDecl whose
    // lexical context is the class, so the namespace/TU walk won't find it.
    // Process it like other implicit declarations instead.
    const auto *VD = dyn_cast<VarDecl>(D);
    if (VD && VD->getPreviousDecl() &&
        !Tracker.isFromThisPTU(VD->getCanonicalDecl(), Cur.ID))
      Cur.ImplicitDecls.insert(D);
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