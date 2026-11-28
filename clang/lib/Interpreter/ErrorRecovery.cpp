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

#include "ErrorRecovery.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Sema/Sema.h"

#define DEBUG_TYPE "clang-repl"

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

/// ASTDeclUnmerger handles two cases based on whether a declaration should be
/// discarded or belongs to a previous PTU: it unlinks discarded declarations
/// from the compiler's shared state, or restores their previous state using
/// the MutationRecord information.
class ASTDeclUnmerger : public DeclVisitor<ASTDeclUnmerger> {
private:
  PTUID ID;
  IncrementalStateTracker &Tracker;
  PTUMutationActions &Actions;
  Sema &SemaRef;
  SemaCacheReverter &SemaCache;
  CodeGenerator *CodeGen;
  MutationRecord *Rec = nullptr;

  llvm::SmallPtrSet<const Decl *, 16> RepairedChains;
  llvm::SmallPtrSet<const DeclContext *, 8> RepairedLexicalContexts;

  bool isFromThisPTU(const void *Ptr) const {
    return Tracker.isFromThisPTU(Ptr, ID);
  }

  bool isDiscarded(const void *Ptr) const {
    return Tracker.isFromThisPTU(Ptr, ID);
  }

  bool hasRecord() const { return Rec != nullptr; }

  MutationRecord &Record() { return *Rec; }

  bool hasNoPreviousDecl(const Decl *D) {
    return D->getPreviousDecl() == nullptr;
  }

  template <typename DeclT> DeclT *findSurvivor(DeclT *D) const {
    for (Decl *It = D->getMostRecentDecl(); It; It = It->getPreviousDecl())
      if (!isFromThisPTU(It))
        return dyn_cast<DeclT>(It);
    return nullptr;
  }

  template <typename DeclT>
  void resetRedeclLink(Redeclarable<DeclT> *DBase, DeclT *Survivor) {
    auto *Latest = static_cast<DeclT *>(DBase);

    // Rebuild First -> ... -> Survivor -> ... -> Latest as
    // First -> ... -> Survivor.
    Latest->getFirstDecl()->RedeclLink.setLatest(Survivor);

    // The chain is circular: a withdrawn declaration still linked into it can
    // never walk back around to itself, so redecls() on one would not
    // terminate. Give each withdrawn declaration a chain of its own.
    for (DeclT *Dead = Latest; Dead != Survivor;) {
      DeclT *Next = Dead->getPreviousDecl();
      Dead->First = Dead;
      Dead->RedeclLink =
          Redeclarable<DeclT>::LatestDeclLink(Tracker.getASTContext());
      Dead = Next;
    }
  }

  template <typename DeclT> void unlinkRedeclChain(DeclT *D) {
    if (hasNoPreviousDecl(D))
      return;
    if (!RepairedChains.insert(D->getCanonicalDecl()).second)
      return;
    DeclT *Latest = D->getMostRecentDecl();
    if (DeclT *Survivor = findSurvivor(D))
      resetRedeclLink(Latest, Survivor);
  }

  /// Remove D from its DeclContext lookup map and restore the previous
  /// declaration if it was replaced.
  void detachFromDCLookup(NamedDecl *ND) {
    // Find any declaration from the previous PTU in the redecl chain that may
    // have been replaced by this declaration.
    NamedDecl *Survivor = findSurvivor(ND);
    DeclarationName Name = ND->getDeclName();

    DeclContext *DC = ND->getDeclContext();
    do {
      DeclContext *Primary = DC->getPrimaryContext();
      if (StoredDeclsMap *Map = Primary->getLookupPtr()) {
        auto Pos = Map->find(Name);
        if (Pos != Map->end()) {
          StoredDeclsList &List = Pos->second;
          List.remove(ND);
          if (Survivor) {
            List.addOrReplaceDecl(Survivor);
          }
          if (List.isNull())
            Map->erase(Pos);
        }
      }
    } while ((DC->isTransparentContext() || DC->isInlineNamespace()) &&
             (DC = DC->getParent()));
  }

  /// Remove entry from "C"'s lookup tables
  void removeFromLookups(NamedDecl *D) {
    if (D->getDeclName().isEmpty())
      return;

    if (D->getDeclName().isIdentifier() && D->getDeclName().getFETokenInfo() &&
        !D->getLangOpts().ObjC && !D->getLangOpts().CPlusPlus)
      SemaRef.IdResolver.RemoveDecl(D);

    ExternCContextDecl *ECCD = Tracker.getASTContext().getExternCContextDecl();
    if (StoredDeclsMap *Map = ECCD->getPrimaryContext()->getLookupPtr()) {
      auto It = Map->find(D->getDeclName());
      if (It != Map->end())
        It->second.remove(D);
    }
  }

  void detachDefData(CXXRecordDecl *RD) {
    for (auto *I : RD->redecls())
      cast<CXXRecordDecl>(I)->DefinitionData = nullptr;
  }

  void detachCommonBase(const RedeclarableTemplateDecl *RT) {
    for (const auto *R : RT->redecls())
      R->Common = nullptr;
  }

  void repairLexicalChain(DeclContext *DC) {
    Decl *Prev = nullptr;
    for (Decl *Cur = DC->FirstDecl; Cur; Cur = Cur->getNextDeclInContext()) {
      if (isFromThisPTU(Cur)) {
        if (Prev) {
          DC->LastDecl = Prev;
          Prev->NextInContextAndBits.setPointer(nullptr);
        } else {
          DC->FirstDecl = DC->LastDecl = nullptr; // nothing survives
        }
        return;
      }
      Prev = Cur;
    }
  }

  void forgetMangledNames(std::initializer_list<GlobalDecl> GDs) {
    if (!CodeGen)
      return;
    // for (GlobalDecl GD : GDs)
    //   CodeGen->forgetDecl(GD);
  }

  void forgetMangledNames(NamedDecl *ND) {
    forgetMangledNames({GlobalDecl(ND)});
  }

  void unlinkCommon(NamedDecl *ND) {
    removeFromLookups(ND);

    Decl *Canon = ND->getCanonicalDecl();
    if (RepairedChains.count(Canon))
      return;

    detachFromDCLookup(ND);
  }

  void VisitDeclContext(DeclContext *DC) {
    for (Decl *D : llvm::SmallVector<Decl *, 8>(DC->decls()))
      Visit(D);
    if (RepairedLexicalContexts.insert(DC).second)
      repairLexicalChain(DC);
  }

  void revertDefinitionArrival(VarDecl *VD) {
    VD->setInit(nullptr);
    VD->setInvalidDecl(false);
  }

  void revertDefinitionArrival(FunctionDecl *FD) {
    FD->setBody(nullptr);
    // Reset the body state so a later PTU sees this as not instantiated.
    FD->setWillHaveBody(false);

    // The reverted body may have caused a substitution failure to mark
    // the function invalid. Clear that state as well.
    FD->setInvalidDecl(false);
  }

  void revertDefinitionArrival(EnumDecl *ED) {
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
    ED->setBeingDefined(false);
  }

  void revertDefinitionArrival(CXXRecordDecl *RD) {
    // 1. DefinitionData pointer -- back to null.
    RD->DefinitionData = nullptr;

    // 2. DeclContext member list -- back to empty.
    RD->FirstDecl = nullptr;
    RD->LastDecl = nullptr;
    DeclContext *Primary = RD->getPrimaryContext();
    if (StoredDeclsMap *Map = Primary->getLookupPtr())
      Map->clear();

    // 3. TagDecl completion bits -- back to "never started". Flipped by
    // startDefinition()/completeDefinition().
    RD->setBeingDefined(false);
    RD->setCompleteDefinition(false);
  }

  // True if Common could still be created later -- i.e. nobody has called
  // getCommonPtr() anywhere in this template's redecl chain yet.
  static bool isCommonPtrValid(const RedeclarableTemplateDecl *RT) {
    return RT->Common != nullptr;
  }

  // True if CanonInjectedTST could still be cached later. Only valid to
  // call once Common itself is confirmed to exist.
  static bool isTemplateCanonInjectedTSTValid(const ClassTemplateDecl *CTD) {
    auto *Ptr = CTD->getCommonPtr();
    return !Ptr->CanonInjectedTST.isNull();
  }

public:
  friend class PTUMutationActions;

  ASTDeclUnmerger(PTUID ID, IncrementalStateTracker &Tracker,
                  PTUMutationActions &Action, Sema &SemaRef,
                  SemaCacheReverter &SemaCache, CodeGenerator *CodeGen)
      : ID(ID), Tracker(Tracker), Actions(Action), SemaRef(SemaRef),
        SemaCache(SemaCache), CodeGen(CodeGen) {}

  static DefinitionDataFootprint
  createDefinitionDataFootprint(const CXXRecordDecl &RD) {
    DefinitionDataFootprint FP;
    const auto &Live = RD.data();
#define FIELD(Name, Width, Merge) FP.Name = Live.Name;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
    return FP;
  }

  static void restoreDefinitionDataFootprint(const DefinitionDataFootprint &FP,
                                             CXXRecordDecl &RD) {
    auto &Live = RD.data();
#define FIELD(Name, Width, Merge) Live.Name = FP.Name;
#include "clang/AST/CXXRecordDeclDefinitionBits.def"
  }

  // Generic across every other footprint type exclusing DefinitionDataFootprint
  // -- SpecializationFootprint, VarSpecializationFootprint,
  // FunctionSpecializationFootprint, MemberSpecializationFootprint.
  template <typename FootprintT, typename OwnerT>
  static FootprintT createFootprint(const OwnerT &Owner) {
    FootprintT FP;
    FP.update(Owner);
    return FP;
  }
  template <typename FootprintT, typename OwnerT>
  static void restoreFootprint(const FootprintT &FP, OwnerT &Owner) {
    FP.restore(Owner);
  }

  void restoreDecl(Decl *D, MutationRecord &R) {
    Rec = &R;
    Visit(D);
    Rec = nullptr;
  }

  void VisitDecl(Decl *D) {
    if (!isDiscarded(D))
      return;

    if (auto *DC = dyn_cast<DeclContext>(D))
      VisitDeclContext(DC);
    if (auto *ND = dyn_cast<NamedDecl>(D)) {
      unlinkCommon(ND);
    }
  }

  void VisitTranslationUnitDecl(TranslationUnitDecl *D) { VisitDeclContext(D); }
  void VisitNamespaceDecl(NamespaceDecl *D);
  void VisitCXXRecordDecl(CXXRecordDecl *RD);
  void VisitFunctionDecl(FunctionDecl *FD);
  void VisitCXXConstructorDecl(CXXConstructorDecl *D);
  void VisitCXXDestructorDecl(CXXDestructorDecl *D);
  void VisitVarDecl(VarDecl *VD);
  void VisitEnumDecl(EnumDecl *ED);
  void VisitClassTemplateDecl(ClassTemplateDecl *CTD);
  void VisitFunctionTemplateDecl(FunctionTemplateDecl *FTD);
  void VisitVarTemplateDecl(VarTemplateDecl *VTD);
  // void VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *TD);
  void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
  void VisitTypedefNameDecl(TypedefNameDecl *D);
  void VisitUsingShadowDecl(UsingShadowDecl *D);
};

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
/// committed or not. It is called from commitDecl to determine whether a new
/// decl needs to be tracked.
///
/// This requires DeclShape S and Decl to determine which mutation kinds
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
    // if (const Type *T = ASTDeclUnmerger::getRawTypeForDecl(RD);
    //     !T || T->isCanonicalUnqualified())
    //   Kinds |= uint32_t(MutationKind::TypeForDecl);
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
        Kinds |= uint32_t(MutationKind::DefinitionData);
    }
    if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
      // If SpecializationKind == TSK_ExplicitSpecialization, there is less
      // chance that the info will be modified later, so we are all set and
      // don't need to track it.
      // A specialization's Kind/point-of-instantiation is overwritten in place,
      // with no separate record of the previous value. The footprint chain is
      // the only place where the previous value is stored.
      if (Spec->getSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationKind::SpecInfo);
    } else if (const auto *MSI = RD->getMemberSpecializationInfo()) {
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationKind::MemberSpecInfo);
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
      Kinds |= uint32_t(MutationKind::ExceptionSpec);
    if (FD->getReturnType()->isUndeducedType())
      // Deduced return type is changed before
      // the listener notify, so save it beforehand.
      Kinds |= uint32_t(MutationKind::DeducedReturnType);
    if (FD->getTemplateSpecializationInfo() &&
        FD->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
      Kinds |= uint32_t(MutationKind::SpecInfo);
    if (const auto *MSI = FD->getMemberSpecializationInfo()) {
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationKind::MemberSpecInfo);
    }

    return Kinds;
  }
  case DeclShape::Var: {
    const auto *VD = cast<VarDecl>(D);
    uint32_t Kinds = 0;
    if (const auto *Spec = dyn_cast<VarTemplateSpecializationDecl>(VD)) {
      if (Spec->getSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationKind::SpecInfo);
    } else if (const auto *MSI = VD->getMemberSpecializationInfo()) {
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationKind::MemberSpecInfo);
    }
    // EvaluatedStmt::WasEvaluated is set the first time this variable is
    // constant-evaluated, which can happen in any PTU long after the VarDecl
    // was created. There is no ASTMutationListener event for this
    // (VarDecl::evaluateValueImpl doesn’t notify anything), so this is a
    // hidden kind that we have to check here.
    if (const EvaluatedStmt *Eval = VD->getEvaluatedStmt();
        !Eval || !Eval->WasEvaluated)
      Kinds |= uint32_t(MutationKind::EvaluatedValue);

    return Kinds;
  }
  case DeclShape::Enum: {
    const auto *ED = cast<EnumDecl>(D);
    uint32_t Kinds = 0;
    // TypeForDecl: EnumDecl is also a TagDecl, so the same two-write pattern
    // and "still open" condition as in the Class case above apply here. See
    // its comment for details.
    // if (const Type *T = ASTDeclUnmerger::getRawTypeForDecl(ED);
    //     !T || T->isCanonicalUnqualified())
    //   Kinds |= uint32_t(MutationKind::TypeForDecl);
    //
    if (const auto *MSI = ED->getMemberSpecializationInfo()) {
      // Same terminal-state exclusion as the other MSI-backed cases above.
      if (MSI->getTemplateSpecializationKind() != TSK_ExplicitSpecialization)
        Kinds |= uint32_t(MutationKind::MemberSpecInfo);
    }
    return Kinds;
  }
  case DeclShape::Template: {
    uint32_t Kinds = 0;
    const RedeclarableTemplateDecl *RT =
        cast<RedeclarableTemplateDecl>(D->getCanonicalDecl());
    if (!ASTDeclUnmerger::isCommonPtrValid(RT) &&
        !HiddenMutationTracker.isTrackedFor(
            RT, uint32_t(MutationKind::TemplateCommon))) {
      // No listener reports these Common changes, so use hidden tracking.
      Kinds |= uint32_t(MutationKind::TemplateCommon);
    } else if (const auto *CTD = dyn_cast<ClassTemplateDecl>(RT)) {
      if (!ASTDeclUnmerger::isTemplateCanonInjectedTSTValid(CTD))
        Kinds |= uint32_t(MutationKind::CanonInjectedTST);
    }
    return Kinds;
  }
  case DeclShape::Typedef: {
    // TypedefDecl/TypeAliasDecl only -- ASTContext::getTypedefType() has its
    // own guard (if (Decl->TypeForDecl) // return), so this is
    // write-once: once non-null, TypeForDecl
    // never changes again for this decl.
    // const auto *TD = cast<TypedefNameDecl>(D);
    // if (!ASTDeclUnmerger::getRawTypeForDecl(TD))
    //   return uint32_t(MutationKind::TypeForDecl);
    return 0;
  }
  case DeclShape::None:
    return 0;
  }
  return 0;
}

template <typename T>
static bool differsFromLast(IncrementalStateTracker &Tracker, const Decl *Owner,
                            MutationKind K, const T &LiveValue) {
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
    if (FlaggedKinds & uint32_t(MutationKind::TypeForDecl)) {
      // No listener at all for this mutation. why it is commented out:
      // Requires friend access to TypeDecl and was added for
      // memory restoration, so we can keep it commented out for now.
      // const Type *Now = ASTDeclUnmerger::getRawTypeForDecl(RD);
      // if (differsFromLast(Tracker, RD, MutationKind::TypeForDecl, Now))
      //   Verified |= uint32_t(MutationKind::TypeForDecl);
    }
    if (FlaggedKinds & uint32_t(MutationKind::SpecInfo)) {
      if (const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD))
        if (differsFromLast(Tracker, Spec, MutationKind::SpecInfo,
                            Footprint<SpecializationFootprint>(*Spec)))
          Verified |= uint32_t(MutationKind::SpecInfo);
    } else if (FlaggedKinds & uint32_t(MutationKind::MemberSpecInfo)) {
      if (differsFromLast(Tracker, RD, MutationKind::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*RD)))
        Verified |= uint32_t(MutationKind::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Function: {
    const auto *FD = cast<FunctionDecl>(D);
    if (FlaggedKinds & uint32_t(MutationKind::SpecInfo)) {
      if (differsFromLast(Tracker, FD, MutationKind::SpecInfo,
                          Footprint<FunctionSpecializationFootprint>(*FD)))
        Verified |= uint32_t(MutationKind::SpecInfo);
    } else if (FlaggedKinds & uint32_t(MutationKind::MemberSpecInfo)) {
      if (differsFromLast(Tracker, FD, MutationKind::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*FD)))
        Verified |= uint32_t(MutationKind::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Var: {
    const auto *VD = cast<VarDecl>(D);
    if (FlaggedKinds & uint32_t(MutationKind::EvaluatedValue) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            VD, uint32_t(MutationKind::EvaluatedValue))) {
      // No chain to compare against -- write-once field.
      const EvaluatedStmt *Eval = VD->getEvaluatedStmt();
      if (Eval && Eval->WasEvaluated)
        Verified |= uint32_t(MutationKind::EvaluatedValue);
    }
    if (FlaggedKinds & uint32_t(MutationKind::SpecInfo)) {
      if (const auto *Spec = dyn_cast<VarTemplateSpecializationDecl>(VD))
        if (differsFromLast(Tracker, Spec, MutationKind::SpecInfo,
                            Footprint<VarSpecializationFootprint>(*Spec)))
          Verified |= uint32_t(MutationKind::SpecInfo);
    } else if (FlaggedKinds & uint32_t(MutationKind::MemberSpecInfo)) {
      if (differsFromLast(Tracker, VD, MutationKind::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*VD)))
        Verified |= uint32_t(MutationKind::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Enum: {
    const auto *ED = cast<EnumDecl>(D);
    if (FlaggedKinds & uint32_t(MutationKind::TypeForDecl)) {
      // const Type *Now = ASTDeclUnmerger::getRawTypeForDecl(ED);
      // if (differsFromLast(Tracker, ED, MutationKind::TypeForDecl, Now))
      //   Verified |= uint32_t(MutationKind::TypeForDecl);
    }
    if (FlaggedKinds & uint32_t(MutationKind::MemberSpecInfo)) {
      if (differsFromLast(Tracker, ED, MutationKind::MemberSpecInfo,
                          Footprint<MemberSpecializationFootprint>(*ED)))
        Verified |= uint32_t(MutationKind::MemberSpecInfo);
    }
    break;
  }
  case DeclShape::Template: {
    // Use the canonical key. commitTemplate() tracks and untrack CommonBase
    // modifications, and CommonPtr is shared across the redeclaration chain.
    // Any modification to CommonPtr is reflected in the canonical declaration.
    const auto *RT = cast<RedeclarableTemplateDecl>(D)->getCanonicalDecl();
    if (FlaggedKinds & uint32_t(MutationKind::TemplateCommon) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            RT, uint32_t(MutationKind::TemplateCommon))) {
      if (ASTDeclUnmerger::isCommonPtrValid(RT))
        Verified |= uint32_t(MutationKind::TemplateCommon);
    }
    if (FlaggedKinds & uint32_t(MutationKind::CanonInjectedTST) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            RT, uint32_t(MutationKind::CanonInjectedTST))) {
      if (const auto *CTD = dyn_cast<ClassTemplateDecl>(RT))
        if (ASTDeclUnmerger::isTemplateCanonInjectedTSTValid(CTD))
          Verified |= uint32_t(MutationKind::CanonInjectedTST);
    }
    break;
  }
  case DeclShape::Typedef: {
    const auto *TD = cast<TypedefNameDecl>(D);
    if (FlaggedKinds & uint32_t(MutationKind::TypeForDecl) &&
        Tracker.getHiddenMutationTracker().isTrackedFor(
            TD, uint32_t(MutationKind::TypeForDecl))) {
      // if (ASTDeclUnmerger::getRawTypeForDecl(TD))
      //   Verified |= uint32_t(MutationKind::TypeForDecl);
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
    uint32_t(MutationKind::SpecInfo) | uint32_t(MutationKind::MemberSpecInfo);

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
    uint32_t Flagged = Rec.MutationFlags & KindsNeedingVerification;
    if (!Flagged)
      return false;
    uint32_t Confirmed = Actions.confirmMutation(D, Rec.S, Flagged);
    /// untrack kind from SweepTracker if confirmed.
    Actions.untrackIfClosed(D, Rec.S, Confirmed);
    Rec.MutationFlags = (Rec.MutationFlags & ~Flagged) | Confirmed;
    return Rec.MutationFlags == 0;
  });
}

static bool isReachableFromThisTU(const Decl *D,
                                  const TranslationUnitDecl *ThisTU) {
  const DeclContext *DC = D->getLexicalDeclContext();
  for (; DC; DC = DC->getLexicalParent())
    if (DC == ThisTU)
      return DC->containsDecl(const_cast<Decl *>(D));
  return false;
}

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::commit ///////////////////////
/// -----------------------------------------------------------------------

/// These are the places where the incremental state of a declaration is
/// registered based on its DeclShape.
///
/// For a declaration created in the current PTU, its state is committed
/// to the tracker. For an existing declaration modified in the current
/// PTU, its previous state is recorded so it can be restored on undo.
///
/// The idea is to keep state tracking limited to a small set of known
/// mutation sites for each DeclShape, instead of tracking changes
/// everywhere in Clang.
///
/// Currently, this covers Class, Function, Enum, Template, etc. Each
/// DeclShape is associated with the mutation kinds that need to be tracked.
///
/// When adding a new mutation site, it should be handled according to
/// the corresponding DeclShape and mutation kind.

bool PTUMutationActions::shouldRecurseInto(const Decl *D) {
  if (isa<NamespaceDecl>(D))
    return true;
  if (const auto *RD = dyn_cast<CXXRecordDecl>(D))
    return RD->isCompleteDefinition();
  return false;
}

void PTUMutationActions::VisitDeclContext(const DeclContext *DC) {
  for (const Decl *D : DC->decls()) {
    if (shouldRecurseInto(D))
      VisitDeclContext(cast<DeclContext>(D));
    commitDecl(D, nullptr);
  }
}

void PTUMutationActions::VisitCXXRecordDecl(const CXXRecordDecl *RD) {
  MutationRecord &Rec = Record();
  if (Rec.has(MutationKind::TypeForDecl)) {
    // Require friend grant to TypeDecl.
    // for (const TagDecl *Redecl : RD->redecls()) {
    //   const auto *Last = Tracker.getFootprints().mostRecent<const Type *>(
    //       Redecl, MutationKind::TypeForDecl);
    //   const Type *LastKnown = Last ? *Last : nullptr;
    //   if (LastKnown != cast<TypeDecl>(Redecl)->getTypeForDecl())
    //     Tracker.commitFootprint(Redecl, MutationKind::TypeForDecl, ID,
    //                             LastKnown);
    // }
  }

  if (Rec.has(MutationKind::DefinitionInstantiate) ||
      Rec.has(MutationKind::DefinitionData)) {
    Tracker.commitFootprint(
        RD, MutationKind::DefinitionData, CurrentID(),
        ASTDeclUnmerger::createDefinitionDataFootprint(*RD));

    if (Rec.has(MutationKind::DefinitionInstantiate)) {
      if (RD->getMemberSpecializationInfo())
        VisitDeclContext(cast<DeclContext>(RD));
    }
  }

  if (Rec.has(MutationKind::SpecInfo)) {
    const auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD);
    if (Spec) {
      Tracker.commitFootprint(
          Spec, MutationKind::SpecInfo, CurrentID(),
          ASTDeclUnmerger::createFootprint<SpecializationFootprint>(*Spec));
      /// Member declarations are instantiated eagerly with the class
      /// definition, egardless of TSK; only their definitions are deferred.
      ///
      /// Walk them here so each member’s MemberSpecializationInfo is tracked as
      /// soon as it is created.
      bool IsNew = Tracker.isFromCurrentPTU(Spec);
      if (IsNew && Spec->getSpecializationKind() == TSK_ImplicitInstantiation)
        VisitDeclContext(cast<DeclContext>(Spec));
      // FIXME: Remove syncSweepTracking from all places
      // (SpecInfo/MemberSpecInfo) once the missing notifier
      // (ASTMutationListener) is added for (SpecInfo/MemberSpecInfo).
      syncSweepTracking(Spec, DeclShape::Class,
                        uint32_t(MutationKind::SpecInfo), IsNew);
    }
  }

  if (Rec.has(MutationKind::MemberSpecInfo)) {
    if (RD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          RD, MutationKind::MemberSpecInfo, CurrentID(),
          ASTDeclUnmerger::createFootprint<MemberSpecializationFootprint>(*RD));
      syncSweepTracking(RD, DeclShape::Class,
                        uint32_t(MutationKind::MemberSpecInfo),
                        Tracker.isFromCurrentPTU(RD));
    }
  }
}

void PTUMutationActions::VisitFunctionDecl(const FunctionDecl *FD) {
  MutationRecord &Rec = Record();
  if (Rec.has(MutationKind::ExceptionSpec) ||
      Rec.has(MutationKind::DeducedReturnType)) {
    for (const FunctionDecl *Redecl : FD->redecls()) {
      if (Tracker.isFromCurrentPTU(FD)) {
        Tracker.commitFootprint(Redecl, MutationKind::ExceptionSpec,
                                CurrentID(), Redecl->getType());
      } else {
        const auto *Last = Tracker.getFootprints().mostRecent<QualType>(
            Redecl, MutationKind::ExceptionSpec);
        QualType LastKnown = Last ? *Last : QualType();
        if (LastKnown != Redecl->getType())
          Tracker.commitFootprint(Redecl, MutationKind::ExceptionSpec,
                                  CurrentID(), LastKnown);
      }
    }
  }

  //   if (!IsNew && Rec.has(MutationKind::DefinitionInstantiate)) {
  //   }

  if (Rec.has(MutationKind::SpecInfo)) {
    assert(FD->getTemplateSpecializationKind() != TSK_Undeclared &&
           "SpecInfoChanged on a function with no specialization info");
    Tracker.commitFootprint(
        FD, MutationKind::SpecInfo, CurrentID(),
        ASTDeclUnmerger::createFootprint<FunctionSpecializationFootprint>(*FD));
    syncSweepTracking(FD, DeclShape::Function, uint32_t(MutationKind::SpecInfo),
                      Tracker.isFromCurrentPTU(FD));
  }

  if (Rec.has(MutationKind::MemberSpecInfo)) {
    if (FD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          FD, MutationKind::MemberSpecInfo, CurrentID(),
          ASTDeclUnmerger::createFootprint<MemberSpecializationFootprint>(*FD));
      syncSweepTracking(FD, DeclShape::Function,
                        uint32_t(MutationKind::MemberSpecInfo),
                        Tracker.isFromCurrentPTU(FD));
    }
  }
}

void PTUMutationActions::VisitVarDecl(const VarDecl *VD) {
  MutationRecord &Rec = Record();
  // EvaluatedStmt / APValue: set on the first constant evaluation of the
  // variable, which can happen in a later PTU.
  if (Rec.has(MutationKind::EvaluatedValue)) {
    // No footprint: this is a write-once field with a fixed null
    // initial state, so there is no baseline to track.
    if (Tracker.isFromCurrentPTU(VD)) {
      // Track the VarDecl so a later sweep can detect when WasEvaluated flips.
      // There is no listener for this mutation.
      HiddenMutationTracker.track(VD, uint32_t(MutationKind::EvaluatedValue));
    } else {
      // This is the confirmation from SweepTracker::sweep(). Since this is a
      // write-once field, it is now settled and no longer needs tracking.
      // restoreVar() will re-track it if a rollback reverts the mutation.
      HiddenMutationTracker.untrack(VD, uint32_t(MutationKind::EvaluatedValue));
    }
  }

  //   if (!IsNew && Rec.has(MutationKind::DefinitionInstantiate)) {
  //   }

  if (Rec.has(MutationKind::SpecInfo)) {
    const auto *Spec = cast<VarTemplateSpecializationDecl>(VD);
    Tracker.commitFootprint(
        Spec, MutationKind::SpecInfo, CurrentID(),
        ASTDeclUnmerger::createFootprint<VarSpecializationFootprint>(*Spec));
    syncSweepTracking(Spec, DeclShape::Var, uint32_t(MutationKind::SpecInfo),
                      Tracker.isFromCurrentPTU(Spec));
  }

  if (Rec.has(MutationKind::MemberSpecInfo)) {
    if (VD->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          VD, MutationKind::MemberSpecInfo, CurrentID(),
          ASTDeclUnmerger::createFootprint<MemberSpecializationFootprint>(*VD));
      syncSweepTracking(VD, DeclShape::Var,
                        uint32_t(MutationKind::MemberSpecInfo),
                        Tracker.isFromCurrentPTU(VD));
    }
  }
}

void PTUMutationActions::VisitEnumDecl(const EnumDecl *ED) {
  MutationRecord &Rec = Record();
  if (Rec.has(MutationKind::TypeForDecl)) {
    // for (const TagDecl *Redecl : ED->redecls()) {
    //   const Type *Now = Redecl->getTypeForDecl();
    //   const Type *LastKnown = nullptr;
    //   if (!Tracker.isFromCurrentPTU(ED)) {
    //     const auto *Last = Tracker.getFootprints().mostRecent<const Type *>(
    //         Redecl, MutationKind::TypeForDecl);
    //     LastKnown = Last ? *Last : nullptr;
    //   }
    //   if (Tracker.isFromCurrentPTU(ED) || (LastKnown != Now))
    //     Tracker.commitFootprint(Redecl, MutationKind::TypeForDecl, ID,
    //                             LastKnown);
    // }
  }

  if (Rec.has(MutationKind::MemberSpecInfo)) {
    if (ED->getMemberSpecializationInfo()) {
      Tracker.commitFootprint(
          ED, MutationKind::MemberSpecInfo, CurrentID(),
          ASTDeclUnmerger::createFootprint<MemberSpecializationFootprint>(*ED));
      syncSweepTracking(ED, DeclShape::Enum,
                        uint32_t(MutationKind::MemberSpecInfo),
                        Tracker.isFromCurrentPTU(ED));
    }
  }
}

void PTUMutationActions::VisitRedeclarableTemplateDecl(
    const RedeclarableTemplateDecl *RT) {
  MutationRecord &Rec = Record();
  const RedeclarableTemplateDecl *RTCanon = RT->getCanonicalDecl();
  if (Rec.has(MutationKind::TemplateCommon)) {
    if (Tracker.isFromCurrentPTU(RT))
      HiddenMutationTracker.track(RTCanon,
                                  uint32_t(MutationKind::TemplateCommon));
    else
      HiddenMutationTracker.untrack(RTCanon,
                                    uint32_t(MutationKind::TemplateCommon));
  }

  if (Rec.has(MutationKind::CanonInjectedTST)) {
    if (Tracker.isFromCurrentPTU(RT))
      HiddenMutationTracker.track(RTCanon,
                                  uint32_t(MutationKind::CanonInjectedTST));
    else
      HiddenMutationTracker.untrack(RTCanon,
                                    uint32_t(MutationKind::CanonInjectedTST));
  }
}

void PTUMutationActions::VisitTypedefNameDecl(const TypedefNameDecl *D) {
  MutationRecord &Rec = Record();
  if (Rec.has(MutationKind::TypeForDecl)) {
    if (Tracker.isFromCurrentPTU(D))
      HiddenMutationTracker.track(D, uint32_t(MutationKind::TypeForDecl));
    else
      HiddenMutationTracker.untrack(D, uint32_t(MutationKind::TypeForDecl));
  }
}

void PTUMutationActions::commitDecl(const Decl *D, MutationRecord *R) {
  if (!D->isDefinedOutsideFunctionOrMethod())
    return;

  MutationRecord NewRec;
  if (!R) {
    DeclShape S = classifyShape(D);
    if (S == DeclShape::None)
      return;

    // Handle cases where baseline info needs to be created, as well as cases
    // where a declaration has already completed all required mutations. For
    // example, member specs may already be fully populated, so we should avoid
    // creating spec info blindly.
    uint32_t Kinds = DeclNeedingTracking(S, D);
    if (!Kinds)
      return;

    NewRec.add(Kinds);
    R = &NewRec;
  }

  MutationRecord *OldRec = Rec;
  Rec = R;
  Visit(D);
  Rec = OldRec;
}

void PTUMutationActions::commit(TranslationUnitDecl *ThisTU) {
  PTUStateInfo &Cur = Tracker.current();
  assert(!Cur.Commited && "Current PTUState is already commited");
  Cur.ThisPTU = ThisTU;

  LLVM_DEBUG({ llvm::dbgs() << "Commit ThisTU\n"; });

  // 1. Verify any unconfirmed mutation (see : KindsNeedingVerification)
  // reported by the listener, whether true or false.
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
    commitDecl(D, &Rec);

  // 4. Cur.ImplicitDecls: Declarations created by this PTU. Register them for
  //    tracking if they have mutation-sensitive state that can be modified by
  //    later PTUs.
  //
  // commitDecl: Decides what to do with declarations created by the current
  // PTU, such as whether a declaration needs to be tracked or not.
  for (const Decl *D : Cur.ImplicitDecls) {
    if (isReachableFromThisTU(D, ThisTU))
      continue;
    if (shouldRecurseInto(D))
      VisitDeclContext(cast<DeclContext>(D));
    commitDecl(D, nullptr);
  }

  // Iterate over all declarations in this TU and register them whether they
  // need explicit tracking.
  VisitDeclContext(ThisTU);

  Cur.Commited = true;
}

/// -----------------------------------------------------------------------
//////////////////////// PTUMutationActions::restore //////////////////////
/// -----------------------------------------------------------------------

/// These are the places where we restore or undo/unlink the incremental state
/// of a decl based on its DeclShape.
///
/// The same DeclShape -> mutation kind relationship used during commit is
/// followed here. For declarations created in the current PTU, unlinks them
/// from the compiler shared state. For existing declarations that were
/// modified, restore brings them back to their previous state.
///
/// When adding a new mutation site that requires restoring or unlinking
/// state, it should be handled according to the corresponding DeclShape
/// and mutation kind.

void ASTDeclUnmerger::VisitNamespaceDecl(NamespaceDecl *D) {
  // we don't care about the decl which complete chain is part belong to
  // current PTU.
  if (!Tracker.isFromThisPTU(D->getCanonicalDecl(), ID))
    VisitDeclContext(D);
  unlinkCommon(D);
  unlinkRedeclChain(D);
}

void ASTDeclUnmerger::VisitCXXRecordDecl(CXXRecordDecl *RD) {
  if (isDiscarded(RD)) {
    VisitDeclContext(RD);
    unlinkCommon(RD);
    if (CXXRecordDecl *Def = RD->getDefinition();
        Def && RD->getPreviousDecl() && isFromThisPTU(Def))
      detachDefData(Def);
    unlinkRedeclChain<TagDecl>(RD);
    return;
  }

  if (!hasRecord())
    return;

  // else perform restoration of decl state.
  if (Record().has(MutationKind::DefinitionInstantiate) ||
      Record().has(MutationKind::DefinitionData)) {
    if (Record().has(MutationKind::DefinitionInstantiate)) {
      revertDefinitionArrival(RD);
    } else if (const auto *Restored =
                   Tracker.getFootprints().mostRecent<DefinitionDataFootprint>(
                       RD, MutationKind::DefinitionData)) {
      restoreDefinitionDataFootprint(*Restored, *RD);
    }
  }

  if (Record().has(MutationKind::TypeForDecl)) {
    // for (const TagDecl *Redecl : RD->redecls()) {
    //   const auto *Last = Tracker.getFootprints().mostRecent<const Type *>(
    //       Redecl, MutationKind::TypeForDecl);
    //   const Type *LastKnown = Last ? *Last : nullptr;
    //   TODO: need to handle?
    // }
  }

  if (Record().has(MutationKind::SpecInfo)) {
    // Only a specialization can carry this kind; a plain CXXRecordDecl
    // reaching here means a note*() wrapper passed the wrong kind.
    auto *Spec = dyn_cast<ClassTemplateSpecializationDecl>(RD);
    assert(Spec && "SpecializationAdded noted on a non-specialization");
    if (const SpecializationFootprint *Restored =
            Tracker.getFootprints().mostRecent<SpecializationFootprint>(
                Spec, MutationKind::SpecInfo))
      restoreFootprint<SpecializationFootprint>(*Restored, *Spec);
    Actions.trackForSweep(Spec, uint32_t(MutationKind::SpecInfo));
  }

  if (Record().has(MutationKind::MemberSpecInfo)) {
    // A nested member of a class template – its MSI kind or
    // point-of-instantiation changed.
    assert(RD->getMemberSpecializationInfo());
    if (const MemberSpecializationFootprint *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                RD, MutationKind::MemberSpecInfo))
      restoreFootprint<MemberSpecializationFootprint>(*Restored, *RD);
    Actions.trackForSweep(RD, uint32_t(MutationKind::MemberSpecInfo));
  }
}

void ASTDeclUnmerger::VisitFunctionDecl(FunctionDecl *FD) {
  // unlink Discarded decl.
  if (isDiscarded(FD)) {
    forgetMangledNames(FD);
    unlinkCommon(FD);
    unlinkRedeclChain(FD);
    return;
  }

  if (!hasRecord())
    return;

  // Restore decl state.

  // FunctionType is shared across the whole redeclaration chain.
  // Resolving a deferred noexcept or deducing an auto return updates the
  // type for every redecl, including ones owned by earlier PTUs, so each
  // redecl needs to be recorded rather than just FD’s own.
  if (Record().has(MutationKind::ExceptionSpec) ||
      Record().has(MutationKind::DeducedReturnType)) {
    for (FunctionDecl *Redecl : FD->redecls()) {
      const auto *Last = Tracker.getFootprints().mostRecent<QualType>(
          Redecl, MutationKind::ExceptionSpec);
      QualType LastKnown = Last ? *Last : QualType();
      if (LastKnown != Redecl->getType())
        Redecl->setType(LastKnown);
    }
  }

  if (Record().has(MutationKind::DefinitionInstantiate)) {
    if (CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(FD)) {
      forgetMangledNames({GlobalDecl(Ctor, Ctor_Complete),
                          GlobalDecl(Ctor, Ctor_Base),
                          GlobalDecl(Ctor, Ctor_Comdat)});
    } else if (CXXDestructorDecl *Dtor = dyn_cast<CXXDestructorDecl>(FD)) {
      forgetMangledNames(
          {GlobalDecl(Dtor, Dtor_Deleting), GlobalDecl(Dtor, Dtor_Complete),
           GlobalDecl(Dtor, Dtor_Base), GlobalDecl(Dtor, Dtor_Comdat)});
    }
    revertDefinitionArrival(FD);
    if (auto *MD = dyn_cast<CXXMethodDecl>(FD); MD && MD->isVirtual())
      SemaCache.restoreVTableUsed(MD->getParent());
  }

  if (Record().has(MutationKind::SpecInfo)) {
    // A function template specialization is a plain FunctionDecl carrying
    // FunctionTemplateSpecializationInfo.
    assert(FD->getTemplateSpecializationKind() != TSK_Undeclared &&
           "SpecInfoChanged on a function with no specialization info");
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<FunctionSpecializationFootprint>(
                FD, MutationKind::SpecInfo))
      restoreFootprint<FunctionSpecializationFootprint>(*Restored, *FD);
    FD->setInstantiationIsPending(false);
    Actions.trackForSweep(FD, uint32_t(MutationKind::SpecInfo));
  }

  if (Record().has(MutationKind::MemberSpecInfo)) {
    assert(FD->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                FD, MutationKind::MemberSpecInfo))
      restoreFootprint<MemberSpecializationFootprint>(*Restored, *FD);
    FD->setInstantiationIsPending(false);
    Actions.trackForSweep(FD, uint32_t(MutationKind::MemberSpecInfo));
  }
}

void ASTDeclUnmerger::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  // Unlinking discared Decl.
  if (isDiscarded(D)) {
    forgetMangledNames({GlobalDecl(D, Ctor_Complete), GlobalDecl(D, Ctor_Base),
                        GlobalDecl(D, Ctor_Comdat)});
    unlinkCommon(D);
    unlinkRedeclChain<FunctionDecl>(D);
    return;
  }

  // Restore decl.
  if (hasRecord())
    VisitFunctionDecl(D);
}

void ASTDeclUnmerger::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  // Unlinking discared Decl.
  if (isDiscarded(D)) {
    forgetMangledNames({GlobalDecl(D, Dtor_Deleting),
                        GlobalDecl(D, Dtor_Complete), GlobalDecl(D, Dtor_Base),
                        GlobalDecl(D, Dtor_Comdat)});
    unlinkCommon(D);
    unlinkRedeclChain<FunctionDecl>(D);
    return;
  }

  // Restore decl.
  if (hasRecord())
    VisitFunctionDecl(D);
}

void ASTDeclUnmerger::VisitVarDecl(VarDecl *VD) {
  // Unlinking discared Decl.
  if (isDiscarded(VD)) {
    if (!isa<ParmVarDecl>(VD) && !VD->getDeclContext()->isDependentContext())
      forgetMangledNames(VD);
    unlinkCommon(VD);
    unlinkRedeclChain(VD);
    return;
  }

  if (!hasRecord())
    return;

  // Restore decl.

  // EvaluatedStmt::WasEvaluated is write-once, so a recorded entry means this
  // PTU confirmed the evaluation.
  if (Record().has(MutationKind::EvaluatedValue)) {
    // VD cached evaluation came from this PTU.
    // Reset it and track it again so a later PTU can re-trigger evaluation.
    if (EvaluatedStmt *Eval = VD->getEvaluatedStmt()) {
      Eval->WasEvaluated = false;
      Eval->Evaluated = APValue();
    }
    Tracker.getHiddenMutationTracker().track(
        VD, uint32_t(MutationKind::EvaluatedValue));
  }

  if (Record().has(MutationKind::DefinitionInstantiate)) {
    revertDefinitionArrival(VD);
    forgetMangledNames(VD);
  }

  if (Record().has(MutationKind::SpecInfo)) {
    auto *Spec = cast<VarTemplateSpecializationDecl>(VD);
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<VarSpecializationFootprint>(
                Spec, MutationKind::SpecInfo))
      restoreFootprint<VarSpecializationFootprint>(*Restored, *Spec);
    Actions.trackForSweep(VD, uint32_t(MutationKind::SpecInfo));
  }

  if (Record().has(MutationKind::MemberSpecInfo)) {
    assert(VD->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                VD, MutationKind::MemberSpecInfo))
      restoreFootprint<MemberSpecializationFootprint>(*Restored, *VD);
    Actions.trackForSweep(VD, uint32_t(MutationKind::MemberSpecInfo));
  }
}

void ASTDeclUnmerger::VisitEnumDecl(EnumDecl *ED) {
  // Unlink Discarded Decl.
  if (isDiscarded(ED)) {
    VisitDeclContext(ED);
    unlinkCommon(ED);
    unlinkRedeclChain<TagDecl>(ED);
    return;
  }

  if (!hasRecord())
    return;

  // Restore Decl state.
  if (Record().has(MutationKind::TypeForDecl)) {
    // for (const TagDecl *Redecl : ED->redecls()) {
    //   const auto *Last = Tracker.getFootprints().mostRecent<const Type *>(
    //       Redecl, MutationKind::TypeForDecl);
    //   TODO: need to handle?
    // }
  }

  if (Record().has(MutationKind::DefinitionInstantiate))
    revertDefinitionArrival(ED);

  if (Record().has(MutationKind::MemberSpecInfo)) {
    assert(ED->getMemberSpecializationInfo());
    if (const auto *Restored =
            Tracker.getFootprints().mostRecent<MemberSpecializationFootprint>(
                ED, MutationKind::MemberSpecInfo))
      restoreFootprint<MemberSpecializationFootprint>(*Restored, *ED);

    Actions.trackForSweep(ED, uint32_t(MutationKind::MemberSpecInfo));
  }
}

void ASTDeclUnmerger::VisitClassTemplateDecl(ClassTemplateDecl *CT) {
  // Unlinking discared Decl.
  if (isDiscarded(CT)) {
    unlinkCommon(CT);
    bool HasSurvivorToReplace =
        !hasNoPreviousDecl(CT) && !isFromThisPTU(CT->getCanonicalDecl());
    if (HasSurvivorToReplace)
      unlinkRedeclChain(cast<RedeclarableTemplateDecl>(CT));

    NamedDecl *Templated = CT->getTemplatedDecl();
    if (auto *RD = dyn_cast<CXXRecordDecl>(Templated)) {
      if (CXXRecordDecl *Def = RD->getDefinition(); Def && isFromThisPTU(Def))
        detachDefData(RD);
      if (HasSurvivorToReplace)
        unlinkRedeclChain<TagDecl>(RD);
    }
  }

  if (!hasRecord())
    return;

  // Restore Decl.

  if (Record().has(MutationKind::TemplateCommon)) {
    // TODO: call ASTDeclUnmerger to detech all decl's common ptr if canon
    // decl of template ie being tracked;
    // ASTDeclUnmerger::resetTemplateCommonBase(
    //     *const_cast<RedeclarableTemplateDecl *>(TD));
    detachCommonBase(CT);
    // Common’s creation was done by this PTU.
    // Re-register both bits, since a later PTU may trigger either one again,
    // and CanonInjectedTST cannot exist until Common exists again first.
    //
    Tracker.getHiddenMutationTracker().track(
        CT->getCanonicalDecl(), uint32_t(MutationKind::TemplateCommon) |
                                    uint32_t(MutationKind::CanonInjectedTST));
  } else if (Record().has(MutationKind::CanonInjectedTST)) {
    if (auto *CTD = dyn_cast<ClassTemplateDecl>(CT)) {
      CTD->getCommonPtr()->CanonInjectedTST = CanQualType();
      Tracker.getHiddenMutationTracker().track(
          CT, uint32_t(MutationKind::CanonInjectedTST));
    }
  }

  if (Record().has(MutationKind::SpecializationAdded)) {
    auto &Specs = CT->getSpecializations();
    while (!Specs.empty()) {
      auto It = Specs.end();
      --It;
      if (!Tracker.isFromThisPTU(&*It, ID))
        break;
      Specs.pop_back();
    }
  }
}

void ASTDeclUnmerger::VisitFunctionTemplateDecl(FunctionTemplateDecl *FTD) {
  // Unlinking discared Decl.
  if (isDiscarded(FTD)) {
    unlinkCommon(FTD);
    bool HasSurvivorToReplace =
        !hasNoPreviousDecl(FTD) && !isFromThisPTU(FTD->getCanonicalDecl());
    if (HasSurvivorToReplace)
      unlinkRedeclChain(cast<RedeclarableTemplateDecl>(FTD));

    if (HasSurvivorToReplace) {
      if (auto *FD = dyn_cast<FunctionDecl>(FTD->getTemplatedDecl()))
        unlinkRedeclChain(FD);
    }
  }

  if (!hasRecord())
    return;

  // Restore Decl.

  if (Record().has(MutationKind::TemplateCommon)) {
    // detech all decl's common ptr if canon decl of template ie being tracked;
    detachCommonBase(FTD);
    // Common’s creation was done by this PTU.
    // Re-register both bits, since a later PTU may trigger either one again.
    //
    Tracker.getHiddenMutationTracker().track(
        FTD->getCanonicalDecl(), uint32_t(MutationKind::TemplateCommon));
  }

  if (Record().has(MutationKind::SpecializationAdded)) {
    auto &Specs = FTD->getSpecializations();
    while (!Specs.empty()) {
      auto It = Specs.end();
      --It;
      if (!Tracker.isFromThisPTU(It->getFunction(), ID))
        break;
      Specs.pop_back();
    }
  }
}

void ASTDeclUnmerger::VisitVarTemplateDecl(VarTemplateDecl *VTD) {
  // Unlinking discared Decl.
  if (isDiscarded(VTD)) {
    unlinkCommon(VTD);
    bool HasSurvivorToReplace =
        !hasNoPreviousDecl(VTD) && !isFromThisPTU(VTD->getCanonicalDecl());
    if (HasSurvivorToReplace)
      unlinkRedeclChain(cast<RedeclarableTemplateDecl>(VTD));

    NamedDecl *Templated = VTD->getTemplatedDecl();
    if (HasSurvivorToReplace) {
      if (auto *VD = dyn_cast<VarDecl>(Templated))
        unlinkRedeclChain(VD);
    }
  }

  if (!hasRecord())
    return;

  // Restore Decl.

  if (Record().has(MutationKind::TemplateCommon)) {
    // detech all decl's common ptr if canon decl of template ie being tracked;
    detachCommonBase(VTD);
    // Common’s creation was done by this PTU.
    // Re-register both bits, since a later PTU may trigger either one again.
    //
    Tracker.getHiddenMutationTracker().track(
        VTD->getCanonicalDecl(), uint32_t(MutationKind::TemplateCommon));
  }

  if (Record().has(MutationKind::SpecializationAdded)) {
    auto &Specs = VTD->getSpecializations();
    while (!Specs.empty()) {
      auto It = Specs.end();
      --It;
      if (!Tracker.isFromThisPTU(&*It, ID))
        break;
      Specs.pop_back();
    }
  }
}

void ASTDeclUnmerger::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  // Unlink Discarded Decl.
  if (isDiscarded(D)) {
    unlinkCommon(D);
    unlinkRedeclChain(D);
    return;
  }
}

void ASTDeclUnmerger::VisitTypedefNameDecl(TypedefNameDecl *D) {
  // Unlink Discarded Decl.
  if (isDiscarded(D)) {
    unlinkCommon(D);
    unlinkRedeclChain(D);
    return;
  }

  if (!hasRecord())
    return;

  if (Record().has(MutationKind::TypeForDecl)) {
    // ASTDeclUnmerger::resetTypeForDecl(TD);
    // HiddenMutationTracker.track(TD, uint32_t(MutationKind::TypeForDecl));
  }
}

void ASTDeclUnmerger::VisitUsingShadowDecl(UsingShadowDecl *D) {
  // Unlink Discarded Decl.
  if (isDiscarded(D)) {
    unlinkCommon(D);
    unlinkRedeclChain(D);
    return;
  }
}

void SemaCacheReverter::restoreSpecialMemberCache(PTUID ID) {
  PTUStateInfo &Cur = Tracker.current();
  if (!Cur.HadImplicitCXXMember)
    return;

  llvm::SmallVector<Sema::SpecialMemberCacheKey, 8> ToRemove;
  for (auto &Entry : SemaRef.SpecialMemberCache) {
    CXXMethodDecl *MD = Entry.second.getMethod();
    if (MD && (Tracker.isFromThisPTU(MD, ID) || Cur.ImplicitDecls.contains(MD)))
      ToRemove.push_back(Entry.first);
  }
  for (const auto &Key : ToRemove)
    SemaRef.SpecialMemberCache.erase(Key);
}

void SemaCacheReverter::restoreVTableUsed(CXXRecordDecl *RD) {
  SemaRef.VTablesUsed.erase(RD->getCanonicalDecl());
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

  ASTDeclUnmerger Umerger(ID, Tracker, *this, SemaRef, SemaCache, CodeGen);

  // 2. Same as the commit() step, but in reverse: commit() updates the
  //    IncrementalStateTracker with the mutated state of declarations from the
  //    previous state. Here, we restore each mutated declaration using its most
  //    recent state in IncrementalStateTracker (i.e., the PTU(N-1) state).
  for (auto &[D, Rec] : Cur.Mutations)
    Umerger.restoreDecl(const_cast<Decl *>(D), Rec);

  // 3. Cur.ImplicitDecls: Declarations created by this PTU. Unlink them from
  //    the compiler's shared state.
  //
  //    This is the counterpart to the commit, but
  //    performs the opposite operation by unlinking the declarations.
  //
  // ASTDeclUnmerger: Determines how to undo declarations mutated or created by
  // the current PTU, such as unlinking declarations from redeclaration chains
  // and DeclContext lookup structures, or restoring declaration state (e.g.,
  // CXXRecordDecl::DefinitionData).

  for (const Decl *D : Cur.ImplicitDecls) {
    if (isReachableFromThisTU(D, ThisTU))
      continue;
    Umerger.Visit(const_cast<Decl *>(D));
  }

  Umerger.Visit(ThisTU);

  // SpecialMemberCache stores the CXXMethodDecl* from a previous
  // LookupSpecialMember() call. If it was created by a PTU being rolled back,
  // the cache entry is stale, so remove it here.
  SemaCache.restoreSpecialMemberCache(ID);

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
/// is mutated, we record its Decl -> DeclShape -> mutation kind relationship.
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
  if (false)
    Cur.noteMutated(D, S, MutationKind::DefinitionInstantiate);

  /// Handle this for MemberSpec because the notifier does not guarantee whether
  /// it fires before or after the mutation. We cannot reliably compare the
  /// state here, so mutation detection is deferred to the verification layer.
  if (S == DeclShape::Enum) {
    if (const EnumDecl *ED = dyn_cast<EnumDecl>(D);
        ED && ED->getMemberSpecializationInfo())
      Cur.noteMutated(D, S, MutationKind::MemberSpecInfo);
  } else if (S == DeclShape::Class) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D);
        RD && RD->getMemberSpecializationInfo())
      Cur.noteMutated(D, S, MutationKind::MemberSpecInfo);
    if (isa<ClassTemplateSpecializationDecl>(D))
      Cur.noteMutated(D, S, MutationKind::SpecInfo);
  }
}

void PTUMutationRecorder::AddedVisibleDecl(const DeclContext *DC,
                                           const Decl *D) {
  PTUStateInfo &Cur = Tracker.current();
  if (D->isImplicit())
    Cur.ImplicitDecls.insert(D);

  if (Tracker.isFromThisPTU(DC, Cur.ID) || Cur.TouchedDCs.contains(DC))
    return;

  Cur.TouchedDCs.insert(DC);
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

  Cur.noteMutated(RD, DeclShape::Class, MutationKind::DefinitionData);
}

template <typename TemplateT, typename SpecT>
void PTUMutationRecorder::noteTemplateDeclMutation(const TemplateT *TD,
                                                   const SpecT *Spec,
                                                   DeclShape S) {
  PTUStateInfo &Cur = Tracker.current();
  Cur.ImplicitDecls.insert(Spec);

  if (Tracker.isFromThisPTU(TD, Cur.ID))
    return;

  Cur.noteMutated(TD, DeclShape::Template, MutationKind::SpecializationAdded);
}

void PTUMutationRecorder::AddedCXXTemplateSpecialization(
    const ClassTemplateDecl *TD, const ClassTemplateSpecializationDecl *D) {
  noteTemplateDeclMutation(TD, D, DeclShape::Class);
}

void PTUMutationRecorder::AddedCXXTemplateSpecialization(
    const VarTemplateDecl *TD, const VarTemplateSpecializationDecl *D) {
  noteTemplateDeclMutation(TD, D, DeclShape::Var);
}

void PTUMutationRecorder::AddedCXXTemplateSpecialization(
    const FunctionTemplateDecl *TD, const FunctionDecl *D) {
  noteTemplateDeclMutation(TD, D, DeclShape::Function);
}

void PTUMutationRecorder::noteExceptionSpecMutation(const FunctionDecl *FD,
                                                    MutationKind K) {
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
  noteExceptionSpecMutation(FD, MutationKind::DeducedReturnType);
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

  Cur.noteMutated(D, classifyShape(D), MutationKind::DefinitionInstantiate);
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
  MutationKind K = MutationKind::None;

  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    S = DeclShape::Function;
    if (FD->getMemberSpecializationInfo())
      K = MutationKind::MemberSpecInfo;
    else if (FD->getTemplateSpecializationKind() != TSK_Undeclared)
      K = MutationKind::SpecInfo;
    else
      return; // neither -- nothing this notifier can be about
  } else if (const auto *VD = dyn_cast<VarDecl>(D)) {
    S = DeclShape::Var;
    if (VD->getMemberSpecializationInfo())
      K = MutationKind::MemberSpecInfo;
    else if (isa<VarTemplateSpecializationDecl>(VD))
      K = MutationKind::SpecInfo;
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