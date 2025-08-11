//===- DeclTemplate.cpp - Template Declaration AST Node Implementation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the C++ related Decl classes for templates.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/ODRHash.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <optional>
#include <utility>

using namespace clang;

//===----------------------------------------------------------------------===//
// TemplateParameterList Implementation
//===----------------------------------------------------------------------===//

template <class TemplateParam>
static bool
DefaultTemplateArgumentContainsUnexpandedPack(const TemplateParam &P) {
  return P.hasDefaultArgument() &&
         P.getDefaultArgument().getArgument().containsUnexpandedParameterPack();
}

TemplateParameterList::TemplateParameterList(const ASTContext &C,
                                             SourceLocation TemplateLoc,
                                             SourceLocation LAngleLoc,
                                             ArrayRef<NamedDecl *> Params,
                                             SourceLocation RAngleLoc,
                                             Expr *RequiresClause)
    : TemplateLoc(TemplateLoc), LAngleLoc(LAngleLoc), RAngleLoc(RAngleLoc),
      NumParams(Params.size()), ContainsUnexpandedParameterPack(false),
      HasRequiresClause(RequiresClause != nullptr),
      HasConstrainedParameters(false) {
  for (unsigned Idx = 0; Idx < NumParams; ++Idx) {
    NamedDecl *P = Params[Idx];
    begin()[Idx] = P;

    bool IsPack = P->isTemplateParameterPack();
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(P)) {
      if (!IsPack && (NTTP->getType()->containsUnexpandedParameterPack() ||
                      DefaultTemplateArgumentContainsUnexpandedPack(*NTTP)))
        ContainsUnexpandedParameterPack = true;
      if (NTTP->hasPlaceholderTypeConstraint())
        HasConstrainedParameters = true;
    } else if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(P)) {
      if (!IsPack &&
          (TTP->getTemplateParameters()->containsUnexpandedParameterPack() ||
           DefaultTemplateArgumentContainsUnexpandedPack(*TTP))) {
        ContainsUnexpandedParameterPack = true;
      }
    } else if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(P)) {
      if (!IsPack && DefaultTemplateArgumentContainsUnexpandedPack(*TTP)) {
        ContainsUnexpandedParameterPack = true;
      } else if (const TypeConstraint *TC = TTP->getTypeConstraint();
                 TC && TC->getImmediatelyDeclaredConstraint()
                           ->containsUnexpandedParameterPack()) {
        ContainsUnexpandedParameterPack = true;
      }
      if (TTP->hasTypeConstraint())
        HasConstrainedParameters = true;
    } else {
      llvm_unreachable("unexpected template parameter type");
    }
  }

  if (HasRequiresClause) {
    if (RequiresClause->containsUnexpandedParameterPack())
      ContainsUnexpandedParameterPack = true;
    *getTrailingObjects<Expr *>() = RequiresClause;
  }
}

bool TemplateParameterList::containsUnexpandedParameterPack() const {
  if (ContainsUnexpandedParameterPack)
    return true;
  if (!HasConstrainedParameters)
    return false;

  // An implicit constrained parameter might have had a use of an unexpanded
  // pack added to it after the template parameter list was created. All
  // implicit parameters are at the end of the parameter list.
  for (const NamedDecl *Param : llvm::reverse(asArray())) {
    if (!Param->isImplicit())
      break;

    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
      const auto *TC = TTP->getTypeConstraint();
      if (TC && TC->getImmediatelyDeclaredConstraint()
                    ->containsUnexpandedParameterPack())
        return true;
    }
  }

  return false;
}

TemplateParameterList *
TemplateParameterList::Create(const ASTContext &C, SourceLocation TemplateLoc,
                              SourceLocation LAngleLoc,
                              ArrayRef<NamedDecl *> Params,
                              SourceLocation RAngleLoc, Expr *RequiresClause) {
  void *Mem = C.Allocate(totalSizeToAlloc<NamedDecl *, Expr *>(
                             Params.size(), RequiresClause ? 1u : 0u),
                         alignof(TemplateParameterList));
  return new (Mem) TemplateParameterList(C, TemplateLoc, LAngleLoc, Params,
                                         RAngleLoc, RequiresClause);
}

void TemplateParameterList::Profile(llvm::FoldingSetNodeID &ID,
                                    const ASTContext &C) const {
  const Expr *RC = getRequiresClause();
  ID.AddBoolean(RC != nullptr);
  if (RC)
    RC->Profile(ID, C, /*Canonical=*/true);
  ID.AddInteger(size());
  for (NamedDecl *D : *this) {
    if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
      ID.AddInteger(0);
      ID.AddBoolean(NTTP->isParameterPack());
      NTTP->getType().getCanonicalType().Profile(ID);
      ID.AddBoolean(NTTP->hasPlaceholderTypeConstraint());
      if (const Expr *E = NTTP->getPlaceholderTypeConstraint())
        E->Profile(ID, C, /*Canonical=*/true);
      continue;
    }
    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(D)) {
      ID.AddInteger(1);
      ID.AddBoolean(TTP->isParameterPack());
      ID.AddBoolean(TTP->hasTypeConstraint());
      if (const TypeConstraint *TC = TTP->getTypeConstraint())
        TC->getImmediatelyDeclaredConstraint()->Profile(ID, C,
                                                        /*Canonical=*/true);
      continue;
    }
    const auto *TTP = cast<TemplateTemplateParmDecl>(D);
    ID.AddInteger(2);
    ID.AddInteger(TTP->templateParameterKind());
    ID.AddBoolean(TTP->isParameterPack());
    TTP->getTemplateParameters()->Profile(ID, C);
  }
}

unsigned TemplateParameterList::getMinRequiredArguments() const {
  unsigned NumRequiredArgs = 0;
  for (const NamedDecl *P : asArray()) {
    if (P->isTemplateParameterPack()) {
      if (UnsignedOrNone Expansions = getExpandedPackSize(P)) {
        NumRequiredArgs += *Expansions;
        continue;
      }
      break;
    }

    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(P)) {
      if (TTP->hasDefaultArgument())
        break;
    } else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(P)) {
      if (NTTP->hasDefaultArgument())
        break;
    } else if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(P);
               TTP && TTP->hasDefaultArgument())
      break;

    ++NumRequiredArgs;
  }

  return NumRequiredArgs;
}

unsigned TemplateParameterList::getDepth() const {
  if (size() == 0)
    return 0;

  const NamedDecl *FirstParm = getParam(0);
  if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(FirstParm))
    return TTP->getDepth();
  else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(FirstParm))
    return NTTP->getDepth();
  else
    return cast<TemplateTemplateParmDecl>(FirstParm)->getDepth();
}

static bool AdoptTemplateParameterList(TemplateParameterList *Params,
                                       DeclContext *Owner) {
  bool Invalid = false;
  for (NamedDecl *P : *Params) {
    P->setDeclContext(Owner);

    if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(P))
      if (AdoptTemplateParameterList(TTP->getTemplateParameters(), Owner))
        Invalid = true;

    if (P->isInvalidDecl())
      Invalid = true;
  }
  return Invalid;
}

void TemplateParameterList::getAssociatedConstraints(
    llvm::SmallVectorImpl<AssociatedConstraint> &ACs) const {
  if (HasConstrainedParameters)
    for (const NamedDecl *Param : *this) {
      if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
        if (const auto *TC = TTP->getTypeConstraint())
          ACs.emplace_back(TC->getImmediatelyDeclaredConstraint(),
                           TC->getArgPackSubstIndex());
      } else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        if (const Expr *E = NTTP->getPlaceholderTypeConstraint())
          ACs.emplace_back(E);
      }
    }
  if (HasRequiresClause)
    ACs.emplace_back(getRequiresClause());
}

bool TemplateParameterList::hasAssociatedConstraints() const {
  return HasRequiresClause || HasConstrainedParameters;
}

ArrayRef<TemplateArgument>
TemplateParameterList::getInjectedTemplateArgs(const ASTContext &Context) {
  if (!InjectedArgs) {
    InjectedArgs = new (Context) TemplateArgument[size()];
    llvm::transform(*this, InjectedArgs, [&](NamedDecl *ND) {
      return Context.getInjectedTemplateArg(ND);
    });
  }
  return {InjectedArgs, NumParams};
}

bool TemplateParameterList::shouldIncludeTypeForArgument(
    const PrintingPolicy &Policy, const TemplateParameterList *TPL,
    unsigned Idx) {
  if (!TPL || Idx >= TPL->size() || Policy.AlwaysIncludeTypeForTemplateArgument)
    return true;
  const NamedDecl *TemplParam = TPL->getParam(Idx);
  if (const auto *ParamValueDecl =
          dyn_cast<NonTypeTemplateParmDecl>(TemplParam))
    if (ParamValueDecl->getType()->getContainedDeducedType())
      return true;
  return false;
}

namespace clang {

void *allocateDefaultArgStorageChain(const ASTContext &C) {
  return new (C) char[sizeof(void*) * 2];
}

} // namespace clang

//===----------------------------------------------------------------------===//
// TemplateDecl Implementation
//===----------------------------------------------------------------------===//

TemplateDecl::TemplateDecl(Kind DK, DeclContext *DC, SourceLocation L,
                           DeclarationName Name, TemplateParameterList *Params,
                           NamedDecl *Decl)
    : NamedDecl(DK, DC, L, Name), TemplatedDecl(Decl), TemplateParams(Params) {}

void TemplateDecl::anchor() {}

void TemplateDecl::getAssociatedConstraints(
    llvm::SmallVectorImpl<AssociatedConstraint> &ACs) const {
  TemplateParams->getAssociatedConstraints(ACs);
  if (auto *FD = dyn_cast_or_null<FunctionDecl>(getTemplatedDecl()))
    if (const AssociatedConstraint &TRC = FD->getTrailingRequiresClause())
      ACs.emplace_back(TRC);
}

bool TemplateDecl::hasAssociatedConstraints() const {
  if (TemplateParams->hasAssociatedConstraints())
    return true;
  if (auto *FD = dyn_cast_or_null<FunctionDecl>(getTemplatedDecl()))
    return static_cast<bool>(FD->getTrailingRequiresClause());
  return false;
}

bool TemplateDecl::isTypeAlias() const {
  switch (getKind()) {
  case TemplateDecl::TypeAliasTemplate:
  case TemplateDecl::BuiltinTemplate:
    return true;
  default:
    return false;
  };
}

//===----------------------------------------------------------------------===//
// RedeclarableTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void RedeclarableTemplateDecl::anchor() {}

RedeclarableTemplateDecl::CommonBase *RedeclarableTemplateDecl::getCommonPtr() const {
  if (Common)
    return Common;

  // Walk the previous-declaration chain until we either find a declaration
  // with a common pointer or we run out of previous declarations.
  SmallVector<const RedeclarableTemplateDecl *, 2> PrevDecls;
  for (const RedeclarableTemplateDecl *Prev = getPreviousDecl(); Prev;
       Prev = Prev->getPreviousDecl()) {
    if (Prev->Common) {
      Common = Prev->Common;
      break;
    }

    PrevDecls.push_back(Prev);
  }

  // If we never found a common pointer, allocate one now.
  if (!Common) {
    // FIXME: If any of the declarations is from an AST file, we probably
    // need an update record to add the common data.

    Common = newCommon(getASTContext());
  }

  // Update any previous declarations we saw with the common pointer.
  for (const RedeclarableTemplateDecl *Prev : PrevDecls)
    Prev->Common = Common;

  return Common;
}

void RedeclarableTemplateDecl::loadLazySpecializationsImpl(
    bool OnlyPartial /*=false*/) const {
  auto *ExternalSource = getASTContext().getExternalSource();
  if (!ExternalSource)
    return;

  ExternalSource->LoadExternalSpecializations(this->getCanonicalDecl(),
                                              OnlyPartial);
}

bool RedeclarableTemplateDecl::loadLazySpecializationsImpl(
    ArrayRef<TemplateArgument> Args, TemplateParameterList *TPL) const {
  auto *ExternalSource = getASTContext().getExternalSource();
  if (!ExternalSource)
    return false;

  // If TPL is not null, it implies that we're loading specializations for
  // partial templates. We need to load all specializations in such cases.
  if (TPL)
    return ExternalSource->LoadExternalSpecializations(this->getCanonicalDecl(),
                                                       /*OnlyPartial=*/false);

  return ExternalSource->LoadExternalSpecializations(this->getCanonicalDecl(),
                                                     Args);
}

template <class EntryType, typename... ProfileArguments>
typename RedeclarableTemplateDecl::SpecEntryTraits<EntryType>::DeclType *
RedeclarableTemplateDecl::findSpecializationLocally(
    llvm::FoldingSetVector<EntryType> &Specs, void *&InsertPos,
    ProfileArguments... ProfileArgs) {
  using SETraits = RedeclarableTemplateDecl::SpecEntryTraits<EntryType>;

  llvm::FoldingSetNodeID ID;
  EntryType::Profile(ID, ProfileArgs..., getASTContext());
  EntryType *Entry = Specs.FindNodeOrInsertPos(ID, InsertPos);
  return Entry ? SETraits::getDecl(Entry)->getMostRecentDecl() : nullptr;
}

template <class EntryType, typename... ProfileArguments>
typename RedeclarableTemplateDecl::SpecEntryTraits<EntryType>::DeclType *
RedeclarableTemplateDecl::findSpecializationImpl(
    llvm::FoldingSetVector<EntryType> &Specs, void *&InsertPos,
    ProfileArguments... ProfileArgs) {

  if (auto *Found = findSpecializationLocally(Specs, InsertPos, ProfileArgs...))
    return Found;

  if (!loadLazySpecializationsImpl(ProfileArgs...))
    return nullptr;

  return findSpecializationLocally(Specs, InsertPos, ProfileArgs...);
}

template<class Derived, class EntryType>
void RedeclarableTemplateDecl::addSpecializationImpl(
    llvm::FoldingSetVector<EntryType> &Specializations, EntryType *Entry,
    void *InsertPos) {
  using SETraits = SpecEntryTraits<EntryType>;

  if (InsertPos) {
#ifndef NDEBUG
    auto Args = SETraits::getTemplateArgs(Entry);
    // Due to hash collisions, it can happen that we load another template
    // specialization with the same hash. This is fine, as long as the next
    // call to findSpecializationImpl does not find a matching Decl for the
    // template arguments.
    loadLazySpecializationsImpl(Args);
    void *CorrectInsertPos;
    assert(!findSpecializationImpl(Specializations, CorrectInsertPos, Args) &&
           InsertPos == CorrectInsertPos &&
           "given incorrect InsertPos for specialization");
#endif
    Specializations.InsertNode(Entry, InsertPos);
  } else {
    EntryType *Existing = Specializations.GetOrInsertNode(Entry);
    (void)Existing;
    assert(SETraits::getDecl(Existing)->isCanonicalDecl() &&
           "non-canonical specialization?");
  }

  if (ASTMutationListener *L = getASTMutationListener())
    L->AddedCXXTemplateSpecialization(cast<Derived>(this),
                                      SETraits::getDecl(Entry));
}

//===----------------------------------------------------------------------===//
// FunctionTemplateDecl Implementation
//===----------------------------------------------------------------------===//

FunctionTemplateDecl *
FunctionTemplateDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             DeclarationName Name,
                             TemplateParameterList *Params, NamedDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, cast<DeclContext>(Decl));
  auto *TD = new (C, DC) FunctionTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

FunctionTemplateDecl *
FunctionTemplateDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID) {
  return new (C, ID) FunctionTemplateDecl(C, nullptr, SourceLocation(),
                                          DeclarationName(), nullptr, nullptr);
}

RedeclarableTemplateDecl::CommonBase *
FunctionTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

void FunctionTemplateDecl::LoadLazySpecializations() const {
  loadLazySpecializationsImpl();
}

llvm::FoldingSetVector<FunctionTemplateSpecializationInfo> &
FunctionTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

FunctionDecl *
FunctionTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                         void *&InsertPos) {
  auto *Common = getCommonPtr();
  return findSpecializationImpl(Common->Specializations, InsertPos, Args);
}

void FunctionTemplateDecl::addSpecialization(
      FunctionTemplateSpecializationInfo *Info, void *InsertPos) {
  auto *Common = getCommonPtr();
  addSpecializationImpl<FunctionTemplateDecl>(Common->Specializations, Info,
                                              InsertPos);
}

void FunctionTemplateDecl::mergePrevDecl(FunctionTemplateDecl *Prev) {
  using Base = RedeclarableTemplateDecl;

  // If we haven't created a common pointer yet, then it can just be created
  // with the usual method.
  if (!Base::Common)
    return;

  Common *ThisCommon = static_cast<Common *>(Base::Common);
  Common *PrevCommon = nullptr;
  SmallVector<FunctionTemplateDecl *, 8> PreviousDecls;
  for (; Prev; Prev = Prev->getPreviousDecl()) {
    if (Prev->Base::Common) {
      PrevCommon = static_cast<Common *>(Prev->Base::Common);
      break;
    }
    PreviousDecls.push_back(Prev);
  }

  // If the previous redecl chain hasn't created a common pointer yet, then just
  // use this common pointer.
  if (!PrevCommon) {
    for (auto *D : PreviousDecls)
      D->Base::Common = ThisCommon;
    return;
  }

  // Ensure we don't leak any important state.
  assert(ThisCommon->Specializations.size() == 0 &&
         "Can't merge incompatible declarations!");

  Base::Common = PrevCommon;
}

//===----------------------------------------------------------------------===//
// ClassTemplateDecl Implementation
//===----------------------------------------------------------------------===//

ClassTemplateDecl *ClassTemplateDecl::Create(ASTContext &C, DeclContext *DC,
                                             SourceLocation L,
                                             DeclarationName Name,
                                             TemplateParameterList *Params,
                                             NamedDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, cast<DeclContext>(Decl));
  auto *TD = new (C, DC) ClassTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

ClassTemplateDecl *ClassTemplateDecl::CreateDeserialized(ASTContext &C,
                                                         GlobalDeclID ID) {
  return new (C, ID) ClassTemplateDecl(C, nullptr, SourceLocation(),
                                       DeclarationName(), nullptr, nullptr);
}

void ClassTemplateDecl::LoadLazySpecializations(
    bool OnlyPartial /*=false*/) const {
  loadLazySpecializationsImpl(OnlyPartial);
}

llvm::FoldingSetVector<ClassTemplateSpecializationDecl> &
ClassTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl> &
ClassTemplateDecl::getPartialSpecializations() const {
  LoadLazySpecializations(/*PartialOnly = */ true);
  return getCommonPtr()->PartialSpecializations;
}

RedeclarableTemplateDecl::CommonBase *
ClassTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

ClassTemplateSpecializationDecl *
ClassTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                      void *&InsertPos) {
  auto *Common = getCommonPtr();
  return findSpecializationImpl(Common->Specializations, InsertPos, Args);
}

void ClassTemplateDecl::AddSpecialization(ClassTemplateSpecializationDecl *D,
                                          void *InsertPos) {
  auto *Common = getCommonPtr();
  addSpecializationImpl<ClassTemplateDecl>(Common->Specializations, D,
                                           InsertPos);
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecialization(
    ArrayRef<TemplateArgument> Args,
    TemplateParameterList *TPL, void *&InsertPos) {
  return findSpecializationImpl(getPartialSpecializations(), InsertPos, Args,
                                TPL);
}

void ClassTemplatePartialSpecializationDecl::Profile(
    llvm::FoldingSetNodeID &ID, ArrayRef<TemplateArgument> TemplateArgs,
    TemplateParameterList *TPL, const ASTContext &Context) {
  ID.AddInteger(TemplateArgs.size());
  for (const TemplateArgument &TemplateArg : TemplateArgs)
    TemplateArg.Profile(ID, Context);
  TPL->Profile(ID, Context);
}

void ClassTemplateDecl::AddPartialSpecialization(
                                      ClassTemplatePartialSpecializationDecl *D,
                                      void *InsertPos) {
  if (InsertPos)
    getPartialSpecializations().InsertNode(D, InsertPos);
  else {
    ClassTemplatePartialSpecializationDecl *Existing
      = getPartialSpecializations().GetOrInsertNode(D);
    (void)Existing;
    assert(Existing->isCanonicalDecl() && "Non-canonical specialization?");
  }

  if (ASTMutationListener *L = getASTMutationListener())
    L->AddedCXXTemplateSpecialization(this, D);
}

void ClassTemplateDecl::getPartialSpecializations(
    SmallVectorImpl<ClassTemplatePartialSpecializationDecl *> &PS) const {
  llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl> &PartialSpecs
    = getPartialSpecializations();
  PS.clear();
  PS.reserve(PartialSpecs.size());
  for (ClassTemplatePartialSpecializationDecl &P : PartialSpecs)
    PS.push_back(P.getMostRecentDecl());
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecialization(QualType T) {
  ASTContext &Context = getASTContext();
  for (ClassTemplatePartialSpecializationDecl &P :
       getPartialSpecializations()) {
    if (Context.hasSameType(P.getCanonicalInjectedSpecializationType(Context),
                            T))
      return P.getMostRecentDecl();
  }

  return nullptr;
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecInstantiatedFromMember(
                                    ClassTemplatePartialSpecializationDecl *D) {
  Decl *DCanon = D->getCanonicalDecl();
  for (ClassTemplatePartialSpecializationDecl &P : getPartialSpecializations()) {
    if (P.getInstantiatedFromMember()->getCanonicalDecl() == DCanon)
      return P.getMostRecentDecl();
  }

  return nullptr;
}

CanQualType ClassTemplateDecl::getCanonicalInjectedSpecializationType(
    const ASTContext &Ctx) const {
  Common *CommonPtr = getCommonPtr();

  if (CommonPtr->CanonInjectedTST.isNull()) {
    SmallVector<TemplateArgument> CanonicalArgs(
        getTemplateParameters()->getInjectedTemplateArgs(Ctx));
    Ctx.canonicalizeTemplateArguments(CanonicalArgs);
    CommonPtr->CanonInjectedTST =
        CanQualType::CreateUnsafe(Ctx.getCanonicalTemplateSpecializationType(
            TemplateName(const_cast<ClassTemplateDecl *>(getCanonicalDecl())),
            CanonicalArgs));
  }
  return CommonPtr->CanonInjectedTST;
}

//===----------------------------------------------------------------------===//
// TemplateTypeParm Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

TemplateTypeParmDecl *TemplateTypeParmDecl::Create(
    const ASTContext &C, DeclContext *DC, SourceLocation KeyLoc,
    SourceLocation NameLoc, unsigned D, unsigned P, IdentifierInfo *Id,
    bool Typename, bool ParameterPack, bool HasTypeConstraint,
    UnsignedOrNone NumExpanded) {
  auto *TTPDecl =
      new (C, DC,
           additionalSizeToAlloc<TypeConstraint>(HasTypeConstraint ? 1 : 0))
      TemplateTypeParmDecl(DC, KeyLoc, NameLoc, Id, Typename,
                           HasTypeConstraint, NumExpanded);
  QualType TTPType = C.getTemplateTypeParmType(D, P, ParameterPack, TTPDecl);
  TTPDecl->setTypeForDecl(TTPType.getTypePtr());
  return TTPDecl;
}

TemplateTypeParmDecl *
TemplateTypeParmDecl::CreateDeserialized(const ASTContext &C, GlobalDeclID ID) {
  return new (C, ID)
      TemplateTypeParmDecl(nullptr, SourceLocation(), SourceLocation(), nullptr,
                           false, false, std::nullopt);
}

TemplateTypeParmDecl *
TemplateTypeParmDecl::CreateDeserialized(const ASTContext &C, GlobalDeclID ID,
                                         bool HasTypeConstraint) {
  return new (C, ID,
              additionalSizeToAlloc<TypeConstraint>(HasTypeConstraint ? 1 : 0))
      TemplateTypeParmDecl(nullptr, SourceLocation(), SourceLocation(), nullptr,
                           false, HasTypeConstraint, std::nullopt);
}

SourceLocation TemplateTypeParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument() ? getDefaultArgument().getLocation()
                              : SourceLocation();
}

SourceRange TemplateTypeParmDecl::getSourceRange() const {
  if (hasDefaultArgument() && !defaultArgumentWasInherited())
    return SourceRange(getBeginLoc(),
                       getDefaultArgument().getSourceRange().getEnd());
  // TypeDecl::getSourceRange returns a range containing name location, which is
  // wrong for unnamed template parameters. e.g:
  // it will return <[[typename>]] instead of <[[typename]]>
  if (getDeclName().isEmpty())
    return SourceRange(getBeginLoc());
  return TypeDecl::getSourceRange();
}

void TemplateTypeParmDecl::setDefaultArgument(
    const ASTContext &C, const TemplateArgumentLoc &DefArg) {
  if (DefArg.getArgument().isNull())
    DefaultArgument.set(nullptr);
  else
    DefaultArgument.set(new (C) TemplateArgumentLoc(DefArg));
}

unsigned TemplateTypeParmDecl::getDepth() const {
  return getTypeForDecl()->castAs<TemplateTypeParmType>()->getDepth();
}

unsigned TemplateTypeParmDecl::getIndex() const {
  return getTypeForDecl()->castAs<TemplateTypeParmType>()->getIndex();
}

bool TemplateTypeParmDecl::isParameterPack() const {
  return getTypeForDecl()->castAs<TemplateTypeParmType>()->isParameterPack();
}

void TemplateTypeParmDecl::setTypeConstraint(
    ConceptReference *Loc, Expr *ImmediatelyDeclaredConstraint,
    UnsignedOrNone ArgPackSubstIndex) {
  assert(HasTypeConstraint &&
         "HasTypeConstraint=true must be passed at construction in order to "
         "call setTypeConstraint");
  assert(!TypeConstraintInitialized &&
         "TypeConstraint was already initialized!");
  new (getTrailingObjects())
      TypeConstraint(Loc, ImmediatelyDeclaredConstraint, ArgPackSubstIndex);
  TypeConstraintInitialized = true;
}

//===----------------------------------------------------------------------===//
// NonTypeTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

NonTypeTemplateParmDecl::NonTypeTemplateParmDecl(
    DeclContext *DC, SourceLocation StartLoc, SourceLocation IdLoc, unsigned D,
    unsigned P, const IdentifierInfo *Id, QualType T, TypeSourceInfo *TInfo,
    ArrayRef<QualType> ExpandedTypes, ArrayRef<TypeSourceInfo *> ExpandedTInfos)
    : DeclaratorDecl(NonTypeTemplateParm, DC, IdLoc, Id, T, TInfo, StartLoc),
      TemplateParmPosition(D, P), ParameterPack(true),
      ExpandedParameterPack(true), NumExpandedTypes(ExpandedTypes.size()) {
  if (!ExpandedTypes.empty() && !ExpandedTInfos.empty()) {
    auto TypesAndInfos =
        getTrailingObjects<std::pair<QualType, TypeSourceInfo *>>();
    for (unsigned I = 0; I != NumExpandedTypes; ++I) {
      new (&TypesAndInfos[I].first) QualType(ExpandedTypes[I]);
      TypesAndInfos[I].second = ExpandedTInfos[I];
    }
  }
}

NonTypeTemplateParmDecl *NonTypeTemplateParmDecl::Create(
    const ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, unsigned D, unsigned P, const IdentifierInfo *Id,
    QualType T, bool ParameterPack, TypeSourceInfo *TInfo) {
  AutoType *AT =
      C.getLangOpts().CPlusPlus20 ? T->getContainedAutoType() : nullptr;
  const bool HasConstraint = AT && AT->isConstrained();
  auto *NTTP =
      new (C, DC,
           additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>, Expr *>(
               0, HasConstraint ? 1 : 0))
          NonTypeTemplateParmDecl(DC, StartLoc, IdLoc, D, P, Id, T,
                                  ParameterPack, TInfo);
  if (HasConstraint)
    NTTP->setPlaceholderTypeConstraint(nullptr);
  return NTTP;
}

NonTypeTemplateParmDecl *NonTypeTemplateParmDecl::Create(
    const ASTContext &C, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, unsigned D, unsigned P, const IdentifierInfo *Id,
    QualType T, TypeSourceInfo *TInfo, ArrayRef<QualType> ExpandedTypes,
    ArrayRef<TypeSourceInfo *> ExpandedTInfos) {
  AutoType *AT = TInfo->getType()->getContainedAutoType();
  const bool HasConstraint = AT && AT->isConstrained();
  auto *NTTP =
      new (C, DC,
           additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>, Expr *>(
               ExpandedTypes.size(), HasConstraint ? 1 : 0))
          NonTypeTemplateParmDecl(DC, StartLoc, IdLoc, D, P, Id, T, TInfo,
                                  ExpandedTypes, ExpandedTInfos);
  if (HasConstraint)
    NTTP->setPlaceholderTypeConstraint(nullptr);
  return NTTP;
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID,
                                            bool HasTypeConstraint) {
  auto *NTTP =
      new (C, ID,
           additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>, Expr *>(
               0, HasTypeConstraint ? 1 : 0))
          NonTypeTemplateParmDecl(nullptr, SourceLocation(), SourceLocation(),
                                  0, 0, nullptr, QualType(), false, nullptr);
  if (HasTypeConstraint)
    NTTP->setPlaceholderTypeConstraint(nullptr);
  return NTTP;
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID,
                                            unsigned NumExpandedTypes,
                                            bool HasTypeConstraint) {
  auto *NTTP =
      new (C, ID,
           additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>, Expr *>(
               NumExpandedTypes, HasTypeConstraint ? 1 : 0))
          NonTypeTemplateParmDecl(nullptr, SourceLocation(), SourceLocation(),
                                  0, 0, nullptr, QualType(), nullptr, {}, {});
  NTTP->NumExpandedTypes = NumExpandedTypes;
  if (HasTypeConstraint)
    NTTP->setPlaceholderTypeConstraint(nullptr);
  return NTTP;
}

SourceRange NonTypeTemplateParmDecl::getSourceRange() const {
  if (hasDefaultArgument() && !defaultArgumentWasInherited())
    return SourceRange(getOuterLocStart(),
                       getDefaultArgument().getSourceRange().getEnd());
  return DeclaratorDecl::getSourceRange();
}

SourceLocation NonTypeTemplateParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument() ? getDefaultArgument().getSourceRange().getBegin()
                              : SourceLocation();
}

void NonTypeTemplateParmDecl::setDefaultArgument(
    const ASTContext &C, const TemplateArgumentLoc &DefArg) {
  if (DefArg.getArgument().isNull())
    DefaultArgument.set(nullptr);
  else
    DefaultArgument.set(new (C) TemplateArgumentLoc(DefArg));
}

//===----------------------------------------------------------------------===//
// TemplateTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

void TemplateTemplateParmDecl::anchor() {}

TemplateTemplateParmDecl::TemplateTemplateParmDecl(
    DeclContext *DC, SourceLocation L, unsigned D, unsigned P,
    IdentifierInfo *Id, TemplateNameKind Kind, bool Typename,
    TemplateParameterList *Params, ArrayRef<TemplateParameterList *> Expansions)
    : TemplateDecl(TemplateTemplateParm, DC, L, Id, Params),
      TemplateParmPosition(D, P), ParameterKind(Kind), Typename(Typename),
      ParameterPack(true), ExpandedParameterPack(true),
      NumExpandedParams(Expansions.size()) {
  llvm::uninitialized_copy(Expansions, getTrailingObjects());
}

TemplateTemplateParmDecl *TemplateTemplateParmDecl::Create(
    const ASTContext &C, DeclContext *DC, SourceLocation L, unsigned D,
    unsigned P, bool ParameterPack, IdentifierInfo *Id, TemplateNameKind Kind,
    bool Typename, TemplateParameterList *Params) {
  return new (C, DC) TemplateTemplateParmDecl(DC, L, D, P, ParameterPack, Id,
                                              Kind, Typename, Params);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC,
                                 SourceLocation L, unsigned D, unsigned P,
                                 IdentifierInfo *Id, TemplateNameKind Kind,
                                 bool Typename, TemplateParameterList *Params,
                                 ArrayRef<TemplateParameterList *> Expansions) {
  return new (C, DC,
              additionalSizeToAlloc<TemplateParameterList *>(Expansions.size()))
      TemplateTemplateParmDecl(DC, L, D, P, Id, Kind, Typename, Params,
                               Expansions);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID) {
  return new (C, ID) TemplateTemplateParmDecl(
      nullptr, SourceLocation(), 0, 0, false, nullptr,
      TemplateNameKind::TNK_Type_template, false, nullptr);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID,
                                             unsigned NumExpansions) {
  auto *TTP =
      new (C, ID, additionalSizeToAlloc<TemplateParameterList *>(NumExpansions))
          TemplateTemplateParmDecl(nullptr, SourceLocation(), 0, 0, nullptr,
                                   TemplateNameKind::TNK_Type_template, false,
                                   nullptr, {});
  TTP->NumExpandedParams = NumExpansions;
  return TTP;
}

SourceLocation TemplateTemplateParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument() ? getDefaultArgument().getLocation()
                              : SourceLocation();
}

void TemplateTemplateParmDecl::setDefaultArgument(
    const ASTContext &C, const TemplateArgumentLoc &DefArg) {
  if (DefArg.getArgument().isNull())
    DefaultArgument.set(nullptr);
  else
    DefaultArgument.set(new (C) TemplateArgumentLoc(DefArg));
}

//===----------------------------------------------------------------------===//
// TemplateArgumentList Implementation
//===----------------------------------------------------------------------===//
TemplateArgumentList::TemplateArgumentList(ArrayRef<TemplateArgument> Args)
    : NumArguments(Args.size()) {
  llvm::uninitialized_copy(Args, getTrailingObjects());
}

TemplateArgumentList *
TemplateArgumentList::CreateCopy(ASTContext &Context,
                                 ArrayRef<TemplateArgument> Args) {
  void *Mem = Context.Allocate(totalSizeToAlloc<TemplateArgument>(Args.size()));
  return new (Mem) TemplateArgumentList(Args);
}

FunctionTemplateSpecializationInfo *FunctionTemplateSpecializationInfo::Create(
    ASTContext &C, FunctionDecl *FD, FunctionTemplateDecl *Template,
    TemplateSpecializationKind TSK, TemplateArgumentList *TemplateArgs,
    const TemplateArgumentListInfo *TemplateArgsAsWritten, SourceLocation POI,
    MemberSpecializationInfo *MSInfo) {
  const ASTTemplateArgumentListInfo *ArgsAsWritten = nullptr;
  if (TemplateArgsAsWritten)
    ArgsAsWritten = ASTTemplateArgumentListInfo::Create(C,
                                                        *TemplateArgsAsWritten);

  void *Mem =
      C.Allocate(totalSizeToAlloc<MemberSpecializationInfo *>(MSInfo ? 1 : 0));
  return new (Mem) FunctionTemplateSpecializationInfo(
      FD, Template, TSK, TemplateArgs, ArgsAsWritten, POI, MSInfo);
}

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

ClassTemplateSpecializationDecl::ClassTemplateSpecializationDecl(
    ASTContext &Context, Kind DK, TagKind TK, DeclContext *DC,
    SourceLocation StartLoc, SourceLocation IdLoc,
    ClassTemplateDecl *SpecializedTemplate, ArrayRef<TemplateArgument> Args,
    bool StrictPackMatch, ClassTemplateSpecializationDecl *PrevDecl)
    : CXXRecordDecl(DK, TK, Context, DC, StartLoc, IdLoc,
                    SpecializedTemplate->getIdentifier(), PrevDecl),
      SpecializedTemplate(SpecializedTemplate),
      TemplateArgs(TemplateArgumentList::CreateCopy(Context, Args)),
      SpecializationKind(TSK_Undeclared), StrictPackMatch(StrictPackMatch) {
  assert(DK == Kind::ClassTemplateSpecialization || StrictPackMatch == false);
}

ClassTemplateSpecializationDecl::ClassTemplateSpecializationDecl(ASTContext &C,
                                                                 Kind DK)
    : CXXRecordDecl(DK, TagTypeKind::Struct, C, nullptr, SourceLocation(),
                    SourceLocation(), nullptr, nullptr),
      SpecializationKind(TSK_Undeclared) {}

ClassTemplateSpecializationDecl *ClassTemplateSpecializationDecl::Create(
    ASTContext &Context, TagKind TK, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, ClassTemplateDecl *SpecializedTemplate,
    ArrayRef<TemplateArgument> Args, bool StrictPackMatch,
    ClassTemplateSpecializationDecl *PrevDecl) {
  auto *Result = new (Context, DC) ClassTemplateSpecializationDecl(
      Context, ClassTemplateSpecialization, TK, DC, StartLoc, IdLoc,
      SpecializedTemplate, Args, StrictPackMatch, PrevDecl);

  // If the template decl is incomplete, copy the external lexical storage from
  // the base template. This allows instantiations of incomplete types to
  // complete using the external AST if the template's declaration came from an
  // external AST.
  if (!SpecializedTemplate->getTemplatedDecl()->isCompleteDefinition())
    Result->setHasExternalLexicalStorage(
      SpecializedTemplate->getTemplatedDecl()->hasExternalLexicalStorage());

  return Result;
}

ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                    GlobalDeclID ID) {
  return new (C, ID)
      ClassTemplateSpecializationDecl(C, ClassTemplateSpecialization);
}

void ClassTemplateSpecializationDecl::getNameForDiagnostic(
    raw_ostream &OS, const PrintingPolicy &Policy, bool Qualified) const {
  NamedDecl::getNameForDiagnostic(OS, Policy, Qualified);

  const auto *PS = dyn_cast<ClassTemplatePartialSpecializationDecl>(this);
  if (const ASTTemplateArgumentListInfo *ArgsAsWritten =
          PS ? PS->getTemplateArgsAsWritten() : nullptr) {
    printTemplateArgumentList(
        OS, ArgsAsWritten->arguments(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  } else {
    const TemplateArgumentList &TemplateArgs = getTemplateArgs();
    printTemplateArgumentList(
        OS, TemplateArgs.asArray(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  }
}

ClassTemplateDecl *
ClassTemplateSpecializationDecl::getSpecializedTemplate() const {
  if (const auto *PartialSpec =
          SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization*>())
    return PartialSpec->PartialSpecialization->getSpecializedTemplate();
  return cast<ClassTemplateDecl *>(SpecializedTemplate);
}

SourceRange
ClassTemplateSpecializationDecl::getSourceRange() const {
  switch (getSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ImplicitInstantiation: {
    llvm::PointerUnion<ClassTemplateDecl *,
                       ClassTemplatePartialSpecializationDecl *>
        Pattern = getSpecializedTemplateOrPartial();
    assert(!Pattern.isNull() &&
           "Class template specialization without pattern?");
    if (const auto *CTPSD =
            dyn_cast<ClassTemplatePartialSpecializationDecl *>(Pattern))
      return CTPSD->getSourceRange();
    return cast<ClassTemplateDecl *>(Pattern)->getSourceRange();
  }
  case TSK_ExplicitSpecialization: {
    SourceRange Range = CXXRecordDecl::getSourceRange();
    if (const ASTTemplateArgumentListInfo *Args = getTemplateArgsAsWritten();
        !isThisDeclarationADefinition() && Args)
      Range.setEnd(Args->getRAngleLoc());
    return Range;
  }
  case TSK_ExplicitInstantiationDeclaration:
  case TSK_ExplicitInstantiationDefinition: {
    SourceRange Range = CXXRecordDecl::getSourceRange();
    if (SourceLocation ExternKW = getExternKeywordLoc(); ExternKW.isValid())
      Range.setBegin(ExternKW);
    else if (SourceLocation TemplateKW = getTemplateKeywordLoc();
             TemplateKW.isValid())
      Range.setBegin(TemplateKW);
    if (const ASTTemplateArgumentListInfo *Args = getTemplateArgsAsWritten())
      Range.setEnd(Args->getRAngleLoc());
    return Range;
  }
  }
  llvm_unreachable("unhandled template specialization kind");
}

void ClassTemplateSpecializationDecl::setExternKeywordLoc(SourceLocation Loc) {
  auto *Info = dyn_cast_if_present<ExplicitInstantiationInfo *>(ExplicitInfo);
  if (!Info) {
    // Don't allocate if the location is invalid.
    if (Loc.isInvalid())
      return;
    Info = new (getASTContext()) ExplicitInstantiationInfo;
    Info->TemplateArgsAsWritten = getTemplateArgsAsWritten();
    ExplicitInfo = Info;
  }
  Info->ExternKeywordLoc = Loc;
}

void ClassTemplateSpecializationDecl::setTemplateKeywordLoc(
    SourceLocation Loc) {
  auto *Info = dyn_cast_if_present<ExplicitInstantiationInfo *>(ExplicitInfo);
  if (!Info) {
    // Don't allocate if the location is invalid.
    if (Loc.isInvalid())
      return;
    Info = new (getASTContext()) ExplicitInstantiationInfo;
    Info->TemplateArgsAsWritten = getTemplateArgsAsWritten();
    ExplicitInfo = Info;
  }
  Info->TemplateKeywordLoc = Loc;
}

//===----------------------------------------------------------------------===//
// ConceptDecl Implementation
//===----------------------------------------------------------------------===//
ConceptDecl *ConceptDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, DeclarationName Name,
                                 TemplateParameterList *Params,
                                 Expr *ConstraintExpr) {
  bool Invalid = AdoptTemplateParameterList(Params, DC);
  auto *TD = new (C, DC) ConceptDecl(DC, L, Name, Params, ConstraintExpr);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

ConceptDecl *ConceptDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID) {
  ConceptDecl *Result = new (C, ID) ConceptDecl(nullptr, SourceLocation(),
                                                DeclarationName(),
                                                nullptr, nullptr);

  return Result;
}

//===----------------------------------------------------------------------===//
// ImplicitConceptSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
ImplicitConceptSpecializationDecl::ImplicitConceptSpecializationDecl(
    DeclContext *DC, SourceLocation SL,
    ArrayRef<TemplateArgument> ConvertedArgs)
    : Decl(ImplicitConceptSpecialization, DC, SL),
      NumTemplateArgs(ConvertedArgs.size()) {
  setTemplateArguments(ConvertedArgs);
}

ImplicitConceptSpecializationDecl::ImplicitConceptSpecializationDecl(
    EmptyShell Empty, unsigned NumTemplateArgs)
    : Decl(ImplicitConceptSpecialization, Empty),
      NumTemplateArgs(NumTemplateArgs) {}

ImplicitConceptSpecializationDecl *ImplicitConceptSpecializationDecl::Create(
    const ASTContext &C, DeclContext *DC, SourceLocation SL,
    ArrayRef<TemplateArgument> ConvertedArgs) {
  return new (C, DC,
              additionalSizeToAlloc<TemplateArgument>(ConvertedArgs.size()))
      ImplicitConceptSpecializationDecl(DC, SL, ConvertedArgs);
}

ImplicitConceptSpecializationDecl *
ImplicitConceptSpecializationDecl::CreateDeserialized(
    const ASTContext &C, GlobalDeclID ID, unsigned NumTemplateArgs) {
  return new (C, ID, additionalSizeToAlloc<TemplateArgument>(NumTemplateArgs))
      ImplicitConceptSpecializationDecl(EmptyShell{}, NumTemplateArgs);
}

void ImplicitConceptSpecializationDecl::setTemplateArguments(
    ArrayRef<TemplateArgument> Converted) {
  assert(Converted.size() == NumTemplateArgs);
  llvm::uninitialized_copy(Converted, getTrailingObjects());
}

//===----------------------------------------------------------------------===//
// ClassTemplatePartialSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
void ClassTemplatePartialSpecializationDecl::anchor() {}

ClassTemplatePartialSpecializationDecl::ClassTemplatePartialSpecializationDecl(
    ASTContext &Context, TagKind TK, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    ClassTemplateDecl *SpecializedTemplate, ArrayRef<TemplateArgument> Args,
    CanQualType CanonInjectedTST,
    ClassTemplatePartialSpecializationDecl *PrevDecl)
    : ClassTemplateSpecializationDecl(
          Context, ClassTemplatePartialSpecialization, TK, DC, StartLoc, IdLoc,
          // Tracking StrictPackMatch for Partial
          // Specializations is not needed.
          SpecializedTemplate, Args, /*StrictPackMatch=*/false, PrevDecl),
      TemplateParams(Params), InstantiatedFromMember(nullptr, false),
      CanonInjectedTST(CanonInjectedTST) {
  if (AdoptTemplateParameterList(Params, this))
    setInvalidDecl();
}

ClassTemplatePartialSpecializationDecl *
ClassTemplatePartialSpecializationDecl::Create(
    ASTContext &Context, TagKind TK, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    ClassTemplateDecl *SpecializedTemplate, ArrayRef<TemplateArgument> Args,
    CanQualType CanonInjectedTST,
    ClassTemplatePartialSpecializationDecl *PrevDecl) {
  auto *Result = new (Context, DC) ClassTemplatePartialSpecializationDecl(
      Context, TK, DC, StartLoc, IdLoc, Params, SpecializedTemplate, Args,
      CanonInjectedTST, PrevDecl);
  Result->setSpecializationKind(TSK_ExplicitSpecialization);
  return Result;
}

ClassTemplatePartialSpecializationDecl *
ClassTemplatePartialSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                           GlobalDeclID ID) {
  return new (C, ID) ClassTemplatePartialSpecializationDecl(C);
}

CanQualType
ClassTemplatePartialSpecializationDecl::getCanonicalInjectedSpecializationType(
    const ASTContext &Ctx) const {
  if (CanonInjectedTST.isNull()) {
    CanonInjectedTST =
        CanQualType::CreateUnsafe(Ctx.getCanonicalTemplateSpecializationType(
            TemplateName(getSpecializedTemplate()->getCanonicalDecl()),
            getTemplateArgs().asArray()));
  }
  return CanonInjectedTST;
}

SourceRange ClassTemplatePartialSpecializationDecl::getSourceRange() const {
  if (const ClassTemplatePartialSpecializationDecl *MT =
          getInstantiatedFromMember();
      MT && !isMemberSpecialization())
    return MT->getSourceRange();
  SourceRange Range = ClassTemplateSpecializationDecl::getSourceRange();
  if (const TemplateParameterList *TPL = getTemplateParameters();
      TPL && !getNumTemplateParameterLists())
    Range.setBegin(TPL->getTemplateLoc());
  return Range;
}

//===----------------------------------------------------------------------===//
// FriendTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void FriendTemplateDecl::anchor() {}

FriendTemplateDecl *
FriendTemplateDecl::Create(ASTContext &Context, DeclContext *DC,
                           SourceLocation L,
                           MutableArrayRef<TemplateParameterList *> Params,
                           FriendUnion Friend, SourceLocation FLoc) {
  TemplateParameterList **TPL = nullptr;
  if (!Params.empty()) {
    TPL = new (Context) TemplateParameterList *[Params.size()];
    llvm::copy(Params, TPL);
  }
  return new (Context, DC)
      FriendTemplateDecl(DC, L, TPL, Params.size(), Friend, FLoc);
}

FriendTemplateDecl *FriendTemplateDecl::CreateDeserialized(ASTContext &C,
                                                           GlobalDeclID ID) {
  return new (C, ID) FriendTemplateDecl(EmptyShell());
}

//===----------------------------------------------------------------------===//
// TypeAliasTemplateDecl Implementation
//===----------------------------------------------------------------------===//

TypeAliasTemplateDecl *
TypeAliasTemplateDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                              DeclarationName Name,
                              TemplateParameterList *Params, NamedDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, DC);
  auto *TD = new (C, DC) TypeAliasTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

TypeAliasTemplateDecl *
TypeAliasTemplateDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID) {
  return new (C, ID) TypeAliasTemplateDecl(C, nullptr, SourceLocation(),
                                           DeclarationName(), nullptr, nullptr);
}

RedeclarableTemplateDecl::CommonBase *
TypeAliasTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

//===----------------------------------------------------------------------===//
// VarTemplateDecl Implementation
//===----------------------------------------------------------------------===//

VarTemplateDecl *VarTemplateDecl::getDefinition() {
  VarTemplateDecl *CurD = this;
  while (CurD) {
    if (CurD->isThisDeclarationADefinition())
      return CurD;
    CurD = CurD->getPreviousDecl();
  }
  return nullptr;
}

VarTemplateDecl *VarTemplateDecl::Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L, DeclarationName Name,
                                         TemplateParameterList *Params,
                                         VarDecl *Decl) {
  bool Invalid = AdoptTemplateParameterList(Params, DC);
  auto *TD = new (C, DC) VarTemplateDecl(C, DC, L, Name, Params, Decl);
  if (Invalid)
    TD->setInvalidDecl();
  return TD;
}

VarTemplateDecl *VarTemplateDecl::CreateDeserialized(ASTContext &C,
                                                     GlobalDeclID ID) {
  return new (C, ID) VarTemplateDecl(C, nullptr, SourceLocation(),
                                     DeclarationName(), nullptr, nullptr);
}

void VarTemplateDecl::LoadLazySpecializations(
    bool OnlyPartial /*=false*/) const {
  loadLazySpecializationsImpl(OnlyPartial);
}

llvm::FoldingSetVector<VarTemplateSpecializationDecl> &
VarTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl> &
VarTemplateDecl::getPartialSpecializations() const {
  LoadLazySpecializations(/*PartialOnly = */ true);
  return getCommonPtr()->PartialSpecializations;
}

RedeclarableTemplateDecl::CommonBase *
VarTemplateDecl::newCommon(ASTContext &C) const {
  auto *CommonPtr = new (C) Common;
  C.addDestruction(CommonPtr);
  return CommonPtr;
}

VarTemplateSpecializationDecl *
VarTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                    void *&InsertPos) {
  auto *Common = getCommonPtr();
  return findSpecializationImpl(Common->Specializations, InsertPos, Args);
}

void VarTemplateDecl::AddSpecialization(VarTemplateSpecializationDecl *D,
                                        void *InsertPos) {
  auto *Common = getCommonPtr();
  addSpecializationImpl<VarTemplateDecl>(Common->Specializations, D, InsertPos);
}

VarTemplatePartialSpecializationDecl *
VarTemplateDecl::findPartialSpecialization(ArrayRef<TemplateArgument> Args,
     TemplateParameterList *TPL, void *&InsertPos) {
  return findSpecializationImpl(getPartialSpecializations(), InsertPos, Args,
                                TPL);
}

void VarTemplatePartialSpecializationDecl::Profile(
    llvm::FoldingSetNodeID &ID, ArrayRef<TemplateArgument> TemplateArgs,
    TemplateParameterList *TPL, const ASTContext &Context) {
  ID.AddInteger(TemplateArgs.size());
  for (const TemplateArgument &TemplateArg : TemplateArgs)
    TemplateArg.Profile(ID, Context);
  TPL->Profile(ID, Context);
}

void VarTemplateDecl::AddPartialSpecialization(
    VarTemplatePartialSpecializationDecl *D, void *InsertPos) {
  if (InsertPos)
    getPartialSpecializations().InsertNode(D, InsertPos);
  else {
    VarTemplatePartialSpecializationDecl *Existing =
        getPartialSpecializations().GetOrInsertNode(D);
    (void)Existing;
    assert(Existing->isCanonicalDecl() && "Non-canonical specialization?");
  }

  if (ASTMutationListener *L = getASTMutationListener())
    L->AddedCXXTemplateSpecialization(this, D);
}

void VarTemplateDecl::getPartialSpecializations(
    SmallVectorImpl<VarTemplatePartialSpecializationDecl *> &PS) const {
  llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl> &PartialSpecs =
      getPartialSpecializations();
  PS.clear();
  PS.reserve(PartialSpecs.size());
  for (VarTemplatePartialSpecializationDecl &P : PartialSpecs)
    PS.push_back(P.getMostRecentDecl());
}

VarTemplatePartialSpecializationDecl *
VarTemplateDecl::findPartialSpecInstantiatedFromMember(
    VarTemplatePartialSpecializationDecl *D) {
  Decl *DCanon = D->getCanonicalDecl();
  for (VarTemplatePartialSpecializationDecl &P : getPartialSpecializations()) {
    if (P.getInstantiatedFromMember()->getCanonicalDecl() == DCanon)
      return P.getMostRecentDecl();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// VarTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

VarTemplateSpecializationDecl::VarTemplateSpecializationDecl(
    Kind DK, ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, VarTemplateDecl *SpecializedTemplate, QualType T,
    TypeSourceInfo *TInfo, StorageClass S, ArrayRef<TemplateArgument> Args)
    : VarDecl(DK, Context, DC, StartLoc, IdLoc,
              SpecializedTemplate->getIdentifier(), T, TInfo, S),
      SpecializedTemplate(SpecializedTemplate),
      TemplateArgs(TemplateArgumentList::CreateCopy(Context, Args)),
      SpecializationKind(TSK_Undeclared), IsCompleteDefinition(false) {}

VarTemplateSpecializationDecl::VarTemplateSpecializationDecl(Kind DK,
                                                             ASTContext &C)
    : VarDecl(DK, C, nullptr, SourceLocation(), SourceLocation(), nullptr,
              QualType(), nullptr, SC_None),
      SpecializationKind(TSK_Undeclared), IsCompleteDefinition(false) {}

VarTemplateSpecializationDecl *VarTemplateSpecializationDecl::Create(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, VarTemplateDecl *SpecializedTemplate, QualType T,
    TypeSourceInfo *TInfo, StorageClass S, ArrayRef<TemplateArgument> Args) {
  return new (Context, DC) VarTemplateSpecializationDecl(
      VarTemplateSpecialization, Context, DC, StartLoc, IdLoc,
      SpecializedTemplate, T, TInfo, S, Args);
}

VarTemplateSpecializationDecl *
VarTemplateSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                  GlobalDeclID ID) {
  return new (C, ID)
      VarTemplateSpecializationDecl(VarTemplateSpecialization, C);
}

void VarTemplateSpecializationDecl::getNameForDiagnostic(
    raw_ostream &OS, const PrintingPolicy &Policy, bool Qualified) const {
  NamedDecl::getNameForDiagnostic(OS, Policy, Qualified);

  const auto *PS = dyn_cast<VarTemplatePartialSpecializationDecl>(this);
  if (const ASTTemplateArgumentListInfo *ArgsAsWritten =
          PS ? PS->getTemplateArgsAsWritten() : nullptr) {
    printTemplateArgumentList(
        OS, ArgsAsWritten->arguments(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  } else {
    const TemplateArgumentList &TemplateArgs = getTemplateArgs();
    printTemplateArgumentList(
        OS, TemplateArgs.asArray(), Policy,
        getSpecializedTemplate()->getTemplateParameters());
  }
}

VarTemplateDecl *VarTemplateSpecializationDecl::getSpecializedTemplate() const {
  if (const auto *PartialSpec =
          SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization *>())
    return PartialSpec->PartialSpecialization->getSpecializedTemplate();
  return cast<VarTemplateDecl *>(SpecializedTemplate);
}

SourceRange VarTemplateSpecializationDecl::getSourceRange() const {
  switch (getSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ImplicitInstantiation: {
    llvm::PointerUnion<VarTemplateDecl *,
                       VarTemplatePartialSpecializationDecl *>
        Pattern = getSpecializedTemplateOrPartial();
    assert(!Pattern.isNull() &&
           "Variable template specialization without pattern?");
    if (const auto *VTPSD =
            dyn_cast<VarTemplatePartialSpecializationDecl *>(Pattern))
      return VTPSD->getSourceRange();
    VarTemplateDecl *VTD = cast<VarTemplateDecl *>(Pattern);
    if (hasInit()) {
      if (VarTemplateDecl *Definition = VTD->getDefinition())
        return Definition->getSourceRange();
    }
    return VTD->getCanonicalDecl()->getSourceRange();
  }
  case TSK_ExplicitSpecialization: {
    SourceRange Range = VarDecl::getSourceRange();
    if (const ASTTemplateArgumentListInfo *Args = getTemplateArgsAsWritten();
        !hasInit() && Args)
      Range.setEnd(Args->getRAngleLoc());
    return Range;
  }
  case TSK_ExplicitInstantiationDeclaration:
  case TSK_ExplicitInstantiationDefinition: {
    SourceRange Range = VarDecl::getSourceRange();
    if (SourceLocation ExternKW = getExternKeywordLoc(); ExternKW.isValid())
      Range.setBegin(ExternKW);
    else if (SourceLocation TemplateKW = getTemplateKeywordLoc();
             TemplateKW.isValid())
      Range.setBegin(TemplateKW);
    if (const ASTTemplateArgumentListInfo *Args = getTemplateArgsAsWritten())
      Range.setEnd(Args->getRAngleLoc());
    return Range;
  }
  }
  llvm_unreachable("unhandled template specialization kind");
}

void VarTemplateSpecializationDecl::setExternKeywordLoc(SourceLocation Loc) {
  auto *Info = dyn_cast_if_present<ExplicitInstantiationInfo *>(ExplicitInfo);
  if (!Info) {
    // Don't allocate if the location is invalid.
    if (Loc.isInvalid())
      return;
    Info = new (getASTContext()) ExplicitInstantiationInfo;
    Info->TemplateArgsAsWritten = getTemplateArgsAsWritten();
    ExplicitInfo = Info;
  }
  Info->ExternKeywordLoc = Loc;
}

void VarTemplateSpecializationDecl::setTemplateKeywordLoc(SourceLocation Loc) {
  auto *Info = dyn_cast_if_present<ExplicitInstantiationInfo *>(ExplicitInfo);
  if (!Info) {
    // Don't allocate if the location is invalid.
    if (Loc.isInvalid())
      return;
    Info = new (getASTContext()) ExplicitInstantiationInfo;
    Info->TemplateArgsAsWritten = getTemplateArgsAsWritten();
    ExplicitInfo = Info;
  }
  Info->TemplateKeywordLoc = Loc;
}

//===----------------------------------------------------------------------===//
// VarTemplatePartialSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

void VarTemplatePartialSpecializationDecl::anchor() {}

VarTemplatePartialSpecializationDecl::VarTemplatePartialSpecializationDecl(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    VarTemplateDecl *SpecializedTemplate, QualType T, TypeSourceInfo *TInfo,
    StorageClass S, ArrayRef<TemplateArgument> Args)
    : VarTemplateSpecializationDecl(VarTemplatePartialSpecialization, Context,
                                    DC, StartLoc, IdLoc, SpecializedTemplate, T,
                                    TInfo, S, Args),
      TemplateParams(Params), InstantiatedFromMember(nullptr, false) {
  if (AdoptTemplateParameterList(Params, DC))
    setInvalidDecl();
}

VarTemplatePartialSpecializationDecl *
VarTemplatePartialSpecializationDecl::Create(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    VarTemplateDecl *SpecializedTemplate, QualType T, TypeSourceInfo *TInfo,
    StorageClass S, ArrayRef<TemplateArgument> Args) {
  auto *Result = new (Context, DC) VarTemplatePartialSpecializationDecl(
      Context, DC, StartLoc, IdLoc, Params, SpecializedTemplate, T, TInfo, S,
      Args);
  Result->setSpecializationKind(TSK_ExplicitSpecialization);
  return Result;
}

VarTemplatePartialSpecializationDecl *
VarTemplatePartialSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                         GlobalDeclID ID) {
  return new (C, ID) VarTemplatePartialSpecializationDecl(C);
}

SourceRange VarTemplatePartialSpecializationDecl::getSourceRange() const {
  if (const VarTemplatePartialSpecializationDecl *MT =
          getInstantiatedFromMember();
      MT && !isMemberSpecialization())
    return MT->getSourceRange();
  SourceRange Range = VarTemplateSpecializationDecl::getSourceRange();
  if (const TemplateParameterList *TPL = getTemplateParameters();
      TPL && !getNumTemplateParameterLists())
    Range.setBegin(TPL->getTemplateLoc());
  return Range;
}

static TemplateParameterList *createBuiltinTemplateParameterList(
    const ASTContext &C, DeclContext *DC, BuiltinTemplateKind BTK) {
  switch (BTK) {
#define CREATE_BUILTIN_TEMPLATE_PARAMETER_LIST
#include "clang/Basic/BuiltinTemplates.inc"
  }

  llvm_unreachable("unhandled BuiltinTemplateKind!");
}

void BuiltinTemplateDecl::anchor() {}

BuiltinTemplateDecl::BuiltinTemplateDecl(const ASTContext &C, DeclContext *DC,
                                         DeclarationName Name,
                                         BuiltinTemplateKind BTK)
    : TemplateDecl(BuiltinTemplate, DC, SourceLocation(), Name,
                   createBuiltinTemplateParameterList(C, DC, BTK)),
      BTK(BTK) {}

TemplateParamObjectDecl *TemplateParamObjectDecl::Create(const ASTContext &C,
                                                         QualType T,
                                                         const APValue &V) {
  DeclContext *DC = C.getTranslationUnitDecl();
  auto *TPOD = new (C, DC) TemplateParamObjectDecl(DC, T, V);
  C.addDestruction(&TPOD->Value);
  return TPOD;
}

TemplateParamObjectDecl *
TemplateParamObjectDecl::CreateDeserialized(ASTContext &C, GlobalDeclID ID) {
  auto *TPOD = new (C, ID) TemplateParamObjectDecl(nullptr, QualType(), APValue());
  C.addDestruction(&TPOD->Value);
  return TPOD;
}

void TemplateParamObjectDecl::printName(llvm::raw_ostream &OS,
                                        const PrintingPolicy &Policy) const {
  OS << "<template param ";
  printAsExpr(OS, Policy);
  OS << ">";
}

void TemplateParamObjectDecl::printAsExpr(llvm::raw_ostream &OS) const {
  printAsExpr(OS, getASTContext().getPrintingPolicy());
}

void TemplateParamObjectDecl::printAsExpr(llvm::raw_ostream &OS,
                                          const PrintingPolicy &Policy) const {
  getType().getUnqualifiedType().print(OS, Policy);
  printAsInit(OS, Policy);
}

void TemplateParamObjectDecl::printAsInit(llvm::raw_ostream &OS) const {
  printAsInit(OS, getASTContext().getPrintingPolicy());
}

void TemplateParamObjectDecl::printAsInit(llvm::raw_ostream &OS,
                                          const PrintingPolicy &Policy) const {
  getValue().printPretty(OS, Policy, getType(), &getASTContext());
}

TemplateParameterList *clang::getReplacedTemplateParameterList(const Decl *D) {
  switch (D->getKind()) {
  case Decl::Kind::CXXRecord:
    return cast<CXXRecordDecl>(D)
        ->getDescribedTemplate()
        ->getTemplateParameters();
  case Decl::Kind::ClassTemplate:
    return cast<ClassTemplateDecl>(D)->getTemplateParameters();
  case Decl::Kind::ClassTemplateSpecialization: {
    const auto *CTSD = cast<ClassTemplateSpecializationDecl>(D);
    auto P = CTSD->getSpecializedTemplateOrPartial();
    if (const auto *CTPSD =
            dyn_cast<ClassTemplatePartialSpecializationDecl *>(P))
      return CTPSD->getTemplateParameters();
    return cast<ClassTemplateDecl *>(P)->getTemplateParameters();
  }
  case Decl::Kind::ClassTemplatePartialSpecialization:
    return cast<ClassTemplatePartialSpecializationDecl>(D)
        ->getTemplateParameters();
  case Decl::Kind::TypeAliasTemplate:
    return cast<TypeAliasTemplateDecl>(D)->getTemplateParameters();
  case Decl::Kind::BuiltinTemplate:
    return cast<BuiltinTemplateDecl>(D)->getTemplateParameters();
  case Decl::Kind::CXXDeductionGuide:
  case Decl::Kind::CXXConversion:
  case Decl::Kind::CXXConstructor:
  case Decl::Kind::CXXDestructor:
  case Decl::Kind::CXXMethod:
  case Decl::Kind::Function:
    return cast<FunctionDecl>(D)
        ->getTemplateSpecializationInfo()
        ->getTemplate()
        ->getTemplateParameters();
  case Decl::Kind::FunctionTemplate:
    return cast<FunctionTemplateDecl>(D)->getTemplateParameters();
  case Decl::Kind::VarTemplate:
    return cast<VarTemplateDecl>(D)->getTemplateParameters();
  case Decl::Kind::VarTemplateSpecialization: {
    const auto *VTSD = cast<VarTemplateSpecializationDecl>(D);
    auto P = VTSD->getSpecializedTemplateOrPartial();
    if (const auto *VTPSD = dyn_cast<VarTemplatePartialSpecializationDecl *>(P))
      return VTPSD->getTemplateParameters();
    return cast<VarTemplateDecl *>(P)->getTemplateParameters();
  }
  case Decl::Kind::VarTemplatePartialSpecialization:
    return cast<VarTemplatePartialSpecializationDecl>(D)
        ->getTemplateParameters();
  case Decl::Kind::TemplateTemplateParm:
    return cast<TemplateTemplateParmDecl>(D)->getTemplateParameters();
  case Decl::Kind::Concept:
    return cast<ConceptDecl>(D)->getTemplateParameters();
  default:
    llvm_unreachable("Unhandled templated declaration kind");
  }
}
