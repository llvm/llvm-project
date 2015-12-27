//===--- DeclTemplate.cpp - Template Declaration AST Node Implementation --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>
using namespace clang;

//===----------------------------------------------------------------------===//
// TemplateParameterList Implementation
//===----------------------------------------------------------------------===//

TemplateParameterList::TemplateParameterList(SourceLocation TemplateLoc,
                                             SourceLocation LAngleLoc,
                                             ArrayRef<NamedDecl *> Params,
                                             SourceLocation RAngleLoc)
  : TemplateLoc(TemplateLoc), LAngleLoc(LAngleLoc), RAngleLoc(RAngleLoc),
    NumParams(Params.size()), ContainsUnexpandedParameterPack(false) {
  assert(this->NumParams == NumParams && "Too many template parameters");
  for (unsigned Idx = 0; Idx < NumParams; ++Idx) {
    NamedDecl *P = Params[Idx];
    begin()[Idx] = P;

    if (!P->isTemplateParameterPack()) {
      if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(P))
        if (NTTP->getType()->containsUnexpandedParameterPack())
          ContainsUnexpandedParameterPack = true;

      if (TemplateTemplateParmDecl *TTP = dyn_cast<TemplateTemplateParmDecl>(P))
        if (TTP->getTemplateParameters()->containsUnexpandedParameterPack())
          ContainsUnexpandedParameterPack = true;

      // FIXME: If a default argument contains an unexpanded parameter pack, the
      // template parameter list does too.
    }
  }
}

TemplateParameterList *TemplateParameterList::Create(
    const ASTContext &C, SourceLocation TemplateLoc, SourceLocation LAngleLoc,
    ArrayRef<NamedDecl *> Params, SourceLocation RAngleLoc) {
  void *Mem = C.Allocate(totalSizeToAlloc<NamedDecl *>(Params.size()),
                         llvm::alignOf<TemplateParameterList>());
  return new (Mem) TemplateParameterList(TemplateLoc, LAngleLoc, Params,
                                         RAngleLoc);
}

unsigned TemplateParameterList::getMinRequiredArguments() const {
  unsigned NumRequiredArgs = 0;
  for (iterator P = const_cast<TemplateParameterList *>(this)->begin(), 
             PEnd = const_cast<TemplateParameterList *>(this)->end(); 
       P != PEnd; ++P) {
    if ((*P)->isTemplateParameterPack()) {
      if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(*P))
        if (NTTP->isExpandedParameterPack()) {
          NumRequiredArgs += NTTP->getNumExpansionTypes();
          continue;
        }
      
      break;
    }
  
    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*P)) {
      if (TTP->hasDefaultArgument())
        break;
    } else if (NonTypeTemplateParmDecl *NTTP 
                                    = dyn_cast<NonTypeTemplateParmDecl>(*P)) {
      if (NTTP->hasDefaultArgument())
        break;
    } else if (cast<TemplateTemplateParmDecl>(*P)->hasDefaultArgument())
      break;
    
    ++NumRequiredArgs;
  }
  
  return NumRequiredArgs;
}

unsigned TemplateParameterList::getDepth() const {
  if (size() == 0)
    return 0;
  
  const NamedDecl *FirstParm = getParam(0);
  if (const TemplateTypeParmDecl *TTP
        = dyn_cast<TemplateTypeParmDecl>(FirstParm))
    return TTP->getDepth();
  else if (const NonTypeTemplateParmDecl *NTTP 
             = dyn_cast<NonTypeTemplateParmDecl>(FirstParm))
    return NTTP->getDepth();
  else
    return cast<TemplateTemplateParmDecl>(FirstParm)->getDepth();
}

static void AdoptTemplateParameterList(TemplateParameterList *Params,
                                       DeclContext *Owner) {
  for (TemplateParameterList::iterator P = Params->begin(), 
                                    PEnd = Params->end();
       P != PEnd; ++P) {
    (*P)->setDeclContext(Owner);
    
    if (TemplateTemplateParmDecl *TTP = dyn_cast<TemplateTemplateParmDecl>(*P))
      AdoptTemplateParameterList(TTP->getTemplateParameters(), Owner);
  }
}

namespace clang {
void *allocateDefaultArgStorageChain(const ASTContext &C) {
  return new (C) char[sizeof(void*) * 2];
}
}

//===----------------------------------------------------------------------===//
// RedeclarableTemplateDecl Implementation
//===----------------------------------------------------------------------===//

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
  for (unsigned I = 0, N = PrevDecls.size(); I != N; ++I)
    PrevDecls[I]->Common = Common;

  return Common;
}

template<class EntryType>
typename RedeclarableTemplateDecl::SpecEntryTraits<EntryType>::DeclType *
RedeclarableTemplateDecl::findSpecializationImpl(
    llvm::FoldingSetVector<EntryType> &Specs, ArrayRef<TemplateArgument> Args,
    void *&InsertPos) {
  typedef SpecEntryTraits<EntryType> SETraits;
  llvm::FoldingSetNodeID ID;
  EntryType::Profile(ID,Args, getASTContext());
  EntryType *Entry = Specs.FindNodeOrInsertPos(ID, InsertPos);
  return Entry ? SETraits::getDecl(Entry)->getMostRecentDecl() : nullptr;
}

template<class Derived, class EntryType>
void RedeclarableTemplateDecl::addSpecializationImpl(
    llvm::FoldingSetVector<EntryType> &Specializations, EntryType *Entry,
    void *InsertPos) {
  typedef SpecEntryTraits<EntryType> SETraits;
  if (InsertPos) {
#ifndef NDEBUG
    void *CorrectInsertPos;
    assert(!findSpecializationImpl(Specializations,
                                   SETraits::getTemplateArgs(Entry),
                                   CorrectInsertPos) &&
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

/// \brief Generate the injected template arguments for the given template
/// parameter list, e.g., for the injected-class-name of a class template.
static void GenerateInjectedTemplateArgs(ASTContext &Context,
                                        TemplateParameterList *Params,
                                         TemplateArgument *Args) {
  for (TemplateParameterList::iterator Param = Params->begin(),
                                    ParamEnd = Params->end();
       Param != ParamEnd; ++Param) {
    TemplateArgument Arg;
    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*Param)) {
      QualType ArgType = Context.getTypeDeclType(TTP);
      if (TTP->isParameterPack())
        ArgType = Context.getPackExpansionType(ArgType, None);

      Arg = TemplateArgument(ArgType);
    } else if (NonTypeTemplateParmDecl *NTTP =
               dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
      Expr *E = new (Context) DeclRefExpr(NTTP, /*enclosing*/ false,
                                  NTTP->getType().getNonLValueExprType(Context),
                                  Expr::getValueKindForType(NTTP->getType()),
                                          NTTP->getLocation());
      
      if (NTTP->isParameterPack())
        E = new (Context) PackExpansionExpr(Context.DependentTy, E,
                                            NTTP->getLocation(), None);
      Arg = TemplateArgument(E);
    } else {
      TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(*Param);
      if (TTP->isParameterPack())
        Arg = TemplateArgument(TemplateName(TTP), Optional<unsigned>());
      else
        Arg = TemplateArgument(TemplateName(TTP));
    }
    
    if ((*Param)->isTemplateParameterPack())
      Arg = TemplateArgument::CreatePackCopy(Context, Arg);

    *Args++ = Arg;
  }
}
                                      
//===----------------------------------------------------------------------===//
// FunctionTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void FunctionTemplateDecl::DeallocateCommon(void *Ptr) {
  static_cast<Common *>(Ptr)->~Common();
}

FunctionTemplateDecl *FunctionTemplateDecl::Create(ASTContext &C,
                                                   DeclContext *DC,
                                                   SourceLocation L,
                                                   DeclarationName Name,
                                               TemplateParameterList *Params,
                                                   NamedDecl *Decl) {
  AdoptTemplateParameterList(Params, cast<DeclContext>(Decl));
  return new (C, DC) FunctionTemplateDecl(C, DC, L, Name, Params, Decl);
}

FunctionTemplateDecl *FunctionTemplateDecl::CreateDeserialized(ASTContext &C,
                                                               unsigned ID) {
  return new (C, ID) FunctionTemplateDecl(C, nullptr, SourceLocation(),
                                          DeclarationName(), nullptr, nullptr);
}

RedeclarableTemplateDecl::CommonBase *
FunctionTemplateDecl::newCommon(ASTContext &C) const {
  Common *CommonPtr = new (C) Common;
  C.AddDeallocation(DeallocateCommon, CommonPtr);
  return CommonPtr;
}

void FunctionTemplateDecl::LoadLazySpecializations() const {
  // Grab the most recent declaration to ensure we've loaded any lazy
  // redeclarations of this template.
  //
  // FIXME: Avoid walking the entire redeclaration chain here.
  Common *CommonPtr = getMostRecentDecl()->getCommonPtr();
  if (CommonPtr->LazySpecializations) {
    ASTContext &Context = getASTContext();
    uint32_t *Specs = CommonPtr->LazySpecializations;
    CommonPtr->LazySpecializations = nullptr;
    for (uint32_t I = 0, N = *Specs++; I != N; ++I)
      (void)Context.getExternalSource()->GetExternalDecl(Specs[I]);
  }
}

llvm::FoldingSetVector<FunctionTemplateSpecializationInfo> &
FunctionTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

FunctionDecl *
FunctionTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                         void *&InsertPos) {
  return findSpecializationImpl(getSpecializations(), Args, InsertPos);
}

void FunctionTemplateDecl::addSpecialization(
      FunctionTemplateSpecializationInfo *Info, void *InsertPos) {
  addSpecializationImpl<FunctionTemplateDecl>(getSpecializations(), Info,
                                              InsertPos);
}

ArrayRef<TemplateArgument> FunctionTemplateDecl::getInjectedTemplateArgs() {
  TemplateParameterList *Params = getTemplateParameters();
  Common *CommonPtr = getCommonPtr();
  if (!CommonPtr->InjectedArgs) {
    CommonPtr->InjectedArgs
      = new (getASTContext()) TemplateArgument[Params->size()];
    GenerateInjectedTemplateArgs(getASTContext(), Params,
                                 CommonPtr->InjectedArgs);
  }

  return llvm::makeArrayRef(CommonPtr->InjectedArgs, Params->size());
}

//===----------------------------------------------------------------------===//
// ClassTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void ClassTemplateDecl::DeallocateCommon(void *Ptr) {
  static_cast<Common *>(Ptr)->~Common();
}

ClassTemplateDecl *ClassTemplateDecl::Create(ASTContext &C,
                                             DeclContext *DC,
                                             SourceLocation L,
                                             DeclarationName Name,
                                             TemplateParameterList *Params,
                                             NamedDecl *Decl,
                                             ClassTemplateDecl *PrevDecl) {
  AdoptTemplateParameterList(Params, cast<DeclContext>(Decl));
  ClassTemplateDecl *New = new (C, DC) ClassTemplateDecl(C, DC, L, Name,
                                                         Params, Decl);
  New->setPreviousDecl(PrevDecl);
  return New;
}

ClassTemplateDecl *ClassTemplateDecl::CreateDeserialized(ASTContext &C,
                                                         unsigned ID) {
  return new (C, ID) ClassTemplateDecl(C, nullptr, SourceLocation(),
                                       DeclarationName(), nullptr, nullptr);
}

void ClassTemplateDecl::LoadLazySpecializations() const {
  // Grab the most recent declaration to ensure we've loaded any lazy
  // redeclarations of this template.
  //
  // FIXME: Avoid walking the entire redeclaration chain here.
  Common *CommonPtr = getMostRecentDecl()->getCommonPtr();
  if (CommonPtr->LazySpecializations) {
    ASTContext &Context = getASTContext();
    uint32_t *Specs = CommonPtr->LazySpecializations;
    CommonPtr->LazySpecializations = nullptr;
    for (uint32_t I = 0, N = *Specs++; I != N; ++I)
      (void)Context.getExternalSource()->GetExternalDecl(Specs[I]);
  }
}

llvm::FoldingSetVector<ClassTemplateSpecializationDecl> &
ClassTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}  

llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl> &
ClassTemplateDecl::getPartialSpecializations() {
  LoadLazySpecializations();
  return getCommonPtr()->PartialSpecializations;
}  

RedeclarableTemplateDecl::CommonBase *
ClassTemplateDecl::newCommon(ASTContext &C) const {
  Common *CommonPtr = new (C) Common;
  C.AddDeallocation(DeallocateCommon, CommonPtr);
  return CommonPtr;
}

ClassTemplateSpecializationDecl *
ClassTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                      void *&InsertPos) {
  return findSpecializationImpl(getSpecializations(), Args, InsertPos);
}

void ClassTemplateDecl::AddSpecialization(ClassTemplateSpecializationDecl *D,
                                          void *InsertPos) {
  addSpecializationImpl<ClassTemplateDecl>(getSpecializations(), D, InsertPos);
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecialization(ArrayRef<TemplateArgument> Args,
                                             void *&InsertPos) {
  return findSpecializationImpl(getPartialSpecializations(), Args, InsertPos);
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
          SmallVectorImpl<ClassTemplatePartialSpecializationDecl *> &PS) {
  llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl> &PartialSpecs
    = getPartialSpecializations();
  PS.clear();
  PS.reserve(PartialSpecs.size());
  for (llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl>::iterator
       P = PartialSpecs.begin(), PEnd = PartialSpecs.end();
       P != PEnd; ++P)
    PS.push_back(P->getMostRecentDecl());
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecialization(QualType T) {
  ASTContext &Context = getASTContext();
  using llvm::FoldingSetVector;
  typedef FoldingSetVector<ClassTemplatePartialSpecializationDecl>::iterator
    partial_spec_iterator;
  for (partial_spec_iterator P = getPartialSpecializations().begin(),
                          PEnd = getPartialSpecializations().end();
       P != PEnd; ++P) {
    if (Context.hasSameType(P->getInjectedSpecializationType(), T))
      return P->getMostRecentDecl();
  }

  return nullptr;
}

ClassTemplatePartialSpecializationDecl *
ClassTemplateDecl::findPartialSpecInstantiatedFromMember(
                                    ClassTemplatePartialSpecializationDecl *D) {
  Decl *DCanon = D->getCanonicalDecl();
  for (llvm::FoldingSetVector<ClassTemplatePartialSpecializationDecl>::iterator
            P = getPartialSpecializations().begin(),
         PEnd = getPartialSpecializations().end();
       P != PEnd; ++P) {
    if (P->getInstantiatedFromMember()->getCanonicalDecl() == DCanon)
      return P->getMostRecentDecl();
  }

  return nullptr;
}

QualType
ClassTemplateDecl::getInjectedClassNameSpecialization() {
  Common *CommonPtr = getCommonPtr();
  if (!CommonPtr->InjectedClassNameType.isNull())
    return CommonPtr->InjectedClassNameType;

  // C++0x [temp.dep.type]p2:
  //  The template argument list of a primary template is a template argument 
  //  list in which the nth template argument has the value of the nth template
  //  parameter of the class template. If the nth template parameter is a 
  //  template parameter pack (14.5.3), the nth template argument is a pack 
  //  expansion (14.5.3) whose pattern is the name of the template parameter 
  //  pack.
  ASTContext &Context = getASTContext();
  TemplateParameterList *Params = getTemplateParameters();
  SmallVector<TemplateArgument, 16> TemplateArgs;
  TemplateArgs.resize(Params->size());
  GenerateInjectedTemplateArgs(getASTContext(), Params, TemplateArgs.data());
  CommonPtr->InjectedClassNameType
    = Context.getTemplateSpecializationType(TemplateName(this),
                                            &TemplateArgs[0],
                                            TemplateArgs.size());
  return CommonPtr->InjectedClassNameType;
}

//===----------------------------------------------------------------------===//
// TemplateTypeParm Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

TemplateTypeParmDecl *
TemplateTypeParmDecl::Create(const ASTContext &C, DeclContext *DC,
                             SourceLocation KeyLoc, SourceLocation NameLoc,
                             unsigned D, unsigned P, IdentifierInfo *Id,
                             bool Typename, bool ParameterPack) {
  TemplateTypeParmDecl *TTPDecl =
    new (C, DC) TemplateTypeParmDecl(DC, KeyLoc, NameLoc, Id, Typename);
  QualType TTPType = C.getTemplateTypeParmType(D, P, ParameterPack, TTPDecl);
  TTPDecl->setTypeForDecl(TTPType.getTypePtr());
  return TTPDecl;
}

TemplateTypeParmDecl *
TemplateTypeParmDecl::CreateDeserialized(const ASTContext &C, unsigned ID) {
  return new (C, ID) TemplateTypeParmDecl(nullptr, SourceLocation(),
                                          SourceLocation(), nullptr, false);
}

SourceLocation TemplateTypeParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument()
             ? getDefaultArgumentInfo()->getTypeLoc().getBeginLoc()
             : SourceLocation();
}

SourceRange TemplateTypeParmDecl::getSourceRange() const {
  if (hasDefaultArgument() && !defaultArgumentWasInherited())
    return SourceRange(getLocStart(),
                       getDefaultArgumentInfo()->getTypeLoc().getEndLoc());
  else
    return TypeDecl::getSourceRange();
}

unsigned TemplateTypeParmDecl::getDepth() const {
  return getTypeForDecl()->getAs<TemplateTypeParmType>()->getDepth();
}

unsigned TemplateTypeParmDecl::getIndex() const {
  return getTypeForDecl()->getAs<TemplateTypeParmType>()->getIndex();
}

bool TemplateTypeParmDecl::isParameterPack() const {
  return getTypeForDecl()->getAs<TemplateTypeParmType>()->isParameterPack();
}

//===----------------------------------------------------------------------===//
// NonTypeTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

NonTypeTemplateParmDecl::NonTypeTemplateParmDecl(DeclContext *DC, 
                                                 SourceLocation StartLoc,
                                                 SourceLocation IdLoc,
                                                 unsigned D, unsigned P,
                                                 IdentifierInfo *Id, 
                                                 QualType T, 
                                                 TypeSourceInfo *TInfo,
                                                 const QualType *ExpandedTypes,
                                                 unsigned NumExpandedTypes,
                                                TypeSourceInfo **ExpandedTInfos)
  : DeclaratorDecl(NonTypeTemplateParm, DC, IdLoc, Id, T, TInfo, StartLoc),
    TemplateParmPosition(D, P), ParameterPack(true),
    ExpandedParameterPack(true), NumExpandedTypes(NumExpandedTypes) {
  if (ExpandedTypes && ExpandedTInfos) {
    auto TypesAndInfos =
        getTrailingObjects<std::pair<QualType, TypeSourceInfo *>>();
    for (unsigned I = 0; I != NumExpandedTypes; ++I) {
      new (&TypesAndInfos[I].first) QualType(ExpandedTypes[I]);
      TypesAndInfos[I].second = ExpandedTInfos[I];
    }
  }
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC,
                                SourceLocation StartLoc, SourceLocation IdLoc,
                                unsigned D, unsigned P, IdentifierInfo *Id,
                                QualType T, bool ParameterPack,
                                TypeSourceInfo *TInfo) {
  return new (C, DC) NonTypeTemplateParmDecl(DC, StartLoc, IdLoc, D, P, Id,
                                             T, ParameterPack, TInfo);
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC, 
                                SourceLocation StartLoc, SourceLocation IdLoc,
                                unsigned D, unsigned P, 
                                IdentifierInfo *Id, QualType T, 
                                TypeSourceInfo *TInfo,
                                const QualType *ExpandedTypes, 
                                unsigned NumExpandedTypes,
                                TypeSourceInfo **ExpandedTInfos) {
  return new (C, DC,
              additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>>(
                  NumExpandedTypes))
      NonTypeTemplateParmDecl(DC, StartLoc, IdLoc, D, P, Id, T, TInfo,
                              ExpandedTypes, NumExpandedTypes, ExpandedTInfos);
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID) NonTypeTemplateParmDecl(nullptr, SourceLocation(),
                                             SourceLocation(), 0, 0, nullptr,
                                             QualType(), false, nullptr);
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID,
                                            unsigned NumExpandedTypes) {
  return new (C, ID,
              additionalSizeToAlloc<std::pair<QualType, TypeSourceInfo *>>(
                  NumExpandedTypes))
      NonTypeTemplateParmDecl(nullptr, SourceLocation(), SourceLocation(), 0, 0,
                              nullptr, QualType(), nullptr, nullptr,
                              NumExpandedTypes, nullptr);
}

SourceRange NonTypeTemplateParmDecl::getSourceRange() const {
  if (hasDefaultArgument() && !defaultArgumentWasInherited())
    return SourceRange(getOuterLocStart(),
                       getDefaultArgument()->getSourceRange().getEnd());
  return DeclaratorDecl::getSourceRange();
}

SourceLocation NonTypeTemplateParmDecl::getDefaultArgumentLoc() const {
  return hasDefaultArgument()
    ? getDefaultArgument()->getSourceRange().getBegin()
    : SourceLocation();
}

//===----------------------------------------------------------------------===//
// TemplateTemplateParmDecl Method Implementations
//===----------------------------------------------------------------------===//

void TemplateTemplateParmDecl::anchor() { }

TemplateTemplateParmDecl::TemplateTemplateParmDecl(
    DeclContext *DC, SourceLocation L, unsigned D, unsigned P,
    IdentifierInfo *Id, TemplateParameterList *Params,
    unsigned NumExpansions, TemplateParameterList * const *Expansions)
  : TemplateDecl(TemplateTemplateParm, DC, L, Id, Params),
    TemplateParmPosition(D, P), ParameterPack(true),
    ExpandedParameterPack(true), NumExpandedParams(NumExpansions) {
  if (Expansions)
    std::uninitialized_copy(Expansions, Expansions + NumExpandedParams,
                            getTrailingObjects<TemplateParameterList *>());
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC,
                                 SourceLocation L, unsigned D, unsigned P,
                                 bool ParameterPack, IdentifierInfo *Id,
                                 TemplateParameterList *Params) {
  return new (C, DC) TemplateTemplateParmDecl(DC, L, D, P, ParameterPack, Id,
                                              Params);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::Create(const ASTContext &C, DeclContext *DC,
                                 SourceLocation L, unsigned D, unsigned P,
                                 IdentifierInfo *Id,
                                 TemplateParameterList *Params,
                                 ArrayRef<TemplateParameterList *> Expansions) {
  return new (C, DC,
              additionalSizeToAlloc<TemplateParameterList *>(Expansions.size()))
      TemplateTemplateParmDecl(DC, L, D, P, Id, Params, Expansions.size(),
                               Expansions.data());
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID) TemplateTemplateParmDecl(nullptr, SourceLocation(), 0, 0,
                                              false, nullptr, nullptr);
}

TemplateTemplateParmDecl *
TemplateTemplateParmDecl::CreateDeserialized(ASTContext &C, unsigned ID,
                                             unsigned NumExpansions) {
  return new (C, ID,
              additionalSizeToAlloc<TemplateParameterList *>(NumExpansions))
      TemplateTemplateParmDecl(nullptr, SourceLocation(), 0, 0, nullptr,
                               nullptr, NumExpansions, nullptr);
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
TemplateArgumentList::TemplateArgumentList(const TemplateArgument *Args,
                                           unsigned NumArgs)
    : Arguments(getTrailingObjects<TemplateArgument>()), NumArguments(NumArgs) {
  std::uninitialized_copy(Args, Args + NumArgs,
                          getTrailingObjects<TemplateArgument>());
}

TemplateArgumentList *
TemplateArgumentList::CreateCopy(ASTContext &Context,
                                 const TemplateArgument *Args,
                                 unsigned NumArgs) {
  void *Mem = Context.Allocate(totalSizeToAlloc<TemplateArgument>(NumArgs));
  return new (Mem) TemplateArgumentList(Args, NumArgs);
}

FunctionTemplateSpecializationInfo *
FunctionTemplateSpecializationInfo::Create(ASTContext &C, FunctionDecl *FD,
                                           FunctionTemplateDecl *Template,
                                           TemplateSpecializationKind TSK,
                                       const TemplateArgumentList *TemplateArgs,
                          const TemplateArgumentListInfo *TemplateArgsAsWritten,
                                           SourceLocation POI) {
  const ASTTemplateArgumentListInfo *ArgsAsWritten = nullptr;
  if (TemplateArgsAsWritten)
    ArgsAsWritten = ASTTemplateArgumentListInfo::Create(C,
                                                        *TemplateArgsAsWritten);

  return new (C) FunctionTemplateSpecializationInfo(FD, Template, TSK,
                                                    TemplateArgs,
                                                    ArgsAsWritten,
                                                    POI);
}

//===----------------------------------------------------------------------===//
// TemplateDecl Implementation
//===----------------------------------------------------------------------===//

void TemplateDecl::anchor() { }

//===----------------------------------------------------------------------===//
// ClassTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
ClassTemplateSpecializationDecl::
ClassTemplateSpecializationDecl(ASTContext &Context, Kind DK, TagKind TK,
                                DeclContext *DC, SourceLocation StartLoc,
                                SourceLocation IdLoc,
                                ClassTemplateDecl *SpecializedTemplate,
                                const TemplateArgument *Args,
                                unsigned NumArgs,
                                ClassTemplateSpecializationDecl *PrevDecl)
  : CXXRecordDecl(DK, TK, Context, DC, StartLoc, IdLoc,
                  SpecializedTemplate->getIdentifier(),
                  PrevDecl),
    SpecializedTemplate(SpecializedTemplate),
    ExplicitInfo(nullptr),
    TemplateArgs(TemplateArgumentList::CreateCopy(Context, Args, NumArgs)),
    SpecializationKind(TSK_Undeclared) {
}

ClassTemplateSpecializationDecl::ClassTemplateSpecializationDecl(ASTContext &C,
                                                                 Kind DK)
    : CXXRecordDecl(DK, TTK_Struct, C, nullptr, SourceLocation(),
                    SourceLocation(), nullptr, nullptr),
      ExplicitInfo(nullptr), SpecializationKind(TSK_Undeclared) {}

ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::Create(ASTContext &Context, TagKind TK,
                                        DeclContext *DC,
                                        SourceLocation StartLoc,
                                        SourceLocation IdLoc,
                                        ClassTemplateDecl *SpecializedTemplate,
                                        const TemplateArgument *Args,
                                        unsigned NumArgs,
                                   ClassTemplateSpecializationDecl *PrevDecl) {
  ClassTemplateSpecializationDecl *Result =
      new (Context, DC) ClassTemplateSpecializationDecl(
          Context, ClassTemplateSpecialization, TK, DC, StartLoc, IdLoc,
          SpecializedTemplate, Args, NumArgs, PrevDecl);
  Result->MayHaveOutOfDateDef = false;

  Context.getTypeDeclType(Result, PrevDecl);
  return Result;
}

ClassTemplateSpecializationDecl *
ClassTemplateSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                    unsigned ID) {
  ClassTemplateSpecializationDecl *Result =
    new (C, ID) ClassTemplateSpecializationDecl(C, ClassTemplateSpecialization);
  Result->MayHaveOutOfDateDef = false;
  return Result;
}

void ClassTemplateSpecializationDecl::getNameForDiagnostic(
    raw_ostream &OS, const PrintingPolicy &Policy, bool Qualified) const {
  NamedDecl::getNameForDiagnostic(OS, Policy, Qualified);

  const TemplateArgumentList &TemplateArgs = getTemplateArgs();
  TemplateSpecializationType::PrintTemplateArgumentList(
      OS, TemplateArgs.data(), TemplateArgs.size(), Policy);
}

ClassTemplateDecl *
ClassTemplateSpecializationDecl::getSpecializedTemplate() const {
  if (SpecializedPartialSpecialization *PartialSpec
      = SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization*>())
    return PartialSpec->PartialSpecialization->getSpecializedTemplate();
  return SpecializedTemplate.get<ClassTemplateDecl*>();
}

SourceRange
ClassTemplateSpecializationDecl::getSourceRange() const {
  if (ExplicitInfo) {
    SourceLocation Begin = getTemplateKeywordLoc();
    if (Begin.isValid()) {
      // Here we have an explicit (partial) specialization or instantiation.
      assert(getSpecializationKind() == TSK_ExplicitSpecialization ||
             getSpecializationKind() == TSK_ExplicitInstantiationDeclaration ||
             getSpecializationKind() == TSK_ExplicitInstantiationDefinition);
      if (getExternLoc().isValid())
        Begin = getExternLoc();
      SourceLocation End = getRBraceLoc();
      if (End.isInvalid())
        End = getTypeAsWritten()->getTypeLoc().getEndLoc();
      return SourceRange(Begin, End);
    }
    // An implicit instantiation of a class template partial specialization
    // uses ExplicitInfo to record the TypeAsWritten, but the source
    // locations should be retrieved from the instantiation pattern.
    typedef ClassTemplatePartialSpecializationDecl CTPSDecl;
    CTPSDecl *ctpsd = const_cast<CTPSDecl*>(cast<CTPSDecl>(this));
    CTPSDecl *inst_from = ctpsd->getInstantiatedFromMember();
    assert(inst_from != nullptr);
    return inst_from->getSourceRange();
  }
  else {
    // No explicit info available.
    llvm::PointerUnion<ClassTemplateDecl *,
                       ClassTemplatePartialSpecializationDecl *>
      inst_from = getInstantiatedFrom();
    if (inst_from.isNull())
      return getSpecializedTemplate()->getSourceRange();
    if (ClassTemplateDecl *ctd = inst_from.dyn_cast<ClassTemplateDecl*>())
      return ctd->getSourceRange();
    return inst_from.get<ClassTemplatePartialSpecializationDecl*>()
      ->getSourceRange();
  }
}

//===----------------------------------------------------------------------===//
// ClassTemplatePartialSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
void ClassTemplatePartialSpecializationDecl::anchor() { }

ClassTemplatePartialSpecializationDecl::
ClassTemplatePartialSpecializationDecl(ASTContext &Context, TagKind TK,
                                       DeclContext *DC,
                                       SourceLocation StartLoc,
                                       SourceLocation IdLoc,
                                       TemplateParameterList *Params,
                                       ClassTemplateDecl *SpecializedTemplate,
                                       const TemplateArgument *Args,
                                       unsigned NumArgs,
                               const ASTTemplateArgumentListInfo *ArgInfos,
                               ClassTemplatePartialSpecializationDecl *PrevDecl)
  : ClassTemplateSpecializationDecl(Context,
                                    ClassTemplatePartialSpecialization,
                                    TK, DC, StartLoc, IdLoc,
                                    SpecializedTemplate,
                                    Args, NumArgs, PrevDecl),
    TemplateParams(Params), ArgsAsWritten(ArgInfos),
    InstantiatedFromMember(nullptr, false)
{
  AdoptTemplateParameterList(Params, this);
}

ClassTemplatePartialSpecializationDecl *
ClassTemplatePartialSpecializationDecl::
Create(ASTContext &Context, TagKind TK,DeclContext *DC,
       SourceLocation StartLoc, SourceLocation IdLoc,
       TemplateParameterList *Params,
       ClassTemplateDecl *SpecializedTemplate,
       const TemplateArgument *Args,
       unsigned NumArgs,
       const TemplateArgumentListInfo &ArgInfos,
       QualType CanonInjectedType,
       ClassTemplatePartialSpecializationDecl *PrevDecl) {
  const ASTTemplateArgumentListInfo *ASTArgInfos =
    ASTTemplateArgumentListInfo::Create(Context, ArgInfos);

  ClassTemplatePartialSpecializationDecl *Result = new (Context, DC)
      ClassTemplatePartialSpecializationDecl(Context, TK, DC, StartLoc, IdLoc,
                                             Params, SpecializedTemplate, Args,
                                             NumArgs, ASTArgInfos, PrevDecl);
  Result->setSpecializationKind(TSK_ExplicitSpecialization);
  Result->MayHaveOutOfDateDef = false;

  Context.getInjectedClassNameType(Result, CanonInjectedType);
  return Result;
}

ClassTemplatePartialSpecializationDecl *
ClassTemplatePartialSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                           unsigned ID) {
  ClassTemplatePartialSpecializationDecl *Result =
      new (C, ID) ClassTemplatePartialSpecializationDecl(C);
  Result->MayHaveOutOfDateDef = false;
  return Result;
}

//===----------------------------------------------------------------------===//
// FriendTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void FriendTemplateDecl::anchor() { }

FriendTemplateDecl *FriendTemplateDecl::Create(ASTContext &Context,
                                               DeclContext *DC,
                                               SourceLocation L,
                                               unsigned NParams,
                                               TemplateParameterList **Params,
                                               FriendUnion Friend,
                                               SourceLocation FLoc) {
  return new (Context, DC) FriendTemplateDecl(DC, L, NParams, Params,
                                              Friend, FLoc);
}

FriendTemplateDecl *FriendTemplateDecl::CreateDeserialized(ASTContext &C,
                                                           unsigned ID) {
  return new (C, ID) FriendTemplateDecl(EmptyShell());
}

//===----------------------------------------------------------------------===//
// TypeAliasTemplateDecl Implementation
//===----------------------------------------------------------------------===//

TypeAliasTemplateDecl *TypeAliasTemplateDecl::Create(ASTContext &C,
                                                     DeclContext *DC,
                                                     SourceLocation L,
                                                     DeclarationName Name,
                                                  TemplateParameterList *Params,
                                                     NamedDecl *Decl) {
  AdoptTemplateParameterList(Params, DC);
  return new (C, DC) TypeAliasTemplateDecl(C, DC, L, Name, Params, Decl);
}

TypeAliasTemplateDecl *TypeAliasTemplateDecl::CreateDeserialized(ASTContext &C,
                                                                 unsigned ID) {
  return new (C, ID) TypeAliasTemplateDecl(C, nullptr, SourceLocation(),
                                           DeclarationName(), nullptr, nullptr);
}

void TypeAliasTemplateDecl::DeallocateCommon(void *Ptr) {
  static_cast<Common *>(Ptr)->~Common();
}
RedeclarableTemplateDecl::CommonBase *
TypeAliasTemplateDecl::newCommon(ASTContext &C) const {
  Common *CommonPtr = new (C) Common;
  C.AddDeallocation(DeallocateCommon, CommonPtr);
  return CommonPtr;
}

//===----------------------------------------------------------------------===//
// ClassScopeFunctionSpecializationDecl Implementation
//===----------------------------------------------------------------------===//

void ClassScopeFunctionSpecializationDecl::anchor() { }

ClassScopeFunctionSpecializationDecl *
ClassScopeFunctionSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                         unsigned ID) {
  return new (C, ID) ClassScopeFunctionSpecializationDecl(
      nullptr, SourceLocation(), nullptr, false, TemplateArgumentListInfo());
}

//===----------------------------------------------------------------------===//
// VarTemplateDecl Implementation
//===----------------------------------------------------------------------===//

void VarTemplateDecl::DeallocateCommon(void *Ptr) {
  static_cast<Common *>(Ptr)->~Common();
}

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
  return new (C, DC) VarTemplateDecl(C, DC, L, Name, Params, Decl);
}

VarTemplateDecl *VarTemplateDecl::CreateDeserialized(ASTContext &C,
                                                     unsigned ID) {
  return new (C, ID) VarTemplateDecl(C, nullptr, SourceLocation(),
                                     DeclarationName(), nullptr, nullptr);
}

// TODO: Unify across class, function and variable templates?
//       May require moving this and Common to RedeclarableTemplateDecl.
void VarTemplateDecl::LoadLazySpecializations() const {
  // Grab the most recent declaration to ensure we've loaded any lazy
  // redeclarations of this template.
  //
  // FIXME: Avoid walking the entire redeclaration chain here.
  Common *CommonPtr = getMostRecentDecl()->getCommonPtr();
  if (CommonPtr->LazySpecializations) {
    ASTContext &Context = getASTContext();
    uint32_t *Specs = CommonPtr->LazySpecializations;
    CommonPtr->LazySpecializations = nullptr;
    for (uint32_t I = 0, N = *Specs++; I != N; ++I)
      (void)Context.getExternalSource()->GetExternalDecl(Specs[I]);
  }
}

llvm::FoldingSetVector<VarTemplateSpecializationDecl> &
VarTemplateDecl::getSpecializations() const {
  LoadLazySpecializations();
  return getCommonPtr()->Specializations;
}

llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl> &
VarTemplateDecl::getPartialSpecializations() {
  LoadLazySpecializations();
  return getCommonPtr()->PartialSpecializations;
}

RedeclarableTemplateDecl::CommonBase *
VarTemplateDecl::newCommon(ASTContext &C) const {
  Common *CommonPtr = new (C) Common;
  C.AddDeallocation(DeallocateCommon, CommonPtr);
  return CommonPtr;
}

VarTemplateSpecializationDecl *
VarTemplateDecl::findSpecialization(ArrayRef<TemplateArgument> Args,
                                    void *&InsertPos) {
  return findSpecializationImpl(getSpecializations(), Args, InsertPos);
}

void VarTemplateDecl::AddSpecialization(VarTemplateSpecializationDecl *D,
                                        void *InsertPos) {
  addSpecializationImpl<VarTemplateDecl>(getSpecializations(), D, InsertPos);
}

VarTemplatePartialSpecializationDecl *
VarTemplateDecl::findPartialSpecialization(ArrayRef<TemplateArgument> Args,
                                           void *&InsertPos) {
  return findSpecializationImpl(getPartialSpecializations(), Args, InsertPos);
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
    SmallVectorImpl<VarTemplatePartialSpecializationDecl *> &PS) {
  llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl> &PartialSpecs =
      getPartialSpecializations();
  PS.clear();
  PS.reserve(PartialSpecs.size());
  for (llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl>::iterator
           P = PartialSpecs.begin(),
           PEnd = PartialSpecs.end();
       P != PEnd; ++P)
    PS.push_back(P->getMostRecentDecl());
}

VarTemplatePartialSpecializationDecl *
VarTemplateDecl::findPartialSpecInstantiatedFromMember(
    VarTemplatePartialSpecializationDecl *D) {
  Decl *DCanon = D->getCanonicalDecl();
  for (llvm::FoldingSetVector<VarTemplatePartialSpecializationDecl>::iterator
           P = getPartialSpecializations().begin(),
           PEnd = getPartialSpecializations().end();
       P != PEnd; ++P) {
    if (P->getInstantiatedFromMember()->getCanonicalDecl() == DCanon)
      return P->getMostRecentDecl();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// VarTemplateSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
VarTemplateSpecializationDecl::VarTemplateSpecializationDecl(
    Kind DK, ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, VarTemplateDecl *SpecializedTemplate, QualType T,
    TypeSourceInfo *TInfo, StorageClass S, const TemplateArgument *Args,
    unsigned NumArgs)
    : VarDecl(DK, Context, DC, StartLoc, IdLoc,
              SpecializedTemplate->getIdentifier(), T, TInfo, S),
      SpecializedTemplate(SpecializedTemplate), ExplicitInfo(nullptr),
      TemplateArgs(TemplateArgumentList::CreateCopy(Context, Args, NumArgs)),
      SpecializationKind(TSK_Undeclared) {}

VarTemplateSpecializationDecl::VarTemplateSpecializationDecl(Kind DK,
                                                             ASTContext &C)
    : VarDecl(DK, C, nullptr, SourceLocation(), SourceLocation(), nullptr,
              QualType(), nullptr, SC_None),
      ExplicitInfo(nullptr), SpecializationKind(TSK_Undeclared) {}

VarTemplateSpecializationDecl *VarTemplateSpecializationDecl::Create(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, VarTemplateDecl *SpecializedTemplate, QualType T,
    TypeSourceInfo *TInfo, StorageClass S, const TemplateArgument *Args,
    unsigned NumArgs) {
  return new (Context, DC) VarTemplateSpecializationDecl(
      VarTemplateSpecialization, Context, DC, StartLoc, IdLoc,
      SpecializedTemplate, T, TInfo, S, Args, NumArgs);
}

VarTemplateSpecializationDecl *
VarTemplateSpecializationDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID)
      VarTemplateSpecializationDecl(VarTemplateSpecialization, C);
}

void VarTemplateSpecializationDecl::getNameForDiagnostic(
    raw_ostream &OS, const PrintingPolicy &Policy, bool Qualified) const {
  NamedDecl::getNameForDiagnostic(OS, Policy, Qualified);

  const TemplateArgumentList &TemplateArgs = getTemplateArgs();
  TemplateSpecializationType::PrintTemplateArgumentList(
      OS, TemplateArgs.data(), TemplateArgs.size(), Policy);
}

VarTemplateDecl *VarTemplateSpecializationDecl::getSpecializedTemplate() const {
  if (SpecializedPartialSpecialization *PartialSpec =
          SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization *>())
    return PartialSpec->PartialSpecialization->getSpecializedTemplate();
  return SpecializedTemplate.get<VarTemplateDecl *>();
}

void VarTemplateSpecializationDecl::setTemplateArgsInfo(
    const TemplateArgumentListInfo &ArgsInfo) {
  unsigned N = ArgsInfo.size();
  TemplateArgsInfo.setLAngleLoc(ArgsInfo.getLAngleLoc());
  TemplateArgsInfo.setRAngleLoc(ArgsInfo.getRAngleLoc());
  for (unsigned I = 0; I != N; ++I)
    TemplateArgsInfo.addArgument(ArgsInfo[I]);
}

//===----------------------------------------------------------------------===//
// VarTemplatePartialSpecializationDecl Implementation
//===----------------------------------------------------------------------===//
void VarTemplatePartialSpecializationDecl::anchor() {}

VarTemplatePartialSpecializationDecl::VarTemplatePartialSpecializationDecl(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    VarTemplateDecl *SpecializedTemplate, QualType T, TypeSourceInfo *TInfo,
    StorageClass S, const TemplateArgument *Args, unsigned NumArgs,
    const ASTTemplateArgumentListInfo *ArgInfos)
    : VarTemplateSpecializationDecl(VarTemplatePartialSpecialization, Context,
                                    DC, StartLoc, IdLoc, SpecializedTemplate, T,
                                    TInfo, S, Args, NumArgs),
      TemplateParams(Params), ArgsAsWritten(ArgInfos),
      InstantiatedFromMember(nullptr, false) {
  // TODO: The template parameters should be in DC by now. Verify.
  // AdoptTemplateParameterList(Params, DC);
}

VarTemplatePartialSpecializationDecl *
VarTemplatePartialSpecializationDecl::Create(
    ASTContext &Context, DeclContext *DC, SourceLocation StartLoc,
    SourceLocation IdLoc, TemplateParameterList *Params,
    VarTemplateDecl *SpecializedTemplate, QualType T, TypeSourceInfo *TInfo,
    StorageClass S, const TemplateArgument *Args, unsigned NumArgs,
    const TemplateArgumentListInfo &ArgInfos) {
  const ASTTemplateArgumentListInfo *ASTArgInfos
    = ASTTemplateArgumentListInfo::Create(Context, ArgInfos);

  VarTemplatePartialSpecializationDecl *Result =
      new (Context, DC) VarTemplatePartialSpecializationDecl(
          Context, DC, StartLoc, IdLoc, Params, SpecializedTemplate, T, TInfo,
          S, Args, NumArgs, ASTArgInfos);
  Result->setSpecializationKind(TSK_ExplicitSpecialization);
  return Result;
}

VarTemplatePartialSpecializationDecl *
VarTemplatePartialSpecializationDecl::CreateDeserialized(ASTContext &C,
                                                         unsigned ID) {
  return new (C, ID) VarTemplatePartialSpecializationDecl(C);
}

static TemplateParameterList *
createMakeIntegerSeqParameterList(const ASTContext &C, DeclContext *DC) {
  // typename T
  auto *T = TemplateTypeParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/1, /*Position=*/0,
      /*Id=*/nullptr, /*Typename=*/true, /*ParameterPack=*/false);
  T->setImplicit(true);

  // T ...Ints
  TypeSourceInfo *TI =
      C.getTrivialTypeSourceInfo(QualType(T->getTypeForDecl(), 0));
  auto *N = NonTypeTemplateParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/1,
      /*Id=*/nullptr, TI->getType(), /*ParameterPack=*/true, TI);
  N->setImplicit(true);

  // <typename T, T ...Ints>
  NamedDecl *P[2] = {T, N};
  auto *TPL = TemplateParameterList::Create(
      C, SourceLocation(), SourceLocation(), P, SourceLocation());

  // template <typename T, ...Ints> class IntSeq
  auto *TemplateTemplateParm = TemplateTemplateParmDecl::Create(
      C, DC, SourceLocation(), /*Depth=*/0, /*Position=*/0,
      /*ParameterPack=*/false, /*Id=*/nullptr, TPL);
  TemplateTemplateParm->setImplicit(true);

  // typename T
  auto *TemplateTypeParm = TemplateTypeParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/1,
      /*Id=*/nullptr, /*Typename=*/true, /*ParameterPack=*/false);
  TemplateTypeParm->setImplicit(true);

  // T N
  TypeSourceInfo *TInfo = C.getTrivialTypeSourceInfo(
      QualType(TemplateTypeParm->getTypeForDecl(), 0));
  auto *NonTypeTemplateParm = NonTypeTemplateParmDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Depth=*/0, /*Position=*/2,
      /*Id=*/nullptr, TInfo->getType(), /*ParameterPack=*/false, TInfo);
  NamedDecl *Params[] = {TemplateTemplateParm, TemplateTypeParm,
                         NonTypeTemplateParm};

  // template <template <typename T, T ...Ints> class IntSeq, typename T, T N>
  return TemplateParameterList::Create(C, SourceLocation(), SourceLocation(),
                                       Params, SourceLocation());
}

static TemplateParameterList *createBuiltinTemplateParameterList(
    const ASTContext &C, DeclContext *DC, BuiltinTemplateKind BTK) {
  switch (BTK) {
  case BTK__make_integer_seq:
    return createMakeIntegerSeqParameterList(C, DC);
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
