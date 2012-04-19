//===--- ASTContext.cpp - Context to hold long-lived AST nodes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Capacity.h"
#include "CXXABI.h"
#include <map>

using namespace clang;

unsigned ASTContext::NumImplicitDefaultConstructors;
unsigned ASTContext::NumImplicitDefaultConstructorsDeclared;
unsigned ASTContext::NumImplicitCopyConstructors;
unsigned ASTContext::NumImplicitCopyConstructorsDeclared;
unsigned ASTContext::NumImplicitMoveConstructors;
unsigned ASTContext::NumImplicitMoveConstructorsDeclared;
unsigned ASTContext::NumImplicitCopyAssignmentOperators;
unsigned ASTContext::NumImplicitCopyAssignmentOperatorsDeclared;
unsigned ASTContext::NumImplicitMoveAssignmentOperators;
unsigned ASTContext::NumImplicitMoveAssignmentOperatorsDeclared;
unsigned ASTContext::NumImplicitDestructors;
unsigned ASTContext::NumImplicitDestructorsDeclared;

enum FloatingRank {
  HalfRank, FloatRank, DoubleRank, LongDoubleRank
};

void 
ASTContext::CanonicalTemplateTemplateParm::Profile(llvm::FoldingSetNodeID &ID, 
                                               TemplateTemplateParmDecl *Parm) {
  ID.AddInteger(Parm->getDepth());
  ID.AddInteger(Parm->getPosition());
  ID.AddBoolean(Parm->isParameterPack());

  TemplateParameterList *Params = Parm->getTemplateParameters();
  ID.AddInteger(Params->size());
  for (TemplateParameterList::const_iterator P = Params->begin(), 
                                          PEnd = Params->end();
       P != PEnd; ++P) {
    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*P)) {
      ID.AddInteger(0);
      ID.AddBoolean(TTP->isParameterPack());
      continue;
    }
    
    if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(*P)) {
      ID.AddInteger(1);
      ID.AddBoolean(NTTP->isParameterPack());
      ID.AddPointer(NTTP->getType().getCanonicalType().getAsOpaquePtr());
      if (NTTP->isExpandedParameterPack()) {
        ID.AddBoolean(true);
        ID.AddInteger(NTTP->getNumExpansionTypes());
        for (unsigned I = 0, N = NTTP->getNumExpansionTypes(); I != N; ++I) {
          QualType T = NTTP->getExpansionType(I);
          ID.AddPointer(T.getCanonicalType().getAsOpaquePtr());
        }
      } else 
        ID.AddBoolean(false);
      continue;
    }
    
    TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(*P);
    ID.AddInteger(2);
    Profile(ID, TTP);
  }
}

TemplateTemplateParmDecl *
ASTContext::getCanonicalTemplateTemplateParmDecl(
                                          TemplateTemplateParmDecl *TTP) const {
  // Check if we already have a canonical template template parameter.
  llvm::FoldingSetNodeID ID;
  CanonicalTemplateTemplateParm::Profile(ID, TTP);
  void *InsertPos = 0;
  CanonicalTemplateTemplateParm *Canonical
    = CanonTemplateTemplateParms.FindNodeOrInsertPos(ID, InsertPos);
  if (Canonical)
    return Canonical->getParam();
  
  // Build a canonical template parameter list.
  TemplateParameterList *Params = TTP->getTemplateParameters();
  SmallVector<NamedDecl *, 4> CanonParams;
  CanonParams.reserve(Params->size());
  for (TemplateParameterList::const_iterator P = Params->begin(), 
                                          PEnd = Params->end();
       P != PEnd; ++P) {
    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*P))
      CanonParams.push_back(
                  TemplateTypeParmDecl::Create(*this, getTranslationUnitDecl(), 
                                               SourceLocation(),
                                               SourceLocation(),
                                               TTP->getDepth(),
                                               TTP->getIndex(), 0, false,
                                               TTP->isParameterPack()));
    else if (NonTypeTemplateParmDecl *NTTP
             = dyn_cast<NonTypeTemplateParmDecl>(*P)) {
      QualType T = getCanonicalType(NTTP->getType());
      TypeSourceInfo *TInfo = getTrivialTypeSourceInfo(T);
      NonTypeTemplateParmDecl *Param;
      if (NTTP->isExpandedParameterPack()) {
        SmallVector<QualType, 2> ExpandedTypes;
        SmallVector<TypeSourceInfo *, 2> ExpandedTInfos;
        for (unsigned I = 0, N = NTTP->getNumExpansionTypes(); I != N; ++I) {
          ExpandedTypes.push_back(getCanonicalType(NTTP->getExpansionType(I)));
          ExpandedTInfos.push_back(
                                getTrivialTypeSourceInfo(ExpandedTypes.back()));
        }
        
        Param = NonTypeTemplateParmDecl::Create(*this, getTranslationUnitDecl(),
                                                SourceLocation(),
                                                SourceLocation(),
                                                NTTP->getDepth(),
                                                NTTP->getPosition(), 0, 
                                                T,
                                                TInfo,
                                                ExpandedTypes.data(),
                                                ExpandedTypes.size(),
                                                ExpandedTInfos.data());
      } else {
        Param = NonTypeTemplateParmDecl::Create(*this, getTranslationUnitDecl(),
                                                SourceLocation(),
                                                SourceLocation(),
                                                NTTP->getDepth(),
                                                NTTP->getPosition(), 0, 
                                                T,
                                                NTTP->isParameterPack(),
                                                TInfo);
      }
      CanonParams.push_back(Param);

    } else
      CanonParams.push_back(getCanonicalTemplateTemplateParmDecl(
                                           cast<TemplateTemplateParmDecl>(*P)));
  }

  TemplateTemplateParmDecl *CanonTTP
    = TemplateTemplateParmDecl::Create(*this, getTranslationUnitDecl(), 
                                       SourceLocation(), TTP->getDepth(),
                                       TTP->getPosition(), 
                                       TTP->isParameterPack(),
                                       0,
                         TemplateParameterList::Create(*this, SourceLocation(),
                                                       SourceLocation(),
                                                       CanonParams.data(),
                                                       CanonParams.size(),
                                                       SourceLocation()));

  // Get the new insert position for the node we care about.
  Canonical = CanonTemplateTemplateParms.FindNodeOrInsertPos(ID, InsertPos);
  assert(Canonical == 0 && "Shouldn't be in the map!");
  (void)Canonical;

  // Create the canonical template template parameter entry.
  Canonical = new (*this) CanonicalTemplateTemplateParm(CanonTTP);
  CanonTemplateTemplateParms.InsertNode(Canonical, InsertPos);
  return CanonTTP;
}

CXXABI *ASTContext::createCXXABI(const TargetInfo &T) {
  if (!LangOpts.CPlusPlus) return 0;

  switch (T.getCXXABI()) {
  case CXXABI_ARM:
    return CreateARMCXXABI(*this);
  case CXXABI_Itanium:
    return CreateItaniumCXXABI(*this);
  case CXXABI_Microsoft:
    return CreateMicrosoftCXXABI(*this);
  }
  llvm_unreachable("Invalid CXXABI type!");
}

static const LangAS::Map *getAddressSpaceMap(const TargetInfo &T,
                                             const LangOptions &LOpts) {
  if (LOpts.FakeAddressSpaceMap) {
    // The fake address space map must have a distinct entry for each
    // language-specific address space.
    static const unsigned FakeAddrSpaceMap[] = {
      1, // opencl_global
      2, // opencl_local
      3  // opencl_constant
    };
    return &FakeAddrSpaceMap;
  } else {
    return &T.getAddressSpaceMap();
  }
}

ASTContext::ASTContext(LangOptions& LOpts, SourceManager &SM,
                       const TargetInfo *t,
                       IdentifierTable &idents, SelectorTable &sels,
                       Builtin::Context &builtins,
                       unsigned size_reserve,
                       bool DelayInitialization) 
  : FunctionProtoTypes(this_()),
    TemplateSpecializationTypes(this_()),
    DependentTemplateSpecializationTypes(this_()),
    SubstTemplateTemplateParmPacks(this_()),
    GlobalNestedNameSpecifier(0), 
    Int128Decl(0), UInt128Decl(0),
    ObjCIdDecl(0), ObjCSelDecl(0), ObjCClassDecl(0), ObjCProtocolClassDecl(0),
    CFConstantStringTypeDecl(0), ObjCInstanceTypeDecl(0),
    FILEDecl(0), 
    jmp_bufDecl(0), sigjmp_bufDecl(0), ucontext_tDecl(0),
    BlockDescriptorType(0), BlockDescriptorExtendedType(0),
    cudaConfigureCallDecl(0),
    NullTypeSourceInfo(QualType()), 
    FirstLocalImport(), LastLocalImport(),
    SourceMgr(SM), LangOpts(LOpts), 
    AddrSpaceMap(0), Target(t), PrintingPolicy(LOpts),
    Idents(idents), Selectors(sels),
    BuiltinInfo(builtins),
    DeclarationNames(*this),
    ExternalSource(0), Listener(0),
    LastSDM(0, 0),
    UniqueBlockByRefTypeID(0) 
{
  if (size_reserve > 0) Types.reserve(size_reserve);
  TUDecl = TranslationUnitDecl::Create(*this);
  
  if (!DelayInitialization) {
    assert(t && "No target supplied for ASTContext initialization");
    InitBuiltinTypes(*t);
  }
}

ASTContext::~ASTContext() {
  // Release the DenseMaps associated with DeclContext objects.
  // FIXME: Is this the ideal solution?
  ReleaseDeclContextMaps();

  // Call all of the deallocation functions.
  for (unsigned I = 0, N = Deallocations.size(); I != N; ++I)
    Deallocations[I].first(Deallocations[I].second);
  
  // ASTRecordLayout objects in ASTRecordLayouts must always be destroyed
  // because they can contain DenseMaps.
  for (llvm::DenseMap<const ObjCContainerDecl*,
       const ASTRecordLayout*>::iterator
       I = ObjCLayouts.begin(), E = ObjCLayouts.end(); I != E; )
    // Increment in loop to prevent using deallocated memory.
    if (ASTRecordLayout *R = const_cast<ASTRecordLayout*>((I++)->second))
      R->Destroy(*this);

  for (llvm::DenseMap<const RecordDecl*, const ASTRecordLayout*>::iterator
       I = ASTRecordLayouts.begin(), E = ASTRecordLayouts.end(); I != E; ) {
    // Increment in loop to prevent using deallocated memory.
    if (ASTRecordLayout *R = const_cast<ASTRecordLayout*>((I++)->second))
      R->Destroy(*this);
  }
  
  for (llvm::DenseMap<const Decl*, AttrVec*>::iterator A = DeclAttrs.begin(),
                                                    AEnd = DeclAttrs.end();
       A != AEnd; ++A)
    A->second->~AttrVec();
}

void ASTContext::AddDeallocation(void (*Callback)(void*), void *Data) {
  Deallocations.push_back(std::make_pair(Callback, Data));
}

void
ASTContext::setExternalSource(OwningPtr<ExternalASTSource> &Source) {
  ExternalSource.reset(Source.take());
}

void ASTContext::PrintStats() const {
  llvm::errs() << "\n*** AST Context Stats:\n";
  llvm::errs() << "  " << Types.size() << " types total.\n";

  unsigned counts[] = {
#define TYPE(Name, Parent) 0,
#define ABSTRACT_TYPE(Name, Parent)
#include "clang/AST/TypeNodes.def"
    0 // Extra
  };

  for (unsigned i = 0, e = Types.size(); i != e; ++i) {
    Type *T = Types[i];
    counts[(unsigned)T->getTypeClass()]++;
  }

  unsigned Idx = 0;
  unsigned TotalBytes = 0;
#define TYPE(Name, Parent)                                              \
  if (counts[Idx])                                                      \
    llvm::errs() << "    " << counts[Idx] << " " << #Name               \
                 << " types\n";                                         \
  TotalBytes += counts[Idx] * sizeof(Name##Type);                       \
  ++Idx;
#define ABSTRACT_TYPE(Name, Parent)
#include "clang/AST/TypeNodes.def"

  llvm::errs() << "Total bytes = " << TotalBytes << "\n";

  // Implicit special member functions.
  llvm::errs() << NumImplicitDefaultConstructorsDeclared << "/"
               << NumImplicitDefaultConstructors
               << " implicit default constructors created\n";
  llvm::errs() << NumImplicitCopyConstructorsDeclared << "/"
               << NumImplicitCopyConstructors
               << " implicit copy constructors created\n";
  if (getLangOpts().CPlusPlus)
    llvm::errs() << NumImplicitMoveConstructorsDeclared << "/"
                 << NumImplicitMoveConstructors
                 << " implicit move constructors created\n";
  llvm::errs() << NumImplicitCopyAssignmentOperatorsDeclared << "/"
               << NumImplicitCopyAssignmentOperators
               << " implicit copy assignment operators created\n";
  if (getLangOpts().CPlusPlus)
    llvm::errs() << NumImplicitMoveAssignmentOperatorsDeclared << "/"
                 << NumImplicitMoveAssignmentOperators
                 << " implicit move assignment operators created\n";
  llvm::errs() << NumImplicitDestructorsDeclared << "/"
               << NumImplicitDestructors
               << " implicit destructors created\n";

  if (ExternalSource.get()) {
    llvm::errs() << "\n";
    ExternalSource->PrintStats();
  }

  BumpAlloc.PrintStats();
}

TypedefDecl *ASTContext::getInt128Decl() const {
  if (!Int128Decl) {
    TypeSourceInfo *TInfo = getTrivialTypeSourceInfo(Int128Ty);
    Int128Decl = TypedefDecl::Create(const_cast<ASTContext &>(*this), 
                                     getTranslationUnitDecl(),
                                     SourceLocation(),
                                     SourceLocation(),
                                     &Idents.get("__int128_t"),
                                     TInfo);
  }
  
  return Int128Decl;
}

TypedefDecl *ASTContext::getUInt128Decl() const {
  if (!UInt128Decl) {
    TypeSourceInfo *TInfo = getTrivialTypeSourceInfo(UnsignedInt128Ty);
    UInt128Decl = TypedefDecl::Create(const_cast<ASTContext &>(*this), 
                                     getTranslationUnitDecl(),
                                     SourceLocation(),
                                     SourceLocation(),
                                     &Idents.get("__uint128_t"),
                                     TInfo);
  }
  
  return UInt128Decl;
}

void ASTContext::InitBuiltinType(CanQualType &R, BuiltinType::Kind K) {
  BuiltinType *Ty = new (*this, TypeAlignment) BuiltinType(K);
  R = CanQualType::CreateUnsafe(QualType(Ty, 0));
  Types.push_back(Ty);
}

void ASTContext::InitBuiltinTypes(const TargetInfo &Target) {
  assert((!this->Target || this->Target == &Target) &&
         "Incorrect target reinitialization");
  assert(VoidTy.isNull() && "Context reinitialized?");

  this->Target = &Target;
  
  ABI.reset(createCXXABI(Target));
  AddrSpaceMap = getAddressSpaceMap(Target, LangOpts);
  
  // C99 6.2.5p19.
  InitBuiltinType(VoidTy,              BuiltinType::Void);

  // C99 6.2.5p2.
  InitBuiltinType(BoolTy,              BuiltinType::Bool);
  // C99 6.2.5p3.
  if (LangOpts.CharIsSigned)
    InitBuiltinType(CharTy,            BuiltinType::Char_S);
  else
    InitBuiltinType(CharTy,            BuiltinType::Char_U);
  // C99 6.2.5p4.
  InitBuiltinType(SignedCharTy,        BuiltinType::SChar);
  InitBuiltinType(ShortTy,             BuiltinType::Short);
  InitBuiltinType(IntTy,               BuiltinType::Int);
  InitBuiltinType(LongTy,              BuiltinType::Long);
  InitBuiltinType(LongLongTy,          BuiltinType::LongLong);

  // C99 6.2.5p6.
  InitBuiltinType(UnsignedCharTy,      BuiltinType::UChar);
  InitBuiltinType(UnsignedShortTy,     BuiltinType::UShort);
  InitBuiltinType(UnsignedIntTy,       BuiltinType::UInt);
  InitBuiltinType(UnsignedLongTy,      BuiltinType::ULong);
  InitBuiltinType(UnsignedLongLongTy,  BuiltinType::ULongLong);

  // C99 6.2.5p10.
  InitBuiltinType(FloatTy,             BuiltinType::Float);
  InitBuiltinType(DoubleTy,            BuiltinType::Double);
  InitBuiltinType(LongDoubleTy,        BuiltinType::LongDouble);

  // GNU extension, 128-bit integers.
  InitBuiltinType(Int128Ty,            BuiltinType::Int128);
  InitBuiltinType(UnsignedInt128Ty,    BuiltinType::UInt128);

  if (LangOpts.CPlusPlus) { // C++ 3.9.1p5
    if (TargetInfo::isTypeSigned(Target.getWCharType()))
      InitBuiltinType(WCharTy,           BuiltinType::WChar_S);
    else  // -fshort-wchar makes wchar_t be unsigned.
      InitBuiltinType(WCharTy,           BuiltinType::WChar_U);
  } else // C99
    WCharTy = getFromTargetType(Target.getWCharType());

  if (LangOpts.CPlusPlus) // C++0x 3.9.1p5, extension for C++
    InitBuiltinType(Char16Ty,           BuiltinType::Char16);
  else // C99
    Char16Ty = getFromTargetType(Target.getChar16Type());

  if (LangOpts.CPlusPlus) // C++0x 3.9.1p5, extension for C++
    InitBuiltinType(Char32Ty,           BuiltinType::Char32);
  else // C99
    Char32Ty = getFromTargetType(Target.getChar32Type());

  // Placeholder type for type-dependent expressions whose type is
  // completely unknown. No code should ever check a type against
  // DependentTy and users should never see it; however, it is here to
  // help diagnose failures to properly check for type-dependent
  // expressions.
  InitBuiltinType(DependentTy,         BuiltinType::Dependent);

  // Placeholder type for functions.
  InitBuiltinType(OverloadTy,          BuiltinType::Overload);

  // Placeholder type for bound members.
  InitBuiltinType(BoundMemberTy,       BuiltinType::BoundMember);

  // Placeholder type for pseudo-objects.
  InitBuiltinType(PseudoObjectTy,      BuiltinType::PseudoObject);

  // "any" type; useful for debugger-like clients.
  InitBuiltinType(UnknownAnyTy,        BuiltinType::UnknownAny);

  // Placeholder type for unbridged ARC casts.
  InitBuiltinType(ARCUnbridgedCastTy,  BuiltinType::ARCUnbridgedCast);

  // C99 6.2.5p11.
  FloatComplexTy      = getComplexType(FloatTy);
  DoubleComplexTy     = getComplexType(DoubleTy);
  LongDoubleComplexTy = getComplexType(LongDoubleTy);

  BuiltinVaListType = QualType();

  // Builtin types for 'id', 'Class', and 'SEL'.
  InitBuiltinType(ObjCBuiltinIdTy, BuiltinType::ObjCId);
  InitBuiltinType(ObjCBuiltinClassTy, BuiltinType::ObjCClass);
  InitBuiltinType(ObjCBuiltinSelTy, BuiltinType::ObjCSel);
  
  // Builtin type for __objc_yes and __objc_no
  ObjCBuiltinBoolTy = (Target.useSignedCharForObjCBool() ?
                       SignedCharTy : BoolTy);
  
  ObjCConstantStringType = QualType();

  // void * type
  VoidPtrTy = getPointerType(VoidTy);

  // nullptr type (C++0x 2.14.7)
  InitBuiltinType(NullPtrTy,           BuiltinType::NullPtr);

  // half type (OpenCL 6.1.1.1) / ARM NEON __fp16
  InitBuiltinType(HalfTy, BuiltinType::Half);
}

DiagnosticsEngine &ASTContext::getDiagnostics() const {
  return SourceMgr.getDiagnostics();
}

AttrVec& ASTContext::getDeclAttrs(const Decl *D) {
  AttrVec *&Result = DeclAttrs[D];
  if (!Result) {
    void *Mem = Allocate(sizeof(AttrVec));
    Result = new (Mem) AttrVec;
  }
    
  return *Result;
}

/// \brief Erase the attributes corresponding to the given declaration.
void ASTContext::eraseDeclAttrs(const Decl *D) { 
  llvm::DenseMap<const Decl*, AttrVec*>::iterator Pos = DeclAttrs.find(D);
  if (Pos != DeclAttrs.end()) {
    Pos->second->~AttrVec();
    DeclAttrs.erase(Pos);
  }
}

MemberSpecializationInfo *
ASTContext::getInstantiatedFromStaticDataMember(const VarDecl *Var) {
  assert(Var->isStaticDataMember() && "Not a static data member");
  llvm::DenseMap<const VarDecl *, MemberSpecializationInfo *>::iterator Pos
    = InstantiatedFromStaticDataMember.find(Var);
  if (Pos == InstantiatedFromStaticDataMember.end())
    return 0;

  return Pos->second;
}

void
ASTContext::setInstantiatedFromStaticDataMember(VarDecl *Inst, VarDecl *Tmpl,
                                                TemplateSpecializationKind TSK,
                                          SourceLocation PointOfInstantiation) {
  assert(Inst->isStaticDataMember() && "Not a static data member");
  assert(Tmpl->isStaticDataMember() && "Not a static data member");
  assert(!InstantiatedFromStaticDataMember[Inst] &&
         "Already noted what static data member was instantiated from");
  InstantiatedFromStaticDataMember[Inst] 
    = new (*this) MemberSpecializationInfo(Tmpl, TSK, PointOfInstantiation);
}

FunctionDecl *ASTContext::getClassScopeSpecializationPattern(
                                                     const FunctionDecl *FD){
  assert(FD && "Specialization is 0");
  llvm::DenseMap<const FunctionDecl*, FunctionDecl *>::const_iterator Pos
    = ClassScopeSpecializationPattern.find(FD);
  if (Pos == ClassScopeSpecializationPattern.end())
    return 0;

  return Pos->second;
}

void ASTContext::setClassScopeSpecializationPattern(FunctionDecl *FD,
                                        FunctionDecl *Pattern) {
  assert(FD && "Specialization is 0");
  assert(Pattern && "Class scope specialization pattern is 0");
  ClassScopeSpecializationPattern[FD] = Pattern;
}

NamedDecl *
ASTContext::getInstantiatedFromUsingDecl(UsingDecl *UUD) {
  llvm::DenseMap<UsingDecl *, NamedDecl *>::const_iterator Pos
    = InstantiatedFromUsingDecl.find(UUD);
  if (Pos == InstantiatedFromUsingDecl.end())
    return 0;

  return Pos->second;
}

void
ASTContext::setInstantiatedFromUsingDecl(UsingDecl *Inst, NamedDecl *Pattern) {
  assert((isa<UsingDecl>(Pattern) ||
          isa<UnresolvedUsingValueDecl>(Pattern) ||
          isa<UnresolvedUsingTypenameDecl>(Pattern)) && 
         "pattern decl is not a using decl");
  assert(!InstantiatedFromUsingDecl[Inst] && "pattern already exists");
  InstantiatedFromUsingDecl[Inst] = Pattern;
}

UsingShadowDecl *
ASTContext::getInstantiatedFromUsingShadowDecl(UsingShadowDecl *Inst) {
  llvm::DenseMap<UsingShadowDecl*, UsingShadowDecl*>::const_iterator Pos
    = InstantiatedFromUsingShadowDecl.find(Inst);
  if (Pos == InstantiatedFromUsingShadowDecl.end())
    return 0;

  return Pos->second;
}

void
ASTContext::setInstantiatedFromUsingShadowDecl(UsingShadowDecl *Inst,
                                               UsingShadowDecl *Pattern) {
  assert(!InstantiatedFromUsingShadowDecl[Inst] && "pattern already exists");
  InstantiatedFromUsingShadowDecl[Inst] = Pattern;
}

FieldDecl *ASTContext::getInstantiatedFromUnnamedFieldDecl(FieldDecl *Field) {
  llvm::DenseMap<FieldDecl *, FieldDecl *>::iterator Pos
    = InstantiatedFromUnnamedFieldDecl.find(Field);
  if (Pos == InstantiatedFromUnnamedFieldDecl.end())
    return 0;

  return Pos->second;
}

void ASTContext::setInstantiatedFromUnnamedFieldDecl(FieldDecl *Inst,
                                                     FieldDecl *Tmpl) {
  assert(!Inst->getDeclName() && "Instantiated field decl is not unnamed");
  assert(!Tmpl->getDeclName() && "Template field decl is not unnamed");
  assert(!InstantiatedFromUnnamedFieldDecl[Inst] &&
         "Already noted what unnamed field was instantiated from");

  InstantiatedFromUnnamedFieldDecl[Inst] = Tmpl;
}

bool ASTContext::ZeroBitfieldFollowsNonBitfield(const FieldDecl *FD, 
                                    const FieldDecl *LastFD) const {
  return (FD->isBitField() && LastFD && !LastFD->isBitField() &&
          FD->getBitWidthValue(*this) == 0);
}

bool ASTContext::ZeroBitfieldFollowsBitfield(const FieldDecl *FD,
                                             const FieldDecl *LastFD) const {
  return (FD->isBitField() && LastFD && LastFD->isBitField() &&
          FD->getBitWidthValue(*this) == 0 &&
          LastFD->getBitWidthValue(*this) != 0);
}

bool ASTContext::BitfieldFollowsBitfield(const FieldDecl *FD,
                                         const FieldDecl *LastFD) const {
  return (FD->isBitField() && LastFD && LastFD->isBitField() &&
          FD->getBitWidthValue(*this) &&
          LastFD->getBitWidthValue(*this));
}

bool ASTContext::NonBitfieldFollowsBitfield(const FieldDecl *FD,
                                         const FieldDecl *LastFD) const {
  return (!FD->isBitField() && LastFD && LastFD->isBitField() &&
          LastFD->getBitWidthValue(*this));
}

bool ASTContext::BitfieldFollowsNonBitfield(const FieldDecl *FD,
                                             const FieldDecl *LastFD) const {
  return (FD->isBitField() && LastFD && !LastFD->isBitField() &&
          FD->getBitWidthValue(*this));
}

ASTContext::overridden_cxx_method_iterator
ASTContext::overridden_methods_begin(const CXXMethodDecl *Method) const {
  llvm::DenseMap<const CXXMethodDecl *, CXXMethodVector>::const_iterator Pos
    = OverriddenMethods.find(Method);
  if (Pos == OverriddenMethods.end())
    return 0;

  return Pos->second.begin();
}

ASTContext::overridden_cxx_method_iterator
ASTContext::overridden_methods_end(const CXXMethodDecl *Method) const {
  llvm::DenseMap<const CXXMethodDecl *, CXXMethodVector>::const_iterator Pos
    = OverriddenMethods.find(Method);
  if (Pos == OverriddenMethods.end())
    return 0;

  return Pos->second.end();
}

unsigned
ASTContext::overridden_methods_size(const CXXMethodDecl *Method) const {
  llvm::DenseMap<const CXXMethodDecl *, CXXMethodVector>::const_iterator Pos
    = OverriddenMethods.find(Method);
  if (Pos == OverriddenMethods.end())
    return 0;

  return Pos->second.size();
}

void ASTContext::addOverriddenMethod(const CXXMethodDecl *Method, 
                                     const CXXMethodDecl *Overridden) {
  OverriddenMethods[Method].push_back(Overridden);
}

void ASTContext::addedLocalImportDecl(ImportDecl *Import) {
  assert(!Import->NextLocalImport && "Import declaration already in the chain");
  assert(!Import->isFromASTFile() && "Non-local import declaration");
  if (!FirstLocalImport) {
    FirstLocalImport = Import;
    LastLocalImport = Import;
    return;
  }
  
  LastLocalImport->NextLocalImport = Import;
  LastLocalImport = Import;
}

//===----------------------------------------------------------------------===//
//                         Type Sizing and Analysis
//===----------------------------------------------------------------------===//

/// getFloatTypeSemantics - Return the APFloat 'semantics' for the specified
/// scalar floating point type.
const llvm::fltSemantics &ASTContext::getFloatTypeSemantics(QualType T) const {
  const BuiltinType *BT = T->getAs<BuiltinType>();
  assert(BT && "Not a floating point type!");
  switch (BT->getKind()) {
  default: llvm_unreachable("Not a floating point type!");
  case BuiltinType::Half:       return Target->getHalfFormat();
  case BuiltinType::Float:      return Target->getFloatFormat();
  case BuiltinType::Double:     return Target->getDoubleFormat();
  case BuiltinType::LongDouble: return Target->getLongDoubleFormat();
  }
}

/// getDeclAlign - Return a conservative estimate of the alignment of the
/// specified decl.  Note that bitfields do not have a valid alignment, so
/// this method will assert on them.
/// If @p RefAsPointee, references are treated like their underlying type
/// (for alignof), else they're treated like pointers (for CodeGen).
CharUnits ASTContext::getDeclAlign(const Decl *D, bool RefAsPointee) const {
  unsigned Align = Target->getCharWidth();

  bool UseAlignAttrOnly = false;
  if (unsigned AlignFromAttr = D->getMaxAlignment()) {
    Align = AlignFromAttr;

    // __attribute__((aligned)) can increase or decrease alignment
    // *except* on a struct or struct member, where it only increases
    // alignment unless 'packed' is also specified.
    //
    // It is an error for alignas to decrease alignment, so we can
    // ignore that possibility;  Sema should diagnose it.
    if (isa<FieldDecl>(D)) {
      UseAlignAttrOnly = D->hasAttr<PackedAttr>() ||
        cast<FieldDecl>(D)->getParent()->hasAttr<PackedAttr>();
    } else {
      UseAlignAttrOnly = true;
    }
  }
  else if (isa<FieldDecl>(D))
      UseAlignAttrOnly = 
        D->hasAttr<PackedAttr>() ||
        cast<FieldDecl>(D)->getParent()->hasAttr<PackedAttr>();

  // If we're using the align attribute only, just ignore everything
  // else about the declaration and its type.
  if (UseAlignAttrOnly) {
    // do nothing

  } else if (const ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    QualType T = VD->getType();
    if (const ReferenceType* RT = T->getAs<ReferenceType>()) {
      if (RefAsPointee)
        T = RT->getPointeeType();
      else
        T = getPointerType(RT->getPointeeType());
    }
    if (!T->isIncompleteType() && !T->isFunctionType()) {
      // Adjust alignments of declarations with array type by the
      // large-array alignment on the target.
      unsigned MinWidth = Target->getLargeArrayMinWidth();
      const ArrayType *arrayType;
      if (MinWidth && (arrayType = getAsArrayType(T))) {
        if (isa<VariableArrayType>(arrayType))
          Align = std::max(Align, Target->getLargeArrayAlign());
        else if (isa<ConstantArrayType>(arrayType) &&
                 MinWidth <= getTypeSize(cast<ConstantArrayType>(arrayType)))
          Align = std::max(Align, Target->getLargeArrayAlign());

        // Walk through any array types while we're at it.
        T = getBaseElementType(arrayType);
      }
      Align = std::max(Align, getPreferredTypeAlign(T.getTypePtr()));
    }

    // Fields can be subject to extra alignment constraints, like if
    // the field is packed, the struct is packed, or the struct has a
    // a max-field-alignment constraint (#pragma pack).  So calculate
    // the actual alignment of the field within the struct, and then
    // (as we're expected to) constrain that by the alignment of the type.
    if (const FieldDecl *field = dyn_cast<FieldDecl>(VD)) {
      // So calculate the alignment of the field.
      const ASTRecordLayout &layout = getASTRecordLayout(field->getParent());

      // Start with the record's overall alignment.
      unsigned fieldAlign = toBits(layout.getAlignment());

      // Use the GCD of that and the offset within the record.
      uint64_t offset = layout.getFieldOffset(field->getFieldIndex());
      if (offset > 0) {
        // Alignment is always a power of 2, so the GCD will be a power of 2,
        // which means we get to do this crazy thing instead of Euclid's.
        uint64_t lowBitOfOffset = offset & (~offset + 1);
        if (lowBitOfOffset < fieldAlign)
          fieldAlign = static_cast<unsigned>(lowBitOfOffset);
      }

      Align = std::min(Align, fieldAlign);
    }
  }

  return toCharUnitsFromBits(Align);
}

std::pair<CharUnits, CharUnits>
ASTContext::getTypeInfoInChars(const Type *T) const {
  std::pair<uint64_t, unsigned> Info = getTypeInfo(T);
  return std::make_pair(toCharUnitsFromBits(Info.first),
                        toCharUnitsFromBits(Info.second));
}

std::pair<CharUnits, CharUnits>
ASTContext::getTypeInfoInChars(QualType T) const {
  return getTypeInfoInChars(T.getTypePtr());
}

std::pair<uint64_t, unsigned> ASTContext::getTypeInfo(const Type *T) const {
  TypeInfoMap::iterator it = MemoizedTypeInfo.find(T);
  if (it != MemoizedTypeInfo.end())
    return it->second;

  std::pair<uint64_t, unsigned> Info = getTypeInfoImpl(T);
  MemoizedTypeInfo.insert(std::make_pair(T, Info));
  return Info;
}

/// getTypeInfoImpl - Return the size of the specified type, in bits.  This
/// method does not work on incomplete types.
///
/// FIXME: Pointers into different addr spaces could have different sizes and
/// alignment requirements: getPointerInfo should take an AddrSpace, this
/// should take a QualType, &c.
std::pair<uint64_t, unsigned>
ASTContext::getTypeInfoImpl(const Type *T) const {
  uint64_t Width=0;
  unsigned Align=8;
  switch (T->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    llvm_unreachable("Should not see dependent types");

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // GCC extension: alignof(function) = 32 bits
    Width = 0;
    Align = 32;
    break;

  case Type::IncompleteArray:
  case Type::VariableArray:
    Width = 0;
    Align = getTypeAlign(cast<ArrayType>(T)->getElementType());
    break;

  case Type::ConstantArray: {
    const ConstantArrayType *CAT = cast<ConstantArrayType>(T);

    std::pair<uint64_t, unsigned> EltInfo = getTypeInfo(CAT->getElementType());
    uint64_t Size = CAT->getSize().getZExtValue();
    assert((Size == 0 || EltInfo.first <= (uint64_t)(-1)/Size) && 
           "Overflow in array type bit size evaluation");
    Width = EltInfo.first*Size;
    Align = EltInfo.second;
    Width = llvm::RoundUpToAlignment(Width, Align);
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    const VectorType *VT = cast<VectorType>(T);
    std::pair<uint64_t, unsigned> EltInfo = getTypeInfo(VT->getElementType());
    Width = EltInfo.first*VT->getNumElements();
    Align = Width;
    // If the alignment is not a power of 2, round up to the next power of 2.
    // This happens for non-power-of-2 length vectors.
    if (Align & (Align-1)) {
      Align = llvm::NextPowerOf2(Align);
      Width = llvm::RoundUpToAlignment(Width, Align);
    }
    break;
  }

  case Type::Builtin:
    switch (cast<BuiltinType>(T)->getKind()) {
    default: llvm_unreachable("Unknown builtin type!");
    case BuiltinType::Void:
      // GCC extension: alignof(void) = 8 bits.
      Width = 0;
      Align = 8;
      break;

    case BuiltinType::Bool:
      Width = Target->getBoolWidth();
      Align = Target->getBoolAlign();
      break;
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::SChar:
      Width = Target->getCharWidth();
      Align = Target->getCharAlign();
      break;
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
      Width = Target->getWCharWidth();
      Align = Target->getWCharAlign();
      break;
    case BuiltinType::Char16:
      Width = Target->getChar16Width();
      Align = Target->getChar16Align();
      break;
    case BuiltinType::Char32:
      Width = Target->getChar32Width();
      Align = Target->getChar32Align();
      break;
    case BuiltinType::UShort:
    case BuiltinType::Short:
      Width = Target->getShortWidth();
      Align = Target->getShortAlign();
      break;
    case BuiltinType::UInt:
    case BuiltinType::Int:
      Width = Target->getIntWidth();
      Align = Target->getIntAlign();
      break;
    case BuiltinType::ULong:
    case BuiltinType::Long:
      Width = Target->getLongWidth();
      Align = Target->getLongAlign();
      break;
    case BuiltinType::ULongLong:
    case BuiltinType::LongLong:
      Width = Target->getLongLongWidth();
      Align = Target->getLongLongAlign();
      break;
    case BuiltinType::Int128:
    case BuiltinType::UInt128:
      Width = 128;
      Align = 128; // int128_t is 128-bit aligned on all targets.
      break;
    case BuiltinType::Half:
      Width = Target->getHalfWidth();
      Align = Target->getHalfAlign();
      break;
    case BuiltinType::Float:
      Width = Target->getFloatWidth();
      Align = Target->getFloatAlign();
      break;
    case BuiltinType::Double:
      Width = Target->getDoubleWidth();
      Align = Target->getDoubleAlign();
      break;
    case BuiltinType::LongDouble:
      Width = Target->getLongDoubleWidth();
      Align = Target->getLongDoubleAlign();
      break;
    case BuiltinType::NullPtr:
      Width = Target->getPointerWidth(0); // C++ 3.9.1p11: sizeof(nullptr_t)
      Align = Target->getPointerAlign(0); //   == sizeof(void*)
      break;
    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      Width = Target->getPointerWidth(0); 
      Align = Target->getPointerAlign(0);
      break;
    }
    break;
  case Type::ObjCObjectPointer:
    Width = Target->getPointerWidth(0);
    Align = Target->getPointerAlign(0);
    break;
  case Type::BlockPointer: {
    unsigned AS = getTargetAddressSpace(
        cast<BlockPointerType>(T)->getPointeeType());
    Width = Target->getPointerWidth(AS);
    Align = Target->getPointerAlign(AS);
    break;
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    // alignof and sizeof should never enter this code path here, so we go
    // the pointer route.
    unsigned AS = getTargetAddressSpace(
        cast<ReferenceType>(T)->getPointeeType());
    Width = Target->getPointerWidth(AS);
    Align = Target->getPointerAlign(AS);
    break;
  }
  case Type::Pointer: {
    unsigned AS = getTargetAddressSpace(cast<PointerType>(T)->getPointeeType());
    Width = Target->getPointerWidth(AS);
    Align = Target->getPointerAlign(AS);
    break;
  }
  case Type::MemberPointer: {
    const MemberPointerType *MPT = cast<MemberPointerType>(T);
    std::pair<uint64_t, unsigned> PtrDiffInfo =
      getTypeInfo(getPointerDiffType());
    Width = PtrDiffInfo.first * ABI->getMemberPointerSize(MPT);
    Align = PtrDiffInfo.second;
    break;
  }
  case Type::Complex: {
    // Complex types have the same alignment as their elements, but twice the
    // size.
    std::pair<uint64_t, unsigned> EltInfo =
      getTypeInfo(cast<ComplexType>(T)->getElementType());
    Width = EltInfo.first*2;
    Align = EltInfo.second;
    break;
  }
  case Type::ObjCObject:
    return getTypeInfo(cast<ObjCObjectType>(T)->getBaseType().getTypePtr());
  case Type::ObjCInterface: {
    const ObjCInterfaceType *ObjCI = cast<ObjCInterfaceType>(T);
    const ASTRecordLayout &Layout = getASTObjCInterfaceLayout(ObjCI->getDecl());
    Width = toBits(Layout.getSize());
    Align = toBits(Layout.getAlignment());
    break;
  }
  case Type::Record:
  case Type::Enum: {
    const TagType *TT = cast<TagType>(T);

    if (TT->getDecl()->isInvalidDecl()) {
      Width = 8;
      Align = 8;
      break;
    }

    if (const EnumType *ET = dyn_cast<EnumType>(TT))
      return getTypeInfo(ET->getDecl()->getIntegerType());

    const RecordType *RT = cast<RecordType>(TT);
    const ASTRecordLayout &Layout = getASTRecordLayout(RT->getDecl());
    Width = toBits(Layout.getSize());
    Align = toBits(Layout.getAlignment());
    break;
  }

  case Type::SubstTemplateTypeParm:
    return getTypeInfo(cast<SubstTemplateTypeParmType>(T)->
                       getReplacementType().getTypePtr());

  case Type::Auto: {
    const AutoType *A = cast<AutoType>(T);
    assert(A->isDeduced() && "Cannot request the size of a dependent type");
    return getTypeInfo(A->getDeducedType().getTypePtr());
  }

  case Type::Paren:
    return getTypeInfo(cast<ParenType>(T)->getInnerType().getTypePtr());

  case Type::Typedef: {
    const TypedefNameDecl *Typedef = cast<TypedefType>(T)->getDecl();
    std::pair<uint64_t, unsigned> Info
      = getTypeInfo(Typedef->getUnderlyingType().getTypePtr());
    // If the typedef has an aligned attribute on it, it overrides any computed
    // alignment we have.  This violates the GCC documentation (which says that
    // attribute(aligned) can only round up) but matches its implementation.
    if (unsigned AttrAlign = Typedef->getMaxAlignment())
      Align = AttrAlign;
    else
      Align = Info.second;
    Width = Info.first;
    break;
  }

  case Type::TypeOfExpr:
    return getTypeInfo(cast<TypeOfExprType>(T)->getUnderlyingExpr()->getType()
                         .getTypePtr());

  case Type::TypeOf:
    return getTypeInfo(cast<TypeOfType>(T)->getUnderlyingType().getTypePtr());

  case Type::Decltype:
    return getTypeInfo(cast<DecltypeType>(T)->getUnderlyingExpr()->getType()
                        .getTypePtr());

  case Type::UnaryTransform:
    return getTypeInfo(cast<UnaryTransformType>(T)->getUnderlyingType());

  case Type::Elaborated:
    return getTypeInfo(cast<ElaboratedType>(T)->getNamedType().getTypePtr());

  case Type::Attributed:
    return getTypeInfo(
                  cast<AttributedType>(T)->getEquivalentType().getTypePtr());

  case Type::TemplateSpecialization: {
    assert(getCanonicalType(T) != T &&
           "Cannot request the size of a dependent type");
    const TemplateSpecializationType *TST = cast<TemplateSpecializationType>(T);
    // A type alias template specialization may refer to a typedef with the
    // aligned attribute on it.
    if (TST->isTypeAlias())
      return getTypeInfo(TST->getAliasedType().getTypePtr());
    else
      return getTypeInfo(getCanonicalType(T));
  }

  case Type::Atomic: {
    std::pair<uint64_t, unsigned> Info
      = getTypeInfo(cast<AtomicType>(T)->getValueType());
    Width = Info.first;
    Align = Info.second;
    if (Width != 0 && Width <= Target->getMaxAtomicPromoteWidth() &&
        llvm::isPowerOf2_64(Width)) {
      // We can potentially perform lock-free atomic operations for this
      // type; promote the alignment appropriately.
      // FIXME: We could potentially promote the width here as well...
      // is that worthwhile?  (Non-struct atomic types generally have
      // power-of-two size anyway, but structs might not.  Requires a bit
      // of implementation work to make sure we zero out the extra bits.)
      Align = static_cast<unsigned>(Width);
    }
  }

  }

  assert(llvm::isPowerOf2_32(Align) && "Alignment must be power of 2");
  return std::make_pair(Width, Align);
}

/// toCharUnitsFromBits - Convert a size in bits to a size in characters.
CharUnits ASTContext::toCharUnitsFromBits(int64_t BitSize) const {
  return CharUnits::fromQuantity(BitSize / getCharWidth());
}

/// toBits - Convert a size in characters to a size in characters.
int64_t ASTContext::toBits(CharUnits CharSize) const {
  return CharSize.getQuantity() * getCharWidth();
}

/// getTypeSizeInChars - Return the size of the specified type, in characters.
/// This method does not work on incomplete types.
CharUnits ASTContext::getTypeSizeInChars(QualType T) const {
  return toCharUnitsFromBits(getTypeSize(T));
}
CharUnits ASTContext::getTypeSizeInChars(const Type *T) const {
  return toCharUnitsFromBits(getTypeSize(T));
}

/// getTypeAlignInChars - Return the ABI-specified alignment of a type, in 
/// characters. This method does not work on incomplete types.
CharUnits ASTContext::getTypeAlignInChars(QualType T) const {
  return toCharUnitsFromBits(getTypeAlign(T));
}
CharUnits ASTContext::getTypeAlignInChars(const Type *T) const {
  return toCharUnitsFromBits(getTypeAlign(T));
}

/// getPreferredTypeAlign - Return the "preferred" alignment of the specified
/// type for the current target in bits.  This can be different than the ABI
/// alignment in cases where it is beneficial for performance to overalign
/// a data type.
unsigned ASTContext::getPreferredTypeAlign(const Type *T) const {
  unsigned ABIAlign = getTypeAlign(T);

  // Double and long long should be naturally aligned if possible.
  if (const ComplexType* CT = T->getAs<ComplexType>())
    T = CT->getElementType().getTypePtr();
  if (T->isSpecificBuiltinType(BuiltinType::Double) ||
      T->isSpecificBuiltinType(BuiltinType::LongLong) ||
      T->isSpecificBuiltinType(BuiltinType::ULongLong))
    return std::max(ABIAlign, (unsigned)getTypeSize(T));

  return ABIAlign;
}

/// DeepCollectObjCIvars -
/// This routine first collects all declared, but not synthesized, ivars in
/// super class and then collects all ivars, including those synthesized for
/// current class. This routine is used for implementation of current class
/// when all ivars, declared and synthesized are known.
///
void ASTContext::DeepCollectObjCIvars(const ObjCInterfaceDecl *OI,
                                      bool leafClass,
                            SmallVectorImpl<const ObjCIvarDecl*> &Ivars) const {
  if (const ObjCInterfaceDecl *SuperClass = OI->getSuperClass())
    DeepCollectObjCIvars(SuperClass, false, Ivars);
  if (!leafClass) {
    for (ObjCInterfaceDecl::ivar_iterator I = OI->ivar_begin(),
         E = OI->ivar_end(); I != E; ++I)
      Ivars.push_back(*I);
  } else {
    ObjCInterfaceDecl *IDecl = const_cast<ObjCInterfaceDecl *>(OI);
    for (const ObjCIvarDecl *Iv = IDecl->all_declared_ivar_begin(); Iv; 
         Iv= Iv->getNextIvar())
      Ivars.push_back(Iv);
  }
}

/// CollectInheritedProtocols - Collect all protocols in current class and
/// those inherited by it.
void ASTContext::CollectInheritedProtocols(const Decl *CDecl,
                          llvm::SmallPtrSet<ObjCProtocolDecl*, 8> &Protocols) {
  if (const ObjCInterfaceDecl *OI = dyn_cast<ObjCInterfaceDecl>(CDecl)) {
    // We can use protocol_iterator here instead of
    // all_referenced_protocol_iterator since we are walking all categories.    
    for (ObjCInterfaceDecl::all_protocol_iterator P = OI->all_referenced_protocol_begin(),
         PE = OI->all_referenced_protocol_end(); P != PE; ++P) {
      ObjCProtocolDecl *Proto = (*P);
      Protocols.insert(Proto->getCanonicalDecl());
      for (ObjCProtocolDecl::protocol_iterator P = Proto->protocol_begin(),
           PE = Proto->protocol_end(); P != PE; ++P) {
        Protocols.insert((*P)->getCanonicalDecl());
        CollectInheritedProtocols(*P, Protocols);
      }
    }
    
    // Categories of this Interface.
    for (const ObjCCategoryDecl *CDeclChain = OI->getCategoryList(); 
         CDeclChain; CDeclChain = CDeclChain->getNextClassCategory())
      CollectInheritedProtocols(CDeclChain, Protocols);
    if (ObjCInterfaceDecl *SD = OI->getSuperClass())
      while (SD) {
        CollectInheritedProtocols(SD, Protocols);
        SD = SD->getSuperClass();
      }
  } else if (const ObjCCategoryDecl *OC = dyn_cast<ObjCCategoryDecl>(CDecl)) {
    for (ObjCCategoryDecl::protocol_iterator P = OC->protocol_begin(),
         PE = OC->protocol_end(); P != PE; ++P) {
      ObjCProtocolDecl *Proto = (*P);
      Protocols.insert(Proto->getCanonicalDecl());
      for (ObjCProtocolDecl::protocol_iterator P = Proto->protocol_begin(),
           PE = Proto->protocol_end(); P != PE; ++P)
        CollectInheritedProtocols(*P, Protocols);
    }
  } else if (const ObjCProtocolDecl *OP = dyn_cast<ObjCProtocolDecl>(CDecl)) {
    for (ObjCProtocolDecl::protocol_iterator P = OP->protocol_begin(),
         PE = OP->protocol_end(); P != PE; ++P) {
      ObjCProtocolDecl *Proto = (*P);
      Protocols.insert(Proto->getCanonicalDecl());
      for (ObjCProtocolDecl::protocol_iterator P = Proto->protocol_begin(),
           PE = Proto->protocol_end(); P != PE; ++P)
        CollectInheritedProtocols(*P, Protocols);
    }
  }
}

unsigned ASTContext::CountNonClassIvars(const ObjCInterfaceDecl *OI) const {
  unsigned count = 0;  
  // Count ivars declared in class extension.
  for (const ObjCCategoryDecl *CDecl = OI->getFirstClassExtension(); CDecl;
       CDecl = CDecl->getNextClassExtension())
    count += CDecl->ivar_size();

  // Count ivar defined in this class's implementation.  This
  // includes synthesized ivars.
  if (ObjCImplementationDecl *ImplDecl = OI->getImplementation())
    count += ImplDecl->ivar_size();

  return count;
}

bool ASTContext::isSentinelNullExpr(const Expr *E) {
  if (!E)
    return false;

  // nullptr_t is always treated as null.
  if (E->getType()->isNullPtrType()) return true;

  if (E->getType()->isAnyPointerType() &&
      E->IgnoreParenCasts()->isNullPointerConstant(*this,
                                                Expr::NPC_ValueDependentIsNull))
    return true;

  // Unfortunately, __null has type 'int'.
  if (isa<GNUNullExpr>(E)) return true;

  return false;
}

/// \brief Get the implementation of ObjCInterfaceDecl,or NULL if none exists.
ObjCImplementationDecl *ASTContext::getObjCImplementation(ObjCInterfaceDecl *D) {
  llvm::DenseMap<ObjCContainerDecl*, ObjCImplDecl*>::iterator
    I = ObjCImpls.find(D);
  if (I != ObjCImpls.end())
    return cast<ObjCImplementationDecl>(I->second);
  return 0;
}
/// \brief Get the implementation of ObjCCategoryDecl, or NULL if none exists.
ObjCCategoryImplDecl *ASTContext::getObjCImplementation(ObjCCategoryDecl *D) {
  llvm::DenseMap<ObjCContainerDecl*, ObjCImplDecl*>::iterator
    I = ObjCImpls.find(D);
  if (I != ObjCImpls.end())
    return cast<ObjCCategoryImplDecl>(I->second);
  return 0;
}

/// \brief Set the implementation of ObjCInterfaceDecl.
void ASTContext::setObjCImplementation(ObjCInterfaceDecl *IFaceD,
                           ObjCImplementationDecl *ImplD) {
  assert(IFaceD && ImplD && "Passed null params");
  ObjCImpls[IFaceD] = ImplD;
}
/// \brief Set the implementation of ObjCCategoryDecl.
void ASTContext::setObjCImplementation(ObjCCategoryDecl *CatD,
                           ObjCCategoryImplDecl *ImplD) {
  assert(CatD && ImplD && "Passed null params");
  ObjCImpls[CatD] = ImplD;
}

ObjCInterfaceDecl *ASTContext::getObjContainingInterface(NamedDecl *ND) const {
  if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(ND->getDeclContext()))
    return ID;
  if (ObjCCategoryDecl *CD = dyn_cast<ObjCCategoryDecl>(ND->getDeclContext()))
    return CD->getClassInterface();
  if (ObjCImplDecl *IMD = dyn_cast<ObjCImplDecl>(ND->getDeclContext()))
    return IMD->getClassInterface();

  return 0;
}

/// \brief Get the copy initialization expression of VarDecl,or NULL if 
/// none exists.
Expr *ASTContext::getBlockVarCopyInits(const VarDecl*VD) {
  assert(VD && "Passed null params");
  assert(VD->hasAttr<BlocksAttr>() && 
         "getBlockVarCopyInits - not __block var");
  llvm::DenseMap<const VarDecl*, Expr*>::iterator
    I = BlockVarCopyInits.find(VD);
  return (I != BlockVarCopyInits.end()) ? cast<Expr>(I->second) : 0;
}

/// \brief Set the copy inialization expression of a block var decl.
void ASTContext::setBlockVarCopyInits(VarDecl*VD, Expr* Init) {
  assert(VD && Init && "Passed null params");
  assert(VD->hasAttr<BlocksAttr>() && 
         "setBlockVarCopyInits - not __block var");
  BlockVarCopyInits[VD] = Init;
}

/// \brief Allocate an uninitialized TypeSourceInfo.
///
/// The caller should initialize the memory held by TypeSourceInfo using
/// the TypeLoc wrappers.
///
/// \param T the type that will be the basis for type source info. This type
/// should refer to how the declarator was written in source code, not to
/// what type semantic analysis resolved the declarator to.
TypeSourceInfo *ASTContext::CreateTypeSourceInfo(QualType T,
                                                 unsigned DataSize) const {
  if (!DataSize)
    DataSize = TypeLoc::getFullDataSizeForType(T);
  else
    assert(DataSize == TypeLoc::getFullDataSizeForType(T) &&
           "incorrect data size provided to CreateTypeSourceInfo!");

  TypeSourceInfo *TInfo =
    (TypeSourceInfo*)BumpAlloc.Allocate(sizeof(TypeSourceInfo) + DataSize, 8);
  new (TInfo) TypeSourceInfo(T);
  return TInfo;
}

TypeSourceInfo *ASTContext::getTrivialTypeSourceInfo(QualType T,
                                                     SourceLocation L) const {
  TypeSourceInfo *DI = CreateTypeSourceInfo(T);
  DI->getTypeLoc().initialize(const_cast<ASTContext &>(*this), L);
  return DI;
}

const ASTRecordLayout &
ASTContext::getASTObjCInterfaceLayout(const ObjCInterfaceDecl *D) const {
  return getObjCLayout(D, 0);
}

const ASTRecordLayout &
ASTContext::getASTObjCImplementationLayout(
                                        const ObjCImplementationDecl *D) const {
  return getObjCLayout(D->getClassInterface(), D);
}

//===----------------------------------------------------------------------===//
//                   Type creation/memoization methods
//===----------------------------------------------------------------------===//

QualType
ASTContext::getExtQualType(const Type *baseType, Qualifiers quals) const {
  unsigned fastQuals = quals.getFastQualifiers();
  quals.removeFastQualifiers();

  // Check if we've already instantiated this type.
  llvm::FoldingSetNodeID ID;
  ExtQuals::Profile(ID, baseType, quals);
  void *insertPos = 0;
  if (ExtQuals *eq = ExtQualNodes.FindNodeOrInsertPos(ID, insertPos)) {
    assert(eq->getQualifiers() == quals);
    return QualType(eq, fastQuals);
  }

  // If the base type is not canonical, make the appropriate canonical type.
  QualType canon;
  if (!baseType->isCanonicalUnqualified()) {
    SplitQualType canonSplit = baseType->getCanonicalTypeInternal().split();
    canonSplit.Quals.addConsistentQualifiers(quals);
    canon = getExtQualType(canonSplit.Ty, canonSplit.Quals);

    // Re-find the insert position.
    (void) ExtQualNodes.FindNodeOrInsertPos(ID, insertPos);
  }

  ExtQuals *eq = new (*this, TypeAlignment) ExtQuals(baseType, canon, quals);
  ExtQualNodes.InsertNode(eq, insertPos);
  return QualType(eq, fastQuals);
}

QualType
ASTContext::getAddrSpaceQualType(QualType T, unsigned AddressSpace) const {
  QualType CanT = getCanonicalType(T);
  if (CanT.getAddressSpace() == AddressSpace)
    return T;

  // If we are composing extended qualifiers together, merge together
  // into one ExtQuals node.
  QualifierCollector Quals;
  const Type *TypeNode = Quals.strip(T);

  // If this type already has an address space specified, it cannot get
  // another one.
  assert(!Quals.hasAddressSpace() &&
         "Type cannot be in multiple addr spaces!");
  Quals.addAddressSpace(AddressSpace);

  return getExtQualType(TypeNode, Quals);
}

QualType ASTContext::getObjCGCQualType(QualType T,
                                       Qualifiers::GC GCAttr) const {
  QualType CanT = getCanonicalType(T);
  if (CanT.getObjCGCAttr() == GCAttr)
    return T;

  if (const PointerType *ptr = T->getAs<PointerType>()) {
    QualType Pointee = ptr->getPointeeType();
    if (Pointee->isAnyPointerType()) {
      QualType ResultType = getObjCGCQualType(Pointee, GCAttr);
      return getPointerType(ResultType);
    }
  }

  // If we are composing extended qualifiers together, merge together
  // into one ExtQuals node.
  QualifierCollector Quals;
  const Type *TypeNode = Quals.strip(T);

  // If this type already has an ObjCGC specified, it cannot get
  // another one.
  assert(!Quals.hasObjCGCAttr() &&
         "Type cannot have multiple ObjCGCs!");
  Quals.addObjCGCAttr(GCAttr);

  return getExtQualType(TypeNode, Quals);
}

const FunctionType *ASTContext::adjustFunctionType(const FunctionType *T,
                                                   FunctionType::ExtInfo Info) {
  if (T->getExtInfo() == Info)
    return T;

  QualType Result;
  if (const FunctionNoProtoType *FNPT = dyn_cast<FunctionNoProtoType>(T)) {
    Result = getFunctionNoProtoType(FNPT->getResultType(), Info);
  } else {
    const FunctionProtoType *FPT = cast<FunctionProtoType>(T);
    FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
    EPI.ExtInfo = Info;
    Result = getFunctionType(FPT->getResultType(), FPT->arg_type_begin(),
                             FPT->getNumArgs(), EPI);
  }

  return cast<FunctionType>(Result.getTypePtr());
}

/// getComplexType - Return the uniqued reference to the type for a complex
/// number with the specified element type.
QualType ASTContext::getComplexType(QualType T) const {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  ComplexType::Profile(ID, T);

  void *InsertPos = 0;
  if (ComplexType *CT = ComplexTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(CT, 0);

  // If the pointee type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical()) {
    Canonical = getComplexType(getCanonicalType(T));

    // Get the new insert position for the node we care about.
    ComplexType *NewIP = ComplexTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }
  ComplexType *New = new (*this, TypeAlignment) ComplexType(T, Canonical);
  Types.push_back(New);
  ComplexTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getPointerType - Return the uniqued reference to the type for a pointer to
/// the specified type.
QualType ASTContext::getPointerType(QualType T) const {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  PointerType::Profile(ID, T);

  void *InsertPos = 0;
  if (PointerType *PT = PointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(PT, 0);

  // If the pointee type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical()) {
    Canonical = getPointerType(getCanonicalType(T));

    // Get the new insert position for the node we care about.
    PointerType *NewIP = PointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }
  PointerType *New = new (*this, TypeAlignment) PointerType(T, Canonical);
  Types.push_back(New);
  PointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getBlockPointerType - Return the uniqued reference to the type for
/// a pointer to the specified block.
QualType ASTContext::getBlockPointerType(QualType T) const {
  assert(T->isFunctionType() && "block of function types only");
  // Unique pointers, to guarantee there is only one block of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  BlockPointerType::Profile(ID, T);

  void *InsertPos = 0;
  if (BlockPointerType *PT =
        BlockPointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(PT, 0);

  // If the block pointee type isn't canonical, this won't be a canonical
  // type either so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical()) {
    Canonical = getBlockPointerType(getCanonicalType(T));

    // Get the new insert position for the node we care about.
    BlockPointerType *NewIP =
      BlockPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }
  BlockPointerType *New
    = new (*this, TypeAlignment) BlockPointerType(T, Canonical);
  Types.push_back(New);
  BlockPointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getLValueReferenceType - Return the uniqued reference to the type for an
/// lvalue reference to the specified type.
QualType
ASTContext::getLValueReferenceType(QualType T, bool SpelledAsLValue) const {
  assert(getCanonicalType(T) != OverloadTy && 
         "Unresolved overloaded function type");
  
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  ReferenceType::Profile(ID, T, SpelledAsLValue);

  void *InsertPos = 0;
  if (LValueReferenceType *RT =
        LValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(RT, 0);

  const ReferenceType *InnerRef = T->getAs<ReferenceType>();

  // If the referencee type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;
  if (!SpelledAsLValue || InnerRef || !T.isCanonical()) {
    QualType PointeeType = (InnerRef ? InnerRef->getPointeeType() : T);
    Canonical = getLValueReferenceType(getCanonicalType(PointeeType));

    // Get the new insert position for the node we care about.
    LValueReferenceType *NewIP =
      LValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }

  LValueReferenceType *New
    = new (*this, TypeAlignment) LValueReferenceType(T, Canonical,
                                                     SpelledAsLValue);
  Types.push_back(New);
  LValueReferenceTypes.InsertNode(New, InsertPos);

  return QualType(New, 0);
}

/// getRValueReferenceType - Return the uniqued reference to the type for an
/// rvalue reference to the specified type.
QualType ASTContext::getRValueReferenceType(QualType T) const {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  ReferenceType::Profile(ID, T, false);

  void *InsertPos = 0;
  if (RValueReferenceType *RT =
        RValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(RT, 0);

  const ReferenceType *InnerRef = T->getAs<ReferenceType>();

  // If the referencee type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;
  if (InnerRef || !T.isCanonical()) {
    QualType PointeeType = (InnerRef ? InnerRef->getPointeeType() : T);
    Canonical = getRValueReferenceType(getCanonicalType(PointeeType));

    // Get the new insert position for the node we care about.
    RValueReferenceType *NewIP =
      RValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }

  RValueReferenceType *New
    = new (*this, TypeAlignment) RValueReferenceType(T, Canonical);
  Types.push_back(New);
  RValueReferenceTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getMemberPointerType - Return the uniqued reference to the type for a
/// member pointer to the specified type, in the specified class.
QualType ASTContext::getMemberPointerType(QualType T, const Type *Cls) const {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  MemberPointerType::Profile(ID, T, Cls);

  void *InsertPos = 0;
  if (MemberPointerType *PT =
      MemberPointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(PT, 0);

  // If the pointee or class type isn't canonical, this won't be a canonical
  // type either, so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical() || !Cls->isCanonicalUnqualified()) {
    Canonical = getMemberPointerType(getCanonicalType(T),getCanonicalType(Cls));

    // Get the new insert position for the node we care about.
    MemberPointerType *NewIP =
      MemberPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }
  MemberPointerType *New
    = new (*this, TypeAlignment) MemberPointerType(T, Cls, Canonical);
  Types.push_back(New);
  MemberPointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getConstantArrayType - Return the unique reference to the type for an
/// array of the specified element type.
QualType ASTContext::getConstantArrayType(QualType EltTy,
                                          const llvm::APInt &ArySizeIn,
                                          ArrayType::ArraySizeModifier ASM,
                                          unsigned IndexTypeQuals) const {
  assert((EltTy->isDependentType() ||
          EltTy->isIncompleteType() || EltTy->isConstantSizeType()) &&
         "Constant array of VLAs is illegal!");

  // Convert the array size into a canonical width matching the pointer size for
  // the target.
  llvm::APInt ArySize(ArySizeIn);
  ArySize =
    ArySize.zextOrTrunc(Target->getPointerWidth(getTargetAddressSpace(EltTy)));

  llvm::FoldingSetNodeID ID;
  ConstantArrayType::Profile(ID, EltTy, ArySize, ASM, IndexTypeQuals);

  void *InsertPos = 0;
  if (ConstantArrayType *ATP =
      ConstantArrayTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(ATP, 0);

  // If the element type isn't canonical or has qualifiers, this won't
  // be a canonical type either, so fill in the canonical type field.
  QualType Canon;
  if (!EltTy.isCanonical() || EltTy.hasLocalQualifiers()) {
    SplitQualType canonSplit = getCanonicalType(EltTy).split();
    Canon = getConstantArrayType(QualType(canonSplit.Ty, 0), ArySize,
                                 ASM, IndexTypeQuals);
    Canon = getQualifiedType(Canon, canonSplit.Quals);

    // Get the new insert position for the node we care about.
    ConstantArrayType *NewIP =
      ConstantArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }

  ConstantArrayType *New = new(*this,TypeAlignment)
    ConstantArrayType(EltTy, Canon, ArySize, ASM, IndexTypeQuals);
  ConstantArrayTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getVariableArrayDecayedType - Turns the given type, which may be
/// variably-modified, into the corresponding type with all the known
/// sizes replaced with [*].
QualType ASTContext::getVariableArrayDecayedType(QualType type) const {
  // Vastly most common case.
  if (!type->isVariablyModifiedType()) return type;

  QualType result;

  SplitQualType split = type.getSplitDesugaredType();
  const Type *ty = split.Ty;
  switch (ty->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    llvm_unreachable("didn't desugar past all non-canonical types?");

  // These types should never be variably-modified.
  case Type::Builtin:
  case Type::Complex:
  case Type::Vector:
  case Type::ExtVector:
  case Type::DependentSizedExtVector:
  case Type::ObjCObject:
  case Type::ObjCInterface:
  case Type::ObjCObjectPointer:
  case Type::Record:
  case Type::Enum:
  case Type::UnresolvedUsing:
  case Type::TypeOfExpr:
  case Type::TypeOf:
  case Type::Decltype:
  case Type::UnaryTransform:
  case Type::DependentName:
  case Type::InjectedClassName:
  case Type::TemplateSpecialization:
  case Type::DependentTemplateSpecialization:
  case Type::TemplateTypeParm:
  case Type::SubstTemplateTypeParmPack:
  case Type::Auto:
  case Type::PackExpansion:
    llvm_unreachable("type should never be variably-modified");

  // These types can be variably-modified but should never need to
  // further decay.
  case Type::FunctionNoProto:
  case Type::FunctionProto:
  case Type::BlockPointer:
  case Type::MemberPointer:
    return type;

  // These types can be variably-modified.  All these modifications
  // preserve structure except as noted by comments.
  // TODO: if we ever care about optimizing VLAs, there are no-op
  // optimizations available here.
  case Type::Pointer:
    result = getPointerType(getVariableArrayDecayedType(
                              cast<PointerType>(ty)->getPointeeType()));
    break;

  case Type::LValueReference: {
    const LValueReferenceType *lv = cast<LValueReferenceType>(ty);
    result = getLValueReferenceType(
                 getVariableArrayDecayedType(lv->getPointeeType()),
                                    lv->isSpelledAsLValue());
    break;
  }

  case Type::RValueReference: {
    const RValueReferenceType *lv = cast<RValueReferenceType>(ty);
    result = getRValueReferenceType(
                 getVariableArrayDecayedType(lv->getPointeeType()));
    break;
  }

  case Type::Atomic: {
    const AtomicType *at = cast<AtomicType>(ty);
    result = getAtomicType(getVariableArrayDecayedType(at->getValueType()));
    break;
  }

  case Type::ConstantArray: {
    const ConstantArrayType *cat = cast<ConstantArrayType>(ty);
    result = getConstantArrayType(
                 getVariableArrayDecayedType(cat->getElementType()),
                                  cat->getSize(),
                                  cat->getSizeModifier(),
                                  cat->getIndexTypeCVRQualifiers());
    break;
  }

  case Type::DependentSizedArray: {
    const DependentSizedArrayType *dat = cast<DependentSizedArrayType>(ty);
    result = getDependentSizedArrayType(
                 getVariableArrayDecayedType(dat->getElementType()),
                                        dat->getSizeExpr(),
                                        dat->getSizeModifier(),
                                        dat->getIndexTypeCVRQualifiers(),
                                        dat->getBracketsRange());
    break;
  }

  // Turn incomplete types into [*] types.
  case Type::IncompleteArray: {
    const IncompleteArrayType *iat = cast<IncompleteArrayType>(ty);
    result = getVariableArrayType(
                 getVariableArrayDecayedType(iat->getElementType()),
                                  /*size*/ 0,
                                  ArrayType::Normal,
                                  iat->getIndexTypeCVRQualifiers(),
                                  SourceRange());
    break;
  }

  // Turn VLA types into [*] types.
  case Type::VariableArray: {
    const VariableArrayType *vat = cast<VariableArrayType>(ty);
    result = getVariableArrayType(
                 getVariableArrayDecayedType(vat->getElementType()),
                                  /*size*/ 0,
                                  ArrayType::Star,
                                  vat->getIndexTypeCVRQualifiers(),
                                  vat->getBracketsRange());
    break;
  }
  }

  // Apply the top-level qualifiers from the original.
  return getQualifiedType(result, split.Quals);
}

/// getVariableArrayType - Returns a non-unique reference to the type for a
/// variable array of the specified element type.
QualType ASTContext::getVariableArrayType(QualType EltTy,
                                          Expr *NumElts,
                                          ArrayType::ArraySizeModifier ASM,
                                          unsigned IndexTypeQuals,
                                          SourceRange Brackets) const {
  // Since we don't unique expressions, it isn't possible to unique VLA's
  // that have an expression provided for their size.
  QualType Canon;
  
  // Be sure to pull qualifiers off the element type.
  if (!EltTy.isCanonical() || EltTy.hasLocalQualifiers()) {
    SplitQualType canonSplit = getCanonicalType(EltTy).split();
    Canon = getVariableArrayType(QualType(canonSplit.Ty, 0), NumElts, ASM,
                                 IndexTypeQuals, Brackets);
    Canon = getQualifiedType(Canon, canonSplit.Quals);
  }
  
  VariableArrayType *New = new(*this, TypeAlignment)
    VariableArrayType(EltTy, Canon, NumElts, ASM, IndexTypeQuals, Brackets);

  VariableArrayTypes.push_back(New);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getDependentSizedArrayType - Returns a non-unique reference to
/// the type for a dependently-sized array of the specified element
/// type.
QualType ASTContext::getDependentSizedArrayType(QualType elementType,
                                                Expr *numElements,
                                                ArrayType::ArraySizeModifier ASM,
                                                unsigned elementTypeQuals,
                                                SourceRange brackets) const {
  assert((!numElements || numElements->isTypeDependent() || 
          numElements->isValueDependent()) &&
         "Size must be type- or value-dependent!");

  // Dependently-sized array types that do not have a specified number
  // of elements will have their sizes deduced from a dependent
  // initializer.  We do no canonicalization here at all, which is okay
  // because they can't be used in most locations.
  if (!numElements) {
    DependentSizedArrayType *newType
      = new (*this, TypeAlignment)
          DependentSizedArrayType(*this, elementType, QualType(),
                                  numElements, ASM, elementTypeQuals,
                                  brackets);
    Types.push_back(newType);
    return QualType(newType, 0);
  }

  // Otherwise, we actually build a new type every time, but we
  // also build a canonical type.

  SplitQualType canonElementType = getCanonicalType(elementType).split();

  void *insertPos = 0;
  llvm::FoldingSetNodeID ID;
  DependentSizedArrayType::Profile(ID, *this,
                                   QualType(canonElementType.Ty, 0),
                                   ASM, elementTypeQuals, numElements);

  // Look for an existing type with these properties.
  DependentSizedArrayType *canonTy =
    DependentSizedArrayTypes.FindNodeOrInsertPos(ID, insertPos);

  // If we don't have one, build one.
  if (!canonTy) {
    canonTy = new (*this, TypeAlignment)
      DependentSizedArrayType(*this, QualType(canonElementType.Ty, 0),
                              QualType(), numElements, ASM, elementTypeQuals,
                              brackets);
    DependentSizedArrayTypes.InsertNode(canonTy, insertPos);
    Types.push_back(canonTy);
  }

  // Apply qualifiers from the element type to the array.
  QualType canon = getQualifiedType(QualType(canonTy,0),
                                    canonElementType.Quals);

  // If we didn't need extra canonicalization for the element type,
  // then just use that as our result.
  if (QualType(canonElementType.Ty, 0) == elementType)
    return canon;

  // Otherwise, we need to build a type which follows the spelling
  // of the element type.
  DependentSizedArrayType *sugaredType
    = new (*this, TypeAlignment)
        DependentSizedArrayType(*this, elementType, canon, numElements,
                                ASM, elementTypeQuals, brackets);
  Types.push_back(sugaredType);
  return QualType(sugaredType, 0);
}

QualType ASTContext::getIncompleteArrayType(QualType elementType,
                                            ArrayType::ArraySizeModifier ASM,
                                            unsigned elementTypeQuals) const {
  llvm::FoldingSetNodeID ID;
  IncompleteArrayType::Profile(ID, elementType, ASM, elementTypeQuals);

  void *insertPos = 0;
  if (IncompleteArrayType *iat =
       IncompleteArrayTypes.FindNodeOrInsertPos(ID, insertPos))
    return QualType(iat, 0);

  // If the element type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.  We also have to pull
  // qualifiers off the element type.
  QualType canon;

  if (!elementType.isCanonical() || elementType.hasLocalQualifiers()) {
    SplitQualType canonSplit = getCanonicalType(elementType).split();
    canon = getIncompleteArrayType(QualType(canonSplit.Ty, 0),
                                   ASM, elementTypeQuals);
    canon = getQualifiedType(canon, canonSplit.Quals);

    // Get the new insert position for the node we care about.
    IncompleteArrayType *existing =
      IncompleteArrayTypes.FindNodeOrInsertPos(ID, insertPos);
    assert(!existing && "Shouldn't be in the map!"); (void) existing;
  }

  IncompleteArrayType *newType = new (*this, TypeAlignment)
    IncompleteArrayType(elementType, canon, ASM, elementTypeQuals);

  IncompleteArrayTypes.InsertNode(newType, insertPos);
  Types.push_back(newType);
  return QualType(newType, 0);
}

/// getVectorType - Return the unique reference to a vector type of
/// the specified element type and size. VectorType must be a built-in type.
QualType ASTContext::getVectorType(QualType vecType, unsigned NumElts,
                                   VectorType::VectorKind VecKind) const {
  assert(vecType->isBuiltinType());

  // Check if we've already instantiated a vector of this type.
  llvm::FoldingSetNodeID ID;
  VectorType::Profile(ID, vecType, NumElts, Type::Vector, VecKind);

  void *InsertPos = 0;
  if (VectorType *VTP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(VTP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!vecType.isCanonical()) {
    Canonical = getVectorType(getCanonicalType(vecType), NumElts, VecKind);

    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }
  VectorType *New = new (*this, TypeAlignment)
    VectorType(vecType, NumElts, Canonical, VecKind);
  VectorTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getExtVectorType - Return the unique reference to an extended vector type of
/// the specified element type and size. VectorType must be a built-in type.
QualType
ASTContext::getExtVectorType(QualType vecType, unsigned NumElts) const {
  assert(vecType->isBuiltinType() || vecType->isDependentType());

  // Check if we've already instantiated a vector of this type.
  llvm::FoldingSetNodeID ID;
  VectorType::Profile(ID, vecType, NumElts, Type::ExtVector,
                      VectorType::GenericVector);
  void *InsertPos = 0;
  if (VectorType *VTP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(VTP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!vecType.isCanonical()) {
    Canonical = getExtVectorType(getCanonicalType(vecType), NumElts);

    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }
  ExtVectorType *New = new (*this, TypeAlignment)
    ExtVectorType(vecType, NumElts, Canonical);
  VectorTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

QualType
ASTContext::getDependentSizedExtVectorType(QualType vecType,
                                           Expr *SizeExpr,
                                           SourceLocation AttrLoc) const {
  llvm::FoldingSetNodeID ID;
  DependentSizedExtVectorType::Profile(ID, *this, getCanonicalType(vecType),
                                       SizeExpr);

  void *InsertPos = 0;
  DependentSizedExtVectorType *Canon
    = DependentSizedExtVectorTypes.FindNodeOrInsertPos(ID, InsertPos);
  DependentSizedExtVectorType *New;
  if (Canon) {
    // We already have a canonical version of this array type; use it as
    // the canonical type for a newly-built type.
    New = new (*this, TypeAlignment)
      DependentSizedExtVectorType(*this, vecType, QualType(Canon, 0),
                                  SizeExpr, AttrLoc);
  } else {
    QualType CanonVecTy = getCanonicalType(vecType);
    if (CanonVecTy == vecType) {
      New = new (*this, TypeAlignment)
        DependentSizedExtVectorType(*this, vecType, QualType(), SizeExpr,
                                    AttrLoc);

      DependentSizedExtVectorType *CanonCheck
        = DependentSizedExtVectorTypes.FindNodeOrInsertPos(ID, InsertPos);
      assert(!CanonCheck && "Dependent-sized ext_vector canonical type broken");
      (void)CanonCheck;
      DependentSizedExtVectorTypes.InsertNode(New, InsertPos);
    } else {
      QualType Canon = getDependentSizedExtVectorType(CanonVecTy, SizeExpr,
                                                      SourceLocation());
      New = new (*this, TypeAlignment) 
        DependentSizedExtVectorType(*this, vecType, Canon, SizeExpr, AttrLoc);
    }
  }

  Types.push_back(New);
  return QualType(New, 0);
}

/// getFunctionNoProtoType - Return a K&R style C function type like 'int()'.
///
QualType
ASTContext::getFunctionNoProtoType(QualType ResultTy,
                                   const FunctionType::ExtInfo &Info) const {
  const CallingConv DefaultCC = Info.getCC();
  const CallingConv CallConv = (LangOpts.MRTD && DefaultCC == CC_Default) ?
                               CC_X86StdCall : DefaultCC;
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  FunctionNoProtoType::Profile(ID, ResultTy, Info);

  void *InsertPos = 0;
  if (FunctionNoProtoType *FT =
        FunctionNoProtoTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(FT, 0);

  QualType Canonical;
  if (!ResultTy.isCanonical() ||
      getCanonicalCallConv(CallConv) != CallConv) {
    Canonical =
      getFunctionNoProtoType(getCanonicalType(ResultTy),
                     Info.withCallingConv(getCanonicalCallConv(CallConv)));

    // Get the new insert position for the node we care about.
    FunctionNoProtoType *NewIP =
      FunctionNoProtoTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }

  FunctionProtoType::ExtInfo newInfo = Info.withCallingConv(CallConv);
  FunctionNoProtoType *New = new (*this, TypeAlignment)
    FunctionNoProtoType(ResultTy, Canonical, newInfo);
  Types.push_back(New);
  FunctionNoProtoTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getFunctionType - Return a normal function type with a typed argument
/// list.  isVariadic indicates whether the argument list includes '...'.
QualType
ASTContext::getFunctionType(QualType ResultTy,
                            const QualType *ArgArray, unsigned NumArgs,
                            const FunctionProtoType::ExtProtoInfo &EPI) const {
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  FunctionProtoType::Profile(ID, ResultTy, ArgArray, NumArgs, EPI, *this);

  void *InsertPos = 0;
  if (FunctionProtoType *FTP =
        FunctionProtoTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(FTP, 0);

  // Determine whether the type being created is already canonical or not.
  bool isCanonical =
    EPI.ExceptionSpecType == EST_None && ResultTy.isCanonical() &&
    !EPI.HasTrailingReturn;
  for (unsigned i = 0; i != NumArgs && isCanonical; ++i)
    if (!ArgArray[i].isCanonicalAsParam())
      isCanonical = false;

  const CallingConv DefaultCC = EPI.ExtInfo.getCC();
  const CallingConv CallConv = (LangOpts.MRTD && DefaultCC == CC_Default) ?
                               CC_X86StdCall : DefaultCC;

  // If this type isn't canonical, get the canonical version of it.
  // The exception spec is not part of the canonical type.
  QualType Canonical;
  if (!isCanonical || getCanonicalCallConv(CallConv) != CallConv) {
    SmallVector<QualType, 16> CanonicalArgs;
    CanonicalArgs.reserve(NumArgs);
    for (unsigned i = 0; i != NumArgs; ++i)
      CanonicalArgs.push_back(getCanonicalParamType(ArgArray[i]));

    FunctionProtoType::ExtProtoInfo CanonicalEPI = EPI;
    CanonicalEPI.HasTrailingReturn = false;
    CanonicalEPI.ExceptionSpecType = EST_None;
    CanonicalEPI.NumExceptions = 0;
    CanonicalEPI.ExtInfo
      = CanonicalEPI.ExtInfo.withCallingConv(getCanonicalCallConv(CallConv));

    Canonical = getFunctionType(getCanonicalType(ResultTy),
                                CanonicalArgs.data(), NumArgs,
                                CanonicalEPI);

    // Get the new insert position for the node we care about.
    FunctionProtoType *NewIP =
      FunctionProtoTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }

  // FunctionProtoType objects are allocated with extra bytes after
  // them for three variable size arrays at the end:
  //  - parameter types
  //  - exception types
  //  - consumed-arguments flags
  // Instead of the exception types, there could be a noexcept
  // expression.
  size_t Size = sizeof(FunctionProtoType) +
                NumArgs * sizeof(QualType);
  if (EPI.ExceptionSpecType == EST_Dynamic)
    Size += EPI.NumExceptions * sizeof(QualType);
  else if (EPI.ExceptionSpecType == EST_ComputedNoexcept) {
    Size += sizeof(Expr*);
  } else if (EPI.ExceptionSpecType == EST_Uninstantiated) {
    Size += 2 * sizeof(FunctionDecl*);
  }
  if (EPI.ConsumedArguments)
    Size += NumArgs * sizeof(bool);

  FunctionProtoType *FTP = (FunctionProtoType*) Allocate(Size, TypeAlignment);
  FunctionProtoType::ExtProtoInfo newEPI = EPI;
  newEPI.ExtInfo = EPI.ExtInfo.withCallingConv(CallConv);
  new (FTP) FunctionProtoType(ResultTy, ArgArray, NumArgs, Canonical, newEPI);
  Types.push_back(FTP);
  FunctionProtoTypes.InsertNode(FTP, InsertPos);
  return QualType(FTP, 0);
}

#ifndef NDEBUG
static bool NeedsInjectedClassNameType(const RecordDecl *D) {
  if (!isa<CXXRecordDecl>(D)) return false;
  const CXXRecordDecl *RD = cast<CXXRecordDecl>(D);
  if (isa<ClassTemplatePartialSpecializationDecl>(RD))
    return true;
  if (RD->getDescribedClassTemplate() &&
      !isa<ClassTemplateSpecializationDecl>(RD))
    return true;
  return false;
}
#endif

/// getInjectedClassNameType - Return the unique reference to the
/// injected class name type for the specified templated declaration.
QualType ASTContext::getInjectedClassNameType(CXXRecordDecl *Decl,
                                              QualType TST) const {
  assert(NeedsInjectedClassNameType(Decl));
  if (Decl->TypeForDecl) {
    assert(isa<InjectedClassNameType>(Decl->TypeForDecl));
  } else if (CXXRecordDecl *PrevDecl = Decl->getPreviousDecl()) {
    assert(PrevDecl->TypeForDecl && "previous declaration has no type");
    Decl->TypeForDecl = PrevDecl->TypeForDecl;
    assert(isa<InjectedClassNameType>(Decl->TypeForDecl));
  } else {
    Type *newType =
      new (*this, TypeAlignment) InjectedClassNameType(Decl, TST);
    Decl->TypeForDecl = newType;
    Types.push_back(newType);
  }
  return QualType(Decl->TypeForDecl, 0);
}

/// getTypeDeclType - Return the unique reference to the type for the
/// specified type declaration.
QualType ASTContext::getTypeDeclTypeSlow(const TypeDecl *Decl) const {
  assert(Decl && "Passed null for Decl param");
  assert(!Decl->TypeForDecl && "TypeForDecl present in slow case");

  if (const TypedefNameDecl *Typedef = dyn_cast<TypedefNameDecl>(Decl))
    return getTypedefType(Typedef);

  assert(!isa<TemplateTypeParmDecl>(Decl) &&
         "Template type parameter types are always available.");

  if (const RecordDecl *Record = dyn_cast<RecordDecl>(Decl)) {
    assert(!Record->getPreviousDecl() &&
           "struct/union has previous declaration");
    assert(!NeedsInjectedClassNameType(Record));
    return getRecordType(Record);
  } else if (const EnumDecl *Enum = dyn_cast<EnumDecl>(Decl)) {
    assert(!Enum->getPreviousDecl() &&
           "enum has previous declaration");
    return getEnumType(Enum);
  } else if (const UnresolvedUsingTypenameDecl *Using =
               dyn_cast<UnresolvedUsingTypenameDecl>(Decl)) {
    Type *newType = new (*this, TypeAlignment) UnresolvedUsingType(Using);
    Decl->TypeForDecl = newType;
    Types.push_back(newType);
  } else
    llvm_unreachable("TypeDecl without a type?");

  return QualType(Decl->TypeForDecl, 0);
}

/// getTypedefType - Return the unique reference to the type for the
/// specified typedef name decl.
QualType
ASTContext::getTypedefType(const TypedefNameDecl *Decl,
                           QualType Canonical) const {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);

  if (Canonical.isNull())
    Canonical = getCanonicalType(Decl->getUnderlyingType());
  TypedefType *newType = new(*this, TypeAlignment)
    TypedefType(Type::Typedef, Decl, Canonical);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType ASTContext::getRecordType(const RecordDecl *Decl) const {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);

  if (const RecordDecl *PrevDecl = Decl->getPreviousDecl())
    if (PrevDecl->TypeForDecl)
      return QualType(Decl->TypeForDecl = PrevDecl->TypeForDecl, 0); 

  RecordType *newType = new (*this, TypeAlignment) RecordType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType ASTContext::getEnumType(const EnumDecl *Decl) const {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);

  if (const EnumDecl *PrevDecl = Decl->getPreviousDecl())
    if (PrevDecl->TypeForDecl)
      return QualType(Decl->TypeForDecl = PrevDecl->TypeForDecl, 0); 

  EnumType *newType = new (*this, TypeAlignment) EnumType(Decl);
  Decl->TypeForDecl = newType;
  Types.push_back(newType);
  return QualType(newType, 0);
}

QualType ASTContext::getAttributedType(AttributedType::Kind attrKind,
                                       QualType modifiedType,
                                       QualType equivalentType) {
  llvm::FoldingSetNodeID id;
  AttributedType::Profile(id, attrKind, modifiedType, equivalentType);

  void *insertPos = 0;
  AttributedType *type = AttributedTypes.FindNodeOrInsertPos(id, insertPos);
  if (type) return QualType(type, 0);

  QualType canon = getCanonicalType(equivalentType);
  type = new (*this, TypeAlignment)
           AttributedType(canon, attrKind, modifiedType, equivalentType);

  Types.push_back(type);
  AttributedTypes.InsertNode(type, insertPos);

  return QualType(type, 0);
}


/// \brief Retrieve a substitution-result type.
QualType
ASTContext::getSubstTemplateTypeParmType(const TemplateTypeParmType *Parm,
                                         QualType Replacement) const {
  assert(Replacement.isCanonical()
         && "replacement types must always be canonical");

  llvm::FoldingSetNodeID ID;
  SubstTemplateTypeParmType::Profile(ID, Parm, Replacement);
  void *InsertPos = 0;
  SubstTemplateTypeParmType *SubstParm
    = SubstTemplateTypeParmTypes.FindNodeOrInsertPos(ID, InsertPos);

  if (!SubstParm) {
    SubstParm = new (*this, TypeAlignment)
      SubstTemplateTypeParmType(Parm, Replacement);
    Types.push_back(SubstParm);
    SubstTemplateTypeParmTypes.InsertNode(SubstParm, InsertPos);
  }

  return QualType(SubstParm, 0);
}

/// \brief Retrieve a 
QualType ASTContext::getSubstTemplateTypeParmPackType(
                                          const TemplateTypeParmType *Parm,
                                              const TemplateArgument &ArgPack) {
#ifndef NDEBUG
  for (TemplateArgument::pack_iterator P = ArgPack.pack_begin(), 
                                    PEnd = ArgPack.pack_end();
       P != PEnd; ++P) {
    assert(P->getKind() == TemplateArgument::Type &&"Pack contains a non-type");
    assert(P->getAsType().isCanonical() && "Pack contains non-canonical type");
  }
#endif
  
  llvm::FoldingSetNodeID ID;
  SubstTemplateTypeParmPackType::Profile(ID, Parm, ArgPack);
  void *InsertPos = 0;
  if (SubstTemplateTypeParmPackType *SubstParm
        = SubstTemplateTypeParmPackTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(SubstParm, 0);
  
  QualType Canon;
  if (!Parm->isCanonicalUnqualified()) {
    Canon = getCanonicalType(QualType(Parm, 0));
    Canon = getSubstTemplateTypeParmPackType(cast<TemplateTypeParmType>(Canon),
                                             ArgPack);
    SubstTemplateTypeParmPackTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  SubstTemplateTypeParmPackType *SubstParm
    = new (*this, TypeAlignment) SubstTemplateTypeParmPackType(Parm, Canon,
                                                               ArgPack);
  Types.push_back(SubstParm);
  SubstTemplateTypeParmTypes.InsertNode(SubstParm, InsertPos);
  return QualType(SubstParm, 0);  
}

/// \brief Retrieve the template type parameter type for a template
/// parameter or parameter pack with the given depth, index, and (optionally)
/// name.
QualType ASTContext::getTemplateTypeParmType(unsigned Depth, unsigned Index,
                                             bool ParameterPack,
                                             TemplateTypeParmDecl *TTPDecl) const {
  llvm::FoldingSetNodeID ID;
  TemplateTypeParmType::Profile(ID, Depth, Index, ParameterPack, TTPDecl);
  void *InsertPos = 0;
  TemplateTypeParmType *TypeParm
    = TemplateTypeParmTypes.FindNodeOrInsertPos(ID, InsertPos);

  if (TypeParm)
    return QualType(TypeParm, 0);

  if (TTPDecl) {
    QualType Canon = getTemplateTypeParmType(Depth, Index, ParameterPack);
    TypeParm = new (*this, TypeAlignment) TemplateTypeParmType(TTPDecl, Canon);

    TemplateTypeParmType *TypeCheck 
      = TemplateTypeParmTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(!TypeCheck && "Template type parameter canonical type broken");
    (void)TypeCheck;
  } else
    TypeParm = new (*this, TypeAlignment)
      TemplateTypeParmType(Depth, Index, ParameterPack);

  Types.push_back(TypeParm);
  TemplateTypeParmTypes.InsertNode(TypeParm, InsertPos);

  return QualType(TypeParm, 0);
}

TypeSourceInfo *
ASTContext::getTemplateSpecializationTypeInfo(TemplateName Name,
                                              SourceLocation NameLoc,
                                        const TemplateArgumentListInfo &Args,
                                              QualType Underlying) const {
  assert(!Name.getAsDependentTemplateName() && 
         "No dependent template names here!");
  QualType TST = getTemplateSpecializationType(Name, Args, Underlying);

  TypeSourceInfo *DI = CreateTypeSourceInfo(TST);
  TemplateSpecializationTypeLoc TL
    = cast<TemplateSpecializationTypeLoc>(DI->getTypeLoc());
  TL.setTemplateKeywordLoc(SourceLocation());
  TL.setTemplateNameLoc(NameLoc);
  TL.setLAngleLoc(Args.getLAngleLoc());
  TL.setRAngleLoc(Args.getRAngleLoc());
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    TL.setArgLocInfo(i, Args[i].getLocInfo());
  return DI;
}

QualType
ASTContext::getTemplateSpecializationType(TemplateName Template,
                                          const TemplateArgumentListInfo &Args,
                                          QualType Underlying) const {
  assert(!Template.getAsDependentTemplateName() && 
         "No dependent template names here!");
  
  unsigned NumArgs = Args.size();

  SmallVector<TemplateArgument, 4> ArgVec;
  ArgVec.reserve(NumArgs);
  for (unsigned i = 0; i != NumArgs; ++i)
    ArgVec.push_back(Args[i].getArgument());

  return getTemplateSpecializationType(Template, ArgVec.data(), NumArgs,
                                       Underlying);
}

#ifndef NDEBUG
static bool hasAnyPackExpansions(const TemplateArgument *Args,
                                 unsigned NumArgs) {
  for (unsigned I = 0; I != NumArgs; ++I)
    if (Args[I].isPackExpansion())
      return true;
  
  return true;
}
#endif

QualType
ASTContext::getTemplateSpecializationType(TemplateName Template,
                                          const TemplateArgument *Args,
                                          unsigned NumArgs,
                                          QualType Underlying) const {
  assert(!Template.getAsDependentTemplateName() && 
         "No dependent template names here!");
  // Look through qualified template names.
  if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
    Template = TemplateName(QTN->getTemplateDecl());
  
  bool IsTypeAlias = 
    Template.getAsTemplateDecl() &&
    isa<TypeAliasTemplateDecl>(Template.getAsTemplateDecl());
  QualType CanonType;
  if (!Underlying.isNull())
    CanonType = getCanonicalType(Underlying);
  else {
    // We can get here with an alias template when the specialization contains
    // a pack expansion that does not match up with a parameter pack.
    assert((!IsTypeAlias || hasAnyPackExpansions(Args, NumArgs)) &&
           "Caller must compute aliased type");
    IsTypeAlias = false;
    CanonType = getCanonicalTemplateSpecializationType(Template, Args,
                                                       NumArgs);
  }

  // Allocate the (non-canonical) template specialization type, but don't
  // try to unique it: these types typically have location information that
  // we don't unique and don't want to lose.
  void *Mem = Allocate(sizeof(TemplateSpecializationType) +
                       sizeof(TemplateArgument) * NumArgs +
                       (IsTypeAlias? sizeof(QualType) : 0),
                       TypeAlignment);
  TemplateSpecializationType *Spec
    = new (Mem) TemplateSpecializationType(Template, Args, NumArgs, CanonType,
                                         IsTypeAlias ? Underlying : QualType());

  Types.push_back(Spec);
  return QualType(Spec, 0);
}

QualType
ASTContext::getCanonicalTemplateSpecializationType(TemplateName Template,
                                                   const TemplateArgument *Args,
                                                   unsigned NumArgs) const {
  assert(!Template.getAsDependentTemplateName() && 
         "No dependent template names here!");

  // Look through qualified template names.
  if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
    Template = TemplateName(QTN->getTemplateDecl());
  
  // Build the canonical template specialization type.
  TemplateName CanonTemplate = getCanonicalTemplateName(Template);
  SmallVector<TemplateArgument, 4> CanonArgs;
  CanonArgs.reserve(NumArgs);
  for (unsigned I = 0; I != NumArgs; ++I)
    CanonArgs.push_back(getCanonicalTemplateArgument(Args[I]));

  // Determine whether this canonical template specialization type already
  // exists.
  llvm::FoldingSetNodeID ID;
  TemplateSpecializationType::Profile(ID, CanonTemplate,
                                      CanonArgs.data(), NumArgs, *this);

  void *InsertPos = 0;
  TemplateSpecializationType *Spec
    = TemplateSpecializationTypes.FindNodeOrInsertPos(ID, InsertPos);

  if (!Spec) {
    // Allocate a new canonical template specialization type.
    void *Mem = Allocate((sizeof(TemplateSpecializationType) +
                          sizeof(TemplateArgument) * NumArgs),
                         TypeAlignment);
    Spec = new (Mem) TemplateSpecializationType(CanonTemplate,
                                                CanonArgs.data(), NumArgs,
                                                QualType(), QualType());
    Types.push_back(Spec);
    TemplateSpecializationTypes.InsertNode(Spec, InsertPos);
  }

  assert(Spec->isDependentType() &&
         "Non-dependent template-id type must have a canonical type");
  return QualType(Spec, 0);
}

QualType
ASTContext::getElaboratedType(ElaboratedTypeKeyword Keyword,
                              NestedNameSpecifier *NNS,
                              QualType NamedType) const {
  llvm::FoldingSetNodeID ID;
  ElaboratedType::Profile(ID, Keyword, NNS, NamedType);

  void *InsertPos = 0;
  ElaboratedType *T = ElaboratedTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  QualType Canon = NamedType;
  if (!Canon.isCanonical()) {
    Canon = getCanonicalType(NamedType);
    ElaboratedType *CheckT = ElaboratedTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckT && "Elaborated canonical type broken");
    (void)CheckT;
  }

  T = new (*this) ElaboratedType(Keyword, NNS, NamedType, Canon);
  Types.push_back(T);
  ElaboratedTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

QualType
ASTContext::getParenType(QualType InnerType) const {
  llvm::FoldingSetNodeID ID;
  ParenType::Profile(ID, InnerType);

  void *InsertPos = 0;
  ParenType *T = ParenTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  QualType Canon = InnerType;
  if (!Canon.isCanonical()) {
    Canon = getCanonicalType(InnerType);
    ParenType *CheckT = ParenTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckT && "Paren canonical type broken");
    (void)CheckT;
  }

  T = new (*this) ParenType(InnerType, Canon);
  Types.push_back(T);
  ParenTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

QualType ASTContext::getDependentNameType(ElaboratedTypeKeyword Keyword,
                                          NestedNameSpecifier *NNS,
                                          const IdentifierInfo *Name,
                                          QualType Canon) const {
  assert(NNS->isDependent() && "nested-name-specifier must be dependent");

  if (Canon.isNull()) {
    NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);
    ElaboratedTypeKeyword CanonKeyword = Keyword;
    if (Keyword == ETK_None)
      CanonKeyword = ETK_Typename;
    
    if (CanonNNS != NNS || CanonKeyword != Keyword)
      Canon = getDependentNameType(CanonKeyword, CanonNNS, Name);
  }

  llvm::FoldingSetNodeID ID;
  DependentNameType::Profile(ID, Keyword, NNS, Name);

  void *InsertPos = 0;
  DependentNameType *T
    = DependentNameTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  T = new (*this) DependentNameType(Keyword, NNS, Name, Canon);
  Types.push_back(T);
  DependentNameTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

QualType
ASTContext::getDependentTemplateSpecializationType(
                                 ElaboratedTypeKeyword Keyword,
                                 NestedNameSpecifier *NNS,
                                 const IdentifierInfo *Name,
                                 const TemplateArgumentListInfo &Args) const {
  // TODO: avoid this copy
  SmallVector<TemplateArgument, 16> ArgCopy;
  for (unsigned I = 0, E = Args.size(); I != E; ++I)
    ArgCopy.push_back(Args[I].getArgument());
  return getDependentTemplateSpecializationType(Keyword, NNS, Name,
                                                ArgCopy.size(),
                                                ArgCopy.data());
}

QualType
ASTContext::getDependentTemplateSpecializationType(
                                 ElaboratedTypeKeyword Keyword,
                                 NestedNameSpecifier *NNS,
                                 const IdentifierInfo *Name,
                                 unsigned NumArgs,
                                 const TemplateArgument *Args) const {
  assert((!NNS || NNS->isDependent()) && 
         "nested-name-specifier must be dependent");

  llvm::FoldingSetNodeID ID;
  DependentTemplateSpecializationType::Profile(ID, *this, Keyword, NNS,
                                               Name, NumArgs, Args);

  void *InsertPos = 0;
  DependentTemplateSpecializationType *T
    = DependentTemplateSpecializationTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);

  ElaboratedTypeKeyword CanonKeyword = Keyword;
  if (Keyword == ETK_None) CanonKeyword = ETK_Typename;

  bool AnyNonCanonArgs = false;
  SmallVector<TemplateArgument, 16> CanonArgs(NumArgs);
  for (unsigned I = 0; I != NumArgs; ++I) {
    CanonArgs[I] = getCanonicalTemplateArgument(Args[I]);
    if (!CanonArgs[I].structurallyEquals(Args[I]))
      AnyNonCanonArgs = true;
  }

  QualType Canon;
  if (AnyNonCanonArgs || CanonNNS != NNS || CanonKeyword != Keyword) {
    Canon = getDependentTemplateSpecializationType(CanonKeyword, CanonNNS,
                                                   Name, NumArgs,
                                                   CanonArgs.data());

    // Find the insert position again.
    DependentTemplateSpecializationTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  void *Mem = Allocate((sizeof(DependentTemplateSpecializationType) +
                        sizeof(TemplateArgument) * NumArgs),
                       TypeAlignment);
  T = new (Mem) DependentTemplateSpecializationType(Keyword, NNS,
                                                    Name, NumArgs, Args, Canon);
  Types.push_back(T);
  DependentTemplateSpecializationTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

QualType ASTContext::getPackExpansionType(QualType Pattern,
                                      llvm::Optional<unsigned> NumExpansions) {
  llvm::FoldingSetNodeID ID;
  PackExpansionType::Profile(ID, Pattern, NumExpansions);

  assert(Pattern->containsUnexpandedParameterPack() &&
         "Pack expansions must expand one or more parameter packs");
  void *InsertPos = 0;
  PackExpansionType *T
    = PackExpansionTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  QualType Canon;
  if (!Pattern.isCanonical()) {
    Canon = getPackExpansionType(getCanonicalType(Pattern), NumExpansions);

    // Find the insert position again.
    PackExpansionTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  T = new (*this) PackExpansionType(Pattern, Canon, NumExpansions);
  Types.push_back(T);
  PackExpansionTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);  
}

/// CmpProtocolNames - Comparison predicate for sorting protocols
/// alphabetically.
static bool CmpProtocolNames(const ObjCProtocolDecl *LHS,
                            const ObjCProtocolDecl *RHS) {
  return LHS->getDeclName() < RHS->getDeclName();
}

static bool areSortedAndUniqued(ObjCProtocolDecl * const *Protocols,
                                unsigned NumProtocols) {
  if (NumProtocols == 0) return true;

  if (Protocols[0]->getCanonicalDecl() != Protocols[0])
    return false;
  
  for (unsigned i = 1; i != NumProtocols; ++i)
    if (!CmpProtocolNames(Protocols[i-1], Protocols[i]) ||
        Protocols[i]->getCanonicalDecl() != Protocols[i])
      return false;
  return true;
}

static void SortAndUniqueProtocols(ObjCProtocolDecl **Protocols,
                                   unsigned &NumProtocols) {
  ObjCProtocolDecl **ProtocolsEnd = Protocols+NumProtocols;

  // Sort protocols, keyed by name.
  std::sort(Protocols, Protocols+NumProtocols, CmpProtocolNames);

  // Canonicalize.
  for (unsigned I = 0, N = NumProtocols; I != N; ++I)
    Protocols[I] = Protocols[I]->getCanonicalDecl();
  
  // Remove duplicates.
  ProtocolsEnd = std::unique(Protocols, ProtocolsEnd);
  NumProtocols = ProtocolsEnd-Protocols;
}

QualType ASTContext::getObjCObjectType(QualType BaseType,
                                       ObjCProtocolDecl * const *Protocols,
                                       unsigned NumProtocols) const {
  // If the base type is an interface and there aren't any protocols
  // to add, then the interface type will do just fine.
  if (!NumProtocols && isa<ObjCInterfaceType>(BaseType))
    return BaseType;

  // Look in the folding set for an existing type.
  llvm::FoldingSetNodeID ID;
  ObjCObjectTypeImpl::Profile(ID, BaseType, Protocols, NumProtocols);
  void *InsertPos = 0;
  if (ObjCObjectType *QT = ObjCObjectTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(QT, 0);

  // Build the canonical type, which has the canonical base type and
  // a sorted-and-uniqued list of protocols.
  QualType Canonical;
  bool ProtocolsSorted = areSortedAndUniqued(Protocols, NumProtocols);
  if (!ProtocolsSorted || !BaseType.isCanonical()) {
    if (!ProtocolsSorted) {
      SmallVector<ObjCProtocolDecl*, 8> Sorted(Protocols,
                                                     Protocols + NumProtocols);
      unsigned UniqueCount = NumProtocols;

      SortAndUniqueProtocols(&Sorted[0], UniqueCount);
      Canonical = getObjCObjectType(getCanonicalType(BaseType),
                                    &Sorted[0], UniqueCount);
    } else {
      Canonical = getObjCObjectType(getCanonicalType(BaseType),
                                    Protocols, NumProtocols);
    }

    // Regenerate InsertPos.
    ObjCObjectTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  unsigned Size = sizeof(ObjCObjectTypeImpl);
  Size += NumProtocols * sizeof(ObjCProtocolDecl *);
  void *Mem = Allocate(Size, TypeAlignment);
  ObjCObjectTypeImpl *T =
    new (Mem) ObjCObjectTypeImpl(Canonical, BaseType, Protocols, NumProtocols);

  Types.push_back(T);
  ObjCObjectTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

/// getObjCObjectPointerType - Return a ObjCObjectPointerType type for
/// the given object type.
QualType ASTContext::getObjCObjectPointerType(QualType ObjectT) const {
  llvm::FoldingSetNodeID ID;
  ObjCObjectPointerType::Profile(ID, ObjectT);

  void *InsertPos = 0;
  if (ObjCObjectPointerType *QT =
              ObjCObjectPointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(QT, 0);

  // Find the canonical object type.
  QualType Canonical;
  if (!ObjectT.isCanonical()) {
    Canonical = getObjCObjectPointerType(getCanonicalType(ObjectT));

    // Regenerate InsertPos.
    ObjCObjectPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  // No match.
  void *Mem = Allocate(sizeof(ObjCObjectPointerType), TypeAlignment);
  ObjCObjectPointerType *QType =
    new (Mem) ObjCObjectPointerType(Canonical, ObjectT);

  Types.push_back(QType);
  ObjCObjectPointerTypes.InsertNode(QType, InsertPos);
  return QualType(QType, 0);
}

/// getObjCInterfaceType - Return the unique reference to the type for the
/// specified ObjC interface decl. The list of protocols is optional.
QualType ASTContext::getObjCInterfaceType(const ObjCInterfaceDecl *Decl,
                                          ObjCInterfaceDecl *PrevDecl) const {
  if (Decl->TypeForDecl)
    return QualType(Decl->TypeForDecl, 0);

  if (PrevDecl) {
    assert(PrevDecl->TypeForDecl && "previous decl has no TypeForDecl");
    Decl->TypeForDecl = PrevDecl->TypeForDecl;
    return QualType(PrevDecl->TypeForDecl, 0);
  }

  // Prefer the definition, if there is one.
  if (const ObjCInterfaceDecl *Def = Decl->getDefinition())
    Decl = Def;
  
  void *Mem = Allocate(sizeof(ObjCInterfaceType), TypeAlignment);
  ObjCInterfaceType *T = new (Mem) ObjCInterfaceType(Decl);
  Decl->TypeForDecl = T;
  Types.push_back(T);
  return QualType(T, 0);
}

/// getTypeOfExprType - Unlike many "get<Type>" functions, we can't unique
/// TypeOfExprType AST's (since expression's are never shared). For example,
/// multiple declarations that refer to "typeof(x)" all contain different
/// DeclRefExpr's. This doesn't effect the type checker, since it operates
/// on canonical type's (which are always unique).
QualType ASTContext::getTypeOfExprType(Expr *tofExpr) const {
  TypeOfExprType *toe;
  if (tofExpr->isTypeDependent()) {
    llvm::FoldingSetNodeID ID;
    DependentTypeOfExprType::Profile(ID, *this, tofExpr);

    void *InsertPos = 0;
    DependentTypeOfExprType *Canon
      = DependentTypeOfExprTypes.FindNodeOrInsertPos(ID, InsertPos);
    if (Canon) {
      // We already have a "canonical" version of an identical, dependent
      // typeof(expr) type. Use that as our canonical type.
      toe = new (*this, TypeAlignment) TypeOfExprType(tofExpr,
                                          QualType((TypeOfExprType*)Canon, 0));
    } else {
      // Build a new, canonical typeof(expr) type.
      Canon
        = new (*this, TypeAlignment) DependentTypeOfExprType(*this, tofExpr);
      DependentTypeOfExprTypes.InsertNode(Canon, InsertPos);
      toe = Canon;
    }
  } else {
    QualType Canonical = getCanonicalType(tofExpr->getType());
    toe = new (*this, TypeAlignment) TypeOfExprType(tofExpr, Canonical);
  }
  Types.push_back(toe);
  return QualType(toe, 0);
}

/// getTypeOfType -  Unlike many "get<Type>" functions, we don't unique
/// TypeOfType AST's. The only motivation to unique these nodes would be
/// memory savings. Since typeof(t) is fairly uncommon, space shouldn't be
/// an issue. This doesn't effect the type checker, since it operates
/// on canonical type's (which are always unique).
QualType ASTContext::getTypeOfType(QualType tofType) const {
  QualType Canonical = getCanonicalType(tofType);
  TypeOfType *tot = new (*this, TypeAlignment) TypeOfType(tofType, Canonical);
  Types.push_back(tot);
  return QualType(tot, 0);
}


/// getDecltypeType -  Unlike many "get<Type>" functions, we don't unique
/// DecltypeType AST's. The only motivation to unique these nodes would be
/// memory savings. Since decltype(t) is fairly uncommon, space shouldn't be
/// an issue. This doesn't effect the type checker, since it operates
/// on canonical types (which are always unique).
QualType ASTContext::getDecltypeType(Expr *e, QualType UnderlyingType) const {
  DecltypeType *dt;
  
  // C++0x [temp.type]p2:
  //   If an expression e involves a template parameter, decltype(e) denotes a
  //   unique dependent type. Two such decltype-specifiers refer to the same 
  //   type only if their expressions are equivalent (14.5.6.1). 
  if (e->isInstantiationDependent()) {
    llvm::FoldingSetNodeID ID;
    DependentDecltypeType::Profile(ID, *this, e);

    void *InsertPos = 0;
    DependentDecltypeType *Canon
      = DependentDecltypeTypes.FindNodeOrInsertPos(ID, InsertPos);
    if (Canon) {
      // We already have a "canonical" version of an equivalent, dependent
      // decltype type. Use that as our canonical type.
      dt = new (*this, TypeAlignment) DecltypeType(e, DependentTy,
                                       QualType((DecltypeType*)Canon, 0));
    } else {
      // Build a new, canonical typeof(expr) type.
      Canon = new (*this, TypeAlignment) DependentDecltypeType(*this, e);
      DependentDecltypeTypes.InsertNode(Canon, InsertPos);
      dt = Canon;
    }
  } else {
    dt = new (*this, TypeAlignment) DecltypeType(e, UnderlyingType, 
                                      getCanonicalType(UnderlyingType));
  }
  Types.push_back(dt);
  return QualType(dt, 0);
}

/// getUnaryTransformationType - We don't unique these, since the memory
/// savings are minimal and these are rare.
QualType ASTContext::getUnaryTransformType(QualType BaseType,
                                           QualType UnderlyingType,
                                           UnaryTransformType::UTTKind Kind)
    const {
  UnaryTransformType *Ty =
    new (*this, TypeAlignment) UnaryTransformType (BaseType, UnderlyingType, 
                                                   Kind,
                                 UnderlyingType->isDependentType() ?
                                 QualType() : getCanonicalType(UnderlyingType));
  Types.push_back(Ty);
  return QualType(Ty, 0);
}

/// getAutoType - We only unique auto types after they've been deduced.
QualType ASTContext::getAutoType(QualType DeducedType) const {
  void *InsertPos = 0;
  if (!DeducedType.isNull()) {
    // Look in the folding set for an existing type.
    llvm::FoldingSetNodeID ID;
    AutoType::Profile(ID, DeducedType);
    if (AutoType *AT = AutoTypes.FindNodeOrInsertPos(ID, InsertPos))
      return QualType(AT, 0);
  }

  AutoType *AT = new (*this, TypeAlignment) AutoType(DeducedType);
  Types.push_back(AT);
  if (InsertPos)
    AutoTypes.InsertNode(AT, InsertPos);
  return QualType(AT, 0);
}

/// getAtomicType - Return the uniqued reference to the atomic type for
/// the given value type.
QualType ASTContext::getAtomicType(QualType T) const {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  AtomicType::Profile(ID, T);

  void *InsertPos = 0;
  if (AtomicType *AT = AtomicTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(AT, 0);

  // If the atomic value type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical()) {
    Canonical = getAtomicType(getCanonicalType(T));

    // Get the new insert position for the node we care about.
    AtomicType *NewIP = AtomicTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); (void)NewIP;
  }
  AtomicType *New = new (*this, TypeAlignment) AtomicType(T, Canonical);
  Types.push_back(New);
  AtomicTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getAutoDeductType - Get type pattern for deducing against 'auto'.
QualType ASTContext::getAutoDeductType() const {
  if (AutoDeductTy.isNull())
    AutoDeductTy = getAutoType(QualType());
  assert(!AutoDeductTy.isNull() && "can't build 'auto' pattern");
  return AutoDeductTy;
}

/// getAutoRRefDeductType - Get type pattern for deducing against 'auto &&'.
QualType ASTContext::getAutoRRefDeductType() const {
  if (AutoRRefDeductTy.isNull())
    AutoRRefDeductTy = getRValueReferenceType(getAutoDeductType());
  assert(!AutoRRefDeductTy.isNull() && "can't build 'auto &&' pattern");
  return AutoRRefDeductTy;
}

/// getTagDeclType - Return the unique reference to the type for the
/// specified TagDecl (struct/union/class/enum) decl.
QualType ASTContext::getTagDeclType(const TagDecl *Decl) const {
  assert (Decl);
  // FIXME: What is the design on getTagDeclType when it requires casting
  // away const?  mutable?
  return getTypeDeclType(const_cast<TagDecl*>(Decl));
}

/// getSizeType - Return the unique type for "size_t" (C99 7.17), the result
/// of the sizeof operator (C99 6.5.3.4p4). The value is target dependent and
/// needs to agree with the definition in <stddef.h>.
CanQualType ASTContext::getSizeType() const {
  return getFromTargetType(Target->getSizeType());
}

/// getIntMaxType - Return the unique type for "intmax_t" (C99 7.18.1.5).
CanQualType ASTContext::getIntMaxType() const {
  return getFromTargetType(Target->getIntMaxType());
}

/// getUIntMaxType - Return the unique type for "uintmax_t" (C99 7.18.1.5).
CanQualType ASTContext::getUIntMaxType() const {
  return getFromTargetType(Target->getUIntMaxType());
}

/// getSignedWCharType - Return the type of "signed wchar_t".
/// Used when in C++, as a GCC extension.
QualType ASTContext::getSignedWCharType() const {
  // FIXME: derive from "Target" ?
  return WCharTy;
}

/// getUnsignedWCharType - Return the type of "unsigned wchar_t".
/// Used when in C++, as a GCC extension.
QualType ASTContext::getUnsignedWCharType() const {
  // FIXME: derive from "Target" ?
  return UnsignedIntTy;
}

/// getPointerDiffType - Return the unique type for "ptrdiff_t" (C99 7.17)
/// defined in <stddef.h>. Pointer - pointer requires this (C99 6.5.6p9).
QualType ASTContext::getPointerDiffType() const {
  return getFromTargetType(Target->getPtrDiffType(0));
}

//===----------------------------------------------------------------------===//
//                              Type Operators
//===----------------------------------------------------------------------===//

CanQualType ASTContext::getCanonicalParamType(QualType T) const {
  // Push qualifiers into arrays, and then discard any remaining
  // qualifiers.
  T = getCanonicalType(T);
  T = getVariableArrayDecayedType(T);
  const Type *Ty = T.getTypePtr();
  QualType Result;
  if (isa<ArrayType>(Ty)) {
    Result = getArrayDecayedType(QualType(Ty,0));
  } else if (isa<FunctionType>(Ty)) {
    Result = getPointerType(QualType(Ty, 0));
  } else {
    Result = QualType(Ty, 0);
  }

  return CanQualType::CreateUnsafe(Result);
}

QualType ASTContext::getUnqualifiedArrayType(QualType type,
                                             Qualifiers &quals) {
  SplitQualType splitType = type.getSplitUnqualifiedType();

  // FIXME: getSplitUnqualifiedType() actually walks all the way to
  // the unqualified desugared type and then drops it on the floor.
  // We then have to strip that sugar back off with
  // getUnqualifiedDesugaredType(), which is silly.
  const ArrayType *AT =
    dyn_cast<ArrayType>(splitType.Ty->getUnqualifiedDesugaredType());

  // If we don't have an array, just use the results in splitType.
  if (!AT) {
    quals = splitType.Quals;
    return QualType(splitType.Ty, 0);
  }

  // Otherwise, recurse on the array's element type.
  QualType elementType = AT->getElementType();
  QualType unqualElementType = getUnqualifiedArrayType(elementType, quals);

  // If that didn't change the element type, AT has no qualifiers, so we
  // can just use the results in splitType.
  if (elementType == unqualElementType) {
    assert(quals.empty()); // from the recursive call
    quals = splitType.Quals;
    return QualType(splitType.Ty, 0);
  }

  // Otherwise, add in the qualifiers from the outermost type, then
  // build the type back up.
  quals.addConsistentQualifiers(splitType.Quals);

  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT)) {
    return getConstantArrayType(unqualElementType, CAT->getSize(),
                                CAT->getSizeModifier(), 0);
  }

  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(AT)) {
    return getIncompleteArrayType(unqualElementType, IAT->getSizeModifier(), 0);
  }

  if (const VariableArrayType *VAT = dyn_cast<VariableArrayType>(AT)) {
    return getVariableArrayType(unqualElementType,
                                VAT->getSizeExpr(),
                                VAT->getSizeModifier(),
                                VAT->getIndexTypeCVRQualifiers(),
                                VAT->getBracketsRange());
  }

  const DependentSizedArrayType *DSAT = cast<DependentSizedArrayType>(AT);
  return getDependentSizedArrayType(unqualElementType, DSAT->getSizeExpr(),
                                    DSAT->getSizeModifier(), 0,
                                    SourceRange());
}

/// UnwrapSimilarPointerTypes - If T1 and T2 are pointer types  that
/// may be similar (C++ 4.4), replaces T1 and T2 with the type that
/// they point to and return true. If T1 and T2 aren't pointer types
/// or pointer-to-member types, or if they are not similar at this
/// level, returns false and leaves T1 and T2 unchanged. Top-level
/// qualifiers on T1 and T2 are ignored. This function will typically
/// be called in a loop that successively "unwraps" pointer and
/// pointer-to-member types to compare them at each level.
bool ASTContext::UnwrapSimilarPointerTypes(QualType &T1, QualType &T2) {
  const PointerType *T1PtrType = T1->getAs<PointerType>(),
                    *T2PtrType = T2->getAs<PointerType>();
  if (T1PtrType && T2PtrType) {
    T1 = T1PtrType->getPointeeType();
    T2 = T2PtrType->getPointeeType();
    return true;
  }
  
  const MemberPointerType *T1MPType = T1->getAs<MemberPointerType>(),
                          *T2MPType = T2->getAs<MemberPointerType>();
  if (T1MPType && T2MPType && 
      hasSameUnqualifiedType(QualType(T1MPType->getClass(), 0), 
                             QualType(T2MPType->getClass(), 0))) {
    T1 = T1MPType->getPointeeType();
    T2 = T2MPType->getPointeeType();
    return true;
  }
  
  if (getLangOpts().ObjC1) {
    const ObjCObjectPointerType *T1OPType = T1->getAs<ObjCObjectPointerType>(),
                                *T2OPType = T2->getAs<ObjCObjectPointerType>();
    if (T1OPType && T2OPType) {
      T1 = T1OPType->getPointeeType();
      T2 = T2OPType->getPointeeType();
      return true;
    }
  }
  
  // FIXME: Block pointers, too?
  
  return false;
}

DeclarationNameInfo
ASTContext::getNameForTemplate(TemplateName Name,
                               SourceLocation NameLoc) const {
  switch (Name.getKind()) {
  case TemplateName::QualifiedTemplate:
  case TemplateName::Template:
    // DNInfo work in progress: CHECKME: what about DNLoc?
    return DeclarationNameInfo(Name.getAsTemplateDecl()->getDeclName(),
                               NameLoc);

  case TemplateName::OverloadedTemplate: {
    OverloadedTemplateStorage *Storage = Name.getAsOverloadedTemplate();
    // DNInfo work in progress: CHECKME: what about DNLoc?
    return DeclarationNameInfo((*Storage->begin())->getDeclName(), NameLoc);
  }

  case TemplateName::DependentTemplate: {
    DependentTemplateName *DTN = Name.getAsDependentTemplateName();
    DeclarationName DName;
    if (DTN->isIdentifier()) {
      DName = DeclarationNames.getIdentifier(DTN->getIdentifier());
      return DeclarationNameInfo(DName, NameLoc);
    } else {
      DName = DeclarationNames.getCXXOperatorName(DTN->getOperator());
      // DNInfo work in progress: FIXME: source locations?
      DeclarationNameLoc DNLoc;
      DNLoc.CXXOperatorName.BeginOpNameLoc = SourceLocation().getRawEncoding();
      DNLoc.CXXOperatorName.EndOpNameLoc = SourceLocation().getRawEncoding();
      return DeclarationNameInfo(DName, NameLoc, DNLoc);
    }
  }

  case TemplateName::SubstTemplateTemplateParm: {
    SubstTemplateTemplateParmStorage *subst
      = Name.getAsSubstTemplateTemplateParm();
    return DeclarationNameInfo(subst->getParameter()->getDeclName(),
                               NameLoc);
  }

  case TemplateName::SubstTemplateTemplateParmPack: {
    SubstTemplateTemplateParmPackStorage *subst
      = Name.getAsSubstTemplateTemplateParmPack();
    return DeclarationNameInfo(subst->getParameterPack()->getDeclName(),
                               NameLoc);
  }
  }

  llvm_unreachable("bad template name kind!");
}

TemplateName ASTContext::getCanonicalTemplateName(TemplateName Name) const {
  switch (Name.getKind()) {
  case TemplateName::QualifiedTemplate:
  case TemplateName::Template: {
    TemplateDecl *Template = Name.getAsTemplateDecl();
    if (TemplateTemplateParmDecl *TTP 
          = dyn_cast<TemplateTemplateParmDecl>(Template))
      Template = getCanonicalTemplateTemplateParmDecl(TTP);
  
    // The canonical template name is the canonical template declaration.
    return TemplateName(cast<TemplateDecl>(Template->getCanonicalDecl()));
  }

  case TemplateName::OverloadedTemplate:
    llvm_unreachable("cannot canonicalize overloaded template");

  case TemplateName::DependentTemplate: {
    DependentTemplateName *DTN = Name.getAsDependentTemplateName();
    assert(DTN && "Non-dependent template names must refer to template decls.");
    return DTN->CanonicalTemplateName;
  }

  case TemplateName::SubstTemplateTemplateParm: {
    SubstTemplateTemplateParmStorage *subst
      = Name.getAsSubstTemplateTemplateParm();
    return getCanonicalTemplateName(subst->getReplacement());
  }

  case TemplateName::SubstTemplateTemplateParmPack: {
    SubstTemplateTemplateParmPackStorage *subst
                                  = Name.getAsSubstTemplateTemplateParmPack();
    TemplateTemplateParmDecl *canonParameter
      = getCanonicalTemplateTemplateParmDecl(subst->getParameterPack());
    TemplateArgument canonArgPack
      = getCanonicalTemplateArgument(subst->getArgumentPack());
    return getSubstTemplateTemplateParmPack(canonParameter, canonArgPack);
  }
  }

  llvm_unreachable("bad template name!");
}

bool ASTContext::hasSameTemplateName(TemplateName X, TemplateName Y) {
  X = getCanonicalTemplateName(X);
  Y = getCanonicalTemplateName(Y);
  return X.getAsVoidPointer() == Y.getAsVoidPointer();
}

TemplateArgument
ASTContext::getCanonicalTemplateArgument(const TemplateArgument &Arg) const {
  switch (Arg.getKind()) {
    case TemplateArgument::Null:
      return Arg;

    case TemplateArgument::Expression:
      return Arg;

    case TemplateArgument::Declaration: {
      if (Decl *D = Arg.getAsDecl())
          return TemplateArgument(D->getCanonicalDecl());
      return TemplateArgument((Decl*)0);
    }

    case TemplateArgument::Template:
      return TemplateArgument(getCanonicalTemplateName(Arg.getAsTemplate()));

    case TemplateArgument::TemplateExpansion:
      return TemplateArgument(getCanonicalTemplateName(
                                         Arg.getAsTemplateOrTemplatePattern()),
                              Arg.getNumTemplateExpansions());

    case TemplateArgument::Integral:
      return TemplateArgument(*Arg.getAsIntegral(),
                              getCanonicalType(Arg.getIntegralType()));

    case TemplateArgument::Type:
      return TemplateArgument(getCanonicalType(Arg.getAsType()));

    case TemplateArgument::Pack: {
      if (Arg.pack_size() == 0)
        return Arg;
      
      TemplateArgument *CanonArgs
        = new (*this) TemplateArgument[Arg.pack_size()];
      unsigned Idx = 0;
      for (TemplateArgument::pack_iterator A = Arg.pack_begin(),
                                        AEnd = Arg.pack_end();
           A != AEnd; (void)++A, ++Idx)
        CanonArgs[Idx] = getCanonicalTemplateArgument(*A);

      return TemplateArgument(CanonArgs, Arg.pack_size());
    }
  }

  // Silence GCC warning
  llvm_unreachable("Unhandled template argument kind");
}

NestedNameSpecifier *
ASTContext::getCanonicalNestedNameSpecifier(NestedNameSpecifier *NNS) const {
  if (!NNS)
    return 0;

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    // Canonicalize the prefix but keep the identifier the same.
    return NestedNameSpecifier::Create(*this,
                         getCanonicalNestedNameSpecifier(NNS->getPrefix()),
                                       NNS->getAsIdentifier());

  case NestedNameSpecifier::Namespace:
    // A namespace is canonical; build a nested-name-specifier with
    // this namespace and no prefix.
    return NestedNameSpecifier::Create(*this, 0, 
                                 NNS->getAsNamespace()->getOriginalNamespace());

  case NestedNameSpecifier::NamespaceAlias:
    // A namespace is canonical; build a nested-name-specifier with
    // this namespace and no prefix.
    return NestedNameSpecifier::Create(*this, 0, 
                                    NNS->getAsNamespaceAlias()->getNamespace()
                                                      ->getOriginalNamespace());

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
    QualType T = getCanonicalType(QualType(NNS->getAsType(), 0));
    
    // If we have some kind of dependent-named type (e.g., "typename T::type"),
    // break it apart into its prefix and identifier, then reconsititute those
    // as the canonical nested-name-specifier. This is required to canonicalize
    // a dependent nested-name-specifier involving typedefs of dependent-name
    // types, e.g.,
    //   typedef typename T::type T1;
    //   typedef typename T1::type T2;
    if (const DependentNameType *DNT = T->getAs<DependentNameType>())
      return NestedNameSpecifier::Create(*this, DNT->getQualifier(), 
                           const_cast<IdentifierInfo *>(DNT->getIdentifier()));

    // Otherwise, just canonicalize the type, and force it to be a TypeSpec.
    // FIXME: Why are TypeSpec and TypeSpecWithTemplate distinct in the
    // first place?
    return NestedNameSpecifier::Create(*this, 0, false,
                                       const_cast<Type*>(T.getTypePtr()));
  }

  case NestedNameSpecifier::Global:
    // The global specifier is canonical and unique.
    return NNS;
  }

  llvm_unreachable("Invalid NestedNameSpecifier::Kind!");
}


const ArrayType *ASTContext::getAsArrayType(QualType T) const {
  // Handle the non-qualified case efficiently.
  if (!T.hasLocalQualifiers()) {
    // Handle the common positive case fast.
    if (const ArrayType *AT = dyn_cast<ArrayType>(T))
      return AT;
  }

  // Handle the common negative case fast.
  if (!isa<ArrayType>(T.getCanonicalType()))
    return 0;

  // Apply any qualifiers from the array type to the element type.  This
  // implements C99 6.7.3p8: "If the specification of an array type includes
  // any type qualifiers, the element type is so qualified, not the array type."

  // If we get here, we either have type qualifiers on the type, or we have
  // sugar such as a typedef in the way.  If we have type qualifiers on the type
  // we must propagate them down into the element type.

  SplitQualType split = T.getSplitDesugaredType();
  Qualifiers qs = split.Quals;

  // If we have a simple case, just return now.
  const ArrayType *ATy = dyn_cast<ArrayType>(split.Ty);
  if (ATy == 0 || qs.empty())
    return ATy;

  // Otherwise, we have an array and we have qualifiers on it.  Push the
  // qualifiers into the array element type and return a new array type.
  QualType NewEltTy = getQualifiedType(ATy->getElementType(), qs);

  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(ATy))
    return cast<ArrayType>(getConstantArrayType(NewEltTy, CAT->getSize(),
                                                CAT->getSizeModifier(),
                                           CAT->getIndexTypeCVRQualifiers()));
  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(ATy))
    return cast<ArrayType>(getIncompleteArrayType(NewEltTy,
                                                  IAT->getSizeModifier(),
                                           IAT->getIndexTypeCVRQualifiers()));

  if (const DependentSizedArrayType *DSAT
        = dyn_cast<DependentSizedArrayType>(ATy))
    return cast<ArrayType>(
                     getDependentSizedArrayType(NewEltTy,
                                                DSAT->getSizeExpr(),
                                                DSAT->getSizeModifier(),
                                              DSAT->getIndexTypeCVRQualifiers(),
                                                DSAT->getBracketsRange()));

  const VariableArrayType *VAT = cast<VariableArrayType>(ATy);
  return cast<ArrayType>(getVariableArrayType(NewEltTy,
                                              VAT->getSizeExpr(),
                                              VAT->getSizeModifier(),
                                              VAT->getIndexTypeCVRQualifiers(),
                                              VAT->getBracketsRange()));
}

QualType ASTContext::getAdjustedParameterType(QualType T) {
  // C99 6.7.5.3p7:
  //   A declaration of a parameter as "array of type" shall be
  //   adjusted to "qualified pointer to type", where the type
  //   qualifiers (if any) are those specified within the [ and ] of
  //   the array type derivation.
  if (T->isArrayType())
    return getArrayDecayedType(T);
  
  // C99 6.7.5.3p8:
  //   A declaration of a parameter as "function returning type"
  //   shall be adjusted to "pointer to function returning type", as
  //   in 6.3.2.1.
  if (T->isFunctionType())
    return getPointerType(T);
  
  return T;  
}

QualType ASTContext::getSignatureParameterType(QualType T) {
  T = getVariableArrayDecayedType(T);
  T = getAdjustedParameterType(T);
  return T.getUnqualifiedType();
}

/// getArrayDecayedType - Return the properly qualified result of decaying the
/// specified array type to a pointer.  This operation is non-trivial when
/// handling typedefs etc.  The canonical type of "T" must be an array type,
/// this returns a pointer to a properly qualified element of the array.
///
/// See C99 6.7.5.3p7 and C99 6.3.2.1p3.
QualType ASTContext::getArrayDecayedType(QualType Ty) const {
  // Get the element type with 'getAsArrayType' so that we don't lose any
  // typedefs in the element type of the array.  This also handles propagation
  // of type qualifiers from the array type into the element type if present
  // (C99 6.7.3p8).
  const ArrayType *PrettyArrayType = getAsArrayType(Ty);
  assert(PrettyArrayType && "Not an array type!");

  QualType PtrTy = getPointerType(PrettyArrayType->getElementType());

  // int x[restrict 4] ->  int *restrict
  return getQualifiedType(PtrTy, PrettyArrayType->getIndexTypeQualifiers());
}

QualType ASTContext::getBaseElementType(const ArrayType *array) const {
  return getBaseElementType(array->getElementType());
}

QualType ASTContext::getBaseElementType(QualType type) const {
  Qualifiers qs;
  while (true) {
    SplitQualType split = type.getSplitDesugaredType();
    const ArrayType *array = split.Ty->getAsArrayTypeUnsafe();
    if (!array) break;

    type = array->getElementType();
    qs.addConsistentQualifiers(split.Quals);
  }

  return getQualifiedType(type, qs);
}

/// getConstantArrayElementCount - Returns number of constant array elements.
uint64_t
ASTContext::getConstantArrayElementCount(const ConstantArrayType *CA)  const {
  uint64_t ElementCount = 1;
  do {
    ElementCount *= CA->getSize().getZExtValue();
    CA = dyn_cast<ConstantArrayType>(CA->getElementType());
  } while (CA);
  return ElementCount;
}

/// getFloatingRank - Return a relative rank for floating point types.
/// This routine will assert if passed a built-in type that isn't a float.
static FloatingRank getFloatingRank(QualType T) {
  if (const ComplexType *CT = T->getAs<ComplexType>())
    return getFloatingRank(CT->getElementType());

  assert(T->getAs<BuiltinType>() && "getFloatingRank(): not a floating type");
  switch (T->getAs<BuiltinType>()->getKind()) {
  default: llvm_unreachable("getFloatingRank(): not a floating type");
  case BuiltinType::Half:       return HalfRank;
  case BuiltinType::Float:      return FloatRank;
  case BuiltinType::Double:     return DoubleRank;
  case BuiltinType::LongDouble: return LongDoubleRank;
  }
}

/// getFloatingTypeOfSizeWithinDomain - Returns a real floating
/// point or a complex type (based on typeDomain/typeSize).
/// 'typeDomain' is a real floating point or complex type.
/// 'typeSize' is a real floating point or complex type.
QualType ASTContext::getFloatingTypeOfSizeWithinDomain(QualType Size,
                                                       QualType Domain) const {
  FloatingRank EltRank = getFloatingRank(Size);
  if (Domain->isComplexType()) {
    switch (EltRank) {
    case HalfRank: llvm_unreachable("Complex half is not supported");
    case FloatRank:      return FloatComplexTy;
    case DoubleRank:     return DoubleComplexTy;
    case LongDoubleRank: return LongDoubleComplexTy;
    }
  }

  assert(Domain->isRealFloatingType() && "Unknown domain!");
  switch (EltRank) {
  case HalfRank: llvm_unreachable("Half ranks are not valid here");
  case FloatRank:      return FloatTy;
  case DoubleRank:     return DoubleTy;
  case LongDoubleRank: return LongDoubleTy;
  }
  llvm_unreachable("getFloatingRank(): illegal value for rank");
}

/// getFloatingTypeOrder - Compare the rank of the two specified floating
/// point types, ignoring the domain of the type (i.e. 'double' ==
/// '_Complex double').  If LHS > RHS, return 1.  If LHS == RHS, return 0. If
/// LHS < RHS, return -1.
int ASTContext::getFloatingTypeOrder(QualType LHS, QualType RHS) const {
  FloatingRank LHSR = getFloatingRank(LHS);
  FloatingRank RHSR = getFloatingRank(RHS);

  if (LHSR == RHSR)
    return 0;
  if (LHSR > RHSR)
    return 1;
  return -1;
}

/// getIntegerRank - Return an integer conversion rank (C99 6.3.1.1p1). This
/// routine will assert if passed a built-in type that isn't an integer or enum,
/// or if it is not canonicalized.
unsigned ASTContext::getIntegerRank(const Type *T) const {
  assert(T->isCanonicalUnqualified() && "T should be canonicalized");

  switch (cast<BuiltinType>(T)->getKind()) {
  default: llvm_unreachable("getIntegerRank(): not a built-in integer");
  case BuiltinType::Bool:
    return 1 + (getIntWidth(BoolTy) << 3);
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
    return 2 + (getIntWidth(CharTy) << 3);
  case BuiltinType::Short:
  case BuiltinType::UShort:
    return 3 + (getIntWidth(ShortTy) << 3);
  case BuiltinType::Int:
  case BuiltinType::UInt:
    return 4 + (getIntWidth(IntTy) << 3);
  case BuiltinType::Long:
  case BuiltinType::ULong:
    return 5 + (getIntWidth(LongTy) << 3);
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
    return 6 + (getIntWidth(LongLongTy) << 3);
  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return 7 + (getIntWidth(Int128Ty) << 3);
  }
}

/// \brief Whether this is a promotable bitfield reference according
/// to C99 6.3.1.1p2, bullet 2 (and GCC extensions).
///
/// \returns the type this bit-field will promote to, or NULL if no
/// promotion occurs.
QualType ASTContext::isPromotableBitField(Expr *E) const {
  if (E->isTypeDependent() || E->isValueDependent())
    return QualType();
  
  FieldDecl *Field = E->getBitField();
  if (!Field)
    return QualType();

  QualType FT = Field->getType();

  uint64_t BitWidth = Field->getBitWidthValue(*this);
  uint64_t IntSize = getTypeSize(IntTy);
  // GCC extension compatibility: if the bit-field size is less than or equal
  // to the size of int, it gets promoted no matter what its type is.
  // For instance, unsigned long bf : 4 gets promoted to signed int.
  if (BitWidth < IntSize)
    return IntTy;

  if (BitWidth == IntSize)
    return FT->isSignedIntegerType() ? IntTy : UnsignedIntTy;

  // Types bigger than int are not subject to promotions, and therefore act
  // like the base type.
  // FIXME: This doesn't quite match what gcc does, but what gcc does here
  // is ridiculous.
  return QualType();
}

/// getPromotedIntegerType - Returns the type that Promotable will
/// promote to: C99 6.3.1.1p2, assuming that Promotable is a promotable
/// integer type.
QualType ASTContext::getPromotedIntegerType(QualType Promotable) const {
  assert(!Promotable.isNull());
  assert(Promotable->isPromotableIntegerType());
  if (const EnumType *ET = Promotable->getAs<EnumType>())
    return ET->getDecl()->getPromotionType();

  if (const BuiltinType *BT = Promotable->getAs<BuiltinType>()) {
    // C++ [conv.prom]: A prvalue of type char16_t, char32_t, or wchar_t
    // (3.9.1) can be converted to a prvalue of the first of the following
    // types that can represent all the values of its underlying type:
    // int, unsigned int, long int, unsigned long int, long long int, or
    // unsigned long long int [...]
    // FIXME: Is there some better way to compute this?
    if (BT->getKind() == BuiltinType::WChar_S ||
        BT->getKind() == BuiltinType::WChar_U ||
        BT->getKind() == BuiltinType::Char16 ||
        BT->getKind() == BuiltinType::Char32) {
      bool FromIsSigned = BT->getKind() == BuiltinType::WChar_S;
      uint64_t FromSize = getTypeSize(BT);
      QualType PromoteTypes[] = { IntTy, UnsignedIntTy, LongTy, UnsignedLongTy,
                                  LongLongTy, UnsignedLongLongTy };
      for (size_t Idx = 0; Idx < llvm::array_lengthof(PromoteTypes); ++Idx) {
        uint64_t ToSize = getTypeSize(PromoteTypes[Idx]);
        if (FromSize < ToSize ||
            (FromSize == ToSize &&
             FromIsSigned == PromoteTypes[Idx]->isSignedIntegerType()))
          return PromoteTypes[Idx];
      }
      llvm_unreachable("char type should fit into long long");
    }
  }

  // At this point, we should have a signed or unsigned integer type.
  if (Promotable->isSignedIntegerType())
    return IntTy;
  uint64_t PromotableSize = getTypeSize(Promotable);
  uint64_t IntSize = getTypeSize(IntTy);
  assert(Promotable->isUnsignedIntegerType() && PromotableSize <= IntSize);
  return (PromotableSize != IntSize) ? IntTy : UnsignedIntTy;
}

/// \brief Recurses in pointer/array types until it finds an objc retainable
/// type and returns its ownership.
Qualifiers::ObjCLifetime ASTContext::getInnerObjCOwnership(QualType T) const {
  while (!T.isNull()) {
    if (T.getObjCLifetime() != Qualifiers::OCL_None)
      return T.getObjCLifetime();
    if (T->isArrayType())
      T = getBaseElementType(T);
    else if (const PointerType *PT = T->getAs<PointerType>())
      T = PT->getPointeeType();
    else if (const ReferenceType *RT = T->getAs<ReferenceType>())
      T = RT->getPointeeType();
    else
      break;
  }

  return Qualifiers::OCL_None;
}

/// getIntegerTypeOrder - Returns the highest ranked integer type:
/// C99 6.3.1.8p1.  If LHS > RHS, return 1.  If LHS == RHS, return 0. If
/// LHS < RHS, return -1.
int ASTContext::getIntegerTypeOrder(QualType LHS, QualType RHS) const {
  const Type *LHSC = getCanonicalType(LHS).getTypePtr();
  const Type *RHSC = getCanonicalType(RHS).getTypePtr();
  if (LHSC == RHSC) return 0;

  bool LHSUnsigned = LHSC->isUnsignedIntegerType();
  bool RHSUnsigned = RHSC->isUnsignedIntegerType();

  unsigned LHSRank = getIntegerRank(LHSC);
  unsigned RHSRank = getIntegerRank(RHSC);

  if (LHSUnsigned == RHSUnsigned) {  // Both signed or both unsigned.
    if (LHSRank == RHSRank) return 0;
    return LHSRank > RHSRank ? 1 : -1;
  }

  // Otherwise, the LHS is signed and the RHS is unsigned or visa versa.
  if (LHSUnsigned) {
    // If the unsigned [LHS] type is larger, return it.
    if (LHSRank >= RHSRank)
      return 1;

    // If the signed type can represent all values of the unsigned type, it
    // wins.  Because we are dealing with 2's complement and types that are
    // powers of two larger than each other, this is always safe.
    return -1;
  }

  // If the unsigned [RHS] type is larger, return it.
  if (RHSRank >= LHSRank)
    return -1;

  // If the signed type can represent all values of the unsigned type, it
  // wins.  Because we are dealing with 2's complement and types that are
  // powers of two larger than each other, this is always safe.
  return 1;
}

static RecordDecl *
CreateRecordDecl(const ASTContext &Ctx, RecordDecl::TagKind TK,
                 DeclContext *DC, IdentifierInfo *Id) {
  SourceLocation Loc;
  if (Ctx.getLangOpts().CPlusPlus)
    return CXXRecordDecl::Create(Ctx, TK, DC, Loc, Loc, Id);
  else
    return RecordDecl::Create(Ctx, TK, DC, Loc, Loc, Id);
}

// getCFConstantStringType - Return the type used for constant CFStrings.
QualType ASTContext::getCFConstantStringType() const {
  if (!CFConstantStringTypeDecl) {
    CFConstantStringTypeDecl =
      CreateRecordDecl(*this, TTK_Struct, TUDecl,
                       &Idents.get("NSConstantString"));
    CFConstantStringTypeDecl->startDefinition();

    QualType FieldTypes[4];

    // const int *isa;
    FieldTypes[0] = getPointerType(IntTy.withConst());
    // int flags;
    FieldTypes[1] = IntTy;
    // const char *str;
    FieldTypes[2] = getPointerType(CharTy.withConst());
    // long length;
    FieldTypes[3] = LongTy;

    // Create fields
    for (unsigned i = 0; i < 4; ++i) {
      FieldDecl *Field = FieldDecl::Create(*this, CFConstantStringTypeDecl,
                                           SourceLocation(),
                                           SourceLocation(), 0,
                                           FieldTypes[i], /*TInfo=*/0,
                                           /*BitWidth=*/0,
                                           /*Mutable=*/false,
                                           /*HasInit=*/false);
      Field->setAccess(AS_public);
      CFConstantStringTypeDecl->addDecl(Field);
    }

    CFConstantStringTypeDecl->completeDefinition();
  }

  return getTagDeclType(CFConstantStringTypeDecl);
}

void ASTContext::setCFConstantStringType(QualType T) {
  const RecordType *Rec = T->getAs<RecordType>();
  assert(Rec && "Invalid CFConstantStringType");
  CFConstantStringTypeDecl = Rec->getDecl();
}

QualType ASTContext::getBlockDescriptorType() const {
  if (BlockDescriptorType)
    return getTagDeclType(BlockDescriptorType);

  RecordDecl *T;
  // FIXME: Needs the FlagAppleBlock bit.
  T = CreateRecordDecl(*this, TTK_Struct, TUDecl,
                       &Idents.get("__block_descriptor"));
  T->startDefinition();
  
  QualType FieldTypes[] = {
    UnsignedLongTy,
    UnsignedLongTy,
  };

  const char *FieldNames[] = {
    "reserved",
    "Size"
  };

  for (size_t i = 0; i < 2; ++i) {
    FieldDecl *Field = FieldDecl::Create(*this, T, SourceLocation(),
                                         SourceLocation(),
                                         &Idents.get(FieldNames[i]),
                                         FieldTypes[i], /*TInfo=*/0,
                                         /*BitWidth=*/0,
                                         /*Mutable=*/false,
                                         /*HasInit=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  T->completeDefinition();

  BlockDescriptorType = T;

  return getTagDeclType(BlockDescriptorType);
}

QualType ASTContext::getBlockDescriptorExtendedType() const {
  if (BlockDescriptorExtendedType)
    return getTagDeclType(BlockDescriptorExtendedType);

  RecordDecl *T;
  // FIXME: Needs the FlagAppleBlock bit.
  T = CreateRecordDecl(*this, TTK_Struct, TUDecl,
                       &Idents.get("__block_descriptor_withcopydispose"));
  T->startDefinition();
  
  QualType FieldTypes[] = {
    UnsignedLongTy,
    UnsignedLongTy,
    getPointerType(VoidPtrTy),
    getPointerType(VoidPtrTy)
  };

  const char *FieldNames[] = {
    "reserved",
    "Size",
    "CopyFuncPtr",
    "DestroyFuncPtr"
  };

  for (size_t i = 0; i < 4; ++i) {
    FieldDecl *Field = FieldDecl::Create(*this, T, SourceLocation(),
                                         SourceLocation(),
                                         &Idents.get(FieldNames[i]),
                                         FieldTypes[i], /*TInfo=*/0,
                                         /*BitWidth=*/0,
                                         /*Mutable=*/false,
                                         /*HasInit=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  T->completeDefinition();

  BlockDescriptorExtendedType = T;

  return getTagDeclType(BlockDescriptorExtendedType);
}

bool ASTContext::BlockRequiresCopying(QualType Ty) const {
  if (Ty->isObjCRetainableType())
    return true;
  if (getLangOpts().CPlusPlus) {
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      return RD->hasConstCopyConstructor();
      
    }
  }
  return false;
}

QualType
ASTContext::BuildByRefType(StringRef DeclName, QualType Ty) const {
  //  type = struct __Block_byref_1_X {
  //    void *__isa;
  //    struct __Block_byref_1_X *__forwarding;
  //    unsigned int __flags;
  //    unsigned int __size;
  //    void *__copy_helper;            // as needed
  //    void *__destroy_help            // as needed
  //    int X;
  //  } *

  bool HasCopyAndDispose = BlockRequiresCopying(Ty);

  // FIXME: Move up
  SmallString<36> Name;
  llvm::raw_svector_ostream(Name) << "__Block_byref_" <<
                                  ++UniqueBlockByRefTypeID << '_' << DeclName;
  RecordDecl *T;
  T = CreateRecordDecl(*this, TTK_Struct, TUDecl, &Idents.get(Name.str()));
  T->startDefinition();
  QualType Int32Ty = IntTy;
  assert(getIntWidth(IntTy) == 32 && "non-32bit int not supported");
  QualType FieldTypes[] = {
    getPointerType(VoidPtrTy),
    getPointerType(getTagDeclType(T)),
    Int32Ty,
    Int32Ty,
    getPointerType(VoidPtrTy),
    getPointerType(VoidPtrTy),
    Ty
  };

  StringRef FieldNames[] = {
    "__isa",
    "__forwarding",
    "__flags",
    "__size",
    "__copy_helper",
    "__destroy_helper",
    DeclName,
  };

  for (size_t i = 0; i < 7; ++i) {
    if (!HasCopyAndDispose && i >=4 && i <= 5)
      continue;
    FieldDecl *Field = FieldDecl::Create(*this, T, SourceLocation(),
                                         SourceLocation(),
                                         &Idents.get(FieldNames[i]),
                                         FieldTypes[i], /*TInfo=*/0,
                                         /*BitWidth=*/0, /*Mutable=*/false,
                                         /*HasInit=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  T->completeDefinition();

  return getPointerType(getTagDeclType(T));
}

TypedefDecl *ASTContext::getObjCInstanceTypeDecl() {
  if (!ObjCInstanceTypeDecl)
    ObjCInstanceTypeDecl = TypedefDecl::Create(*this, 
                                               getTranslationUnitDecl(),
                                               SourceLocation(), 
                                               SourceLocation(),
                                               &Idents.get("instancetype"), 
                                     getTrivialTypeSourceInfo(getObjCIdType()));
  return ObjCInstanceTypeDecl;
}

// This returns true if a type has been typedefed to BOOL:
// typedef <type> BOOL;
static bool isTypeTypedefedAsBOOL(QualType T) {
  if (const TypedefType *TT = dyn_cast<TypedefType>(T))
    if (IdentifierInfo *II = TT->getDecl()->getIdentifier())
      return II->isStr("BOOL");

  return false;
}

/// getObjCEncodingTypeSize returns size of type for objective-c encoding
/// purpose.
CharUnits ASTContext::getObjCEncodingTypeSize(QualType type) const {
  if (!type->isIncompleteArrayType() && type->isIncompleteType())
    return CharUnits::Zero();
  
  CharUnits sz = getTypeSizeInChars(type);

  // Make all integer and enum types at least as large as an int
  if (sz.isPositive() && type->isIntegralOrEnumerationType())
    sz = std::max(sz, getTypeSizeInChars(IntTy));
  // Treat arrays as pointers, since that's how they're passed in.
  else if (type->isArrayType())
    sz = getTypeSizeInChars(VoidPtrTy);
  return sz;
}

static inline 
std::string charUnitsToString(const CharUnits &CU) {
  return llvm::itostr(CU.getQuantity());
}

/// getObjCEncodingForBlock - Return the encoded type for this block
/// declaration.
std::string ASTContext::getObjCEncodingForBlock(const BlockExpr *Expr) const {
  std::string S;

  const BlockDecl *Decl = Expr->getBlockDecl();
  QualType BlockTy =
      Expr->getType()->getAs<BlockPointerType>()->getPointeeType();
  // Encode result type.
  getObjCEncodingForType(BlockTy->getAs<FunctionType>()->getResultType(), S);
  // Compute size of all parameters.
  // Start with computing size of a pointer in number of bytes.
  // FIXME: There might(should) be a better way of doing this computation!
  SourceLocation Loc;
  CharUnits PtrSize = getTypeSizeInChars(VoidPtrTy);
  CharUnits ParmOffset = PtrSize;
  for (BlockDecl::param_const_iterator PI = Decl->param_begin(),
       E = Decl->param_end(); PI != E; ++PI) {
    QualType PType = (*PI)->getType();
    CharUnits sz = getObjCEncodingTypeSize(PType);
    assert (sz.isPositive() && "BlockExpr - Incomplete param type");
    ParmOffset += sz;
  }
  // Size of the argument frame
  S += charUnitsToString(ParmOffset);
  // Block pointer and offset.
  S += "@?0";
  
  // Argument types.
  ParmOffset = PtrSize;
  for (BlockDecl::param_const_iterator PI = Decl->param_begin(), E =
       Decl->param_end(); PI != E; ++PI) {
    ParmVarDecl *PVDecl = *PI;
    QualType PType = PVDecl->getOriginalType(); 
    if (const ArrayType *AT =
          dyn_cast<ArrayType>(PType->getCanonicalTypeInternal())) {
      // Use array's original type only if it has known number of
      // elements.
      if (!isa<ConstantArrayType>(AT))
        PType = PVDecl->getType();
    } else if (PType->isFunctionType())
      PType = PVDecl->getType();
    getObjCEncodingForType(PType, S);
    S += charUnitsToString(ParmOffset);
    ParmOffset += getObjCEncodingTypeSize(PType);
  }

  return S;
}

bool ASTContext::getObjCEncodingForFunctionDecl(const FunctionDecl *Decl,
                                                std::string& S) {
  // Encode result type.
  getObjCEncodingForType(Decl->getResultType(), S);
  CharUnits ParmOffset;
  // Compute size of all parameters.
  for (FunctionDecl::param_const_iterator PI = Decl->param_begin(),
       E = Decl->param_end(); PI != E; ++PI) {
    QualType PType = (*PI)->getType();
    CharUnits sz = getObjCEncodingTypeSize(PType);
    if (sz.isZero())
      return true;
    
    assert (sz.isPositive() && 
        "getObjCEncodingForFunctionDecl - Incomplete param type");
    ParmOffset += sz;
  }
  S += charUnitsToString(ParmOffset);
  ParmOffset = CharUnits::Zero();

  // Argument types.
  for (FunctionDecl::param_const_iterator PI = Decl->param_begin(),
       E = Decl->param_end(); PI != E; ++PI) {
    ParmVarDecl *PVDecl = *PI;
    QualType PType = PVDecl->getOriginalType();
    if (const ArrayType *AT =
          dyn_cast<ArrayType>(PType->getCanonicalTypeInternal())) {
      // Use array's original type only if it has known number of
      // elements.
      if (!isa<ConstantArrayType>(AT))
        PType = PVDecl->getType();
    } else if (PType->isFunctionType())
      PType = PVDecl->getType();
    getObjCEncodingForType(PType, S);
    S += charUnitsToString(ParmOffset);
    ParmOffset += getObjCEncodingTypeSize(PType);
  }
  
  return false;
}

/// getObjCEncodingForMethodParameter - Return the encoded type for a single
/// method parameter or return type. If Extended, include class names and 
/// block object types.
void ASTContext::getObjCEncodingForMethodParameter(Decl::ObjCDeclQualifier QT,
                                                   QualType T, std::string& S,
                                                   bool Extended) const {
  // Encode type qualifer, 'in', 'inout', etc. for the parameter.
  getObjCEncodingForTypeQualifier(QT, S);
  // Encode parameter type.
  getObjCEncodingForTypeImpl(T, S, true, true, 0,
                             true     /*OutermostType*/,
                             false    /*EncodingProperty*/, 
                             false    /*StructField*/, 
                             Extended /*EncodeBlockParameters*/, 
                             Extended /*EncodeClassNames*/);
}

/// getObjCEncodingForMethodDecl - Return the encoded type for this method
/// declaration.
bool ASTContext::getObjCEncodingForMethodDecl(const ObjCMethodDecl *Decl,
                                              std::string& S, 
                                              bool Extended) const {
  // FIXME: This is not very efficient.
  // Encode return type.
  getObjCEncodingForMethodParameter(Decl->getObjCDeclQualifier(), 
                                    Decl->getResultType(), S, Extended);
  // Compute size of all parameters.
  // Start with computing size of a pointer in number of bytes.
  // FIXME: There might(should) be a better way of doing this computation!
  SourceLocation Loc;
  CharUnits PtrSize = getTypeSizeInChars(VoidPtrTy);
  // The first two arguments (self and _cmd) are pointers; account for
  // their size.
  CharUnits ParmOffset = 2 * PtrSize;
  for (ObjCMethodDecl::param_const_iterator PI = Decl->param_begin(),
       E = Decl->sel_param_end(); PI != E; ++PI) {
    QualType PType = (*PI)->getType();
    CharUnits sz = getObjCEncodingTypeSize(PType);
    if (sz.isZero())
      return true;
    
    assert (sz.isPositive() && 
        "getObjCEncodingForMethodDecl - Incomplete param type");
    ParmOffset += sz;
  }
  S += charUnitsToString(ParmOffset);
  S += "@0:";
  S += charUnitsToString(PtrSize);

  // Argument types.
  ParmOffset = 2 * PtrSize;
  for (ObjCMethodDecl::param_const_iterator PI = Decl->param_begin(),
       E = Decl->sel_param_end(); PI != E; ++PI) {
    const ParmVarDecl *PVDecl = *PI;
    QualType PType = PVDecl->getOriginalType();
    if (const ArrayType *AT =
          dyn_cast<ArrayType>(PType->getCanonicalTypeInternal())) {
      // Use array's original type only if it has known number of
      // elements.
      if (!isa<ConstantArrayType>(AT))
        PType = PVDecl->getType();
    } else if (PType->isFunctionType())
      PType = PVDecl->getType();
    getObjCEncodingForMethodParameter(PVDecl->getObjCDeclQualifier(), 
                                      PType, S, Extended);
    S += charUnitsToString(ParmOffset);
    ParmOffset += getObjCEncodingTypeSize(PType);
  }
  
  return false;
}

/// getObjCEncodingForPropertyDecl - Return the encoded type for this
/// property declaration. If non-NULL, Container must be either an
/// ObjCCategoryImplDecl or ObjCImplementationDecl; it should only be
/// NULL when getting encodings for protocol properties.
/// Property attributes are stored as a comma-delimited C string. The simple
/// attributes readonly and bycopy are encoded as single characters. The
/// parametrized attributes, getter=name, setter=name, and ivar=name, are
/// encoded as single characters, followed by an identifier. Property types
/// are also encoded as a parametrized attribute. The characters used to encode
/// these attributes are defined by the following enumeration:
/// @code
/// enum PropertyAttributes {
/// kPropertyReadOnly = 'R',   // property is read-only.
/// kPropertyBycopy = 'C',     // property is a copy of the value last assigned
/// kPropertyByref = '&',  // property is a reference to the value last assigned
/// kPropertyDynamic = 'D',    // property is dynamic
/// kPropertyGetter = 'G',     // followed by getter selector name
/// kPropertySetter = 'S',     // followed by setter selector name
/// kPropertyInstanceVariable = 'V'  // followed by instance variable  name
/// kPropertyType = 'T'              // followed by old-style type encoding.
/// kPropertyWeak = 'W'              // 'weak' property
/// kPropertyStrong = 'P'            // property GC'able
/// kPropertyNonAtomic = 'N'         // property non-atomic
/// };
/// @endcode
void ASTContext::getObjCEncodingForPropertyDecl(const ObjCPropertyDecl *PD,
                                                const Decl *Container,
                                                std::string& S) const {
  // Collect information from the property implementation decl(s).
  bool Dynamic = false;
  ObjCPropertyImplDecl *SynthesizePID = 0;

  // FIXME: Duplicated code due to poor abstraction.
  if (Container) {
    if (const ObjCCategoryImplDecl *CID =
        dyn_cast<ObjCCategoryImplDecl>(Container)) {
      for (ObjCCategoryImplDecl::propimpl_iterator
             i = CID->propimpl_begin(), e = CID->propimpl_end();
           i != e; ++i) {
        ObjCPropertyImplDecl *PID = *i;
        if (PID->getPropertyDecl() == PD) {
          if (PID->getPropertyImplementation()==ObjCPropertyImplDecl::Dynamic) {
            Dynamic = true;
          } else {
            SynthesizePID = PID;
          }
        }
      }
    } else {
      const ObjCImplementationDecl *OID=cast<ObjCImplementationDecl>(Container);
      for (ObjCCategoryImplDecl::propimpl_iterator
             i = OID->propimpl_begin(), e = OID->propimpl_end();
           i != e; ++i) {
        ObjCPropertyImplDecl *PID = *i;
        if (PID->getPropertyDecl() == PD) {
          if (PID->getPropertyImplementation()==ObjCPropertyImplDecl::Dynamic) {
            Dynamic = true;
          } else {
            SynthesizePID = PID;
          }
        }
      }
    }
  }

  // FIXME: This is not very efficient.
  S = "T";

  // Encode result type.
  // GCC has some special rules regarding encoding of properties which
  // closely resembles encoding of ivars.
  getObjCEncodingForTypeImpl(PD->getType(), S, true, true, 0,
                             true /* outermost type */,
                             true /* encoding for property */);

  if (PD->isReadOnly()) {
    S += ",R";
  } else {
    switch (PD->getSetterKind()) {
    case ObjCPropertyDecl::Assign: break;
    case ObjCPropertyDecl::Copy:   S += ",C"; break;
    case ObjCPropertyDecl::Retain: S += ",&"; break;
    case ObjCPropertyDecl::Weak:   S += ",W"; break;
    }
  }

  // It really isn't clear at all what this means, since properties
  // are "dynamic by default".
  if (Dynamic)
    S += ",D";

  if (PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_nonatomic)
    S += ",N";

  if (PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_getter) {
    S += ",G";
    S += PD->getGetterName().getAsString();
  }

  if (PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_setter) {
    S += ",S";
    S += PD->getSetterName().getAsString();
  }

  if (SynthesizePID) {
    const ObjCIvarDecl *OID = SynthesizePID->getPropertyIvarDecl();
    S += ",V";
    S += OID->getNameAsString();
  }

  // FIXME: OBJCGC: weak & strong
}

/// getLegacyIntegralTypeEncoding -
/// Another legacy compatibility encoding: 32-bit longs are encoded as
/// 'l' or 'L' , but not always.  For typedefs, we need to use
/// 'i' or 'I' instead if encoding a struct field, or a pointer!
///
void ASTContext::getLegacyIntegralTypeEncoding (QualType &PointeeTy) const {
  if (isa<TypedefType>(PointeeTy.getTypePtr())) {
    if (const BuiltinType *BT = PointeeTy->getAs<BuiltinType>()) {
      if (BT->getKind() == BuiltinType::ULong && getIntWidth(PointeeTy) == 32)
        PointeeTy = UnsignedIntTy;
      else
        if (BT->getKind() == BuiltinType::Long && getIntWidth(PointeeTy) == 32)
          PointeeTy = IntTy;
    }
  }
}

void ASTContext::getObjCEncodingForType(QualType T, std::string& S,
                                        const FieldDecl *Field) const {
  // We follow the behavior of gcc, expanding structures which are
  // directly pointed to, and expanding embedded structures. Note that
  // these rules are sufficient to prevent recursive encoding of the
  // same type.
  getObjCEncodingForTypeImpl(T, S, true, true, Field,
                             true /* outermost type */);
}

static char ObjCEncodingForPrimitiveKind(const ASTContext *C, QualType T) {
    switch (T->getAs<BuiltinType>()->getKind()) {
    default: llvm_unreachable("Unhandled builtin type kind");
    case BuiltinType::Void:       return 'v';
    case BuiltinType::Bool:       return 'B';
    case BuiltinType::Char_U:
    case BuiltinType::UChar:      return 'C';
    case BuiltinType::UShort:     return 'S';
    case BuiltinType::UInt:       return 'I';
    case BuiltinType::ULong:
        return C->getIntWidth(T) == 32 ? 'L' : 'Q';
    case BuiltinType::UInt128:    return 'T';
    case BuiltinType::ULongLong:  return 'Q';
    case BuiltinType::Char_S:
    case BuiltinType::SChar:      return 'c';
    case BuiltinType::Short:      return 's';
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
    case BuiltinType::Int:        return 'i';
    case BuiltinType::Long:
      return C->getIntWidth(T) == 32 ? 'l' : 'q';
    case BuiltinType::LongLong:   return 'q';
    case BuiltinType::Int128:     return 't';
    case BuiltinType::Float:      return 'f';
    case BuiltinType::Double:     return 'd';
    case BuiltinType::LongDouble: return 'D';
    }
}

static char ObjCEncodingForEnumType(const ASTContext *C, const EnumType *ET) {
  EnumDecl *Enum = ET->getDecl();
  
  // The encoding of an non-fixed enum type is always 'i', regardless of size.
  if (!Enum->isFixed())
    return 'i';
  
  // The encoding of a fixed enum type matches its fixed underlying type.
  return ObjCEncodingForPrimitiveKind(C, Enum->getIntegerType());
}

static void EncodeBitField(const ASTContext *Ctx, std::string& S,
                           QualType T, const FieldDecl *FD) {
  assert(FD->isBitField() && "not a bitfield - getObjCEncodingForTypeImpl");
  S += 'b';
  // The NeXT runtime encodes bit fields as b followed by the number of bits.
  // The GNU runtime requires more information; bitfields are encoded as b,
  // then the offset (in bits) of the first element, then the type of the
  // bitfield, then the size in bits.  For example, in this structure:
  //
  // struct
  // {
  //    int integer;
  //    int flags:2;
  // };
  // On a 32-bit system, the encoding for flags would be b2 for the NeXT
  // runtime, but b32i2 for the GNU runtime.  The reason for this extra
  // information is not especially sensible, but we're stuck with it for
  // compatibility with GCC, although providing it breaks anything that
  // actually uses runtime introspection and wants to work on both runtimes...
  if (!Ctx->getLangOpts().NeXTRuntime) {
    const RecordDecl *RD = FD->getParent();
    const ASTRecordLayout &RL = Ctx->getASTRecordLayout(RD);
    S += llvm::utostr(RL.getFieldOffset(FD->getFieldIndex()));
    if (const EnumType *ET = T->getAs<EnumType>())
      S += ObjCEncodingForEnumType(Ctx, ET);
    else
      S += ObjCEncodingForPrimitiveKind(Ctx, T);
  }
  S += llvm::utostr(FD->getBitWidthValue(*Ctx));
}

// FIXME: Use SmallString for accumulating string.
void ASTContext::getObjCEncodingForTypeImpl(QualType T, std::string& S,
                                            bool ExpandPointedToStructures,
                                            bool ExpandStructures,
                                            const FieldDecl *FD,
                                            bool OutermostType,
                                            bool EncodingProperty,
                                            bool StructField,
                                            bool EncodeBlockParameters,
                                            bool EncodeClassNames) const {
  if (T->getAs<BuiltinType>()) {
    if (FD && FD->isBitField())
      return EncodeBitField(this, S, T, FD);
    S += ObjCEncodingForPrimitiveKind(this, T);
    return;
  }

  if (const ComplexType *CT = T->getAs<ComplexType>()) {
    S += 'j';
    getObjCEncodingForTypeImpl(CT->getElementType(), S, false, false, 0, false,
                               false);
    return;
  }
  
  // encoding for pointer or r3eference types.
  QualType PointeeTy;
  if (const PointerType *PT = T->getAs<PointerType>()) {
    if (PT->isObjCSelType()) {
      S += ':';
      return;
    }
    PointeeTy = PT->getPointeeType();
  }
  else if (const ReferenceType *RT = T->getAs<ReferenceType>())
    PointeeTy = RT->getPointeeType();
  if (!PointeeTy.isNull()) {
    bool isReadOnly = false;
    // For historical/compatibility reasons, the read-only qualifier of the
    // pointee gets emitted _before_ the '^'.  The read-only qualifier of
    // the pointer itself gets ignored, _unless_ we are looking at a typedef!
    // Also, do not emit the 'r' for anything but the outermost type!
    if (isa<TypedefType>(T.getTypePtr())) {
      if (OutermostType && T.isConstQualified()) {
        isReadOnly = true;
        S += 'r';
      }
    } else if (OutermostType) {
      QualType P = PointeeTy;
      while (P->getAs<PointerType>())
        P = P->getAs<PointerType>()->getPointeeType();
      if (P.isConstQualified()) {
        isReadOnly = true;
        S += 'r';
      }
    }
    if (isReadOnly) {
      // Another legacy compatibility encoding. Some ObjC qualifier and type
      // combinations need to be rearranged.
      // Rewrite "in const" from "nr" to "rn"
      if (StringRef(S).endswith("nr"))
        S.replace(S.end()-2, S.end(), "rn");
    }

    if (PointeeTy->isCharType()) {
      // char pointer types should be encoded as '*' unless it is a
      // type that has been typedef'd to 'BOOL'.
      if (!isTypeTypedefedAsBOOL(PointeeTy)) {
        S += '*';
        return;
      }
    } else if (const RecordType *RTy = PointeeTy->getAs<RecordType>()) {
      // GCC binary compat: Need to convert "struct objc_class *" to "#".
      if (RTy->getDecl()->getIdentifier() == &Idents.get("objc_class")) {
        S += '#';
        return;
      }
      // GCC binary compat: Need to convert "struct objc_object *" to "@".
      if (RTy->getDecl()->getIdentifier() == &Idents.get("objc_object")) {
        S += '@';
        return;
      }
      // fall through...
    }
    S += '^';
    getLegacyIntegralTypeEncoding(PointeeTy);

    getObjCEncodingForTypeImpl(PointeeTy, S, false, ExpandPointedToStructures,
                               NULL);
    return;
  }
  
  if (const ArrayType *AT =
      // Ignore type qualifiers etc.
        dyn_cast<ArrayType>(T->getCanonicalTypeInternal())) {
    if (isa<IncompleteArrayType>(AT) && !StructField) {
      // Incomplete arrays are encoded as a pointer to the array element.
      S += '^';

      getObjCEncodingForTypeImpl(AT->getElementType(), S,
                                 false, ExpandStructures, FD);
    } else {
      S += '[';

      if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT)) {
        if (getTypeSize(CAT->getElementType()) == 0)
          S += '0';
        else
          S += llvm::utostr(CAT->getSize().getZExtValue());
      } else {
        //Variable length arrays are encoded as a regular array with 0 elements.
        assert((isa<VariableArrayType>(AT) || isa<IncompleteArrayType>(AT)) &&
               "Unknown array type!");
        S += '0';
      }

      getObjCEncodingForTypeImpl(AT->getElementType(), S,
                                 false, ExpandStructures, FD);
      S += ']';
    }
    return;
  }

  if (T->getAs<FunctionType>()) {
    S += '?';
    return;
  }

  if (const RecordType *RTy = T->getAs<RecordType>()) {
    RecordDecl *RDecl = RTy->getDecl();
    S += RDecl->isUnion() ? '(' : '{';
    // Anonymous structures print as '?'
    if (const IdentifierInfo *II = RDecl->getIdentifier()) {
      S += II->getName();
      if (ClassTemplateSpecializationDecl *Spec
          = dyn_cast<ClassTemplateSpecializationDecl>(RDecl)) {
        const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
        std::string TemplateArgsStr
          = TemplateSpecializationType::PrintTemplateArgumentList(
                                            TemplateArgs.data(),
                                            TemplateArgs.size(),
                                            (*this).getPrintingPolicy());

        S += TemplateArgsStr;
      }
    } else {
      S += '?';
    }
    if (ExpandStructures) {
      S += '=';
      if (!RDecl->isUnion()) {
        getObjCEncodingForStructureImpl(RDecl, S, FD);
      } else {
        for (RecordDecl::field_iterator Field = RDecl->field_begin(),
                                     FieldEnd = RDecl->field_end();
             Field != FieldEnd; ++Field) {
          if (FD) {
            S += '"';
            S += Field->getNameAsString();
            S += '"';
          }

          // Special case bit-fields.
          if (Field->isBitField()) {
            getObjCEncodingForTypeImpl(Field->getType(), S, false, true,
                                       (*Field));
          } else {
            QualType qt = Field->getType();
            getLegacyIntegralTypeEncoding(qt);
            getObjCEncodingForTypeImpl(qt, S, false, true,
                                       FD, /*OutermostType*/false,
                                       /*EncodingProperty*/false,
                                       /*StructField*/true);
          }
        }
      }
    }
    S += RDecl->isUnion() ? ')' : '}';
    return;
  }
  
  if (const EnumType *ET = T->getAs<EnumType>()) {
    if (FD && FD->isBitField())
      EncodeBitField(this, S, T, FD);
    else
      S += ObjCEncodingForEnumType(this, ET);
    return;
  }

  if (const BlockPointerType *BT = T->getAs<BlockPointerType>()) {
    S += "@?"; // Unlike a pointer-to-function, which is "^?".
    if (EncodeBlockParameters) {
      const FunctionType *FT = BT->getPointeeType()->getAs<FunctionType>();
      
      S += '<';
      // Block return type
      getObjCEncodingForTypeImpl(FT->getResultType(), S, 
                                 ExpandPointedToStructures, ExpandStructures, 
                                 FD, 
                                 false /* OutermostType */, 
                                 EncodingProperty, 
                                 false /* StructField */, 
                                 EncodeBlockParameters, 
                                 EncodeClassNames);
      // Block self
      S += "@?";
      // Block parameters
      if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(FT)) {
        for (FunctionProtoType::arg_type_iterator I = FPT->arg_type_begin(),
               E = FPT->arg_type_end(); I && (I != E); ++I) {
          getObjCEncodingForTypeImpl(*I, S, 
                                     ExpandPointedToStructures, 
                                     ExpandStructures, 
                                     FD, 
                                     false /* OutermostType */, 
                                     EncodingProperty, 
                                     false /* StructField */, 
                                     EncodeBlockParameters, 
                                     EncodeClassNames);
        }
      }
      S += '>';
    }
    return;
  }

  // Ignore protocol qualifiers when mangling at this level.
  if (const ObjCObjectType *OT = T->getAs<ObjCObjectType>())
    T = OT->getBaseType();

  if (const ObjCInterfaceType *OIT = T->getAs<ObjCInterfaceType>()) {
    // @encode(class_name)
    ObjCInterfaceDecl *OI = OIT->getDecl();
    S += '{';
    const IdentifierInfo *II = OI->getIdentifier();
    S += II->getName();
    S += '=';
    SmallVector<const ObjCIvarDecl*, 32> Ivars;
    DeepCollectObjCIvars(OI, true, Ivars);
    for (unsigned i = 0, e = Ivars.size(); i != e; ++i) {
      const FieldDecl *Field = cast<FieldDecl>(Ivars[i]);
      if (Field->isBitField())
        getObjCEncodingForTypeImpl(Field->getType(), S, false, true, Field);
      else
        getObjCEncodingForTypeImpl(Field->getType(), S, false, true, FD);
    }
    S += '}';
    return;
  }

  if (const ObjCObjectPointerType *OPT = T->getAs<ObjCObjectPointerType>()) {
    if (OPT->isObjCIdType()) {
      S += '@';
      return;
    }

    if (OPT->isObjCClassType() || OPT->isObjCQualifiedClassType()) {
      // FIXME: Consider if we need to output qualifiers for 'Class<p>'.
      // Since this is a binary compatibility issue, need to consult with runtime
      // folks. Fortunately, this is a *very* obsure construct.
      S += '#';
      return;
    }

    if (OPT->isObjCQualifiedIdType()) {
      getObjCEncodingForTypeImpl(getObjCIdType(), S,
                                 ExpandPointedToStructures,
                                 ExpandStructures, FD);
      if (FD || EncodingProperty || EncodeClassNames) {
        // Note that we do extended encoding of protocol qualifer list
        // Only when doing ivar or property encoding.
        S += '"';
        for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
             E = OPT->qual_end(); I != E; ++I) {
          S += '<';
          S += (*I)->getNameAsString();
          S += '>';
        }
        S += '"';
      }
      return;
    }

    QualType PointeeTy = OPT->getPointeeType();
    if (!EncodingProperty &&
        isa<TypedefType>(PointeeTy.getTypePtr())) {
      // Another historical/compatibility reason.
      // We encode the underlying type which comes out as
      // {...};
      S += '^';
      getObjCEncodingForTypeImpl(PointeeTy, S,
                                 false, ExpandPointedToStructures,
                                 NULL);
      return;
    }

    S += '@';
    if (OPT->getInterfaceDecl() && 
        (FD || EncodingProperty || EncodeClassNames)) {
      S += '"';
      S += OPT->getInterfaceDecl()->getIdentifier()->getName();
      for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
           E = OPT->qual_end(); I != E; ++I) {
        S += '<';
        S += (*I)->getNameAsString();
        S += '>';
      }
      S += '"';
    }
    return;
  }

  // gcc just blithely ignores member pointers.
  // TODO: maybe there should be a mangling for these
  if (T->getAs<MemberPointerType>())
    return;
  
  if (T->isVectorType()) {
    // This matches gcc's encoding, even though technically it is
    // insufficient.
    // FIXME. We should do a better job than gcc.
    return;
  }
  
  llvm_unreachable("@encode for type not implemented!");
}

void ASTContext::getObjCEncodingForStructureImpl(RecordDecl *RDecl,
                                                 std::string &S,
                                                 const FieldDecl *FD,
                                                 bool includeVBases) const {
  assert(RDecl && "Expected non-null RecordDecl");
  assert(!RDecl->isUnion() && "Should not be called for unions");
  if (!RDecl->getDefinition())
    return;

  CXXRecordDecl *CXXRec = dyn_cast<CXXRecordDecl>(RDecl);
  std::multimap<uint64_t, NamedDecl *> FieldOrBaseOffsets;
  const ASTRecordLayout &layout = getASTRecordLayout(RDecl);

  if (CXXRec) {
    for (CXXRecordDecl::base_class_iterator
           BI = CXXRec->bases_begin(),
           BE = CXXRec->bases_end(); BI != BE; ++BI) {
      if (!BI->isVirtual()) {
        CXXRecordDecl *base = BI->getType()->getAsCXXRecordDecl();
        if (base->isEmpty())
          continue;
        uint64_t offs = layout.getBaseClassOffsetInBits(base);
        FieldOrBaseOffsets.insert(FieldOrBaseOffsets.upper_bound(offs),
                                  std::make_pair(offs, base));
      }
    }
  }
  
  unsigned i = 0;
  for (RecordDecl::field_iterator Field = RDecl->field_begin(),
                               FieldEnd = RDecl->field_end();
       Field != FieldEnd; ++Field, ++i) {
    uint64_t offs = layout.getFieldOffset(i);
    FieldOrBaseOffsets.insert(FieldOrBaseOffsets.upper_bound(offs),
                              std::make_pair(offs, *Field));
  }

  if (CXXRec && includeVBases) {
    for (CXXRecordDecl::base_class_iterator
           BI = CXXRec->vbases_begin(),
           BE = CXXRec->vbases_end(); BI != BE; ++BI) {
      CXXRecordDecl *base = BI->getType()->getAsCXXRecordDecl();
      if (base->isEmpty())
        continue;
      uint64_t offs = layout.getVBaseClassOffsetInBits(base);
      if (FieldOrBaseOffsets.find(offs) == FieldOrBaseOffsets.end())
        FieldOrBaseOffsets.insert(FieldOrBaseOffsets.end(),
                                  std::make_pair(offs, base));
    }
  }

  CharUnits size;
  if (CXXRec) {
    size = includeVBases ? layout.getSize() : layout.getNonVirtualSize();
  } else {
    size = layout.getSize();
  }

  uint64_t CurOffs = 0;
  std::multimap<uint64_t, NamedDecl *>::iterator
    CurLayObj = FieldOrBaseOffsets.begin();

  if ((CurLayObj != FieldOrBaseOffsets.end() && CurLayObj->first != 0) ||
      (CurLayObj == FieldOrBaseOffsets.end() &&
         CXXRec && CXXRec->isDynamicClass())) {
    assert(CXXRec && CXXRec->isDynamicClass() &&
           "Offset 0 was empty but no VTable ?");
    if (FD) {
      S += "\"_vptr$";
      std::string recname = CXXRec->getNameAsString();
      if (recname.empty()) recname = "?";
      S += recname;
      S += '"';
    }
    S += "^^?";
    CurOffs += getTypeSize(VoidPtrTy);
  }

  if (!RDecl->hasFlexibleArrayMember()) {
    // Mark the end of the structure.
    uint64_t offs = toBits(size);
    FieldOrBaseOffsets.insert(FieldOrBaseOffsets.upper_bound(offs),
                              std::make_pair(offs, (NamedDecl*)0));
  }

  for (; CurLayObj != FieldOrBaseOffsets.end(); ++CurLayObj) {
    assert(CurOffs <= CurLayObj->first);

    if (CurOffs < CurLayObj->first) {
      uint64_t padding = CurLayObj->first - CurOffs; 
      // FIXME: There doesn't seem to be a way to indicate in the encoding that
      // packing/alignment of members is different that normal, in which case
      // the encoding will be out-of-sync with the real layout.
      // If the runtime switches to just consider the size of types without
      // taking into account alignment, we could make padding explicit in the
      // encoding (e.g. using arrays of chars). The encoding strings would be
      // longer then though.
      CurOffs += padding;
    }

    NamedDecl *dcl = CurLayObj->second;
    if (dcl == 0)
      break; // reached end of structure.

    if (CXXRecordDecl *base = dyn_cast<CXXRecordDecl>(dcl)) {
      // We expand the bases without their virtual bases since those are going
      // in the initial structure. Note that this differs from gcc which
      // expands virtual bases each time one is encountered in the hierarchy,
      // making the encoding type bigger than it really is.
      getObjCEncodingForStructureImpl(base, S, FD, /*includeVBases*/false);
      assert(!base->isEmpty());
      CurOffs += toBits(getASTRecordLayout(base).getNonVirtualSize());
    } else {
      FieldDecl *field = cast<FieldDecl>(dcl);
      if (FD) {
        S += '"';
        S += field->getNameAsString();
        S += '"';
      }

      if (field->isBitField()) {
        EncodeBitField(this, S, field->getType(), field);
        CurOffs += field->getBitWidthValue(*this);
      } else {
        QualType qt = field->getType();
        getLegacyIntegralTypeEncoding(qt);
        getObjCEncodingForTypeImpl(qt, S, false, true, FD,
                                   /*OutermostType*/false,
                                   /*EncodingProperty*/false,
                                   /*StructField*/true);
        CurOffs += getTypeSize(field->getType());
      }
    }
  }
}

void ASTContext::getObjCEncodingForTypeQualifier(Decl::ObjCDeclQualifier QT,
                                                 std::string& S) const {
  if (QT & Decl::OBJC_TQ_In)
    S += 'n';
  if (QT & Decl::OBJC_TQ_Inout)
    S += 'N';
  if (QT & Decl::OBJC_TQ_Out)
    S += 'o';
  if (QT & Decl::OBJC_TQ_Bycopy)
    S += 'O';
  if (QT & Decl::OBJC_TQ_Byref)
    S += 'R';
  if (QT & Decl::OBJC_TQ_Oneway)
    S += 'V';
}

void ASTContext::setBuiltinVaListType(QualType T) {
  assert(BuiltinVaListType.isNull() && "__builtin_va_list type already set!");

  BuiltinVaListType = T;
}

TypedefDecl *ASTContext::getObjCIdDecl() const {
  if (!ObjCIdDecl) {
    QualType T = getObjCObjectType(ObjCBuiltinIdTy, 0, 0);
    T = getObjCObjectPointerType(T);
    TypeSourceInfo *IdInfo = getTrivialTypeSourceInfo(T);
    ObjCIdDecl = TypedefDecl::Create(const_cast<ASTContext &>(*this),
                                     getTranslationUnitDecl(),
                                     SourceLocation(), SourceLocation(),
                                     &Idents.get("id"), IdInfo);
  }
  
  return ObjCIdDecl;
}

TypedefDecl *ASTContext::getObjCSelDecl() const {
  if (!ObjCSelDecl) {
    QualType SelT = getPointerType(ObjCBuiltinSelTy);
    TypeSourceInfo *SelInfo = getTrivialTypeSourceInfo(SelT);
    ObjCSelDecl = TypedefDecl::Create(const_cast<ASTContext &>(*this),
                                      getTranslationUnitDecl(),
                                      SourceLocation(), SourceLocation(),
                                      &Idents.get("SEL"), SelInfo);
  }
  return ObjCSelDecl;
}

TypedefDecl *ASTContext::getObjCClassDecl() const {
  if (!ObjCClassDecl) {
    QualType T = getObjCObjectType(ObjCBuiltinClassTy, 0, 0);
    T = getObjCObjectPointerType(T);
    TypeSourceInfo *ClassInfo = getTrivialTypeSourceInfo(T);
    ObjCClassDecl = TypedefDecl::Create(const_cast<ASTContext &>(*this),
                                        getTranslationUnitDecl(),
                                        SourceLocation(), SourceLocation(),
                                        &Idents.get("Class"), ClassInfo);
  }
  
  return ObjCClassDecl;
}

ObjCInterfaceDecl *ASTContext::getObjCProtocolDecl() const {
  if (!ObjCProtocolClassDecl) {
    ObjCProtocolClassDecl 
      = ObjCInterfaceDecl::Create(*this, getTranslationUnitDecl(), 
                                  SourceLocation(),
                                  &Idents.get("Protocol"),
                                  /*PrevDecl=*/0,
                                  SourceLocation(), true);    
  }
  
  return ObjCProtocolClassDecl;
}

void ASTContext::setObjCConstantStringInterface(ObjCInterfaceDecl *Decl) {
  assert(ObjCConstantStringType.isNull() &&
         "'NSConstantString' type already set!");

  ObjCConstantStringType = getObjCInterfaceType(Decl);
}

/// \brief Retrieve the template name that corresponds to a non-empty
/// lookup.
TemplateName
ASTContext::getOverloadedTemplateName(UnresolvedSetIterator Begin,
                                      UnresolvedSetIterator End) const {
  unsigned size = End - Begin;
  assert(size > 1 && "set is not overloaded!");

  void *memory = Allocate(sizeof(OverloadedTemplateStorage) +
                          size * sizeof(FunctionTemplateDecl*));
  OverloadedTemplateStorage *OT = new(memory) OverloadedTemplateStorage(size);

  NamedDecl **Storage = OT->getStorage();
  for (UnresolvedSetIterator I = Begin; I != End; ++I) {
    NamedDecl *D = *I;
    assert(isa<FunctionTemplateDecl>(D) ||
           (isa<UsingShadowDecl>(D) &&
            isa<FunctionTemplateDecl>(D->getUnderlyingDecl())));
    *Storage++ = D;
  }

  return TemplateName(OT);
}

/// \brief Retrieve the template name that represents a qualified
/// template name such as \c std::vector.
TemplateName
ASTContext::getQualifiedTemplateName(NestedNameSpecifier *NNS,
                                     bool TemplateKeyword,
                                     TemplateDecl *Template) const {
  assert(NNS && "Missing nested-name-specifier in qualified template name");
  
  // FIXME: Canonicalization?
  llvm::FoldingSetNodeID ID;
  QualifiedTemplateName::Profile(ID, NNS, TemplateKeyword, Template);

  void *InsertPos = 0;
  QualifiedTemplateName *QTN =
    QualifiedTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
  if (!QTN) {
    QTN = new (*this,4) QualifiedTemplateName(NNS, TemplateKeyword, Template);
    QualifiedTemplateNames.InsertNode(QTN, InsertPos);
  }

  return TemplateName(QTN);
}

/// \brief Retrieve the template name that represents a dependent
/// template name such as \c MetaFun::template apply.
TemplateName
ASTContext::getDependentTemplateName(NestedNameSpecifier *NNS,
                                     const IdentifierInfo *Name) const {
  assert((!NNS || NNS->isDependent()) &&
         "Nested name specifier must be dependent");

  llvm::FoldingSetNodeID ID;
  DependentTemplateName::Profile(ID, NNS, Name);

  void *InsertPos = 0;
  DependentTemplateName *QTN =
    DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);

  if (QTN)
    return TemplateName(QTN);

  NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);
  if (CanonNNS == NNS) {
    QTN = new (*this,4) DependentTemplateName(NNS, Name);
  } else {
    TemplateName Canon = getDependentTemplateName(CanonNNS, Name);
    QTN = new (*this,4) DependentTemplateName(NNS, Name, Canon);
    DependentTemplateName *CheckQTN =
      DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckQTN && "Dependent type name canonicalization broken");
    (void)CheckQTN;
  }

  DependentTemplateNames.InsertNode(QTN, InsertPos);
  return TemplateName(QTN);
}

/// \brief Retrieve the template name that represents a dependent
/// template name such as \c MetaFun::template operator+.
TemplateName 
ASTContext::getDependentTemplateName(NestedNameSpecifier *NNS,
                                     OverloadedOperatorKind Operator) const {
  assert((!NNS || NNS->isDependent()) &&
         "Nested name specifier must be dependent");
  
  llvm::FoldingSetNodeID ID;
  DependentTemplateName::Profile(ID, NNS, Operator);
  
  void *InsertPos = 0;
  DependentTemplateName *QTN
    = DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
  
  if (QTN)
    return TemplateName(QTN);
  
  NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);
  if (CanonNNS == NNS) {
    QTN = new (*this,4) DependentTemplateName(NNS, Operator);
  } else {
    TemplateName Canon = getDependentTemplateName(CanonNNS, Operator);
    QTN = new (*this,4) DependentTemplateName(NNS, Operator, Canon);
    
    DependentTemplateName *CheckQTN
      = DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckQTN && "Dependent template name canonicalization broken");
    (void)CheckQTN;
  }
  
  DependentTemplateNames.InsertNode(QTN, InsertPos);
  return TemplateName(QTN);
}

TemplateName 
ASTContext::getSubstTemplateTemplateParm(TemplateTemplateParmDecl *param,
                                         TemplateName replacement) const {
  llvm::FoldingSetNodeID ID;
  SubstTemplateTemplateParmStorage::Profile(ID, param, replacement);
  
  void *insertPos = 0;
  SubstTemplateTemplateParmStorage *subst
    = SubstTemplateTemplateParms.FindNodeOrInsertPos(ID, insertPos);
  
  if (!subst) {
    subst = new (*this) SubstTemplateTemplateParmStorage(param, replacement);
    SubstTemplateTemplateParms.InsertNode(subst, insertPos);
  }

  return TemplateName(subst);
}

TemplateName 
ASTContext::getSubstTemplateTemplateParmPack(TemplateTemplateParmDecl *Param,
                                       const TemplateArgument &ArgPack) const {
  ASTContext &Self = const_cast<ASTContext &>(*this);
  llvm::FoldingSetNodeID ID;
  SubstTemplateTemplateParmPackStorage::Profile(ID, Self, Param, ArgPack);
  
  void *InsertPos = 0;
  SubstTemplateTemplateParmPackStorage *Subst
    = SubstTemplateTemplateParmPacks.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!Subst) {
    Subst = new (*this) SubstTemplateTemplateParmPackStorage(Param, 
                                                           ArgPack.pack_size(),
                                                         ArgPack.pack_begin());
    SubstTemplateTemplateParmPacks.InsertNode(Subst, InsertPos);
  }

  return TemplateName(Subst);
}

/// getFromTargetType - Given one of the integer types provided by
/// TargetInfo, produce the corresponding type. The unsigned @p Type
/// is actually a value of type @c TargetInfo::IntType.
CanQualType ASTContext::getFromTargetType(unsigned Type) const {
  switch (Type) {
  case TargetInfo::NoInt: return CanQualType();
  case TargetInfo::SignedShort: return ShortTy;
  case TargetInfo::UnsignedShort: return UnsignedShortTy;
  case TargetInfo::SignedInt: return IntTy;
  case TargetInfo::UnsignedInt: return UnsignedIntTy;
  case TargetInfo::SignedLong: return LongTy;
  case TargetInfo::UnsignedLong: return UnsignedLongTy;
  case TargetInfo::SignedLongLong: return LongLongTy;
  case TargetInfo::UnsignedLongLong: return UnsignedLongLongTy;
  }

  llvm_unreachable("Unhandled TargetInfo::IntType value");
}

//===----------------------------------------------------------------------===//
//                        Type Predicates.
//===----------------------------------------------------------------------===//

/// getObjCGCAttr - Returns one of GCNone, Weak or Strong objc's
/// garbage collection attribute.
///
Qualifiers::GC ASTContext::getObjCGCAttrKind(QualType Ty) const {
  if (getLangOpts().getGC() == LangOptions::NonGC)
    return Qualifiers::GCNone;

  assert(getLangOpts().ObjC1);
  Qualifiers::GC GCAttrs = Ty.getObjCGCAttr();

  // Default behaviour under objective-C's gc is for ObjC pointers
  // (or pointers to them) be treated as though they were declared
  // as __strong.
  if (GCAttrs == Qualifiers::GCNone) {
    if (Ty->isObjCObjectPointerType() || Ty->isBlockPointerType())
      return Qualifiers::Strong;
    else if (Ty->isPointerType())
      return getObjCGCAttrKind(Ty->getAs<PointerType>()->getPointeeType());
  } else {
    // It's not valid to set GC attributes on anything that isn't a
    // pointer.
#ifndef NDEBUG
    QualType CT = Ty->getCanonicalTypeInternal();
    while (const ArrayType *AT = dyn_cast<ArrayType>(CT))
      CT = AT->getElementType();
    assert(CT->isAnyPointerType() || CT->isBlockPointerType());
#endif
  }
  return GCAttrs;
}

//===----------------------------------------------------------------------===//
//                        Type Compatibility Testing
//===----------------------------------------------------------------------===//

/// areCompatVectorTypes - Return true if the two specified vector types are
/// compatible.
static bool areCompatVectorTypes(const VectorType *LHS,
                                 const VectorType *RHS) {
  assert(LHS->isCanonicalUnqualified() && RHS->isCanonicalUnqualified());
  return LHS->getElementType() == RHS->getElementType() &&
         LHS->getNumElements() == RHS->getNumElements();
}

bool ASTContext::areCompatibleVectorTypes(QualType FirstVec,
                                          QualType SecondVec) {
  assert(FirstVec->isVectorType() && "FirstVec should be a vector type");
  assert(SecondVec->isVectorType() && "SecondVec should be a vector type");

  if (hasSameUnqualifiedType(FirstVec, SecondVec))
    return true;

  // Treat Neon vector types and most AltiVec vector types as if they are the
  // equivalent GCC vector types.
  const VectorType *First = FirstVec->getAs<VectorType>();
  const VectorType *Second = SecondVec->getAs<VectorType>();
  if (First->getNumElements() == Second->getNumElements() &&
      hasSameType(First->getElementType(), Second->getElementType()) &&
      First->getVectorKind() != VectorType::AltiVecPixel &&
      First->getVectorKind() != VectorType::AltiVecBool &&
      Second->getVectorKind() != VectorType::AltiVecPixel &&
      Second->getVectorKind() != VectorType::AltiVecBool)
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// ObjCQualifiedIdTypesAreCompatible - Compatibility testing for qualified id's.
//===----------------------------------------------------------------------===//

/// ProtocolCompatibleWithProtocol - return 'true' if 'lProto' is in the
/// inheritance hierarchy of 'rProto'.
bool
ASTContext::ProtocolCompatibleWithProtocol(ObjCProtocolDecl *lProto,
                                           ObjCProtocolDecl *rProto) const {
  if (declaresSameEntity(lProto, rProto))
    return true;
  for (ObjCProtocolDecl::protocol_iterator PI = rProto->protocol_begin(),
       E = rProto->protocol_end(); PI != E; ++PI)
    if (ProtocolCompatibleWithProtocol(lProto, *PI))
      return true;
  return false;
}

/// QualifiedIdConformsQualifiedId - compare id<p,...> with id<p1,...>
/// return true if lhs's protocols conform to rhs's protocol; false
/// otherwise.
bool ASTContext::QualifiedIdConformsQualifiedId(QualType lhs, QualType rhs) {
  if (lhs->isObjCQualifiedIdType() && rhs->isObjCQualifiedIdType())
    return ObjCQualifiedIdTypesAreCompatible(lhs, rhs, false);
  return false;
}

/// ObjCQualifiedClassTypesAreCompatible - compare  Class<p,...> and
/// Class<p1, ...>.
bool ASTContext::ObjCQualifiedClassTypesAreCompatible(QualType lhs, 
                                                      QualType rhs) {
  const ObjCObjectPointerType *lhsQID = lhs->getAs<ObjCObjectPointerType>();
  const ObjCObjectPointerType *rhsOPT = rhs->getAs<ObjCObjectPointerType>();
  assert ((lhsQID && rhsOPT) && "ObjCQualifiedClassTypesAreCompatible");
  
  for (ObjCObjectPointerType::qual_iterator I = lhsQID->qual_begin(),
       E = lhsQID->qual_end(); I != E; ++I) {
    bool match = false;
    ObjCProtocolDecl *lhsProto = *I;
    for (ObjCObjectPointerType::qual_iterator J = rhsOPT->qual_begin(),
         E = rhsOPT->qual_end(); J != E; ++J) {
      ObjCProtocolDecl *rhsProto = *J;
      if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto)) {
        match = true;
        break;
      }
    }
    if (!match)
      return false;
  }
  return true;
}

/// ObjCQualifiedIdTypesAreCompatible - We know that one of lhs/rhs is an
/// ObjCQualifiedIDType.
bool ASTContext::ObjCQualifiedIdTypesAreCompatible(QualType lhs, QualType rhs,
                                                   bool compare) {
  // Allow id<P..> and an 'id' or void* type in all cases.
  if (lhs->isVoidPointerType() ||
      lhs->isObjCIdType() || lhs->isObjCClassType())
    return true;
  else if (rhs->isVoidPointerType() ||
           rhs->isObjCIdType() || rhs->isObjCClassType())
    return true;

  if (const ObjCObjectPointerType *lhsQID = lhs->getAsObjCQualifiedIdType()) {
    const ObjCObjectPointerType *rhsOPT = rhs->getAs<ObjCObjectPointerType>();

    if (!rhsOPT) return false;

    if (rhsOPT->qual_empty()) {
      // If the RHS is a unqualified interface pointer "NSString*",
      // make sure we check the class hierarchy.
      if (ObjCInterfaceDecl *rhsID = rhsOPT->getInterfaceDecl()) {
        for (ObjCObjectPointerType::qual_iterator I = lhsQID->qual_begin(),
             E = lhsQID->qual_end(); I != E; ++I) {
          // when comparing an id<P> on lhs with a static type on rhs,
          // see if static class implements all of id's protocols, directly or
          // through its super class and categories.
          if (!rhsID->ClassImplementsProtocol(*I, true))
            return false;
        }
      }
      // If there are no qualifiers and no interface, we have an 'id'.
      return true;
    }
    // Both the right and left sides have qualifiers.
    for (ObjCObjectPointerType::qual_iterator I = lhsQID->qual_begin(),
         E = lhsQID->qual_end(); I != E; ++I) {
      ObjCProtocolDecl *lhsProto = *I;
      bool match = false;

      // when comparing an id<P> on lhs with a static type on rhs,
      // see if static class implements all of id's protocols, directly or
      // through its super class and categories.
      for (ObjCObjectPointerType::qual_iterator J = rhsOPT->qual_begin(),
           E = rhsOPT->qual_end(); J != E; ++J) {
        ObjCProtocolDecl *rhsProto = *J;
        if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto) ||
            (compare && ProtocolCompatibleWithProtocol(rhsProto, lhsProto))) {
          match = true;
          break;
        }
      }
      // If the RHS is a qualified interface pointer "NSString<P>*",
      // make sure we check the class hierarchy.
      if (ObjCInterfaceDecl *rhsID = rhsOPT->getInterfaceDecl()) {
        for (ObjCObjectPointerType::qual_iterator I = lhsQID->qual_begin(),
             E = lhsQID->qual_end(); I != E; ++I) {
          // when comparing an id<P> on lhs with a static type on rhs,
          // see if static class implements all of id's protocols, directly or
          // through its super class and categories.
          if (rhsID->ClassImplementsProtocol(*I, true)) {
            match = true;
            break;
          }
        }
      }
      if (!match)
        return false;
    }

    return true;
  }

  const ObjCObjectPointerType *rhsQID = rhs->getAsObjCQualifiedIdType();
  assert(rhsQID && "One of the LHS/RHS should be id<x>");

  if (const ObjCObjectPointerType *lhsOPT =
        lhs->getAsObjCInterfacePointerType()) {
    // If both the right and left sides have qualifiers.
    for (ObjCObjectPointerType::qual_iterator I = lhsOPT->qual_begin(),
         E = lhsOPT->qual_end(); I != E; ++I) {
      ObjCProtocolDecl *lhsProto = *I;
      bool match = false;

      // when comparing an id<P> on rhs with a static type on lhs,
      // see if static class implements all of id's protocols, directly or
      // through its super class and categories.
      // First, lhs protocols in the qualifier list must be found, direct
      // or indirect in rhs's qualifier list or it is a mismatch.
      for (ObjCObjectPointerType::qual_iterator J = rhsQID->qual_begin(),
           E = rhsQID->qual_end(); J != E; ++J) {
        ObjCProtocolDecl *rhsProto = *J;
        if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto) ||
            (compare && ProtocolCompatibleWithProtocol(rhsProto, lhsProto))) {
          match = true;
          break;
        }
      }
      if (!match)
        return false;
    }
    
    // Static class's protocols, or its super class or category protocols
    // must be found, direct or indirect in rhs's qualifier list or it is a mismatch.
    if (ObjCInterfaceDecl *lhsID = lhsOPT->getInterfaceDecl()) {
      llvm::SmallPtrSet<ObjCProtocolDecl *, 8> LHSInheritedProtocols;
      CollectInheritedProtocols(lhsID, LHSInheritedProtocols);
      // This is rather dubious but matches gcc's behavior. If lhs has
      // no type qualifier and its class has no static protocol(s)
      // assume that it is mismatch.
      if (LHSInheritedProtocols.empty() && lhsOPT->qual_empty())
        return false;
      for (llvm::SmallPtrSet<ObjCProtocolDecl*,8>::iterator I =
           LHSInheritedProtocols.begin(),
           E = LHSInheritedProtocols.end(); I != E; ++I) {
        bool match = false;
        ObjCProtocolDecl *lhsProto = (*I);
        for (ObjCObjectPointerType::qual_iterator J = rhsQID->qual_begin(),
             E = rhsQID->qual_end(); J != E; ++J) {
          ObjCProtocolDecl *rhsProto = *J;
          if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto) ||
              (compare && ProtocolCompatibleWithProtocol(rhsProto, lhsProto))) {
            match = true;
            break;
          }
        }
        if (!match)
          return false;
      }
    }
    return true;
  }
  return false;
}

/// canAssignObjCInterfaces - Return true if the two interface types are
/// compatible for assignment from RHS to LHS.  This handles validation of any
/// protocol qualifiers on the LHS or RHS.
///
bool ASTContext::canAssignObjCInterfaces(const ObjCObjectPointerType *LHSOPT,
                                         const ObjCObjectPointerType *RHSOPT) {
  const ObjCObjectType* LHS = LHSOPT->getObjectType();
  const ObjCObjectType* RHS = RHSOPT->getObjectType();

  // If either type represents the built-in 'id' or 'Class' types, return true.
  if (LHS->isObjCUnqualifiedIdOrClass() ||
      RHS->isObjCUnqualifiedIdOrClass())
    return true;

  if (LHS->isObjCQualifiedId() || RHS->isObjCQualifiedId())
    return ObjCQualifiedIdTypesAreCompatible(QualType(LHSOPT,0),
                                             QualType(RHSOPT,0),
                                             false);
  
  if (LHS->isObjCQualifiedClass() && RHS->isObjCQualifiedClass())
    return ObjCQualifiedClassTypesAreCompatible(QualType(LHSOPT,0),
                                                QualType(RHSOPT,0));
  
  // If we have 2 user-defined types, fall into that path.
  if (LHS->getInterface() && RHS->getInterface())
    return canAssignObjCInterfaces(LHS, RHS);

  return false;
}

/// canAssignObjCInterfacesInBlockPointer - This routine is specifically written
/// for providing type-safety for objective-c pointers used to pass/return 
/// arguments in block literals. When passed as arguments, passing 'A*' where
/// 'id' is expected is not OK. Passing 'Sub *" where 'Super *" is expected is
/// not OK. For the return type, the opposite is not OK.
bool ASTContext::canAssignObjCInterfacesInBlockPointer(
                                         const ObjCObjectPointerType *LHSOPT,
                                         const ObjCObjectPointerType *RHSOPT,
                                         bool BlockReturnType) {
  if (RHSOPT->isObjCBuiltinType() || LHSOPT->isObjCIdType())
    return true;
  
  if (LHSOPT->isObjCBuiltinType()) {
    return RHSOPT->isObjCBuiltinType() || RHSOPT->isObjCQualifiedIdType();
  }
  
  if (LHSOPT->isObjCQualifiedIdType() || RHSOPT->isObjCQualifiedIdType())
    return ObjCQualifiedIdTypesAreCompatible(QualType(LHSOPT,0),
                                             QualType(RHSOPT,0),
                                             false);
  
  const ObjCInterfaceType* LHS = LHSOPT->getInterfaceType();
  const ObjCInterfaceType* RHS = RHSOPT->getInterfaceType();
  if (LHS && RHS)  { // We have 2 user-defined types.
    if (LHS != RHS) {
      if (LHS->getDecl()->isSuperClassOf(RHS->getDecl()))
        return BlockReturnType;
      if (RHS->getDecl()->isSuperClassOf(LHS->getDecl()))
        return !BlockReturnType;
    }
    else
      return true;
  }
  return false;
}

/// getIntersectionOfProtocols - This routine finds the intersection of set
/// of protocols inherited from two distinct objective-c pointer objects.
/// It is used to build composite qualifier list of the composite type of
/// the conditional expression involving two objective-c pointer objects.
static 
void getIntersectionOfProtocols(ASTContext &Context,
                                const ObjCObjectPointerType *LHSOPT,
                                const ObjCObjectPointerType *RHSOPT,
      SmallVectorImpl<ObjCProtocolDecl *> &IntersectionOfProtocols) {
  
  const ObjCObjectType* LHS = LHSOPT->getObjectType();
  const ObjCObjectType* RHS = RHSOPT->getObjectType();
  assert(LHS->getInterface() && "LHS must have an interface base");
  assert(RHS->getInterface() && "RHS must have an interface base");
  
  llvm::SmallPtrSet<ObjCProtocolDecl *, 8> InheritedProtocolSet;
  unsigned LHSNumProtocols = LHS->getNumProtocols();
  if (LHSNumProtocols > 0)
    InheritedProtocolSet.insert(LHS->qual_begin(), LHS->qual_end());
  else {
    llvm::SmallPtrSet<ObjCProtocolDecl *, 8> LHSInheritedProtocols;
    Context.CollectInheritedProtocols(LHS->getInterface(),
                                      LHSInheritedProtocols);
    InheritedProtocolSet.insert(LHSInheritedProtocols.begin(), 
                                LHSInheritedProtocols.end());
  }
  
  unsigned RHSNumProtocols = RHS->getNumProtocols();
  if (RHSNumProtocols > 0) {
    ObjCProtocolDecl **RHSProtocols =
      const_cast<ObjCProtocolDecl **>(RHS->qual_begin());
    for (unsigned i = 0; i < RHSNumProtocols; ++i)
      if (InheritedProtocolSet.count(RHSProtocols[i]))
        IntersectionOfProtocols.push_back(RHSProtocols[i]);
  } else {
    llvm::SmallPtrSet<ObjCProtocolDecl *, 8> RHSInheritedProtocols;
    Context.CollectInheritedProtocols(RHS->getInterface(),
                                      RHSInheritedProtocols);
    for (llvm::SmallPtrSet<ObjCProtocolDecl*,8>::iterator I = 
         RHSInheritedProtocols.begin(),
         E = RHSInheritedProtocols.end(); I != E; ++I) 
      if (InheritedProtocolSet.count((*I)))
        IntersectionOfProtocols.push_back((*I));
  }
}

/// areCommonBaseCompatible - Returns common base class of the two classes if
/// one found. Note that this is O'2 algorithm. But it will be called as the
/// last type comparison in a ?-exp of ObjC pointer types before a 
/// warning is issued. So, its invokation is extremely rare.
QualType ASTContext::areCommonBaseCompatible(
                                          const ObjCObjectPointerType *Lptr,
                                          const ObjCObjectPointerType *Rptr) {
  const ObjCObjectType *LHS = Lptr->getObjectType();
  const ObjCObjectType *RHS = Rptr->getObjectType();
  const ObjCInterfaceDecl* LDecl = LHS->getInterface();
  const ObjCInterfaceDecl* RDecl = RHS->getInterface();
  if (!LDecl || !RDecl || (declaresSameEntity(LDecl, RDecl)))
    return QualType();
  
  do {
    LHS = cast<ObjCInterfaceType>(getObjCInterfaceType(LDecl));
    if (canAssignObjCInterfaces(LHS, RHS)) {
      SmallVector<ObjCProtocolDecl *, 8> Protocols;
      getIntersectionOfProtocols(*this, Lptr, Rptr, Protocols);

      QualType Result = QualType(LHS, 0);
      if (!Protocols.empty())
        Result = getObjCObjectType(Result, Protocols.data(), Protocols.size());
      Result = getObjCObjectPointerType(Result);
      return Result;
    }
  } while ((LDecl = LDecl->getSuperClass()));
    
  return QualType();
}

bool ASTContext::canAssignObjCInterfaces(const ObjCObjectType *LHS,
                                         const ObjCObjectType *RHS) {
  assert(LHS->getInterface() && "LHS is not an interface type");
  assert(RHS->getInterface() && "RHS is not an interface type");

  // Verify that the base decls are compatible: the RHS must be a subclass of
  // the LHS.
  if (!LHS->getInterface()->isSuperClassOf(RHS->getInterface()))
    return false;

  // RHS must have a superset of the protocols in the LHS.  If the LHS is not
  // protocol qualified at all, then we are good.
  if (LHS->getNumProtocols() == 0)
    return true;

  // Okay, we know the LHS has protocol qualifiers.  If the RHS doesn't, 
  // more detailed analysis is required.
  if (RHS->getNumProtocols() == 0) {
    // OK, if LHS is a superclass of RHS *and*
    // this superclass is assignment compatible with LHS.
    // false otherwise.
    bool IsSuperClass = 
      LHS->getInterface()->isSuperClassOf(RHS->getInterface());
    if (IsSuperClass) {
      // OK if conversion of LHS to SuperClass results in narrowing of types
      // ; i.e., SuperClass may implement at least one of the protocols
      // in LHS's protocol list. Example, SuperObj<P1> = lhs<P1,P2> is ok.
      // But not SuperObj<P1,P2,P3> = lhs<P1,P2>.
      llvm::SmallPtrSet<ObjCProtocolDecl *, 8> SuperClassInheritedProtocols;
      CollectInheritedProtocols(RHS->getInterface(), SuperClassInheritedProtocols);
      // If super class has no protocols, it is not a match.
      if (SuperClassInheritedProtocols.empty())
        return false;
      
      for (ObjCObjectType::qual_iterator LHSPI = LHS->qual_begin(),
           LHSPE = LHS->qual_end();
           LHSPI != LHSPE; LHSPI++) {
        bool SuperImplementsProtocol = false;
        ObjCProtocolDecl *LHSProto = (*LHSPI);
        
        for (llvm::SmallPtrSet<ObjCProtocolDecl*,8>::iterator I =
             SuperClassInheritedProtocols.begin(),
             E = SuperClassInheritedProtocols.end(); I != E; ++I) {
          ObjCProtocolDecl *SuperClassProto = (*I);
          if (SuperClassProto->lookupProtocolNamed(LHSProto->getIdentifier())) {
            SuperImplementsProtocol = true;
            break;
          }
        }
        if (!SuperImplementsProtocol)
          return false;
      }
      return true;
    }
    return false;
  }

  for (ObjCObjectType::qual_iterator LHSPI = LHS->qual_begin(),
                                     LHSPE = LHS->qual_end();
       LHSPI != LHSPE; LHSPI++) {
    bool RHSImplementsProtocol = false;

    // If the RHS doesn't implement the protocol on the left, the types
    // are incompatible.
    for (ObjCObjectType::qual_iterator RHSPI = RHS->qual_begin(),
                                       RHSPE = RHS->qual_end();
         RHSPI != RHSPE; RHSPI++) {
      if ((*RHSPI)->lookupProtocolNamed((*LHSPI)->getIdentifier())) {
        RHSImplementsProtocol = true;
        break;
      }
    }
    // FIXME: For better diagnostics, consider passing back the protocol name.
    if (!RHSImplementsProtocol)
      return false;
  }
  // The RHS implements all protocols listed on the LHS.
  return true;
}

bool ASTContext::areComparableObjCPointerTypes(QualType LHS, QualType RHS) {
  // get the "pointed to" types
  const ObjCObjectPointerType *LHSOPT = LHS->getAs<ObjCObjectPointerType>();
  const ObjCObjectPointerType *RHSOPT = RHS->getAs<ObjCObjectPointerType>();

  if (!LHSOPT || !RHSOPT)
    return false;

  return canAssignObjCInterfaces(LHSOPT, RHSOPT) ||
         canAssignObjCInterfaces(RHSOPT, LHSOPT);
}

bool ASTContext::canBindObjCObjectType(QualType To, QualType From) {
  return canAssignObjCInterfaces(
                getObjCObjectPointerType(To)->getAs<ObjCObjectPointerType>(),
                getObjCObjectPointerType(From)->getAs<ObjCObjectPointerType>());
}

/// typesAreCompatible - C99 6.7.3p9: For two qualified types to be compatible,
/// both shall have the identically qualified version of a compatible type.
/// C99 6.2.7p1: Two types have compatible types if their types are the
/// same. See 6.7.[2,3,5] for additional rules.
bool ASTContext::typesAreCompatible(QualType LHS, QualType RHS,
                                    bool CompareUnqualified) {
  if (getLangOpts().CPlusPlus)
    return hasSameType(LHS, RHS);
  
  return !mergeTypes(LHS, RHS, false, CompareUnqualified).isNull();
}

bool ASTContext::propertyTypesAreCompatible(QualType LHS, QualType RHS) {
  return typesAreCompatible(LHS, RHS);
}

bool ASTContext::typesAreBlockPointerCompatible(QualType LHS, QualType RHS) {
  return !mergeTypes(LHS, RHS, true).isNull();
}

/// mergeTransparentUnionType - if T is a transparent union type and a member
/// of T is compatible with SubType, return the merged type, else return
/// QualType()
QualType ASTContext::mergeTransparentUnionType(QualType T, QualType SubType,
                                               bool OfBlockPointer,
                                               bool Unqualified) {
  if (const RecordType *UT = T->getAsUnionType()) {
    RecordDecl *UD = UT->getDecl();
    if (UD->hasAttr<TransparentUnionAttr>()) {
      for (RecordDecl::field_iterator it = UD->field_begin(),
           itend = UD->field_end(); it != itend; ++it) {
        QualType ET = it->getType().getUnqualifiedType();
        QualType MT = mergeTypes(ET, SubType, OfBlockPointer, Unqualified);
        if (!MT.isNull())
          return MT;
      }
    }
  }

  return QualType();
}

/// mergeFunctionArgumentTypes - merge two types which appear as function
/// argument types
QualType ASTContext::mergeFunctionArgumentTypes(QualType lhs, QualType rhs, 
                                                bool OfBlockPointer,
                                                bool Unqualified) {
  // GNU extension: two types are compatible if they appear as a function
  // argument, one of the types is a transparent union type and the other
  // type is compatible with a union member
  QualType lmerge = mergeTransparentUnionType(lhs, rhs, OfBlockPointer,
                                              Unqualified);
  if (!lmerge.isNull())
    return lmerge;

  QualType rmerge = mergeTransparentUnionType(rhs, lhs, OfBlockPointer,
                                              Unqualified);
  if (!rmerge.isNull())
    return rmerge;

  return mergeTypes(lhs, rhs, OfBlockPointer, Unqualified);
}

QualType ASTContext::mergeFunctionTypes(QualType lhs, QualType rhs, 
                                        bool OfBlockPointer,
                                        bool Unqualified) {
  const FunctionType *lbase = lhs->getAs<FunctionType>();
  const FunctionType *rbase = rhs->getAs<FunctionType>();
  const FunctionProtoType *lproto = dyn_cast<FunctionProtoType>(lbase);
  const FunctionProtoType *rproto = dyn_cast<FunctionProtoType>(rbase);
  bool allLTypes = true;
  bool allRTypes = true;

  // Check return type
  QualType retType;
  if (OfBlockPointer) {
    QualType RHS = rbase->getResultType();
    QualType LHS = lbase->getResultType();
    bool UnqualifiedResult = Unqualified;
    if (!UnqualifiedResult)
      UnqualifiedResult = (!RHS.hasQualifiers() && LHS.hasQualifiers());
    retType = mergeTypes(LHS, RHS, true, UnqualifiedResult, true);
  }
  else
    retType = mergeTypes(lbase->getResultType(), rbase->getResultType(), false,
                         Unqualified);
  if (retType.isNull()) return QualType();
  
  if (Unqualified)
    retType = retType.getUnqualifiedType();

  CanQualType LRetType = getCanonicalType(lbase->getResultType());
  CanQualType RRetType = getCanonicalType(rbase->getResultType());
  if (Unqualified) {
    LRetType = LRetType.getUnqualifiedType();
    RRetType = RRetType.getUnqualifiedType();
  }
  
  if (getCanonicalType(retType) != LRetType)
    allLTypes = false;
  if (getCanonicalType(retType) != RRetType)
    allRTypes = false;

  // FIXME: double check this
  // FIXME: should we error if lbase->getRegParmAttr() != 0 &&
  //                           rbase->getRegParmAttr() != 0 &&
  //                           lbase->getRegParmAttr() != rbase->getRegParmAttr()?
  FunctionType::ExtInfo lbaseInfo = lbase->getExtInfo();
  FunctionType::ExtInfo rbaseInfo = rbase->getExtInfo();

  // Compatible functions must have compatible calling conventions
  if (!isSameCallConv(lbaseInfo.getCC(), rbaseInfo.getCC()))
    return QualType();

  // Regparm is part of the calling convention.
  if (lbaseInfo.getHasRegParm() != rbaseInfo.getHasRegParm())
    return QualType();
  if (lbaseInfo.getRegParm() != rbaseInfo.getRegParm())
    return QualType();

  if (lbaseInfo.getProducesResult() != rbaseInfo.getProducesResult())
    return QualType();

  // functypes which return are preferred over those that do not.
  if (lbaseInfo.getNoReturn() && !rbaseInfo.getNoReturn())
    allLTypes = false;
  else if (!lbaseInfo.getNoReturn() && rbaseInfo.getNoReturn())
    allRTypes = false;
  // FIXME: some uses, e.g. conditional exprs, really want this to be 'both'.
  bool NoReturn = lbaseInfo.getNoReturn() || rbaseInfo.getNoReturn();

  FunctionType::ExtInfo einfo = lbaseInfo.withNoReturn(NoReturn);

  if (lproto && rproto) { // two C99 style function prototypes
    assert(!lproto->hasExceptionSpec() && !rproto->hasExceptionSpec() &&
           "C++ shouldn't be here");
    unsigned lproto_nargs = lproto->getNumArgs();
    unsigned rproto_nargs = rproto->getNumArgs();

    // Compatible functions must have the same number of arguments
    if (lproto_nargs != rproto_nargs)
      return QualType();

    // Variadic and non-variadic functions aren't compatible
    if (lproto->isVariadic() != rproto->isVariadic())
      return QualType();

    if (lproto->getTypeQuals() != rproto->getTypeQuals())
      return QualType();

    if (LangOpts.ObjCAutoRefCount &&
        !FunctionTypesMatchOnNSConsumedAttrs(rproto, lproto))
      return QualType();
      
    // Check argument compatibility
    SmallVector<QualType, 10> types;
    for (unsigned i = 0; i < lproto_nargs; i++) {
      QualType largtype = lproto->getArgType(i).getUnqualifiedType();
      QualType rargtype = rproto->getArgType(i).getUnqualifiedType();
      QualType argtype = mergeFunctionArgumentTypes(largtype, rargtype,
                                                    OfBlockPointer,
                                                    Unqualified);
      if (argtype.isNull()) return QualType();
      
      if (Unqualified)
        argtype = argtype.getUnqualifiedType();
      
      types.push_back(argtype);
      if (Unqualified) {
        largtype = largtype.getUnqualifiedType();
        rargtype = rargtype.getUnqualifiedType();
      }
      
      if (getCanonicalType(argtype) != getCanonicalType(largtype))
        allLTypes = false;
      if (getCanonicalType(argtype) != getCanonicalType(rargtype))
        allRTypes = false;
    }
      
    if (allLTypes) return lhs;
    if (allRTypes) return rhs;

    FunctionProtoType::ExtProtoInfo EPI = lproto->getExtProtoInfo();
    EPI.ExtInfo = einfo;
    return getFunctionType(retType, types.begin(), types.size(), EPI);
  }

  if (lproto) allRTypes = false;
  if (rproto) allLTypes = false;

  const FunctionProtoType *proto = lproto ? lproto : rproto;
  if (proto) {
    assert(!proto->hasExceptionSpec() && "C++ shouldn't be here");
    if (proto->isVariadic()) return QualType();
    // Check that the types are compatible with the types that
    // would result from default argument promotions (C99 6.7.5.3p15).
    // The only types actually affected are promotable integer
    // types and floats, which would be passed as a different
    // type depending on whether the prototype is visible.
    unsigned proto_nargs = proto->getNumArgs();
    for (unsigned i = 0; i < proto_nargs; ++i) {
      QualType argTy = proto->getArgType(i);
      
      // Look at the promotion type of enum types, since that is the type used
      // to pass enum values.
      if (const EnumType *Enum = argTy->getAs<EnumType>())
        argTy = Enum->getDecl()->getPromotionType();
      
      if (argTy->isPromotableIntegerType() ||
          getCanonicalType(argTy).getUnqualifiedType() == FloatTy)
        return QualType();
    }

    if (allLTypes) return lhs;
    if (allRTypes) return rhs;

    FunctionProtoType::ExtProtoInfo EPI = proto->getExtProtoInfo();
    EPI.ExtInfo = einfo;
    return getFunctionType(retType, proto->arg_type_begin(),
                           proto->getNumArgs(), EPI);
  }

  if (allLTypes) return lhs;
  if (allRTypes) return rhs;
  return getFunctionNoProtoType(retType, einfo);
}

QualType ASTContext::mergeTypes(QualType LHS, QualType RHS, 
                                bool OfBlockPointer,
                                bool Unqualified, bool BlockReturnType) {
  // C++ [expr]: If an expression initially has the type "reference to T", the
  // type is adjusted to "T" prior to any further analysis, the expression
  // designates the object or function denoted by the reference, and the
  // expression is an lvalue unless the reference is an rvalue reference and
  // the expression is a function call (possibly inside parentheses).
  assert(!LHS->getAs<ReferenceType>() && "LHS is a reference type?");
  assert(!RHS->getAs<ReferenceType>() && "RHS is a reference type?");

  if (Unqualified) {
    LHS = LHS.getUnqualifiedType();
    RHS = RHS.getUnqualifiedType();
  }
  
  QualType LHSCan = getCanonicalType(LHS),
           RHSCan = getCanonicalType(RHS);

  // If two types are identical, they are compatible.
  if (LHSCan == RHSCan)
    return LHS;

  // If the qualifiers are different, the types aren't compatible... mostly.
  Qualifiers LQuals = LHSCan.getLocalQualifiers();
  Qualifiers RQuals = RHSCan.getLocalQualifiers();
  if (LQuals != RQuals) {
    // If any of these qualifiers are different, we have a type
    // mismatch.
    if (LQuals.getCVRQualifiers() != RQuals.getCVRQualifiers() ||
        LQuals.getAddressSpace() != RQuals.getAddressSpace() ||
        LQuals.getObjCLifetime() != RQuals.getObjCLifetime())
      return QualType();

    // Exactly one GC qualifier difference is allowed: __strong is
    // okay if the other type has no GC qualifier but is an Objective
    // C object pointer (i.e. implicitly strong by default).  We fix
    // this by pretending that the unqualified type was actually
    // qualified __strong.
    Qualifiers::GC GC_L = LQuals.getObjCGCAttr();
    Qualifiers::GC GC_R = RQuals.getObjCGCAttr();
    assert((GC_L != GC_R) && "unequal qualifier sets had only equal elements");

    if (GC_L == Qualifiers::Weak || GC_R == Qualifiers::Weak)
      return QualType();

    if (GC_L == Qualifiers::Strong && RHSCan->isObjCObjectPointerType()) {
      return mergeTypes(LHS, getObjCGCQualType(RHS, Qualifiers::Strong));
    }
    if (GC_R == Qualifiers::Strong && LHSCan->isObjCObjectPointerType()) {
      return mergeTypes(getObjCGCQualType(LHS, Qualifiers::Strong), RHS);
    }
    return QualType();
  }

  // Okay, qualifiers are equal.

  Type::TypeClass LHSClass = LHSCan->getTypeClass();
  Type::TypeClass RHSClass = RHSCan->getTypeClass();

  // We want to consider the two function types to be the same for these
  // comparisons, just force one to the other.
  if (LHSClass == Type::FunctionProto) LHSClass = Type::FunctionNoProto;
  if (RHSClass == Type::FunctionProto) RHSClass = Type::FunctionNoProto;

  // Same as above for arrays
  if (LHSClass == Type::VariableArray || LHSClass == Type::IncompleteArray)
    LHSClass = Type::ConstantArray;
  if (RHSClass == Type::VariableArray || RHSClass == Type::IncompleteArray)
    RHSClass = Type::ConstantArray;

  // ObjCInterfaces are just specialized ObjCObjects.
  if (LHSClass == Type::ObjCInterface) LHSClass = Type::ObjCObject;
  if (RHSClass == Type::ObjCInterface) RHSClass = Type::ObjCObject;

  // Canonicalize ExtVector -> Vector.
  if (LHSClass == Type::ExtVector) LHSClass = Type::Vector;
  if (RHSClass == Type::ExtVector) RHSClass = Type::Vector;

  // If the canonical type classes don't match.
  if (LHSClass != RHSClass) {
    // C99 6.7.2.2p4: Each enumerated type shall be compatible with char,
    // a signed integer type, or an unsigned integer type.
    // Compatibility is based on the underlying type, not the promotion
    // type.
    if (const EnumType* ETy = LHS->getAs<EnumType>()) {
      QualType TINT = ETy->getDecl()->getIntegerType();
      if (!TINT.isNull() && hasSameType(TINT, RHSCan.getUnqualifiedType()))
        return RHS;
    }
    if (const EnumType* ETy = RHS->getAs<EnumType>()) {
      QualType TINT = ETy->getDecl()->getIntegerType();
      if (!TINT.isNull() && hasSameType(TINT, LHSCan.getUnqualifiedType()))
        return LHS;
    }
    // allow block pointer type to match an 'id' type.
    if (OfBlockPointer && !BlockReturnType) {
       if (LHS->isObjCIdType() && RHS->isBlockPointerType())
         return LHS;
      if (RHS->isObjCIdType() && LHS->isBlockPointerType())
        return RHS;
    }
    
    return QualType();
  }

  // The canonical type classes match.
  switch (LHSClass) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  case Type::LValueReference:
  case Type::RValueReference:
  case Type::MemberPointer:
    llvm_unreachable("C++ should never be in mergeTypes");

  case Type::ObjCInterface:
  case Type::IncompleteArray:
  case Type::VariableArray:
  case Type::FunctionProto:
  case Type::ExtVector:
    llvm_unreachable("Types are eliminated above");

  case Type::Pointer:
  {
    // Merge two pointer types, while trying to preserve typedef info
    QualType LHSPointee = LHS->getAs<PointerType>()->getPointeeType();
    QualType RHSPointee = RHS->getAs<PointerType>()->getPointeeType();
    if (Unqualified) {
      LHSPointee = LHSPointee.getUnqualifiedType();
      RHSPointee = RHSPointee.getUnqualifiedType();
    }
    QualType ResultType = mergeTypes(LHSPointee, RHSPointee, false, 
                                     Unqualified);
    if (ResultType.isNull()) return QualType();
    if (getCanonicalType(LHSPointee) == getCanonicalType(ResultType))
      return LHS;
    if (getCanonicalType(RHSPointee) == getCanonicalType(ResultType))
      return RHS;
    return getPointerType(ResultType);
  }
  case Type::BlockPointer:
  {
    // Merge two block pointer types, while trying to preserve typedef info
    QualType LHSPointee = LHS->getAs<BlockPointerType>()->getPointeeType();
    QualType RHSPointee = RHS->getAs<BlockPointerType>()->getPointeeType();
    if (Unqualified) {
      LHSPointee = LHSPointee.getUnqualifiedType();
      RHSPointee = RHSPointee.getUnqualifiedType();
    }
    QualType ResultType = mergeTypes(LHSPointee, RHSPointee, OfBlockPointer,
                                     Unqualified);
    if (ResultType.isNull()) return QualType();
    if (getCanonicalType(LHSPointee) == getCanonicalType(ResultType))
      return LHS;
    if (getCanonicalType(RHSPointee) == getCanonicalType(ResultType))
      return RHS;
    return getBlockPointerType(ResultType);
  }
  case Type::Atomic:
  {
    // Merge two pointer types, while trying to preserve typedef info
    QualType LHSValue = LHS->getAs<AtomicType>()->getValueType();
    QualType RHSValue = RHS->getAs<AtomicType>()->getValueType();
    if (Unqualified) {
      LHSValue = LHSValue.getUnqualifiedType();
      RHSValue = RHSValue.getUnqualifiedType();
    }
    QualType ResultType = mergeTypes(LHSValue, RHSValue, false, 
                                     Unqualified);
    if (ResultType.isNull()) return QualType();
    if (getCanonicalType(LHSValue) == getCanonicalType(ResultType))
      return LHS;
    if (getCanonicalType(RHSValue) == getCanonicalType(ResultType))
      return RHS;
    return getAtomicType(ResultType);
  }
  case Type::ConstantArray:
  {
    const ConstantArrayType* LCAT = getAsConstantArrayType(LHS);
    const ConstantArrayType* RCAT = getAsConstantArrayType(RHS);
    if (LCAT && RCAT && RCAT->getSize() != LCAT->getSize())
      return QualType();

    QualType LHSElem = getAsArrayType(LHS)->getElementType();
    QualType RHSElem = getAsArrayType(RHS)->getElementType();
    if (Unqualified) {
      LHSElem = LHSElem.getUnqualifiedType();
      RHSElem = RHSElem.getUnqualifiedType();
    }
    
    QualType ResultType = mergeTypes(LHSElem, RHSElem, false, Unqualified);
    if (ResultType.isNull()) return QualType();
    if (LCAT && getCanonicalType(LHSElem) == getCanonicalType(ResultType))
      return LHS;
    if (RCAT && getCanonicalType(RHSElem) == getCanonicalType(ResultType))
      return RHS;
    if (LCAT) return getConstantArrayType(ResultType, LCAT->getSize(),
                                          ArrayType::ArraySizeModifier(), 0);
    if (RCAT) return getConstantArrayType(ResultType, RCAT->getSize(),
                                          ArrayType::ArraySizeModifier(), 0);
    const VariableArrayType* LVAT = getAsVariableArrayType(LHS);
    const VariableArrayType* RVAT = getAsVariableArrayType(RHS);
    if (LVAT && getCanonicalType(LHSElem) == getCanonicalType(ResultType))
      return LHS;
    if (RVAT && getCanonicalType(RHSElem) == getCanonicalType(ResultType))
      return RHS;
    if (LVAT) {
      // FIXME: This isn't correct! But tricky to implement because
      // the array's size has to be the size of LHS, but the type
      // has to be different.
      return LHS;
    }
    if (RVAT) {
      // FIXME: This isn't correct! But tricky to implement because
      // the array's size has to be the size of RHS, but the type
      // has to be different.
      return RHS;
    }
    if (getCanonicalType(LHSElem) == getCanonicalType(ResultType)) return LHS;
    if (getCanonicalType(RHSElem) == getCanonicalType(ResultType)) return RHS;
    return getIncompleteArrayType(ResultType,
                                  ArrayType::ArraySizeModifier(), 0);
  }
  case Type::FunctionNoProto:
    return mergeFunctionTypes(LHS, RHS, OfBlockPointer, Unqualified);
  case Type::Record:
  case Type::Enum:
    return QualType();
  case Type::Builtin:
    // Only exactly equal builtin types are compatible, which is tested above.
    return QualType();
  case Type::Complex:
    // Distinct complex types are incompatible.
    return QualType();
  case Type::Vector:
    // FIXME: The merged type should be an ExtVector!
    if (areCompatVectorTypes(LHSCan->getAs<VectorType>(),
                             RHSCan->getAs<VectorType>()))
      return LHS;
    return QualType();
  case Type::ObjCObject: {
    // Check if the types are assignment compatible.
    // FIXME: This should be type compatibility, e.g. whether
    // "LHS x; RHS x;" at global scope is legal.
    const ObjCObjectType* LHSIface = LHS->getAs<ObjCObjectType>();
    const ObjCObjectType* RHSIface = RHS->getAs<ObjCObjectType>();
    if (canAssignObjCInterfaces(LHSIface, RHSIface))
      return LHS;

    return QualType();
  }
  case Type::ObjCObjectPointer: {
    if (OfBlockPointer) {
      if (canAssignObjCInterfacesInBlockPointer(
                                          LHS->getAs<ObjCObjectPointerType>(),
                                          RHS->getAs<ObjCObjectPointerType>(),
                                          BlockReturnType))
        return LHS;
      return QualType();
    }
    if (canAssignObjCInterfaces(LHS->getAs<ObjCObjectPointerType>(),
                                RHS->getAs<ObjCObjectPointerType>()))
      return LHS;

    return QualType();
  }
  }

  llvm_unreachable("Invalid Type::Class!");
}

bool ASTContext::FunctionTypesMatchOnNSConsumedAttrs(
                   const FunctionProtoType *FromFunctionType,
                   const FunctionProtoType *ToFunctionType) {
  if (FromFunctionType->hasAnyConsumedArgs() != 
      ToFunctionType->hasAnyConsumedArgs())
    return false;
  FunctionProtoType::ExtProtoInfo FromEPI = 
    FromFunctionType->getExtProtoInfo();
  FunctionProtoType::ExtProtoInfo ToEPI = 
    ToFunctionType->getExtProtoInfo();
  if (FromEPI.ConsumedArguments && ToEPI.ConsumedArguments)
    for (unsigned ArgIdx = 0, NumArgs = FromFunctionType->getNumArgs();
         ArgIdx != NumArgs; ++ArgIdx)  {
      if (FromEPI.ConsumedArguments[ArgIdx] != 
          ToEPI.ConsumedArguments[ArgIdx])
        return false;
    }
  return true;
}

/// mergeObjCGCQualifiers - This routine merges ObjC's GC attribute of 'LHS' and
/// 'RHS' attributes and returns the merged version; including for function
/// return types.
QualType ASTContext::mergeObjCGCQualifiers(QualType LHS, QualType RHS) {
  QualType LHSCan = getCanonicalType(LHS),
  RHSCan = getCanonicalType(RHS);
  // If two types are identical, they are compatible.
  if (LHSCan == RHSCan)
    return LHS;
  if (RHSCan->isFunctionType()) {
    if (!LHSCan->isFunctionType())
      return QualType();
    QualType OldReturnType = 
      cast<FunctionType>(RHSCan.getTypePtr())->getResultType();
    QualType NewReturnType =
      cast<FunctionType>(LHSCan.getTypePtr())->getResultType();
    QualType ResReturnType = 
      mergeObjCGCQualifiers(NewReturnType, OldReturnType);
    if (ResReturnType.isNull())
      return QualType();
    if (ResReturnType == NewReturnType || ResReturnType == OldReturnType) {
      // id foo(); ... __strong id foo(); or: __strong id foo(); ... id foo();
      // In either case, use OldReturnType to build the new function type.
      const FunctionType *F = LHS->getAs<FunctionType>();
      if (const FunctionProtoType *FPT = cast<FunctionProtoType>(F)) {
        FunctionProtoType::ExtProtoInfo EPI = FPT->getExtProtoInfo();
        EPI.ExtInfo = getFunctionExtInfo(LHS);
        QualType ResultType
          = getFunctionType(OldReturnType, FPT->arg_type_begin(),
                            FPT->getNumArgs(), EPI);
        return ResultType;
      }
    }
    return QualType();
  }
  
  // If the qualifiers are different, the types can still be merged.
  Qualifiers LQuals = LHSCan.getLocalQualifiers();
  Qualifiers RQuals = RHSCan.getLocalQualifiers();
  if (LQuals != RQuals) {
    // If any of these qualifiers are different, we have a type mismatch.
    if (LQuals.getCVRQualifiers() != RQuals.getCVRQualifiers() ||
        LQuals.getAddressSpace() != RQuals.getAddressSpace())
      return QualType();
    
    // Exactly one GC qualifier difference is allowed: __strong is
    // okay if the other type has no GC qualifier but is an Objective
    // C object pointer (i.e. implicitly strong by default).  We fix
    // this by pretending that the unqualified type was actually
    // qualified __strong.
    Qualifiers::GC GC_L = LQuals.getObjCGCAttr();
    Qualifiers::GC GC_R = RQuals.getObjCGCAttr();
    assert((GC_L != GC_R) && "unequal qualifier sets had only equal elements");
    
    if (GC_L == Qualifiers::Weak || GC_R == Qualifiers::Weak)
      return QualType();
    
    if (GC_L == Qualifiers::Strong)
      return LHS;
    if (GC_R == Qualifiers::Strong)
      return RHS;
    return QualType();
  }
  
  if (LHSCan->isObjCObjectPointerType() && RHSCan->isObjCObjectPointerType()) {
    QualType LHSBaseQT = LHS->getAs<ObjCObjectPointerType>()->getPointeeType();
    QualType RHSBaseQT = RHS->getAs<ObjCObjectPointerType>()->getPointeeType();
    QualType ResQT = mergeObjCGCQualifiers(LHSBaseQT, RHSBaseQT);
    if (ResQT == LHSBaseQT)
      return LHS;
    if (ResQT == RHSBaseQT)
      return RHS;
  }
  return QualType();
}

//===----------------------------------------------------------------------===//
//                         Integer Predicates
//===----------------------------------------------------------------------===//

unsigned ASTContext::getIntWidth(QualType T) const {
  if (const EnumType *ET = dyn_cast<EnumType>(T))
    T = ET->getDecl()->getIntegerType();
  if (T->isBooleanType())
    return 1;
  // For builtin types, just use the standard type sizing method
  return (unsigned)getTypeSize(T);
}

QualType ASTContext::getCorrespondingUnsignedType(QualType T) {
  assert(T->hasSignedIntegerRepresentation() && "Unexpected type");
  
  // Turn <4 x signed int> -> <4 x unsigned int>
  if (const VectorType *VTy = T->getAs<VectorType>())
    return getVectorType(getCorrespondingUnsignedType(VTy->getElementType()),
                         VTy->getNumElements(), VTy->getVectorKind());

  // For enums, we return the unsigned version of the base type.
  if (const EnumType *ETy = T->getAs<EnumType>())
    T = ETy->getDecl()->getIntegerType();
  
  const BuiltinType *BTy = T->getAs<BuiltinType>();
  assert(BTy && "Unexpected signed integer type");
  switch (BTy->getKind()) {
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
    return UnsignedCharTy;
  case BuiltinType::Short:
    return UnsignedShortTy;
  case BuiltinType::Int:
    return UnsignedIntTy;
  case BuiltinType::Long:
    return UnsignedLongTy;
  case BuiltinType::LongLong:
    return UnsignedLongLongTy;
  case BuiltinType::Int128:
    return UnsignedInt128Ty;
  default:
    llvm_unreachable("Unexpected signed integer type");
  }
}

ASTMutationListener::~ASTMutationListener() { }


//===----------------------------------------------------------------------===//
//                          Builtin Type Computation
//===----------------------------------------------------------------------===//

/// DecodeTypeFromStr - This decodes one type descriptor from Str, advancing the
/// pointer over the consumed characters.  This returns the resultant type.  If
/// AllowTypeModifiers is false then modifier like * are not parsed, just basic
/// types.  This allows "v2i*" to be parsed as a pointer to a v2i instead of
/// a vector of "i*".
///
/// RequiresICE is filled in on return to indicate whether the value is required
/// to be an Integer Constant Expression.
static QualType DecodeTypeFromStr(const char *&Str, const ASTContext &Context,
                                  ASTContext::GetBuiltinTypeError &Error,
                                  bool &RequiresICE,
                                  bool AllowTypeModifiers) {
  // Modifiers.
  int HowLong = 0;
  bool Signed = false, Unsigned = false;
  RequiresICE = false;
  
  // Read the prefixed modifiers first.
  bool Done = false;
  while (!Done) {
    switch (*Str++) {
    default: Done = true; --Str; break;
    case 'I':
      RequiresICE = true;
      break;
    case 'S':
      assert(!Unsigned && "Can't use both 'S' and 'U' modifiers!");
      assert(!Signed && "Can't use 'S' modifier multiple times!");
      Signed = true;
      break;
    case 'U':
      assert(!Signed && "Can't use both 'S' and 'U' modifiers!");
      assert(!Unsigned && "Can't use 'S' modifier multiple times!");
      Unsigned = true;
      break;
    case 'L':
      assert(HowLong <= 2 && "Can't have LLLL modifier");
      ++HowLong;
      break;
    }
  }

  QualType Type;

  // Read the base type.
  switch (*Str++) {
  default: llvm_unreachable("Unknown builtin type letter!");
  case 'v':
    assert(HowLong == 0 && !Signed && !Unsigned &&
           "Bad modifiers used with 'v'!");
    Type = Context.VoidTy;
    break;
  case 'f':
    assert(HowLong == 0 && !Signed && !Unsigned &&
           "Bad modifiers used with 'f'!");
    Type = Context.FloatTy;
    break;
  case 'd':
    assert(HowLong < 2 && !Signed && !Unsigned &&
           "Bad modifiers used with 'd'!");
    if (HowLong)
      Type = Context.LongDoubleTy;
    else
      Type = Context.DoubleTy;
    break;
  case 's':
    assert(HowLong == 0 && "Bad modifiers used with 's'!");
    if (Unsigned)
      Type = Context.UnsignedShortTy;
    else
      Type = Context.ShortTy;
    break;
  case 'i':
    if (HowLong == 3)
      Type = Unsigned ? Context.UnsignedInt128Ty : Context.Int128Ty;
    else if (HowLong == 2)
      Type = Unsigned ? Context.UnsignedLongLongTy : Context.LongLongTy;
    else if (HowLong == 1)
      Type = Unsigned ? Context.UnsignedLongTy : Context.LongTy;
    else
      Type = Unsigned ? Context.UnsignedIntTy : Context.IntTy;
    break;
  case 'c':
    assert(HowLong == 0 && "Bad modifiers used with 'c'!");
    if (Signed)
      Type = Context.SignedCharTy;
    else if (Unsigned)
      Type = Context.UnsignedCharTy;
    else
      Type = Context.CharTy;
    break;
  case 'b': // boolean
    assert(HowLong == 0 && !Signed && !Unsigned && "Bad modifiers for 'b'!");
    Type = Context.BoolTy;
    break;
  case 'z':  // size_t.
    assert(HowLong == 0 && !Signed && !Unsigned && "Bad modifiers for 'z'!");
    Type = Context.getSizeType();
    break;
  case 'F':
    Type = Context.getCFConstantStringType();
    break;
  case 'G':
    Type = Context.getObjCIdType();
    break;
  case 'H':
    Type = Context.getObjCSelType();
    break;
  case 'a':
    Type = Context.getBuiltinVaListType();
    assert(!Type.isNull() && "builtin va list type not initialized!");
    break;
  case 'A':
    // This is a "reference" to a va_list; however, what exactly
    // this means depends on how va_list is defined. There are two
    // different kinds of va_list: ones passed by value, and ones
    // passed by reference.  An example of a by-value va_list is
    // x86, where va_list is a char*. An example of by-ref va_list
    // is x86-64, where va_list is a __va_list_tag[1]. For x86,
    // we want this argument to be a char*&; for x86-64, we want
    // it to be a __va_list_tag*.
    Type = Context.getBuiltinVaListType();
    assert(!Type.isNull() && "builtin va list type not initialized!");
    if (Type->isArrayType())
      Type = Context.getArrayDecayedType(Type);
    else
      Type = Context.getLValueReferenceType(Type);
    break;
  case 'V': {
    char *End;
    unsigned NumElements = strtoul(Str, &End, 10);
    assert(End != Str && "Missing vector size");
    Str = End;

    QualType ElementType = DecodeTypeFromStr(Str, Context, Error, 
                                             RequiresICE, false);
    assert(!RequiresICE && "Can't require vector ICE");
    
    // TODO: No way to make AltiVec vectors in builtins yet.
    Type = Context.getVectorType(ElementType, NumElements,
                                 VectorType::GenericVector);
    break;
  }
  case 'X': {
    QualType ElementType = DecodeTypeFromStr(Str, Context, Error, RequiresICE,
                                             false);
    assert(!RequiresICE && "Can't require complex ICE");
    Type = Context.getComplexType(ElementType);
    break;
  }  
  case 'Y' : {
    Type = Context.getPointerDiffType();
    break;
  }
  case 'P':
    Type = Context.getFILEType();
    if (Type.isNull()) {
      Error = ASTContext::GE_Missing_stdio;
      return QualType();
    }
    break;
  case 'J':
    if (Signed)
      Type = Context.getsigjmp_bufType();
    else
      Type = Context.getjmp_bufType();

    if (Type.isNull()) {
      Error = ASTContext::GE_Missing_setjmp;
      return QualType();
    }
    break;
  case 'K':
    assert(HowLong == 0 && !Signed && !Unsigned && "Bad modifiers for 'K'!");
    Type = Context.getucontext_tType();

    if (Type.isNull()) {
      Error = ASTContext::GE_Missing_ucontext;
      return QualType();
    }
    break;
  }

  // If there are modifiers and if we're allowed to parse them, go for it.
  Done = !AllowTypeModifiers;
  while (!Done) {
    switch (char c = *Str++) {
    default: Done = true; --Str; break;
    case '*':
    case '&': {
      // Both pointers and references can have their pointee types
      // qualified with an address space.
      char *End;
      unsigned AddrSpace = strtoul(Str, &End, 10);
      if (End != Str && AddrSpace != 0) {
        Type = Context.getAddrSpaceQualType(Type, AddrSpace);
        Str = End;
      }
      if (c == '*')
        Type = Context.getPointerType(Type);
      else
        Type = Context.getLValueReferenceType(Type);
      break;
    }
    // FIXME: There's no way to have a built-in with an rvalue ref arg.
    case 'C':
      Type = Type.withConst();
      break;
    case 'D':
      Type = Context.getVolatileType(Type);
      break;
    case 'R':
      Type = Type.withRestrict();
      break;
    }
  }
  
  assert((!RequiresICE || Type->isIntegralOrEnumerationType()) &&
         "Integer constant 'I' type must be an integer"); 

  return Type;
}

/// GetBuiltinType - Return the type for the specified builtin.
QualType ASTContext::GetBuiltinType(unsigned Id,
                                    GetBuiltinTypeError &Error,
                                    unsigned *IntegerConstantArgs) const {
  const char *TypeStr = BuiltinInfo.GetTypeString(Id);

  SmallVector<QualType, 8> ArgTypes;

  bool RequiresICE = false;
  Error = GE_None;
  QualType ResType = DecodeTypeFromStr(TypeStr, *this, Error,
                                       RequiresICE, true);
  if (Error != GE_None)
    return QualType();
  
  assert(!RequiresICE && "Result of intrinsic cannot be required to be an ICE");
  
  while (TypeStr[0] && TypeStr[0] != '.') {
    QualType Ty = DecodeTypeFromStr(TypeStr, *this, Error, RequiresICE, true);
    if (Error != GE_None)
      return QualType();

    // If this argument is required to be an IntegerConstantExpression and the
    // caller cares, fill in the bitmask we return.
    if (RequiresICE && IntegerConstantArgs)
      *IntegerConstantArgs |= 1 << ArgTypes.size();
    
    // Do array -> pointer decay.  The builtin should use the decayed type.
    if (Ty->isArrayType())
      Ty = getArrayDecayedType(Ty);

    ArgTypes.push_back(Ty);
  }

  assert((TypeStr[0] != '.' || TypeStr[1] == 0) &&
         "'.' should only occur at end of builtin type list!");

  FunctionType::ExtInfo EI;
  if (BuiltinInfo.isNoReturn(Id)) EI = EI.withNoReturn(true);

  bool Variadic = (TypeStr[0] == '.');

  // We really shouldn't be making a no-proto type here, especially in C++.
  if (ArgTypes.empty() && Variadic)
    return getFunctionNoProtoType(ResType, EI);

  FunctionProtoType::ExtProtoInfo EPI;
  EPI.ExtInfo = EI;
  EPI.Variadic = Variadic;

  return getFunctionType(ResType, ArgTypes.data(), ArgTypes.size(), EPI);
}

GVALinkage ASTContext::GetGVALinkageForFunction(const FunctionDecl *FD) {
  GVALinkage External = GVA_StrongExternal;

  Linkage L = FD->getLinkage();
  switch (L) {
  case NoLinkage:
  case InternalLinkage:
  case UniqueExternalLinkage:
    return GVA_Internal;
    
  case ExternalLinkage:
    switch (FD->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      External = GVA_StrongExternal;
      break;

    case TSK_ExplicitInstantiationDefinition:
      return GVA_ExplicitTemplateInstantiation;

    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ImplicitInstantiation:
      External = GVA_TemplateInstantiation;
      break;
    }
  }

  if (!FD->isInlined())
    return External;
    
  if (!getLangOpts().CPlusPlus || FD->hasAttr<GNUInlineAttr>()) {
    // GNU or C99 inline semantics. Determine whether this symbol should be
    // externally visible.
    if (FD->isInlineDefinitionExternallyVisible())
      return External;

    // C99 inline semantics, where the symbol is not externally visible.
    return GVA_C99Inline;
  }

  // C++0x [temp.explicit]p9:
  //   [ Note: The intent is that an inline function that is the subject of 
  //   an explicit instantiation declaration will still be implicitly 
  //   instantiated when used so that the body can be considered for 
  //   inlining, but that no out-of-line copy of the inline function would be
  //   generated in the translation unit. -- end note ]
  if (FD->getTemplateSpecializationKind() 
                                       == TSK_ExplicitInstantiationDeclaration)
    return GVA_C99Inline;

  return GVA_CXXInline;
}

GVALinkage ASTContext::GetGVALinkageForVariable(const VarDecl *VD) {
  // If this is a static data member, compute the kind of template
  // specialization. Otherwise, this variable is not part of a
  // template.
  TemplateSpecializationKind TSK = TSK_Undeclared;
  if (VD->isStaticDataMember())
    TSK = VD->getTemplateSpecializationKind();

  Linkage L = VD->getLinkage();
  if (L == ExternalLinkage && getLangOpts().CPlusPlus &&
      VD->getType()->getLinkage() == UniqueExternalLinkage)
    L = UniqueExternalLinkage;

  switch (L) {
  case NoLinkage:
  case InternalLinkage:
  case UniqueExternalLinkage:
    return GVA_Internal;

  case ExternalLinkage:
    switch (TSK) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      return GVA_StrongExternal;

    case TSK_ExplicitInstantiationDeclaration:
      llvm_unreachable("Variable should not be instantiated");
      // Fall through to treat this like any other instantiation.
        
    case TSK_ExplicitInstantiationDefinition:
      return GVA_ExplicitTemplateInstantiation;

    case TSK_ImplicitInstantiation:
      return GVA_TemplateInstantiation;      
    }
  }

  llvm_unreachable("Invalid Linkage!");
}

bool ASTContext::DeclMustBeEmitted(const Decl *D) {
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (!VD->isFileVarDecl())
      return false;
  } else if (!isa<FunctionDecl>(D))
    return false;

  // Weak references don't produce any output by themselves.
  if (D->hasAttr<WeakRefAttr>())
    return false;

  // Aliases and used decls are required.
  if (D->hasAttr<AliasAttr>() || D->hasAttr<UsedAttr>())
    return true;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // Forward declarations aren't required.
    if (!FD->doesThisDeclarationHaveABody())
      return FD->doesDeclarationForceExternallyVisibleDefinition();

    // Constructors and destructors are required.
    if (FD->hasAttr<ConstructorAttr>() || FD->hasAttr<DestructorAttr>())
      return true;
    
    // The key function for a class is required.
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
      const CXXRecordDecl *RD = MD->getParent();
      if (MD->isOutOfLine() && RD->isDynamicClass()) {
        const CXXMethodDecl *KeyFunc = getKeyFunction(RD);
        if (KeyFunc && KeyFunc->getCanonicalDecl() == MD->getCanonicalDecl())
          return true;
      }
    }

    GVALinkage Linkage = GetGVALinkageForFunction(FD);

    // static, static inline, always_inline, and extern inline functions can
    // always be deferred.  Normal inline functions can be deferred in C99/C++.
    // Implicit template instantiations can also be deferred in C++.
    if (Linkage == GVA_Internal  || Linkage == GVA_C99Inline ||
        Linkage == GVA_CXXInline || Linkage == GVA_TemplateInstantiation)
      return false;
    return true;
  }
  
  const VarDecl *VD = cast<VarDecl>(D);
  assert(VD->isFileVarDecl() && "Expected file scoped var");

  if (VD->isThisDeclarationADefinition() == VarDecl::DeclarationOnly)
    return false;

  // Structs that have non-trivial constructors or destructors are required.

  // FIXME: Handle references.
  // FIXME: Be more selective about which constructors we care about.
  if (const RecordType *RT = VD->getType()->getAs<RecordType>()) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      if (RD->hasDefinition() && !(RD->hasTrivialDefaultConstructor() &&
                                   RD->hasTrivialCopyConstructor() &&
                                   RD->hasTrivialMoveConstructor() &&
                                   RD->hasTrivialDestructor()))
        return true;
    }
  }

  GVALinkage L = GetGVALinkageForVariable(VD);
  if (L == GVA_Internal || L == GVA_TemplateInstantiation) {
    if (!(VD->getInit() && VD->getInit()->HasSideEffects(*this)))
      return false;
  }

  return true;
}

CallingConv ASTContext::getDefaultMethodCallConv() {
  // Pass through to the C++ ABI object
  return ABI->getDefaultMethodCallConv();
}

bool ASTContext::isNearlyEmpty(const CXXRecordDecl *RD) const {
  // Pass through to the C++ ABI object
  return ABI->isNearlyEmpty(RD);
}

MangleContext *ASTContext::createMangleContext() {
  switch (Target->getCXXABI()) {
  case CXXABI_ARM:
  case CXXABI_Itanium:
    return createItaniumMangleContext(*this, getDiagnostics());
  case CXXABI_Microsoft:
    return createMicrosoftMangleContext(*this, getDiagnostics());
  }
  llvm_unreachable("Unsupported ABI");
}

CXXABI::~CXXABI() {}

size_t ASTContext::getSideTableAllocatedMemory() const {
  return ASTRecordLayouts.getMemorySize()
    + llvm::capacity_in_bytes(ObjCLayouts)
    + llvm::capacity_in_bytes(KeyFunctions)
    + llvm::capacity_in_bytes(ObjCImpls)
    + llvm::capacity_in_bytes(BlockVarCopyInits)
    + llvm::capacity_in_bytes(DeclAttrs)
    + llvm::capacity_in_bytes(InstantiatedFromStaticDataMember)
    + llvm::capacity_in_bytes(InstantiatedFromUsingDecl)
    + llvm::capacity_in_bytes(InstantiatedFromUsingShadowDecl)
    + llvm::capacity_in_bytes(InstantiatedFromUnnamedFieldDecl)
    + llvm::capacity_in_bytes(OverriddenMethods)
    + llvm::capacity_in_bytes(Types)
    + llvm::capacity_in_bytes(VariableArrayTypes)
    + llvm::capacity_in_bytes(ClassScopeSpecializationPattern);
}

unsigned ASTContext::getLambdaManglingNumber(CXXMethodDecl *CallOperator) {
  CXXRecordDecl *Lambda = CallOperator->getParent();
  return LambdaMangleContexts[Lambda->getDeclContext()]
           .getManglingNumber(CallOperator);
}


void ASTContext::setParameterIndex(const ParmVarDecl *D, unsigned int index) {
  ParamIndices[D] = index;
}

unsigned ASTContext::getParameterIndex(const ParmVarDecl *D) const {
  ParameterIndexTable::const_iterator I = ParamIndices.find(D);
  assert(I != ParamIndices.end() && 
         "ParmIndices lacks entry set by ParmVarDecl");
  return I->second;
}
