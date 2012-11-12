//===--- ASTReaderDecl.cpp - Decl Deserialization ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ASTReader::ReadDeclRecord method, which is the
// entrypoint for loading a decl.
//
//===----------------------------------------------------------------------===//

#include "ASTCommon.h"
#include "ASTReaderInternals.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/SaveAndRestore.h"
using namespace clang;
using namespace clang::serialization;

//===----------------------------------------------------------------------===//
// Declaration deserialization
//===----------------------------------------------------------------------===//

namespace clang {
  class ASTDeclReader : public DeclVisitor<ASTDeclReader, void> {
    ASTReader &Reader;
    ModuleFile &F;
    const DeclID ThisDeclID;
    const unsigned RawLocation;
    typedef ASTReader::RecordData RecordData;
    const RecordData &Record;
    unsigned &Idx;
    TypeID TypeIDForTypeDecl;
    
    DeclID DeclContextIDForTemplateParmDecl;
    DeclID LexicalDeclContextIDForTemplateParmDecl;

    bool HasPendingBody;

    uint64_t GetCurrentCursorOffset();
    
    SourceLocation ReadSourceLocation(const RecordData &R, unsigned &I) {
      return Reader.ReadSourceLocation(F, R, I);
    }
    
    SourceRange ReadSourceRange(const RecordData &R, unsigned &I) {
      return Reader.ReadSourceRange(F, R, I);
    }
    
    TypeSourceInfo *GetTypeSourceInfo(const RecordData &R, unsigned &I) {
      return Reader.GetTypeSourceInfo(F, R, I);
    }
    
    serialization::DeclID ReadDeclID(const RecordData &R, unsigned &I) {
      return Reader.ReadDeclID(F, R, I);
    }
    
    Decl *ReadDecl(const RecordData &R, unsigned &I) {
      return Reader.ReadDecl(F, R, I);
    }

    template<typename T>
    T *ReadDeclAs(const RecordData &R, unsigned &I) {
      return Reader.ReadDeclAs<T>(F, R, I);
    }

    void ReadQualifierInfo(QualifierInfo &Info,
                           const RecordData &R, unsigned &I) {
      Reader.ReadQualifierInfo(F, Info, R, I);
    }
    
    void ReadDeclarationNameLoc(DeclarationNameLoc &DNLoc, DeclarationName Name,
                                const RecordData &R, unsigned &I) {
      Reader.ReadDeclarationNameLoc(F, DNLoc, Name, R, I);
    }
    
    void ReadDeclarationNameInfo(DeclarationNameInfo &NameInfo,
                                const RecordData &R, unsigned &I) {
      Reader.ReadDeclarationNameInfo(F, NameInfo, R, I);
    }

    serialization::SubmoduleID readSubmoduleID(const RecordData &R, 
                                               unsigned &I) {
      if (I >= R.size())
        return 0;
      
      return Reader.getGlobalSubmoduleID(F, R[I++]);
    }
    
    Module *readModule(const RecordData &R, unsigned &I) {
      return Reader.getSubmodule(readSubmoduleID(R, I));
    }
    
    void ReadCXXDefinitionData(struct CXXRecordDecl::DefinitionData &Data,
                               const RecordData &R, unsigned &I);

    /// \brief RAII class used to capture the first ID within a redeclaration
    /// chain and to introduce it into the list of pending redeclaration chains
    /// on destruction.
    ///
    /// The caller can choose not to introduce this ID into the redeclaration
    /// chain by calling \c suppress().
    class RedeclarableResult {
      ASTReader &Reader;
      GlobalDeclID FirstID;
      mutable bool Owning;
      
      void operator=(RedeclarableResult &) LLVM_DELETED_FUNCTION;
      
    public:
      RedeclarableResult(ASTReader &Reader, GlobalDeclID FirstID)
        : Reader(Reader), FirstID(FirstID), Owning(true) { }

      RedeclarableResult(const RedeclarableResult &Other)
        : Reader(Other.Reader), FirstID(Other.FirstID), Owning(Other.Owning) 
      { 
        Other.Owning = false;
      }

      ~RedeclarableResult() {
        // FIXME: We want to suppress this when the declaration is local to
        // a function, since there's no reason to search other AST files
        // for redeclarations (they can't exist). However, this is hard to 
        // do locally because the declaration hasn't necessarily loaded its
        // declaration context yet. Also, local externs still have the function
        // as their (semantic) declaration context, which is wrong and would
        // break this optimize.
        
        if (FirstID && Owning && Reader.PendingDeclChainsKnown.insert(FirstID))
          Reader.PendingDeclChains.push_back(FirstID);
      }
      
      /// \brief Retrieve the first ID.
      GlobalDeclID getFirstID() const { return FirstID; }
      
      /// \brief Do not introduce this declaration ID into the set of pending
      /// declaration chains.
      void suppress() {
        Owning = false;
      }
    };
    
    /// \brief Class used to capture the result of searching for an existing
    /// declaration of a specific kind and name, along with the ability
    /// to update the place where this result was found (the declaration
    /// chain hanging off an identifier or the DeclContext we searched in)
    /// if requested.
    class FindExistingResult {
      ASTReader &Reader;
      NamedDecl *New;
      NamedDecl *Existing;
      mutable bool AddResult;
      
      void operator=(FindExistingResult&) LLVM_DELETED_FUNCTION;
      
    public:
      FindExistingResult(ASTReader &Reader)
        : Reader(Reader), New(0), Existing(0), AddResult(false) { }
      
      FindExistingResult(ASTReader &Reader, NamedDecl *New, NamedDecl *Existing)
        : Reader(Reader), New(New), Existing(Existing), AddResult(true) { }
      
      FindExistingResult(const FindExistingResult &Other)
        : Reader(Other.Reader), New(Other.New), Existing(Other.Existing), 
          AddResult(Other.AddResult)
      {
        Other.AddResult = false;
      }
      
      ~FindExistingResult();
      
      /// \brief Suppress the addition of this result into the known set of
      /// names.
      void suppress() { AddResult = false; }
      
      operator NamedDecl*() const { return Existing; }
      
      template<typename T>
      operator T*() const { return dyn_cast_or_null<T>(Existing); }
    };
    
    FindExistingResult findExisting(NamedDecl *D);
    
  public:
    ASTDeclReader(ASTReader &Reader, ModuleFile &F,
                  DeclID thisDeclID,
                  unsigned RawLocation,
                  const RecordData &Record, unsigned &Idx)
      : Reader(Reader), F(F), ThisDeclID(thisDeclID),
        RawLocation(RawLocation), Record(Record), Idx(Idx),
        TypeIDForTypeDecl(0), HasPendingBody(false) { }

    static void attachPreviousDecl(Decl *D, Decl *previous);
    static void attachLatestDecl(Decl *D, Decl *latest);

    /// \brief Determine whether this declaration has a pending body.
    bool hasPendingBody() const { return HasPendingBody; }

    void Visit(Decl *D);

    void UpdateDecl(Decl *D, ModuleFile &ModuleFile,
                    const RecordData &Record);

    static void setNextObjCCategory(ObjCCategoryDecl *Cat,
                                    ObjCCategoryDecl *Next) {
      Cat->NextClassCategory = Next;
    }

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitLabelDecl(LabelDecl *LD);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeDecl(TypeDecl *TD);
    void VisitTypedefNameDecl(TypedefNameDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
    void VisitTypeAliasDecl(TypeAliasDecl *TD);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    void VisitTagDecl(TagDecl *TD);
    void VisitEnumDecl(EnumDecl *ED);
    void VisitRecordDecl(RecordDecl *RD);
    void VisitCXXRecordDecl(CXXRecordDecl *D);
    void VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
                                       ClassScopeFunctionSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    void VisitValueDecl(ValueDecl *VD);
    void VisitEnumConstantDecl(EnumConstantDecl *ECD);
    void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    void VisitDeclaratorDecl(DeclaratorDecl *DD);
    void VisitFunctionDecl(FunctionDecl *FD);
    void VisitCXXMethodDecl(CXXMethodDecl *D);
    void VisitCXXConstructorDecl(CXXConstructorDecl *D);
    void VisitCXXDestructorDecl(CXXDestructorDecl *D);
    void VisitCXXConversionDecl(CXXConversionDecl *D);
    void VisitFieldDecl(FieldDecl *FD);
    void VisitIndirectFieldDecl(IndirectFieldDecl *FD);
    void VisitVarDecl(VarDecl *VD);
    void VisitImplicitParamDecl(ImplicitParamDecl *PD);
    void VisitParmVarDecl(ParmVarDecl *PD);
    void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    void VisitTemplateDecl(TemplateDecl *D);
    RedeclarableResult VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    void VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *AD);
    void VisitImportDecl(ImportDecl *D);
    void VisitAccessSpecDecl(AccessSpecDecl *D);
    void VisitFriendDecl(FriendDecl *D);
    void VisitFriendTemplateDecl(FriendTemplateDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitBlockDecl(BlockDecl *BD);

    std::pair<uint64_t, uint64_t> VisitDeclContext(DeclContext *DC);
    
    template<typename T> 
    RedeclarableResult VisitRedeclarable(Redeclarable<T> *D);

    template<typename T>
    void mergeRedeclarable(Redeclarable<T> *D, RedeclarableResult &Redecl);
    
    // FIXME: Reorder according to DeclNodes.td?
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCContainerDecl(ObjCContainerDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCIvarDecl(ObjCIvarDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCImplDecl(ObjCImplDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
  };
}

uint64_t ASTDeclReader::GetCurrentCursorOffset() {
  return F.DeclsCursor.GetCurrentBitNo() + F.GlobalBitOffset;
}

void ASTDeclReader::Visit(Decl *D) {
  DeclVisitor<ASTDeclReader, void>::Visit(D);

  if (DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D)) {
    if (DD->DeclInfo) {
      DeclaratorDecl::ExtInfo *Info =
          DD->DeclInfo.get<DeclaratorDecl::ExtInfo *>();
      Info->TInfo =
          GetTypeSourceInfo(Record, Idx);
    }
    else {
      DD->DeclInfo = GetTypeSourceInfo(Record, Idx);
    }
  }

  if (TypeDecl *TD = dyn_cast<TypeDecl>(D)) {
    // if we have a fully initialized TypeDecl, we can safely read its type now.
    TD->setTypeForDecl(Reader.GetType(TypeIDForTypeDecl).getTypePtrOrNull());
  } else if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    // if we have a fully initialized TypeDecl, we can safely read its type now.
    ID->TypeForDecl = Reader.GetType(TypeIDForTypeDecl).getTypePtrOrNull();
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // FunctionDecl's body was written last after all other Stmts/Exprs.
    // We only read it if FD doesn't already have a body (e.g., from another
    // module).
    // FIXME: Also consider = default and = delete.
    // FIXME: Can we diagnose ODR violations somehow?
    if (Record[Idx++]) {
      Reader.PendingBodies[FD] = GetCurrentCursorOffset();
      HasPendingBody = true;
    }
  } else if (D->isTemplateParameter()) {
    // If we have a fully initialized template parameter, we can now
    // set its DeclContext.
    DeclContext *SemaDC = cast<DeclContext>(
                              Reader.GetDecl(DeclContextIDForTemplateParmDecl));
    DeclContext *LexicalDC = cast<DeclContext>(
                       Reader.GetDecl(LexicalDeclContextIDForTemplateParmDecl));
    D->setDeclContextsImpl(SemaDC, LexicalDC, Reader.getContext());
  }
}

void ASTDeclReader::VisitDecl(Decl *D) {
  if (D->isTemplateParameter()) {
    // We don't want to deserialize the DeclContext of a template
    // parameter immediately, because the template parameter might be
    // used in the formulation of its DeclContext. Use the translation
    // unit DeclContext as a placeholder.
    DeclContextIDForTemplateParmDecl = ReadDeclID(Record, Idx);
    LexicalDeclContextIDForTemplateParmDecl = ReadDeclID(Record, Idx);
    D->setDeclContext(Reader.getContext().getTranslationUnitDecl()); 
  } else {
    DeclContext *SemaDC = ReadDeclAs<DeclContext>(Record, Idx);
    DeclContext *LexicalDC = ReadDeclAs<DeclContext>(Record, Idx);
    // Avoid calling setLexicalDeclContext() directly because it uses
    // Decl::getASTContext() internally which is unsafe during derialization.
    D->setDeclContextsImpl(SemaDC, LexicalDC, Reader.getContext());
  }
  D->setLocation(Reader.ReadSourceLocation(F, RawLocation));
  D->setInvalidDecl(Record[Idx++]);
  if (Record[Idx++]) { // hasAttrs
    AttrVec Attrs;
    Reader.ReadAttributes(F, Attrs, Record, Idx);
    // Avoid calling setAttrs() directly because it uses Decl::getASTContext()
    // internally which is unsafe during derialization.
    D->setAttrsImpl(Attrs, Reader.getContext());
  }
  D->setImplicit(Record[Idx++]);
  D->setUsed(Record[Idx++]);
  D->setReferenced(Record[Idx++]);
  D->setTopLevelDeclInObjCContainer(Record[Idx++]);
  D->setAccess((AccessSpecifier)Record[Idx++]);
  D->FromASTFile = true;
  D->setModulePrivate(Record[Idx++]);
  D->Hidden = D->isModulePrivate();
  
  // Determine whether this declaration is part of a (sub)module. If so, it
  // may not yet be visible.
  if (unsigned SubmoduleID = readSubmoduleID(Record, Idx)) {
    // Store the owning submodule ID in the declaration.
    D->setOwningModuleID(SubmoduleID);
    
    // Module-private declarations are never visible, so there is no work to do.
    if (!D->isModulePrivate()) {
      if (Module *Owner = Reader.getSubmodule(SubmoduleID)) {
        if (Owner->NameVisibility != Module::AllVisible) {
          // The owning module is not visible. Mark this declaration as hidden.
          D->Hidden = true;
          
          // Note that this declaration was hidden because its owning module is 
          // not yet visible.
          Reader.HiddenNamesMap[Owner].push_back(D);
        }
      }
    }
  }
}

void ASTDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  llvm_unreachable("Translation units are not serialized");
}

void ASTDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(F, Record, Idx));
}

void ASTDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  TD->setLocStart(ReadSourceLocation(Record, Idx));
  // Delay type reading until after we have fully initialized the decl.
  TypeIDForTypeDecl = Reader.getGlobalTypeID(F, Record[Idx++]);
}

void ASTDeclReader::VisitTypedefNameDecl(TypedefNameDecl *TD) {
  RedeclarableResult Redecl = VisitRedeclarable(TD);
  VisitTypeDecl(TD);
  
  TD->setTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
  mergeRedeclarable(TD, Redecl);
}

void ASTDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  VisitTypedefNameDecl(TD);
}

void ASTDeclReader::VisitTypeAliasDecl(TypeAliasDecl *TD) {
  VisitTypedefNameDecl(TD);
}

void ASTDeclReader::VisitTagDecl(TagDecl *TD) {
  RedeclarableResult Redecl = VisitRedeclarable(TD);
  VisitTypeDecl(TD);
  
  TD->IdentifierNamespace = Record[Idx++];
  TD->setTagKind((TagDecl::TagKind)Record[Idx++]);
  TD->setCompleteDefinition(Record[Idx++]);
  TD->setEmbeddedInDeclarator(Record[Idx++]);
  TD->setFreeStanding(Record[Idx++]);
  TD->setRBraceLoc(ReadSourceLocation(Record, Idx));
  
  if (Record[Idx++]) { // hasExtInfo
    TagDecl::ExtInfo *Info = new (Reader.getContext()) TagDecl::ExtInfo();
    ReadQualifierInfo(*Info, Record, Idx);
    TD->TypedefNameDeclOrQualifier = Info;
  } else
    TD->setTypedefNameForAnonDecl(ReadDeclAs<TypedefNameDecl>(Record, Idx));

  mergeRedeclarable(TD, Redecl);  
}

void ASTDeclReader::VisitEnumDecl(EnumDecl *ED) {
  VisitTagDecl(ED);
  if (TypeSourceInfo *TI = Reader.GetTypeSourceInfo(F, Record, Idx))
    ED->setIntegerTypeSourceInfo(TI);
  else
    ED->setIntegerType(Reader.readType(F, Record, Idx));
  ED->setPromotionType(Reader.readType(F, Record, Idx));
  ED->setNumPositiveBits(Record[Idx++]);
  ED->setNumNegativeBits(Record[Idx++]);
  ED->IsScoped = Record[Idx++];
  ED->IsScopedUsingClassTag = Record[Idx++];
  ED->IsFixed = Record[Idx++];

  if (EnumDecl *InstED = ReadDeclAs<EnumDecl>(Record, Idx)) {
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    ED->setInstantiationOfMemberEnum(Reader.getContext(), InstED, TSK);
    ED->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
  }
}

void ASTDeclReader::VisitRecordDecl(RecordDecl *RD) {
  VisitTagDecl(RD);
  RD->setHasFlexibleArrayMember(Record[Idx++]);
  RD->setAnonymousStructOrUnion(Record[Idx++]);
  RD->setHasObjectMember(Record[Idx++]);
}

void ASTDeclReader::VisitValueDecl(ValueDecl *VD) {
  VisitNamedDecl(VD);
  VD->setType(Reader.readType(F, Record, Idx));
}

void ASTDeclReader::VisitEnumConstantDecl(EnumConstantDecl *ECD) {
  VisitValueDecl(ECD);
  if (Record[Idx++])
    ECD->setInitExpr(Reader.ReadExpr(F));
  ECD->setInitVal(Reader.ReadAPSInt(Record, Idx));
}

void ASTDeclReader::VisitDeclaratorDecl(DeclaratorDecl *DD) {
  VisitValueDecl(DD);
  DD->setInnerLocStart(ReadSourceLocation(Record, Idx));
  if (Record[Idx++]) { // hasExtInfo
    DeclaratorDecl::ExtInfo *Info
        = new (Reader.getContext()) DeclaratorDecl::ExtInfo();
    ReadQualifierInfo(*Info, Record, Idx);
    DD->DeclInfo = Info;
  }
}

void ASTDeclReader::VisitFunctionDecl(FunctionDecl *FD) {
  RedeclarableResult Redecl = VisitRedeclarable(FD);
  VisitDeclaratorDecl(FD);

  ReadDeclarationNameLoc(FD->DNLoc, FD->getDeclName(), Record, Idx);
  FD->IdentifierNamespace = Record[Idx++];
  
  // FunctionDecl's body is handled last at ASTDeclReader::Visit,
  // after everything else is read.
  
  FD->SClass = (StorageClass)Record[Idx++];
  FD->SClassAsWritten = (StorageClass)Record[Idx++];
  FD->IsInline = Record[Idx++];
  FD->IsInlineSpecified = Record[Idx++];
  FD->IsVirtualAsWritten = Record[Idx++];
  FD->IsPure = Record[Idx++];
  FD->HasInheritedPrototype = Record[Idx++];
  FD->HasWrittenPrototype = Record[Idx++];
  FD->IsDeleted = Record[Idx++];
  FD->IsTrivial = Record[Idx++];
  FD->IsDefaulted = Record[Idx++];
  FD->IsExplicitlyDefaulted = Record[Idx++];
  FD->HasImplicitReturnZero = Record[Idx++];
  FD->IsConstexpr = Record[Idx++];
  FD->EndRangeLoc = ReadSourceLocation(Record, Idx);

  switch ((FunctionDecl::TemplatedKind)Record[Idx++]) {
  case FunctionDecl::TK_NonTemplate:
    mergeRedeclarable(FD, Redecl);      
    break;
  case FunctionDecl::TK_FunctionTemplate:
    FD->setDescribedFunctionTemplate(ReadDeclAs<FunctionTemplateDecl>(Record, 
                                                                      Idx));
    break;
  case FunctionDecl::TK_MemberSpecialization: {
    FunctionDecl *InstFD = ReadDeclAs<FunctionDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    FD->setInstantiationOfMemberFunction(Reader.getContext(), InstFD, TSK);
    FD->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
    break;
  }
  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    FunctionTemplateDecl *Template = ReadDeclAs<FunctionTemplateDecl>(Record, 
                                                                      Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    
    // Template arguments.
    SmallVector<TemplateArgument, 8> TemplArgs;
    Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
    
    // Template args as written.
    SmallVector<TemplateArgumentLoc, 8> TemplArgLocs;
    SourceLocation LAngleLoc, RAngleLoc;
    bool HasTemplateArgumentsAsWritten = Record[Idx++];
    if (HasTemplateArgumentsAsWritten) {
      unsigned NumTemplateArgLocs = Record[Idx++];
      TemplArgLocs.reserve(NumTemplateArgLocs);
      for (unsigned i=0; i != NumTemplateArgLocs; ++i)
        TemplArgLocs.push_back(
            Reader.ReadTemplateArgumentLoc(F, Record, Idx));
  
      LAngleLoc = ReadSourceLocation(Record, Idx);
      RAngleLoc = ReadSourceLocation(Record, Idx);
    }
    
    SourceLocation POI = ReadSourceLocation(Record, Idx);

    ASTContext &C = Reader.getContext();
    TemplateArgumentList *TemplArgList
      = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), TemplArgs.size());
    TemplateArgumentListInfo TemplArgsInfo(LAngleLoc, RAngleLoc);
    for (unsigned i=0, e = TemplArgLocs.size(); i != e; ++i)
      TemplArgsInfo.addArgument(TemplArgLocs[i]);
    FunctionTemplateSpecializationInfo *FTInfo
        = FunctionTemplateSpecializationInfo::Create(C, FD, Template, TSK,
                                                     TemplArgList,
                             HasTemplateArgumentsAsWritten ? &TemplArgsInfo : 0,
                                                     POI);
    FD->TemplateOrSpecialization = FTInfo;

    if (FD->isCanonicalDecl()) { // if canonical add to template's set.
      // The template that contains the specializations set. It's not safe to
      // use getCanonicalDecl on Template since it may still be initializing.
      FunctionTemplateDecl *CanonTemplate
        = ReadDeclAs<FunctionTemplateDecl>(Record, Idx);
      // Get the InsertPos by FindNodeOrInsertPos() instead of calling
      // InsertNode(FTInfo) directly to avoid the getASTContext() call in
      // FunctionTemplateSpecializationInfo's Profile().
      // We avoid getASTContext because a decl in the parent hierarchy may
      // be initializing.
      llvm::FoldingSetNodeID ID;
      FunctionTemplateSpecializationInfo::Profile(ID, TemplArgs.data(),
                                                  TemplArgs.size(), C);
      void *InsertPos = 0;
      CanonTemplate->getSpecializations().FindNodeOrInsertPos(ID, InsertPos);
      if (InsertPos)
        CanonTemplate->getSpecializations().InsertNode(FTInfo, InsertPos);
      else
        assert(0 && "Another specialization already inserted!");
    }
    break;
  }
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
    // Templates.
    UnresolvedSet<8> TemplDecls;
    unsigned NumTemplates = Record[Idx++];
    while (NumTemplates--)
      TemplDecls.addDecl(ReadDeclAs<NamedDecl>(Record, Idx));
    
    // Templates args.
    TemplateArgumentListInfo TemplArgs;
    unsigned NumArgs = Record[Idx++];
    while (NumArgs--)
      TemplArgs.addArgument(Reader.ReadTemplateArgumentLoc(F, Record, Idx));
    TemplArgs.setLAngleLoc(ReadSourceLocation(Record, Idx));
    TemplArgs.setRAngleLoc(ReadSourceLocation(Record, Idx));
    
    FD->setDependentTemplateSpecialization(Reader.getContext(),
                                           TemplDecls, TemplArgs);
    break;
  }
  }

  // Read in the parameters.
  unsigned NumParams = Record[Idx++];
  SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(ReadDeclAs<ParmVarDecl>(Record, Idx));
  FD->setParams(Reader.getContext(), Params);
}

void ASTDeclReader::VisitObjCMethodDecl(ObjCMethodDecl *MD) {
  VisitNamedDecl(MD);
  if (Record[Idx++]) {
    // Load the body on-demand. Most clients won't care, because method
    // definitions rarely show up in headers.
    Reader.PendingBodies[MD] = GetCurrentCursorOffset();
    HasPendingBody = true;
    MD->setSelfDecl(ReadDeclAs<ImplicitParamDecl>(Record, Idx));
    MD->setCmdDecl(ReadDeclAs<ImplicitParamDecl>(Record, Idx));
  }
  MD->setInstanceMethod(Record[Idx++]);
  MD->setVariadic(Record[Idx++]);
  MD->setPropertyAccessor(Record[Idx++]);
  MD->setDefined(Record[Idx++]);
  MD->IsOverriding = Record[Idx++];

  MD->IsRedeclaration = Record[Idx++];
  MD->HasRedeclaration = Record[Idx++];
  if (MD->HasRedeclaration)
    Reader.getContext().setObjCMethodRedeclaration(MD,
                                       ReadDeclAs<ObjCMethodDecl>(Record, Idx));

  MD->setDeclImplementation((ObjCMethodDecl::ImplementationControl)Record[Idx++]);
  MD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  MD->SetRelatedResultType(Record[Idx++]);
  MD->setResultType(Reader.readType(F, Record, Idx));
  MD->setResultTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
  MD->DeclEndLoc = ReadSourceLocation(Record, Idx);
  unsigned NumParams = Record[Idx++];
  SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(ReadDeclAs<ParmVarDecl>(Record, Idx));

  MD->SelLocsKind = Record[Idx++];
  unsigned NumStoredSelLocs = Record[Idx++];
  SmallVector<SourceLocation, 16> SelLocs;
  SelLocs.reserve(NumStoredSelLocs);
  for (unsigned i = 0; i != NumStoredSelLocs; ++i)
    SelLocs.push_back(ReadSourceLocation(Record, Idx));

  MD->setParamsAndSelLocs(Reader.getContext(), Params, SelLocs);
}

void ASTDeclReader::VisitObjCContainerDecl(ObjCContainerDecl *CD) {
  VisitNamedDecl(CD);
  CD->setAtStartLoc(ReadSourceLocation(Record, Idx));
  CD->setAtEndRange(ReadSourceRange(Record, Idx));
}

void ASTDeclReader::VisitObjCInterfaceDecl(ObjCInterfaceDecl *ID) {
  RedeclarableResult Redecl = VisitRedeclarable(ID);
  VisitObjCContainerDecl(ID);
  TypeIDForTypeDecl = Reader.getGlobalTypeID(F, Record[Idx++]);
  mergeRedeclarable(ID, Redecl);
  
  if (Record[Idx++]) {
    // Read the definition.
    ID->allocateDefinitionData();
    
    // Set the definition data of the canonical declaration, so other
    // redeclarations will see it.
    ID->getCanonicalDecl()->Data = ID->Data;
    
    ObjCInterfaceDecl::DefinitionData &Data = ID->data();
    
    // Read the superclass.
    Data.SuperClass = ReadDeclAs<ObjCInterfaceDecl>(Record, Idx);
    Data.SuperClassLoc = ReadSourceLocation(Record, Idx);

    Data.EndLoc = ReadSourceLocation(Record, Idx);
    
    // Read the directly referenced protocols and their SourceLocations.
    unsigned NumProtocols = Record[Idx++];
    SmallVector<ObjCProtocolDecl *, 16> Protocols;
    Protocols.reserve(NumProtocols);
    for (unsigned I = 0; I != NumProtocols; ++I)
      Protocols.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
    SmallVector<SourceLocation, 16> ProtoLocs;
    ProtoLocs.reserve(NumProtocols);
    for (unsigned I = 0; I != NumProtocols; ++I)
      ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
    ID->setProtocolList(Protocols.data(), NumProtocols, ProtoLocs.data(),
                        Reader.getContext());
  
    // Read the transitive closure of protocols referenced by this class.
    NumProtocols = Record[Idx++];
    Protocols.clear();
    Protocols.reserve(NumProtocols);
    for (unsigned I = 0; I != NumProtocols; ++I)
      Protocols.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
    ID->data().AllReferencedProtocols.set(Protocols.data(), NumProtocols,
                                          Reader.getContext());
  
    // We will rebuild this list lazily.
    ID->setIvarList(0);
    
    // Note that we have deserialized a definition.
    Reader.PendingDefinitions.insert(ID);
    
    // Note that we've loaded this Objective-C class.
    Reader.ObjCClassesLoaded.push_back(ID);
  } else {
    ID->Data = ID->getCanonicalDecl()->Data;
  }
}

void ASTDeclReader::VisitObjCIvarDecl(ObjCIvarDecl *IVD) {
  VisitFieldDecl(IVD);
  IVD->setAccessControl((ObjCIvarDecl::AccessControl)Record[Idx++]);
  // This field will be built lazily.
  IVD->setNextIvar(0);
  bool synth = Record[Idx++];
  IVD->setSynthesize(synth);
}

void ASTDeclReader::VisitObjCProtocolDecl(ObjCProtocolDecl *PD) {
  RedeclarableResult Redecl = VisitRedeclarable(PD);
  VisitObjCContainerDecl(PD);
  mergeRedeclarable(PD, Redecl);
  
  if (Record[Idx++]) {
    // Read the definition.
    PD->allocateDefinitionData();
    
    // Set the definition data of the canonical declaration, so other
    // redeclarations will see it.
    PD->getCanonicalDecl()->Data = PD->Data;

    unsigned NumProtoRefs = Record[Idx++];
    SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
    ProtoRefs.reserve(NumProtoRefs);
    for (unsigned I = 0; I != NumProtoRefs; ++I)
      ProtoRefs.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
    SmallVector<SourceLocation, 16> ProtoLocs;
    ProtoLocs.reserve(NumProtoRefs);
    for (unsigned I = 0; I != NumProtoRefs; ++I)
      ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
    PD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                        Reader.getContext());
    
    // Note that we have deserialized a definition.
    Reader.PendingDefinitions.insert(PD);
  } else {
    PD->Data = PD->getCanonicalDecl()->Data;
  }
}

void ASTDeclReader::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *FD) {
  VisitFieldDecl(FD);
}

void ASTDeclReader::VisitObjCCategoryDecl(ObjCCategoryDecl *CD) {
  VisitObjCContainerDecl(CD);
  CD->setCategoryNameLoc(ReadSourceLocation(Record, Idx));
  CD->setIvarLBraceLoc(ReadSourceLocation(Record, Idx));
  CD->setIvarRBraceLoc(ReadSourceLocation(Record, Idx));
  
  // Note that this category has been deserialized. We do this before
  // deserializing the interface declaration, so that it will consider this
  /// category.
  Reader.CategoriesDeserialized.insert(CD);

  CD->ClassInterface = ReadDeclAs<ObjCInterfaceDecl>(Record, Idx);
  unsigned NumProtoRefs = Record[Idx++];
  SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
  SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
  CD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                      Reader.getContext());
}

void ASTDeclReader::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *CAD) {
  VisitNamedDecl(CAD);
  CAD->setClassInterface(ReadDeclAs<ObjCInterfaceDecl>(Record, Idx));
}

void ASTDeclReader::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  D->setAtLoc(ReadSourceLocation(Record, Idx));
  D->setLParenLoc(ReadSourceLocation(Record, Idx));
  D->setType(GetTypeSourceInfo(Record, Idx));
  // FIXME: stable encoding
  D->setPropertyAttributes(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
  D->setPropertyAttributesAsWritten(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
  // FIXME: stable encoding
  D->setPropertyImplementation(
                            (ObjCPropertyDecl::PropertyControl)Record[Idx++]);
  D->setGetterName(Reader.ReadDeclarationName(F,Record, Idx).getObjCSelector());
  D->setSetterName(Reader.ReadDeclarationName(F,Record, Idx).getObjCSelector());
  D->setGetterMethodDecl(ReadDeclAs<ObjCMethodDecl>(Record, Idx));
  D->setSetterMethodDecl(ReadDeclAs<ObjCMethodDecl>(Record, Idx));
  D->setPropertyIvarDecl(ReadDeclAs<ObjCIvarDecl>(Record, Idx));
}

void ASTDeclReader::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitObjCContainerDecl(D);
  D->setClassInterface(ReadDeclAs<ObjCInterfaceDecl>(Record, Idx));
}

void ASTDeclReader::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  D->setIdentifier(Reader.GetIdentifierInfo(F, Record, Idx));
  D->CategoryNameLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  D->setSuperClass(ReadDeclAs<ObjCInterfaceDecl>(Record, Idx));
  D->setIvarLBraceLoc(ReadSourceLocation(Record, Idx));
  D->setIvarRBraceLoc(ReadSourceLocation(Record, Idx));
  D->setHasNonZeroConstructors(Record[Idx++]);
  D->setHasDestructors(Record[Idx++]);
  llvm::tie(D->IvarInitializers, D->NumIvarInitializers)
      = Reader.ReadCXXCtorInitializers(F, Record, Idx);
}


void ASTDeclReader::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  D->setAtLoc(ReadSourceLocation(Record, Idx));
  D->setPropertyDecl(ReadDeclAs<ObjCPropertyDecl>(Record, Idx));
  D->PropertyIvarDecl = ReadDeclAs<ObjCIvarDecl>(Record, Idx);
  D->IvarLoc = ReadSourceLocation(Record, Idx);
  D->setGetterCXXConstructor(Reader.ReadExpr(F));
  D->setSetterCXXAssignment(Reader.ReadExpr(F));
}

void ASTDeclReader::VisitFieldDecl(FieldDecl *FD) {
  VisitDeclaratorDecl(FD);
  FD->Mutable = Record[Idx++];
  if (int BitWidthOrInitializer = Record[Idx++]) {
    FD->InitializerOrBitWidth.setInt(BitWidthOrInitializer - 1);
    FD->InitializerOrBitWidth.setPointer(Reader.ReadExpr(F));
  }
  if (!FD->getDeclName()) {
    if (FieldDecl *Tmpl = ReadDeclAs<FieldDecl>(Record, Idx))
      Reader.getContext().setInstantiatedFromUnnamedFieldDecl(FD, Tmpl);
  }
}

void ASTDeclReader::VisitIndirectFieldDecl(IndirectFieldDecl *FD) {
  VisitValueDecl(FD);

  FD->ChainingSize = Record[Idx++];
  assert(FD->ChainingSize >= 2 && "Anonymous chaining must be >= 2");
  FD->Chaining = new (Reader.getContext())NamedDecl*[FD->ChainingSize];

  for (unsigned I = 0; I != FD->ChainingSize; ++I)
    FD->Chaining[I] = ReadDeclAs<NamedDecl>(Record, Idx);
}

void ASTDeclReader::VisitVarDecl(VarDecl *VD) {
  RedeclarableResult Redecl = VisitRedeclarable(VD);
  VisitDeclaratorDecl(VD);
  
  VD->VarDeclBits.SClass = (StorageClass)Record[Idx++];
  VD->VarDeclBits.SClassAsWritten = (StorageClass)Record[Idx++];
  VD->VarDeclBits.ThreadSpecified = Record[Idx++];
  VD->VarDeclBits.InitStyle = Record[Idx++];
  VD->VarDeclBits.ExceptionVar = Record[Idx++];
  VD->VarDeclBits.NRVOVariable = Record[Idx++];
  VD->VarDeclBits.CXXForRangeDecl = Record[Idx++];
  VD->VarDeclBits.ARCPseudoStrong = Record[Idx++];
  VD->VarDeclBits.IsConstexpr = Record[Idx++];
  
  // Only true variables (not parameters or implicit parameters) can be merged.
  if (VD->getKind() == Decl::Var)
    mergeRedeclarable(VD, Redecl);
  
  if (uint64_t Val = Record[Idx++]) {
    VD->setInit(Reader.ReadExpr(F));
    if (Val > 1) {
      EvaluatedStmt *Eval = VD->ensureEvaluatedStmt();
      Eval->CheckedICE = true;
      Eval->IsICE = Val == 3;
    }
  }

  if (Record[Idx++]) { // HasMemberSpecializationInfo.
    VarDecl *Tmpl = ReadDeclAs<VarDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    Reader.getContext().setInstantiatedFromStaticDataMember(VD, Tmpl, TSK,POI);
  }
}

void ASTDeclReader::VisitImplicitParamDecl(ImplicitParamDecl *PD) {
  VisitVarDecl(PD);
}

void ASTDeclReader::VisitParmVarDecl(ParmVarDecl *PD) {
  VisitVarDecl(PD);
  unsigned isObjCMethodParam = Record[Idx++];
  unsigned scopeDepth = Record[Idx++];
  unsigned scopeIndex = Record[Idx++];
  unsigned declQualifier = Record[Idx++];
  if (isObjCMethodParam) {
    assert(scopeDepth == 0);
    PD->setObjCMethodScopeInfo(scopeIndex);
    PD->ParmVarDeclBits.ScopeDepthOrObjCQuals = declQualifier;
  } else {
    PD->setScopeInfo(scopeDepth, scopeIndex);
  }
  PD->ParmVarDeclBits.IsKNRPromoted = Record[Idx++];
  PD->ParmVarDeclBits.HasInheritedDefaultArg = Record[Idx++];
  if (Record[Idx++]) // hasUninstantiatedDefaultArg.
    PD->setUninstantiatedDefaultArg(Reader.ReadExpr(F));
}

void ASTDeclReader::VisitFileScopeAsmDecl(FileScopeAsmDecl *AD) {
  VisitDecl(AD);
  AD->setAsmString(cast<StringLiteral>(Reader.ReadExpr(F)));
  AD->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitBlockDecl(BlockDecl *BD) {
  VisitDecl(BD);
  BD->setBody(cast_or_null<CompoundStmt>(Reader.ReadStmt(F)));
  BD->setSignatureAsWritten(GetTypeSourceInfo(Record, Idx));
  unsigned NumParams = Record[Idx++];
  SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(ReadDeclAs<ParmVarDecl>(Record, Idx));
  BD->setParams(Params);

  BD->setIsVariadic(Record[Idx++]);
  BD->setBlockMissingReturnType(Record[Idx++]);
  BD->setIsConversionFromLambda(Record[Idx++]);

  bool capturesCXXThis = Record[Idx++];
  unsigned numCaptures = Record[Idx++];
  SmallVector<BlockDecl::Capture, 16> captures;
  captures.reserve(numCaptures);
  for (unsigned i = 0; i != numCaptures; ++i) {
    VarDecl *decl = ReadDeclAs<VarDecl>(Record, Idx);
    unsigned flags = Record[Idx++];
    bool byRef = (flags & 1);
    bool nested = (flags & 2);
    Expr *copyExpr = ((flags & 4) ? Reader.ReadExpr(F) : 0);

    captures.push_back(BlockDecl::Capture(decl, byRef, nested, copyExpr));
  }
  BD->setCaptures(Reader.getContext(), captures.begin(),
                  captures.end(), capturesCXXThis);
}

void ASTDeclReader::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  VisitDecl(D);
  D->setLanguage((LinkageSpecDecl::LanguageIDs)Record[Idx++]);
  D->setExternLoc(ReadSourceLocation(Record, Idx));
  D->setRBraceLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitLabelDecl(LabelDecl *D) {
  VisitNamedDecl(D);
  D->setLocStart(ReadSourceLocation(Record, Idx));
}


void ASTDeclReader::VisitNamespaceDecl(NamespaceDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarable(D);
  VisitNamedDecl(D);
  D->setInline(Record[Idx++]);
  D->LocStart = ReadSourceLocation(Record, Idx);
  D->RBraceLoc = ReadSourceLocation(Record, Idx);
  mergeRedeclarable(D, Redecl);

  if (Redecl.getFirstID() == ThisDeclID) {
    // Each module has its own anonymous namespace, which is disjoint from
    // any other module's anonymous namespaces, so don't attach the anonymous
    // namespace at all.
    NamespaceDecl *Anon = ReadDeclAs<NamespaceDecl>(Record, Idx);
    if (F.Kind != MK_Module)
      D->setAnonymousNamespace(Anon);
  } else {
    // Link this namespace back to the first declaration, which has already
    // been deserialized.
    D->AnonOrFirstNamespaceAndInline.setPointer(D->getFirstDeclaration());
  }
}

void ASTDeclReader::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  VisitNamedDecl(D);
  D->NamespaceLoc = ReadSourceLocation(Record, Idx);
  D->IdentLoc = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  D->Namespace = ReadDeclAs<NamedDecl>(Record, Idx);
}

void ASTDeclReader::VisitUsingDecl(UsingDecl *D) {
  VisitNamedDecl(D);
  D->setUsingLocation(ReadSourceLocation(Record, Idx));
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  ReadDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record, Idx);
  D->FirstUsingShadow.setPointer(ReadDeclAs<UsingShadowDecl>(Record, Idx));
  D->setTypeName(Record[Idx++]);
  if (NamedDecl *Pattern = ReadDeclAs<NamedDecl>(Record, Idx))
    Reader.getContext().setInstantiatedFromUsingDecl(D, Pattern);
}

void ASTDeclReader::VisitUsingShadowDecl(UsingShadowDecl *D) {
  VisitNamedDecl(D);
  D->setTargetDecl(ReadDeclAs<NamedDecl>(Record, Idx));
  D->UsingOrNextShadow = ReadDeclAs<NamedDecl>(Record, Idx);
  UsingShadowDecl *Pattern = ReadDeclAs<UsingShadowDecl>(Record, Idx);
  if (Pattern)
    Reader.getContext().setInstantiatedFromUsingShadowDecl(D, Pattern);
}

void ASTDeclReader::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  VisitNamedDecl(D);
  D->UsingLoc = ReadSourceLocation(Record, Idx);
  D->NamespaceLoc = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  D->NominatedNamespace = ReadDeclAs<NamedDecl>(Record, Idx);
  D->CommonAncestor = ReadDeclAs<DeclContext>(Record, Idx);
}

void ASTDeclReader::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  VisitValueDecl(D);
  D->setUsingLoc(ReadSourceLocation(Record, Idx));
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  ReadDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record, Idx);
}

void ASTDeclReader::VisitUnresolvedUsingTypenameDecl(
                                               UnresolvedUsingTypenameDecl *D) {
  VisitTypeDecl(D);
  D->TypenameLocation = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
}

void ASTDeclReader::ReadCXXDefinitionData(
                                   struct CXXRecordDecl::DefinitionData &Data,
                                   const RecordData &Record, unsigned &Idx) {
  // Note: the caller has deserialized the IsLambda bit already.
  Data.UserDeclaredConstructor = Record[Idx++];
  Data.UserDeclaredCopyConstructor = Record[Idx++];
  Data.UserDeclaredMoveConstructor = Record[Idx++];
  Data.UserDeclaredCopyAssignment = Record[Idx++];
  Data.UserDeclaredMoveAssignment = Record[Idx++];
  Data.UserDeclaredDestructor = Record[Idx++];
  Data.Aggregate = Record[Idx++];
  Data.PlainOldData = Record[Idx++];
  Data.Empty = Record[Idx++];
  Data.Polymorphic = Record[Idx++];
  Data.Abstract = Record[Idx++];
  Data.IsStandardLayout = Record[Idx++];
  Data.HasNoNonEmptyBases = Record[Idx++];
  Data.HasPrivateFields = Record[Idx++];
  Data.HasProtectedFields = Record[Idx++];
  Data.HasPublicFields = Record[Idx++];
  Data.HasMutableFields = Record[Idx++];
  Data.HasOnlyCMembers = Record[Idx++];
  Data.HasInClassInitializer = Record[Idx++];
  Data.HasTrivialDefaultConstructor = Record[Idx++];
  Data.HasConstexprNonCopyMoveConstructor = Record[Idx++];
  Data.DefaultedDefaultConstructorIsConstexpr = Record[Idx++];
  Data.HasConstexprDefaultConstructor = Record[Idx++];
  Data.HasTrivialCopyConstructor = Record[Idx++];
  Data.HasTrivialMoveConstructor = Record[Idx++];
  Data.HasTrivialCopyAssignment = Record[Idx++];
  Data.HasTrivialMoveAssignment = Record[Idx++];
  Data.HasTrivialDestructor = Record[Idx++];
  Data.HasIrrelevantDestructor = Record[Idx++];
  Data.HasNonLiteralTypeFieldsOrBases = Record[Idx++];
  Data.ComputedVisibleConversions = Record[Idx++];
  Data.UserProvidedDefaultConstructor = Record[Idx++];
  Data.DeclaredDefaultConstructor = Record[Idx++];
  Data.DeclaredCopyConstructor = Record[Idx++];
  Data.DeclaredMoveConstructor = Record[Idx++];
  Data.DeclaredCopyAssignment = Record[Idx++];
  Data.DeclaredMoveAssignment = Record[Idx++];
  Data.DeclaredDestructor = Record[Idx++];
  Data.FailedImplicitMoveConstructor = Record[Idx++];
  Data.FailedImplicitMoveAssignment = Record[Idx++];

  Data.NumBases = Record[Idx++];
  if (Data.NumBases)
    Data.Bases = Reader.readCXXBaseSpecifiers(F, Record, Idx);
  Data.NumVBases = Record[Idx++];
  if (Data.NumVBases)
    Data.VBases = Reader.readCXXBaseSpecifiers(F, Record, Idx);
  
  Reader.ReadUnresolvedSet(F, Data.Conversions, Record, Idx);
  Reader.ReadUnresolvedSet(F, Data.VisibleConversions, Record, Idx);
  assert(Data.Definition && "Data.Definition should be already set!");
  Data.FirstFriend = ReadDeclAs<FriendDecl>(Record, Idx);
  
  if (Data.IsLambda) {
    typedef LambdaExpr::Capture Capture;
    CXXRecordDecl::LambdaDefinitionData &Lambda
      = static_cast<CXXRecordDecl::LambdaDefinitionData &>(Data);
    Lambda.Dependent = Record[Idx++];
    Lambda.NumCaptures = Record[Idx++];
    Lambda.NumExplicitCaptures = Record[Idx++];
    Lambda.ManglingNumber = Record[Idx++];
    Lambda.ContextDecl = ReadDecl(Record, Idx);
    Lambda.Captures 
      = (Capture*)Reader.Context.Allocate(sizeof(Capture)*Lambda.NumCaptures);
    Capture *ToCapture = Lambda.Captures;
    Lambda.MethodTyInfo = GetTypeSourceInfo(Record, Idx);
    for (unsigned I = 0, N = Lambda.NumCaptures; I != N; ++I) {
      SourceLocation Loc = ReadSourceLocation(Record, Idx);
      bool IsImplicit = Record[Idx++];
      LambdaCaptureKind Kind = static_cast<LambdaCaptureKind>(Record[Idx++]);
      VarDecl *Var = ReadDeclAs<VarDecl>(Record, Idx);
      SourceLocation EllipsisLoc = ReadSourceLocation(Record, Idx);
      *ToCapture++ = Capture(Loc, IsImplicit, Kind, Var, EllipsisLoc);
    }
  }
}

void ASTDeclReader::VisitCXXRecordDecl(CXXRecordDecl *D) {
  VisitRecordDecl(D);

  ASTContext &C = Reader.getContext();
  if (Record[Idx++]) {
    // Determine whether this is a lambda closure type, so that we can
    // allocate the appropriate DefinitionData structure.
    bool IsLambda = Record[Idx++];
    if (IsLambda)
      D->DefinitionData = new (C) CXXRecordDecl::LambdaDefinitionData(D, 0,
                                                                      false);
    else
      D->DefinitionData = new (C) struct CXXRecordDecl::DefinitionData(D);
    
    // Propagate the DefinitionData pointer to the canonical declaration, so
    // that all other deserialized declarations will see it.
    // FIXME: Complain if there already is a DefinitionData!
    D->getCanonicalDecl()->DefinitionData = D->DefinitionData;
    
    ReadCXXDefinitionData(*D->DefinitionData, Record, Idx);
    
    // Note that we have deserialized a definition. Any declarations 
    // deserialized before this one will be be given the DefinitionData pointer
    // at the end.
    Reader.PendingDefinitions.insert(D);
  } else {
    // Propagate DefinitionData pointer from the canonical declaration.
    D->DefinitionData = D->getCanonicalDecl()->DefinitionData;    
  }

  enum CXXRecKind {
    CXXRecNotTemplate = 0, CXXRecTemplate, CXXRecMemberSpecialization
  };
  switch ((CXXRecKind)Record[Idx++]) {
  case CXXRecNotTemplate:
    break;
  case CXXRecTemplate:
    D->TemplateOrInstantiation = ReadDeclAs<ClassTemplateDecl>(Record, Idx);
    break;
  case CXXRecMemberSpecialization: {
    CXXRecordDecl *RD = ReadDeclAs<CXXRecordDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    MemberSpecializationInfo *MSI = new (C) MemberSpecializationInfo(RD, TSK);
    MSI->setPointOfInstantiation(POI);
    D->TemplateOrInstantiation = MSI;
    break;
  }
  }

  // Load the key function to avoid deserializing every method so we can
  // compute it.
  if (D->IsCompleteDefinition) {
    if (CXXMethodDecl *Key = ReadDeclAs<CXXMethodDecl>(Record, Idx))
      C.KeyFunctions[D] = Key;
  }
}

void ASTDeclReader::VisitCXXMethodDecl(CXXMethodDecl *D) {
  VisitFunctionDecl(D);
  unsigned NumOverridenMethods = Record[Idx++];
  while (NumOverridenMethods--) {
    // Avoid invariant checking of CXXMethodDecl::addOverriddenMethod,
    // MD may be initializing.
    if (CXXMethodDecl *MD = ReadDeclAs<CXXMethodDecl>(Record, Idx))
      Reader.getContext().addOverriddenMethod(D, MD);
  }
}

void ASTDeclReader::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  VisitCXXMethodDecl(D);
  
  D->IsExplicitSpecified = Record[Idx++];
  D->ImplicitlyDefined = Record[Idx++];
  llvm::tie(D->CtorInitializers, D->NumCtorInitializers)
      = Reader.ReadCXXCtorInitializers(F, Record, Idx);
}

void ASTDeclReader::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  VisitCXXMethodDecl(D);

  D->ImplicitlyDefined = Record[Idx++];
  D->OperatorDelete = ReadDeclAs<FunctionDecl>(Record, Idx);
}

void ASTDeclReader::VisitCXXConversionDecl(CXXConversionDecl *D) {
  VisitCXXMethodDecl(D);
  D->IsExplicitSpecified = Record[Idx++];
}

void ASTDeclReader::VisitImportDecl(ImportDecl *D) {
  VisitDecl(D);
  D->ImportedAndComplete.setPointer(readModule(Record, Idx));
  D->ImportedAndComplete.setInt(Record[Idx++]);
  SourceLocation *StoredLocs = reinterpret_cast<SourceLocation *>(D + 1);
  for (unsigned I = 0, N = Record.back(); I != N; ++I)
    StoredLocs[I] = ReadSourceLocation(Record, Idx);
  ++Idx; // The number of stored source locations.
}

void ASTDeclReader::VisitAccessSpecDecl(AccessSpecDecl *D) {
  VisitDecl(D);
  D->setColonLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitFriendDecl(FriendDecl *D) {
  VisitDecl(D);
  if (Record[Idx++])
    D->Friend = GetTypeSourceInfo(Record, Idx);
  else
    D->Friend = ReadDeclAs<NamedDecl>(Record, Idx);
  D->NextFriend = Record[Idx++];
  D->UnsupportedFriend = (Record[Idx++] != 0);
  D->FriendLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitFriendTemplateDecl(FriendTemplateDecl *D) {
  VisitDecl(D);
  unsigned NumParams = Record[Idx++];
  D->NumParams = NumParams;
  D->Params = new TemplateParameterList*[NumParams];
  for (unsigned i = 0; i != NumParams; ++i)
    D->Params[i] = Reader.ReadTemplateParameterList(F, Record, Idx);
  if (Record[Idx++]) // HasFriendDecl
    D->Friend = ReadDeclAs<NamedDecl>(Record, Idx);
  else
    D->Friend = GetTypeSourceInfo(Record, Idx);
  D->FriendLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitTemplateDecl(TemplateDecl *D) {
  VisitNamedDecl(D);

  NamedDecl *TemplatedDecl = ReadDeclAs<NamedDecl>(Record, Idx);
  TemplateParameterList* TemplateParams
      = Reader.ReadTemplateParameterList(F, Record, Idx); 
  D->init(TemplatedDecl, TemplateParams);
}

ASTDeclReader::RedeclarableResult 
ASTDeclReader::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarable(D);

  // Make sure we've allocated the Common pointer first. We do this before
  // VisitTemplateDecl so that getCommonPtr() can be used during initialization.
  RedeclarableTemplateDecl *CanonD = D->getCanonicalDecl();
  if (!CanonD->Common) {
    CanonD->Common = CanonD->newCommon(Reader.getContext());
    Reader.PendingDefinitions.insert(CanonD);
  }
  D->Common = CanonD->Common;

  // If this is the first declaration of the template, fill in the information
  // for the 'common' pointer.
  if (ThisDeclID == Redecl.getFirstID()) {
    if (RedeclarableTemplateDecl *RTD
          = ReadDeclAs<RedeclarableTemplateDecl>(Record, Idx)) {
      assert(RTD->getKind() == D->getKind() &&
             "InstantiatedFromMemberTemplate kind mismatch");
      D->setInstantiatedFromMemberTemplate(RTD);
      if (Record[Idx++])
        D->setMemberSpecialization();
    }
  }

  VisitTemplateDecl(D);
  D->IdentifierNamespace = Record[Idx++];

  mergeRedeclarable(D, Redecl);

  return Redecl;
}

void ASTDeclReader::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarableTemplateDecl(D);

  if (ThisDeclID == Redecl.getFirstID()) {
    // This ClassTemplateDecl owns a CommonPtr; read it to keep track of all of
    // the specializations.
    SmallVector<serialization::DeclID, 2> SpecIDs;
    SpecIDs.push_back(0);
    
    // Specializations.
    unsigned Size = Record[Idx++];
    SpecIDs[0] += Size;
    for (unsigned I = 0; I != Size; ++I)
      SpecIDs.push_back(ReadDeclID(Record, Idx));

    // Partial specializations.
    Size = Record[Idx++];
    SpecIDs[0] += Size;
    for (unsigned I = 0; I != Size; ++I)
      SpecIDs.push_back(ReadDeclID(Record, Idx));

    ClassTemplateDecl::Common *CommonPtr = D->getCommonPtr();
    if (SpecIDs[0]) {
      typedef serialization::DeclID DeclID;
      
      // FIXME: Append specializations!
      CommonPtr->LazySpecializations
        = new (Reader.getContext()) DeclID [SpecIDs.size()];
      memcpy(CommonPtr->LazySpecializations, SpecIDs.data(), 
             SpecIDs.size() * sizeof(DeclID));
    }
    
    CommonPtr->InjectedClassNameType = Reader.readType(F, Record, Idx);
  }
}

void ASTDeclReader::VisitClassTemplateSpecializationDecl(
                                           ClassTemplateSpecializationDecl *D) {
  VisitCXXRecordDecl(D);
  
  ASTContext &C = Reader.getContext();
  if (Decl *InstD = ReadDecl(Record, Idx)) {
    if (ClassTemplateDecl *CTD = dyn_cast<ClassTemplateDecl>(InstD)) {
      D->SpecializedTemplate = CTD;
    } else {
      SmallVector<TemplateArgument, 8> TemplArgs;
      Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
      TemplateArgumentList *ArgList
        = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), 
                                           TemplArgs.size());
      ClassTemplateSpecializationDecl::SpecializedPartialSpecialization *PS
          = new (C) ClassTemplateSpecializationDecl::
                                             SpecializedPartialSpecialization();
      PS->PartialSpecialization
          = cast<ClassTemplatePartialSpecializationDecl>(InstD);
      PS->TemplateArgs = ArgList;
      D->SpecializedTemplate = PS;
    }
  }

  // Explicit info.
  if (TypeSourceInfo *TyInfo = GetTypeSourceInfo(Record, Idx)) {
    ClassTemplateSpecializationDecl::ExplicitSpecializationInfo *ExplicitInfo
        = new (C) ClassTemplateSpecializationDecl::ExplicitSpecializationInfo;
    ExplicitInfo->TypeAsWritten = TyInfo;
    ExplicitInfo->ExternLoc = ReadSourceLocation(Record, Idx);
    ExplicitInfo->TemplateKeywordLoc = ReadSourceLocation(Record, Idx);
    D->ExplicitInfo = ExplicitInfo;
  }

  SmallVector<TemplateArgument, 8> TemplArgs;
  Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
  D->TemplateArgs = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), 
                                                     TemplArgs.size());
  D->PointOfInstantiation = ReadSourceLocation(Record, Idx);
  D->SpecializationKind = (TemplateSpecializationKind)Record[Idx++];

  bool writtenAsCanonicalDecl = Record[Idx++];
  if (writtenAsCanonicalDecl) {
    ClassTemplateDecl *CanonPattern = ReadDeclAs<ClassTemplateDecl>(Record,Idx);
    if (D->isCanonicalDecl()) { // It's kept in the folding set.
      if (ClassTemplatePartialSpecializationDecl *Partial
                        = dyn_cast<ClassTemplatePartialSpecializationDecl>(D)) {
  CanonPattern->getCommonPtr()->PartialSpecializations.GetOrInsertNode(Partial);
      } else {
        CanonPattern->getCommonPtr()->Specializations.GetOrInsertNode(D);
      }
    }
  }
}

void ASTDeclReader::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);

  ASTContext &C = Reader.getContext();
  D->TemplateParams = Reader.ReadTemplateParameterList(F, Record, Idx);

  unsigned NumArgs = Record[Idx++];
  if (NumArgs) {
    D->NumArgsAsWritten = NumArgs;
    D->ArgsAsWritten = new (C) TemplateArgumentLoc[NumArgs];
    for (unsigned i=0; i != NumArgs; ++i)
      D->ArgsAsWritten[i] = Reader.ReadTemplateArgumentLoc(F, Record, Idx);
  }

  D->SequenceNumber = Record[Idx++];

  // These are read/set from/to the first declaration.
  if (D->getPreviousDecl() == 0) {
    D->InstantiatedFromMember.setPointer(
      ReadDeclAs<ClassTemplatePartialSpecializationDecl>(Record, Idx));
    D->InstantiatedFromMember.setInt(Record[Idx++]);
  }
}

void ASTDeclReader::VisitClassScopeFunctionSpecializationDecl(
                                    ClassScopeFunctionSpecializationDecl *D) {
  VisitDecl(D);
  D->Specialization = ReadDeclAs<CXXMethodDecl>(Record, Idx);
}

void ASTDeclReader::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarableTemplateDecl(D);

  if (ThisDeclID == Redecl.getFirstID()) {
    // This FunctionTemplateDecl owns a CommonPtr; read it.

    // Read the function specialization declarations.
    // FunctionTemplateDecl's FunctionTemplateSpecializationInfos are filled
    // when reading the specialized FunctionDecl.
    unsigned NumSpecs = Record[Idx++];
    while (NumSpecs--)
      (void)ReadDecl(Record, Idx);
  }
}

void ASTDeclReader::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  VisitTypeDecl(D);

  D->setDeclaredWithTypename(Record[Idx++]);

  bool Inherited = Record[Idx++];
  TypeSourceInfo *DefArg = GetTypeSourceInfo(Record, Idx);
  D->setDefaultArgument(DefArg, Inherited);
}

void ASTDeclReader::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  VisitDeclaratorDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  if (D->isExpandedParameterPack()) {
    void **Data = reinterpret_cast<void **>(D + 1);
    for (unsigned I = 0, N = D->getNumExpansionTypes(); I != N; ++I) {
      Data[2*I] = Reader.readType(F, Record, Idx).getAsOpaquePtr();
      Data[2*I + 1] = GetTypeSourceInfo(Record, Idx);
    }
  } else {
    // Rest of NonTypeTemplateParmDecl.
    D->ParameterPack = Record[Idx++];
    if (Record[Idx++]) {
      Expr *DefArg = Reader.ReadExpr(F);
      bool Inherited = Record[Idx++];
      D->setDefaultArgument(DefArg, Inherited);
   }
  }
}

void ASTDeclReader::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  VisitTemplateDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  if (D->isExpandedParameterPack()) {
    void **Data = reinterpret_cast<void **>(D + 1);
    for (unsigned I = 0, N = D->getNumExpansionTemplateParameters();
         I != N; ++I)
      Data[I] = Reader.ReadTemplateParameterList(F, Record, Idx);
  } else {
    // Rest of TemplateTemplateParmDecl.
    TemplateArgumentLoc Arg = Reader.ReadTemplateArgumentLoc(F, Record, Idx);
    bool IsInherited = Record[Idx++];
    D->setDefaultArgument(Arg, IsInherited);
    D->ParameterPack = Record[Idx++];
  }
}

void ASTDeclReader::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);
}

void ASTDeclReader::VisitStaticAssertDecl(StaticAssertDecl *D) {
  VisitDecl(D);
  D->AssertExprAndFailed.setPointer(Reader.ReadExpr(F));
  D->AssertExprAndFailed.setInt(Record[Idx++]);
  D->Message = cast<StringLiteral>(Reader.ReadExpr(F));
  D->RParenLoc = ReadSourceLocation(Record, Idx);
}

std::pair<uint64_t, uint64_t>
ASTDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = Record[Idx++];
  uint64_t VisibleOffset = Record[Idx++];
  return std::make_pair(LexicalOffset, VisibleOffset);
}

template <typename T>
ASTDeclReader::RedeclarableResult 
ASTDeclReader::VisitRedeclarable(Redeclarable<T> *D) {
  DeclID FirstDeclID = ReadDeclID(Record, Idx);
  
  // 0 indicates that this declaration was the only declaration of its entity,
  // and is used for space optimization.
  if (FirstDeclID == 0)
    FirstDeclID = ThisDeclID;
  
  T *FirstDecl = cast_or_null<T>(Reader.GetDecl(FirstDeclID));
  if (FirstDecl != D) {
    // We delay loading of the redeclaration chain to avoid deeply nested calls.
    // We temporarily set the first (canonical) declaration as the previous one
    // which is the one that matters and mark the real previous DeclID to be
    // loaded & attached later on.
    D->RedeclLink = Redeclarable<T>::PreviousDeclLink(FirstDecl);
  }    
  
  // Note that this declaration has been deserialized.
  Reader.RedeclsDeserialized.insert(static_cast<T *>(D));
                             
  // The result structure takes care to note that we need to load the 
  // other declaration chains for this ID.
  return RedeclarableResult(Reader, FirstDeclID);
}

/// \brief Attempts to merge the given declaration (D) with another declaration
/// of the same entity.
template<typename T>
void ASTDeclReader::mergeRedeclarable(Redeclarable<T> *D, 
                                      RedeclarableResult &Redecl) {
  // If modules are not available, there is no reason to perform this merge.
  if (!Reader.getContext().getLangOpts().Modules)
    return;
  
  if (FindExistingResult ExistingRes = findExisting(static_cast<T*>(D))) {
    if (T *Existing = ExistingRes) {
      T *ExistingCanon = Existing->getCanonicalDecl();
      T *DCanon = static_cast<T*>(D)->getCanonicalDecl();
      if (ExistingCanon != DCanon) {
        // Have our redeclaration link point back at the canonical declaration
        // of the existing declaration, so that this declaration has the 
        // appropriate canonical declaration.
        D->RedeclLink = Redeclarable<T>::PreviousDeclLink(ExistingCanon);
        
        // When we merge a namespace, update its pointer to the first namespace.
        if (NamespaceDecl *Namespace
              = dyn_cast<NamespaceDecl>(static_cast<T*>(D))) {
          Namespace->AnonOrFirstNamespaceAndInline.setPointer(
            static_cast<NamespaceDecl *>(static_cast<void*>(ExistingCanon)));
        }
        
        // Don't introduce DCanon into the set of pending declaration chains.
        Redecl.suppress();
        
        // Introduce ExistingCanon into the set of pending declaration chains,
        // if in fact it came from a module file.
        if (ExistingCanon->isFromASTFile()) {
          GlobalDeclID ExistingCanonID = ExistingCanon->getGlobalID();
          assert(ExistingCanonID && "Unrecorded canonical declaration ID?");
          if (Reader.PendingDeclChainsKnown.insert(ExistingCanonID))
            Reader.PendingDeclChains.push_back(ExistingCanonID);
        }
        
        // If this declaration was the canonical declaration, make a note of 
        // that. We accept the linear algorithm here because the number of 
        // unique canonical declarations of an entity should always be tiny.
        if (DCanon == static_cast<T*>(D)) {
          SmallVectorImpl<DeclID> &Merged = Reader.MergedDecls[ExistingCanon];
          if (std::find(Merged.begin(), Merged.end(), Redecl.getFirstID())
                == Merged.end())
            Merged.push_back(Redecl.getFirstID());
          
          // If ExistingCanon did not come from a module file, introduce the
          // first declaration that *does* come from a module file to the 
          // set of pending declaration chains, so that we merge this 
          // declaration.
          if (!ExistingCanon->isFromASTFile() &&
              Reader.PendingDeclChainsKnown.insert(Redecl.getFirstID()))
            Reader.PendingDeclChains.push_back(Merged[0]);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Attribute Reading
//===----------------------------------------------------------------------===//

/// \brief Reads attributes from the current stream position.
void ASTReader::ReadAttributes(ModuleFile &F, AttrVec &Attrs,
                               const RecordData &Record, unsigned &Idx) {
  for (unsigned i = 0, e = Record[Idx++]; i != e; ++i) {
    Attr *New = 0;
    attr::Kind Kind = (attr::Kind)Record[Idx++];
    SourceRange Range = ReadSourceRange(F, Record, Idx);

#include "clang/Serialization/AttrPCHRead.inc"

    assert(New && "Unable to decode attribute?");
    Attrs.push_back(New);
  }
}

//===----------------------------------------------------------------------===//
// ASTReader Implementation
//===----------------------------------------------------------------------===//

/// \brief Note that we have loaded the declaration with the given
/// Index.
///
/// This routine notes that this declaration has already been loaded,
/// so that future GetDecl calls will return this declaration rather
/// than trying to load a new declaration.
inline void ASTReader::LoadedDecl(unsigned Index, Decl *D) {
  assert(!DeclsLoaded[Index] && "Decl loaded twice?");
  DeclsLoaded[Index] = D;
}


/// \brief Determine whether the consumer will be interested in seeing
/// this declaration (via HandleTopLevelDecl).
///
/// This routine should return true for anything that might affect
/// code generation, e.g., inline function definitions, Objective-C
/// declarations with metadata, etc.
static bool isConsumerInterestedIn(Decl *D, bool HasBody) {
  // An ObjCMethodDecl is never considered as "interesting" because its
  // implementation container always is.

  if (isa<FileScopeAsmDecl>(D) || 
      isa<ObjCProtocolDecl>(D) || 
      isa<ObjCImplDecl>(D))
    return true;
  if (VarDecl *Var = dyn_cast<VarDecl>(D))
    return Var->isFileVarDecl() &&
           Var->isThisDeclarationADefinition() == VarDecl::Definition;
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D))
    return Func->doesThisDeclarationHaveABody() || HasBody;
  
  return false;
}

/// \brief Get the correct cursor and offset for loading a declaration.
ASTReader::RecordLocation
ASTReader::DeclCursorForID(DeclID ID, unsigned &RawLocation) {
  // See if there's an override.
  DeclReplacementMap::iterator It = ReplacedDecls.find(ID);
  if (It != ReplacedDecls.end()) {
    RawLocation = It->second.RawLoc;
    return RecordLocation(It->second.Mod, It->second.Offset);
  }

  GlobalDeclMapType::iterator I = GlobalDeclMap.find(ID);
  assert(I != GlobalDeclMap.end() && "Corrupted global declaration map");
  ModuleFile *M = I->second;
  const DeclOffset &
    DOffs =  M->DeclOffsets[ID - M->BaseDeclID - NUM_PREDEF_DECL_IDS];
  RawLocation = DOffs.Loc;
  return RecordLocation(M, DOffs.BitOffset);
}

ASTReader::RecordLocation ASTReader::getLocalBitOffset(uint64_t GlobalOffset) {
  ContinuousRangeMap<uint64_t, ModuleFile*, 4>::iterator I
    = GlobalBitOffsetsMap.find(GlobalOffset);

  assert(I != GlobalBitOffsetsMap.end() && "Corrupted global bit offsets map");
  return RecordLocation(I->second, GlobalOffset - I->second->GlobalBitOffset);
}

uint64_t ASTReader::getGlobalBitOffset(ModuleFile &M, uint32_t LocalOffset) {
  return LocalOffset + M.GlobalBitOffset;
}

/// \brief Determine whether the two declarations refer to the same entity.
static bool isSameEntity(NamedDecl *X, NamedDecl *Y) {
  assert(X->getDeclName() == Y->getDeclName() && "Declaration name mismatch!");
  
  if (X == Y)
    return true;
  
  // Must be in the same context.
  if (!X->getDeclContext()->getRedeclContext()->Equals(
         Y->getDeclContext()->getRedeclContext()))
    return false;

  // Two typedefs refer to the same entity if they have the same underlying
  // type.
  if (TypedefNameDecl *TypedefX = dyn_cast<TypedefNameDecl>(X))
    if (TypedefNameDecl *TypedefY = dyn_cast<TypedefNameDecl>(Y))
      return X->getASTContext().hasSameType(TypedefX->getUnderlyingType(),
                                            TypedefY->getUnderlyingType());
  
  // Must have the same kind.
  if (X->getKind() != Y->getKind())
    return false;
    
  // Objective-C classes and protocols with the same name always match.
  if (isa<ObjCInterfaceDecl>(X) || isa<ObjCProtocolDecl>(X))
    return true;
  
  // Compatible tags match.
  if (TagDecl *TagX = dyn_cast<TagDecl>(X)) {
    TagDecl *TagY = cast<TagDecl>(Y);
    return (TagX->getTagKind() == TagY->getTagKind()) ||
      ((TagX->getTagKind() == TTK_Struct || TagX->getTagKind() == TTK_Class ||
        TagX->getTagKind() == TTK_Interface) &&
       (TagY->getTagKind() == TTK_Struct || TagY->getTagKind() == TTK_Class ||
        TagY->getTagKind() == TTK_Interface));
  }
  
  // Functions with the same type and linkage match.
  // FIXME: This needs to cope with function templates, merging of 
  //prototyped/non-prototyped functions, etc.
  if (FunctionDecl *FuncX = dyn_cast<FunctionDecl>(X)) {
    FunctionDecl *FuncY = cast<FunctionDecl>(Y);
    return (FuncX->getLinkage() == FuncY->getLinkage()) &&
      FuncX->getASTContext().hasSameType(FuncX->getType(), FuncY->getType());
  }

  // Variables with the same type and linkage match.
  if (VarDecl *VarX = dyn_cast<VarDecl>(X)) {
    VarDecl *VarY = cast<VarDecl>(Y);
    return (VarX->getLinkage() == VarY->getLinkage()) &&
      VarX->getASTContext().hasSameType(VarX->getType(), VarY->getType());
  }
  
  // Namespaces with the same name and inlinedness match.
  if (NamespaceDecl *NamespaceX = dyn_cast<NamespaceDecl>(X)) {
    NamespaceDecl *NamespaceY = cast<NamespaceDecl>(Y);
    return NamespaceX->isInline() == NamespaceY->isInline();
  }

  // Identical template names and kinds match.
  if (isa<TemplateDecl>(X))
    return true;

  // FIXME: Many other cases to implement.
  return false;
}

ASTDeclReader::FindExistingResult::~FindExistingResult() {
  if (!AddResult || Existing)
    return;
  
  if (New->getDeclContext()->getRedeclContext()->isTranslationUnit()
      && Reader.SemaObj) {
    Reader.SemaObj->IdResolver.tryAddTopLevelDecl(New, New->getDeclName());
  } else {
    DeclContext *DC = New->getLexicalDeclContext();
    if (DC->isNamespace())
      DC->addDecl(New);
  }
}

ASTDeclReader::FindExistingResult ASTDeclReader::findExisting(NamedDecl *D) {
  DeclarationName Name = D->getDeclName();
  if (!Name) {
    // Don't bother trying to find unnamed declarations.
    FindExistingResult Result(Reader, D, /*Existing=*/0);
    Result.suppress();
    return Result;
  }
  
  DeclContext *DC = D->getDeclContext()->getRedeclContext();
  if (!DC->isFileContext())
    return FindExistingResult(Reader);
  
  if (DC->isTranslationUnit() && Reader.SemaObj) {
    IdentifierResolver &IdResolver = Reader.SemaObj->IdResolver;
    for (IdentifierResolver::iterator I = IdResolver.begin(Name), 
                                   IEnd = IdResolver.end();
         I != IEnd; ++I) {
      if (isSameEntity(*I, D))
        return FindExistingResult(Reader, D, *I);
    }
  }

  if (DC->isNamespace()) {
    for (DeclContext::lookup_result R = DC->lookup(Name);
         R.first != R.second; ++R.first) {
      if (isSameEntity(*R.first, D))
        return FindExistingResult(Reader, D, *R.first);
    }
  }
  
  return FindExistingResult(Reader, D, /*Existing=*/0);
}

void ASTDeclReader::attachPreviousDecl(Decl *D, Decl *previous) {
  assert(D && previous);
  if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    TD->RedeclLink.setNext(cast<TagDecl>(previous));
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    FD->RedeclLink.setNext(cast<FunctionDecl>(previous));
  } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    VD->RedeclLink.setNext(cast<VarDecl>(previous));
  } else if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    TD->RedeclLink.setNext(cast<TypedefNameDecl>(previous));
  } else if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    ID->RedeclLink.setNext(cast<ObjCInterfaceDecl>(previous));
  } else if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D)) {
    PD->RedeclLink.setNext(cast<ObjCProtocolDecl>(previous));
  } else if (NamespaceDecl *ND = dyn_cast<NamespaceDecl>(D)) {
    ND->RedeclLink.setNext(cast<NamespaceDecl>(previous));
  } else {
    RedeclarableTemplateDecl *TD = cast<RedeclarableTemplateDecl>(D);
    TD->RedeclLink.setNext(cast<RedeclarableTemplateDecl>(previous));
  }
}

void ASTDeclReader::attachLatestDecl(Decl *D, Decl *Latest) {
  assert(D && Latest);
  if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    TD->RedeclLink
      = Redeclarable<TagDecl>::LatestDeclLink(cast<TagDecl>(Latest));
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    FD->RedeclLink 
      = Redeclarable<FunctionDecl>::LatestDeclLink(cast<FunctionDecl>(Latest));
  } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    VD->RedeclLink
      = Redeclarable<VarDecl>::LatestDeclLink(cast<VarDecl>(Latest));
  } else if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    TD->RedeclLink
      = Redeclarable<TypedefNameDecl>::LatestDeclLink(
                                                cast<TypedefNameDecl>(Latest));
  } else if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    ID->RedeclLink
      = Redeclarable<ObjCInterfaceDecl>::LatestDeclLink(
                                              cast<ObjCInterfaceDecl>(Latest));
  } else if (ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D)) {
    PD->RedeclLink
      = Redeclarable<ObjCProtocolDecl>::LatestDeclLink(
                                                cast<ObjCProtocolDecl>(Latest));
  } else if (NamespaceDecl *ND = dyn_cast<NamespaceDecl>(D)) {
    ND->RedeclLink
      = Redeclarable<NamespaceDecl>::LatestDeclLink(
                                                   cast<NamespaceDecl>(Latest));
  } else {
    RedeclarableTemplateDecl *TD = cast<RedeclarableTemplateDecl>(D);
    TD->RedeclLink
      = Redeclarable<RedeclarableTemplateDecl>::LatestDeclLink(
                                        cast<RedeclarableTemplateDecl>(Latest));
  }
}

ASTReader::MergedDeclsMap::iterator
ASTReader::combineStoredMergedDecls(Decl *Canon, GlobalDeclID CanonID) {
  // If we don't have any stored merged declarations, just look in the
  // merged declarations set.
  StoredMergedDeclsMap::iterator StoredPos = StoredMergedDecls.find(CanonID);
  if (StoredPos == StoredMergedDecls.end())
    return MergedDecls.find(Canon);

  // Append the stored merged declarations to the merged declarations set.
  MergedDeclsMap::iterator Pos = MergedDecls.find(Canon);
  if (Pos == MergedDecls.end())
    Pos = MergedDecls.insert(std::make_pair(Canon, 
                                            SmallVector<DeclID, 2>())).first;
  Pos->second.append(StoredPos->second.begin(), StoredPos->second.end());
  StoredMergedDecls.erase(StoredPos);
  
  // Sort and uniquify the set of merged declarations.
  llvm::array_pod_sort(Pos->second.begin(), Pos->second.end());
  Pos->second.erase(std::unique(Pos->second.begin(), Pos->second.end()),
                    Pos->second.end());
  return Pos;
}

void ASTReader::loadAndAttachPreviousDecl(Decl *D, serialization::DeclID ID) {
  Decl *previous = GetDecl(ID);
  ASTDeclReader::attachPreviousDecl(D, previous);
}

/// \brief Read the declaration at the given offset from the AST file.
Decl *ASTReader::ReadDeclRecord(DeclID ID) {
  unsigned Index = ID - NUM_PREDEF_DECL_IDS;
  unsigned RawLocation = 0;
  RecordLocation Loc = DeclCursorForID(ID, RawLocation);
  llvm::BitstreamCursor &DeclsCursor = Loc.F->DeclsCursor;
  // Keep track of where we are in the stream, then jump back there
  // after reading this declaration.
  SavedStreamPosition SavedPosition(DeclsCursor);

  ReadingKindTracker ReadingKind(Read_Decl, *this);

  // Note that we are loading a declaration record.
  Deserializing ADecl(this);

  DeclsCursor.JumpToBit(Loc.Offset);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  unsigned Idx = 0;
  ASTDeclReader Reader(*this, *Loc.F, ID, RawLocation, Record,Idx);

  Decl *D = 0;
  switch ((DeclCode)DeclsCursor.ReadRecord(Code, Record)) {
  case DECL_CONTEXT_LEXICAL:
  case DECL_CONTEXT_VISIBLE:
    llvm_unreachable("Record cannot be de-serialized with ReadDeclRecord");
  case DECL_TYPEDEF:
    D = TypedefDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_TYPEALIAS:
    D = TypeAliasDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_ENUM:
    D = EnumDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_RECORD:
    D = RecordDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_ENUM_CONSTANT:
    D = EnumConstantDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FUNCTION:
    D = FunctionDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_LINKAGE_SPEC:
    D = LinkageSpecDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_LABEL:
    D = LabelDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_NAMESPACE:
    D = NamespaceDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_NAMESPACE_ALIAS:
    D = NamespaceAliasDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_USING:
    D = UsingDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_USING_SHADOW:
    D = UsingShadowDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_USING_DIRECTIVE:
    D = UsingDirectiveDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_UNRESOLVED_USING_VALUE:
    D = UnresolvedUsingValueDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_UNRESOLVED_USING_TYPENAME:
    D = UnresolvedUsingTypenameDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_RECORD:
    D = CXXRecordDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_METHOD:
    D = CXXMethodDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_CONSTRUCTOR:
    D = CXXConstructorDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_DESTRUCTOR:
    D = CXXDestructorDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_CONVERSION:
    D = CXXConversionDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_ACCESS_SPEC:
    D = AccessSpecDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FRIEND:
    D = FriendDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FRIEND_TEMPLATE:
    D = FriendTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_TEMPLATE:
    D = ClassTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_TEMPLATE_SPECIALIZATION:
    D = ClassTemplateSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION:
    D = ClassTemplatePartialSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_SCOPE_FUNCTION_SPECIALIZATION:
    D = ClassScopeFunctionSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FUNCTION_TEMPLATE:
    D = FunctionTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_TEMPLATE_TYPE_PARM:
    D = TemplateTypeParmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_NON_TYPE_TEMPLATE_PARM:
    D = NonTypeTemplateParmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_EXPANDED_NON_TYPE_TEMPLATE_PARM_PACK:
    D = NonTypeTemplateParmDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_TEMPLATE_TEMPLATE_PARM:
    D = TemplateTemplateParmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_EXPANDED_TEMPLATE_TEMPLATE_PARM_PACK:
    D = TemplateTemplateParmDecl::CreateDeserialized(Context, ID,
                                                     Record[Idx++]);
    break;
  case DECL_TYPE_ALIAS_TEMPLATE:
    D = TypeAliasTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_STATIC_ASSERT:
    D = StaticAssertDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_METHOD:
    D = ObjCMethodDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_INTERFACE:
    D = ObjCInterfaceDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_IVAR:
    D = ObjCIvarDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_PROTOCOL:
    D = ObjCProtocolDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_AT_DEFS_FIELD:
    D = ObjCAtDefsFieldDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_CATEGORY:
    D = ObjCCategoryDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_CATEGORY_IMPL:
    D = ObjCCategoryImplDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_IMPLEMENTATION:
    D = ObjCImplementationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_COMPATIBLE_ALIAS:
    D = ObjCCompatibleAliasDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_PROPERTY:
    D = ObjCPropertyDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_PROPERTY_IMPL:
    D = ObjCPropertyImplDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FIELD:
    D = FieldDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_INDIRECTFIELD:
    D = IndirectFieldDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_VAR:
    D = VarDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_IMPLICIT_PARAM:
    D = ImplicitParamDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_PARM_VAR:
    D = ParmVarDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FILE_SCOPE_ASM:
    D = FileScopeAsmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_BLOCK:
    D = BlockDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_BASE_SPECIFIERS:
    Error("attempt to read a C++ base-specifier record as a declaration");
    return 0;
  case DECL_IMPORT:
    // Note: last entry of the ImportDecl record is the number of stored source 
    // locations.
    D = ImportDecl::CreateDeserialized(Context, ID, Record.back());
    break;
  }

  assert(D && "Unknown declaration reading AST file");
  LoadedDecl(Index, D);
  // Set the DeclContext before doing any deserialization, to make sure internal
  // calls to Decl::getASTContext() by Decl's methods will find the
  // TranslationUnitDecl without crashing.
  D->setDeclContext(Context.getTranslationUnitDecl());
  Reader.Visit(D);

  // If this declaration is also a declaration context, get the
  // offsets for its tables of lexical and visible declarations.
  if (DeclContext *DC = dyn_cast<DeclContext>(D)) {
    std::pair<uint64_t, uint64_t> Offsets = Reader.VisitDeclContext(DC);
    if (Offsets.first || Offsets.second) {
      if (Offsets.first != 0)
        DC->setHasExternalLexicalStorage(true);
      if (Offsets.second != 0)
        DC->setHasExternalVisibleStorage(true);
      if (ReadDeclContextStorage(*Loc.F, DeclsCursor, Offsets, 
                                 Loc.F->DeclContextInfos[DC]))
        return 0;
    }

    // Now add the pending visible updates for this decl context, if it has any.
    DeclContextVisibleUpdatesPending::iterator I =
        PendingVisibleUpdates.find(ID);
    if (I != PendingVisibleUpdates.end()) {
      // There are updates. This means the context has external visible
      // storage, even if the original stored version didn't.
      DC->setHasExternalVisibleStorage(true);
      DeclContextVisibleUpdates &U = I->second;
      for (DeclContextVisibleUpdates::iterator UI = U.begin(), UE = U.end();
           UI != UE; ++UI) {
        DeclContextInfo &Info = UI->second->DeclContextInfos[DC];
        delete Info.NameLookupTableData;
        Info.NameLookupTableData = UI->first;
      }
      PendingVisibleUpdates.erase(I);
    }

    if (!DC->hasExternalVisibleStorage() && DC->hasExternalLexicalStorage())
      DC->setMustBuildLookupTable();
  }
  assert(Idx == Record.size());

  // Load any relevant update records.
  loadDeclUpdateRecords(ID, D);

  // Load the categories after recursive loading is finished.
  if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(D))
    if (Class->isThisDeclarationADefinition())
      loadObjCCategories(ID, Class);
  
  // If we have deserialized a declaration that has a definition the
  // AST consumer might need to know about, queue it.
  // We don't pass it to the consumer immediately because we may be in recursive
  // loading, and some declarations may still be initializing.
  if (isConsumerInterestedIn(D, Reader.hasPendingBody()))
    InterestingDecls.push_back(D);

  return D;
}

void ASTReader::loadDeclUpdateRecords(serialization::DeclID ID, Decl *D) {
  // The declaration may have been modified by files later in the chain.
  // If this is the case, read the record containing the updates from each file
  // and pass it to ASTDeclReader to make the modifications.
  DeclUpdateOffsetsMap::iterator UpdI = DeclUpdateOffsets.find(ID);
  if (UpdI != DeclUpdateOffsets.end()) {
    FileOffsetsTy &UpdateOffsets = UpdI->second;
    for (FileOffsetsTy::iterator
         I = UpdateOffsets.begin(), E = UpdateOffsets.end(); I != E; ++I) {
      ModuleFile *F = I->first;
      uint64_t Offset = I->second;
      llvm::BitstreamCursor &Cursor = F->DeclsCursor;
      SavedStreamPosition SavedPosition(Cursor);
      Cursor.JumpToBit(Offset);
      RecordData Record;
      unsigned Code = Cursor.ReadCode();
      unsigned RecCode = Cursor.ReadRecord(Code, Record);
      (void)RecCode;
      assert(RecCode == DECL_UPDATES && "Expected DECL_UPDATES record!");
      
      unsigned Idx = 0;
      ASTDeclReader Reader(*this, *F, ID, 0, Record, Idx);
      Reader.UpdateDecl(D, *F, Record);
    }
  }
}

namespace {
  struct CompareLocalRedeclarationsInfoToID {
    bool operator()(const LocalRedeclarationsInfo &X, DeclID Y) {
      return X.FirstID < Y;
    }

    bool operator()(DeclID X, const LocalRedeclarationsInfo &Y) {
      return X < Y.FirstID;
    }

    bool operator()(const LocalRedeclarationsInfo &X, 
                    const LocalRedeclarationsInfo &Y) {
      return X.FirstID < Y.FirstID;
    }
    bool operator()(DeclID X, DeclID Y) {
      return X < Y;
    }
  };
  
  /// \brief Module visitor class that finds all of the redeclarations of a 
  /// 
  class RedeclChainVisitor {
    ASTReader &Reader;
    SmallVectorImpl<DeclID> &SearchDecls;
    llvm::SmallPtrSet<Decl *, 16> &Deserialized;
    GlobalDeclID CanonID;
    llvm::SmallVector<Decl *, 4> Chain;
    
  public:
    RedeclChainVisitor(ASTReader &Reader, SmallVectorImpl<DeclID> &SearchDecls,
                       llvm::SmallPtrSet<Decl *, 16> &Deserialized,
                       GlobalDeclID CanonID)
      : Reader(Reader), SearchDecls(SearchDecls), Deserialized(Deserialized),
        CanonID(CanonID) { 
      for (unsigned I = 0, N = SearchDecls.size(); I != N; ++I)
        addToChain(Reader.GetDecl(SearchDecls[I]));
    }
    
    static bool visit(ModuleFile &M, bool Preorder, void *UserData) {
      if (Preorder)
        return false;
      
      return static_cast<RedeclChainVisitor *>(UserData)->visit(M);
    }
    
    void addToChain(Decl *D) {
      if (!D)
        return;
      
      if (Deserialized.erase(D))
        Chain.push_back(D);
    }
    
    void searchForID(ModuleFile &M, GlobalDeclID GlobalID) {
      // Map global ID of the first declaration down to the local ID
      // used in this module file.
      DeclID ID = Reader.mapGlobalIDToModuleFileGlobalID(M, GlobalID);
      if (!ID)
        return;
      
      // Perform a binary search to find the local redeclarations for this
      // declaration (if any).
      const LocalRedeclarationsInfo *Result
        = std::lower_bound(M.RedeclarationsMap,
                           M.RedeclarationsMap + M.LocalNumRedeclarationsInMap, 
                           ID, CompareLocalRedeclarationsInfoToID());
      if (Result == M.RedeclarationsMap + M.LocalNumRedeclarationsInMap ||
          Result->FirstID != ID) {
        // If we have a previously-canonical singleton declaration that was 
        // merged into another redeclaration chain, create a trivial chain
        // for this single declaration so that it will get wired into the 
        // complete redeclaration chain.
        if (GlobalID != CanonID && 
            GlobalID - NUM_PREDEF_DECL_IDS >= M.BaseDeclID && 
            GlobalID - NUM_PREDEF_DECL_IDS < M.BaseDeclID + M.LocalNumDecls) {
          addToChain(Reader.GetDecl(GlobalID));
        }
        
        return;
      }
      
      // Dig out all of the redeclarations.
      unsigned Offset = Result->Offset;
      unsigned N = M.RedeclarationChains[Offset];
      M.RedeclarationChains[Offset++] = 0; // Don't try to deserialize again
      for (unsigned I = 0; I != N; ++I)
        addToChain(Reader.GetLocalDecl(M, M.RedeclarationChains[Offset++]));
    }
    
    bool visit(ModuleFile &M) {
      // Visit each of the declarations.
      for (unsigned I = 0, N = SearchDecls.size(); I != N; ++I)
        searchForID(M, SearchDecls[I]);
      return false;
    }
    
    ArrayRef<Decl *> getChain() const {
      return Chain;
    }
  };
}

void ASTReader::loadPendingDeclChain(serialization::GlobalDeclID ID) {
  Decl *D = GetDecl(ID);  
  Decl *CanonDecl = D->getCanonicalDecl();
  
  // Determine the set of declaration IDs we'll be searching for.
  llvm::SmallVector<DeclID, 1> SearchDecls;
  GlobalDeclID CanonID = 0;
  if (D == CanonDecl) {
    SearchDecls.push_back(ID); // Always first.
    CanonID = ID;
  }
  MergedDeclsMap::iterator MergedPos = combineStoredMergedDecls(CanonDecl, ID);
  if (MergedPos != MergedDecls.end())
    SearchDecls.append(MergedPos->second.begin(), MergedPos->second.end());
  
  // Build up the list of redeclarations.
  RedeclChainVisitor Visitor(*this, SearchDecls, RedeclsDeserialized, CanonID);
  ModuleMgr.visitDepthFirst(&RedeclChainVisitor::visit, &Visitor);
  
  // Retrieve the chains.
  ArrayRef<Decl *> Chain = Visitor.getChain();
  if (Chain.empty())
    return;
    
  // Hook up the chains.
  Decl *MostRecent = CanonDecl->getMostRecentDecl();
  for (unsigned I = 0, N = Chain.size(); I != N; ++I) {
    if (Chain[I] == CanonDecl)
      continue;
    
    ASTDeclReader::attachPreviousDecl(Chain[I], MostRecent);
    MostRecent = Chain[I];
  }
  
  ASTDeclReader::attachLatestDecl(CanonDecl, MostRecent);  
}

namespace {
  struct CompareObjCCategoriesInfo {
    bool operator()(const ObjCCategoriesInfo &X, DeclID Y) {
      return X.DefinitionID < Y;
    }
    
    bool operator()(DeclID X, const ObjCCategoriesInfo &Y) {
      return X < Y.DefinitionID;
    }
    
    bool operator()(const ObjCCategoriesInfo &X, 
                    const ObjCCategoriesInfo &Y) {
      return X.DefinitionID < Y.DefinitionID;
    }
    bool operator()(DeclID X, DeclID Y) {
      return X < Y;
    }
  };

  /// \brief Given an ObjC interface, goes through the modules and links to the
  /// interface all the categories for it.
  class ObjCCategoriesVisitor {
    ASTReader &Reader;
    serialization::GlobalDeclID InterfaceID;
    ObjCInterfaceDecl *Interface;
    llvm::SmallPtrSet<ObjCCategoryDecl *, 16> &Deserialized;
    unsigned PreviousGeneration;
    ObjCCategoryDecl *Tail;
    llvm::DenseMap<DeclarationName, ObjCCategoryDecl *> NameCategoryMap;
    
    void add(ObjCCategoryDecl *Cat) {
      // Only process each category once.
      if (!Deserialized.erase(Cat))
        return;
      
      // Check for duplicate categories.
      if (Cat->getDeclName()) {
        ObjCCategoryDecl *&Existing = NameCategoryMap[Cat->getDeclName()];
        if (Existing && 
            Reader.getOwningModuleFile(Existing) 
                                          != Reader.getOwningModuleFile(Cat)) {
          // FIXME: We should not warn for duplicates in diamond:
          //
          //   MT     //
          //  /  \    //
          // ML  MR   //
          //  \  /    //
          //   MB     //
          //
          // If there are duplicates in ML/MR, there will be warning when 
          // creating MB *and* when importing MB. We should not warn when 
          // importing.
          Reader.Diag(Cat->getLocation(), diag::warn_dup_category_def)
            << Interface->getDeclName() << Cat->getDeclName();
          Reader.Diag(Existing->getLocation(), diag::note_previous_definition);
        } else if (!Existing) {
          // Record this category.
          Existing = Cat;
        }
      }
      
      // Add this category to the end of the chain.
      if (Tail)
        ASTDeclReader::setNextObjCCategory(Tail, Cat);
      else
        Interface->setCategoryList(Cat);
      Tail = Cat;
    }
    
  public:
    ObjCCategoriesVisitor(ASTReader &Reader,
                          serialization::GlobalDeclID InterfaceID,
                          ObjCInterfaceDecl *Interface,
                        llvm::SmallPtrSet<ObjCCategoryDecl *, 16> &Deserialized,
                          unsigned PreviousGeneration)
      : Reader(Reader), InterfaceID(InterfaceID), Interface(Interface),
        Deserialized(Deserialized), PreviousGeneration(PreviousGeneration),
        Tail(0) 
    {
      // Populate the name -> category map with the set of known categories.
      for (ObjCCategoryDecl *Cat = Interface->getCategoryList(); Cat;
           Cat = Cat->getNextClassCategory()) {
        if (Cat->getDeclName())
          NameCategoryMap[Cat->getDeclName()] = Cat;
        
        // Keep track of the tail of the category list.
        Tail = Cat;
      }
    }

    static bool visit(ModuleFile &M, void *UserData) {
      return static_cast<ObjCCategoriesVisitor *>(UserData)->visit(M);
    }

    bool visit(ModuleFile &M) {
      // If we've loaded all of the category information we care about from
      // this module file, we're done.
      if (M.Generation <= PreviousGeneration)
        return true;
      
      // Map global ID of the definition down to the local ID used in this 
      // module file. If there is no such mapping, we'll find nothing here
      // (or in any module it imports).
      DeclID LocalID = Reader.mapGlobalIDToModuleFileGlobalID(M, InterfaceID);
      if (!LocalID)
        return true;

      // Perform a binary search to find the local redeclarations for this
      // declaration (if any).
      const ObjCCategoriesInfo *Result
        = std::lower_bound(M.ObjCCategoriesMap,
                           M.ObjCCategoriesMap + M.LocalNumObjCCategoriesInMap, 
                           LocalID, CompareObjCCategoriesInfo());
      if (Result == M.ObjCCategoriesMap + M.LocalNumObjCCategoriesInMap ||
          Result->DefinitionID != LocalID) {
        // We didn't find anything. If the class definition is in this module
        // file, then the module files it depends on cannot have any categories,
        // so suppress further lookup.
        return Reader.isDeclIDFromModule(InterfaceID, M);
      }
      
      // We found something. Dig out all of the categories.
      unsigned Offset = Result->Offset;
      unsigned N = M.ObjCCategories[Offset];
      M.ObjCCategories[Offset++] = 0; // Don't try to deserialize again
      for (unsigned I = 0; I != N; ++I)
        add(cast_or_null<ObjCCategoryDecl>(
              Reader.GetLocalDecl(M, M.ObjCCategories[Offset++])));
      return true;
    }
  };
}

void ASTReader::loadObjCCategories(serialization::GlobalDeclID ID,
                                   ObjCInterfaceDecl *D,
                                   unsigned PreviousGeneration) {
  ObjCCategoriesVisitor Visitor(*this, ID, D, CategoriesDeserialized,
                                PreviousGeneration);
  ModuleMgr.visit(ObjCCategoriesVisitor::visit, &Visitor);
}

void ASTDeclReader::UpdateDecl(Decl *D, ModuleFile &ModuleFile,
                               const RecordData &Record) {
  unsigned Idx = 0;
  while (Idx < Record.size()) {
    switch ((DeclUpdateKind)Record[Idx++]) {
    case UPD_CXX_ADDED_IMPLICIT_MEMBER:
      cast<CXXRecordDecl>(D)->addedMember(Reader.ReadDecl(ModuleFile, Record, Idx));
      break;

    case UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION:
      // It will be added to the template's specializations set when loaded.
      (void)Reader.ReadDecl(ModuleFile, Record, Idx);
      break;

    case UPD_CXX_ADDED_ANONYMOUS_NAMESPACE: {
      NamespaceDecl *Anon
        = Reader.ReadDeclAs<NamespaceDecl>(ModuleFile, Record, Idx);
      
      // Each module has its own anonymous namespace, which is disjoint from
      // any other module's anonymous namespaces, so don't attach the anonymous
      // namespace at all.
      if (ModuleFile.Kind != MK_Module) {
        if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(D))
          TU->setAnonymousNamespace(Anon);
        else
          cast<NamespaceDecl>(D)->setAnonymousNamespace(Anon);
      }
      break;
    }

    case UPD_CXX_INSTANTIATED_STATIC_DATA_MEMBER:
      cast<VarDecl>(D)->getMemberSpecializationInfo()->setPointOfInstantiation(
          Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      break;
    }
  }
}
