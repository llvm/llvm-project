//===- IndexingContext.cpp - Higher level API functions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"
#include "CIndexDiagnostic.h"
#include "CXTranslationUnit.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Frontend/ASTUnit.h"

using namespace clang;
using namespace cxindex;
using namespace cxcursor;

IndexingContext::ObjCProtocolListInfo::ObjCProtocolListInfo(
                                    const ObjCProtocolList &ProtList,
                                    IndexingContext &IdxCtx,
                                    ScratchAlloc &SA) {
  ObjCInterfaceDecl::protocol_loc_iterator LI = ProtList.loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator
         I = ProtList.begin(), E = ProtList.end(); I != E; ++I, ++LI) {
    SourceLocation Loc = *LI;
    ObjCProtocolDecl *PD = *I;
    ProtEntities.push_back(EntityInfo());
    IdxCtx.getEntityInfo(PD, ProtEntities.back(), SA);
    CXIdxObjCProtocolRefInfo ProtInfo = { nullptr,
                                MakeCursorObjCProtocolRef(PD, Loc, IdxCtx.CXTU),
                                IdxCtx.getIndexLoc(Loc) };
    ProtInfos.push_back(ProtInfo);

    if (IdxCtx.shouldSuppressRefs())
      IdxCtx.markEntityOccurrenceInFile(PD, Loc);
  }

  for (unsigned i = 0, e = ProtInfos.size(); i != e; ++i)
    ProtInfos[i].protocol = &ProtEntities[i];

  for (unsigned i = 0, e = ProtInfos.size(); i != e; ++i)
    Prots.push_back(&ProtInfos[i]);
}


IBOutletCollectionInfo::IBOutletCollectionInfo(
                                          const IBOutletCollectionInfo &other)
  : AttrInfo(CXIdxAttr_IBOutletCollection, other.cursor, other.loc, other.A) {

  IBCollInfo.attrInfo = this;
  IBCollInfo.classCursor = other.IBCollInfo.classCursor;
  IBCollInfo.classLoc = other.IBCollInfo.classLoc;
  if (other.IBCollInfo.objcClass) {
    ClassInfo = other.ClassInfo;
    IBCollInfo.objcClass = &ClassInfo;
  } else
    IBCollInfo.objcClass = nullptr;
}

AttrListInfo::AttrListInfo(const Decl *D, IndexingContext &IdxCtx)
  : SA(IdxCtx), ref_cnt(0) {

  if (!D->hasAttrs())
    return;

  for (const auto *A : D->attrs()) {
    CXCursor C = MakeCXCursor(A, D, IdxCtx.CXTU);
    CXIdxLoc Loc =  IdxCtx.getIndexLoc(A->getLocation());
    switch (C.kind) {
    default:
      Attrs.push_back(AttrInfo(CXIdxAttr_Unexposed, C, Loc, A));
      break;
    case CXCursor_IBActionAttr:
      Attrs.push_back(AttrInfo(CXIdxAttr_IBAction, C, Loc, A));
      break;
    case CXCursor_IBOutletAttr:
      Attrs.push_back(AttrInfo(CXIdxAttr_IBOutlet, C, Loc, A));
      break;
    case CXCursor_IBOutletCollectionAttr:
      IBCollAttrs.push_back(IBOutletCollectionInfo(C, Loc, A));
      break;
    }
  }

  for (unsigned i = 0, e = IBCollAttrs.size(); i != e; ++i) {
    IBOutletCollectionInfo &IBInfo = IBCollAttrs[i];
    CXAttrs.push_back(&IBInfo);

    const IBOutletCollectionAttr *
      IBAttr = cast<IBOutletCollectionAttr>(IBInfo.A);
    SourceLocation InterfaceLocStart =
        IBAttr->getInterfaceLoc()->getTypeLoc().getLocStart();
    IBInfo.IBCollInfo.attrInfo = &IBInfo;
    IBInfo.IBCollInfo.classLoc = IdxCtx.getIndexLoc(InterfaceLocStart);
    IBInfo.IBCollInfo.objcClass = nullptr;
    IBInfo.IBCollInfo.classCursor = clang_getNullCursor();
    QualType Ty = IBAttr->getInterface();
    if (const ObjCObjectType *ObjectTy = Ty->getAs<ObjCObjectType>()) {
      if (const ObjCInterfaceDecl *InterD = ObjectTy->getInterface()) {
        IdxCtx.getEntityInfo(InterD, IBInfo.ClassInfo, SA);
        IBInfo.IBCollInfo.objcClass = &IBInfo.ClassInfo;
        IBInfo.IBCollInfo.classCursor =
            MakeCursorObjCClassRef(InterD, InterfaceLocStart, IdxCtx.CXTU);
      }
    }
  }

  for (unsigned i = 0, e = Attrs.size(); i != e; ++i)
    CXAttrs.push_back(&Attrs[i]);
}

IntrusiveRefCntPtr<AttrListInfo>
AttrListInfo::create(const Decl *D, IndexingContext &IdxCtx) {
  ScratchAlloc SA(IdxCtx);
  AttrListInfo *attrs = SA.allocate<AttrListInfo>();
  return new (attrs) AttrListInfo(D, IdxCtx);
}

IndexingContext::CXXBasesListInfo::CXXBasesListInfo(const CXXRecordDecl *D,
                                   IndexingContext &IdxCtx,
                                   ScratchAlloc &SA) {
  for (const auto &Base : D->bases()) {
    BaseEntities.push_back(EntityInfo());
    const NamedDecl *BaseD = nullptr;
    QualType T = Base.getType();
    SourceLocation Loc = getBaseLoc(Base);

    if (const TypedefType *TDT = T->getAs<TypedefType>()) {
      BaseD = TDT->getDecl();
    } else if (const TemplateSpecializationType *
          TST = T->getAs<TemplateSpecializationType>()) {
      BaseD = TST->getTemplateName().getAsTemplateDecl();
    } else if (const RecordType *RT = T->getAs<RecordType>()) {
      BaseD = RT->getDecl();
    }

    if (BaseD)
      IdxCtx.getEntityInfo(BaseD, BaseEntities.back(), SA);
    CXIdxBaseClassInfo BaseInfo = { nullptr,
                         MakeCursorCXXBaseSpecifier(&Base, IdxCtx.CXTU),
                         IdxCtx.getIndexLoc(Loc) };
    BaseInfos.push_back(BaseInfo);
  }

  for (unsigned i = 0, e = BaseInfos.size(); i != e; ++i) {
    if (BaseEntities[i].name && BaseEntities[i].USR)
      BaseInfos[i].base = &BaseEntities[i];
  }

  for (unsigned i = 0, e = BaseInfos.size(); i != e; ++i)
    CXBases.push_back(&BaseInfos[i]);
}

SourceLocation IndexingContext::CXXBasesListInfo::getBaseLoc(
                                           const CXXBaseSpecifier &Base) const {
  SourceLocation Loc = Base.getSourceRange().getBegin();
  TypeLoc TL;
  if (Base.getTypeSourceInfo())
    TL = Base.getTypeSourceInfo()->getTypeLoc();
  if (TL.isNull())
    return Loc;

  if (QualifiedTypeLoc QL = TL.getAs<QualifiedTypeLoc>())
    TL = QL.getUnqualifiedLoc();

  if (ElaboratedTypeLoc EL = TL.getAs<ElaboratedTypeLoc>())
    return EL.getNamedTypeLoc().getBeginLoc();
  if (DependentNameTypeLoc DL = TL.getAs<DependentNameTypeLoc>())
    return DL.getNameLoc();
  if (DependentTemplateSpecializationTypeLoc DTL =
          TL.getAs<DependentTemplateSpecializationTypeLoc>())
    return DTL.getTemplateNameLoc();

  return Loc;
}

const char *ScratchAlloc::toCStr(StringRef Str) {
  if (Str.empty())
    return "";
  if (Str.data()[Str.size()] == '\0')
    return Str.data();
  return copyCStr(Str);
}

const char *ScratchAlloc::copyCStr(StringRef Str) {
  char *buf = IdxCtx.StrScratch.Allocate<char>(Str.size() + 1);
  std::uninitialized_copy(Str.begin(), Str.end(), buf);
  buf[Str.size()] = '\0';
  return buf;
}

void IndexingContext::setASTContext(ASTContext &ctx) {
  Ctx = &ctx;
  cxtu::getASTUnit(CXTU)->setASTContext(&ctx);
}

void IndexingContext::setPreprocessor(Preprocessor &PP) {
  cxtu::getASTUnit(CXTU)->setPreprocessor(&PP);
}

bool IndexingContext::isFunctionLocalDecl(const Decl *D) {
  assert(D);

  if (!D->getParentFunctionOrMethod())
    return false;

  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    switch (ND->getFormalLinkage()) {
    case NoLinkage:
    case VisibleNoLinkage:
    case InternalLinkage:
      return true;
    case UniqueExternalLinkage:
      llvm_unreachable("Not a sema linkage");
    case ExternalLinkage:
      return false;
    }
  }

  return true;
}

bool IndexingContext::shouldAbort() {
  if (!CB.abortQuery)
    return false;
  return CB.abortQuery(ClientData, nullptr);
}

void IndexingContext::enteredMainFile(const FileEntry *File) {
  if (File && CB.enteredMainFile) {
    CXIdxClientFile idxFile =
      CB.enteredMainFile(ClientData,
                         static_cast<CXFile>(const_cast<FileEntry *>(File)),
                         nullptr);
    FileMap[File] = idxFile;
  }
}

void IndexingContext::ppIncludedFile(SourceLocation hashLoc,
                                     StringRef filename,
                                     const FileEntry *File,
                                     bool isImport, bool isAngled,
                                     bool isModuleImport) {
  if (!CB.ppIncludedFile)
    return;

  ScratchAlloc SA(*this);
  CXIdxIncludedFileInfo Info = { getIndexLoc(hashLoc),
                                 SA.toCStr(filename),
                                 static_cast<CXFile>(
                                   const_cast<FileEntry *>(File)),
                                 isImport, isAngled, isModuleImport };
  CXIdxClientFile idxFile = CB.ppIncludedFile(ClientData, &Info);
  FileMap[File] = idxFile;
}

void IndexingContext::importedModule(const ImportDecl *ImportD) {
  if (!CB.importedASTFile)
    return;

  Module *Mod = ImportD->getImportedModule();
  if (!Mod)
    return;

  CXIdxImportedASTFileInfo Info = {
                                    static_cast<CXFile>(
                                    const_cast<FileEntry *>(Mod->getASTFile())),
                                    Mod,
                                    getIndexLoc(ImportD->getLocation()),
                                    ImportD->isImplicit()
                                  };
  CXIdxClientASTFile astFile = CB.importedASTFile(ClientData, &Info);
  (void)astFile;
}

void IndexingContext::importedPCH(const FileEntry *File) {
  if (!CB.importedASTFile)
    return;

  CXIdxImportedASTFileInfo Info = {
                                    static_cast<CXFile>(
                                      const_cast<FileEntry *>(File)),
                                    /*module=*/nullptr,
                                    getIndexLoc(SourceLocation()),
                                    /*isImplicit=*/false
                                  };
  CXIdxClientASTFile astFile = CB.importedASTFile(ClientData, &Info);
  (void)astFile;
}

void IndexingContext::startedTranslationUnit() {
  CXIdxClientContainer idxCont = nullptr;
  if (CB.startedTranslationUnit)
    idxCont = CB.startedTranslationUnit(ClientData, nullptr);
  addContainerInMap(Ctx->getTranslationUnitDecl(), idxCont);
}

void IndexingContext::handleDiagnosticSet(CXDiagnostic CXDiagSet) {
  if (!CB.diagnostic)
    return;

  CB.diagnostic(ClientData, CXDiagSet, nullptr);
}

bool IndexingContext::handleDecl(const NamedDecl *D,
                                 SourceLocation Loc, CXCursor Cursor,
                                 DeclInfo &DInfo,
                                 const DeclContext *LexicalDC) {
  if (!CB.indexDeclaration || !D)
    return false;
  if (D->isImplicit() && shouldIgnoreIfImplicit(D))
    return false;

  ScratchAlloc SA(*this);
  getEntityInfo(D, DInfo.EntInfo, SA);
  if ((!shouldIndexFunctionLocalSymbols() && !DInfo.EntInfo.USR)
      || Loc.isInvalid())
    return false;

  if (!LexicalDC)
    LexicalDC = D->getLexicalDeclContext();

  if (shouldSuppressRefs())
    markEntityOccurrenceInFile(D, Loc);
  
  DInfo.entityInfo = &DInfo.EntInfo;
  DInfo.cursor = Cursor;
  DInfo.loc = getIndexLoc(Loc);
  DInfo.isImplicit = D->isImplicit();

  DInfo.attributes = DInfo.EntInfo.attributes;
  DInfo.numAttributes = DInfo.EntInfo.numAttributes;

  getContainerInfo(D->getDeclContext(), DInfo.SemanticContainer);
  DInfo.semanticContainer = &DInfo.SemanticContainer;

  if (LexicalDC == D->getDeclContext()) {
    DInfo.lexicalContainer = &DInfo.SemanticContainer;
  } else if (isTemplateImplicitInstantiation(D)) {
    // Implicit instantiations have the lexical context of where they were
    // instantiated first. We choose instead the semantic context because:
    // 1) at the time that we see the instantiation we have not seen the
    //   function where it occurred yet.
    // 2) the lexical context of the first instantiation is not useful
    //   information anyway.
    DInfo.lexicalContainer = &DInfo.SemanticContainer;
  } else {
    getContainerInfo(LexicalDC, DInfo.LexicalContainer);
    DInfo.lexicalContainer = &DInfo.LexicalContainer;
  }

  if (DInfo.isContainer) {
    getContainerInfo(getEntityContainer(D), DInfo.DeclAsContainer);
    DInfo.declAsContainer = &DInfo.DeclAsContainer;
  }

  CB.indexDeclaration(ClientData, &DInfo);
  return true;
}

bool IndexingContext::handleObjCContainer(const ObjCContainerDecl *D,
                                          SourceLocation Loc, CXCursor Cursor,
                                          ObjCContainerDeclInfo &ContDInfo) {
  ContDInfo.ObjCContDeclInfo.declInfo = &ContDInfo;
  return handleDecl(D, Loc, Cursor, ContDInfo);
}

bool IndexingContext::handleFunction(const FunctionDecl *D) {
  bool isDef = D->isThisDeclarationADefinition();
  bool isContainer = isDef;
  bool isSkipped = false;
  if (D->hasSkippedBody()) {
    isSkipped = true;
    isDef = true;
    isContainer = false;
  }

  DeclInfo DInfo(!D->isFirstDecl(), isDef, isContainer);
  if (isSkipped)
    DInfo.flags |= CXIdxDeclFlag_Skipped;
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleVar(const VarDecl *D) {
  DeclInfo DInfo(!D->isFirstDecl(), D->isThisDeclarationADefinition(),
                 /*isContainer=*/false);
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleField(const FieldDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/false, /*isDefinition=*/true,
                 /*isContainer=*/false);
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleMSProperty(const MSPropertyDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/false, /*isDefinition=*/true,
                 /*isContainer=*/false);
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleEnumerator(const EnumConstantDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/false, /*isDefinition=*/true,
                 /*isContainer=*/false);
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleTagDecl(const TagDecl *D) {
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(D))
    return handleCXXRecordDecl(CXXRD, D);

  DeclInfo DInfo(!D->isFirstDecl(), D->isThisDeclarationADefinition(),
                 D->isThisDeclarationADefinition());
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleTypedefName(const TypedefNameDecl *D) {
  DeclInfo DInfo(!D->isFirstDecl(), /*isDefinition=*/true,
                 /*isContainer=*/false);
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleObjCInterface(const ObjCInterfaceDecl *D) {
  // For @class forward declarations, suppress them the same way as references.
  if (!D->isThisDeclarationADefinition()) {
    if (shouldSuppressRefs() && markEntityOccurrenceInFile(D, D->getLocation()))
      return false; // already occurred.

    // FIXME: This seems like the wrong definition for redeclaration.
    bool isRedeclaration = D->hasDefinition() || D->getPreviousDecl();
    ObjCContainerDeclInfo ContDInfo(/*isForwardRef=*/true, isRedeclaration,
                                    /*isImplementation=*/false);
    return handleObjCContainer(D, D->getLocation(),
                               MakeCursorObjCClassRef(D, D->getLocation(),
                                                      CXTU), 
                               ContDInfo);
  }

  ScratchAlloc SA(*this);

  CXIdxBaseClassInfo BaseClass;
  EntityInfo BaseEntity;
  BaseClass.cursor = clang_getNullCursor();
  if (ObjCInterfaceDecl *SuperD = D->getSuperClass()) {
    getEntityInfo(SuperD, BaseEntity, SA);
    SourceLocation SuperLoc = D->getSuperClassLoc();
    BaseClass.base = &BaseEntity;
    BaseClass.cursor = MakeCursorObjCSuperClassRef(SuperD, SuperLoc, CXTU);
    BaseClass.loc = getIndexLoc(SuperLoc);

    if (shouldSuppressRefs())
      markEntityOccurrenceInFile(SuperD, SuperLoc);
  }
  
  ObjCProtocolList EmptyProtoList;
  ObjCProtocolListInfo ProtInfo(D->isThisDeclarationADefinition() 
                                  ? D->getReferencedProtocols()
                                  : EmptyProtoList, 
                                *this, SA);
  
  ObjCInterfaceDeclInfo InterInfo(D);
  InterInfo.ObjCProtoListInfo = ProtInfo.getListInfo();
  InterInfo.ObjCInterDeclInfo.containerInfo = &InterInfo.ObjCContDeclInfo;
  InterInfo.ObjCInterDeclInfo.superInfo = D->getSuperClass() ? &BaseClass
                                                             : nullptr;
  InterInfo.ObjCInterDeclInfo.protocols = &InterInfo.ObjCProtoListInfo;

  return handleObjCContainer(D, D->getLocation(), getCursor(D), InterInfo);
}

bool IndexingContext::handleObjCImplementation(
                                              const ObjCImplementationDecl *D) {
  ObjCContainerDeclInfo ContDInfo(/*isForwardRef=*/false,
                      /*isRedeclaration=*/true,
                      /*isImplementation=*/true);
  return handleObjCContainer(D, D->getLocation(), getCursor(D), ContDInfo);
}

bool IndexingContext::handleObjCProtocol(const ObjCProtocolDecl *D) {
  if (!D->isThisDeclarationADefinition()) {
    if (shouldSuppressRefs() && markEntityOccurrenceInFile(D, D->getLocation()))
      return false; // already occurred.
    
    // FIXME: This seems like the wrong definition for redeclaration.
    bool isRedeclaration = D->hasDefinition() || D->getPreviousDecl();
    ObjCContainerDeclInfo ContDInfo(/*isForwardRef=*/true,
                                    isRedeclaration,
                                    /*isImplementation=*/false);
    return handleObjCContainer(D, D->getLocation(), 
                               MakeCursorObjCProtocolRef(D, D->getLocation(),
                                                         CXTU),
                               ContDInfo);    
  }
  
  ScratchAlloc SA(*this);
  ObjCProtocolList EmptyProtoList;
  ObjCProtocolListInfo ProtListInfo(D->isThisDeclarationADefinition()
                                      ? D->getReferencedProtocols()
                                      : EmptyProtoList,
                                    *this, SA);
  
  ObjCProtocolDeclInfo ProtInfo(D);
  ProtInfo.ObjCProtoRefListInfo = ProtListInfo.getListInfo();

  return handleObjCContainer(D, D->getLocation(), getCursor(D), ProtInfo);
}

bool IndexingContext::handleObjCCategory(const ObjCCategoryDecl *D) {
  ScratchAlloc SA(*this);

  ObjCCategoryDeclInfo CatDInfo(/*isImplementation=*/false);
  EntityInfo ClassEntity;
  const ObjCInterfaceDecl *IFaceD = D->getClassInterface();
  SourceLocation ClassLoc = D->getLocation();
  SourceLocation CategoryLoc = D->IsClassExtension() ? ClassLoc
                                                     : D->getCategoryNameLoc();
  getEntityInfo(IFaceD, ClassEntity, SA);

  if (shouldSuppressRefs())
    markEntityOccurrenceInFile(IFaceD, ClassLoc);

  ObjCProtocolListInfo ProtInfo(D->getReferencedProtocols(), *this, SA);
  
  CatDInfo.ObjCCatDeclInfo.containerInfo = &CatDInfo.ObjCContDeclInfo;
  if (IFaceD) {
    CatDInfo.ObjCCatDeclInfo.objcClass = &ClassEntity;
    CatDInfo.ObjCCatDeclInfo.classCursor =
        MakeCursorObjCClassRef(IFaceD, ClassLoc, CXTU);
  } else {
    CatDInfo.ObjCCatDeclInfo.objcClass = nullptr;
    CatDInfo.ObjCCatDeclInfo.classCursor = clang_getNullCursor();
  }
  CatDInfo.ObjCCatDeclInfo.classLoc = getIndexLoc(ClassLoc);
  CatDInfo.ObjCProtoListInfo = ProtInfo.getListInfo();
  CatDInfo.ObjCCatDeclInfo.protocols = &CatDInfo.ObjCProtoListInfo;

  return handleObjCContainer(D, CategoryLoc, getCursor(D), CatDInfo);
}

bool IndexingContext::handleObjCCategoryImpl(const ObjCCategoryImplDecl *D) {
  ScratchAlloc SA(*this);

  const ObjCCategoryDecl *CatD = D->getCategoryDecl();
  ObjCCategoryDeclInfo CatDInfo(/*isImplementation=*/true);
  EntityInfo ClassEntity;
  const ObjCInterfaceDecl *IFaceD = CatD->getClassInterface();
  SourceLocation ClassLoc = D->getLocation();
  SourceLocation CategoryLoc = D->getCategoryNameLoc();
  getEntityInfo(IFaceD, ClassEntity, SA);

  if (shouldSuppressRefs())
    markEntityOccurrenceInFile(IFaceD, ClassLoc);

  CatDInfo.ObjCCatDeclInfo.containerInfo = &CatDInfo.ObjCContDeclInfo;
  if (IFaceD) {
    CatDInfo.ObjCCatDeclInfo.objcClass = &ClassEntity;
    CatDInfo.ObjCCatDeclInfo.classCursor =
        MakeCursorObjCClassRef(IFaceD, ClassLoc, CXTU);
  } else {
    CatDInfo.ObjCCatDeclInfo.objcClass = nullptr;
    CatDInfo.ObjCCatDeclInfo.classCursor = clang_getNullCursor();
  }
  CatDInfo.ObjCCatDeclInfo.classLoc = getIndexLoc(ClassLoc);
  CatDInfo.ObjCCatDeclInfo.protocols = nullptr;

  return handleObjCContainer(D, CategoryLoc, getCursor(D), CatDInfo);
}

bool IndexingContext::handleObjCMethod(const ObjCMethodDecl *D) {
  bool isDef = D->isThisDeclarationADefinition();
  bool isContainer = isDef;
  bool isSkipped = false;
  if (D->hasSkippedBody()) {
    isSkipped = true;
    isDef = true;
    isContainer = false;
  }

  DeclInfo DInfo(!D->isCanonicalDecl(), isDef, isContainer);
  if (isSkipped)
    DInfo.flags |= CXIdxDeclFlag_Skipped;
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleSynthesizedObjCProperty(
                                                const ObjCPropertyImplDecl *D) {
  ObjCPropertyDecl *PD = D->getPropertyDecl();
  return handleReference(PD, D->getLocation(), getCursor(D), nullptr,
                         D->getDeclContext());
}

bool IndexingContext::handleSynthesizedObjCMethod(const ObjCMethodDecl *D,
                                                  SourceLocation Loc,
                                                 const DeclContext *LexicalDC) {
  DeclInfo DInfo(/*isRedeclaration=*/true, /*isDefinition=*/true,
                 /*isContainer=*/false);
  return handleDecl(D, Loc, getCursor(D), DInfo, LexicalDC);
}

bool IndexingContext::handleObjCProperty(const ObjCPropertyDecl *D) {
  ScratchAlloc SA(*this);

  ObjCPropertyDeclInfo DInfo;
  EntityInfo GetterEntity;
  EntityInfo SetterEntity;

  DInfo.ObjCPropDeclInfo.declInfo = &DInfo;

  if (ObjCMethodDecl *Getter = D->getGetterMethodDecl()) {
    getEntityInfo(Getter, GetterEntity, SA);
    DInfo.ObjCPropDeclInfo.getter = &GetterEntity;
  } else {
    DInfo.ObjCPropDeclInfo.getter = nullptr;
  }
  if (ObjCMethodDecl *Setter = D->getSetterMethodDecl()) {
    getEntityInfo(Setter, SetterEntity, SA);
    DInfo.ObjCPropDeclInfo.setter = &SetterEntity;
  } else {
    DInfo.ObjCPropDeclInfo.setter = nullptr;
  }

  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleNamespace(const NamespaceDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/!D->isOriginalNamespace(),
                 /*isDefinition=*/true,
                 /*isContainer=*/true);
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleClassTemplate(const ClassTemplateDecl *D) {
  return handleCXXRecordDecl(D->getTemplatedDecl(), D);
}

bool IndexingContext::handleFunctionTemplate(const FunctionTemplateDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/!D->isCanonicalDecl(),
                 /*isDefinition=*/D->isThisDeclarationADefinition(),
                 /*isContainer=*/D->isThisDeclarationADefinition());
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleTypeAliasTemplate(const TypeAliasTemplateDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/!D->isCanonicalDecl(),
                 /*isDefinition=*/true, /*isContainer=*/false);
  return handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

bool IndexingContext::handleReference(const NamedDecl *D, SourceLocation Loc,
                                      const NamedDecl *Parent,
                                      const DeclContext *DC,
                                      const Expr *E,
                                      CXIdxEntityRefKind Kind) {
  if (!D)
    return false;

  CXCursor Cursor = E ? MakeCXCursor(E, cast<Decl>(DC), CXTU)
                      : getRefCursor(D, Loc);
  return handleReference(D, Loc, Cursor, Parent, DC, E, Kind);
}

bool IndexingContext::handleReference(const NamedDecl *D, SourceLocation Loc,
                                      CXCursor Cursor,
                                      const NamedDecl *Parent,
                                      const DeclContext *DC,
                                      const Expr *E,
                                      CXIdxEntityRefKind Kind) {
  if (!CB.indexEntityReference)
    return false;

  if (!D)
    return false;
  if (Loc.isInvalid())
    return false;
  if (!shouldIndexFunctionLocalSymbols() && isFunctionLocalDecl(D))
    return false;
  if (isNotFromSourceFile(D->getLocation()))
    return false;
  if (D->isImplicit() && shouldIgnoreIfImplicit(D))
    return false;

  if (shouldSuppressRefs()) {
    if (markEntityOccurrenceInFile(D, Loc))
      return false; // already occurred.
  }

  ScratchAlloc SA(*this);
  EntityInfo RefEntity, ParentEntity;
  getEntityInfo(D, RefEntity, SA);
  if (!RefEntity.USR)
    return false;

  getEntityInfo(Parent, ParentEntity, SA);

  ContainerInfo Container;
  getContainerInfo(DC, Container);

  CXIdxEntityRefInfo Info = { Kind,
                              Cursor,
                              getIndexLoc(Loc),
                              &RefEntity,
                              Parent ? &ParentEntity : nullptr,
                              &Container };
  CB.indexEntityReference(ClientData, &Info);
  return true;
}

bool IndexingContext::isNotFromSourceFile(SourceLocation Loc) const {
  if (Loc.isInvalid())
    return true;
  SourceManager &SM = Ctx->getSourceManager();
  SourceLocation FileLoc = SM.getFileLoc(Loc);
  FileID FID = SM.getFileID(FileLoc);
  return SM.getFileEntryForID(FID) == nullptr;
}

void IndexingContext::addContainerInMap(const DeclContext *DC,
                                        CXIdxClientContainer container) {
  if (!DC)
    return;

  ContainerMapTy::iterator I = ContainerMap.find(DC);
  if (I == ContainerMap.end()) {
    if (container)
      ContainerMap[DC] = container;
    return;
  }
  // Allow changing the container of a previously seen DeclContext so we
  // can handle invalid user code, like a function re-definition.
  if (container)
    I->second = container;
  else
    ContainerMap.erase(I);
}

CXIdxClientEntity IndexingContext::getClientEntity(const Decl *D) const {
  if (!D)
    return nullptr;
  EntityMapTy::const_iterator I = EntityMap.find(D);
  if (I == EntityMap.end())
    return nullptr;
  return I->second;
}

void IndexingContext::setClientEntity(const Decl *D, CXIdxClientEntity client) {
  if (!D)
    return;
  EntityMap[D] = client;
}

bool IndexingContext::handleCXXRecordDecl(const CXXRecordDecl *RD,
                                          const NamedDecl *OrigD) {
  if (RD->isThisDeclarationADefinition()) {
    ScratchAlloc SA(*this);
    CXXClassDeclInfo CXXDInfo(/*isRedeclaration=*/!OrigD->isCanonicalDecl(),
                           /*isDefinition=*/RD->isThisDeclarationADefinition());
    CXXBasesListInfo BaseList(RD, *this, SA);
    CXXDInfo.CXXClassInfo.declInfo = &CXXDInfo;
    CXXDInfo.CXXClassInfo.bases = BaseList.getBases();
    CXXDInfo.CXXClassInfo.numBases = BaseList.getNumBases();

    if (shouldSuppressRefs()) {
      // Go through bases and mark them as referenced.
      for (unsigned i = 0, e = BaseList.getNumBases(); i != e; ++i) {
        const CXIdxBaseClassInfo *baseInfo = BaseList.getBases()[i];
        if (baseInfo->base) {
          const NamedDecl *BaseD = BaseList.BaseEntities[i].Dcl;
          SourceLocation
            Loc = SourceLocation::getFromRawEncoding(baseInfo->loc.int_data);
          markEntityOccurrenceInFile(BaseD, Loc);
        }
      }
    }

    return handleDecl(OrigD, OrigD->getLocation(), getCursor(OrigD), CXXDInfo);
  }

  DeclInfo DInfo(/*isRedeclaration=*/!OrigD->isCanonicalDecl(),
                 /*isDefinition=*/RD->isThisDeclarationADefinition(),
                 /*isContainer=*/RD->isThisDeclarationADefinition());
  return handleDecl(OrigD, OrigD->getLocation(), getCursor(OrigD), DInfo);
}

bool IndexingContext::markEntityOccurrenceInFile(const NamedDecl *D,
                                                 SourceLocation Loc) {
  if (!D || Loc.isInvalid())
    return true;

  SourceManager &SM = Ctx->getSourceManager();
  D = getEntityDecl(D);
  
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(SM.getFileLoc(Loc));
  FileID FID = LocInfo.first;
  if (FID.isInvalid())
    return true;
  
  const FileEntry *FE = SM.getFileEntryForID(FID);
  if (!FE)
    return true;
  RefFileOccurrence RefOccur(FE, D);
  std::pair<llvm::DenseSet<RefFileOccurrence>::iterator, bool>
  res = RefFileOccurrences.insert(RefOccur);
  if (!res.second)
    return true; // already in map.

  return false;
}

const NamedDecl *IndexingContext::getEntityDecl(const NamedDecl *D) const {
  assert(D);
  D = cast<NamedDecl>(D->getCanonicalDecl());

  if (const ObjCImplementationDecl *
               ImplD = dyn_cast<ObjCImplementationDecl>(D)) {
    return getEntityDecl(ImplD->getClassInterface());

  } else if (const ObjCCategoryImplDecl *
               CatImplD = dyn_cast<ObjCCategoryImplDecl>(D)) {
    return getEntityDecl(CatImplD->getCategoryDecl());
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FunctionTemplateDecl *TemplD = FD->getDescribedFunctionTemplate())
      return getEntityDecl(TemplD);
  } else if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    if (ClassTemplateDecl *TemplD = RD->getDescribedClassTemplate())
      return getEntityDecl(TemplD);
  }

  return D;
}

const DeclContext *
IndexingContext::getEntityContainer(const Decl *D) const {
  const DeclContext *DC = dyn_cast<DeclContext>(D);
  if (DC)
    return DC;

  if (const ClassTemplateDecl *ClassTempl = dyn_cast<ClassTemplateDecl>(D)) {
    DC = ClassTempl->getTemplatedDecl();
  } else if (const FunctionTemplateDecl *
          FuncTempl = dyn_cast<FunctionTemplateDecl>(D)) {
    DC = FuncTempl->getTemplatedDecl();
  }

  return DC;
}

CXIdxClientContainer
IndexingContext::getClientContainerForDC(const DeclContext *DC) const {
  if (!DC)
    return nullptr;

  ContainerMapTy::const_iterator I = ContainerMap.find(DC);
  if (I == ContainerMap.end())
    return nullptr;

  return I->second;
}

CXIdxClientFile IndexingContext::getIndexFile(const FileEntry *File) {
  if (!File)
    return nullptr;

  FileMapTy::iterator FI = FileMap.find(File);
  if (FI != FileMap.end())
    return FI->second;

  return nullptr;
}

CXIdxLoc IndexingContext::getIndexLoc(SourceLocation Loc) const {
  CXIdxLoc idxLoc =  { {nullptr, nullptr}, 0 };
  if (Loc.isInvalid())
    return idxLoc;

  idxLoc.ptr_data[0] = const_cast<IndexingContext *>(this);
  idxLoc.int_data = Loc.getRawEncoding();
  return idxLoc;
}

void IndexingContext::translateLoc(SourceLocation Loc,
                                   CXIdxClientFile *indexFile, CXFile *file,
                                   unsigned *line, unsigned *column,
                                   unsigned *offset) {
  if (Loc.isInvalid())
    return;

  SourceManager &SM = Ctx->getSourceManager();
  Loc = SM.getFileLoc(Loc);

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
  FileID FID = LocInfo.first;
  unsigned FileOffset = LocInfo.second;

  if (FID.isInvalid())
    return;
  
  const FileEntry *FE = SM.getFileEntryForID(FID);
  if (indexFile)
    *indexFile = getIndexFile(FE);
  if (file)
    *file = const_cast<FileEntry *>(FE);
  if (line)
    *line = SM.getLineNumber(FID, FileOffset);
  if (column)
    *column = SM.getColumnNumber(FID, FileOffset);
  if (offset)
    *offset = FileOffset;
}

void IndexingContext::getEntityInfo(const NamedDecl *D,
                                    EntityInfo &EntityInfo,
                                    ScratchAlloc &SA) {
  if (!D)
    return;

  D = getEntityDecl(D);
  EntityInfo.cursor = getCursor(D);
  EntityInfo.Dcl = D;
  EntityInfo.IndexCtx = this;
  EntityInfo.kind = CXIdxEntity_Unexposed;
  EntityInfo.templateKind = CXIdxEntity_NonTemplate;
  EntityInfo.lang = CXIdxEntityLang_C;

  if (D->hasAttrs()) {
    EntityInfo.AttrList = AttrListInfo::create(D, *this);
    EntityInfo.attributes = EntityInfo.AttrList->getAttrs();
    EntityInfo.numAttributes = EntityInfo.AttrList->getNumAttrs();
  }

  if (const TagDecl *TD = dyn_cast<TagDecl>(D)) {
    switch (TD->getTagKind()) {
    case TTK_Struct:
      EntityInfo.kind = CXIdxEntity_Struct; break;
    case TTK_Union:
      EntityInfo.kind = CXIdxEntity_Union; break;
    case TTK_Class:
      EntityInfo.kind = CXIdxEntity_CXXClass;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    case TTK_Interface:
      EntityInfo.kind = CXIdxEntity_CXXInterface;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    case TTK_Enum:
      EntityInfo.kind = CXIdxEntity_Enum; break;
    }

    if (const CXXRecordDecl *CXXRec = dyn_cast<CXXRecordDecl>(D))
      if (!CXXRec->isCLike())
        EntityInfo.lang = CXIdxEntityLang_CXX;

    if (isa<ClassTemplatePartialSpecializationDecl>(D)) {
      EntityInfo.templateKind = CXIdxEntity_TemplatePartialSpecialization;
    } else if (isa<ClassTemplateSpecializationDecl>(D)) {
      EntityInfo.templateKind = CXIdxEntity_TemplateSpecialization;
    }

  } else {
    switch (D->getKind()) {
    case Decl::Typedef:
      EntityInfo.kind = CXIdxEntity_Typedef; break;
    case Decl::Function:
      EntityInfo.kind = CXIdxEntity_Function;
      break;
    case Decl::ParmVar:
      EntityInfo.kind = CXIdxEntity_Variable;
      break;
    case Decl::Var:
      EntityInfo.kind = CXIdxEntity_Variable;
      if (isa<CXXRecordDecl>(D->getDeclContext())) {
        EntityInfo.kind = CXIdxEntity_CXXStaticVariable;
        EntityInfo.lang = CXIdxEntityLang_CXX;
      }
      break;
    case Decl::Field:
      EntityInfo.kind = CXIdxEntity_Field;
      if (const CXXRecordDecl *
            CXXRec = dyn_cast<CXXRecordDecl>(D->getDeclContext())) {
        // FIXME: isPOD check is not sufficient, a POD can contain methods,
        // we want a isCStructLike check.
        if (!CXXRec->isPOD())
          EntityInfo.lang = CXIdxEntityLang_CXX;
      }
      break;
    case Decl::EnumConstant:
      EntityInfo.kind = CXIdxEntity_EnumConstant; break;
    case Decl::ObjCInterface:
      EntityInfo.kind = CXIdxEntity_ObjCClass;
      EntityInfo.lang = CXIdxEntityLang_ObjC;
      break;
    case Decl::ObjCProtocol:
      EntityInfo.kind = CXIdxEntity_ObjCProtocol;
      EntityInfo.lang = CXIdxEntityLang_ObjC;
      break;
    case Decl::ObjCCategory:
      EntityInfo.kind = CXIdxEntity_ObjCCategory;
      EntityInfo.lang = CXIdxEntityLang_ObjC;
      break;
    case Decl::ObjCMethod:
      if (cast<ObjCMethodDecl>(D)->isInstanceMethod())
        EntityInfo.kind = CXIdxEntity_ObjCInstanceMethod;
      else
        EntityInfo.kind = CXIdxEntity_ObjCClassMethod;
      EntityInfo.lang = CXIdxEntityLang_ObjC;
      break;
    case Decl::ObjCProperty:
      EntityInfo.kind = CXIdxEntity_ObjCProperty;
      EntityInfo.lang = CXIdxEntityLang_ObjC;
      break;
    case Decl::ObjCIvar:
      EntityInfo.kind = CXIdxEntity_ObjCIvar;
      EntityInfo.lang = CXIdxEntityLang_ObjC;
      break;
    case Decl::Namespace:
      EntityInfo.kind = CXIdxEntity_CXXNamespace;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    case Decl::NamespaceAlias:
      EntityInfo.kind = CXIdxEntity_CXXNamespaceAlias;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    case Decl::CXXConstructor:
      EntityInfo.kind = CXIdxEntity_CXXConstructor;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    case Decl::CXXDestructor:
      EntityInfo.kind = CXIdxEntity_CXXDestructor;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    case Decl::CXXConversion:
      EntityInfo.kind = CXIdxEntity_CXXConversionFunction;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    case Decl::CXXMethod: {
      const CXXMethodDecl *MD = cast<CXXMethodDecl>(D);
      if (MD->isStatic())
        EntityInfo.kind = CXIdxEntity_CXXStaticMethod;
      else
        EntityInfo.kind = CXIdxEntity_CXXInstanceMethod;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    }
    case Decl::ClassTemplate:
      EntityInfo.kind = CXIdxEntity_CXXClass;
      EntityInfo.templateKind = CXIdxEntity_Template;
      break;
    case Decl::FunctionTemplate:
      EntityInfo.kind = CXIdxEntity_Function;
      EntityInfo.templateKind = CXIdxEntity_Template;
      if (const CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(
                           cast<FunctionTemplateDecl>(D)->getTemplatedDecl())) {
        if (isa<CXXConstructorDecl>(MD))
          EntityInfo.kind = CXIdxEntity_CXXConstructor;
        else if (isa<CXXDestructorDecl>(MD))
          EntityInfo.kind = CXIdxEntity_CXXDestructor;
        else if (isa<CXXConversionDecl>(MD))
          EntityInfo.kind = CXIdxEntity_CXXConversionFunction;
        else {
          if (MD->isStatic())
            EntityInfo.kind = CXIdxEntity_CXXStaticMethod;
          else
            EntityInfo.kind = CXIdxEntity_CXXInstanceMethod;
        }
      }
      break;
    case Decl::TypeAliasTemplate:
      EntityInfo.kind = CXIdxEntity_CXXTypeAlias;
      EntityInfo.templateKind = CXIdxEntity_Template;
      break;
    case Decl::TypeAlias:
      EntityInfo.kind = CXIdxEntity_CXXTypeAlias;
      EntityInfo.lang = CXIdxEntityLang_CXX;
      break;
    default:
      break;
    }
  }

  if (EntityInfo.kind == CXIdxEntity_Unexposed)
    return;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->getTemplatedKind() ==
          FunctionDecl::TK_FunctionTemplateSpecialization)
      EntityInfo.templateKind = CXIdxEntity_TemplateSpecialization;
  }

  if (EntityInfo.templateKind != CXIdxEntity_NonTemplate)
    EntityInfo.lang = CXIdxEntityLang_CXX;

  if (IdentifierInfo *II = D->getIdentifier()) {
    EntityInfo.name = SA.toCStr(II->getName());

  } else if (isa<TagDecl>(D) || isa<FieldDecl>(D) || isa<NamespaceDecl>(D)) {
    EntityInfo.name = nullptr; // anonymous tag/field/namespace.

  } else {
    SmallString<256> StrBuf;
    {
      llvm::raw_svector_ostream OS(StrBuf);
      D->printName(OS);
    }
    EntityInfo.name = SA.copyCStr(StrBuf.str());
  }

  {
    SmallString<512> StrBuf;
    bool Ignore = getDeclCursorUSR(D, StrBuf);
    if (Ignore) {
      EntityInfo.USR = nullptr;
    } else {
      EntityInfo.USR = SA.copyCStr(StrBuf.str());
    }
  }
}

void IndexingContext::getContainerInfo(const DeclContext *DC,
                                       ContainerInfo &ContInfo) {
  ContInfo.cursor = getCursor(cast<Decl>(DC));
  ContInfo.DC = DC;
  ContInfo.IndexCtx = this;
}

CXCursor IndexingContext::getRefCursor(const NamedDecl *D, SourceLocation Loc) {
  if (const TypeDecl *TD = dyn_cast<TypeDecl>(D))
    return MakeCursorTypeRef(TD, Loc, CXTU);
  if (const ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D))
    return MakeCursorObjCClassRef(ID, Loc, CXTU);
  if (const ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D))
    return MakeCursorObjCProtocolRef(PD, Loc, CXTU);
  if (const TemplateDecl *Template = dyn_cast<TemplateDecl>(D))
    return MakeCursorTemplateRef(Template, Loc, CXTU);
  if (const NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(D))
    return MakeCursorNamespaceRef(Namespace, Loc, CXTU);
  if (const NamespaceAliasDecl *Namespace = dyn_cast<NamespaceAliasDecl>(D))
    return MakeCursorNamespaceRef(Namespace, Loc, CXTU);
  if (const FieldDecl *Field = dyn_cast<FieldDecl>(D))
    return MakeCursorMemberRef(Field, Loc, CXTU);
  if (const VarDecl *Var = dyn_cast<VarDecl>(D))
    return MakeCursorVariableRef(Var, Loc, CXTU);
  
  return clang_getNullCursor();
}

bool IndexingContext::shouldIgnoreIfImplicit(const Decl *D) {
  if (isa<ObjCInterfaceDecl>(D))
    return false;
  if (isa<ObjCCategoryDecl>(D))
    return false;
  if (isa<ObjCIvarDecl>(D))
    return false;
  if (isa<ObjCMethodDecl>(D))
    return false;
  if (isa<ImportDecl>(D))
    return false;
  return true;
}

bool IndexingContext::isTemplateImplicitInstantiation(const Decl *D) {
  if (const ClassTemplateSpecializationDecl *
        SD = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    return SD->getSpecializationKind() == TSK_ImplicitInstantiation;
  }
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    return FD->getTemplateSpecializationKind() == TSK_ImplicitInstantiation;
  }
  return false;
}
