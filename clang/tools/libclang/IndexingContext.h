//===- IndexingContext.h - Higher level API functions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Index_Internal.h"
#include "CXCursor.h"

#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclGroup.h"
#include "llvm/ADT/DenseSet.h"
#include <deque>

namespace clang {
  class FileEntry;
  class ObjCPropertyDecl;
  class ClassTemplateDecl;
  class FunctionTemplateDecl;
  class TypeAliasTemplateDecl;
  class ClassTemplateSpecializationDecl;

namespace cxindex {
  class IndexingContext;
  class AttrListInfo;

class ScratchAlloc {
  IndexingContext &IdxCtx;

public:
  explicit ScratchAlloc(IndexingContext &indexCtx);
  ScratchAlloc(const ScratchAlloc &SA);

  ~ScratchAlloc();

  const char *toCStr(StringRef Str);
  const char *copyCStr(StringRef Str);

  template <typename T>
  T *allocate();
};

struct EntityInfo : public CXIdxEntityInfo {
  const NamedDecl *Dcl;
  IndexingContext *IndexCtx;
  IntrusiveRefCntPtr<AttrListInfo> AttrList;

  EntityInfo() {
    name = USR = 0;
    attributes = 0;
    numAttributes = 0;
  }
};

struct ContainerInfo : public CXIdxContainerInfo {
  const DeclContext *DC;
  IndexingContext *IndexCtx;
};
  
struct DeclInfo : public CXIdxDeclInfo {
  enum DInfoKind {
    Info_Decl,

    Info_ObjCContainer,
      Info_ObjCInterface,
      Info_ObjCProtocol,
      Info_ObjCCategory,

    Info_ObjCProperty,

    Info_CXXClass
  };
  
  DInfoKind Kind;

  EntityInfo EntInfo;
  ContainerInfo SemanticContainer;
  ContainerInfo LexicalContainer;
  ContainerInfo DeclAsContainer;

  DeclInfo(bool isRedeclaration, bool isDefinition, bool isContainer)
    : Kind(Info_Decl) {
    this->isRedeclaration = isRedeclaration;
    this->isDefinition = isDefinition;
    this->isContainer = isContainer;
    attributes = 0;
    numAttributes = 0;
    declAsContainer = semanticContainer = lexicalContainer = 0;
  }
  DeclInfo(DInfoKind K,
           bool isRedeclaration, bool isDefinition, bool isContainer)
    : Kind(K) {
    this->isRedeclaration = isRedeclaration;
    this->isDefinition = isDefinition;
    this->isContainer = isContainer;
    attributes = 0;
    numAttributes = 0;
    declAsContainer = semanticContainer = lexicalContainer = 0;
  }
};

struct ObjCContainerDeclInfo : public DeclInfo {
  CXIdxObjCContainerDeclInfo ObjCContDeclInfo;

  ObjCContainerDeclInfo(bool isForwardRef,
                        bool isRedeclaration,
                        bool isImplementation)
    : DeclInfo(Info_ObjCContainer, isRedeclaration,
               /*isDefinition=*/!isForwardRef, /*isContainer=*/!isForwardRef) {
    init(isForwardRef, isImplementation);
  }
  ObjCContainerDeclInfo(DInfoKind K,
                        bool isForwardRef,
                        bool isRedeclaration,
                        bool isImplementation)
    : DeclInfo(K, isRedeclaration, /*isDefinition=*/!isForwardRef,
               /*isContainer=*/!isForwardRef) {
    init(isForwardRef, isImplementation);
  }

  static bool classof(const DeclInfo *D) {
    return Info_ObjCContainer <= D->Kind && D->Kind <= Info_ObjCCategory;
  }

private:
  void init(bool isForwardRef, bool isImplementation) {
    if (isForwardRef)
      ObjCContDeclInfo.kind = CXIdxObjCContainer_ForwardRef;
    else if (isImplementation)
      ObjCContDeclInfo.kind = CXIdxObjCContainer_Implementation;
    else
      ObjCContDeclInfo.kind = CXIdxObjCContainer_Interface;
  }
};

struct ObjCInterfaceDeclInfo : public ObjCContainerDeclInfo {
  CXIdxObjCInterfaceDeclInfo ObjCInterDeclInfo;
  CXIdxObjCProtocolRefListInfo ObjCProtoListInfo;

  ObjCInterfaceDeclInfo(const ObjCInterfaceDecl *D)
    : ObjCContainerDeclInfo(Info_ObjCInterface,
                            /*isForwardRef=*/false,
                          /*isRedeclaration=*/D->getPreviousDecl() != 0,
                            /*isImplementation=*/false) { }

  static bool classof(const DeclInfo *D) {
    return D->Kind == Info_ObjCInterface;
  }
};

struct ObjCProtocolDeclInfo : public ObjCContainerDeclInfo {
  CXIdxObjCProtocolRefListInfo ObjCProtoRefListInfo;

  ObjCProtocolDeclInfo(const ObjCProtocolDecl *D)
    : ObjCContainerDeclInfo(Info_ObjCProtocol,
                            /*isForwardRef=*/false,
                            /*isRedeclaration=*/D->getPreviousDecl(),
                            /*isImplementation=*/false) { }

  static bool classof(const DeclInfo *D) {
    return D->Kind == Info_ObjCProtocol;
  }
};

struct ObjCCategoryDeclInfo : public ObjCContainerDeclInfo {
  CXIdxObjCCategoryDeclInfo ObjCCatDeclInfo;
  CXIdxObjCProtocolRefListInfo ObjCProtoListInfo;

  explicit ObjCCategoryDeclInfo(bool isImplementation)
    : ObjCContainerDeclInfo(Info_ObjCCategory,
                            /*isForwardRef=*/false,
                            /*isRedeclaration=*/isImplementation,
                            /*isImplementation=*/isImplementation) { }

  static bool classof(const DeclInfo *D) {
    return D->Kind == Info_ObjCCategory;
  }
};

struct ObjCPropertyDeclInfo : public DeclInfo {
  CXIdxObjCPropertyDeclInfo ObjCPropDeclInfo;

  ObjCPropertyDeclInfo()
    : DeclInfo(Info_ObjCProperty,
               /*isRedeclaration=*/false, /*isDefinition=*/false,
               /*isContainer=*/false) { }

  static bool classof(const DeclInfo *D) {
    return D->Kind == Info_ObjCProperty;
  }
};

struct CXXClassDeclInfo : public DeclInfo {
  CXIdxCXXClassDeclInfo CXXClassInfo;

  CXXClassDeclInfo(bool isRedeclaration, bool isDefinition)
    : DeclInfo(Info_CXXClass, isRedeclaration, isDefinition, isDefinition) { }

  static bool classof(const DeclInfo *D) {
    return D->Kind == Info_CXXClass;
  }
};

struct AttrInfo : public CXIdxAttrInfo {
  const Attr *A;

  AttrInfo(CXIdxAttrKind Kind, CXCursor C, CXIdxLoc Loc, const Attr *A) {
    kind = Kind;
    cursor = C;
    loc = Loc;
    this->A = A;
  }
};

struct IBOutletCollectionInfo : public AttrInfo {
  EntityInfo ClassInfo;
  CXIdxIBOutletCollectionAttrInfo IBCollInfo;

  IBOutletCollectionInfo(CXCursor C, CXIdxLoc Loc, const Attr *A) :
    AttrInfo(CXIdxAttr_IBOutletCollection, C, Loc, A) {
    assert(C.kind == CXCursor_IBOutletCollectionAttr);
    IBCollInfo.objcClass = 0;
  }

  IBOutletCollectionInfo(const IBOutletCollectionInfo &other);

  static bool classof(const AttrInfo *A) {
    return A->kind == CXIdxAttr_IBOutletCollection;
  }
};

class AttrListInfo {
  ScratchAlloc SA;

  SmallVector<AttrInfo, 2> Attrs;
  SmallVector<IBOutletCollectionInfo, 2> IBCollAttrs;
  SmallVector<CXIdxAttrInfo *, 2> CXAttrs;
  unsigned ref_cnt;

  AttrListInfo(const AttrListInfo &) LLVM_DELETED_FUNCTION;
  void operator=(const AttrListInfo &) LLVM_DELETED_FUNCTION;
public:
  AttrListInfo(const Decl *D, IndexingContext &IdxCtx);

  static IntrusiveRefCntPtr<AttrListInfo> create(const Decl *D,
                                                 IndexingContext &IdxCtx);

  const CXIdxAttrInfo *const *getAttrs() const {
    if (CXAttrs.empty())
      return 0;
    return CXAttrs.data();
  }
  unsigned getNumAttrs() const { return (unsigned)CXAttrs.size(); }

  /// \brief Retain/Release only useful when we allocate a AttrListInfo from the
  /// BumpPtrAllocator, and not from the stack; so that we keep a pointer
  // in the EntityInfo
  void Retain() { ++ref_cnt; }
  void Release() {
    assert (ref_cnt > 0 && "Reference count is already zero.");
    if (--ref_cnt == 0) {
      // Memory is allocated from a BumpPtrAllocator, no need to delete it.
      this->~AttrListInfo();
    }
  }
};

struct RefFileOccurence {
  const FileEntry *File;
  const Decl *Dcl;

  RefFileOccurence(const FileEntry *File, const Decl *Dcl)
    : File(File), Dcl(Dcl) { }
};

class IndexingContext {
  ASTContext *Ctx;
  CXClientData ClientData;
  IndexerCallbacks &CB;
  unsigned IndexOptions;
  CXTranslationUnit CXTU;
  
  typedef llvm::DenseMap<const FileEntry *, CXIdxClientFile> FileMapTy;
  typedef llvm::DenseMap<const DeclContext *, CXIdxClientContainer>
    ContainerMapTy;
  typedef llvm::DenseMap<const Decl *, CXIdxClientEntity> EntityMapTy;

  FileMapTy FileMap;
  ContainerMapTy ContainerMap;
  EntityMapTy EntityMap;

  llvm::DenseSet<RefFileOccurence> RefFileOccurences;

  std::deque<DeclGroupRef> TUDeclsInObjCContainer;
  
  llvm::BumpPtrAllocator StrScratch;
  unsigned StrAdapterCount;
  friend class ScratchAlloc;

  struct ObjCProtocolListInfo {
    SmallVector<CXIdxObjCProtocolRefInfo, 4> ProtInfos;
    SmallVector<EntityInfo, 4> ProtEntities;
    SmallVector<CXIdxObjCProtocolRefInfo *, 4> Prots;

    CXIdxObjCProtocolRefListInfo getListInfo() const {
      CXIdxObjCProtocolRefListInfo Info = { Prots.data(),
                                            (unsigned)Prots.size() };
      return Info;
    }

    ObjCProtocolListInfo(const ObjCProtocolList &ProtList,
                         IndexingContext &IdxCtx,
                         ScratchAlloc &SA);
  };

  struct CXXBasesListInfo {
    SmallVector<CXIdxBaseClassInfo, 4> BaseInfos;
    SmallVector<EntityInfo, 4> BaseEntities;
    SmallVector<CXIdxBaseClassInfo *, 4> CXBases;

    const CXIdxBaseClassInfo *const *getBases() const {
      return CXBases.data();
    }
    unsigned getNumBases() const { return (unsigned)CXBases.size(); }

    CXXBasesListInfo(const CXXRecordDecl *D,
                     IndexingContext &IdxCtx, ScratchAlloc &SA);

  private:
    SourceLocation getBaseLoc(const CXXBaseSpecifier &Base) const;
  };

  friend class AttrListInfo;

public:
  IndexingContext(CXClientData clientData, IndexerCallbacks &indexCallbacks,
                  unsigned indexOptions, CXTranslationUnit cxTU)
    : Ctx(0), ClientData(clientData), CB(indexCallbacks),
      IndexOptions(indexOptions), CXTU(cxTU),
      StrScratch(/*size=*/1024), StrAdapterCount(0) { }

  ASTContext &getASTContext() const { return *Ctx; }

  void setASTContext(ASTContext &ctx);
  void setPreprocessor(Preprocessor &PP);

  bool shouldSuppressRefs() const {
    return IndexOptions & CXIndexOpt_SuppressRedundantRefs;
  }

  bool shouldIndexFunctionLocalSymbols() const {
    return IndexOptions & CXIndexOpt_IndexFunctionLocalSymbols;
  }

  bool shouldIndexImplicitTemplateInsts() const {
    return IndexOptions & CXIndexOpt_IndexImplicitTemplateInstantiations;
  }

  static bool isFunctionLocalDecl(const Decl *D);

  bool shouldAbort();

  bool hasDiagnosticCallback() const { return CB.diagnostic; }

  void enteredMainFile(const FileEntry *File);

  void ppIncludedFile(SourceLocation hashLoc,
                      StringRef filename, const FileEntry *File,
                      bool isImport, bool isAngled, bool isModuleImport);

  void importedModule(const ImportDecl *ImportD);
  void importedPCH(const FileEntry *File);

  void startedTranslationUnit();

  void indexDecl(const Decl *D);

  void indexTagDecl(const TagDecl *D);

  void indexTypeSourceInfo(TypeSourceInfo *TInfo, const NamedDecl *Parent,
                           const DeclContext *DC = 0);

  void indexTypeLoc(TypeLoc TL, const NamedDecl *Parent,
                    const DeclContext *DC = 0);

  void indexNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
                                   const NamedDecl *Parent,
                                   const DeclContext *DC = 0);

  void indexDeclContext(const DeclContext *DC);
  
  void indexBody(const Stmt *S, const NamedDecl *Parent,
                 const DeclContext *DC = 0);

  void handleDiagnosticSet(CXDiagnosticSet CXDiagSet);

  bool handleFunction(const FunctionDecl *FD);

  bool handleVar(const VarDecl *D);

  bool handleField(const FieldDecl *D);

  bool handleEnumerator(const EnumConstantDecl *D);

  bool handleTagDecl(const TagDecl *D);
  
  bool handleTypedefName(const TypedefNameDecl *D);

  bool handleObjCInterface(const ObjCInterfaceDecl *D);
  bool handleObjCImplementation(const ObjCImplementationDecl *D);

  bool handleObjCProtocol(const ObjCProtocolDecl *D);

  bool handleObjCCategory(const ObjCCategoryDecl *D);
  bool handleObjCCategoryImpl(const ObjCCategoryImplDecl *D);

  bool handleObjCMethod(const ObjCMethodDecl *D);

  bool handleSynthesizedObjCProperty(const ObjCPropertyImplDecl *D);
  bool handleSynthesizedObjCMethod(const ObjCMethodDecl *D, SourceLocation Loc,
                                   const DeclContext *LexicalDC);

  bool handleObjCProperty(const ObjCPropertyDecl *D);

  bool handleNamespace(const NamespaceDecl *D);

  bool handleClassTemplate(const ClassTemplateDecl *D);
  bool handleFunctionTemplate(const FunctionTemplateDecl *D);
  bool handleTypeAliasTemplate(const TypeAliasTemplateDecl *D);

  bool handleReference(const NamedDecl *D, SourceLocation Loc, CXCursor Cursor,
                       const NamedDecl *Parent,
                       const DeclContext *DC,
                       const Expr *E = 0,
                       CXIdxEntityRefKind Kind = CXIdxEntityRef_Direct);

  bool handleReference(const NamedDecl *D, SourceLocation Loc,
                       const NamedDecl *Parent,
                       const DeclContext *DC,
                       const Expr *E = 0,
                       CXIdxEntityRefKind Kind = CXIdxEntityRef_Direct);

  bool isNotFromSourceFile(SourceLocation Loc) const;

  void indexTopLevelDecl(const Decl *D);
  void indexTUDeclsInObjCContainer();
  void indexDeclGroupRef(DeclGroupRef DG);

  void addTUDeclInObjCContainer(DeclGroupRef DG) {
    TUDeclsInObjCContainer.push_back(DG);
  }

  void translateLoc(SourceLocation Loc, CXIdxClientFile *indexFile, CXFile *file,
                    unsigned *line, unsigned *column, unsigned *offset);

  CXIdxClientContainer getClientContainerForDC(const DeclContext *DC) const;
  void addContainerInMap(const DeclContext *DC, CXIdxClientContainer container);

  CXIdxClientEntity getClientEntity(const Decl *D) const;
  void setClientEntity(const Decl *D, CXIdxClientEntity client);

  static bool isTemplateImplicitInstantiation(const Decl *D);

private:
  bool handleDecl(const NamedDecl *D,
                  SourceLocation Loc, CXCursor Cursor,
                  DeclInfo &DInfo,
                  const DeclContext *LexicalDC = 0);

  bool handleObjCContainer(const ObjCContainerDecl *D,
                           SourceLocation Loc, CXCursor Cursor,
                           ObjCContainerDeclInfo &ContDInfo);

  bool handleCXXRecordDecl(const CXXRecordDecl *RD, const NamedDecl *OrigD);

  bool markEntityOccurrenceInFile(const NamedDecl *D, SourceLocation Loc);

  const NamedDecl *getEntityDecl(const NamedDecl *D) const;

  const DeclContext *getEntityContainer(const Decl *D) const;

  CXIdxClientFile getIndexFile(const FileEntry *File);
  
  CXIdxLoc getIndexLoc(SourceLocation Loc) const;

  void getEntityInfo(const NamedDecl *D,
                     EntityInfo &EntityInfo,
                     ScratchAlloc &SA);

  void getContainerInfo(const DeclContext *DC, ContainerInfo &ContInfo);

  CXCursor getCursor(const Decl *D) {
    return cxcursor::MakeCXCursor(const_cast<Decl*>(D), CXTU);
  }

  CXCursor getRefCursor(const NamedDecl *D, SourceLocation Loc);

  static bool shouldIgnoreIfImplicit(const Decl *D);
};

inline ScratchAlloc::ScratchAlloc(IndexingContext &idxCtx) : IdxCtx(idxCtx) {
  ++IdxCtx.StrAdapterCount;
}
inline ScratchAlloc::ScratchAlloc(const ScratchAlloc &SA) : IdxCtx(SA.IdxCtx) {
  ++IdxCtx.StrAdapterCount;
}

inline ScratchAlloc::~ScratchAlloc() {
  --IdxCtx.StrAdapterCount;
  if (IdxCtx.StrAdapterCount == 0)
    IdxCtx.StrScratch.Reset();
}

template <typename T>
inline T *ScratchAlloc::allocate() {
  return IdxCtx.StrScratch.Allocate<T>();
}

}} // end clang::cxindex

namespace llvm {
  /// Define DenseMapInfo so that FileID's can be used as keys in DenseMap and
  /// DenseSets.
  template <>
  struct DenseMapInfo<clang::cxindex::RefFileOccurence> {
    static inline clang::cxindex::RefFileOccurence getEmptyKey() {
      return clang::cxindex::RefFileOccurence(0, 0);
    }

    static inline clang::cxindex::RefFileOccurence getTombstoneKey() {
      return clang::cxindex::RefFileOccurence((const clang::FileEntry *)~0,
                                              (const clang::Decl *)~0);
    }

    static unsigned getHashValue(clang::cxindex::RefFileOccurence S) {
      typedef std::pair<const clang::FileEntry *, const clang::Decl *> PairTy;
      return DenseMapInfo<PairTy>::getHashValue(PairTy(S.File, S.Dcl));
    }

    static bool isEqual(clang::cxindex::RefFileOccurence LHS,
                        clang::cxindex::RefFileOccurence RHS) {
      return LHS.File == RHS.File && LHS.Dcl == RHS.Dcl;
    }
  };
}
