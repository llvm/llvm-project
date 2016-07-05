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

#include "clang/Serialization/ASTReader.h"
#include "ASTCommon.h"
#include "ASTReaderInternals.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Sema/SemaDiagnostic.h"
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
    uint64_t Offset;
    const DeclID ThisDeclID;
    const SourceLocation ThisDeclLoc;
    typedef ASTReader::RecordData RecordData;
    const RecordData &Record;
    unsigned &Idx;
    TypeID TypeIDForTypeDecl;
    unsigned AnonymousDeclNumber;
    GlobalDeclID NamedDeclForTagDecl;
    IdentifierInfo *TypedefNameForLinkage;

    bool HasPendingBody;

    ///\brief A flag to carry the information for a decl from the entity is
    /// used. We use it to delay the marking of the canonical decl as used until
    /// the entire declaration is deserialized and merged.
    bool IsDeclMarkedUsed;

    uint64_t GetCurrentCursorOffset();

    uint64_t ReadLocalOffset(const RecordData &R, unsigned &I) {
      uint64_t LocalOffset = R[I++];
      assert(LocalOffset < Offset && "offset point after current record");
      return LocalOffset ? Offset - LocalOffset : 0;
    }

    uint64_t ReadGlobalOffset(ModuleFile &F, const RecordData &R, unsigned &I) {
      uint64_t Local = ReadLocalOffset(R, I);
      return Local ? Reader.getGlobalBitOffset(F, Local) : 0;
    }

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

    std::string ReadString(const RecordData &R, unsigned &I) {
      return Reader.ReadString(R, I);
    }

    void ReadDeclIDList(SmallVectorImpl<DeclID> &IDs) {
      for (unsigned I = 0, Size = Record[Idx++]; I != Size; ++I)
        IDs.push_back(ReadDeclID(Record, Idx));
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

    void ReadCXXRecordDefinition(CXXRecordDecl *D, bool Update);
    void ReadCXXDefinitionData(struct CXXRecordDecl::DefinitionData &Data,
                               const RecordData &R, unsigned &I);
    void MergeDefinitionData(CXXRecordDecl *D,
                             struct CXXRecordDecl::DefinitionData &&NewDD);

    static NamedDecl *getAnonymousDeclForMerging(ASTReader &Reader,
                                                 DeclContext *DC,
                                                 unsigned Index);
    static void setAnonymousDeclForMerging(ASTReader &Reader, DeclContext *DC,
                                           unsigned Index, NamedDecl *D);

    /// Results from loading a RedeclarableDecl.
    class RedeclarableResult {
      GlobalDeclID FirstID;
      Decl *MergeWith;
      bool IsKeyDecl;

    public:
      RedeclarableResult(GlobalDeclID FirstID, Decl *MergeWith, bool IsKeyDecl)
          : FirstID(FirstID), MergeWith(MergeWith), IsKeyDecl(IsKeyDecl) {}

      /// \brief Retrieve the first ID.
      GlobalDeclID getFirstID() const { return FirstID; }

      /// \brief Is this declaration a key declaration?
      bool isKeyDecl() const { return IsKeyDecl; }

      /// \brief Get a known declaration that this should be merged with, if
      /// any.
      Decl *getKnownMergeTarget() const { return MergeWith; }
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

      unsigned AnonymousDeclNumber;
      IdentifierInfo *TypedefNameForLinkage;

      void operator=(FindExistingResult&) = delete;

    public:
      FindExistingResult(ASTReader &Reader)
          : Reader(Reader), New(nullptr), Existing(nullptr), AddResult(false),
            AnonymousDeclNumber(0), TypedefNameForLinkage(nullptr) {}

      FindExistingResult(ASTReader &Reader, NamedDecl *New, NamedDecl *Existing,
                         unsigned AnonymousDeclNumber,
                         IdentifierInfo *TypedefNameForLinkage)
          : Reader(Reader), New(New), Existing(Existing), AddResult(true),
            AnonymousDeclNumber(AnonymousDeclNumber),
            TypedefNameForLinkage(TypedefNameForLinkage) {}

      FindExistingResult(const FindExistingResult &Other)
          : Reader(Other.Reader), New(Other.New), Existing(Other.Existing),
            AddResult(Other.AddResult),
            AnonymousDeclNumber(Other.AnonymousDeclNumber),
            TypedefNameForLinkage(Other.TypedefNameForLinkage) {
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

    static DeclContext *getPrimaryContextForMerging(ASTReader &Reader,
                                                    DeclContext *DC);
    FindExistingResult findExisting(NamedDecl *D);

  public:
    ASTDeclReader(ASTReader &Reader, ASTReader::RecordLocation Loc,
                  DeclID thisDeclID, SourceLocation ThisDeclLoc,
                  const RecordData &Record, unsigned &Idx)
        : Reader(Reader), F(*Loc.F), Offset(Loc.Offset), ThisDeclID(thisDeclID),
          ThisDeclLoc(ThisDeclLoc), Record(Record), Idx(Idx),
          TypeIDForTypeDecl(0), NamedDeclForTagDecl(0),
          TypedefNameForLinkage(nullptr), HasPendingBody(false),
          IsDeclMarkedUsed(false) {}

    template <typename DeclT>
    static Decl *getMostRecentDeclImpl(Redeclarable<DeclT> *D);
    static Decl *getMostRecentDeclImpl(...);
    static Decl *getMostRecentDecl(Decl *D);

    template <typename DeclT>
    static void attachPreviousDeclImpl(ASTReader &Reader,
                                       Redeclarable<DeclT> *D, Decl *Previous,
                                       Decl *Canon);
    static void attachPreviousDeclImpl(ASTReader &Reader, ...);
    static void attachPreviousDecl(ASTReader &Reader, Decl *D, Decl *Previous,
                                   Decl *Canon);

    template <typename DeclT>
    static void attachLatestDeclImpl(Redeclarable<DeclT> *D, Decl *Latest);
    static void attachLatestDeclImpl(...);
    static void attachLatestDecl(Decl *D, Decl *latest);

    template <typename DeclT>
    static void markIncompleteDeclChainImpl(Redeclarable<DeclT> *D);
    static void markIncompleteDeclChainImpl(...);

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
    void VisitPragmaCommentDecl(PragmaCommentDecl *D);
    void VisitPragmaDetectMismatchDecl(PragmaDetectMismatchDecl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitLabelDecl(LabelDecl *LD);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeDecl(TypeDecl *TD);
    RedeclarableResult VisitTypedefNameDecl(TypedefNameDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
    void VisitTypeAliasDecl(TypeAliasDecl *TD);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    RedeclarableResult VisitTagDecl(TagDecl *TD);
    void VisitEnumDecl(EnumDecl *ED);
    RedeclarableResult VisitRecordDeclImpl(RecordDecl *RD);
    void VisitRecordDecl(RecordDecl *RD) { VisitRecordDeclImpl(RD); }
    RedeclarableResult VisitCXXRecordDeclImpl(CXXRecordDecl *D);
    void VisitCXXRecordDecl(CXXRecordDecl *D) { VisitCXXRecordDeclImpl(D); }
    RedeclarableResult VisitClassTemplateSpecializationDeclImpl(
                                            ClassTemplateSpecializationDecl *D);
    void VisitClassTemplateSpecializationDecl(
        ClassTemplateSpecializationDecl *D) {
      VisitClassTemplateSpecializationDeclImpl(D);
    }
    void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
                                       ClassScopeFunctionSpecializationDecl *D);
    RedeclarableResult
    VisitVarTemplateSpecializationDeclImpl(VarTemplateSpecializationDecl *D);
    void VisitVarTemplateSpecializationDecl(VarTemplateSpecializationDecl *D) {
      VisitVarTemplateSpecializationDeclImpl(D);
    }
    void VisitVarTemplatePartialSpecializationDecl(
        VarTemplatePartialSpecializationDecl *D);
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
    void VisitMSPropertyDecl(MSPropertyDecl *FD);
    void VisitIndirectFieldDecl(IndirectFieldDecl *FD);
    RedeclarableResult VisitVarDeclImpl(VarDecl *D);
    void VisitVarDecl(VarDecl *VD) { VisitVarDeclImpl(VD); }
    void VisitImplicitParamDecl(ImplicitParamDecl *PD);
    void VisitParmVarDecl(ParmVarDecl *PD);
    void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    DeclID VisitTemplateDecl(TemplateDecl *D);
    RedeclarableResult VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitBuiltinTemplateDecl(BuiltinTemplateDecl *D);
    void VisitVarTemplateDecl(VarTemplateDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    void VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);
    void VisitConstructorUsingShadowDecl(ConstructorUsingShadowDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *AD);
    void VisitImportDecl(ImportDecl *D);
    void VisitAccessSpecDecl(AccessSpecDecl *D);
    void VisitFriendDecl(FriendDecl *D);
    void VisitFriendTemplateDecl(FriendTemplateDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitBlockDecl(BlockDecl *BD);
    void VisitCapturedDecl(CapturedDecl *CD);
    void VisitEmptyDecl(EmptyDecl *D);

    std::pair<uint64_t, uint64_t> VisitDeclContext(DeclContext *DC);

    template<typename T>
    RedeclarableResult VisitRedeclarable(Redeclarable<T> *D);

    template<typename T>
    void mergeRedeclarable(Redeclarable<T> *D, RedeclarableResult &Redecl,
                           DeclID TemplatePatternID = 0);

    template<typename T>
    void mergeRedeclarable(Redeclarable<T> *D, T *Existing,
                           RedeclarableResult &Redecl,
                           DeclID TemplatePatternID = 0);

    template<typename T>
    void mergeMergeable(Mergeable<T> *D);

    void mergeTemplatePattern(RedeclarableTemplateDecl *D,
                              RedeclarableTemplateDecl *Existing,
                              DeclID DsID, bool IsKeyDecl);

    ObjCTypeParamList *ReadObjCTypeParamList();

    // FIXME: Reorder according to DeclNodes.td?
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCTypeParamDecl(ObjCTypeParamDecl *D);
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
    void VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D);
    void VisitOMPDeclareReductionDecl(OMPDeclareReductionDecl *D);
    void VisitOMPCapturedExprDecl(OMPCapturedExprDecl *D);

    /// We've merged the definition \p MergedDef into the existing definition
    /// \p Def. Ensure that \p Def is made visible whenever \p MergedDef is made
    /// visible.
    void mergeDefinitionVisibility(NamedDecl *Def, NamedDecl *MergedDef) {
      if (Def->isHidden()) {
        // If MergedDef is visible or becomes visible, make the definition visible.
        if (!MergedDef->isHidden())
          Def->Hidden = false;
        else if (Reader.getContext().getLangOpts().ModulesLocalVisibility) {
          Reader.getContext().mergeDefinitionIntoModule(
              Def, MergedDef->getImportedOwningModule(),
              /*NotifyListeners*/ false);
          Reader.PendingMergedDefinitionsToDeduplicate.insert(Def);
        } else {
          auto SubmoduleID = MergedDef->getOwningModuleID();
          assert(SubmoduleID && "hidden definition in no module");
          Reader.HiddenNamesMap[Reader.getSubmodule(SubmoduleID)].push_back(Def);
        }
      }
    }
  };
} // end namespace clang

namespace {
/// Iterator over the redeclarations of a declaration that have already
/// been merged into the same redeclaration chain.
template<typename DeclT>
class MergedRedeclIterator {
  DeclT *Start, *Canonical, *Current;
public:
  MergedRedeclIterator() : Current(nullptr) {}
  MergedRedeclIterator(DeclT *Start)
      : Start(Start), Canonical(nullptr), Current(Start) {}

  DeclT *operator*() { return Current; }

  MergedRedeclIterator &operator++() {
    if (Current->isFirstDecl()) {
      Canonical = Current;
      Current = Current->getMostRecentDecl();
    } else
      Current = Current->getPreviousDecl();

    // If we started in the merged portion, we'll reach our start position
    // eventually. Otherwise, we'll never reach it, but the second declaration
    // we reached was the canonical declaration, so stop when we see that one
    // again.
    if (Current == Start || Current == Canonical)
      Current = nullptr;
    return *this;
  }

  friend bool operator!=(const MergedRedeclIterator &A,
                         const MergedRedeclIterator &B) {
    return A.Current != B.Current;
  }
};
} // end anonymous namespace

template<typename DeclT>
llvm::iterator_range<MergedRedeclIterator<DeclT>> merged_redecls(DeclT *D) {
  return llvm::make_range(MergedRedeclIterator<DeclT>(D),
                          MergedRedeclIterator<DeclT>());
}

uint64_t ASTDeclReader::GetCurrentCursorOffset() {
  return F.DeclsCursor.GetCurrentBitNo() + F.GlobalBitOffset;
}

void ASTDeclReader::Visit(Decl *D) {
  DeclVisitor<ASTDeclReader, void>::Visit(D);

  // At this point we have deserialized and merged the decl and it is safe to
  // update its canonical decl to signal that the entire entity is used.
  D->getCanonicalDecl()->Used |= IsDeclMarkedUsed;
  IsDeclMarkedUsed = false;

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
    // We have a fully initialized TypeDecl. Read its type now.
    TD->setTypeForDecl(Reader.GetType(TypeIDForTypeDecl).getTypePtrOrNull());

    // If this is a tag declaration with a typedef name for linkage, it's safe
    // to load that typedef now.
    if (NamedDeclForTagDecl)
      cast<TagDecl>(D)->TypedefNameDeclOrQualifier =
          cast<TypedefNameDecl>(Reader.GetDecl(NamedDeclForTagDecl));
  } else if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    // if we have a fully initialized TypeDecl, we can safely read its type now.
    ID->TypeForDecl = Reader.GetType(TypeIDForTypeDecl).getTypePtrOrNull();
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // FunctionDecl's body was written last after all other Stmts/Exprs.
    // We only read it if FD doesn't already have a body (e.g., from another
    // module).
    // FIXME: Can we diagnose ODR violations somehow?
    if (Record[Idx++]) {
      if (auto *CD = dyn_cast<CXXConstructorDecl>(FD)) {
        CD->NumCtorInitializers = Record[Idx++];
        if (CD->NumCtorInitializers)
          CD->CtorInitializers = ReadGlobalOffset(F, Record, Idx);
      }
      Reader.PendingBodies[FD] = GetCurrentCursorOffset();
      HasPendingBody = true;
    }
  }
}

void ASTDeclReader::VisitDecl(Decl *D) {
  if (D->isTemplateParameter() || D->isTemplateParameterPack() ||
      isa<ParmVarDecl>(D)) {
    // We don't want to deserialize the DeclContext of a template
    // parameter or of a parameter of a function template immediately.   These
    // entities might be used in the formulation of its DeclContext (for
    // example, a function parameter can be used in decltype() in trailing
    // return type of the function).  Use the translation unit DeclContext as a
    // placeholder.
    GlobalDeclID SemaDCIDForTemplateParmDecl = ReadDeclID(Record, Idx);
    GlobalDeclID LexicalDCIDForTemplateParmDecl = ReadDeclID(Record, Idx);
    if (!LexicalDCIDForTemplateParmDecl)
      LexicalDCIDForTemplateParmDecl = SemaDCIDForTemplateParmDecl;
    Reader.addPendingDeclContextInfo(D,
                                     SemaDCIDForTemplateParmDecl,
                                     LexicalDCIDForTemplateParmDecl);
    D->setDeclContext(Reader.getContext().getTranslationUnitDecl()); 
  } else {
    DeclContext *SemaDC = ReadDeclAs<DeclContext>(Record, Idx);
    DeclContext *LexicalDC = ReadDeclAs<DeclContext>(Record, Idx);
    if (!LexicalDC)
      LexicalDC = SemaDC;
    DeclContext *MergedSemaDC = Reader.MergedDeclContexts.lookup(SemaDC);
    // Avoid calling setLexicalDeclContext() directly because it uses
    // Decl::getASTContext() internally which is unsafe during derialization.
    D->setDeclContextsImpl(MergedSemaDC ? MergedSemaDC : SemaDC, LexicalDC,
                           Reader.getContext());
  }
  D->setLocation(ThisDeclLoc);
  D->setInvalidDecl(Record[Idx++]);
  if (Record[Idx++]) { // hasAttrs
    AttrVec Attrs;
    Reader.ReadAttributes(F, Attrs, Record, Idx);
    // Avoid calling setAttrs() directly because it uses Decl::getASTContext()
    // internally which is unsafe during derialization.
    D->setAttrsImpl(Attrs, Reader.getContext());
  }
  D->setImplicit(Record[Idx++]);
  D->Used = Record[Idx++];
  IsDeclMarkedUsed |= D->Used;
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

    if (D->Hidden) {
      // Module-private declarations are never visible, so there is no work to do.
    } else if (Reader.getContext().getLangOpts().ModulesLocalVisibility) {
      // If local visibility is being tracked, this declaration will become
      // hidden and visible as the owning module does. Inform Sema that this
      // declaration might not be visible.
      D->Hidden = true;
    } else if (Module *Owner = Reader.getSubmodule(SubmoduleID)) {
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

void ASTDeclReader::VisitPragmaCommentDecl(PragmaCommentDecl *D) {
  VisitDecl(D);
  D->setLocation(ReadSourceLocation(Record, Idx));
  D->CommentKind = (PragmaMSCommentKind)Record[Idx++];
  std::string Arg = ReadString(Record, Idx);
  memcpy(D->getTrailingObjects<char>(), Arg.data(), Arg.size());
  D->getTrailingObjects<char>()[Arg.size()] = '\0';
}

void ASTDeclReader::VisitPragmaDetectMismatchDecl(PragmaDetectMismatchDecl *D) {
  VisitDecl(D);
  D->setLocation(ReadSourceLocation(Record, Idx));
  std::string Name = ReadString(Record, Idx);
  memcpy(D->getTrailingObjects<char>(), Name.data(), Name.size());
  D->getTrailingObjects<char>()[Name.size()] = '\0';

  D->ValueStart = Name.size() + 1;
  std::string Value = ReadString(Record, Idx);
  memcpy(D->getTrailingObjects<char>() + D->ValueStart, Value.data(),
         Value.size());
  D->getTrailingObjects<char>()[D->ValueStart + Value.size()] = '\0';
}

void ASTDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  llvm_unreachable("Translation units are not serialized");
}

void ASTDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(F, Record, Idx));
  AnonymousDeclNumber = Record[Idx++];
}

void ASTDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  TD->setLocStart(ReadSourceLocation(Record, Idx));
  // Delay type reading until after we have fully initialized the decl.
  TypeIDForTypeDecl = Reader.getGlobalTypeID(F, Record[Idx++]);
}

ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitTypedefNameDecl(TypedefNameDecl *TD) {
  RedeclarableResult Redecl = VisitRedeclarable(TD);
  VisitTypeDecl(TD);
  TypeSourceInfo *TInfo = GetTypeSourceInfo(Record, Idx);
  if (Record[Idx++]) { // isModed
    QualType modedT = Reader.readType(F, Record, Idx);
    TD->setModedTypeSourceInfo(TInfo, modedT);
  } else
    TD->setTypeSourceInfo(TInfo);
  return Redecl;
}

void ASTDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  RedeclarableResult Redecl = VisitTypedefNameDecl(TD);
  mergeRedeclarable(TD, Redecl);
}

void ASTDeclReader::VisitTypeAliasDecl(TypeAliasDecl *TD) {
  RedeclarableResult Redecl = VisitTypedefNameDecl(TD);
  if (auto *Template = ReadDeclAs<TypeAliasTemplateDecl>(Record, Idx))
    // Merged when we merge the template.
    TD->setDescribedAliasTemplate(Template);
  else
    mergeRedeclarable(TD, Redecl);
}

ASTDeclReader::RedeclarableResult ASTDeclReader::VisitTagDecl(TagDecl *TD) {
  RedeclarableResult Redecl = VisitRedeclarable(TD);
  VisitTypeDecl(TD);
  
  TD->IdentifierNamespace = Record[Idx++];
  TD->setTagKind((TagDecl::TagKind)Record[Idx++]);
  if (!isa<CXXRecordDecl>(TD))
    TD->setCompleteDefinition(Record[Idx++]);
  TD->setEmbeddedInDeclarator(Record[Idx++]);
  TD->setFreeStanding(Record[Idx++]);
  TD->setCompleteDefinitionRequired(Record[Idx++]);
  TD->setRBraceLoc(ReadSourceLocation(Record, Idx));
  
  switch (Record[Idx++]) {
  case 0:
    break;
  case 1: { // ExtInfo
    TagDecl::ExtInfo *Info = new (Reader.getContext()) TagDecl::ExtInfo();
    ReadQualifierInfo(*Info, Record, Idx);
    TD->TypedefNameDeclOrQualifier = Info;
    break;
  }
  case 2: // TypedefNameForAnonDecl
    NamedDeclForTagDecl = ReadDeclID(Record, Idx);
    TypedefNameForLinkage = Reader.GetIdentifierInfo(F, Record, Idx);
    break;
  default:
    llvm_unreachable("unexpected tag info kind");
  }

  if (!isa<CXXRecordDecl>(TD))
    mergeRedeclarable(TD, Redecl);
  return Redecl;
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

  // If this is a definition subject to the ODR, and we already have a
  // definition, merge this one into it.
  if (ED->IsCompleteDefinition &&
      Reader.getContext().getLangOpts().Modules &&
      Reader.getContext().getLangOpts().CPlusPlus) {
    EnumDecl *&OldDef = Reader.EnumDefinitions[ED->getCanonicalDecl()];
    if (!OldDef) {
      // This is the first time we've seen an imported definition. Look for a
      // local definition before deciding that we are the first definition.
      for (auto *D : merged_redecls(ED->getCanonicalDecl())) {
        if (!D->isFromASTFile() && D->isCompleteDefinition()) {
          OldDef = D;
          break;
        }
      }
    }
    if (OldDef) {
      Reader.MergedDeclContexts.insert(std::make_pair(ED, OldDef));
      ED->IsCompleteDefinition = false;
      mergeDefinitionVisibility(OldDef, ED);
    } else {
      OldDef = ED;
    }
  }

  if (EnumDecl *InstED = ReadDeclAs<EnumDecl>(Record, Idx)) {
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    ED->setInstantiationOfMemberEnum(Reader.getContext(), InstED, TSK);
    ED->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
  }
}

ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitRecordDeclImpl(RecordDecl *RD) {
  RedeclarableResult Redecl = VisitTagDecl(RD);
  RD->setHasFlexibleArrayMember(Record[Idx++]);
  RD->setAnonymousStructOrUnion(Record[Idx++]);
  RD->setHasObjectMember(Record[Idx++]);
  RD->setHasVolatileMember(Record[Idx++]);
  return Redecl;
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
  mergeMergeable(ECD);
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
  FD->HasSkippedBody = Record[Idx++];
  FD->IsLateTemplateParsed = Record[Idx++];
  FD->setCachedLinkage(Linkage(Record[Idx++]));
  FD->EndRangeLoc = ReadSourceLocation(Record, Idx);

  switch ((FunctionDecl::TemplatedKind)Record[Idx++]) {
  case FunctionDecl::TK_NonTemplate:
    mergeRedeclarable(FD, Redecl);
    break;
  case FunctionDecl::TK_FunctionTemplate:
    // Merged when we merge the template.
    FD->setDescribedFunctionTemplate(ReadDeclAs<FunctionTemplateDecl>(Record, 
                                                                      Idx));
    break;
  case FunctionDecl::TK_MemberSpecialization: {
    FunctionDecl *InstFD = ReadDeclAs<FunctionDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    FD->setInstantiationOfMemberFunction(Reader.getContext(), InstFD, TSK);
    FD->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
    mergeRedeclarable(FD, Redecl);
    break;
  }
  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    FunctionTemplateDecl *Template = ReadDeclAs<FunctionTemplateDecl>(Record, 
                                                                      Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    
    // Template arguments.
    SmallVector<TemplateArgument, 8> TemplArgs;
    Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx,
                                    /*Canonicalize*/ true);

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
      = TemplateArgumentList::CreateCopy(C, TemplArgs);
    TemplateArgumentListInfo TemplArgsInfo(LAngleLoc, RAngleLoc);
    for (unsigned i=0, e = TemplArgLocs.size(); i != e; ++i)
      TemplArgsInfo.addArgument(TemplArgLocs[i]);
    FunctionTemplateSpecializationInfo *FTInfo
        = FunctionTemplateSpecializationInfo::Create(C, FD, Template, TSK,
                                                     TemplArgList,
                             HasTemplateArgumentsAsWritten ? &TemplArgsInfo
                                                           : nullptr,
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
      FunctionTemplateSpecializationInfo::Profile(ID, TemplArgs, C);
      void *InsertPos = nullptr;
      FunctionTemplateDecl::Common *CommonPtr = CanonTemplate->getCommonPtr();
      FunctionTemplateSpecializationInfo *ExistingInfo =
          CommonPtr->Specializations.FindNodeOrInsertPos(ID, InsertPos);
      if (InsertPos)
        CommonPtr->Specializations.InsertNode(FTInfo, InsertPos);
      else {
        assert(Reader.getContext().getLangOpts().Modules &&
               "already deserialized this template specialization");
        mergeRedeclarable(FD, ExistingInfo->Function, Redecl);
      }
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
    // These are not merged; we don't need to merge redeclarations of dependent
    // template friends.
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
  MD->HasSkippedBody = Record[Idx++];

  MD->IsRedeclaration = Record[Idx++];
  MD->HasRedeclaration = Record[Idx++];
  if (MD->HasRedeclaration)
    Reader.getContext().setObjCMethodRedeclaration(MD,
                                       ReadDeclAs<ObjCMethodDecl>(Record, Idx));

  MD->setDeclImplementation((ObjCMethodDecl::ImplementationControl)Record[Idx++]);
  MD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  MD->SetRelatedResultType(Record[Idx++]);
  MD->setReturnType(Reader.readType(F, Record, Idx));
  MD->setReturnTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
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

void ASTDeclReader::VisitObjCTypeParamDecl(ObjCTypeParamDecl *D) {
  VisitTypedefNameDecl(D);

  D->Variance = Record[Idx++];
  D->Index = Record[Idx++];
  D->VarianceLoc = ReadSourceLocation(Record, Idx);
  D->ColonLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitObjCContainerDecl(ObjCContainerDecl *CD) {
  VisitNamedDecl(CD);
  CD->setAtStartLoc(ReadSourceLocation(Record, Idx));
  CD->setAtEndRange(ReadSourceRange(Record, Idx));
}

ObjCTypeParamList *ASTDeclReader::ReadObjCTypeParamList() {
  unsigned numParams = Record[Idx++];
  if (numParams == 0)
    return nullptr;

  SmallVector<ObjCTypeParamDecl *, 4> typeParams;
  typeParams.reserve(numParams);
  for (unsigned i = 0; i != numParams; ++i) {
    auto typeParam = ReadDeclAs<ObjCTypeParamDecl>(Record, Idx);
    if (!typeParam)
      return nullptr;

    typeParams.push_back(typeParam);
  }

  SourceLocation lAngleLoc = ReadSourceLocation(Record, Idx);
  SourceLocation rAngleLoc = ReadSourceLocation(Record, Idx);

  return ObjCTypeParamList::create(Reader.getContext(), lAngleLoc,
                                   typeParams, rAngleLoc);
}

void ASTDeclReader::VisitObjCInterfaceDecl(ObjCInterfaceDecl *ID) {
  RedeclarableResult Redecl = VisitRedeclarable(ID);
  VisitObjCContainerDecl(ID);
  TypeIDForTypeDecl = Reader.getGlobalTypeID(F, Record[Idx++]);
  mergeRedeclarable(ID, Redecl);

  ID->TypeParamList = ReadObjCTypeParamList();
  if (Record[Idx++]) {
    // Read the definition.
    ID->allocateDefinitionData();
    
    // Set the definition data of the canonical declaration, so other
    // redeclarations will see it.
    ID->getCanonicalDecl()->Data = ID->Data;
    
    ObjCInterfaceDecl::DefinitionData &Data = ID->data();
    
    // Read the superclass.
    Data.SuperClassTInfo = GetTypeSourceInfo(Record, Idx);

    Data.EndLoc = ReadSourceLocation(Record, Idx);
    Data.HasDesignatedInitializers = Record[Idx++];
    
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
    ID->setIvarList(nullptr);

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
  IVD->setNextIvar(nullptr);
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
  CD->TypeParamList = ReadObjCTypeParamList();
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
  QualType T = Reader.readType(F, Record, Idx);
  TypeSourceInfo *TSI = GetTypeSourceInfo(Record, Idx);
  D->setType(T, TSI);
  D->setPropertyAttributes(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
  D->setPropertyAttributesAsWritten(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
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
  D->SuperLoc = ReadSourceLocation(Record, Idx);
  D->setIvarLBraceLoc(ReadSourceLocation(Record, Idx));
  D->setIvarRBraceLoc(ReadSourceLocation(Record, Idx));
  D->setHasNonZeroConstructors(Record[Idx++]);
  D->setHasDestructors(Record[Idx++]);
  D->NumIvarInitializers = Record[Idx++];
  if (D->NumIvarInitializers)
    D->IvarInitializers = ReadGlobalOffset(F, Record, Idx);
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
    FD->InitStorage.setInt(
          static_cast<FieldDecl::InitStorageKind>(BitWidthOrInitializer - 1));
    if (FD->InitStorage.getInt() == FieldDecl::ISK_CapturedVLAType) {
      // Read captured variable length array.
      FD->InitStorage.setPointer(
          Reader.readType(F, Record, Idx).getAsOpaquePtr());
    } else {
      FD->InitStorage.setPointer(Reader.ReadExpr(F));
    }
  }
  if (!FD->getDeclName()) {
    if (FieldDecl *Tmpl = ReadDeclAs<FieldDecl>(Record, Idx))
      Reader.getContext().setInstantiatedFromUnnamedFieldDecl(FD, Tmpl);
  }
  mergeMergeable(FD);
}

void ASTDeclReader::VisitMSPropertyDecl(MSPropertyDecl *PD) {
  VisitDeclaratorDecl(PD);
  PD->GetterId = Reader.GetIdentifierInfo(F, Record, Idx);
  PD->SetterId = Reader.GetIdentifierInfo(F, Record, Idx);
}

void ASTDeclReader::VisitIndirectFieldDecl(IndirectFieldDecl *FD) {
  VisitValueDecl(FD);

  FD->ChainingSize = Record[Idx++];
  assert(FD->ChainingSize >= 2 && "Anonymous chaining must be >= 2");
  FD->Chaining = new (Reader.getContext())NamedDecl*[FD->ChainingSize];

  for (unsigned I = 0; I != FD->ChainingSize; ++I)
    FD->Chaining[I] = ReadDeclAs<NamedDecl>(Record, Idx);

  mergeMergeable(FD);
}

ASTDeclReader::RedeclarableResult ASTDeclReader::VisitVarDeclImpl(VarDecl *VD) {
  RedeclarableResult Redecl = VisitRedeclarable(VD);
  VisitDeclaratorDecl(VD);

  VD->VarDeclBits.SClass = (StorageClass)Record[Idx++];
  VD->VarDeclBits.TSCSpec = Record[Idx++];
  VD->VarDeclBits.InitStyle = Record[Idx++];
  if (!isa<ParmVarDecl>(VD)) {
    VD->NonParmVarDeclBits.ExceptionVar = Record[Idx++];
    VD->NonParmVarDeclBits.NRVOVariable = Record[Idx++];
    VD->NonParmVarDeclBits.CXXForRangeDecl = Record[Idx++];
    VD->NonParmVarDeclBits.ARCPseudoStrong = Record[Idx++];
    VD->NonParmVarDeclBits.IsInline = Record[Idx++];
    VD->NonParmVarDeclBits.IsInlineSpecified = Record[Idx++];
    VD->NonParmVarDeclBits.IsConstexpr = Record[Idx++];
    VD->NonParmVarDeclBits.IsInitCapture = Record[Idx++];
    VD->NonParmVarDeclBits.PreviousDeclInSameBlockScope = Record[Idx++];
  }
  Linkage VarLinkage = Linkage(Record[Idx++]);
  VD->setCachedLinkage(VarLinkage);

  // Reconstruct the one piece of the IdentifierNamespace that we need.
  if (VD->getStorageClass() == SC_Extern && VarLinkage != NoLinkage &&
      VD->getLexicalDeclContext()->isFunctionOrMethod())
    VD->setLocalExternDecl();

  if (uint64_t Val = Record[Idx++]) {
    VD->setInit(Reader.ReadExpr(F));
    if (Val > 1) {
      EvaluatedStmt *Eval = VD->ensureEvaluatedStmt();
      Eval->CheckedICE = true;
      Eval->IsICE = Val == 3;
    }
  }

  enum VarKind {
    VarNotTemplate = 0, VarTemplate, StaticDataMemberSpecialization
  };
  switch ((VarKind)Record[Idx++]) {
  case VarNotTemplate:
    // Only true variables (not parameters or implicit parameters) can be
    // merged; the other kinds are not really redeclarable at all.
    if (!isa<ParmVarDecl>(VD) && !isa<ImplicitParamDecl>(VD) &&
        !isa<VarTemplateSpecializationDecl>(VD))
      mergeRedeclarable(VD, Redecl);
    break;
  case VarTemplate:
    // Merged when we merge the template.
    VD->setDescribedVarTemplate(ReadDeclAs<VarTemplateDecl>(Record, Idx));
    break;
  case StaticDataMemberSpecialization: { // HasMemberSpecializationInfo.
    VarDecl *Tmpl = ReadDeclAs<VarDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    Reader.getContext().setInstantiatedFromStaticDataMember(VD, Tmpl, TSK,POI);
    mergeRedeclarable(VD, Redecl);
    break;
  }
  }

  return Redecl;
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

  // FIXME: If this is a redeclaration of a function from another module, handle
  // inheritance of default arguments.
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
    Expr *copyExpr = ((flags & 4) ? Reader.ReadExpr(F) : nullptr);

    captures.push_back(BlockDecl::Capture(decl, byRef, nested, copyExpr));
  }
  BD->setCaptures(Reader.getContext(), captures, capturesCXXThis);
}

void ASTDeclReader::VisitCapturedDecl(CapturedDecl *CD) {
  VisitDecl(CD);
  unsigned ContextParamPos = Record[Idx++];
  CD->setNothrow(Record[Idx++] != 0);
  // Body is set by VisitCapturedStmt.
  for (unsigned I = 0; I < CD->NumParams; ++I) {
    if (I != ContextParamPos)
      CD->setParam(I, ReadDeclAs<ImplicitParamDecl>(Record, Idx));
    else
      CD->setContextParam(I, ReadDeclAs<ImplicitParamDecl>(Record, Idx));
  }
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

  // Defer loading the anonymous namespace until we've finished merging
  // this namespace; loading it might load a later declaration of the
  // same namespace, and we have an invariant that older declarations
  // get merged before newer ones try to merge.
  GlobalDeclID AnonNamespace = 0;
  if (Redecl.getFirstID() == ThisDeclID) {
    AnonNamespace = ReadDeclID(Record, Idx);
  } else {
    // Link this namespace back to the first declaration, which has already
    // been deserialized.
    D->AnonOrFirstNamespaceAndInline.setPointer(D->getFirstDecl());
  }

  mergeRedeclarable(D, Redecl);

  if (AnonNamespace) {
    // Each module has its own anonymous namespace, which is disjoint from
    // any other module's anonymous namespaces, so don't attach the anonymous
    // namespace at all.
    NamespaceDecl *Anon = cast<NamespaceDecl>(Reader.GetDecl(AnonNamespace));
    if (F.Kind != MK_ImplicitModule && F.Kind != MK_ExplicitModule)
      D->setAnonymousNamespace(Anon);
  }
}

void ASTDeclReader::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarable(D);
  VisitNamedDecl(D);
  D->NamespaceLoc = ReadSourceLocation(Record, Idx);
  D->IdentLoc = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  D->Namespace = ReadDeclAs<NamedDecl>(Record, Idx);
  mergeRedeclarable(D, Redecl);
}

void ASTDeclReader::VisitUsingDecl(UsingDecl *D) {
  VisitNamedDecl(D);
  D->setUsingLoc(ReadSourceLocation(Record, Idx));
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  ReadDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record, Idx);
  D->FirstUsingShadow.setPointer(ReadDeclAs<UsingShadowDecl>(Record, Idx));
  D->setTypename(Record[Idx++]);
  if (NamedDecl *Pattern = ReadDeclAs<NamedDecl>(Record, Idx))
    Reader.getContext().setInstantiatedFromUsingDecl(D, Pattern);
  mergeMergeable(D);
}

void ASTDeclReader::VisitUsingShadowDecl(UsingShadowDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarable(D);
  VisitNamedDecl(D);
  D->setTargetDecl(ReadDeclAs<NamedDecl>(Record, Idx));
  D->UsingOrNextShadow = ReadDeclAs<NamedDecl>(Record, Idx);
  UsingShadowDecl *Pattern = ReadDeclAs<UsingShadowDecl>(Record, Idx);
  if (Pattern)
    Reader.getContext().setInstantiatedFromUsingShadowDecl(D, Pattern);
  mergeRedeclarable(D, Redecl);
}

void ASTDeclReader::VisitConstructorUsingShadowDecl(
    ConstructorUsingShadowDecl *D) {
  VisitUsingShadowDecl(D);
  D->NominatedBaseClassShadowDecl =
      ReadDeclAs<ConstructorUsingShadowDecl>(Record, Idx);
  D->ConstructedBaseClassShadowDecl =
      ReadDeclAs<ConstructorUsingShadowDecl>(Record, Idx);
  D->IsVirtual = Record[Idx++];
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
  mergeMergeable(D);
}

void ASTDeclReader::VisitUnresolvedUsingTypenameDecl(
                                               UnresolvedUsingTypenameDecl *D) {
  VisitTypeDecl(D);
  D->TypenameLocation = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  mergeMergeable(D);
}

void ASTDeclReader::ReadCXXDefinitionData(
                                   struct CXXRecordDecl::DefinitionData &Data,
                                   const RecordData &Record, unsigned &Idx) {
  // Note: the caller has deserialized the IsLambda bit already.
  Data.UserDeclaredConstructor = Record[Idx++];
  Data.UserDeclaredSpecialMembers = Record[Idx++];
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
  Data.HasVariantMembers = Record[Idx++];
  Data.HasOnlyCMembers = Record[Idx++];
  Data.HasInClassInitializer = Record[Idx++];
  Data.HasUninitializedReferenceMember = Record[Idx++];
  Data.HasUninitializedFields = Record[Idx++];
  Data.HasInheritedConstructor = Record[Idx++];
  Data.HasInheritedAssignment = Record[Idx++];
  Data.NeedOverloadResolutionForMoveConstructor = Record[Idx++];
  Data.NeedOverloadResolutionForMoveAssignment = Record[Idx++];
  Data.NeedOverloadResolutionForDestructor = Record[Idx++];
  Data.DefaultedMoveConstructorIsDeleted = Record[Idx++];
  Data.DefaultedMoveAssignmentIsDeleted = Record[Idx++];
  Data.DefaultedDestructorIsDeleted = Record[Idx++];
  Data.HasTrivialSpecialMembers = Record[Idx++];
  Data.DeclaredNonTrivialSpecialMembers = Record[Idx++];
  Data.HasIrrelevantDestructor = Record[Idx++];
  Data.HasConstexprNonCopyMoveConstructor = Record[Idx++];
  Data.HasDefaultedDefaultConstructor = Record[Idx++];
  Data.DefaultedDefaultConstructorIsConstexpr = Record[Idx++];
  Data.HasConstexprDefaultConstructor = Record[Idx++];
  Data.HasNonLiteralTypeFieldsOrBases = Record[Idx++];
  Data.ComputedVisibleConversions = Record[Idx++];
  Data.UserProvidedDefaultConstructor = Record[Idx++];
  Data.DeclaredSpecialMembers = Record[Idx++];
  Data.ImplicitCopyConstructorHasConstParam = Record[Idx++];
  Data.ImplicitCopyAssignmentHasConstParam = Record[Idx++];
  Data.HasDeclaredCopyConstructorWithConstParam = Record[Idx++];
  Data.HasDeclaredCopyAssignmentWithConstParam = Record[Idx++];

  Data.NumBases = Record[Idx++];
  if (Data.NumBases)
    Data.Bases = ReadGlobalOffset(F, Record, Idx);
  Data.NumVBases = Record[Idx++];
  if (Data.NumVBases)
    Data.VBases = ReadGlobalOffset(F, Record, Idx);
  
  Reader.ReadUnresolvedSet(F, Data.Conversions, Record, Idx);
  Reader.ReadUnresolvedSet(F, Data.VisibleConversions, Record, Idx);
  assert(Data.Definition && "Data.Definition should be already set!");
  Data.FirstFriend = ReadDeclID(Record, Idx);

  if (Data.IsLambda) {
    typedef LambdaCapture Capture;
    CXXRecordDecl::LambdaDefinitionData &Lambda
      = static_cast<CXXRecordDecl::LambdaDefinitionData &>(Data);
    Lambda.Dependent = Record[Idx++];
    Lambda.IsGenericLambda = Record[Idx++];
    Lambda.CaptureDefault = Record[Idx++];
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
      switch (Kind) {
      case LCK_StarThis: 
      case LCK_This:
      case LCK_VLAType:
        *ToCapture++ = Capture(Loc, IsImplicit, Kind, nullptr,SourceLocation());
        break;
      case LCK_ByCopy:
      case LCK_ByRef:
        VarDecl *Var = ReadDeclAs<VarDecl>(Record, Idx);
        SourceLocation EllipsisLoc = ReadSourceLocation(Record, Idx);
        *ToCapture++ = Capture(Loc, IsImplicit, Kind, Var, EllipsisLoc);
        break;
      }
    }
  }
}

void ASTDeclReader::MergeDefinitionData(
    CXXRecordDecl *D, struct CXXRecordDecl::DefinitionData &&MergeDD) {
  assert(D->DefinitionData &&
         "merging class definition into non-definition");
  auto &DD = *D->DefinitionData;

  if (DD.Definition != MergeDD.Definition) {
    // Track that we merged the definitions.
    Reader.MergedDeclContexts.insert(std::make_pair(MergeDD.Definition,
                                                    DD.Definition));
    Reader.PendingDefinitions.erase(MergeDD.Definition);
    MergeDD.Definition->IsCompleteDefinition = false;
    mergeDefinitionVisibility(DD.Definition, MergeDD.Definition);
    assert(Reader.Lookups.find(MergeDD.Definition) == Reader.Lookups.end() &&
           "already loaded pending lookups for merged definition");
  }

  auto PFDI = Reader.PendingFakeDefinitionData.find(&DD);
  if (PFDI != Reader.PendingFakeDefinitionData.end() &&
      PFDI->second == ASTReader::PendingFakeDefinitionKind::Fake) {
    // We faked up this definition data because we found a class for which we'd
    // not yet loaded the definition. Replace it with the real thing now.
    assert(!DD.IsLambda && !MergeDD.IsLambda && "faked up lambda definition?");
    PFDI->second = ASTReader::PendingFakeDefinitionKind::FakeLoaded;

    // Don't change which declaration is the definition; that is required
    // to be invariant once we select it.
    auto *Def = DD.Definition;
    DD = std::move(MergeDD);
    DD.Definition = Def;
    return;
  }

  // FIXME: Move this out into a .def file?
  bool DetectedOdrViolation = false;
#define OR_FIELD(Field) DD.Field |= MergeDD.Field;
#define MATCH_FIELD(Field) \
    DetectedOdrViolation |= DD.Field != MergeDD.Field; \
    OR_FIELD(Field)
  MATCH_FIELD(UserDeclaredConstructor)
  MATCH_FIELD(UserDeclaredSpecialMembers)
  MATCH_FIELD(Aggregate)
  MATCH_FIELD(PlainOldData)
  MATCH_FIELD(Empty)
  MATCH_FIELD(Polymorphic)
  MATCH_FIELD(Abstract)
  MATCH_FIELD(IsStandardLayout)
  MATCH_FIELD(HasNoNonEmptyBases)
  MATCH_FIELD(HasPrivateFields)
  MATCH_FIELD(HasProtectedFields)
  MATCH_FIELD(HasPublicFields)
  MATCH_FIELD(HasMutableFields)
  MATCH_FIELD(HasVariantMembers)
  MATCH_FIELD(HasOnlyCMembers)
  MATCH_FIELD(HasInClassInitializer)
  MATCH_FIELD(HasUninitializedReferenceMember)
  MATCH_FIELD(HasUninitializedFields)
  MATCH_FIELD(HasInheritedConstructor)
  MATCH_FIELD(HasInheritedAssignment)
  MATCH_FIELD(NeedOverloadResolutionForMoveConstructor)
  MATCH_FIELD(NeedOverloadResolutionForMoveAssignment)
  MATCH_FIELD(NeedOverloadResolutionForDestructor)
  MATCH_FIELD(DefaultedMoveConstructorIsDeleted)
  MATCH_FIELD(DefaultedMoveAssignmentIsDeleted)
  MATCH_FIELD(DefaultedDestructorIsDeleted)
  OR_FIELD(HasTrivialSpecialMembers)
  OR_FIELD(DeclaredNonTrivialSpecialMembers)
  MATCH_FIELD(HasIrrelevantDestructor)
  OR_FIELD(HasConstexprNonCopyMoveConstructor)
  OR_FIELD(HasDefaultedDefaultConstructor)
  MATCH_FIELD(DefaultedDefaultConstructorIsConstexpr)
  OR_FIELD(HasConstexprDefaultConstructor)
  MATCH_FIELD(HasNonLiteralTypeFieldsOrBases)
  // ComputedVisibleConversions is handled below.
  MATCH_FIELD(UserProvidedDefaultConstructor)
  OR_FIELD(DeclaredSpecialMembers)
  MATCH_FIELD(ImplicitCopyConstructorHasConstParam)
  MATCH_FIELD(ImplicitCopyAssignmentHasConstParam)
  OR_FIELD(HasDeclaredCopyConstructorWithConstParam)
  OR_FIELD(HasDeclaredCopyAssignmentWithConstParam)
  MATCH_FIELD(IsLambda)
#undef OR_FIELD
#undef MATCH_FIELD

  if (DD.NumBases != MergeDD.NumBases || DD.NumVBases != MergeDD.NumVBases)
    DetectedOdrViolation = true;
  // FIXME: Issue a diagnostic if the base classes don't match when we come
  // to lazily load them.

  // FIXME: Issue a diagnostic if the list of conversion functions doesn't
  // match when we come to lazily load them.
  if (MergeDD.ComputedVisibleConversions && !DD.ComputedVisibleConversions) {
    DD.VisibleConversions = std::move(MergeDD.VisibleConversions);
    DD.ComputedVisibleConversions = true;
  }

  // FIXME: Issue a diagnostic if FirstFriend doesn't match when we come to
  // lazily load it.

  if (DD.IsLambda) {
    // FIXME: ODR-checking for merging lambdas (this happens, for instance,
    // when they occur within the body of a function template specialization).
  }

  if (DetectedOdrViolation)
    Reader.PendingOdrMergeFailures[DD.Definition].push_back(MergeDD.Definition);
}

void ASTDeclReader::ReadCXXRecordDefinition(CXXRecordDecl *D, bool Update) {
  struct CXXRecordDecl::DefinitionData *DD;
  ASTContext &C = Reader.getContext();

  // Determine whether this is a lambda closure type, so that we can
  // allocate the appropriate DefinitionData structure.
  bool IsLambda = Record[Idx++];
  if (IsLambda)
    DD = new (C) CXXRecordDecl::LambdaDefinitionData(D, nullptr, false, false,
                                                     LCD_None);
  else
    DD = new (C) struct CXXRecordDecl::DefinitionData(D);

  ReadCXXDefinitionData(*DD, Record, Idx);

  // We might already have a definition for this record. This can happen either
  // because we're reading an update record, or because we've already done some
  // merging. Either way, just merge into it.
  CXXRecordDecl *Canon = D->getCanonicalDecl();
  if (Canon->DefinitionData) {
    MergeDefinitionData(Canon, std::move(*DD));
    D->DefinitionData = Canon->DefinitionData;
    return;
  }

  // Mark this declaration as being a definition.
  D->IsCompleteDefinition = true;
  D->DefinitionData = DD;

  // If this is not the first declaration or is an update record, we can have
  // other redeclarations already. Make a note that we need to propagate the
  // DefinitionData pointer onto them.
  if (Update || Canon != D) {
    Canon->DefinitionData = D->DefinitionData;
    Reader.PendingDefinitions.insert(D);
  }
}

ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitCXXRecordDeclImpl(CXXRecordDecl *D) {
  RedeclarableResult Redecl = VisitRecordDeclImpl(D);

  ASTContext &C = Reader.getContext();

  enum CXXRecKind {
    CXXRecNotTemplate = 0, CXXRecTemplate, CXXRecMemberSpecialization
  };
  switch ((CXXRecKind)Record[Idx++]) {
  case CXXRecNotTemplate:
    // Merged when we merge the folding set entry in the primary template.
    if (!isa<ClassTemplateSpecializationDecl>(D))
      mergeRedeclarable(D, Redecl);
    break;
  case CXXRecTemplate: {
    // Merged when we merge the template.
    ClassTemplateDecl *Template = ReadDeclAs<ClassTemplateDecl>(Record, Idx);
    D->TemplateOrInstantiation = Template;
    if (!Template->getTemplatedDecl()) {
      // We've not actually loaded the ClassTemplateDecl yet, because we're
      // currently being loaded as its pattern. Rely on it to set up our
      // TypeForDecl (see VisitClassTemplateDecl).
      //
      // Beware: we do not yet know our canonical declaration, and may still
      // get merged once the surrounding class template has got off the ground.
      TypeIDForTypeDecl = 0;
    }
    break;
  }
  case CXXRecMemberSpecialization: {
    CXXRecordDecl *RD = ReadDeclAs<CXXRecordDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    MemberSpecializationInfo *MSI = new (C) MemberSpecializationInfo(RD, TSK);
    MSI->setPointOfInstantiation(POI);
    D->TemplateOrInstantiation = MSI;
    mergeRedeclarable(D, Redecl);
    break;
  }
  }

  bool WasDefinition = Record[Idx++];
  if (WasDefinition)
    ReadCXXRecordDefinition(D, /*Update*/false);
  else
    // Propagate DefinitionData pointer from the canonical declaration.
    D->DefinitionData = D->getCanonicalDecl()->DefinitionData;

  // Lazily load the key function to avoid deserializing every method so we can
  // compute it.
  if (WasDefinition) {
    DeclID KeyFn = ReadDeclID(Record, Idx);
    if (KeyFn && D->IsCompleteDefinition)
      // FIXME: This is wrong for the ARM ABI, where some other module may have
      // made this function no longer be a key function. We need an update
      // record or similar for that case.
      C.KeyFunctions[D] = KeyFn;
  }

  return Redecl;
}

void ASTDeclReader::VisitCXXMethodDecl(CXXMethodDecl *D) {
  VisitFunctionDecl(D);

  unsigned NumOverridenMethods = Record[Idx++];
  if (D->isCanonicalDecl()) {
    while (NumOverridenMethods--) {
      // Avoid invariant checking of CXXMethodDecl::addOverriddenMethod,
      // MD may be initializing.
      if (CXXMethodDecl *MD = ReadDeclAs<CXXMethodDecl>(Record, Idx))
        Reader.getContext().addOverriddenMethod(D, MD->getCanonicalDecl());
    }
  } else {
    // We don't care about which declarations this used to override; we get
    // the relevant information from the canonical declaration.
    Idx += NumOverridenMethods;
  }
}

void ASTDeclReader::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  // We need the inherited constructor information to merge the declaration,
  // so we have to read it before we call VisitCXXMethodDecl.
  if (D->isInheritingConstructor()) {
    auto *Shadow = ReadDeclAs<ConstructorUsingShadowDecl>(Record, Idx);
    auto *Ctor = ReadDeclAs<CXXConstructorDecl>(Record, Idx);
    *D->getTrailingObjects<InheritedConstructor>() =
        InheritedConstructor(Shadow, Ctor);
  }

  VisitCXXMethodDecl(D);

  D->IsExplicitSpecified = Record[Idx++];
}

void ASTDeclReader::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  VisitCXXMethodDecl(D);

  if (auto *OperatorDelete = ReadDeclAs<FunctionDecl>(Record, Idx)) {
    auto *Canon = cast<CXXDestructorDecl>(D->getCanonicalDecl());
    // FIXME: Check consistency if we have an old and new operator delete.
    if (!Canon->OperatorDelete)
      Canon->OperatorDelete = OperatorDelete;
  }
}

void ASTDeclReader::VisitCXXConversionDecl(CXXConversionDecl *D) {
  VisitCXXMethodDecl(D);
  D->IsExplicitSpecified = Record[Idx++];
}

void ASTDeclReader::VisitImportDecl(ImportDecl *D) {
  VisitDecl(D);
  D->ImportedAndComplete.setPointer(readModule(Record, Idx));
  D->ImportedAndComplete.setInt(Record[Idx++]);
  SourceLocation *StoredLocs = D->getTrailingObjects<SourceLocation>();
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
  if (Record[Idx++]) // hasFriendDecl
    D->Friend = ReadDeclAs<NamedDecl>(Record, Idx);
  else
    D->Friend = GetTypeSourceInfo(Record, Idx);
  for (unsigned i = 0; i != D->NumTPLists; ++i)
    D->getTrailingObjects<TemplateParameterList *>()[i] =
        Reader.ReadTemplateParameterList(F, Record, Idx);
  D->NextFriend = ReadDeclID(Record, Idx);
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

DeclID ASTDeclReader::VisitTemplateDecl(TemplateDecl *D) {
  VisitNamedDecl(D);

  DeclID PatternID = ReadDeclID(Record, Idx);
  NamedDecl *TemplatedDecl = cast_or_null<NamedDecl>(Reader.GetDecl(PatternID));
  TemplateParameterList* TemplateParams
      = Reader.ReadTemplateParameterList(F, Record, Idx); 
  D->init(TemplatedDecl, TemplateParams);

  return PatternID;
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

  DeclID PatternID = VisitTemplateDecl(D);
  D->IdentifierNamespace = Record[Idx++];

  mergeRedeclarable(D, Redecl, PatternID);

  // If we merged the template with a prior declaration chain, merge the common
  // pointer.
  // FIXME: Actually merge here, don't just overwrite.
  D->Common = D->getCanonicalDecl()->Common;

  return Redecl;
}

static DeclID *newDeclIDList(ASTContext &Context, DeclID *Old,
                             SmallVectorImpl<DeclID> &IDs) {
  assert(!IDs.empty() && "no IDs to add to list");
  if (Old) {
    IDs.insert(IDs.end(), Old + 1, Old + 1 + Old[0]);
    std::sort(IDs.begin(), IDs.end());
    IDs.erase(std::unique(IDs.begin(), IDs.end()), IDs.end());
  }

  auto *Result = new (Context) DeclID[1 + IDs.size()];
  *Result = IDs.size();
  std::copy(IDs.begin(), IDs.end(), Result + 1);
  return Result;
}

void ASTDeclReader::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarableTemplateDecl(D);

  if (ThisDeclID == Redecl.getFirstID()) {
    // This ClassTemplateDecl owns a CommonPtr; read it to keep track of all of
    // the specializations.
    SmallVector<serialization::DeclID, 32> SpecIDs;
    ReadDeclIDList(SpecIDs);

    if (!SpecIDs.empty()) {
      auto *CommonPtr = D->getCommonPtr();
      CommonPtr->LazySpecializations = newDeclIDList(
          Reader.getContext(), CommonPtr->LazySpecializations, SpecIDs);
    }
  }

  if (D->getTemplatedDecl()->TemplateOrInstantiation) {
    // We were loaded before our templated declaration was. We've not set up
    // its corresponding type yet (see VisitCXXRecordDeclImpl), so reconstruct
    // it now.
    Reader.Context.getInjectedClassNameType(
        D->getTemplatedDecl(), D->getInjectedClassNameSpecialization());
  }
}

void ASTDeclReader::VisitBuiltinTemplateDecl(BuiltinTemplateDecl *D) {
  llvm_unreachable("BuiltinTemplates are not serialized");
}

/// TODO: Unify with ClassTemplateDecl version?
///       May require unifying ClassTemplateDecl and
///        VarTemplateDecl beyond TemplateDecl...
void ASTDeclReader::VisitVarTemplateDecl(VarTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarableTemplateDecl(D);

  if (ThisDeclID == Redecl.getFirstID()) {
    // This VarTemplateDecl owns a CommonPtr; read it to keep track of all of
    // the specializations.
    SmallVector<serialization::DeclID, 32> SpecIDs;
    ReadDeclIDList(SpecIDs);

    if (!SpecIDs.empty()) {
      auto *CommonPtr = D->getCommonPtr();
      CommonPtr->LazySpecializations = newDeclIDList(
          Reader.getContext(), CommonPtr->LazySpecializations, SpecIDs);
    }
  }
}

ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitClassTemplateSpecializationDeclImpl(
    ClassTemplateSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitCXXRecordDeclImpl(D);
  
  ASTContext &C = Reader.getContext();
  if (Decl *InstD = ReadDecl(Record, Idx)) {
    if (ClassTemplateDecl *CTD = dyn_cast<ClassTemplateDecl>(InstD)) {
      D->SpecializedTemplate = CTD;
    } else {
      SmallVector<TemplateArgument, 8> TemplArgs;
      Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
      TemplateArgumentList *ArgList
        = TemplateArgumentList::CreateCopy(C, TemplArgs);
      ClassTemplateSpecializationDecl::SpecializedPartialSpecialization *PS
          = new (C) ClassTemplateSpecializationDecl::
                                             SpecializedPartialSpecialization();
      PS->PartialSpecialization
          = cast<ClassTemplatePartialSpecializationDecl>(InstD);
      PS->TemplateArgs = ArgList;
      D->SpecializedTemplate = PS;
    }
  }

  SmallVector<TemplateArgument, 8> TemplArgs;
  Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx,
                                  /*Canonicalize*/ true);
  D->TemplateArgs = TemplateArgumentList::CreateCopy(C, TemplArgs);
  D->PointOfInstantiation = ReadSourceLocation(Record, Idx);
  D->SpecializationKind = (TemplateSpecializationKind)Record[Idx++];

  bool writtenAsCanonicalDecl = Record[Idx++];
  if (writtenAsCanonicalDecl) {
    ClassTemplateDecl *CanonPattern = ReadDeclAs<ClassTemplateDecl>(Record,Idx);
    if (D->isCanonicalDecl()) { // It's kept in the folding set.
      // Set this as, or find, the canonical declaration for this specialization
      ClassTemplateSpecializationDecl *CanonSpec;
      if (ClassTemplatePartialSpecializationDecl *Partial =
              dyn_cast<ClassTemplatePartialSpecializationDecl>(D)) {
        CanonSpec = CanonPattern->getCommonPtr()->PartialSpecializations
            .GetOrInsertNode(Partial);
      } else {
        CanonSpec =
            CanonPattern->getCommonPtr()->Specializations.GetOrInsertNode(D);
      }
      // If there was already a canonical specialization, merge into it.
      if (CanonSpec != D) {
        mergeRedeclarable<TagDecl>(D, CanonSpec, Redecl);

        // This declaration might be a definition. Merge with any existing
        // definition.
        if (auto *DDD = D->DefinitionData) {
          if (CanonSpec->DefinitionData)
            MergeDefinitionData(CanonSpec, std::move(*DDD));
          else
            CanonSpec->DefinitionData = D->DefinitionData;
        }
        D->DefinitionData = CanonSpec->DefinitionData;
      }
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

  return Redecl;
}

void ASTDeclReader::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitClassTemplateSpecializationDeclImpl(D);

  D->TemplateParams = Reader.ReadTemplateParameterList(F, Record, Idx);
  D->ArgsAsWritten = Reader.ReadASTTemplateArgumentListInfo(F, Record, Idx);

  // These are read/set from/to the first declaration.
  if (ThisDeclID == Redecl.getFirstID()) {
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
    SmallVector<serialization::DeclID, 32> SpecIDs;
    ReadDeclIDList(SpecIDs);

    if (!SpecIDs.empty()) {
      auto *CommonPtr = D->getCommonPtr();
      CommonPtr->LazySpecializations = newDeclIDList(
          Reader.getContext(), CommonPtr->LazySpecializations, SpecIDs);
    }
  }
}

/// TODO: Unify with ClassTemplateSpecializationDecl version?
///       May require unifying ClassTemplate(Partial)SpecializationDecl and
///        VarTemplate(Partial)SpecializationDecl with a new data
///        structure Template(Partial)SpecializationDecl, and
///        using Template(Partial)SpecializationDecl as input type.
ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitVarTemplateSpecializationDeclImpl(
    VarTemplateSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitVarDeclImpl(D);

  ASTContext &C = Reader.getContext();
  if (Decl *InstD = ReadDecl(Record, Idx)) {
    if (VarTemplateDecl *VTD = dyn_cast<VarTemplateDecl>(InstD)) {
      D->SpecializedTemplate = VTD;
    } else {
      SmallVector<TemplateArgument, 8> TemplArgs;
      Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
      TemplateArgumentList *ArgList = TemplateArgumentList::CreateCopy(
          C, TemplArgs);
      VarTemplateSpecializationDecl::SpecializedPartialSpecialization *PS =
          new (C)
          VarTemplateSpecializationDecl::SpecializedPartialSpecialization();
      PS->PartialSpecialization =
          cast<VarTemplatePartialSpecializationDecl>(InstD);
      PS->TemplateArgs = ArgList;
      D->SpecializedTemplate = PS;
    }
  }

  // Explicit info.
  if (TypeSourceInfo *TyInfo = GetTypeSourceInfo(Record, Idx)) {
    VarTemplateSpecializationDecl::ExplicitSpecializationInfo *ExplicitInfo =
        new (C) VarTemplateSpecializationDecl::ExplicitSpecializationInfo;
    ExplicitInfo->TypeAsWritten = TyInfo;
    ExplicitInfo->ExternLoc = ReadSourceLocation(Record, Idx);
    ExplicitInfo->TemplateKeywordLoc = ReadSourceLocation(Record, Idx);
    D->ExplicitInfo = ExplicitInfo;
  }

  SmallVector<TemplateArgument, 8> TemplArgs;
  Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx,
                                  /*Canonicalize*/ true);
  D->TemplateArgs = TemplateArgumentList::CreateCopy(C, TemplArgs);
  D->PointOfInstantiation = ReadSourceLocation(Record, Idx);
  D->SpecializationKind = (TemplateSpecializationKind)Record[Idx++];

  bool writtenAsCanonicalDecl = Record[Idx++];
  if (writtenAsCanonicalDecl) {
    VarTemplateDecl *CanonPattern = ReadDeclAs<VarTemplateDecl>(Record, Idx);
    if (D->isCanonicalDecl()) { // It's kept in the folding set.
      // FIXME: If it's already present, merge it.
      if (VarTemplatePartialSpecializationDecl *Partial =
              dyn_cast<VarTemplatePartialSpecializationDecl>(D)) {
        CanonPattern->getCommonPtr()->PartialSpecializations
            .GetOrInsertNode(Partial);
      } else {
        CanonPattern->getCommonPtr()->Specializations.GetOrInsertNode(D);
      }
    }
  }

  return Redecl;
}

/// TODO: Unify with ClassTemplatePartialSpecializationDecl version?
///       May require unifying ClassTemplate(Partial)SpecializationDecl and
///        VarTemplate(Partial)SpecializationDecl with a new data
///        structure Template(Partial)SpecializationDecl, and
///        using Template(Partial)SpecializationDecl as input type.
void ASTDeclReader::VisitVarTemplatePartialSpecializationDecl(
    VarTemplatePartialSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitVarTemplateSpecializationDeclImpl(D);

  D->TemplateParams = Reader.ReadTemplateParameterList(F, Record, Idx);
  D->ArgsAsWritten = Reader.ReadASTTemplateArgumentListInfo(F, Record, Idx);

  // These are read/set from/to the first declaration.
  if (ThisDeclID == Redecl.getFirstID()) {
    D->InstantiatedFromMember.setPointer(
        ReadDeclAs<VarTemplatePartialSpecializationDecl>(Record, Idx));
    D->InstantiatedFromMember.setInt(Record[Idx++]);
  }
}

void ASTDeclReader::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  VisitTypeDecl(D);

  D->setDeclaredWithTypename(Record[Idx++]);

  if (Record[Idx++])
    D->setDefaultArgument(GetTypeSourceInfo(Record, Idx));
}

void ASTDeclReader::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  VisitDeclaratorDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  if (D->isExpandedParameterPack()) {
    auto TypesAndInfos =
        D->getTrailingObjects<std::pair<QualType, TypeSourceInfo *>>();
    for (unsigned I = 0, N = D->getNumExpansionTypes(); I != N; ++I) {
      new (&TypesAndInfos[I].first) QualType(Reader.readType(F, Record, Idx));
      TypesAndInfos[I].second = GetTypeSourceInfo(Record, Idx);
    }
  } else {
    // Rest of NonTypeTemplateParmDecl.
    D->ParameterPack = Record[Idx++];
    if (Record[Idx++])
      D->setDefaultArgument(Reader.ReadExpr(F));
  }
}

void ASTDeclReader::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  VisitTemplateDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  if (D->isExpandedParameterPack()) {
    TemplateParameterList **Data =
        D->getTrailingObjects<TemplateParameterList *>();
    for (unsigned I = 0, N = D->getNumExpansionTemplateParameters();
         I != N; ++I)
      Data[I] = Reader.ReadTemplateParameterList(F, Record, Idx);
  } else {
    // Rest of TemplateTemplateParmDecl.
    D->ParameterPack = Record[Idx++];
    if (Record[Idx++])
      D->setDefaultArgument(Reader.getContext(),
                            Reader.ReadTemplateArgumentLoc(F, Record, Idx));
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

void ASTDeclReader::VisitEmptyDecl(EmptyDecl *D) {
  VisitDecl(D);
}

std::pair<uint64_t, uint64_t>
ASTDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = ReadLocalOffset(Record, Idx);
  uint64_t VisibleOffset = ReadLocalOffset(Record, Idx);
  return std::make_pair(LexicalOffset, VisibleOffset);
}

template <typename T>
ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitRedeclarable(Redeclarable<T> *D) {
  DeclID FirstDeclID = ReadDeclID(Record, Idx);
  Decl *MergeWith = nullptr;

  bool IsKeyDecl = ThisDeclID == FirstDeclID;
  bool IsFirstLocalDecl = false;

  uint64_t RedeclOffset = 0;

  // 0 indicates that this declaration was the only declaration of its entity,
  // and is used for space optimization.
  if (FirstDeclID == 0) {
    FirstDeclID = ThisDeclID;
    IsKeyDecl = true;
    IsFirstLocalDecl = true;
  } else if (unsigned N = Record[Idx++]) {
    // This declaration was the first local declaration, but may have imported
    // other declarations.
    IsKeyDecl = N == 1;
    IsFirstLocalDecl = true;

    // We have some declarations that must be before us in our redeclaration
    // chain. Read them now, and remember that we ought to merge with one of
    // them.
    // FIXME: Provide a known merge target to the second and subsequent such
    // declaration.
    for (unsigned I = 0; I != N - 1; ++I)
      MergeWith = ReadDecl(Record, Idx/*, MergeWith*/);

    RedeclOffset = ReadLocalOffset(Record, Idx);
  } else {
    // This declaration was not the first local declaration. Read the first
    // local declaration now, to trigger the import of other redeclarations.
    (void)ReadDecl(Record, Idx);
  }

  T *FirstDecl = cast_or_null<T>(Reader.GetDecl(FirstDeclID));
  if (FirstDecl != D) {
    // We delay loading of the redeclaration chain to avoid deeply nested calls.
    // We temporarily set the first (canonical) declaration as the previous one
    // which is the one that matters and mark the real previous DeclID to be
    // loaded & attached later on.
    D->RedeclLink = Redeclarable<T>::PreviousDeclLink(FirstDecl);
    D->First = FirstDecl->getCanonicalDecl();
  }    

  T *DAsT = static_cast<T*>(D);

  // Note that we need to load local redeclarations of this decl and build a
  // decl chain for them. This must happen *after* we perform the preloading
  // above; this ensures that the redeclaration chain is built in the correct
  // order.
  if (IsFirstLocalDecl)
    Reader.PendingDeclChains.push_back(std::make_pair(DAsT, RedeclOffset));

  return RedeclarableResult(FirstDeclID, MergeWith, IsKeyDecl);
}

/// \brief Attempts to merge the given declaration (D) with another declaration
/// of the same entity.
template<typename T>
void ASTDeclReader::mergeRedeclarable(Redeclarable<T> *DBase,
                                      RedeclarableResult &Redecl,
                                      DeclID TemplatePatternID) {
  T *D = static_cast<T*>(DBase);

  // If modules are not available, there is no reason to perform this merge.
  if (!Reader.getContext().getLangOpts().Modules)
    return;

  // If we're not the canonical declaration, we don't need to merge.
  if (!DBase->isFirstDecl())
    return;

  if (auto *Existing = Redecl.getKnownMergeTarget())
    // We already know of an existing declaration we should merge with.
    mergeRedeclarable(D, cast<T>(Existing), Redecl, TemplatePatternID);
  else if (FindExistingResult ExistingRes = findExisting(D))
    if (T *Existing = ExistingRes)
      mergeRedeclarable(D, Existing, Redecl, TemplatePatternID);
}

/// \brief "Cast" to type T, asserting if we don't have an implicit conversion.
/// We use this to put code in a template that will only be valid for certain
/// instantiations.
template<typename T> static T assert_cast(T t) { return t; }
template<typename T> static T assert_cast(...) {
  llvm_unreachable("bad assert_cast");
}

/// \brief Merge together the pattern declarations from two template
/// declarations.
void ASTDeclReader::mergeTemplatePattern(RedeclarableTemplateDecl *D,
                                         RedeclarableTemplateDecl *Existing,
                                         DeclID DsID, bool IsKeyDecl) {
  auto *DPattern = D->getTemplatedDecl();
  auto *ExistingPattern = Existing->getTemplatedDecl();
  RedeclarableResult Result(DPattern->getCanonicalDecl()->getGlobalID(),
                            /*MergeWith*/ ExistingPattern, IsKeyDecl);

  if (auto *DClass = dyn_cast<CXXRecordDecl>(DPattern)) {
    // Merge with any existing definition.
    // FIXME: This is duplicated in several places. Refactor.
    auto *ExistingClass =
        cast<CXXRecordDecl>(ExistingPattern)->getCanonicalDecl();
    if (auto *DDD = DClass->DefinitionData) {
      if (ExistingClass->DefinitionData) {
        MergeDefinitionData(ExistingClass, std::move(*DDD));
      } else {
        ExistingClass->DefinitionData = DClass->DefinitionData;
        // We may have skipped this before because we thought that DClass
        // was the canonical declaration.
        Reader.PendingDefinitions.insert(DClass);
      }
    }
    DClass->DefinitionData = ExistingClass->DefinitionData;

    return mergeRedeclarable(DClass, cast<TagDecl>(ExistingPattern),
                             Result);
  }
  if (auto *DFunction = dyn_cast<FunctionDecl>(DPattern))
    return mergeRedeclarable(DFunction, cast<FunctionDecl>(ExistingPattern),
                             Result);
  if (auto *DVar = dyn_cast<VarDecl>(DPattern))
    return mergeRedeclarable(DVar, cast<VarDecl>(ExistingPattern), Result);
  if (auto *DAlias = dyn_cast<TypeAliasDecl>(DPattern))
    return mergeRedeclarable(DAlias, cast<TypedefNameDecl>(ExistingPattern),
                             Result);
  llvm_unreachable("merged an unknown kind of redeclarable template");
}

/// \brief Attempts to merge the given declaration (D) with another declaration
/// of the same entity.
template<typename T>
void ASTDeclReader::mergeRedeclarable(Redeclarable<T> *DBase, T *Existing,
                                      RedeclarableResult &Redecl,
                                      DeclID TemplatePatternID) {
  T *D = static_cast<T*>(DBase);
  T *ExistingCanon = Existing->getCanonicalDecl();
  T *DCanon = D->getCanonicalDecl();
  if (ExistingCanon != DCanon) {
    assert(DCanon->getGlobalID() == Redecl.getFirstID() &&
           "already merged this declaration");

    // Have our redeclaration link point back at the canonical declaration
    // of the existing declaration, so that this declaration has the
    // appropriate canonical declaration.
    D->RedeclLink = Redeclarable<T>::PreviousDeclLink(ExistingCanon);
    D->First = ExistingCanon;
    ExistingCanon->Used |= D->Used;
    D->Used = false;

    // When we merge a namespace, update its pointer to the first namespace.
    // We cannot have loaded any redeclarations of this declaration yet, so
    // there's nothing else that needs to be updated.
    if (auto *Namespace = dyn_cast<NamespaceDecl>(D))
      Namespace->AnonOrFirstNamespaceAndInline.setPointer(
          assert_cast<NamespaceDecl*>(ExistingCanon));

    // When we merge a template, merge its pattern.
    if (auto *DTemplate = dyn_cast<RedeclarableTemplateDecl>(D))
      mergeTemplatePattern(
          DTemplate, assert_cast<RedeclarableTemplateDecl*>(ExistingCanon),
          TemplatePatternID, Redecl.isKeyDecl());

    // If this declaration is a key declaration, make a note of that.
    if (Redecl.isKeyDecl())
      Reader.KeyDecls[ExistingCanon].push_back(Redecl.getFirstID());
  }
}

/// \brief Attempts to merge the given declaration (D) with another declaration
/// of the same entity, for the case where the entity is not actually
/// redeclarable. This happens, for instance, when merging the fields of
/// identical class definitions from two different modules.
template<typename T>
void ASTDeclReader::mergeMergeable(Mergeable<T> *D) {
  // If modules are not available, there is no reason to perform this merge.
  if (!Reader.getContext().getLangOpts().Modules)
    return;

  // ODR-based merging is only performed in C++. In C, identically-named things
  // in different translation units are not redeclarations (but may still have
  // compatible types).
  if (!Reader.getContext().getLangOpts().CPlusPlus)
    return;

  if (FindExistingResult ExistingRes = findExisting(static_cast<T*>(D)))
    if (T *Existing = ExistingRes)
      Reader.Context.setPrimaryMergedDecl(static_cast<T*>(D),
                                          Existing->getCanonicalDecl());
}

void ASTDeclReader::VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D) {
  VisitDecl(D);
  unsigned NumVars = D->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i) {
    Vars.push_back(Reader.ReadExpr(F));
  }
  D->setVars(Vars);
}

void ASTDeclReader::VisitOMPDeclareReductionDecl(OMPDeclareReductionDecl *D) {
  VisitValueDecl(D);
  D->setLocation(Reader.ReadSourceLocation(F, Record, Idx));
  D->setCombiner(Reader.ReadExpr(F));
  D->setInitializer(Reader.ReadExpr(F));
  D->PrevDeclInScope = Reader.ReadDeclID(F, Record, Idx);
}

void ASTDeclReader::VisitOMPCapturedExprDecl(OMPCapturedExprDecl *D) {
  VisitVarDecl(D);
}

//===----------------------------------------------------------------------===//
// Attribute Reading
//===----------------------------------------------------------------------===//

/// \brief Reads attributes from the current stream position.
void ASTReader::ReadAttributes(ModuleFile &F, AttrVec &Attrs,
                               const RecordData &Record, unsigned &Idx) {
  for (unsigned i = 0, e = Record[Idx++]; i != e; ++i) {
    Attr *New = nullptr;
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
      isa<ObjCImplDecl>(D) ||
      isa<ImportDecl>(D) ||
      isa<PragmaCommentDecl>(D) ||
      isa<PragmaDetectMismatchDecl>(D))
    return true;
  if (isa<OMPThreadPrivateDecl>(D) || isa<OMPDeclareReductionDecl>(D))
    return !D->getDeclContext()->isFunctionOrMethod();
  if (VarDecl *Var = dyn_cast<VarDecl>(D))
    return Var->isFileVarDecl() &&
           Var->isThisDeclarationADefinition() == VarDecl::Definition;
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D))
    return Func->doesThisDeclarationHaveABody() || HasBody;
  
  return false;
}

/// \brief Get the correct cursor and offset for loading a declaration.
ASTReader::RecordLocation
ASTReader::DeclCursorForID(DeclID ID, SourceLocation &Loc) {
  GlobalDeclMapType::iterator I = GlobalDeclMap.find(ID);
  assert(I != GlobalDeclMap.end() && "Corrupted global declaration map");
  ModuleFile *M = I->second;
  const DeclOffset &DOffs =
      M->DeclOffsets[ID - M->BaseDeclID - NUM_PREDEF_DECL_IDS];
  Loc = TranslateSourceLocation(*M, DOffs.getLocation());
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

static bool isSameTemplateParameterList(const TemplateParameterList *X,
                                        const TemplateParameterList *Y);

/// \brief Determine whether two template parameters are similar enough
/// that they may be used in declarations of the same template.
static bool isSameTemplateParameter(const NamedDecl *X,
                                    const NamedDecl *Y) {
  if (X->getKind() != Y->getKind())
    return false;

  if (const TemplateTypeParmDecl *TX = dyn_cast<TemplateTypeParmDecl>(X)) {
    const TemplateTypeParmDecl *TY = cast<TemplateTypeParmDecl>(Y);
    return TX->isParameterPack() == TY->isParameterPack();
  }

  if (const NonTypeTemplateParmDecl *TX = dyn_cast<NonTypeTemplateParmDecl>(X)) {
    const NonTypeTemplateParmDecl *TY = cast<NonTypeTemplateParmDecl>(Y);
    return TX->isParameterPack() == TY->isParameterPack() &&
           TX->getASTContext().hasSameType(TX->getType(), TY->getType());
  }

  const TemplateTemplateParmDecl *TX = cast<TemplateTemplateParmDecl>(X);
  const TemplateTemplateParmDecl *TY = cast<TemplateTemplateParmDecl>(Y);
  return TX->isParameterPack() == TY->isParameterPack() &&
         isSameTemplateParameterList(TX->getTemplateParameters(),
                                     TY->getTemplateParameters());
}

static NamespaceDecl *getNamespace(const NestedNameSpecifier *X) {
  if (auto *NS = X->getAsNamespace())
    return NS;
  if (auto *NAS = X->getAsNamespaceAlias())
    return NAS->getNamespace();
  return nullptr;
}

static bool isSameQualifier(const NestedNameSpecifier *X,
                            const NestedNameSpecifier *Y) {
  if (auto *NSX = getNamespace(X)) {
    auto *NSY = getNamespace(Y);
    if (!NSY || NSX->getCanonicalDecl() != NSY->getCanonicalDecl())
      return false;
  } else if (X->getKind() != Y->getKind())
    return false;

  // FIXME: For namespaces and types, we're permitted to check that the entity
  // is named via the same tokens. We should probably do so.
  switch (X->getKind()) {
  case NestedNameSpecifier::Identifier:
    if (X->getAsIdentifier() != Y->getAsIdentifier())
      return false;
    break;
  case NestedNameSpecifier::Namespace:
  case NestedNameSpecifier::NamespaceAlias:
    // We've already checked that we named the same namespace.
    break;
  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    if (X->getAsType()->getCanonicalTypeInternal() !=
        Y->getAsType()->getCanonicalTypeInternal())
      return false;
    break;
  case NestedNameSpecifier::Global:
  case NestedNameSpecifier::Super:
    return true;
  }

  // Recurse into earlier portion of NNS, if any.
  auto *PX = X->getPrefix();
  auto *PY = Y->getPrefix();
  if (PX && PY)
    return isSameQualifier(PX, PY);
  return !PX && !PY;
}

/// \brief Determine whether two template parameter lists are similar enough
/// that they may be used in declarations of the same template.
static bool isSameTemplateParameterList(const TemplateParameterList *X,
                                        const TemplateParameterList *Y) {
  if (X->size() != Y->size())
    return false;

  for (unsigned I = 0, N = X->size(); I != N; ++I)
    if (!isSameTemplateParameter(X->getParam(I), Y->getParam(I)))
      return false;

  return true;
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

  if (isa<ClassTemplateSpecializationDecl>(X)) {
    // No need to handle these here: we merge them when adding them to the
    // template.
    return false;
  }

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
  // FIXME: This needs to cope with merging of prototyped/non-prototyped
  // functions, etc.
  if (FunctionDecl *FuncX = dyn_cast<FunctionDecl>(X)) {
    FunctionDecl *FuncY = cast<FunctionDecl>(Y);
    if (CXXConstructorDecl *CtorX = dyn_cast<CXXConstructorDecl>(X)) {
      CXXConstructorDecl *CtorY = cast<CXXConstructorDecl>(Y);
      if (CtorX->getInheritedConstructor() &&
          !isSameEntity(CtorX->getInheritedConstructor().getConstructor(),
                        CtorY->getInheritedConstructor().getConstructor()))
        return false;
    }
    return (FuncX->getLinkageInternal() == FuncY->getLinkageInternal()) &&
      FuncX->getASTContext().hasSameType(FuncX->getType(), FuncY->getType());
  }

  // Variables with the same type and linkage match.
  if (VarDecl *VarX = dyn_cast<VarDecl>(X)) {
    VarDecl *VarY = cast<VarDecl>(Y);
    if (VarX->getLinkageInternal() == VarY->getLinkageInternal()) {
      ASTContext &C = VarX->getASTContext();
      if (C.hasSameType(VarX->getType(), VarY->getType()))
        return true;

      // We can get decls with different types on the redecl chain. Eg.
      // template <typename T> struct S { static T Var[]; }; // #1
      // template <typename T> T S<T>::Var[sizeof(T)]; // #2
      // Only? happens when completing an incomplete array type. In this case
      // when comparing #1 and #2 we should go through their element type.
      const ArrayType *VarXTy = C.getAsArrayType(VarX->getType());
      const ArrayType *VarYTy = C.getAsArrayType(VarY->getType());
      if (!VarXTy || !VarYTy)
        return false;
      if (VarXTy->isIncompleteArrayType() || VarYTy->isIncompleteArrayType())
        return C.hasSameType(VarXTy->getElementType(), VarYTy->getElementType());
    }
    return false;
  }

  // Namespaces with the same name and inlinedness match.
  if (NamespaceDecl *NamespaceX = dyn_cast<NamespaceDecl>(X)) {
    NamespaceDecl *NamespaceY = cast<NamespaceDecl>(Y);
    return NamespaceX->isInline() == NamespaceY->isInline();
  }

  // Identical template names and kinds match if their template parameter lists
  // and patterns match.
  if (TemplateDecl *TemplateX = dyn_cast<TemplateDecl>(X)) {
    TemplateDecl *TemplateY = cast<TemplateDecl>(Y);
    return isSameEntity(TemplateX->getTemplatedDecl(),
                        TemplateY->getTemplatedDecl()) &&
           isSameTemplateParameterList(TemplateX->getTemplateParameters(),
                                       TemplateY->getTemplateParameters());
  }

  // Fields with the same name and the same type match.
  if (FieldDecl *FDX = dyn_cast<FieldDecl>(X)) {
    FieldDecl *FDY = cast<FieldDecl>(Y);
    // FIXME: Also check the bitwidth is odr-equivalent, if any.
    return X->getASTContext().hasSameType(FDX->getType(), FDY->getType());
  }

  // Indirect fields with the same target field match.
  if (auto *IFDX = dyn_cast<IndirectFieldDecl>(X)) {
    auto *IFDY = cast<IndirectFieldDecl>(Y);
    return IFDX->getAnonField()->getCanonicalDecl() ==
           IFDY->getAnonField()->getCanonicalDecl();
  }

  // Enumerators with the same name match.
  if (isa<EnumConstantDecl>(X))
    // FIXME: Also check the value is odr-equivalent.
    return true;

  // Using shadow declarations with the same target match.
  if (UsingShadowDecl *USX = dyn_cast<UsingShadowDecl>(X)) {
    UsingShadowDecl *USY = cast<UsingShadowDecl>(Y);
    return USX->getTargetDecl() == USY->getTargetDecl();
  }

  // Using declarations with the same qualifier match. (We already know that
  // the name matches.)
  if (auto *UX = dyn_cast<UsingDecl>(X)) {
    auto *UY = cast<UsingDecl>(Y);
    return isSameQualifier(UX->getQualifier(), UY->getQualifier()) &&
           UX->hasTypename() == UY->hasTypename() &&
           UX->isAccessDeclaration() == UY->isAccessDeclaration();
  }
  if (auto *UX = dyn_cast<UnresolvedUsingValueDecl>(X)) {
    auto *UY = cast<UnresolvedUsingValueDecl>(Y);
    return isSameQualifier(UX->getQualifier(), UY->getQualifier()) &&
           UX->isAccessDeclaration() == UY->isAccessDeclaration();
  }
  if (auto *UX = dyn_cast<UnresolvedUsingTypenameDecl>(X))
    return isSameQualifier(
        UX->getQualifier(),
        cast<UnresolvedUsingTypenameDecl>(Y)->getQualifier());

  // Namespace alias definitions with the same target match.
  if (auto *NAX = dyn_cast<NamespaceAliasDecl>(X)) {
    auto *NAY = cast<NamespaceAliasDecl>(Y);
    return NAX->getNamespace()->Equals(NAY->getNamespace());
  }

  return false;
}

/// Find the context in which we should search for previous declarations when
/// looking for declarations to merge.
DeclContext *ASTDeclReader::getPrimaryContextForMerging(ASTReader &Reader,
                                                        DeclContext *DC) {
  if (NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC))
    return ND->getOriginalNamespace();

  if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC)) {
    // Try to dig out the definition.
    auto *DD = RD->DefinitionData;
    if (!DD)
      DD = RD->getCanonicalDecl()->DefinitionData;

    // If there's no definition yet, then DC's definition is added by an update
    // record, but we've not yet loaded that update record. In this case, we
    // commit to DC being the canonical definition now, and will fix this when
    // we load the update record.
    if (!DD) {
      DD = new (Reader.Context) struct CXXRecordDecl::DefinitionData(RD);
      RD->IsCompleteDefinition = true;
      RD->DefinitionData = DD;
      RD->getCanonicalDecl()->DefinitionData = DD;

      // Track that we did this horrible thing so that we can fix it later.
      Reader.PendingFakeDefinitionData.insert(
          std::make_pair(DD, ASTReader::PendingFakeDefinitionKind::Fake));
    }

    return DD->Definition;
  }

  if (EnumDecl *ED = dyn_cast<EnumDecl>(DC))
    return ED->getASTContext().getLangOpts().CPlusPlus? ED->getDefinition()
                                                      : nullptr;

  // We can see the TU here only if we have no Sema object. In that case,
  // there's no TU scope to look in, so using the DC alone is sufficient.
  if (auto *TU = dyn_cast<TranslationUnitDecl>(DC))
    return TU;

  return nullptr;
}

ASTDeclReader::FindExistingResult::~FindExistingResult() {
  // Record that we had a typedef name for linkage whether or not we merge
  // with that declaration.
  if (TypedefNameForLinkage) {
    DeclContext *DC = New->getDeclContext()->getRedeclContext();
    Reader.ImportedTypedefNamesForLinkage.insert(
        std::make_pair(std::make_pair(DC, TypedefNameForLinkage), New));
    return;
  }

  if (!AddResult || Existing)
    return;

  DeclarationName Name = New->getDeclName();
  DeclContext *DC = New->getDeclContext()->getRedeclContext();
  if (needsAnonymousDeclarationNumber(New)) {
    setAnonymousDeclForMerging(Reader, New->getLexicalDeclContext(),
                               AnonymousDeclNumber, New);
  } else if (DC->isTranslationUnit() &&
             !Reader.getContext().getLangOpts().CPlusPlus) {
    if (Reader.getIdResolver().tryAddTopLevelDecl(New, Name))
      Reader.PendingFakeLookupResults[Name.getAsIdentifierInfo()]
            .push_back(New);
  } else if (DeclContext *MergeDC = getPrimaryContextForMerging(Reader, DC)) {
    // Add the declaration to its redeclaration context so later merging
    // lookups will find it.
    MergeDC->makeDeclVisibleInContextImpl(New, /*Internal*/true);
  }
}

/// Find the declaration that should be merged into, given the declaration found
/// by name lookup. If we're merging an anonymous declaration within a typedef,
/// we need a matching typedef, and we merge with the type inside it.
static NamedDecl *getDeclForMerging(NamedDecl *Found,
                                    bool IsTypedefNameForLinkage) {
  if (!IsTypedefNameForLinkage)
    return Found;

  // If we found a typedef declaration that gives a name to some other
  // declaration, then we want that inner declaration. Declarations from
  // AST files are handled via ImportedTypedefNamesForLinkage.
  if (Found->isFromASTFile())
    return nullptr;

  if (auto *TND = dyn_cast<TypedefNameDecl>(Found))
    return TND->getAnonDeclWithTypedefName();

  return nullptr;
}

NamedDecl *ASTDeclReader::getAnonymousDeclForMerging(ASTReader &Reader,
                                                     DeclContext *DC,
                                                     unsigned Index) {
  // If the lexical context has been merged, look into the now-canonical
  // definition.
  if (auto *Merged = Reader.MergedDeclContexts.lookup(DC))
    DC = Merged;

  // If we've seen this before, return the canonical declaration.
  auto &Previous = Reader.AnonymousDeclarationsForMerging[DC];
  if (Index < Previous.size() && Previous[Index])
    return Previous[Index];

  // If this is the first time, but we have parsed a declaration of the context,
  // build the anonymous declaration list from the parsed declaration.
  if (!cast<Decl>(DC)->isFromASTFile()) {
    numberAnonymousDeclsWithin(DC, [&](NamedDecl *ND, unsigned Number) {
      if (Previous.size() == Number)
        Previous.push_back(cast<NamedDecl>(ND->getCanonicalDecl()));
      else
        Previous[Number] = cast<NamedDecl>(ND->getCanonicalDecl());
    });
  }

  return Index < Previous.size() ? Previous[Index] : nullptr;
}

void ASTDeclReader::setAnonymousDeclForMerging(ASTReader &Reader,
                                               DeclContext *DC, unsigned Index,
                                               NamedDecl *D) {
  if (auto *Merged = Reader.MergedDeclContexts.lookup(DC))
    DC = Merged;

  auto &Previous = Reader.AnonymousDeclarationsForMerging[DC];
  if (Index >= Previous.size())
    Previous.resize(Index + 1);
  if (!Previous[Index])
    Previous[Index] = D;
}

ASTDeclReader::FindExistingResult ASTDeclReader::findExisting(NamedDecl *D) {
  DeclarationName Name = TypedefNameForLinkage ? TypedefNameForLinkage
                                               : D->getDeclName();

  if (!Name && !needsAnonymousDeclarationNumber(D)) {
    // Don't bother trying to find unnamed declarations that are in
    // unmergeable contexts.
    FindExistingResult Result(Reader, D, /*Existing=*/nullptr,
                              AnonymousDeclNumber, TypedefNameForLinkage);
    Result.suppress();
    return Result;
  }

  DeclContext *DC = D->getDeclContext()->getRedeclContext();
  if (TypedefNameForLinkage) {
    auto It = Reader.ImportedTypedefNamesForLinkage.find(
        std::make_pair(DC, TypedefNameForLinkage));
    if (It != Reader.ImportedTypedefNamesForLinkage.end())
      if (isSameEntity(It->second, D))
        return FindExistingResult(Reader, D, It->second, AnonymousDeclNumber,
                                  TypedefNameForLinkage);
    // Go on to check in other places in case an existing typedef name
    // was not imported.
  }

  if (needsAnonymousDeclarationNumber(D)) {
    // This is an anonymous declaration that we may need to merge. Look it up
    // in its context by number.
    if (auto *Existing = getAnonymousDeclForMerging(
            Reader, D->getLexicalDeclContext(), AnonymousDeclNumber))
      if (isSameEntity(Existing, D))
        return FindExistingResult(Reader, D, Existing, AnonymousDeclNumber,
                                  TypedefNameForLinkage);
  } else if (DC->isTranslationUnit() &&
             !Reader.getContext().getLangOpts().CPlusPlus) {
    IdentifierResolver &IdResolver = Reader.getIdResolver();

    // Temporarily consider the identifier to be up-to-date. We don't want to
    // cause additional lookups here.
    class UpToDateIdentifierRAII {
      IdentifierInfo *II;
      bool WasOutToDate;

    public:
      explicit UpToDateIdentifierRAII(IdentifierInfo *II)
        : II(II), WasOutToDate(false)
      {
        if (II) {
          WasOutToDate = II->isOutOfDate();
          if (WasOutToDate)
            II->setOutOfDate(false);
        }
      }

      ~UpToDateIdentifierRAII() {
        if (WasOutToDate)
          II->setOutOfDate(true);
      }
    } UpToDate(Name.getAsIdentifierInfo());

    for (IdentifierResolver::iterator I = IdResolver.begin(Name), 
                                   IEnd = IdResolver.end();
         I != IEnd; ++I) {
      if (NamedDecl *Existing = getDeclForMerging(*I, TypedefNameForLinkage))
        if (isSameEntity(Existing, D))
          return FindExistingResult(Reader, D, Existing, AnonymousDeclNumber,
                                    TypedefNameForLinkage);
    }
  } else if (DeclContext *MergeDC = getPrimaryContextForMerging(Reader, DC)) {
    DeclContext::lookup_result R = MergeDC->noload_lookup(Name);
    for (DeclContext::lookup_iterator I = R.begin(), E = R.end(); I != E; ++I) {
      if (NamedDecl *Existing = getDeclForMerging(*I, TypedefNameForLinkage))
        if (isSameEntity(Existing, D))
          return FindExistingResult(Reader, D, Existing, AnonymousDeclNumber,
                                    TypedefNameForLinkage);
    }
  } else {
    // Not in a mergeable context.
    return FindExistingResult(Reader);
  }

  // If this declaration is from a merged context, make a note that we need to
  // check that the canonical definition of that context contains the decl.
  //
  // FIXME: We should do something similar if we merge two definitions of the
  // same template specialization into the same CXXRecordDecl.
  auto MergedDCIt = Reader.MergedDeclContexts.find(D->getLexicalDeclContext());
  if (MergedDCIt != Reader.MergedDeclContexts.end() &&
      MergedDCIt->second == D->getDeclContext())
    Reader.PendingOdrMergeChecks.push_back(D);

  return FindExistingResult(Reader, D, /*Existing=*/nullptr,
                            AnonymousDeclNumber, TypedefNameForLinkage);
}

template<typename DeclT>
Decl *ASTDeclReader::getMostRecentDeclImpl(Redeclarable<DeclT> *D) {
  return D->RedeclLink.getLatestNotUpdated();
}
Decl *ASTDeclReader::getMostRecentDeclImpl(...) {
  llvm_unreachable("getMostRecentDecl on non-redeclarable declaration");
}

Decl *ASTDeclReader::getMostRecentDecl(Decl *D) {
  assert(D);

  switch (D->getKind()) {
#define ABSTRACT_DECL(TYPE)
#define DECL(TYPE, BASE)                               \
  case Decl::TYPE:                                     \
    return getMostRecentDeclImpl(cast<TYPE##Decl>(D));
#include "clang/AST/DeclNodes.inc"
  }
  llvm_unreachable("unknown decl kind");
}

Decl *ASTReader::getMostRecentExistingDecl(Decl *D) {
  return ASTDeclReader::getMostRecentDecl(D->getCanonicalDecl());
}

template<typename DeclT>
void ASTDeclReader::attachPreviousDeclImpl(ASTReader &Reader,
                                           Redeclarable<DeclT> *D,
                                           Decl *Previous, Decl *Canon) {
  D->RedeclLink.setPrevious(cast<DeclT>(Previous));
  D->First = cast<DeclT>(Previous)->First;
}

namespace clang {
template<>
void ASTDeclReader::attachPreviousDeclImpl(ASTReader &Reader,
                                           Redeclarable<FunctionDecl> *D,
                                           Decl *Previous, Decl *Canon) {
  FunctionDecl *FD = static_cast<FunctionDecl*>(D);
  FunctionDecl *PrevFD = cast<FunctionDecl>(Previous);

  FD->RedeclLink.setPrevious(PrevFD);
  FD->First = PrevFD->First;

  // If the previous declaration is an inline function declaration, then this
  // declaration is too.
  if (PrevFD->IsInline != FD->IsInline) {
    // FIXME: [dcl.fct.spec]p4:
    //   If a function with external linkage is declared inline in one
    //   translation unit, it shall be declared inline in all translation
    //   units in which it appears.
    //
    // Be careful of this case:
    //
    // module A:
    //   template<typename T> struct X { void f(); };
    //   template<typename T> inline void X<T>::f() {}
    //
    // module B instantiates the declaration of X<int>::f
    // module C instantiates the definition of X<int>::f
    //
    // If module B and C are merged, we do not have a violation of this rule.
    FD->IsInline = true;
  }

  // If we need to propagate an exception specification along the redecl
  // chain, make a note of that so that we can do so later.
  auto *FPT = FD->getType()->getAs<FunctionProtoType>();
  auto *PrevFPT = PrevFD->getType()->getAs<FunctionProtoType>();
  if (FPT && PrevFPT) {
    bool IsUnresolved = isUnresolvedExceptionSpec(FPT->getExceptionSpecType());
    bool WasUnresolved =
        isUnresolvedExceptionSpec(PrevFPT->getExceptionSpecType());
    if (IsUnresolved != WasUnresolved)
      Reader.PendingExceptionSpecUpdates.insert(
          std::make_pair(Canon, IsUnresolved ? PrevFD : FD));
  }
}
} // end namespace clang

void ASTDeclReader::attachPreviousDeclImpl(ASTReader &Reader, ...) {
  llvm_unreachable("attachPreviousDecl on non-redeclarable declaration");
}

/// Inherit the default template argument from \p From to \p To. Returns
/// \c false if there is no default template for \p From.
template <typename ParmDecl>
static bool inheritDefaultTemplateArgument(ASTContext &Context, ParmDecl *From,
                                           Decl *ToD) {
  auto *To = cast<ParmDecl>(ToD);
  if (!From->hasDefaultArgument())
    return false;
  To->setInheritedDefaultArgument(Context, From);
  return true;
}

static void inheritDefaultTemplateArguments(ASTContext &Context,
                                            TemplateDecl *From,
                                            TemplateDecl *To) {
  auto *FromTP = From->getTemplateParameters();
  auto *ToTP = To->getTemplateParameters();
  assert(FromTP->size() == ToTP->size() && "merged mismatched templates?");

  for (unsigned I = 0, N = FromTP->size(); I != N; ++I) {
    NamedDecl *FromParam = FromTP->getParam(N - I - 1);
    if (FromParam->isParameterPack())
      continue;
    NamedDecl *ToParam = ToTP->getParam(N - I - 1);

    if (auto *FTTP = dyn_cast<TemplateTypeParmDecl>(FromParam)) {
      if (!inheritDefaultTemplateArgument(Context, FTTP, ToParam))
        break;
    } else if (auto *FNTTP = dyn_cast<NonTypeTemplateParmDecl>(FromParam)) {
      if (!inheritDefaultTemplateArgument(Context, FNTTP, ToParam))
        break;
    } else {
      if (!inheritDefaultTemplateArgument(
              Context, cast<TemplateTemplateParmDecl>(FromParam), ToParam))
        break;
    }
  }
}

void ASTDeclReader::attachPreviousDecl(ASTReader &Reader, Decl *D,
                                       Decl *Previous, Decl *Canon) {
  assert(D && Previous);

  switch (D->getKind()) {
#define ABSTRACT_DECL(TYPE)
#define DECL(TYPE, BASE)                                                  \
  case Decl::TYPE:                                                        \
    attachPreviousDeclImpl(Reader, cast<TYPE##Decl>(D), Previous, Canon); \
    break;
#include "clang/AST/DeclNodes.inc"
  }

  // If the declaration was visible in one module, a redeclaration of it in
  // another module remains visible even if it wouldn't be visible by itself.
  //
  // FIXME: In this case, the declaration should only be visible if a module
  //        that makes it visible has been imported.
  D->IdentifierNamespace |=
      Previous->IdentifierNamespace &
      (Decl::IDNS_Ordinary | Decl::IDNS_Tag | Decl::IDNS_Type);

  // If the declaration declares a template, it may inherit default arguments
  // from the previous declaration.
  if (TemplateDecl *TD = dyn_cast<TemplateDecl>(D))
    inheritDefaultTemplateArguments(Reader.getContext(),
                                    cast<TemplateDecl>(Previous), TD);
}

template<typename DeclT>
void ASTDeclReader::attachLatestDeclImpl(Redeclarable<DeclT> *D, Decl *Latest) {
  D->RedeclLink.setLatest(cast<DeclT>(Latest));
}
void ASTDeclReader::attachLatestDeclImpl(...) {
  llvm_unreachable("attachLatestDecl on non-redeclarable declaration");
}

void ASTDeclReader::attachLatestDecl(Decl *D, Decl *Latest) {
  assert(D && Latest);

  switch (D->getKind()) {
#define ABSTRACT_DECL(TYPE)
#define DECL(TYPE, BASE)                                  \
  case Decl::TYPE:                                        \
    attachLatestDeclImpl(cast<TYPE##Decl>(D), Latest); \
    break;
#include "clang/AST/DeclNodes.inc"
  }
}

template<typename DeclT>
void ASTDeclReader::markIncompleteDeclChainImpl(Redeclarable<DeclT> *D) {
  D->RedeclLink.markIncomplete();
}
void ASTDeclReader::markIncompleteDeclChainImpl(...) {
  llvm_unreachable("markIncompleteDeclChain on non-redeclarable declaration");
}

void ASTReader::markIncompleteDeclChain(Decl *D) {
  switch (D->getKind()) {
#define ABSTRACT_DECL(TYPE)
#define DECL(TYPE, BASE)                                             \
  case Decl::TYPE:                                                   \
    ASTDeclReader::markIncompleteDeclChainImpl(cast<TYPE##Decl>(D)); \
    break;
#include "clang/AST/DeclNodes.inc"
  }
}

/// \brief Read the declaration at the given offset from the AST file.
Decl *ASTReader::ReadDeclRecord(DeclID ID) {
  unsigned Index = ID - NUM_PREDEF_DECL_IDS;
  SourceLocation DeclLoc;
  RecordLocation Loc = DeclCursorForID(ID, DeclLoc);
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
  ASTDeclReader Reader(*this, Loc, ID, DeclLoc, Record,Idx);

  Decl *D = nullptr;
  switch ((DeclCode)DeclsCursor.readRecord(Code, Record)) {
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
  case DECL_CONSTRUCTOR_USING_SHADOW:
    D = ConstructorUsingShadowDecl::CreateDeserialized(Context, ID);
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
    D = CXXConstructorDecl::CreateDeserialized(Context, ID, false);
    break;
  case DECL_CXX_INHERITED_CONSTRUCTOR:
    D = CXXConstructorDecl::CreateDeserialized(Context, ID, true);
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
    D = FriendDecl::CreateDeserialized(Context, ID, Record[Idx++]);
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
  case DECL_VAR_TEMPLATE:
    D = VarTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_VAR_TEMPLATE_SPECIALIZATION:
    D = VarTemplateSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_VAR_TEMPLATE_PARTIAL_SPECIALIZATION:
    D = VarTemplatePartialSpecializationDecl::CreateDeserialized(Context, ID);
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
  case DECL_MS_PROPERTY:
    D = MSPropertyDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CAPTURED:
    D = CapturedDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_CXX_BASE_SPECIFIERS:
    Error("attempt to read a C++ base-specifier record as a declaration");
    return nullptr;
  case DECL_CXX_CTOR_INITIALIZERS:
    Error("attempt to read a C++ ctor initializer record as a declaration");
    return nullptr;
  case DECL_IMPORT:
    // Note: last entry of the ImportDecl record is the number of stored source 
    // locations.
    D = ImportDecl::CreateDeserialized(Context, ID, Record.back());
    break;
  case DECL_OMP_THREADPRIVATE:
    D = OMPThreadPrivateDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_OMP_DECLARE_REDUCTION:
    D = OMPDeclareReductionDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OMP_CAPTUREDEXPR:
    D = OMPCapturedExprDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_PRAGMA_COMMENT:
    D = PragmaCommentDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_PRAGMA_DETECT_MISMATCH:
    D = PragmaDetectMismatchDecl::CreateDeserialized(Context, ID,
                                                     Record[Idx++]);
    break;
  case DECL_EMPTY:
    D = EmptyDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_TYPE_PARAM:
    D = ObjCTypeParamDecl::CreateDeserialized(Context, ID);
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
    if (Offsets.first &&
        ReadLexicalDeclContextStorage(*Loc.F, DeclsCursor, Offsets.first, DC))
      return nullptr;
    if (Offsets.second &&
        ReadVisibleDeclContextStorage(*Loc.F, DeclsCursor, Offsets.second, ID))
      return nullptr;
  }
  assert(Idx == Record.size());

  // Load any relevant update records.
  PendingUpdateRecords.push_back(std::make_pair(ID, D));

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
    auto UpdateOffsets = std::move(UpdI->second);
    DeclUpdateOffsets.erase(UpdI);

    bool WasInteresting = isConsumerInterestedIn(D, false);
    for (auto &FileAndOffset : UpdateOffsets) {
      ModuleFile *F = FileAndOffset.first;
      uint64_t Offset = FileAndOffset.second;
      llvm::BitstreamCursor &Cursor = F->DeclsCursor;
      SavedStreamPosition SavedPosition(Cursor);
      Cursor.JumpToBit(Offset);
      RecordData Record;
      unsigned Code = Cursor.ReadCode();
      unsigned RecCode = Cursor.readRecord(Code, Record);
      (void)RecCode;
      assert(RecCode == DECL_UPDATES && "Expected DECL_UPDATES record!");

      unsigned Idx = 0;
      ASTDeclReader Reader(*this, RecordLocation(F, Offset), ID,
                           SourceLocation(), Record, Idx);
      Reader.UpdateDecl(D, *F, Record);

      // We might have made this declaration interesting. If so, remember that
      // we need to hand it off to the consumer.
      if (!WasInteresting &&
          isConsumerInterestedIn(D, Reader.hasPendingBody())) {
        InterestingDecls.push_back(D);
        WasInteresting = true;
      }
    }
  }

  // Load the pending visible updates for this decl context, if it has any.
  auto I = PendingVisibleUpdates.find(ID);
  if (I != PendingVisibleUpdates.end()) {
    auto VisibleUpdates = std::move(I->second);
    PendingVisibleUpdates.erase(I);

    auto *DC = cast<DeclContext>(D)->getPrimaryContext();
    for (const PendingVisibleUpdate &Update : VisibleUpdates)
      Lookups[DC].Table.add(
          Update.Mod, Update.Data,
          reader::ASTDeclContextNameLookupTrait(*this, *Update.Mod));
    DC->setHasExternalVisibleStorage(true);
  }
}

void ASTReader::loadPendingDeclChain(Decl *FirstLocal, uint64_t LocalOffset) {
  // Attach FirstLocal to the end of the decl chain.
  Decl *CanonDecl = FirstLocal->getCanonicalDecl();
  if (FirstLocal != CanonDecl) {
    Decl *PrevMostRecent = ASTDeclReader::getMostRecentDecl(CanonDecl);
    ASTDeclReader::attachPreviousDecl(
        *this, FirstLocal, PrevMostRecent ? PrevMostRecent : CanonDecl,
        CanonDecl);
  }

  if (!LocalOffset) {
    ASTDeclReader::attachLatestDecl(CanonDecl, FirstLocal);
    return;
  }

  // Load the list of other redeclarations from this module file.
  ModuleFile *M = getOwningModuleFile(FirstLocal);
  assert(M && "imported decl from no module file");

  llvm::BitstreamCursor &Cursor = M->DeclsCursor;
  SavedStreamPosition SavedPosition(Cursor);
  Cursor.JumpToBit(LocalOffset);

  RecordData Record;
  unsigned Code = Cursor.ReadCode();
  unsigned RecCode = Cursor.readRecord(Code, Record);
  (void)RecCode;
  assert(RecCode == LOCAL_REDECLARATIONS && "expected LOCAL_REDECLARATIONS record!");

  // FIXME: We have several different dispatches on decl kind here; maybe
  // we should instead generate one loop per kind and dispatch up-front?
  Decl *MostRecent = FirstLocal;
  for (unsigned I = 0, N = Record.size(); I != N; ++I) {
    auto *D = GetLocalDecl(*M, Record[N - I - 1]);
    ASTDeclReader::attachPreviousDecl(*this, D, MostRecent, CanonDecl);
    MostRecent = D;
  }
  ASTDeclReader::attachLatestDecl(CanonDecl, MostRecent);
}

namespace {
  /// \brief Given an ObjC interface, goes through the modules and links to the
  /// interface all the categories for it.
  class ObjCCategoriesVisitor {
    ASTReader &Reader;
    serialization::GlobalDeclID InterfaceID;
    ObjCInterfaceDecl *Interface;
    llvm::SmallPtrSetImpl<ObjCCategoryDecl *> &Deserialized;
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
        Interface->setCategoryListRaw(Cat);
      Tail = Cat;
    }
    
  public:
    ObjCCategoriesVisitor(ASTReader &Reader,
                          serialization::GlobalDeclID InterfaceID,
                          ObjCInterfaceDecl *Interface,
                        llvm::SmallPtrSetImpl<ObjCCategoryDecl *> &Deserialized,
                          unsigned PreviousGeneration)
      : Reader(Reader), InterfaceID(InterfaceID), Interface(Interface),
        Deserialized(Deserialized), PreviousGeneration(PreviousGeneration),
        Tail(nullptr)
    {
      // Populate the name -> category map with the set of known categories.
      for (auto *Cat : Interface->known_categories()) {
        if (Cat->getDeclName())
          NameCategoryMap[Cat->getDeclName()] = Cat;
        
        // Keep track of the tail of the category list.
        Tail = Cat;
      }
    }

    bool operator()(ModuleFile &M) {
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
      const ObjCCategoriesInfo Compare = { LocalID, 0 };
      const ObjCCategoriesInfo *Result
        = std::lower_bound(M.ObjCCategoriesMap,
                           M.ObjCCategoriesMap + M.LocalNumObjCCategoriesInMap, 
                           Compare);
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
} // end anonymous namespace

void ASTReader::loadObjCCategories(serialization::GlobalDeclID ID,
                                   ObjCInterfaceDecl *D,
                                   unsigned PreviousGeneration) {
  ObjCCategoriesVisitor Visitor(*this, ID, D, CategoriesDeserialized,
                                PreviousGeneration);
  ModuleMgr.visit(Visitor);
}

template<typename DeclT, typename Fn>
static void forAllLaterRedecls(DeclT *D, Fn F) {
  F(D);

  // Check whether we've already merged D into its redeclaration chain.
  // MostRecent may or may not be nullptr if D has not been merged. If
  // not, walk the merged redecl chain and see if it's there.
  auto *MostRecent = D->getMostRecentDecl();
  bool Found = false;
  for (auto *Redecl = MostRecent; Redecl && !Found;
       Redecl = Redecl->getPreviousDecl())
    Found = (Redecl == D);

  // If this declaration is merged, apply the functor to all later decls.
  if (Found) {
    for (auto *Redecl = MostRecent; Redecl != D;
         Redecl = Redecl->getPreviousDecl())
      F(Redecl);
  }
}

void ASTDeclReader::UpdateDecl(Decl *D, ModuleFile &ModuleFile,
                               const RecordData &Record) {
  while (Idx < Record.size()) {
    switch ((DeclUpdateKind)Record[Idx++]) {
    case UPD_CXX_ADDED_IMPLICIT_MEMBER: {
      auto *RD = cast<CXXRecordDecl>(D);
      // FIXME: If we also have an update record for instantiating the
      // definition of D, we need that to happen before we get here.
      Decl *MD = Reader.ReadDecl(ModuleFile, Record, Idx);
      assert(MD && "couldn't read decl from update record");
      // FIXME: We should call addHiddenDecl instead, to add the member
      // to its DeclContext.
      RD->addedMember(MD);
      break;
    }

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
      if (ModuleFile.Kind != MK_ImplicitModule &&
          ModuleFile.Kind != MK_ExplicitModule) {
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

    case UPD_CXX_INSTANTIATED_DEFAULT_ARGUMENT: {
      auto Param = cast<ParmVarDecl>(D);

      // We have to read the default argument regardless of whether we use it
      // so that hypothetical further update records aren't messed up.
      // TODO: Add a function to skip over the next expr record.
      auto DefaultArg = Reader.ReadExpr(F);

      // Only apply the update if the parameter still has an uninstantiated
      // default argument.
      if (Param->hasUninstantiatedDefaultArg())
        Param->setDefaultArg(DefaultArg);
      break;
    }

    case UPD_CXX_ADDED_FUNCTION_DEFINITION: {
      FunctionDecl *FD = cast<FunctionDecl>(D);
      if (Reader.PendingBodies[FD]) {
        // FIXME: Maybe check for ODR violations.
        // It's safe to stop now because this update record is always last.
        return;
      }

      if (Record[Idx++]) {
        // Maintain AST consistency: any later redeclarations of this function
        // are inline if this one is. (We might have merged another declaration
        // into this one.)
        forAllLaterRedecls(FD, [](FunctionDecl *FD) {
          FD->setImplicitlyInline();
        });
      }
      FD->setInnerLocStart(Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      if (auto *CD = dyn_cast<CXXConstructorDecl>(FD)) {
        CD->NumCtorInitializers = Record[Idx++];
        if (CD->NumCtorInitializers)
          CD->CtorInitializers = ReadGlobalOffset(F, Record, Idx);
      }
      // Store the offset of the body so we can lazily load it later.
      Reader.PendingBodies[FD] = GetCurrentCursorOffset();
      HasPendingBody = true;
      assert(Idx == Record.size() && "lazy body must be last");
      break;
    }

    case UPD_CXX_INSTANTIATED_CLASS_DEFINITION: {
      auto *RD = cast<CXXRecordDecl>(D);
      auto *OldDD = RD->getCanonicalDecl()->DefinitionData;
      bool HadRealDefinition =
          OldDD && (OldDD->Definition != RD ||
                    !Reader.PendingFakeDefinitionData.count(OldDD));
      ReadCXXRecordDefinition(RD, /*Update*/true);

      // Visible update is handled separately.
      uint64_t LexicalOffset = ReadLocalOffset(Record, Idx);
      if (!HadRealDefinition && LexicalOffset) {
        Reader.ReadLexicalDeclContextStorage(ModuleFile, ModuleFile.DeclsCursor,
                                             LexicalOffset, RD);
        Reader.PendingFakeDefinitionData.erase(OldDD);
      }

      auto TSK = (TemplateSpecializationKind)Record[Idx++];
      SourceLocation POI = Reader.ReadSourceLocation(ModuleFile, Record, Idx);
      if (MemberSpecializationInfo *MSInfo =
              RD->getMemberSpecializationInfo()) {
        MSInfo->setTemplateSpecializationKind(TSK);
        MSInfo->setPointOfInstantiation(POI);
      } else {
        ClassTemplateSpecializationDecl *Spec =
            cast<ClassTemplateSpecializationDecl>(RD);
        Spec->setTemplateSpecializationKind(TSK);
        Spec->setPointOfInstantiation(POI);

        if (Record[Idx++]) {
          auto PartialSpec =
              ReadDeclAs<ClassTemplatePartialSpecializationDecl>(Record, Idx);
          SmallVector<TemplateArgument, 8> TemplArgs;
          Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
          auto *TemplArgList = TemplateArgumentList::CreateCopy(
              Reader.getContext(), TemplArgs);

          // FIXME: If we already have a partial specialization set,
          // check that it matches.
          if (!Spec->getSpecializedTemplateOrPartial()
                   .is<ClassTemplatePartialSpecializationDecl *>())
            Spec->setInstantiationOf(PartialSpec, TemplArgList);
        }
      }

      RD->setTagKind((TagTypeKind)Record[Idx++]);
      RD->setLocation(Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      RD->setLocStart(Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      RD->setRBraceLoc(Reader.ReadSourceLocation(ModuleFile, Record, Idx));

      if (Record[Idx++]) {
        AttrVec Attrs;
        Reader.ReadAttributes(F, Attrs, Record, Idx);
        D->setAttrsImpl(Attrs, Reader.getContext());
      }
      break;
    }

    case UPD_CXX_RESOLVED_DTOR_DELETE: {
      // Set the 'operator delete' directly to avoid emitting another update
      // record.
      auto *Del = Reader.ReadDeclAs<FunctionDecl>(ModuleFile, Record, Idx);
      auto *First = cast<CXXDestructorDecl>(D->getCanonicalDecl());
      // FIXME: Check consistency if we have an old and new operator delete.
      if (!First->OperatorDelete)
        First->OperatorDelete = Del;
      break;
    }

    case UPD_CXX_RESOLVED_EXCEPTION_SPEC: {
      FunctionProtoType::ExceptionSpecInfo ESI;
      SmallVector<QualType, 8> ExceptionStorage;
      Reader.readExceptionSpec(ModuleFile, ExceptionStorage, ESI, Record, Idx);

      // Update this declaration's exception specification, if needed.
      auto *FD = cast<FunctionDecl>(D);
      auto *FPT = FD->getType()->castAs<FunctionProtoType>();
      // FIXME: If the exception specification is already present, check that it
      // matches.
      if (isUnresolvedExceptionSpec(FPT->getExceptionSpecType())) {
        FD->setType(Reader.Context.getFunctionType(
            FPT->getReturnType(), FPT->getParamTypes(),
            FPT->getExtProtoInfo().withExceptionSpec(ESI)));

        // When we get to the end of deserializing, see if there are other decls
        // that we need to propagate this exception specification onto.
        Reader.PendingExceptionSpecUpdates.insert(
            std::make_pair(FD->getCanonicalDecl(), FD));
      }
      break;
    }

    case UPD_CXX_DEDUCED_RETURN_TYPE: {
      // FIXME: Also do this when merging redecls.
      QualType DeducedResultType = Reader.readType(ModuleFile, Record, Idx);
      for (auto *Redecl : merged_redecls(D)) {
        // FIXME: If the return type is already deduced, check that it matches.
        FunctionDecl *FD = cast<FunctionDecl>(Redecl);
        Reader.Context.adjustDeducedFunctionResultType(FD, DeducedResultType);
      }
      break;
    }

    case UPD_DECL_MARKED_USED: {
      // FIXME: This doesn't send the right notifications if there are
      // ASTMutationListeners other than an ASTWriter.

      // Maintain AST consistency: any later redeclarations are used too.
      D->setIsUsed();
      break;
    }

    case UPD_MANGLING_NUMBER:
      Reader.Context.setManglingNumber(cast<NamedDecl>(D), Record[Idx++]);
      break;

    case UPD_STATIC_LOCAL_NUMBER:
      Reader.Context.setStaticLocalNumber(cast<VarDecl>(D), Record[Idx++]);
      break;

    case UPD_DECL_MARKED_OPENMP_THREADPRIVATE:
      D->addAttr(OMPThreadPrivateDeclAttr::CreateImplicit(
          Reader.Context, ReadSourceRange(Record, Idx)));
      break;

    case UPD_DECL_EXPORTED: {
      unsigned SubmoduleID = readSubmoduleID(Record, Idx);
      auto *Exported = cast<NamedDecl>(D);
      if (auto *TD = dyn_cast<TagDecl>(Exported))
        Exported = TD->getDefinition();
      Module *Owner = SubmoduleID ? Reader.getSubmodule(SubmoduleID) : nullptr;
      if (Reader.getContext().getLangOpts().ModulesLocalVisibility) {
        // FIXME: This doesn't send the right notifications if there are
        // ASTMutationListeners other than an ASTWriter.
        Reader.getContext().mergeDefinitionIntoModule(
            cast<NamedDecl>(Exported), Owner,
            /*NotifyListeners*/ false);
        Reader.PendingMergedDefinitionsToDeduplicate.insert(
            cast<NamedDecl>(Exported));
      } else if (Owner && Owner->NameVisibility != Module::AllVisible) {
        // If Owner is made visible at some later point, make this declaration
        // visible too.
        Reader.HiddenNamesMap[Owner].push_back(Exported);
      } else {
        // The declaration is now visible.
        Exported->Hidden = false;
      }
      break;
    }

    case UPD_DECL_MARKED_OPENMP_DECLARETARGET:
    case UPD_ADDED_ATTR_TO_RECORD:
      AttrVec Attrs;
      Reader.ReadAttributes(F, Attrs, Record, Idx);
      assert(Attrs.size() == 1);
      D->addAttr(Attrs[0]);
      break;
    }
  }
}
