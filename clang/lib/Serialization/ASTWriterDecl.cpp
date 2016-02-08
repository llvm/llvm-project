//===--- ASTWriterDecl.cpp - Declaration Serialization --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements serialization for Declarations.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTWriter.h"
#include "ASTCommon.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;
using namespace serialization;

//===----------------------------------------------------------------------===//
// Declaration serialization
//===----------------------------------------------------------------------===//

namespace clang {
  class ASTDeclWriter : public DeclVisitor<ASTDeclWriter, void> {

    ASTWriter &Writer;
    ASTContext &Context;
    typedef ASTWriter::RecordData RecordData;
    RecordData &Record;

  public:
    serialization::DeclCode Code;
    unsigned AbbrevToUse;

    ASTDeclWriter(ASTWriter &Writer, ASTContext &Context, RecordData &Record)
      : Writer(Writer), Context(Context), Record(Record) {
    }

    void Visit(Decl *D);

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    void VisitNamedDecl(NamedDecl *D);
    void VisitLabelDecl(LabelDecl *LD);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeDecl(TypeDecl *D);
    void VisitTypedefNameDecl(TypedefNameDecl *D);
    void VisitTypedefDecl(TypedefDecl *D);
    void VisitTypeAliasDecl(TypeAliasDecl *D);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    void VisitTagDecl(TagDecl *D);
    void VisitEnumDecl(EnumDecl *D);
    void VisitRecordDecl(RecordDecl *D);
    void VisitCXXRecordDecl(CXXRecordDecl *D);
    void VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
    void VisitVarTemplateSpecializationDecl(VarTemplateSpecializationDecl *D);
    void VisitVarTemplatePartialSpecializationDecl(
        VarTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
                                       ClassScopeFunctionSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    void VisitValueDecl(ValueDecl *D);
    void VisitEnumConstantDecl(EnumConstantDecl *D);
    void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    void VisitDeclaratorDecl(DeclaratorDecl *D);
    void VisitFunctionDecl(FunctionDecl *D);
    void VisitCXXMethodDecl(CXXMethodDecl *D);
    void VisitCXXConstructorDecl(CXXConstructorDecl *D);
    void VisitCXXDestructorDecl(CXXDestructorDecl *D);
    void VisitCXXConversionDecl(CXXConversionDecl *D);
    void VisitFieldDecl(FieldDecl *D);
    void VisitMSPropertyDecl(MSPropertyDecl *D);
    void VisitIndirectFieldDecl(IndirectFieldDecl *D);
    void VisitVarDecl(VarDecl *D);
    void VisitImplicitParamDecl(ImplicitParamDecl *D);
    void VisitParmVarDecl(ParmVarDecl *D);
    void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    void VisitTemplateDecl(TemplateDecl *D);
    void VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitVarTemplateDecl(VarTemplateDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    void VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    void VisitImportDecl(ImportDecl *D);
    void VisitAccessSpecDecl(AccessSpecDecl *D);
    void VisitFriendDecl(FriendDecl *D);
    void VisitFriendTemplateDecl(FriendTemplateDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitBlockDecl(BlockDecl *D);
    void VisitCapturedDecl(CapturedDecl *D);
    void VisitEmptyDecl(EmptyDecl *D);

    void VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset,
                          uint64_t VisibleOffset);
    template <typename T> void VisitRedeclarable(Redeclarable<T> *D);


    // FIXME: Put in the same order is DeclNodes.td?
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
    void VisitOMPCapturedFieldDecl(OMPCapturedFieldDecl *D);

    /// Add an Objective-C type parameter list to the given record.
    void AddObjCTypeParamList(ObjCTypeParamList *typeParams) {
      // Empty type parameter list.
      if (!typeParams) {
        Record.push_back(0);
        return;
      }

      Record.push_back(typeParams->size());
      for (auto typeParam : *typeParams) {
        Writer.AddDeclRef(typeParam, Record);
      }
      Writer.AddSourceLocation(typeParams->getLAngleLoc(), Record);
      Writer.AddSourceLocation(typeParams->getRAngleLoc(), Record);
    }

    void AddFunctionDefinition(const FunctionDecl *FD) {
      assert(FD->doesThisDeclarationHaveABody());
      if (auto *CD = dyn_cast<CXXConstructorDecl>(FD)) {
        Record.push_back(CD->NumCtorInitializers);
        if (CD->NumCtorInitializers)
          Writer.AddCXXCtorInitializersRef(
              llvm::makeArrayRef(CD->init_begin(), CD->init_end()), Record);
      }
      Writer.AddStmt(FD->getBody());
    }

    /// Add to the record the first declaration from each module file that
    /// provides a declaration of D. The intent is to provide a sufficient
    /// set such that reloading this set will load all current redeclarations.
    void AddFirstDeclFromEachModule(const Decl *D, bool IncludeLocal) {
      llvm::MapVector<ModuleFile*, const Decl*> Firsts;
      // FIXME: We can skip entries that we know are implied by others.
      for (const Decl *R = D->getMostRecentDecl(); R; R = R->getPreviousDecl()) {
        if (R->isFromASTFile())
          Firsts[Writer.Chain->getOwningModuleFile(R)] = R;
        else if (IncludeLocal)
          Firsts[nullptr] = R;
      }
      for (const auto &F : Firsts)
        Writer.AddDeclRef(F.second, Record);
    }

    /// Get the specialization decl from an entry in the specialization list.
    template <typename EntryType>
    typename RedeclarableTemplateDecl::SpecEntryTraits<EntryType>::DeclType *
    getSpecializationDecl(EntryType &T) {
      return RedeclarableTemplateDecl::SpecEntryTraits<EntryType>::getDecl(&T);
    }

    /// Get the list of partial specializations from a template's common ptr.
    template<typename T>
    decltype(T::PartialSpecializations) &getPartialSpecializations(T *Common) {
      return Common->PartialSpecializations;
    }
    ArrayRef<Decl> getPartialSpecializations(FunctionTemplateDecl::Common *) {
      return None;
    }

    template<typename Decl>
    void AddTemplateSpecializations(Decl *D) {
      auto *Common = D->getCommonPtr();

      // If we have any lazy specializations, and the external AST source is
      // our chained AST reader, we can just write out the DeclIDs. Otherwise,
      // we need to resolve them to actual declarations.
      if (Writer.Chain != Writer.Context->getExternalSource() &&
          Common->LazySpecializations) {
        D->LoadLazySpecializations();
        assert(!Common->LazySpecializations);
      }

      auto &Specializations = Common->Specializations;
      auto &&PartialSpecializations = getPartialSpecializations(Common);
      ArrayRef<DeclID> LazySpecializations;
      if (auto *LS = Common->LazySpecializations)
        LazySpecializations = llvm::makeArrayRef(LS + 1, LS[0]);

      // Add a slot to the record for the number of specializations.
      unsigned I = Record.size();
      Record.push_back(0);

      for (auto &Entry : Specializations) {
        auto *D = getSpecializationDecl(Entry);
        assert(D->isCanonicalDecl() && "non-canonical decl in set");
        AddFirstDeclFromEachModule(D, /*IncludeLocal*/true);
      }
      for (auto &Entry : PartialSpecializations) {
        auto *D = getSpecializationDecl(Entry);
        assert(D->isCanonicalDecl() && "non-canonical decl in set");
        AddFirstDeclFromEachModule(D, /*IncludeLocal*/true);
      }
      Record.append(LazySpecializations.begin(), LazySpecializations.end());

      // Update the size entry we added earlier.
      Record[I] = Record.size() - I - 1;
    }

    /// Ensure that this template specialization is associated with the specified
    /// template on reload.
    void RegisterTemplateSpecialization(const Decl *Template,
                                        const Decl *Specialization) {
      Template = Template->getCanonicalDecl();

      // If the canonical template is local, we'll write out this specialization
      // when we emit it.
      // FIXME: We can do the same thing if there is any local declaration of
      // the template, to avoid emitting an update record.
      if (!Template->isFromASTFile())
        return;

      // We only need to associate the first local declaration of the
      // specialization. The other declarations will get pulled in by it.
      if (Writer.getFirstLocalDecl(Specialization) != Specialization)
        return;

      Writer.DeclUpdates[Template].push_back(ASTWriter::DeclUpdate(
          UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION, Specialization));
    }
  };
}

void ASTDeclWriter::Visit(Decl *D) {
  DeclVisitor<ASTDeclWriter>::Visit(D);

  // Source locations require array (variable-length) abbreviations.  The
  // abbreviation infrastructure requires that arrays are encoded last, so
  // we handle it here in the case of those classes derived from DeclaratorDecl
  if (DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D)) {
    Writer.AddTypeSourceInfo(DD->getTypeSourceInfo(), Record);
  }

  // Handle FunctionDecl's body here and write it after all other Stmts/Exprs
  // have been written. We want it last because we will not read it back when
  // retrieving it from the AST, we'll just lazily set the offset. 
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    Record.push_back(FD->doesThisDeclarationHaveABody());
    if (FD->doesThisDeclarationHaveABody())
      AddFunctionDefinition(FD);
  }
}

void ASTDeclWriter::VisitDecl(Decl *D) {
  Writer.AddDeclRef(cast_or_null<Decl>(D->getDeclContext()), Record);
  if (D->getDeclContext() != D->getLexicalDeclContext())
    Writer.AddDeclRef(cast_or_null<Decl>(D->getLexicalDeclContext()), Record);
  else
    Record.push_back(0);
  Record.push_back(D->isInvalidDecl());
  Record.push_back(D->hasAttrs());
  if (D->hasAttrs())
    Writer.WriteAttributes(llvm::makeArrayRef(D->getAttrs().begin(),
                                              D->getAttrs().size()), Record);
  Record.push_back(D->isImplicit());
  Record.push_back(D->isUsed(false));
  Record.push_back(D->isReferenced());
  Record.push_back(D->isTopLevelDeclInObjCContainer());
  Record.push_back(D->getAccess());
  Record.push_back(D->isModulePrivate());
  Record.push_back(Writer.inferSubmoduleIDFromLocation(D->getLocation()));

  // If this declaration injected a name into a context different from its
  // lexical context, and that context is an imported namespace, we need to
  // update its visible declarations to include this name.
  //
  // This happens when we instantiate a class with a friend declaration or a
  // function with a local extern declaration, for instance.
  //
  // FIXME: Can we handle this in AddedVisibleDecl instead?
  if (D->isOutOfLine()) {
    auto *DC = D->getDeclContext();
    while (auto *NS = dyn_cast<NamespaceDecl>(DC->getRedeclContext())) {
      if (!NS->isFromASTFile())
        break;
      Writer.UpdatedDeclContexts.insert(NS->getPrimaryContext());
      if (!NS->isInlineNamespace())
        break;
      DC = NS->getParent();
    }
  }
}

void ASTDeclWriter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  llvm_unreachable("Translation units aren't directly serialized");
}

void ASTDeclWriter::VisitNamedDecl(NamedDecl *D) {
  VisitDecl(D);
  Writer.AddDeclarationName(D->getDeclName(), Record);
  Record.push_back(needsAnonymousDeclarationNumber(D)
                       ? Writer.getAnonymousDeclarationNumber(D)
                       : 0);
}

void ASTDeclWriter::VisitTypeDecl(TypeDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getLocStart(), Record);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
}

void ASTDeclWriter::VisitTypedefNameDecl(TypedefNameDecl *D) {
  VisitRedeclarable(D);
  VisitTypeDecl(D);
  Writer.AddTypeSourceInfo(D->getTypeSourceInfo(), Record);
  Record.push_back(D->isModed());
  if (D->isModed())
    Writer.AddTypeRef(D->getUnderlyingType(), Record);
}

void ASTDeclWriter::VisitTypedefDecl(TypedefDecl *D) {
  VisitTypedefNameDecl(D);
  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      D->getFirstDecl() == D->getMostRecentDecl() &&
      !D->isInvalidDecl() &&
      !D->isTopLevelDeclInObjCContainer() &&
      !D->isModulePrivate() &&
      !needsAnonymousDeclarationNumber(D) &&
      D->getDeclName().getNameKind() == DeclarationName::Identifier)
    AbbrevToUse = Writer.getDeclTypedefAbbrev();

  Code = serialization::DECL_TYPEDEF;
}

void ASTDeclWriter::VisitTypeAliasDecl(TypeAliasDecl *D) {
  VisitTypedefNameDecl(D);
  Writer.AddDeclRef(D->getDescribedAliasTemplate(), Record);
  Code = serialization::DECL_TYPEALIAS;
}

void ASTDeclWriter::VisitTagDecl(TagDecl *D) {
  VisitRedeclarable(D);
  VisitTypeDecl(D);
  Record.push_back(D->getIdentifierNamespace());
  Record.push_back((unsigned)D->getTagKind()); // FIXME: stable encoding
  if (!isa<CXXRecordDecl>(D))
    Record.push_back(D->isCompleteDefinition());
  Record.push_back(D->isEmbeddedInDeclarator());
  Record.push_back(D->isFreeStanding());
  Record.push_back(D->isCompleteDefinitionRequired());
  Writer.AddSourceLocation(D->getRBraceLoc(), Record);

  if (D->hasExtInfo()) {
    Record.push_back(1);
    Writer.AddQualifierInfo(*D->getExtInfo(), Record);
  } else if (auto *TD = D->getTypedefNameForAnonDecl()) {
    Record.push_back(2);
    Writer.AddDeclRef(TD, Record);
    Writer.AddIdentifierRef(TD->getDeclName().getAsIdentifierInfo(), Record);
  } else {
    Record.push_back(0);
  }
}

void ASTDeclWriter::VisitEnumDecl(EnumDecl *D) {
  VisitTagDecl(D);
  Writer.AddTypeSourceInfo(D->getIntegerTypeSourceInfo(), Record);
  if (!D->getIntegerTypeSourceInfo())
    Writer.AddTypeRef(D->getIntegerType(), Record);
  Writer.AddTypeRef(D->getPromotionType(), Record);
  Record.push_back(D->getNumPositiveBits());
  Record.push_back(D->getNumNegativeBits());
  Record.push_back(D->isScoped());
  Record.push_back(D->isScopedUsingClassTag());
  Record.push_back(D->isFixed());
  if (MemberSpecializationInfo *MemberInfo = D->getMemberSpecializationInfo()) {
    Writer.AddDeclRef(MemberInfo->getInstantiatedFrom(), Record);
    Record.push_back(MemberInfo->getTemplateSpecializationKind());
    Writer.AddSourceLocation(MemberInfo->getPointOfInstantiation(), Record);
  } else {
    Writer.AddDeclRef(nullptr, Record);
  }

  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      !D->isUsed(false) &&
      !D->hasExtInfo() &&
      !D->getTypedefNameForAnonDecl() &&
      D->getFirstDecl() == D->getMostRecentDecl() &&
      !D->isInvalidDecl() &&
      !D->isReferenced() &&
      !D->isTopLevelDeclInObjCContainer() &&
      D->getAccess() == AS_none &&
      !D->isModulePrivate() &&
      !CXXRecordDecl::classofKind(D->getKind()) &&
      !D->getIntegerTypeSourceInfo() &&
      !D->getMemberSpecializationInfo() &&
      !needsAnonymousDeclarationNumber(D) &&
      D->getDeclName().getNameKind() == DeclarationName::Identifier)
    AbbrevToUse = Writer.getDeclEnumAbbrev();

  Code = serialization::DECL_ENUM;
}

void ASTDeclWriter::VisitRecordDecl(RecordDecl *D) {
  VisitTagDecl(D);
  Record.push_back(D->hasFlexibleArrayMember());
  Record.push_back(D->isAnonymousStructOrUnion());
  Record.push_back(D->hasObjectMember());
  Record.push_back(D->hasVolatileMember());

  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      !D->isUsed(false) &&
      !D->hasExtInfo() &&
      !D->getTypedefNameForAnonDecl() &&
      D->getFirstDecl() == D->getMostRecentDecl() &&
      !D->isInvalidDecl() &&
      !D->isReferenced() &&
      !D->isTopLevelDeclInObjCContainer() &&
      D->getAccess() == AS_none &&
      !D->isModulePrivate() &&
      !CXXRecordDecl::classofKind(D->getKind()) &&
      !needsAnonymousDeclarationNumber(D) &&
      D->getDeclName().getNameKind() == DeclarationName::Identifier)
    AbbrevToUse = Writer.getDeclRecordAbbrev();

  Code = serialization::DECL_RECORD;
}

void ASTDeclWriter::VisitValueDecl(ValueDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(D->getType(), Record);
}

void ASTDeclWriter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->getInitExpr()? 1 : 0);
  if (D->getInitExpr())
    Writer.AddStmt(D->getInitExpr());
  Writer.AddAPSInt(D->getInitVal(), Record);

  Code = serialization::DECL_ENUM_CONSTANT;
}

void ASTDeclWriter::VisitDeclaratorDecl(DeclaratorDecl *D) {
  VisitValueDecl(D);
  Writer.AddSourceLocation(D->getInnerLocStart(), Record);
  Record.push_back(D->hasExtInfo());
  if (D->hasExtInfo())
    Writer.AddQualifierInfo(*D->getExtInfo(), Record);
}

void ASTDeclWriter::VisitFunctionDecl(FunctionDecl *D) {
  VisitRedeclarable(D);
  VisitDeclaratorDecl(D);
  Writer.AddDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record);
  Record.push_back(D->getIdentifierNamespace());
  
  // FunctionDecl's body is handled last at ASTWriterDecl::Visit,
  // after everything else is written.
  
  Record.push_back((int)D->SClass); // FIXME: stable encoding
  Record.push_back(D->IsInline);
  Record.push_back(D->IsInlineSpecified);
  Record.push_back(D->IsVirtualAsWritten);
  Record.push_back(D->IsPure);
  Record.push_back(D->HasInheritedPrototype);
  Record.push_back(D->HasWrittenPrototype);
  Record.push_back(D->IsDeleted);
  Record.push_back(D->IsTrivial);
  Record.push_back(D->IsDefaulted);
  Record.push_back(D->IsExplicitlyDefaulted);
  Record.push_back(D->HasImplicitReturnZero);
  Record.push_back(D->IsConstexpr);
  Record.push_back(D->HasSkippedBody);
  Record.push_back(D->IsLateTemplateParsed);
  Record.push_back(D->getLinkageInternal());
  Writer.AddSourceLocation(D->getLocEnd(), Record);

  Record.push_back(D->getTemplatedKind());
  switch (D->getTemplatedKind()) {
  case FunctionDecl::TK_NonTemplate:
    break;
  case FunctionDecl::TK_FunctionTemplate:
    Writer.AddDeclRef(D->getDescribedFunctionTemplate(), Record);
    break;
  case FunctionDecl::TK_MemberSpecialization: {
    MemberSpecializationInfo *MemberInfo = D->getMemberSpecializationInfo();
    Writer.AddDeclRef(MemberInfo->getInstantiatedFrom(), Record);
    Record.push_back(MemberInfo->getTemplateSpecializationKind());
    Writer.AddSourceLocation(MemberInfo->getPointOfInstantiation(), Record);
    break;
  }
  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    FunctionTemplateSpecializationInfo *
      FTSInfo = D->getTemplateSpecializationInfo();

    RegisterTemplateSpecialization(FTSInfo->getTemplate(), D);

    Writer.AddDeclRef(FTSInfo->getTemplate(), Record);
    Record.push_back(FTSInfo->getTemplateSpecializationKind());
    
    // Template arguments.
    Writer.AddTemplateArgumentList(FTSInfo->TemplateArguments, Record);
    
    // Template args as written.
    Record.push_back(FTSInfo->TemplateArgumentsAsWritten != nullptr);
    if (FTSInfo->TemplateArgumentsAsWritten) {
      Record.push_back(FTSInfo->TemplateArgumentsAsWritten->NumTemplateArgs);
      for (int i=0, e = FTSInfo->TemplateArgumentsAsWritten->NumTemplateArgs;
             i!=e; ++i)
        Writer.AddTemplateArgumentLoc((*FTSInfo->TemplateArgumentsAsWritten)[i],
                                      Record);
      Writer.AddSourceLocation(FTSInfo->TemplateArgumentsAsWritten->LAngleLoc,
                               Record);
      Writer.AddSourceLocation(FTSInfo->TemplateArgumentsAsWritten->RAngleLoc,
                               Record);
    }
    
    Writer.AddSourceLocation(FTSInfo->getPointOfInstantiation(), Record);

    if (D->isCanonicalDecl()) {
      // Write the template that contains the specializations set. We will
      // add a FunctionTemplateSpecializationInfo to it when reading.
      Writer.AddDeclRef(FTSInfo->getTemplate()->getCanonicalDecl(), Record);
    }
    break;
  }
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
    DependentFunctionTemplateSpecializationInfo *
      DFTSInfo = D->getDependentSpecializationInfo();
    
    // Templates.
    Record.push_back(DFTSInfo->getNumTemplates());
    for (int i=0, e = DFTSInfo->getNumTemplates(); i != e; ++i)
      Writer.AddDeclRef(DFTSInfo->getTemplate(i), Record);
    
    // Templates args.
    Record.push_back(DFTSInfo->getNumTemplateArgs());
    for (int i=0, e = DFTSInfo->getNumTemplateArgs(); i != e; ++i)
      Writer.AddTemplateArgumentLoc(DFTSInfo->getTemplateArg(i), Record);
    Writer.AddSourceLocation(DFTSInfo->getLAngleLoc(), Record);
    Writer.AddSourceLocation(DFTSInfo->getRAngleLoc(), Record);
    break;
  }
  }

  Record.push_back(D->param_size());
  for (auto P : D->params())
    Writer.AddDeclRef(P, Record);
  Code = serialization::DECL_FUNCTION;
}

void ASTDeclWriter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  VisitNamedDecl(D);
  // FIXME: convert to LazyStmtPtr?
  // Unlike C/C++, method bodies will never be in header files.
  bool HasBodyStuff = D->getBody() != nullptr     ||
                      D->getSelfDecl() != nullptr || D->getCmdDecl() != nullptr;
  Record.push_back(HasBodyStuff);
  if (HasBodyStuff) {
    Writer.AddStmt(D->getBody());
    Writer.AddDeclRef(D->getSelfDecl(), Record);
    Writer.AddDeclRef(D->getCmdDecl(), Record);
  }
  Record.push_back(D->isInstanceMethod());
  Record.push_back(D->isVariadic());
  Record.push_back(D->isPropertyAccessor());
  Record.push_back(D->isDefined());
  Record.push_back(D->IsOverriding);
  Record.push_back(D->HasSkippedBody);

  Record.push_back(D->IsRedeclaration);
  Record.push_back(D->HasRedeclaration);
  if (D->HasRedeclaration) {
    assert(Context.getObjCMethodRedeclaration(D));
    Writer.AddDeclRef(Context.getObjCMethodRedeclaration(D), Record);
  }

  // FIXME: stable encoding for @required/@optional
  Record.push_back(D->getImplementationControl());
  // FIXME: stable encoding for in/out/inout/bycopy/byref/oneway/nullability
  Record.push_back(D->getObjCDeclQualifier());
  Record.push_back(D->hasRelatedResultType());
  Writer.AddTypeRef(D->getReturnType(), Record);
  Writer.AddTypeSourceInfo(D->getReturnTypeSourceInfo(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Record.push_back(D->param_size());
  for (const auto *P : D->params())
    Writer.AddDeclRef(P, Record);

  Record.push_back(D->SelLocsKind);
  unsigned NumStoredSelLocs = D->getNumStoredSelLocs();
  SourceLocation *SelLocs = D->getStoredSelLocs();
  Record.push_back(NumStoredSelLocs);
  for (unsigned i = 0; i != NumStoredSelLocs; ++i)
    Writer.AddSourceLocation(SelLocs[i], Record);

  Code = serialization::DECL_OBJC_METHOD;
}

void ASTDeclWriter::VisitObjCTypeParamDecl(ObjCTypeParamDecl *D) {
  VisitTypedefNameDecl(D);
  Record.push_back(D->Variance);
  Record.push_back(D->Index);
  Writer.AddSourceLocation(D->VarianceLoc, Record);
  Writer.AddSourceLocation(D->ColonLoc, Record);

  Code = serialization::DECL_OBJC_TYPE_PARAM;
}

void ASTDeclWriter::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getAtStartLoc(), Record);
  Writer.AddSourceRange(D->getAtEndRange(), Record);
  // Abstract class (no need to define a stable serialization::DECL code).
}

void ASTDeclWriter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  VisitRedeclarable(D);
  VisitObjCContainerDecl(D);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
  AddObjCTypeParamList(D->TypeParamList);

  Record.push_back(D->isThisDeclarationADefinition());
  if (D->isThisDeclarationADefinition()) {
    // Write the DefinitionData
    ObjCInterfaceDecl::DefinitionData &Data = D->data();
    
    Writer.AddTypeSourceInfo(D->getSuperClassTInfo(), Record);
    Writer.AddSourceLocation(D->getEndOfDefinitionLoc(), Record);
    Record.push_back(Data.HasDesignatedInitializers);

    // Write out the protocols that are directly referenced by the @interface.
    Record.push_back(Data.ReferencedProtocols.size());
    for (const auto *P : D->protocols())
      Writer.AddDeclRef(P, Record);
    for (const auto &PL : D->protocol_locs())
      Writer.AddSourceLocation(PL, Record);
    
    // Write out the protocols that are transitively referenced.
    Record.push_back(Data.AllReferencedProtocols.size());
    for (ObjCList<ObjCProtocolDecl>::iterator
              P = Data.AllReferencedProtocols.begin(),
           PEnd = Data.AllReferencedProtocols.end();
         P != PEnd; ++P)
      Writer.AddDeclRef(*P, Record);

    
    if (ObjCCategoryDecl *Cat = D->getCategoryListRaw()) {
      // Ensure that we write out the set of categories for this class.
      Writer.ObjCClassesWithCategories.insert(D);
      
      // Make sure that the categories get serialized.
      for (; Cat; Cat = Cat->getNextClassCategoryRaw())
        (void)Writer.GetDeclRef(Cat);
    }
  }  
  
  Code = serialization::DECL_OBJC_INTERFACE;
}

void ASTDeclWriter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  VisitFieldDecl(D);
  // FIXME: stable encoding for @public/@private/@protected/@package
  Record.push_back(D->getAccessControl());
  Record.push_back(D->getSynthesize());

  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      !D->isUsed(false) &&
      !D->isInvalidDecl() &&
      !D->isReferenced() &&
      !D->isModulePrivate() &&
      !D->getBitWidth() &&
      !D->hasExtInfo() &&
      D->getDeclName())
    AbbrevToUse = Writer.getDeclObjCIvarAbbrev();

  Code = serialization::DECL_OBJC_IVAR;
}

void ASTDeclWriter::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  VisitRedeclarable(D);
  VisitObjCContainerDecl(D);
  
  Record.push_back(D->isThisDeclarationADefinition());
  if (D->isThisDeclarationADefinition()) {
    Record.push_back(D->protocol_size());
    for (const auto *I : D->protocols())
      Writer.AddDeclRef(I, Record);
    for (const auto &PL : D->protocol_locs())
      Writer.AddSourceLocation(PL, Record);
  }
  
  Code = serialization::DECL_OBJC_PROTOCOL;
}

void ASTDeclWriter::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D) {
  VisitFieldDecl(D);
  Code = serialization::DECL_OBJC_AT_DEFS_FIELD;
}

void ASTDeclWriter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddSourceLocation(D->getCategoryNameLoc(), Record);
  Writer.AddSourceLocation(D->getIvarLBraceLoc(), Record);
  Writer.AddSourceLocation(D->getIvarRBraceLoc(), Record);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  AddObjCTypeParamList(D->TypeParamList);
  Record.push_back(D->protocol_size());
  for (const auto *I : D->protocols())
    Writer.AddDeclRef(I, Record);
  for (const auto &PL : D->protocol_locs())
    Writer.AddSourceLocation(PL, Record);
  Code = serialization::DECL_OBJC_CATEGORY;
}

void ASTDeclWriter::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D) {
  VisitNamedDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  Code = serialization::DECL_OBJC_COMPATIBLE_ALIAS;
}

void ASTDeclWriter::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getAtLoc(), Record);
  Writer.AddSourceLocation(D->getLParenLoc(), Record);
  Writer.AddTypeRef(D->getType(), Record);
  Writer.AddTypeSourceInfo(D->getTypeSourceInfo(), Record);
  // FIXME: stable encoding
  Record.push_back((unsigned)D->getPropertyAttributes());
  Record.push_back((unsigned)D->getPropertyAttributesAsWritten());
  // FIXME: stable encoding
  Record.push_back((unsigned)D->getPropertyImplementation());
  Writer.AddDeclarationName(D->getGetterName(), Record);
  Writer.AddDeclarationName(D->getSetterName(), Record);
  Writer.AddDeclRef(D->getGetterMethodDecl(), Record);
  Writer.AddDeclRef(D->getSetterMethodDecl(), Record);
  Writer.AddDeclRef(D->getPropertyIvarDecl(), Record);
  Code = serialization::DECL_OBJC_PROPERTY;
}

void ASTDeclWriter::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  // Abstract class (no need to define a stable serialization::DECL code).
}

void ASTDeclWriter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  Writer.AddIdentifierRef(D->getIdentifier(), Record);
  Writer.AddSourceLocation(D->getCategoryNameLoc(), Record);
  Code = serialization::DECL_OBJC_CATEGORY_IMPL;
}

void ASTDeclWriter::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  Writer.AddDeclRef(D->getSuperClass(), Record);
  Writer.AddSourceLocation(D->getSuperClassLoc(), Record);
  Writer.AddSourceLocation(D->getIvarLBraceLoc(), Record);
  Writer.AddSourceLocation(D->getIvarRBraceLoc(), Record);
  Record.push_back(D->hasNonZeroConstructors());
  Record.push_back(D->hasDestructors());
  Record.push_back(D->NumIvarInitializers);
  if (D->NumIvarInitializers)
    Writer.AddCXXCtorInitializersRef(
        llvm::makeArrayRef(D->init_begin(), D->init_end()), Record);
  Code = serialization::DECL_OBJC_IMPLEMENTATION;
}

void ASTDeclWriter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  Writer.AddSourceLocation(D->getLocStart(), Record);
  Writer.AddDeclRef(D->getPropertyDecl(), Record);
  Writer.AddDeclRef(D->getPropertyIvarDecl(), Record);
  Writer.AddSourceLocation(D->getPropertyIvarDeclLoc(), Record);
  Writer.AddStmt(D->getGetterCXXConstructor());
  Writer.AddStmt(D->getSetterCXXAssignment());
  Code = serialization::DECL_OBJC_PROPERTY_IMPL;
}

void ASTDeclWriter::VisitFieldDecl(FieldDecl *D) {
  VisitDeclaratorDecl(D);
  Record.push_back(D->isMutable());
  if (D->InitStorage.getInt() == FieldDecl::ISK_BitWidthOrNothing &&
      D->InitStorage.getPointer() == nullptr) {
    Record.push_back(0);
  } else if (D->InitStorage.getInt() == FieldDecl::ISK_CapturedVLAType) {
    Record.push_back(D->InitStorage.getInt() + 1);
    Writer.AddTypeRef(
        QualType(static_cast<Type *>(D->InitStorage.getPointer()), 0),
        Record);
  } else {
    Record.push_back(D->InitStorage.getInt() + 1);
    Writer.AddStmt(static_cast<Expr *>(D->InitStorage.getPointer()));
  }
  if (!D->getDeclName())
    Writer.AddDeclRef(Context.getInstantiatedFromUnnamedFieldDecl(D), Record);

  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      !D->isUsed(false) &&
      !D->isInvalidDecl() &&
      !D->isReferenced() &&
      !D->isTopLevelDeclInObjCContainer() &&
      !D->isModulePrivate() &&
      !D->getBitWidth() &&
      !D->hasInClassInitializer() &&
      !D->hasExtInfo() &&
      !ObjCIvarDecl::classofKind(D->getKind()) &&
      !ObjCAtDefsFieldDecl::classofKind(D->getKind()) &&
      D->getDeclName())
    AbbrevToUse = Writer.getDeclFieldAbbrev();

  Code = serialization::DECL_FIELD;
}

void ASTDeclWriter::VisitMSPropertyDecl(MSPropertyDecl *D) {
  VisitDeclaratorDecl(D);
  Writer.AddIdentifierRef(D->getGetterId(), Record);
  Writer.AddIdentifierRef(D->getSetterId(), Record);
  Code = serialization::DECL_MS_PROPERTY;
}

void ASTDeclWriter::VisitIndirectFieldDecl(IndirectFieldDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->getChainingSize());

  for (const auto *P : D->chain())
    Writer.AddDeclRef(P, Record);
  Code = serialization::DECL_INDIRECTFIELD;
}

void ASTDeclWriter::VisitVarDecl(VarDecl *D) {
  VisitRedeclarable(D);
  VisitDeclaratorDecl(D);
  Record.push_back(D->getStorageClass());
  Record.push_back(D->getTSCSpec());
  Record.push_back(D->getInitStyle());
  if (!isa<ParmVarDecl>(D)) {
    Record.push_back(D->isExceptionVariable());
    Record.push_back(D->isNRVOVariable());
    Record.push_back(D->isCXXForRangeDecl());
    Record.push_back(D->isARCPseudoStrong());
    Record.push_back(D->isConstexpr());
    Record.push_back(D->isInitCapture());
    Record.push_back(D->isPreviousDeclInSameBlockScope());
  }
  Record.push_back(D->getLinkageInternal());

  if (D->getInit()) {
    Record.push_back(!D->isInitKnownICE() ? 1 : (D->isInitICE() ? 3 : 2));
    Writer.AddStmt(D->getInit());
  } else {
    Record.push_back(0);
  }
  
  enum {
    VarNotTemplate = 0, VarTemplate, StaticDataMemberSpecialization
  };
  if (VarTemplateDecl *TemplD = D->getDescribedVarTemplate()) {
    Record.push_back(VarTemplate);
    Writer.AddDeclRef(TemplD, Record);
  } else if (MemberSpecializationInfo *SpecInfo
               = D->getMemberSpecializationInfo()) {
    Record.push_back(StaticDataMemberSpecialization);
    Writer.AddDeclRef(SpecInfo->getInstantiatedFrom(), Record);
    Record.push_back(SpecInfo->getTemplateSpecializationKind());
    Writer.AddSourceLocation(SpecInfo->getPointOfInstantiation(), Record);
  } else {
    Record.push_back(VarNotTemplate);
  }

  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      !D->isUsed(false) &&
      !D->isInvalidDecl() &&
      !D->isReferenced() &&
      !D->isTopLevelDeclInObjCContainer() &&
      D->getAccess() == AS_none &&
      !D->isModulePrivate() &&
      !needsAnonymousDeclarationNumber(D) &&
      D->getDeclName().getNameKind() == DeclarationName::Identifier &&
      !D->hasExtInfo() &&
      D->getFirstDecl() == D->getMostRecentDecl() &&
      D->getInitStyle() == VarDecl::CInit &&
      D->getInit() == nullptr &&
      !isa<ParmVarDecl>(D) &&
      !isa<VarTemplateSpecializationDecl>(D) &&
      !D->isConstexpr() &&
      !D->isInitCapture() &&
      !D->isPreviousDeclInSameBlockScope() &&
      !D->getMemberSpecializationInfo())
    AbbrevToUse = Writer.getDeclVarAbbrev();

  Code = serialization::DECL_VAR;
}

void ASTDeclWriter::VisitImplicitParamDecl(ImplicitParamDecl *D) {
  VisitVarDecl(D);
  Code = serialization::DECL_IMPLICIT_PARAM;
}

void ASTDeclWriter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
  Record.push_back(D->isObjCMethodParameter());
  Record.push_back(D->getFunctionScopeDepth());
  Record.push_back(D->getFunctionScopeIndex());
  Record.push_back(D->getObjCDeclQualifier()); // FIXME: stable encoding
  Record.push_back(D->isKNRPromoted());
  Record.push_back(D->hasInheritedDefaultArg());
  Record.push_back(D->hasUninstantiatedDefaultArg());
  if (D->hasUninstantiatedDefaultArg())
    Writer.AddStmt(D->getUninstantiatedDefaultArg());
  Code = serialization::DECL_PARM_VAR;

  assert(!D->isARCPseudoStrong()); // can be true of ImplicitParamDecl

  // If the assumptions about the DECL_PARM_VAR abbrev are true, use it.  Here
  // we dynamically check for the properties that we optimize for, but don't
  // know are true of all PARM_VAR_DECLs.
  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      !D->hasAttrs() &&
      !D->hasExtInfo() &&
      !D->isImplicit() &&
      !D->isUsed(false) &&
      !D->isInvalidDecl() &&
      !D->isReferenced() &&
      D->getAccess() == AS_none &&
      !D->isModulePrivate() &&
      D->getStorageClass() == 0 &&
      D->getInitStyle() == VarDecl::CInit && // Can params have anything else?
      D->getFunctionScopeDepth() == 0 &&
      D->getObjCDeclQualifier() == 0 &&
      !D->isKNRPromoted() &&
      !D->hasInheritedDefaultArg() &&
      D->getInit() == nullptr &&
      !D->hasUninstantiatedDefaultArg())  // No default expr.
    AbbrevToUse = Writer.getDeclParmVarAbbrev();

  // Check things we know are true of *every* PARM_VAR_DECL, which is more than
  // just us assuming it.
  assert(!D->getTSCSpec() && "PARM_VAR_DECL can't use TLS");
  assert(D->getAccess() == AS_none && "PARM_VAR_DECL can't be public/private");
  assert(!D->isExceptionVariable() && "PARM_VAR_DECL can't be exception var");
  assert(D->getPreviousDecl() == nullptr && "PARM_VAR_DECL can't be redecl");
  assert(!D->isStaticDataMember() &&
         "PARM_VAR_DECL can't be static data member");
}

void ASTDeclWriter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getAsmString());
  Writer.AddSourceLocation(D->getRParenLoc(), Record);
  Code = serialization::DECL_FILE_SCOPE_ASM;
}

void ASTDeclWriter::VisitEmptyDecl(EmptyDecl *D) {
  VisitDecl(D);
  Code = serialization::DECL_EMPTY;
}

void ASTDeclWriter::VisitBlockDecl(BlockDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getBody());
  Writer.AddTypeSourceInfo(D->getSignatureAsWritten(), Record);
  Record.push_back(D->param_size());
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Record.push_back(D->isVariadic());
  Record.push_back(D->blockMissingReturnType());
  Record.push_back(D->isConversionFromLambda());
  Record.push_back(D->capturesCXXThis());
  Record.push_back(D->getNumCaptures());
  for (const auto &capture : D->captures()) {
    Writer.AddDeclRef(capture.getVariable(), Record);

    unsigned flags = 0;
    if (capture.isByRef()) flags |= 1;
    if (capture.isNested()) flags |= 2;
    if (capture.hasCopyExpr()) flags |= 4;
    Record.push_back(flags);

    if (capture.hasCopyExpr()) Writer.AddStmt(capture.getCopyExpr());
  }

  Code = serialization::DECL_BLOCK;
}

void ASTDeclWriter::VisitCapturedDecl(CapturedDecl *CD) {
  Record.push_back(CD->getNumParams());
  VisitDecl(CD);
  Record.push_back(CD->getContextParamPosition());
  Record.push_back(CD->isNothrow() ? 1 : 0);
  // Body is stored by VisitCapturedStmt.
  for (unsigned I = 0; I < CD->getNumParams(); ++I)
    Writer.AddDeclRef(CD->getParam(I), Record);
  Code = serialization::DECL_CAPTURED;
}

void ASTDeclWriter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  VisitDecl(D);
  Record.push_back(D->getLanguage());
  Writer.AddSourceLocation(D->getExternLoc(), Record);
  Writer.AddSourceLocation(D->getRBraceLoc(), Record);
  Code = serialization::DECL_LINKAGE_SPEC;
}

void ASTDeclWriter::VisitLabelDecl(LabelDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getLocStart(), Record);
  Code = serialization::DECL_LABEL;
}


void ASTDeclWriter::VisitNamespaceDecl(NamespaceDecl *D) {
  VisitRedeclarable(D);
  VisitNamedDecl(D);
  Record.push_back(D->isInline());
  Writer.AddSourceLocation(D->getLocStart(), Record);
  Writer.AddSourceLocation(D->getRBraceLoc(), Record);

  if (D->isOriginalNamespace())
    Writer.AddDeclRef(D->getAnonymousNamespace(), Record);
  Code = serialization::DECL_NAMESPACE;

  if (Writer.hasChain() && D->isAnonymousNamespace() && 
      D == D->getMostRecentDecl()) {
    // This is a most recent reopening of the anonymous namespace. If its parent
    // is in a previous PCH (or is the TU), mark that parent for update, because
    // the original namespace always points to the latest re-opening of its
    // anonymous namespace.
    Decl *Parent = cast<Decl>(
        D->getParent()->getRedeclContext()->getPrimaryContext());
    if (Parent->isFromASTFile() || isa<TranslationUnitDecl>(Parent)) {
      Writer.DeclUpdates[Parent].push_back(
          ASTWriter::DeclUpdate(UPD_CXX_ADDED_ANONYMOUS_NAMESPACE, D));
    }
  }
}

void ASTDeclWriter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  VisitRedeclarable(D);
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getNamespaceLoc(), Record);
  Writer.AddSourceLocation(D->getTargetNameLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(D->getQualifierLoc(), Record);
  Writer.AddDeclRef(D->getNamespace(), Record);
  Code = serialization::DECL_NAMESPACE_ALIAS;
}

void ASTDeclWriter::VisitUsingDecl(UsingDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getUsingLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(D->getQualifierLoc(), Record);
  Writer.AddDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record);
  Writer.AddDeclRef(D->FirstUsingShadow.getPointer(), Record);
  Record.push_back(D->hasTypename());
  Writer.AddDeclRef(Context.getInstantiatedFromUsingDecl(D), Record);
  Code = serialization::DECL_USING;
}

void ASTDeclWriter::VisitUsingShadowDecl(UsingShadowDecl *D) {
  VisitRedeclarable(D);
  VisitNamedDecl(D);
  Writer.AddDeclRef(D->getTargetDecl(), Record);
  Writer.AddDeclRef(D->UsingOrNextShadow, Record);
  Writer.AddDeclRef(Context.getInstantiatedFromUsingShadowDecl(D), Record);
  Code = serialization::DECL_USING_SHADOW;
}

void ASTDeclWriter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getUsingLoc(), Record);
  Writer.AddSourceLocation(D->getNamespaceKeyLocation(), Record);
  Writer.AddNestedNameSpecifierLoc(D->getQualifierLoc(), Record);
  Writer.AddDeclRef(D->getNominatedNamespace(), Record);
  Writer.AddDeclRef(dyn_cast<Decl>(D->getCommonAncestor()), Record);
  Code = serialization::DECL_USING_DIRECTIVE;
}

void ASTDeclWriter::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  VisitValueDecl(D);
  Writer.AddSourceLocation(D->getUsingLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(D->getQualifierLoc(), Record);
  Writer.AddDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record);
  Code = serialization::DECL_UNRESOLVED_USING_VALUE;
}

void ASTDeclWriter::VisitUnresolvedUsingTypenameDecl(
                                               UnresolvedUsingTypenameDecl *D) {
  VisitTypeDecl(D);
  Writer.AddSourceLocation(D->getTypenameLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(D->getQualifierLoc(), Record);
  Code = serialization::DECL_UNRESOLVED_USING_TYPENAME;
}

void ASTDeclWriter::VisitCXXRecordDecl(CXXRecordDecl *D) {
  VisitRecordDecl(D);

  enum {
    CXXRecNotTemplate = 0, CXXRecTemplate, CXXRecMemberSpecialization
  };
  if (ClassTemplateDecl *TemplD = D->getDescribedClassTemplate()) {
    Record.push_back(CXXRecTemplate);
    Writer.AddDeclRef(TemplD, Record);
  } else if (MemberSpecializationInfo *MSInfo
               = D->getMemberSpecializationInfo()) {
    Record.push_back(CXXRecMemberSpecialization);
    Writer.AddDeclRef(MSInfo->getInstantiatedFrom(), Record);
    Record.push_back(MSInfo->getTemplateSpecializationKind());
    Writer.AddSourceLocation(MSInfo->getPointOfInstantiation(), Record);
  } else {
    Record.push_back(CXXRecNotTemplate);
  }

  Record.push_back(D->isThisDeclarationADefinition());
  if (D->isThisDeclarationADefinition())
    Writer.AddCXXDefinitionData(D, Record);

  // Store (what we currently believe to be) the key function to avoid
  // deserializing every method so we can compute it.
  if (D->IsCompleteDefinition)
    Writer.AddDeclRef(Context.getCurrentKeyFunction(D), Record);

  Code = serialization::DECL_CXX_RECORD;
}

void ASTDeclWriter::VisitCXXMethodDecl(CXXMethodDecl *D) {
  VisitFunctionDecl(D);
  if (D->isCanonicalDecl()) {
    Record.push_back(D->size_overridden_methods());
    for (CXXMethodDecl::method_iterator
           I = D->begin_overridden_methods(), E = D->end_overridden_methods();
           I != E; ++I)
      Writer.AddDeclRef(*I, Record);
  } else {
    // We only need to record overridden methods once for the canonical decl.
    Record.push_back(0);
  }

  if (D->getDeclContext() == D->getLexicalDeclContext() &&
      D->getFirstDecl() == D->getMostRecentDecl() &&
      !D->isInvalidDecl() &&
      !D->hasAttrs() &&
      !D->isTopLevelDeclInObjCContainer() &&
      D->getDeclName().getNameKind() == DeclarationName::Identifier &&
      !D->hasExtInfo() &&
      !D->hasInheritedPrototype() &&
      D->hasWrittenPrototype())
    AbbrevToUse = Writer.getDeclCXXMethodAbbrev();

  Code = serialization::DECL_CXX_METHOD;
}

void ASTDeclWriter::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  VisitCXXMethodDecl(D);

  Writer.AddDeclRef(D->getInheritedConstructor(), Record);
  Record.push_back(D->IsExplicitSpecified);

  Code = serialization::DECL_CXX_CONSTRUCTOR;
}

void ASTDeclWriter::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  VisitCXXMethodDecl(D);

  Writer.AddDeclRef(D->getOperatorDelete(), Record);

  Code = serialization::DECL_CXX_DESTRUCTOR;
}

void ASTDeclWriter::VisitCXXConversionDecl(CXXConversionDecl *D) {
  VisitCXXMethodDecl(D);
  Record.push_back(D->IsExplicitSpecified);
  Code = serialization::DECL_CXX_CONVERSION;
}

void ASTDeclWriter::VisitImportDecl(ImportDecl *D) {
  VisitDecl(D);
  Record.push_back(Writer.getSubmoduleID(D->getImportedModule()));
  ArrayRef<SourceLocation> IdentifierLocs = D->getIdentifierLocs();
  Record.push_back(!IdentifierLocs.empty());
  if (IdentifierLocs.empty()) {
    Writer.AddSourceLocation(D->getLocEnd(), Record);
    Record.push_back(1);
  } else {
    for (unsigned I = 0, N = IdentifierLocs.size(); I != N; ++I)
      Writer.AddSourceLocation(IdentifierLocs[I], Record);
    Record.push_back(IdentifierLocs.size());
  }
  // Note: the number of source locations must always be the last element in
  // the record.
  Code = serialization::DECL_IMPORT;
}

void ASTDeclWriter::VisitAccessSpecDecl(AccessSpecDecl *D) {
  VisitDecl(D);
  Writer.AddSourceLocation(D->getColonLoc(), Record);
  Code = serialization::DECL_ACCESS_SPEC;
}

void ASTDeclWriter::VisitFriendDecl(FriendDecl *D) {
  // Record the number of friend type template parameter lists here
  // so as to simplify memory allocation during deserialization.
  Record.push_back(D->NumTPLists);
  VisitDecl(D);
  bool hasFriendDecl = D->Friend.is<NamedDecl*>();
  Record.push_back(hasFriendDecl);
  if (hasFriendDecl)
    Writer.AddDeclRef(D->getFriendDecl(), Record);
  else
    Writer.AddTypeSourceInfo(D->getFriendType(), Record);
  for (unsigned i = 0; i < D->NumTPLists; ++i)
    Writer.AddTemplateParameterList(D->getFriendTypeTemplateParameterList(i),
                                    Record);
  Writer.AddDeclRef(D->getNextFriend(), Record);
  Record.push_back(D->UnsupportedFriend);
  Writer.AddSourceLocation(D->FriendLoc, Record);
  Code = serialization::DECL_FRIEND;
}

void ASTDeclWriter::VisitFriendTemplateDecl(FriendTemplateDecl *D) {
  VisitDecl(D);
  Record.push_back(D->getNumTemplateParameters());
  for (unsigned i = 0, e = D->getNumTemplateParameters(); i != e; ++i)
    Writer.AddTemplateParameterList(D->getTemplateParameterList(i), Record);
  Record.push_back(D->getFriendDecl() != nullptr);
  if (D->getFriendDecl())
    Writer.AddDeclRef(D->getFriendDecl(), Record);
  else
    Writer.AddTypeSourceInfo(D->getFriendType(), Record);
  Writer.AddSourceLocation(D->getFriendLoc(), Record);
  Code = serialization::DECL_FRIEND_TEMPLATE;
}

void ASTDeclWriter::VisitTemplateDecl(TemplateDecl *D) {
  VisitNamedDecl(D);

  Writer.AddDeclRef(D->getTemplatedDecl(), Record);
  Writer.AddTemplateParameterList(D->getTemplateParameters(), Record);
}

void ASTDeclWriter::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D) {
  VisitRedeclarable(D);

  // Emit data to initialize CommonOrPrev before VisitTemplateDecl so that
  // getCommonPtr() can be used while this is still initializing.
  if (D->isFirstDecl()) {
    // This declaration owns the 'common' pointer, so serialize that data now.
    Writer.AddDeclRef(D->getInstantiatedFromMemberTemplate(), Record);
    if (D->getInstantiatedFromMemberTemplate())
      Record.push_back(D->isMemberSpecialization());
  }
  
  VisitTemplateDecl(D);
  Record.push_back(D->getIdentifierNamespace());
}

void ASTDeclWriter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->isFirstDecl())
    AddTemplateSpecializations(D);
  Code = serialization::DECL_CLASS_TEMPLATE;
}

void ASTDeclWriter::VisitClassTemplateSpecializationDecl(
                                           ClassTemplateSpecializationDecl *D) {
  RegisterTemplateSpecialization(D->getSpecializedTemplate(), D);

  VisitCXXRecordDecl(D);

  llvm::PointerUnion<ClassTemplateDecl *,
                     ClassTemplatePartialSpecializationDecl *> InstFrom
    = D->getSpecializedTemplateOrPartial();
  if (Decl *InstFromD = InstFrom.dyn_cast<ClassTemplateDecl *>()) {
    Writer.AddDeclRef(InstFromD, Record);
  } else {
    Writer.AddDeclRef(InstFrom.get<ClassTemplatePartialSpecializationDecl *>(),
                      Record);
    Writer.AddTemplateArgumentList(&D->getTemplateInstantiationArgs(), Record);
  }

  Writer.AddTemplateArgumentList(&D->getTemplateArgs(), Record);
  Writer.AddSourceLocation(D->getPointOfInstantiation(), Record);
  Record.push_back(D->getSpecializationKind());
  Record.push_back(D->isCanonicalDecl());

  if (D->isCanonicalDecl()) {
    // When reading, we'll add it to the folding set of the following template. 
    Writer.AddDeclRef(D->getSpecializedTemplate()->getCanonicalDecl(), Record);
  }

  // Explicit info.
  Writer.AddTypeSourceInfo(D->getTypeAsWritten(), Record);
  if (D->getTypeAsWritten()) {
    Writer.AddSourceLocation(D->getExternLoc(), Record);
    Writer.AddSourceLocation(D->getTemplateKeywordLoc(), Record);
  }

  Code = serialization::DECL_CLASS_TEMPLATE_SPECIALIZATION;
}

void ASTDeclWriter::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);

  Writer.AddTemplateParameterList(D->getTemplateParameters(), Record);
  Writer.AddASTTemplateArgumentListInfo(D->getTemplateArgsAsWritten(), Record);

  // These are read/set from/to the first declaration.
  if (D->getPreviousDecl() == nullptr) {
    Writer.AddDeclRef(D->getInstantiatedFromMember(), Record);
    Record.push_back(D->isMemberSpecialization());
  }

  Code = serialization::DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION;
}

void ASTDeclWriter::VisitVarTemplateDecl(VarTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->isFirstDecl())
    AddTemplateSpecializations(D);
  Code = serialization::DECL_VAR_TEMPLATE;
}

void ASTDeclWriter::VisitVarTemplateSpecializationDecl(
    VarTemplateSpecializationDecl *D) {
  RegisterTemplateSpecialization(D->getSpecializedTemplate(), D);

  VisitVarDecl(D);

  llvm::PointerUnion<VarTemplateDecl *, VarTemplatePartialSpecializationDecl *>
  InstFrom = D->getSpecializedTemplateOrPartial();
  if (Decl *InstFromD = InstFrom.dyn_cast<VarTemplateDecl *>()) {
    Writer.AddDeclRef(InstFromD, Record);
  } else {
    Writer.AddDeclRef(InstFrom.get<VarTemplatePartialSpecializationDecl *>(),
                      Record);
    Writer.AddTemplateArgumentList(&D->getTemplateInstantiationArgs(), Record);
  }

  // Explicit info.
  Writer.AddTypeSourceInfo(D->getTypeAsWritten(), Record);
  if (D->getTypeAsWritten()) {
    Writer.AddSourceLocation(D->getExternLoc(), Record);
    Writer.AddSourceLocation(D->getTemplateKeywordLoc(), Record);
  }

  Writer.AddTemplateArgumentList(&D->getTemplateArgs(), Record);
  Writer.AddSourceLocation(D->getPointOfInstantiation(), Record);
  Record.push_back(D->getSpecializationKind());
  Record.push_back(D->isCanonicalDecl());

  if (D->isCanonicalDecl()) {
    // When reading, we'll add it to the folding set of the following template.
    Writer.AddDeclRef(D->getSpecializedTemplate()->getCanonicalDecl(), Record);
  }

  Code = serialization::DECL_VAR_TEMPLATE_SPECIALIZATION;
}

void ASTDeclWriter::VisitVarTemplatePartialSpecializationDecl(
    VarTemplatePartialSpecializationDecl *D) {
  VisitVarTemplateSpecializationDecl(D);

  Writer.AddTemplateParameterList(D->getTemplateParameters(), Record);
  Writer.AddASTTemplateArgumentListInfo(D->getTemplateArgsAsWritten(), Record);

  // These are read/set from/to the first declaration.
  if (D->getPreviousDecl() == nullptr) {
    Writer.AddDeclRef(D->getInstantiatedFromMember(), Record);
    Record.push_back(D->isMemberSpecialization());
  }

  Code = serialization::DECL_VAR_TEMPLATE_PARTIAL_SPECIALIZATION;
}

void ASTDeclWriter::VisitClassScopeFunctionSpecializationDecl(
                                    ClassScopeFunctionSpecializationDecl *D) {
  VisitDecl(D);
  Writer.AddDeclRef(D->getSpecialization(), Record);
  Code = serialization::DECL_CLASS_SCOPE_FUNCTION_SPECIALIZATION;
}


void ASTDeclWriter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->isFirstDecl())
    AddTemplateSpecializations(D);
  Code = serialization::DECL_FUNCTION_TEMPLATE;
}

void ASTDeclWriter::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  VisitTypeDecl(D);

  Record.push_back(D->wasDeclaredWithTypename());

  bool OwnsDefaultArg = D->hasDefaultArgument() &&
                        !D->defaultArgumentWasInherited();
  Record.push_back(OwnsDefaultArg);
  if (OwnsDefaultArg)
    Writer.AddTypeSourceInfo(D->getDefaultArgumentInfo(), Record);

  Code = serialization::DECL_TEMPLATE_TYPE_PARM;
}

void ASTDeclWriter::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  // For an expanded parameter pack, record the number of expansion types here
  // so that it's easier for deserialization to allocate the right amount of
  // memory.
  if (D->isExpandedParameterPack())
    Record.push_back(D->getNumExpansionTypes());
  
  VisitDeclaratorDecl(D);
  // TemplateParmPosition.
  Record.push_back(D->getDepth());
  Record.push_back(D->getPosition());
  
  if (D->isExpandedParameterPack()) {
    for (unsigned I = 0, N = D->getNumExpansionTypes(); I != N; ++I) {
      Writer.AddTypeRef(D->getExpansionType(I), Record);
      Writer.AddTypeSourceInfo(D->getExpansionTypeSourceInfo(I), Record);
    }
      
    Code = serialization::DECL_EXPANDED_NON_TYPE_TEMPLATE_PARM_PACK;
  } else {
    // Rest of NonTypeTemplateParmDecl.
    Record.push_back(D->isParameterPack());
    bool OwnsDefaultArg = D->hasDefaultArgument() &&
                          !D->defaultArgumentWasInherited();
    Record.push_back(OwnsDefaultArg);
    if (OwnsDefaultArg)
      Writer.AddStmt(D->getDefaultArgument());
    Code = serialization::DECL_NON_TYPE_TEMPLATE_PARM;
  }
}

void ASTDeclWriter::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  // For an expanded parameter pack, record the number of expansion types here
  // so that it's easier for deserialization to allocate the right amount of
  // memory.
  if (D->isExpandedParameterPack())
    Record.push_back(D->getNumExpansionTemplateParameters());

  VisitTemplateDecl(D);
  // TemplateParmPosition.
  Record.push_back(D->getDepth());
  Record.push_back(D->getPosition());

  if (D->isExpandedParameterPack()) {
    for (unsigned I = 0, N = D->getNumExpansionTemplateParameters();
         I != N; ++I)
      Writer.AddTemplateParameterList(D->getExpansionTemplateParameters(I),
                                      Record);
    Code = serialization::DECL_EXPANDED_TEMPLATE_TEMPLATE_PARM_PACK;
  } else {
    // Rest of TemplateTemplateParmDecl.
    Record.push_back(D->isParameterPack());
    bool OwnsDefaultArg = D->hasDefaultArgument() &&
                          !D->defaultArgumentWasInherited();
    Record.push_back(OwnsDefaultArg);
    if (OwnsDefaultArg)
      Writer.AddTemplateArgumentLoc(D->getDefaultArgument(), Record);
    Code = serialization::DECL_TEMPLATE_TEMPLATE_PARM;
  }
}

void ASTDeclWriter::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);
  Code = serialization::DECL_TYPE_ALIAS_TEMPLATE;
}

void ASTDeclWriter::VisitStaticAssertDecl(StaticAssertDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getAssertExpr());
  Record.push_back(D->isFailed());
  Writer.AddStmt(D->getMessage());
  Writer.AddSourceLocation(D->getRParenLoc(), Record);
  Code = serialization::DECL_STATIC_ASSERT;
}

/// \brief Emit the DeclContext part of a declaration context decl.
///
/// \param LexicalOffset the offset at which the DECL_CONTEXT_LEXICAL
/// block for this declaration context is stored. May be 0 to indicate
/// that there are no declarations stored within this context.
///
/// \param VisibleOffset the offset at which the DECL_CONTEXT_VISIBLE
/// block for this declaration context is stored. May be 0 to indicate
/// that there are no declarations visible from this context. Note
/// that this value will not be emitted for non-primary declaration
/// contexts.
void ASTDeclWriter::VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset,
                                     uint64_t VisibleOffset) {
  Record.push_back(LexicalOffset);
  Record.push_back(VisibleOffset);
}

const Decl *ASTWriter::getFirstLocalDecl(const Decl *D) {
  /// \brief Is this a local declaration (that is, one that will be written to
  /// our AST file)? This is the case for declarations that are neither imported
  /// from another AST file nor predefined.
  auto IsLocalDecl = [&](const Decl *D) -> bool {
    if (D->isFromASTFile())
      return false;
    auto I = DeclIDs.find(D);
    return (I == DeclIDs.end() || I->second >= NUM_PREDEF_DECL_IDS);
  };

  assert(IsLocalDecl(D) && "expected a local declaration");

  const Decl *Canon = D->getCanonicalDecl();
  if (IsLocalDecl(Canon))
    return Canon;

  const Decl *&CacheEntry = FirstLocalDeclCache[Canon];
  if (CacheEntry)
    return CacheEntry;

  for (const Decl *Redecl = D; Redecl; Redecl = Redecl->getPreviousDecl())
    if (IsLocalDecl(Redecl))
      D = Redecl;
  return CacheEntry = D;
}

template <typename T>
void ASTDeclWriter::VisitRedeclarable(Redeclarable<T> *D) {
  T *First = D->getFirstDecl();
  T *MostRecent = First->getMostRecentDecl();
  T *DAsT = static_cast<T *>(D);
  if (MostRecent != First) {
    assert(isRedeclarableDeclKind(DAsT->getKind()) &&
           "Not considered redeclarable?");

    Writer.AddDeclRef(First, Record);

    // Write out a list of local redeclarations of this declaration if it's the
    // first local declaration in the chain.
    const Decl *FirstLocal = Writer.getFirstLocalDecl(DAsT);
    if (DAsT == FirstLocal) {
      // Emit a list of all imported first declarations so that we can be sure
      // that all redeclarations visible to this module are before D in the
      // redecl chain.
      unsigned I = Record.size();
      Record.push_back(0);
      if (Writer.Chain)
        AddFirstDeclFromEachModule(DAsT, /*IncludeLocal*/false);
      // This is the number of imported first declarations + 1.
      Record[I] = Record.size() - I;

      // Collect the set of local redeclarations of this declaration, from
      // newest to oldest.
      RecordData LocalRedecls;
      for (const Decl *Prev = FirstLocal->getMostRecentDecl();
           Prev != FirstLocal; Prev = Prev->getPreviousDecl())
        if (!Prev->isFromASTFile())
          Writer.AddDeclRef(Prev, LocalRedecls);

      // If we have any redecls, write them now as a separate record preceding
      // the declaration itself.
      if (LocalRedecls.empty())
        Record.push_back(0);
      else {
        Record.push_back(Writer.Stream.GetCurrentBitNo());
        Writer.Stream.EmitRecord(LOCAL_REDECLARATIONS, LocalRedecls);
      }
    } else {
      Record.push_back(0);
      Writer.AddDeclRef(FirstLocal, Record);
    }

    // Make sure that we serialize both the previous and the most-recent 
    // declarations, which (transitively) ensures that all declarations in the
    // chain get serialized.
    //
    // FIXME: This is not correct; when we reach an imported declaration we
    // won't emit its previous declaration.
    (void)Writer.GetDeclRef(D->getPreviousDecl());
    (void)Writer.GetDeclRef(MostRecent);
  } else {
    // We use the sentinel value 0 to indicate an only declaration.
    Record.push_back(0);
  }
}

void ASTDeclWriter::VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D) {
  Record.push_back(D->varlist_size());
  VisitDecl(D);
  for (auto *I : D->varlists())
    Writer.AddStmt(I);
  Code = serialization::DECL_OMP_THREADPRIVATE;
}

void ASTDeclWriter::VisitOMPCapturedFieldDecl(OMPCapturedFieldDecl *D) {
  VisitVarDecl(D);
  Code = serialization::DECL_OMP_CAPTUREDFIELD;
}

//===----------------------------------------------------------------------===//
// ASTWriter Implementation
//===----------------------------------------------------------------------===//

void ASTWriter::WriteDeclAbbrevs() {
  using namespace llvm;

  BitCodeAbbrev *Abv;

  // Abbreviation for DECL_FIELD
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_FIELD));
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(0));                       // isUsed
  Abv->Add(BitCodeAbbrevOp(0));                       // isReferenced
  Abv->Add(BitCodeAbbrevOp(0));                   // TopLevelDeclInObjCContainer
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2));  // AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  Abv->Add(BitCodeAbbrevOp(0));                       // AnonDeclNumber
  // ValueDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  // DeclaratorDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // InnerStartLoc
  Abv->Add(BitCodeAbbrevOp(0));                       // hasExtInfo
  // FieldDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isMutable
  Abv->Add(BitCodeAbbrevOp(0));                       //getBitWidth
  // Type Source Info
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // TypeLoc
  DeclFieldAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for DECL_OBJC_IVAR
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_OBJC_IVAR));
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(0));                       // isUsed
  Abv->Add(BitCodeAbbrevOp(0));                       // isReferenced
  Abv->Add(BitCodeAbbrevOp(0));                   // TopLevelDeclInObjCContainer
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2));  // AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  Abv->Add(BitCodeAbbrevOp(0));                       // AnonDeclNumber
  // ValueDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  // DeclaratorDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // InnerStartLoc
  Abv->Add(BitCodeAbbrevOp(0));                       // hasExtInfo
  // FieldDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isMutable
  Abv->Add(BitCodeAbbrevOp(0));                       //getBitWidth
  // ObjC Ivar
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // getAccessControl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // getSynthesize
  // Type Source Info
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // TypeLoc
  DeclObjCIvarAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for DECL_ENUM
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_ENUM));
  // Redeclarable
  Abv->Add(BitCodeAbbrevOp(0));                       // No redeclaration
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(0));                       // isUsed
  Abv->Add(BitCodeAbbrevOp(0));                       // isReferenced
  Abv->Add(BitCodeAbbrevOp(0));                   // TopLevelDeclInObjCContainer
  Abv->Add(BitCodeAbbrevOp(AS_none));                 // C++ AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  Abv->Add(BitCodeAbbrevOp(0));                       // AnonDeclNumber
  // TypeDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Source Location
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type Ref
  // TagDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // IdentifierNamespace
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // getTagKind
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isCompleteDefinition
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // EmbeddedInDeclarator
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsFreeStanding
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsCompleteDefinitionRequired
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // SourceLocation
  Abv->Add(BitCodeAbbrevOp(0));                         // ExtInfoKind
  // EnumDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // AddTypeRef
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // IntegerType
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // getPromotionType
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // getNumPositiveBits
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // getNumNegativeBits
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isScoped
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isScopedUsingClassTag
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isFixed
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // InstantiatedMembEnum
  // DC
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // LexicalOffset
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // VisibleOffset
  DeclEnumAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for DECL_RECORD
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_RECORD));
  // Redeclarable
  Abv->Add(BitCodeAbbrevOp(0));                       // No redeclaration
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(0));                       // isUsed
  Abv->Add(BitCodeAbbrevOp(0));                       // isReferenced
  Abv->Add(BitCodeAbbrevOp(0));                   // TopLevelDeclInObjCContainer
  Abv->Add(BitCodeAbbrevOp(AS_none));                 // C++ AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  Abv->Add(BitCodeAbbrevOp(0));                       // AnonDeclNumber
  // TypeDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Source Location
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type Ref
  // TagDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // IdentifierNamespace
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // getTagKind
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isCompleteDefinition
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // EmbeddedInDeclarator
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsFreeStanding
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsCompleteDefinitionRequired
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // SourceLocation
  Abv->Add(BitCodeAbbrevOp(0));                         // ExtInfoKind
  // RecordDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // FlexibleArrayMember
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // AnonymousStructUnion
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // hasObjectMember
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // hasVolatileMember
  // DC
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // LexicalOffset
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // VisibleOffset
  DeclRecordAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for DECL_PARM_VAR
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_PARM_VAR));
  // Redeclarable
  Abv->Add(BitCodeAbbrevOp(0));                       // No redeclaration
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(0));                       // isUsed
  Abv->Add(BitCodeAbbrevOp(0));                       // isReferenced
  Abv->Add(BitCodeAbbrevOp(0));                   // TopLevelDeclInObjCContainer
  Abv->Add(BitCodeAbbrevOp(AS_none));                 // C++ AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  Abv->Add(BitCodeAbbrevOp(0));                       // AnonDeclNumber
  // ValueDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  // DeclaratorDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // InnerStartLoc
  Abv->Add(BitCodeAbbrevOp(0));                       // hasExtInfo
  // VarDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // StorageClass
  Abv->Add(BitCodeAbbrevOp(0));                       // getTSCSpec
  Abv->Add(BitCodeAbbrevOp(0));                       // hasCXXDirectInitializer
  Abv->Add(BitCodeAbbrevOp(0));                       // Linkage
  Abv->Add(BitCodeAbbrevOp(0));                       // HasInit
  Abv->Add(BitCodeAbbrevOp(0));                   // HasMemberSpecializationInfo
  // ParmVarDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsObjCMethodParameter
  Abv->Add(BitCodeAbbrevOp(0));                       // ScopeDepth
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // ScopeIndex
  Abv->Add(BitCodeAbbrevOp(0));                       // ObjCDeclQualifier
  Abv->Add(BitCodeAbbrevOp(0));                       // KNRPromoted
  Abv->Add(BitCodeAbbrevOp(0));                       // HasInheritedDefaultArg
  Abv->Add(BitCodeAbbrevOp(0));                   // HasUninstantiatedDefaultArg
  // Type Source Info
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // TypeLoc
  DeclParmVarAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for DECL_TYPEDEF
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_TYPEDEF));
  // Redeclarable
  Abv->Add(BitCodeAbbrevOp(0));                       // No redeclaration
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isUsed
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isReferenced
  Abv->Add(BitCodeAbbrevOp(0));                   // TopLevelDeclInObjCContainer
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // C++ AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  Abv->Add(BitCodeAbbrevOp(0));                       // AnonDeclNumber
  // TypeDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Source Location
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type Ref
  // TypedefDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // TypeLoc
  DeclTypedefAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for DECL_VAR
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_VAR));
  // Redeclarable
  Abv->Add(BitCodeAbbrevOp(0));                       // No redeclaration
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(0));                       // isUsed
  Abv->Add(BitCodeAbbrevOp(0));                       // isReferenced
  Abv->Add(BitCodeAbbrevOp(0));                   // TopLevelDeclInObjCContainer
  Abv->Add(BitCodeAbbrevOp(AS_none));                 // C++ AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  Abv->Add(BitCodeAbbrevOp(0));                       // AnonDeclNumber
  // ValueDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  // DeclaratorDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // InnerStartLoc
  Abv->Add(BitCodeAbbrevOp(0));                       // hasExtInfo
  // VarDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // StorageClass
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // getTSCSpec
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // CXXDirectInitializer
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isExceptionVariable
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isNRVOVariable
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isCXXForRangeDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isARCPseudoStrong
  Abv->Add(BitCodeAbbrevOp(0));                         // isConstexpr
  Abv->Add(BitCodeAbbrevOp(0));                         // isInitCapture
  Abv->Add(BitCodeAbbrevOp(0));                         // isPrevDeclInSameScope
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); // Linkage
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // HasInit
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // HasMemberSpecInfo
  // Type Source Info
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // TypeLoc
  DeclVarAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for DECL_CXX_METHOD
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_CXX_METHOD));
  // RedeclarableDecl
  Abv->Add(BitCodeAbbrevOp(0));                         // CanonicalDecl
  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // DeclContext
  Abv->Add(BitCodeAbbrevOp(0));                         // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(0));                         // Invalid
  Abv->Add(BitCodeAbbrevOp(0));                         // HasAttrs
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Implicit
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Used
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Referenced
  Abv->Add(BitCodeAbbrevOp(0));                         // InObjCContainer
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // Access
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // ModulePrivate
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // SubmoduleID
  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(DeclarationName::Identifier)); // NameKind
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // Identifier
  Abv->Add(BitCodeAbbrevOp(0));                         // AnonDeclNumber
  // ValueDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // Type
  // DeclaratorDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // InnerLocStart
  Abv->Add(BitCodeAbbrevOp(0));                         // HasExtInfo
  // FunctionDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 11)); // IDNS
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); // StorageClass
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Inline
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // InlineSpecified
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // VirtualAsWritten
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Pure
  Abv->Add(BitCodeAbbrevOp(0));                         // HasInheritedProto
  Abv->Add(BitCodeAbbrevOp(1));                         // HasWrittenProto
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Deleted
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Trivial
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Defaulted
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // ExplicitlyDefaulted
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // ImplicitReturnZero
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Constexpr
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // SkippedBody
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // LateParsed
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); // Linkage
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));   // LocEnd
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); // TemplateKind
  // This Array slurps the rest of the record. Fortunately we want to encode
  // (nearly) all the remaining (variable number of) fields in the same way.
  //
  // This is the function template information if any, then
  //         NumParams and Params[] from FunctionDecl, and
  //         NumOverriddenMethods, OverriddenMethods[] from CXXMethodDecl.
  //
  //  Add an AbbrevOp for 'size then elements' and use it here.
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));
  DeclCXXMethodAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for EXPR_DECL_REF
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::EXPR_DECL_REF));
  //Stmt
  //Expr
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //TypeDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //ValueDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //InstantiationDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //UnexpandedParamPack
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetValueKind
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetObjectKind
  //DeclRefExpr
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //HasQualifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //GetDeclFound
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //ExplicitTemplateArgs
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //HadMultipleCandidates
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                           1)); // RefersToEnclosingVariableOrCapture
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclRef
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Location
  DeclRefExprAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for EXPR_INTEGER_LITERAL
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::EXPR_INTEGER_LITERAL));
  //Stmt
  //Expr
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //TypeDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //ValueDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //InstantiationDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //UnexpandedParamPack
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetValueKind
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetObjectKind
  //Integer Literal
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Location
  Abv->Add(BitCodeAbbrevOp(32));                      // Bit Width
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Value
  IntegerLiteralAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for EXPR_CHARACTER_LITERAL
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::EXPR_CHARACTER_LITERAL));
  //Stmt
  //Expr
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //TypeDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //ValueDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //InstantiationDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //UnexpandedParamPack
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetValueKind
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetObjectKind
  //Character Literal
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // getValue
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Location
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); // getKind
  CharacterLiteralAbbrev = Stream.EmitAbbrev(Abv);

  // Abbreviation for EXPR_IMPLICIT_CAST
  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::EXPR_IMPLICIT_CAST));
  // Stmt
  // Expr
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //TypeDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //ValueDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //InstantiationDependent
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); //UnexpandedParamPack
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetValueKind
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3)); //GetObjectKind
  // CastExpr
  Abv->Add(BitCodeAbbrevOp(0)); // PathSize
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 6)); // CastKind
  // ImplicitCastExpr
  ExprImplicitCastAbbrev = Stream.EmitAbbrev(Abv);

  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_CONTEXT_LEXICAL));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  DeclContextLexicalAbbrev = Stream.EmitAbbrev(Abv);

  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_CONTEXT_VISIBLE));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  DeclContextVisibleLookupAbbrev = Stream.EmitAbbrev(Abv);
}

/// isRequiredDecl - Check if this is a "required" Decl, which must be seen by
/// consumers of the AST.
///
/// Such decls will always be deserialized from the AST file, so we would like
/// this to be as restrictive as possible. Currently the predicate is driven by
/// code generation requirements, if other clients have a different notion of
/// what is "required" then we may have to consider an alternate scheme where
/// clients can iterate over the top-level decls and get information on them,
/// without necessary deserializing them. We could explicitly require such
/// clients to use a separate API call to "realize" the decl. This should be
/// relatively painless since they would presumably only do it for top-level
/// decls.
static bool isRequiredDecl(const Decl *D, ASTContext &Context,
                           bool WritingModule) {
  // An ObjCMethodDecl is never considered as "required" because its
  // implementation container always is.

  // File scoped assembly or obj-c implementation must be seen.
  if (isa<FileScopeAsmDecl>(D) || isa<ObjCImplDecl>(D))
    return true;

  // ImportDecl is used by codegen to determine the set of imported modules to
  // search for inputs for automatic linking; include it if it has a semantic
  // effect.
  if (isa<ImportDecl>(D) && !WritingModule)
    return true;

  return Context.DeclMustBeEmitted(D);
}

void ASTWriter::WriteDecl(ASTContext &Context, Decl *D) {
  // Switch case IDs are per Decl.
  ClearSwitchCaseIDs();

  RecordData Record;
  ASTDeclWriter W(*this, Context, Record);

  // Determine the ID for this declaration.
  serialization::DeclID ID;
  assert(!D->isFromASTFile() && "should not be emitting imported decl");
  serialization::DeclID &IDR = DeclIDs[D];
  if (IDR == 0)
    IDR = NextDeclID++;
    
  ID = IDR;

  bool isReplacingADecl = ID < FirstDeclID;

  // If this declaration is also a DeclContext, write blocks for the
  // declarations that lexically stored inside its context and those
  // declarations that are visible from its context. These blocks
  // are written before the declaration itself so that we can put
  // their offsets into the record for the declaration.
  uint64_t LexicalOffset = 0;
  uint64_t VisibleOffset = 0;
  DeclContext *DC = dyn_cast<DeclContext>(D);
  if (DC) {
    if (isReplacingADecl) {
      // It is replacing a decl from a chained PCH; make sure that the
      // DeclContext is fully loaded.
      if (DC->hasExternalLexicalStorage())
        DC->LoadLexicalDeclsFromExternalStorage();
      if (DC->hasExternalVisibleStorage())
        Chain->completeVisibleDeclsMap(DC);
    }
    LexicalOffset = WriteDeclContextLexicalBlock(Context, DC);
    VisibleOffset = WriteDeclContextVisibleBlock(Context, DC);
  }
  
  // Build a record for this declaration
  Record.clear();
  W.Code = (serialization::DeclCode)0;
  W.AbbrevToUse = 0;
  W.Visit(D);
  if (DC) W.VisitDeclContext(DC, LexicalOffset, VisibleOffset);

  if (isReplacingADecl) {
    // We're replacing a decl in a previous file.
    ReplacedDecls.push_back(ReplacedDeclInfo(ID, Stream.GetCurrentBitNo(),
                                             D->getLocation()));
  } else {
    unsigned Index = ID - FirstDeclID;

    // Record the offset for this declaration
    SourceLocation Loc = D->getLocation();
    if (DeclOffsets.size() == Index)
      DeclOffsets.push_back(DeclOffset(Loc, Stream.GetCurrentBitNo()));
    else if (DeclOffsets.size() < Index) {
      DeclOffsets.resize(Index+1);
      DeclOffsets[Index].setLocation(Loc);
      DeclOffsets[Index].BitOffset = Stream.GetCurrentBitNo();
    }

    SourceManager &SM = Context.getSourceManager();
    if (Loc.isValid() && SM.isLocalSourceLocation(Loc))
      associateDeclWithFile(D, ID);
  }

  if (!W.Code)
    llvm::report_fatal_error(StringRef("unexpected declaration kind '") +
                            D->getDeclKindName() + "'");
  Stream.EmitRecord(W.Code, Record, W.AbbrevToUse);

  // Flush any expressions, base specifiers, and ctor initializers that
  // were written as part of this declaration.
  FlushPendingAfterDecl();

  // Note declarations that should be deserialized eagerly so that we can add
  // them to a record in the AST file later.
  if (isRequiredDecl(D, Context, WritingModule))
    EagerlyDeserializedDecls.push_back(ID);
}

void ASTWriter::AddFunctionDefinition(const FunctionDecl *FD,
                                      RecordData &Record) {
  ClearSwitchCaseIDs();

  ASTDeclWriter W(*this, FD->getASTContext(), Record);
  W.AddFunctionDefinition(FD);
}
