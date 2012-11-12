//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"

#include "clang/AST/DeclVisitor.h"

using namespace clang;
using namespace cxindex;

namespace {

class IndexingDeclVisitor : public DeclVisitor<IndexingDeclVisitor, bool> {
  IndexingContext &IndexCtx;

public:
  explicit IndexingDeclVisitor(IndexingContext &indexCtx)
    : IndexCtx(indexCtx) { }

  void handleDeclarator(DeclaratorDecl *D, const NamedDecl *Parent = 0) {
    if (!Parent) Parent = D;

    if (!IndexCtx.shouldIndexFunctionLocalSymbols()) {
      IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), Parent);
      IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), Parent);
    } else {
      if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
        IndexCtx.handleVar(Parm);
      } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        for (FunctionDecl::param_iterator
               PI = FD->param_begin(), PE = FD->param_end(); PI != PE; ++PI) {
          IndexCtx.handleVar(*PI);
        }
      }
    }
  }

  void handleObjCMethod(ObjCMethodDecl *D) {
    IndexCtx.handleObjCMethod(D);
    if (D->isImplicit())
      return;

    IndexCtx.indexTypeSourceInfo(D->getResultTypeSourceInfo(), D);
    for (ObjCMethodDecl::param_iterator
           I = D->param_begin(), E = D->param_end(); I != E; ++I)
      handleDeclarator(*I, D);

    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, D);
      }
    }
  }

  bool VisitFunctionDecl(FunctionDecl *D) {
    IndexCtx.handleFunction(D);
    handleDeclarator(D);

    if (CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(D)) {
      // Constructor initializers.
      for (CXXConstructorDecl::init_iterator I = Ctor->init_begin(),
                                             E = Ctor->init_end();
           I != E; ++I) {
        CXXCtorInitializer *Init = *I;
        if (Init->isWritten()) {
          IndexCtx.indexTypeSourceInfo(Init->getTypeSourceInfo(), D);
          if (const FieldDecl *Member = Init->getAnyMember())
            IndexCtx.handleReference(Member, Init->getMemberLocation(), D, D);
          IndexCtx.indexBody(Init->getInit(), D, D);
        }
      }
    }

    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, D);
      }
    }
    return true;
  }

  bool VisitVarDecl(VarDecl *D) {
    IndexCtx.handleVar(D);
    handleDeclarator(D);
    IndexCtx.indexBody(D->getInit(), D);
    return true;
  }

  bool VisitFieldDecl(FieldDecl *D) {
    IndexCtx.handleField(D);
    handleDeclarator(D);
    if (D->isBitField())
      IndexCtx.indexBody(D->getBitWidth(), D);
    else if (D->hasInClassInitializer())
      IndexCtx.indexBody(D->getInClassInitializer(), D);
    return true;
  }
  
  bool VisitEnumConstantDecl(EnumConstantDecl *D) {
    IndexCtx.handleEnumerator(D);
    IndexCtx.indexBody(D->getInitExpr(), D);
    return true;
  }

  bool VisitTypedefNameDecl(TypedefNameDecl *D) {
    IndexCtx.handleTypedefName(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitTagDecl(TagDecl *D) {
    // Non-free standing tags are handled in indexTypeSourceInfo.
    if (D->isFreeStanding())
      IndexCtx.indexTagDecl(D);
    return true;
  }

  bool VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
    IndexCtx.handleObjCInterface(D);

    if (D->isThisDeclarationADefinition()) {
      IndexCtx.indexTUDeclsInObjCContainer();
      IndexCtx.indexDeclContext(D);
    }
    return true;
  }

  bool VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
    IndexCtx.handleObjCProtocol(D);

    if (D->isThisDeclarationADefinition()) {
      IndexCtx.indexTUDeclsInObjCContainer();
      IndexCtx.indexDeclContext(D);
    }
    return true;
  }

  bool VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
    const ObjCInterfaceDecl *Class = D->getClassInterface();
    if (!Class)
      return true;

    if (Class->isImplicitInterfaceDecl())
      IndexCtx.handleObjCInterface(Class);

    IndexCtx.handleObjCImplementation(D);

    IndexCtx.indexTUDeclsInObjCContainer();

    // Index the ivars first to make sure the synthesized ivars are indexed
    // before indexing the methods that can reference them.
    for (ObjCImplementationDecl::ivar_iterator
           IvarI = D->ivar_begin(),
           IvarE = D->ivar_end(); IvarI != IvarE; ++IvarI) {
      IndexCtx.indexDecl(*IvarI);
    }
    for (DeclContext::decl_iterator
           I = D->decls_begin(), E = D->decls_end(); I != E; ++I) {
      if (!isa<ObjCIvarDecl>(*I))
        IndexCtx.indexDecl(*I);
    }

    return true;
  }

  bool VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
    IndexCtx.handleObjCCategory(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
    const ObjCCategoryDecl *Cat = D->getCategoryDecl();
    if (!Cat)
      return true;

    IndexCtx.handleObjCCategoryImpl(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCMethodDecl(ObjCMethodDecl *D) {
    // Methods associated with a property, even user-declared ones, are
    // handled when we handle the property.
    if (D->isPropertyAccessor())
      return true;

    handleObjCMethod(D);
    return true;
  }

  bool VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
    if (ObjCMethodDecl *MD = D->getGetterMethodDecl())
      if (MD->getLexicalDeclContext() == D->getLexicalDeclContext())
        handleObjCMethod(MD);
    if (ObjCMethodDecl *MD = D->getSetterMethodDecl())
      if (MD->getLexicalDeclContext() == D->getLexicalDeclContext())
        handleObjCMethod(MD);
    IndexCtx.handleObjCProperty(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
    ObjCPropertyDecl *PD = D->getPropertyDecl();
    IndexCtx.handleSynthesizedObjCProperty(D);

    if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
      return true;
    assert(D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize);
    
    if (ObjCIvarDecl *IvarD = D->getPropertyIvarDecl()) {
      if (!IvarD->getSynthesize())
        IndexCtx.handleReference(IvarD, D->getPropertyIvarDeclLoc(), 0,
                                 D->getDeclContext());
    }

    if (ObjCMethodDecl *MD = PD->getGetterMethodDecl()) {
      if (MD->isPropertyAccessor())
        IndexCtx.handleSynthesizedObjCMethod(MD, D->getLocation(),
                                             D->getLexicalDeclContext());
    }
    if (ObjCMethodDecl *MD = PD->getSetterMethodDecl()) {
      if (MD->isPropertyAccessor())
        IndexCtx.handleSynthesizedObjCMethod(MD, D->getLocation(),
                                             D->getLexicalDeclContext());
    }
    return true;
  }

  bool VisitNamespaceDecl(NamespaceDecl *D) {
    IndexCtx.handleNamespace(D);
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitUsingDecl(UsingDecl *D) {
    // FIXME: Parent for the following is CXIdxEntity_Unexposed with no USR,
    // we should do better.

    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), D);
    for (UsingDecl::shadow_iterator
           I = D->shadow_begin(), E = D->shadow_end(); I != E; ++I) {
      IndexCtx.handleReference((*I)->getUnderlyingDecl(), D->getLocation(),
                               D, D->getLexicalDeclContext());
    }
    return true;
  }

  bool VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
    // FIXME: Parent for the following is CXIdxEntity_Unexposed with no USR,
    // we should do better.

    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), D);
    IndexCtx.handleReference(D->getNominatedNamespaceAsWritten(),
                             D->getLocation(), D, D->getLexicalDeclContext());
    return true;
  }

  bool VisitClassTemplateDecl(ClassTemplateDecl *D) {
    IndexCtx.handleClassTemplate(D);
    if (D->isThisDeclarationADefinition())
      IndexCtx.indexDeclContext(D->getTemplatedDecl());
    return true;
  }

  bool VisitClassTemplateSpecializationDecl(
                                           ClassTemplateSpecializationDecl *D) {
    // FIXME: Notify subsequent callbacks if info comes from implicit
    // instantiation.
    if (D->isThisDeclarationADefinition() &&
        (IndexCtx.shouldIndexImplicitTemplateInsts() ||
         !IndexCtx.isTemplateImplicitInstantiation(D)))
      IndexCtx.indexTagDecl(D);
    return true;
  }

  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
    IndexCtx.handleFunctionTemplate(D);
    FunctionDecl *FD = D->getTemplatedDecl();
    handleDeclarator(FD, D);
    if (FD->isThisDeclarationADefinition()) {
      const Stmt *Body = FD->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, FD);
      }
    }
    return true;
  }

  bool VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
    IndexCtx.handleTypeAliasTemplate(D);
    IndexCtx.indexTypeSourceInfo(D->getTemplatedDecl()->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitImportDecl(ImportDecl *D) {
    IndexCtx.importedModule(D);
    return true;
  }
};

} // anonymous namespace

void IndexingContext::indexDecl(const Decl *D) {
  if (D->isImplicit() && shouldIgnoreIfImplicit(D))
    return;

  bool Handled = IndexingDeclVisitor(*this).Visit(const_cast<Decl*>(D));
  if (!Handled && isa<DeclContext>(D))
    indexDeclContext(cast<DeclContext>(D));
}

void IndexingContext::indexDeclContext(const DeclContext *DC) {
  for (DeclContext::decl_iterator
         I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I) {
    indexDecl(*I);
  }
}

void IndexingContext::indexTopLevelDecl(const Decl *D) {
  if (isNotFromSourceFile(D->getLocation()))
    return;

  if (isa<ObjCMethodDecl>(D))
    return; // Wait for the objc container.

  indexDecl(D);
}

void IndexingContext::indexDeclGroupRef(DeclGroupRef DG) {
  for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I)
    indexTopLevelDecl(*I);
}

void IndexingContext::indexTUDeclsInObjCContainer() {
  while (!TUDeclsInObjCContainer.empty()) {
    DeclGroupRef DG = TUDeclsInObjCContainer.front();
    TUDeclsInObjCContainer.pop_front();
    indexDeclGroupRef(DG);
  }
}
