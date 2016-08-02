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

class IndexingDeclVisitor : public ConstDeclVisitor<IndexingDeclVisitor, bool> {
  IndexingContext &IndexCtx;

public:
  explicit IndexingDeclVisitor(IndexingContext &indexCtx)
    : IndexCtx(indexCtx) { }

  /// \brief Returns true if the given method has been defined explicitly by the
  /// user.
  static bool hasUserDefined(const ObjCMethodDecl *D,
                             const ObjCImplDecl *Container) {
    const ObjCMethodDecl *MD = Container->getMethod(D->getSelector(),
                                                    D->isInstanceMethod());
    return MD && !MD->isImplicit() && MD->isThisDeclarationADefinition();
  }

  void handleDeclarator(const DeclaratorDecl *D,
                        const NamedDecl *Parent = nullptr) {
    if (!Parent) Parent = D;

    if (!IndexCtx.shouldIndexFunctionLocalSymbols()) {
      IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), Parent);
      IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), Parent);
    } else {
      if (const ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
        IndexCtx.handleVar(Parm);
      } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        for (auto PI : FD->params()) {
          IndexCtx.handleVar(PI);
        }
      }
    }
  }

  void handleObjCMethod(const ObjCMethodDecl *D) {
    IndexCtx.handleObjCMethod(D);
    if (D->isImplicit())
      return;

    IndexCtx.indexTypeSourceInfo(D->getReturnTypeSourceInfo(), D);
    for (const auto *I : D->params())
      handleDeclarator(I, D);

    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, D);
      }
    }
  }

  bool VisitFunctionDecl(const FunctionDecl *D) {
    IndexCtx.handleFunction(D);
    handleDeclarator(D);

    if (const CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(D)) {
      // Constructor initializers.
      for (const auto *Init : Ctor->inits()) {
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

  bool VisitVarDecl(const VarDecl *D) {
    IndexCtx.handleVar(D);
    handleDeclarator(D);
    IndexCtx.indexBody(D->getInit(), D);
    return true;
  }

  bool VisitFieldDecl(const FieldDecl *D) {
    IndexCtx.handleField(D);
    handleDeclarator(D);
    if (D->isBitField())
      IndexCtx.indexBody(D->getBitWidth(), D);
    else if (D->hasInClassInitializer())
      IndexCtx.indexBody(D->getInClassInitializer(), D);
    return true;
  }

  bool VisitMSPropertyDecl(const MSPropertyDecl *D) {
    handleDeclarator(D);
    return true;
  }

  bool VisitEnumConstantDecl(const EnumConstantDecl *D) {
    IndexCtx.handleEnumerator(D);
    IndexCtx.indexBody(D->getInitExpr(), D);
    return true;
  }

  bool VisitTypedefNameDecl(const TypedefNameDecl *D) {
    IndexCtx.handleTypedefName(D);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitTagDecl(const TagDecl *D) {
    // Non-free standing tags are handled in indexTypeSourceInfo.
    if (D->isFreeStanding())
      IndexCtx.indexTagDecl(D);
    return true;
  }

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
    IndexCtx.handleObjCInterface(D);

    if (D->isThisDeclarationADefinition()) {
      IndexCtx.indexTUDeclsInObjCContainer();
      IndexCtx.indexDeclContext(D);
    }
    return true;
  }

  bool VisitObjCProtocolDecl(const ObjCProtocolDecl *D) {
    IndexCtx.handleObjCProtocol(D);

    if (D->isThisDeclarationADefinition()) {
      IndexCtx.indexTUDeclsInObjCContainer();
      IndexCtx.indexDeclContext(D);
    }
    return true;
  }

  bool VisitObjCImplementationDecl(const ObjCImplementationDecl *D) {
    const ObjCInterfaceDecl *Class = D->getClassInterface();
    if (!Class)
      return true;

    if (Class->isImplicitInterfaceDecl())
      IndexCtx.handleObjCInterface(Class);

    IndexCtx.handleObjCImplementation(D);

    IndexCtx.indexTUDeclsInObjCContainer();

    // Index the ivars first to make sure the synthesized ivars are indexed
    // before indexing the methods that can reference them.
    for (const auto *IvarI : D->ivars())
      IndexCtx.indexDecl(IvarI);
    for (const auto *I : D->decls()) {
      if (!isa<ObjCIvarDecl>(I))
        IndexCtx.indexDecl(I);
    }

    return true;
  }

  bool VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
    IndexCtx.handleObjCCategory(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D) {
    const ObjCCategoryDecl *Cat = D->getCategoryDecl();
    if (!Cat)
      return true;

    IndexCtx.handleObjCCategoryImpl(D);

    IndexCtx.indexTUDeclsInObjCContainer();
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitObjCMethodDecl(const ObjCMethodDecl *D) {
    // Methods associated with a property, even user-declared ones, are
    // handled when we handle the property.
    if (D->isPropertyAccessor())
      return true;

    handleObjCMethod(D);
    return true;
  }

  bool VisitObjCPropertyDecl(const ObjCPropertyDecl *D) {
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

  bool VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D) {
    ObjCPropertyDecl *PD = D->getPropertyDecl();
    IndexCtx.handleSynthesizedObjCProperty(D);

    if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
      return true;
    assert(D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize);
    
    if (ObjCIvarDecl *IvarD = D->getPropertyIvarDecl()) {
      if (!IvarD->getSynthesize())
        IndexCtx.handleReference(IvarD, D->getPropertyIvarDeclLoc(), nullptr,
                                 D->getDeclContext());
    }

    if (ObjCMethodDecl *MD = PD->getGetterMethodDecl()) {
      if (MD->isPropertyAccessor() &&
          !hasUserDefined(MD, cast<ObjCImplDecl>(D->getDeclContext())))
        IndexCtx.handleSynthesizedObjCMethod(MD, D->getLocation(),
                                             D->getLexicalDeclContext());
    }
    if (ObjCMethodDecl *MD = PD->getSetterMethodDecl()) {
      if (MD->isPropertyAccessor() &&
          !hasUserDefined(MD, cast<ObjCImplDecl>(D->getDeclContext())))
        IndexCtx.handleSynthesizedObjCMethod(MD, D->getLocation(),
                                             D->getLexicalDeclContext());
    }
    return true;
  }

  bool VisitNamespaceDecl(const NamespaceDecl *D) {
    IndexCtx.handleNamespace(D);
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitUsingDecl(const UsingDecl *D) {
    // FIXME: Parent for the following is CXIdxEntity_Unexposed with no USR,
    // we should do better.

    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), D);
    for (const auto *I : D->shadows())
      IndexCtx.handleReference(I->getUnderlyingDecl(), D->getLocation(), D,
                               D->getLexicalDeclContext());
    return true;
  }

  bool VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
    // FIXME: Parent for the following is CXIdxEntity_Unexposed with no USR,
    // we should do better.

    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), D);
    IndexCtx.handleReference(D->getNominatedNamespaceAsWritten(),
                             D->getLocation(), D, D->getLexicalDeclContext());
    return true;
  }

  bool VisitClassTemplateDecl(const ClassTemplateDecl *D) {
    IndexCtx.handleClassTemplate(D);
    if (D->isThisDeclarationADefinition())
      IndexCtx.indexDeclContext(D->getTemplatedDecl());
    return true;
  }

  bool VisitClassTemplateSpecializationDecl(const
                                           ClassTemplateSpecializationDecl *D) {
    // FIXME: Notify subsequent callbacks if info comes from implicit
    // instantiation.
    if (D->isThisDeclarationADefinition() &&
        (IndexCtx.shouldIndexImplicitTemplateInsts() ||
         !IndexCtx.isTemplateImplicitInstantiation(D)))
      IndexCtx.indexTagDecl(D);
    return true;
  }

  bool VisitFunctionTemplateDecl(const FunctionTemplateDecl *D) {
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

  bool VisitTypeAliasTemplateDecl(const TypeAliasTemplateDecl *D) {
    IndexCtx.handleTypeAliasTemplate(D);
    IndexCtx.indexTypeSourceInfo(D->getTemplatedDecl()->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitImportDecl(const ImportDecl *D) {
    IndexCtx.importedModule(D);
    return true;
  }
};

} // anonymous namespace

void IndexingContext::indexDecl(const Decl *D) {
  if (D->isImplicit() && shouldIgnoreIfImplicit(D))
    return;

  bool Handled = IndexingDeclVisitor(*this).Visit(D);
  if (!Handled && isa<DeclContext>(D))
    indexDeclContext(cast<DeclContext>(D));
}

void IndexingContext::indexDeclContext(const DeclContext *DC) {
  for (const auto *I : DC->decls())
    indexDecl(I);
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
