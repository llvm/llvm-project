//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"

#include "RecursiveASTVisitor.h"

using namespace clang;
using namespace cxindex;

namespace {

class TypeIndexer : public cxindex::RecursiveASTVisitor<TypeIndexer> {
  IndexingContext &IndexCtx;
  const NamedDecl *Parent;
  const DeclContext *ParentDC;

public:
  TypeIndexer(IndexingContext &indexCtx, const NamedDecl *parent,
              const DeclContext *DC)
    : IndexCtx(indexCtx), Parent(parent), ParentDC(DC) { }
  
  bool shouldWalkTypesOfTypeLocs() const { return false; }

  bool VisitTypedefTypeLoc(TypedefTypeLoc TL) {
    IndexCtx.handleReference(TL.getTypedefNameDecl(), TL.getNameLoc(),
                             Parent, ParentDC);
    return true;
  }

  bool TraverseNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS) {
    IndexCtx.indexNestedNameSpecifierLoc(NNS, Parent, ParentDC);
    return true;
  }

  bool VisitTagTypeLoc(TagTypeLoc TL) {
    TagDecl *D = TL.getDecl();
    if (D->getParentFunctionOrMethod())
      return true;

    if (TL.isDefinition()) {
      IndexCtx.indexTagDecl(D);
      return true;
    }

    if (D->getLocation() == TL.getNameLoc())
      IndexCtx.handleTagDecl(D);
    else
      IndexCtx.handleReference(D, TL.getNameLoc(),
                               Parent, ParentDC);
    return true;
  }

  bool VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
    IndexCtx.handleReference(TL.getIFaceDecl(), TL.getNameLoc(),
                             Parent, ParentDC);
    return true;
  }

  bool VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
    for (unsigned i = 0, e = TL.getNumProtocols(); i != e; ++i) {
      IndexCtx.handleReference(TL.getProtocol(i), TL.getProtocolLoc(i),
                               Parent, ParentDC);
    }
    return true;
  }

  bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc TL) {
    if (const TemplateSpecializationType *T = TL.getTypePtr()) {
      if (IndexCtx.shouldIndexImplicitTemplateInsts()) {
        if (CXXRecordDecl *RD = T->getAsCXXRecordDecl())
          IndexCtx.handleReference(RD, TL.getTemplateNameLoc(),
                                   Parent, ParentDC);
      } else {
        if (const TemplateDecl *D = T->getTemplateName().getAsTemplateDecl())
          IndexCtx.handleReference(D, TL.getTemplateNameLoc(),
                                   Parent, ParentDC);
      }
    }
    return true;
  }

  bool TraverseStmt(Stmt *S) {
    IndexCtx.indexBody(S, Parent, ParentDC);
    return true;
  }
};

} // anonymous namespace

void IndexingContext::indexTypeSourceInfo(TypeSourceInfo *TInfo,
                                          const NamedDecl *Parent,
                                          const DeclContext *DC) {
  if (!TInfo || TInfo->getTypeLoc().isNull())
    return;
  
  indexTypeLoc(TInfo->getTypeLoc(), Parent, DC);
}

void IndexingContext::indexTypeLoc(TypeLoc TL,
                                   const NamedDecl *Parent,
                                   const DeclContext *DC) {
  if (TL.isNull())
    return;

  if (DC == 0)
    DC = Parent->getLexicalDeclContext();
  TypeIndexer(*this, Parent, DC).TraverseTypeLoc(TL);
}

void IndexingContext::indexNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
                                                  const NamedDecl *Parent,
                                                  const DeclContext *DC) {
  if (!NNS)
    return;

  if (NestedNameSpecifierLoc Prefix = NNS.getPrefix())
    indexNestedNameSpecifierLoc(Prefix, Parent, DC);

  if (DC == 0)
    DC = Parent->getLexicalDeclContext();
  SourceLocation Loc = NNS.getSourceRange().getBegin();

  switch (NNS.getNestedNameSpecifier()->getKind()) {
  case NestedNameSpecifier::Identifier:
  case NestedNameSpecifier::Global:
    break;

  case NestedNameSpecifier::Namespace:
    handleReference(NNS.getNestedNameSpecifier()->getAsNamespace(),
                    Loc, Parent, DC);
    break;
  case NestedNameSpecifier::NamespaceAlias:
    handleReference(NNS.getNestedNameSpecifier()->getAsNamespaceAlias(),
                    Loc, Parent, DC);
    break;

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    indexTypeLoc(NNS.getTypeLoc(), Parent, DC);
    break;
  }
}

void IndexingContext::indexTagDecl(const TagDecl *D) {
  if (handleTagDecl(D)) {
    if (D->isThisDeclarationADefinition())
      indexDeclContext(D);
  }
}
