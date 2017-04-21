//===- IndexDecl.cpp - Indexing declarations ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/AST/DeclVisitor.h"

using namespace clang;
using namespace index;

#define TRY_DECL(D,CALL_EXPR)                                                  \
  do {                                                                         \
    if (!IndexCtx.shouldIndex(D)) return true;                                 \
    if (!CALL_EXPR)                                                            \
      return false;                                                            \
  } while (0)

#define TRY_TO(CALL_EXPR)                                                      \
  do {                                                                         \
    if (!CALL_EXPR)                                                            \
      return false;                                                            \
  } while (0)

namespace {

class IndexingDeclVisitor : public ConstDeclVisitor<IndexingDeclVisitor, bool> {
  IndexingContext &IndexCtx;

public:
  explicit IndexingDeclVisitor(IndexingContext &indexCtx)
    : IndexCtx(indexCtx) { }

  bool Handled = true;

  bool VisitDecl(const Decl *D) {
    Handled = false;
    return true;
  }

  /// \brief Returns true if the given method has been defined explicitly by the
  /// user.
  static bool hasUserDefined(const ObjCMethodDecl *D,
                             const ObjCImplDecl *Container) {
    const ObjCMethodDecl *MD = Container->getMethod(D->getSelector(),
                                                    D->isInstanceMethod());
    return MD && !MD->isImplicit() && MD->isThisDeclarationADefinition();
  }

  void handleDeclarator(const DeclaratorDecl *D,
                        const NamedDecl *Parent = nullptr,
                        bool isIBType = false) {
    if (!Parent) Parent = D;

    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), Parent,
                                 Parent->getLexicalDeclContext(),
                                 /*isBase=*/false, isIBType);
    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), Parent);
    if (IndexCtx.shouldIndexFunctionLocalSymbols()) {
      // Only index parameters in definitions, parameters in declarations are
      // not useful.
      if (const ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
        auto *DC = Parm->getDeclContext();
        if (auto *FD = dyn_cast<FunctionDecl>(DC)) {
          if (FD->isThisDeclarationADefinition())
            IndexCtx.handleDecl(Parm);
        } else if (auto *MD = dyn_cast<ObjCMethodDecl>(DC)) {
          if (MD->isThisDeclarationADefinition())
            IndexCtx.handleDecl(Parm);
        } else {
          IndexCtx.handleDecl(Parm);
        }
      } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        if (FD->isThisDeclarationADefinition()) {
          for (auto PI : FD->parameters()) {
            IndexCtx.handleDecl(PI);
          }
        }
      }
    }
  }

  bool handleObjCMethod(const ObjCMethodDecl *D,
                        const ObjCPropertyDecl *AssociatedProp = nullptr) {
    SmallVector<SymbolRelation, 4> Relations;
    SmallVector<const ObjCMethodDecl*, 4> Overriden;

    D->getOverriddenMethods(Overriden);
    for(auto overridden: Overriden) {
      Relations.emplace_back((unsigned) SymbolRole::RelationOverrideOf,
                             overridden);
    }
    if (AssociatedProp)
      Relations.emplace_back((unsigned)SymbolRole::RelationAccessorOf,
                             AssociatedProp);

    // getLocation() returns beginning token of a method declaration, but for
    // indexing purposes we want to point to the base name.
    SourceLocation MethodLoc = D->getSelectorStartLoc();
    if (MethodLoc.isInvalid())
      MethodLoc = D->getLocation();

    SourceLocation AttrLoc;

    // check for (getter=/setter=)
    if (AssociatedProp) {
      bool isGetter = !D->param_size();
      AttrLoc = isGetter ?
        AssociatedProp->getGetterNameLoc():
        AssociatedProp->getSetterNameLoc();
    }

    SymbolRoleSet Roles = (SymbolRoleSet)SymbolRole::Dynamic;
    if (D->isImplicit()) {
      if (AttrLoc.isValid()) {
        MethodLoc = AttrLoc;
      } else {
        Roles |= (SymbolRoleSet)SymbolRole::Implicit;
      }
    } else if (AttrLoc.isValid()) {
      IndexCtx.handleReference(D, AttrLoc, cast<NamedDecl>(D->getDeclContext()),
                               D->getDeclContext(), 0);
    }

    TRY_DECL(D, IndexCtx.handleDecl(D, MethodLoc, Roles, Relations));
    IndexCtx.indexTypeSourceInfo(D->getReturnTypeSourceInfo(), D);
    bool hasIBActionAndFirst = D->hasAttr<IBActionAttr>();
    for (const auto *I : D->parameters()) {
      handleDeclarator(I, D, /*isIBType=*/hasIBActionAndFirst);
      hasIBActionAndFirst = false;
    }

    if (D->isThisDeclarationADefinition()) {
      const Stmt *Body = D->getBody();
      if (Body) {
        IndexCtx.indexBody(Body, D, D);
      }
    }
    return true;
  }

  bool VisitFunctionDecl(const FunctionDecl *D) {
    if (D->isDeleted())
      return true;

    SymbolRoleSet Roles{};
    SmallVector<SymbolRelation, 4> Relations;
    if (auto *CXXMD = dyn_cast<CXXMethodDecl>(D)) {
      if (CXXMD->isVirtual())
        Roles |= (unsigned)SymbolRole::Dynamic;
      for (auto I = CXXMD->begin_overridden_methods(),
           E = CXXMD->end_overridden_methods(); I != E; ++I) {
        Relations.emplace_back((unsigned)SymbolRole::RelationOverrideOf, *I);
      }
    }

    TRY_DECL(D, IndexCtx.handleDecl(D, Roles, Relations));
    handleDeclarator(D);

    if (const CXXConstructorDecl *Ctor = dyn_cast<CXXConstructorDecl>(D)) {
      IndexCtx.handleReference(Ctor->getParent(), Ctor->getLocation(),
                               Ctor->getParent(), Ctor->getDeclContext());

      // Constructor initializers.
      for (const auto *Init : Ctor->inits()) {
        if (Init->isWritten()) {
          IndexCtx.indexTypeSourceInfo(Init->getTypeSourceInfo(), D);
          if (const FieldDecl *Member = Init->getAnyMember())
            IndexCtx.handleReference(Member, Init->getMemberLocation(), D, D,
                                     (unsigned)SymbolRole::Write);
          IndexCtx.indexBody(Init->getInit(), D, D);
        }
      }
    } else if (const CXXDestructorDecl *Dtor = dyn_cast<CXXDestructorDecl>(D)) {
      if (auto TypeNameInfo = Dtor->getNameInfo().getNamedTypeInfo()) {
        IndexCtx.handleReference(Dtor->getParent(),
                                 TypeNameInfo->getTypeLoc().getLocStart(),
                                 Dtor->getParent(), Dtor->getDeclContext());
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
    TRY_DECL(D, IndexCtx.handleDecl(D));
    handleDeclarator(D);
    IndexCtx.indexBody(D->getInit(), D);
    return true;
  }

  bool VisitFieldDecl(const FieldDecl *D) {
    TRY_DECL(D, IndexCtx.handleDecl(D));
    handleDeclarator(D);
    if (D->isBitField())
      IndexCtx.indexBody(D->getBitWidth(), D);
    else if (D->hasInClassInitializer())
      IndexCtx.indexBody(D->getInClassInitializer(), D);
    return true;
  }

  bool VisitObjCIvarDecl(const ObjCIvarDecl *D) {
    if (D->getSynthesize()) {
      // handled in VisitObjCPropertyImplDecl
      return true;
    }
    TRY_DECL(D, IndexCtx.handleDecl(D));
    handleDeclarator(D);
    return true;
  }

  bool VisitMSPropertyDecl(const MSPropertyDecl *D) {
    handleDeclarator(D);
    return true;
  }

  bool VisitEnumConstantDecl(const EnumConstantDecl *D) {
    TRY_DECL(D, IndexCtx.handleDecl(D));
    IndexCtx.indexBody(D->getInitExpr(), D);
    return true;
  }

  bool VisitTypedefNameDecl(const TypedefNameDecl *D) {
    if (!D->isTransparentTag()) {
      TRY_DECL(D, IndexCtx.handleDecl(D));
      IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    }
    return true;
  }

  bool VisitTagDecl(const TagDecl *D) {
    // Non-free standing tags are handled in indexTypeSourceInfo.
    if (D->isFreeStanding()) {
      if (D->isThisDeclarationADefinition()) {
        IndexCtx.indexTagDecl(D);
      } else {
        auto *Parent = dyn_cast<NamedDecl>(D->getDeclContext());
        return IndexCtx.handleReference(D, D->getLocation(), Parent,
                                        D->getLexicalDeclContext(),
                                        SymbolRoleSet());
      }
    }
    return true;
  }

  bool handleReferencedProtocols(const ObjCProtocolList &ProtList,
                                 const ObjCContainerDecl *ContD,
                                 SourceLocation SuperLoc) {
    ObjCInterfaceDecl::protocol_loc_iterator LI = ProtList.loc_begin();
    for (ObjCInterfaceDecl::protocol_iterator
         I = ProtList.begin(), E = ProtList.end(); I != E; ++I, ++LI) {
      SourceLocation Loc = *LI;
      ObjCProtocolDecl *PD = *I;
      SymbolRoleSet roles{};
      if (Loc == SuperLoc)
        roles |= (SymbolRoleSet)SymbolRole::Implicit;
      TRY_TO(IndexCtx.handleReference(PD, Loc, ContD, ContD, roles,
          SymbolRelation{(unsigned)SymbolRole::RelationBaseOf, ContD}));
    }
    return true;
  }

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *D) {
    if (D->isThisDeclarationADefinition()) {
      TRY_DECL(D, IndexCtx.handleDecl(D));
      SourceLocation SuperLoc = D->getSuperClassLoc();
      if (auto *SuperD = D->getSuperClass()) {
        bool hasSuperTypedef = false;
        if (auto *TInfo = D->getSuperClassTInfo()) {
          if (auto *TT = TInfo->getType()->getAs<TypedefType>()) {
            if (auto *TD = TT->getDecl()) {
              hasSuperTypedef = true;
              TRY_TO(IndexCtx.handleReference(TD, SuperLoc, D, D,
                                              SymbolRoleSet()));
            }
          }
        }
        SymbolRoleSet superRoles{};
        if (hasSuperTypedef)
          superRoles |= (SymbolRoleSet)SymbolRole::Implicit;
        TRY_TO(IndexCtx.handleReference(SuperD, SuperLoc, D, D, superRoles,
            SymbolRelation{(unsigned)SymbolRole::RelationBaseOf, D}));
      }
      TRY_TO(handleReferencedProtocols(D->getReferencedProtocols(), D,
                                       SuperLoc));
      TRY_TO(IndexCtx.indexDeclContext(D));
    } else {
      return IndexCtx.handleReference(D, D->getLocation(), nullptr,
                                      D->getDeclContext(), SymbolRoleSet());
    }
    return true;
  }

  bool VisitObjCProtocolDecl(const ObjCProtocolDecl *D) {
    if (D->isThisDeclarationADefinition()) {
      TRY_DECL(D, IndexCtx.handleDecl(D));
      TRY_TO(handleReferencedProtocols(D->getReferencedProtocols(), D,
                                       /*superLoc=*/SourceLocation()));
      TRY_TO(IndexCtx.indexDeclContext(D));
    } else {
      return IndexCtx.handleReference(D, D->getLocation(), nullptr,
                                      D->getDeclContext(), SymbolRoleSet());
    }
    return true;
  }

  bool VisitObjCImplementationDecl(const ObjCImplementationDecl *D) {
    const ObjCInterfaceDecl *Class = D->getClassInterface();
    if (!Class)
      return true;

    if (Class->isImplicitInterfaceDecl())
      IndexCtx.handleDecl(Class);

    TRY_DECL(D, IndexCtx.handleDecl(D));

    // Visit implicit @synthesize property implementations first as their
    // location is reported at the name of the @implementation block. This
    // serves no purpose other than to simplify the FileCheck-based tests.
    for (const auto *I : D->property_impls()) {
      if (I->getLocation().isInvalid())
        IndexCtx.indexDecl(I);
    }
    for (const auto *I : D->decls()) {
      if (!isa<ObjCPropertyImplDecl>(I) ||
          cast<ObjCPropertyImplDecl>(I)->getLocation().isValid())
        IndexCtx.indexDecl(I);
    }

    return true;
  }

  bool VisitObjCCategoryDecl(const ObjCCategoryDecl *D) {
    if (!IndexCtx.shouldIndex(D))
      return true;
    const ObjCInterfaceDecl *C = D->getClassInterface();
    if (!C)
      return true;
    TRY_TO(IndexCtx.handleReference(C, D->getLocation(), D, D, SymbolRoleSet(),
                                   SymbolRelation{
                                     (unsigned)SymbolRole::RelationExtendedBy, D
                                   }));
    SourceLocation CategoryLoc = D->getCategoryNameLoc();
    if (!CategoryLoc.isValid())
      CategoryLoc = D->getLocation();
    TRY_TO(IndexCtx.handleDecl(D, CategoryLoc));
    TRY_TO(handleReferencedProtocols(D->getReferencedProtocols(), D,
                                     /*superLoc=*/SourceLocation()));
    TRY_TO(IndexCtx.indexDeclContext(D));
    return true;
  }

  bool VisitObjCCategoryImplDecl(const ObjCCategoryImplDecl *D) {
    const ObjCCategoryDecl *Cat = D->getCategoryDecl();
    if (!Cat)
      return true;
    const ObjCInterfaceDecl *C = D->getClassInterface();
    if (C)
      TRY_TO(IndexCtx.handleReference(C, D->getLocation(), D, D,
                                      SymbolRoleSet()));
    SourceLocation CategoryLoc = D->getCategoryNameLoc();
    if (!CategoryLoc.isValid())
      CategoryLoc = D->getLocation();
    TRY_DECL(D, IndexCtx.handleDecl(D, CategoryLoc));
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
        handleObjCMethod(MD, D);
    if (ObjCMethodDecl *MD = D->getSetterMethodDecl())
      if (MD->getLexicalDeclContext() == D->getLexicalDeclContext())
        handleObjCMethod(MD, D);
    TRY_DECL(D, IndexCtx.handleDecl(D));
    if (IBOutletCollectionAttr *attr = D->getAttr<IBOutletCollectionAttr>())
      IndexCtx.indexTypeSourceInfo(attr->getInterfaceLoc(), D,
                                   D->getLexicalDeclContext(), false, true);
    IndexCtx.indexTypeSourceInfo(D->getTypeSourceInfo(), D);
    return true;
  }

  bool VisitObjCPropertyImplDecl(const ObjCPropertyImplDecl *D) {
    ObjCPropertyDecl *PD = D->getPropertyDecl();
    auto *Container = cast<ObjCImplDecl>(D->getDeclContext());
    SourceLocation Loc = D->getLocation();
    SymbolRoleSet Roles = 0;
    SmallVector<SymbolRelation, 1> Relations;

    if (ObjCIvarDecl *ID = D->getPropertyIvarDecl())
      Relations.push_back({(SymbolRoleSet)SymbolRole::RelationAccessorOf, ID});
    if (Loc.isInvalid()) {
      Loc = Container->getLocation();
      Roles |= (SymbolRoleSet)SymbolRole::Implicit;
    }
    TRY_DECL(D, IndexCtx.handleDecl(D, Loc, Roles, Relations));

    if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic)
      return true;

    assert(D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize);
    if (ObjCMethodDecl *MD = PD->getGetterMethodDecl()) {
      if (MD->isPropertyAccessor() &&
          !hasUserDefined(MD, Container))
        IndexCtx.handleDecl(MD, Loc, SymbolRoleSet(SymbolRole::Implicit), {},
                            Container);
    }
    if (ObjCMethodDecl *MD = PD->getSetterMethodDecl()) {
      if (MD->isPropertyAccessor() &&
          !hasUserDefined(MD, Container))
        IndexCtx.handleDecl(MD, Loc, SymbolRoleSet(SymbolRole::Implicit), {},
                            Container);
    }
    if (ObjCIvarDecl *IvarD = D->getPropertyIvarDecl()) {
      if (IvarD->getSynthesize()) {
        // For synthesized ivars, use the location of its name in the
        // corresponding @synthesize. If there isn't one, use the containing
        // @implementation's location, rather than the property's location,
        // otherwise the header file containing the @interface will have different
        // indexing contents based on whether the @implementation was present or
        // not in the translation unit.
        SymbolRoleSet IvarRoles = 0;
        SourceLocation IvarLoc = D->getPropertyIvarDeclLoc();
        if (D->getLocation().isInvalid()) {
          IvarLoc = Container->getLocation();
          IvarRoles = (SymbolRoleSet)SymbolRole::Implicit;
        } else if (D->getLocation() == IvarLoc) {
          IvarRoles = (SymbolRoleSet)SymbolRole::Implicit;
        }
        TRY_DECL(IvarD, IndexCtx.handleDecl(IvarD, IvarLoc, IvarRoles));
      } else {
        IndexCtx.handleReference(IvarD, D->getPropertyIvarDeclLoc(), nullptr,
                                 D->getDeclContext(), SymbolRoleSet());
      }
    }
    return true;
  }

  bool VisitNamespaceDecl(const NamespaceDecl *D) {
    TRY_DECL(D, IndexCtx.handleDecl(D));
    IndexCtx.indexDeclContext(D);
    return true;
  }

  bool VisitUsingDecl(const UsingDecl *D) {
    const DeclContext *DC = D->getDeclContext()->getRedeclContext();
    const NamedDecl *Parent = dyn_cast<NamedDecl>(DC);

    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), Parent,
                                         D->getLexicalDeclContext());
    for (const auto *I : D->shadows())
      IndexCtx.handleReference(I->getUnderlyingDecl(), D->getLocation(), Parent,
                               D->getLexicalDeclContext(), SymbolRoleSet());
    return true;
  }

  bool VisitUsingDirectiveDecl(const UsingDirectiveDecl *D) {
    const DeclContext *DC = D->getDeclContext()->getRedeclContext();
    const NamedDecl *Parent = dyn_cast<NamedDecl>(DC);

    IndexCtx.indexNestedNameSpecifierLoc(D->getQualifierLoc(), Parent,
                                         D->getLexicalDeclContext());
    return IndexCtx.handleReference(D->getNominatedNamespaceAsWritten(),
                                    D->getLocation(), Parent,
                                    D->getLexicalDeclContext(),
                                    SymbolRoleSet());
  }

  bool VisitClassTemplateSpecializationDecl(const
                                           ClassTemplateSpecializationDecl *D) {
    // FIXME: Notify subsequent callbacks if info comes from implicit
    // instantiation.
    if (D->isThisDeclarationADefinition()) {
      llvm::PointerUnion<ClassTemplateDecl *,
                         ClassTemplatePartialSpecializationDecl *>
          Template = D->getSpecializedTemplateOrPartial();
      const Decl *SpecializationOf =
          Template.is<ClassTemplateDecl *>()
              ? (Decl *)Template.get<ClassTemplateDecl *>()
              : Template.get<ClassTemplatePartialSpecializationDecl *>();
      IndexCtx.indexTagDecl(
          D, SymbolRelation(SymbolRoleSet(SymbolRole::RelationSpecializationOf),
                            SpecializationOf));
    }
    return true;
  }

  bool VisitTemplateDecl(const TemplateDecl *D) {
    // FIXME: Template parameters.
    return Visit(D->getTemplatedDecl());
  }

  bool VisitFriendDecl(const FriendDecl *D) {
    if (auto ND = D->getFriendDecl()) {
      // FIXME: Ignore a class template in a dependent context, these are not
      // linked properly with their redeclarations, ending up with duplicate
      // USRs.
      // See comment "Friend templates are visible in fairly strange ways." in
      // SemaTemplate.cpp which precedes code that prevents the friend template
      // from becoming visible from the enclosing context.
      if (isa<ClassTemplateDecl>(ND) && D->getDeclContext()->isDependentContext())
        return true;
      return Visit(ND);
    }
    if (auto Ty = D->getFriendType()) {
      IndexCtx.indexTypeSourceInfo(Ty, cast<NamedDecl>(D->getDeclContext()));
    }
    return true;
  }

  bool VisitImportDecl(const ImportDecl *D) {
    return IndexCtx.importedModule(D);
  }
};

} // anonymous namespace

bool IndexingContext::indexDecl(const Decl *D) {
  if (D->isImplicit() && shouldIgnoreIfImplicit(D))
    return true;

  if (isTemplateImplicitInstantiation(D))
    return true;

  IndexingDeclVisitor Visitor(*this);
  bool ShouldContinue = Visitor.Visit(D);
  if (!ShouldContinue)
    return false;

  if (!Visitor.Handled && isa<DeclContext>(D))
    return indexDeclContext(cast<DeclContext>(D));

  return true;
}

bool IndexingContext::indexDeclContext(const DeclContext *DC) {
  for (const auto *I : DC->decls())
    if (!indexDecl(I))
      return false;
  return true;
}

bool IndexingContext::indexTopLevelDecl(const Decl *D) {
  if (D->getLocation().isInvalid())
    return true;

  if (isa<ObjCMethodDecl>(D))
    return true; // Wait for the objc container.

  return indexDecl(D);
}

bool IndexingContext::indexDeclGroupRef(DeclGroupRef DG) {
  for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I)
    if (!indexTopLevelDecl(*I))
      return false;
  return true;
}
