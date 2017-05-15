//===- IndexingContext.cpp - Indexing context data ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;
using namespace index;

static bool isGeneratedDecl(const Decl *D) {
  if (auto *attr = D->getAttr<ExternalSourceSymbolAttr>()) {
    return attr->getGeneratedDeclaration();
  }
  return false;
}

bool IndexingContext::shouldIndex(const Decl *D) {
  return !isGeneratedDecl(D);
}

const LangOptions &IndexingContext::getLangOpts() const {
  return Ctx->getLangOpts();
}

bool IndexingContext::shouldIndexFunctionLocalSymbols() const {
  return IndexOpts.IndexFunctionLocals;
}

bool IndexingContext::handleDecl(const Decl *D,
                                 SymbolRoleSet Roles,
                                 ArrayRef<SymbolRelation> Relations) {
  return handleDecl(D, D->getLocation(), Roles, Relations);
}

bool IndexingContext::handleDecl(const Decl *D, SourceLocation Loc,
                                 SymbolRoleSet Roles,
                                 ArrayRef<SymbolRelation> Relations,
                                 const DeclContext *DC) {
  if (!DC)
    DC = D->getDeclContext();

  const Decl *OrigD = D;
  if (isa<ObjCPropertyImplDecl>(D)) {
    D = cast<ObjCPropertyImplDecl>(D)->getPropertyDecl();
  }
  return handleDeclOccurrence(D, Loc, /*IsRef=*/false, cast<Decl>(DC),
                              Roles, Relations,
                              nullptr, OrigD, DC);
}

bool IndexingContext::handleReference(const NamedDecl *D, SourceLocation Loc,
                                      const NamedDecl *Parent,
                                      const DeclContext *DC,
                                      SymbolRoleSet Roles,
                                      ArrayRef<SymbolRelation> Relations,
                                      const Expr *RefE,
                                      const Decl *RefD) {
  if (!shouldIndexFunctionLocalSymbols() && isFunctionLocalSymbol(D))
    return true;

  if (isa<NonTypeTemplateParmDecl>(D) || isa<TemplateTypeParmDecl>(D))
    return true;
    
  return handleDeclOccurrence(D, Loc, /*IsRef=*/true, Parent, Roles, Relations,
                              RefE, RefD, DC);
}

bool IndexingContext::importedModule(const ImportDecl *ImportD) {
  SourceLocation Loc;
  auto IdLocs = ImportD->getIdentifierLocs();
  if (!IdLocs.empty())
    Loc = IdLocs.front();
  else
    Loc = ImportD->getLocation();
  SourceManager &SM = Ctx->getSourceManager();
  Loc = SM.getFileLoc(Loc);
  if (Loc.isInvalid())
    return true;

  FileID FID;
  unsigned Offset;
  std::tie(FID, Offset) = SM.getDecomposedLoc(Loc);
  if (FID.isInvalid())
    return true;

  bool Invalid = false;
  const SrcMgr::SLocEntry &SEntry = SM.getSLocEntry(FID, &Invalid);
  if (Invalid || !SEntry.isFile())
    return true;

  if (SEntry.getFile().getFileCharacteristic() != SrcMgr::C_User) {
    switch (IndexOpts.SystemSymbolFilter) {
    case IndexingOptions::SystemSymbolFilterKind::None:
      return true;
    case IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly:
    case IndexingOptions::SystemSymbolFilterKind::All:
      break;
    }
  }

  SymbolRoleSet Roles = (unsigned)SymbolRole::Declaration;
  if (ImportD->isImplicit())
    Roles |= (unsigned)SymbolRole::Implicit;

  return DataConsumer.handleModuleOccurence(ImportD, Roles, FID, Offset);
}

bool IndexingContext::isTemplateImplicitInstantiation(const Decl *D) {
  TemplateSpecializationKind TKind = TSK_Undeclared;
  if (const ClassTemplateSpecializationDecl *
      SD = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    TKind = SD->getSpecializationKind();
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    TKind = FD->getTemplateSpecializationKind();
  } else if (auto *VD = dyn_cast<VarDecl>(D)) {
    TKind = VD->getTemplateSpecializationKind();
  } else if (isa<FieldDecl>(D)) {
    if (const auto *Parent =
            dyn_cast<ClassTemplateSpecializationDecl>(D->getDeclContext()))
      TKind = Parent->getSpecializationKind();
  }
  switch (TKind) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      return false;
    case TSK_ImplicitInstantiation:
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      return true;
  }
  llvm_unreachable("invalid TemplateSpecializationKind");
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

static const Decl *adjustTemplateImplicitInstantiation(const Decl *D) {
  if (const ClassTemplateSpecializationDecl *
      SD = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    return SD->getTemplateInstantiationPattern();
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    return FD->getTemplateInstantiationPattern();
  } else if (auto *VD = dyn_cast<VarDecl>(D)) {
    return VD->getTemplateInstantiationPattern();
  } else if (const auto *FD = dyn_cast<FieldDecl>(D)) {
    if (const auto *Parent =
            dyn_cast<ClassTemplateSpecializationDecl>(D->getDeclContext())) {
      const CXXRecordDecl *Pattern = Parent->getTemplateInstantiationPattern();
      for (const NamedDecl *ND : Pattern->lookup(FD->getDeclName())) {
        if (ND->isImplicit())
          continue;
        if (isa<FieldDecl>(ND))
          return ND;
      }
    }
  }
  return nullptr;
}

static bool isDeclADefinition(const Decl *D, const DeclContext *ContainerDC, ASTContext &Ctx) {
  if (auto VD = dyn_cast<VarDecl>(D))
    return VD->isThisDeclarationADefinition(Ctx);

  if (auto FD = dyn_cast<FunctionDecl>(D))
    return FD->isThisDeclarationADefinition();

  if (auto TD = dyn_cast<TagDecl>(D))
    return TD->isThisDeclarationADefinition();

  if (auto MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->isThisDeclarationADefinition() || isa<ObjCImplDecl>(ContainerDC);

  if (isa<TypedefNameDecl>(D) ||
      isa<EnumConstantDecl>(D) ||
      isa<FieldDecl>(D) ||
      isa<MSPropertyDecl>(D) ||
      isa<ObjCImplDecl>(D) ||
      isa<ObjCPropertyImplDecl>(D))
    return true;

  return false;
}

static const Decl *adjustParent(const Decl *Parent) {
  if (!Parent)
    return nullptr;
  for (;; Parent = cast<Decl>(Parent->getDeclContext())) {
    if (isa<TranslationUnitDecl>(Parent))
      return nullptr;
    if (isa<LinkageSpecDecl>(Parent) || isa<BlockDecl>(Parent))
      continue;
    if (auto NS = dyn_cast<NamespaceDecl>(Parent)) {
      if (NS->isAnonymousNamespace())
        continue;
    } else if (auto RD = dyn_cast<RecordDecl>(Parent)) {
      if (RD->isAnonymousStructOrUnion())
        continue;
    } else if (auto FD = dyn_cast<FieldDecl>(Parent)) {
      if (FD->getDeclName().isEmpty())
        continue;
    }
    return Parent;
  }
}

static const Decl *getCanonicalDecl(const Decl *D) {
  D = D->getCanonicalDecl();
  if (auto TD = dyn_cast<TemplateDecl>(D)) {
    D = TD->getTemplatedDecl();
    assert(D->isCanonicalDecl());
  }

  return D;
}

static bool shouldReportOccurrenceForSystemDeclOnlyMode(
    bool IsRef, SymbolRoleSet Roles, ArrayRef<SymbolRelation> Relations) {
  if (!IsRef)
    return true;

  auto acceptForRelation = [](SymbolRoleSet roles) -> bool {
    bool accept = false;
    applyForEachSymbolRoleInterruptible(roles, [&accept](SymbolRole r) -> bool {
      switch (r) {
      case SymbolRole::RelationChildOf:
      case SymbolRole::RelationBaseOf:
      case SymbolRole::RelationOverrideOf:
      case SymbolRole::RelationExtendedBy:
      case SymbolRole::RelationAccessorOf:
      case SymbolRole::RelationIBTypeOf:
        accept = true;
        return false;
      case SymbolRole::Declaration:
      case SymbolRole::Definition:
      case SymbolRole::Reference:
      case SymbolRole::Read:
      case SymbolRole::Write:
      case SymbolRole::Call:
      case SymbolRole::Dynamic:
      case SymbolRole::AddressOf:
      case SymbolRole::Implicit:
      case SymbolRole::RelationReceivedBy:
      case SymbolRole::RelationCalledBy:
      case SymbolRole::RelationContainedBy:
      case SymbolRole::RelationSpecializationOf:
        return true;
      }
      llvm_unreachable("Unsupported SymbolRole value!");
    });
    return accept;
  };

  for (auto &Rel : Relations) {
    if (acceptForRelation(Rel.Roles))
      return true;
  }

  return false;
}

bool IndexingContext::handleDeclOccurrence(const Decl *D, SourceLocation Loc,
                                           bool IsRef, const Decl *Parent,
                                           SymbolRoleSet Roles,
                                           ArrayRef<SymbolRelation> Relations,
                                           const Expr *OrigE,
                                           const Decl *OrigD,
                                           const DeclContext *ContainerDC) {
  if (D->isImplicit() && !isa<ObjCMethodDecl>(D))
    return true;
  if (!isa<NamedDecl>(D) ||
      (cast<NamedDecl>(D)->getDeclName().isEmpty() &&
       !isa<TagDecl>(D) && !isa<ObjCCategoryDecl>(D)))
    return true;

  SourceManager &SM = Ctx->getSourceManager();
  Loc = SM.getFileLoc(Loc);
  if (Loc.isInvalid())
    return true;

  FileID FID;
  unsigned Offset;
  std::tie(FID, Offset) = SM.getDecomposedLoc(Loc);
  if (FID.isInvalid())
    return true;

  bool Invalid = false;
  const SrcMgr::SLocEntry &SEntry = SM.getSLocEntry(FID, &Invalid);
  if (Invalid || !SEntry.isFile())
    return true;

  if (SEntry.getFile().getFileCharacteristic() != SrcMgr::C_User) {
    switch (IndexOpts.SystemSymbolFilter) {
    case IndexingOptions::SystemSymbolFilterKind::None:
      return true;
    case IndexingOptions::SystemSymbolFilterKind::DeclarationsOnly:
      if (!shouldReportOccurrenceForSystemDeclOnlyMode(IsRef, Roles, Relations))
        return true;
      break;
    case IndexingOptions::SystemSymbolFilterKind::All:
      break;
    }
  }

  if (isTemplateImplicitInstantiation(D)) {
    if (!IsRef)
      return true;
    D = adjustTemplateImplicitInstantiation(D);
    if (!D)
      return true;
    assert(!isTemplateImplicitInstantiation(D));
  }

  if (!OrigD)
    OrigD = D;

  if (IsRef)
    Roles |= (unsigned)SymbolRole::Reference;
  else if (isDeclADefinition(OrigD, ContainerDC, *Ctx))
    Roles |= (unsigned)SymbolRole::Definition;
  else
    Roles |= (unsigned)SymbolRole::Declaration;

  D = getCanonicalDecl(D);
  Parent = adjustParent(Parent);
  if (Parent)
    Parent = getCanonicalDecl(Parent);

  SmallVector<SymbolRelation, 6> FinalRelations;
  FinalRelations.reserve(Relations.size()+1);

  auto addRelation = [&](SymbolRelation Rel) {
    auto It = std::find_if(FinalRelations.begin(), FinalRelations.end(),
                [&](SymbolRelation Elem)->bool {
                  return Elem.RelatedSymbol == Rel.RelatedSymbol;
                });
    if (It != FinalRelations.end()) {
      It->Roles |= Rel.Roles;
    } else {
      FinalRelations.push_back(Rel);
    }
    Roles |= Rel.Roles;
  };

  if (Parent) {
    if (IsRef || (!isa<ParmVarDecl>(D) && isFunctionLocalSymbol(D))) {
      addRelation(SymbolRelation{
        (unsigned)SymbolRole::RelationContainedBy,
        Parent
      });
    } else {
      addRelation(SymbolRelation{
        (unsigned)SymbolRole::RelationChildOf,
        Parent
      });
    }
  }

  for (auto &Rel : Relations) {
    addRelation(SymbolRelation(Rel.Roles,
                               Rel.RelatedSymbol->getCanonicalDecl()));
  }

  IndexDataConsumer::ASTNodeInfo Node{ OrigE, OrigD, Parent, ContainerDC };
  return DataConsumer.handleDeclOccurence(D, Roles, FinalRelations, FID, Offset,
                                          Node);
}
