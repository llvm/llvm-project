//===--- IndexSymbol.cpp - Types and functions for indexing symbols -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexSymbol.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"

using namespace clang;
using namespace clang::index;

SymbolInfo index::getSymbolInfo(const Decl *D) {
  assert(D);
  SymbolInfo Info;
  Info.Kind = SymbolKind::Unknown;
  Info.TemplateKind = SymbolCXXTemplateKind::NonTemplate;
  Info.Lang = SymbolLanguage::C;

  if (const TagDecl *TD = dyn_cast<TagDecl>(D)) {
    switch (TD->getTagKind()) {
    case TTK_Struct:
      Info.Kind = SymbolKind::Struct; break;
    case TTK_Union:
      Info.Kind = SymbolKind::Union; break;
    case TTK_Class:
      Info.Kind = SymbolKind::CXXClass;
      Info.Lang = SymbolLanguage::CXX;
      break;
    case TTK_Interface:
      Info.Kind = SymbolKind::CXXInterface;
      Info.Lang = SymbolLanguage::CXX;
      break;
    case TTK_Enum:
      Info.Kind = SymbolKind::Enum; break;
    }

    if (const CXXRecordDecl *CXXRec = dyn_cast<CXXRecordDecl>(D))
      if (!CXXRec->isCLike())
        Info.Lang = SymbolLanguage::CXX;

    if (isa<ClassTemplatePartialSpecializationDecl>(D)) {
      Info.TemplateKind = SymbolCXXTemplateKind::TemplatePartialSpecialization;
    } else if (isa<ClassTemplateSpecializationDecl>(D)) {
      Info.TemplateKind = SymbolCXXTemplateKind::TemplateSpecialization;
    }

  } else {
    switch (D->getKind()) {
    case Decl::Typedef:
      Info.Kind = SymbolKind::Typedef; break;
    case Decl::Function:
      Info.Kind = SymbolKind::Function;
      break;
    case Decl::ParmVar:
      Info.Kind = SymbolKind::Variable;
      break;
    case Decl::Var:
      Info.Kind = SymbolKind::Variable;
      if (isa<CXXRecordDecl>(D->getDeclContext())) {
        Info.Kind = SymbolKind::CXXStaticVariable;
        Info.Lang = SymbolLanguage::CXX;
      }
      break;
    case Decl::Field:
      Info.Kind = SymbolKind::Field;
      if (const CXXRecordDecl *
            CXXRec = dyn_cast<CXXRecordDecl>(D->getDeclContext())) {
        if (!CXXRec->isCLike())
          Info.Lang = SymbolLanguage::CXX;
      }
      break;
    case Decl::EnumConstant:
      Info.Kind = SymbolKind::EnumConstant; break;
    case Decl::ObjCInterface:
    case Decl::ObjCImplementation:
      Info.Kind = SymbolKind::ObjCClass;
      Info.Lang = SymbolLanguage::ObjC;
      break;
    case Decl::ObjCProtocol:
      Info.Kind = SymbolKind::ObjCProtocol;
      Info.Lang = SymbolLanguage::ObjC;
      break;
    case Decl::ObjCCategory:
    case Decl::ObjCCategoryImpl:
      Info.Kind = SymbolKind::ObjCCategory;
      Info.Lang = SymbolLanguage::ObjC;
      break;
    case Decl::ObjCMethod:
      if (cast<ObjCMethodDecl>(D)->isInstanceMethod())
        Info.Kind = SymbolKind::ObjCInstanceMethod;
      else
        Info.Kind = SymbolKind::ObjCClassMethod;
      Info.Lang = SymbolLanguage::ObjC;
      break;
    case Decl::ObjCProperty:
      Info.Kind = SymbolKind::ObjCProperty;
      Info.Lang = SymbolLanguage::ObjC;
      break;
    case Decl::ObjCIvar:
      Info.Kind = SymbolKind::ObjCIvar;
      Info.Lang = SymbolLanguage::ObjC;
      break;
    case Decl::Namespace:
      Info.Kind = SymbolKind::CXXNamespace;
      Info.Lang = SymbolLanguage::CXX;
      break;
    case Decl::NamespaceAlias:
      Info.Kind = SymbolKind::CXXNamespaceAlias;
      Info.Lang = SymbolLanguage::CXX;
      break;
    case Decl::CXXConstructor:
      Info.Kind = SymbolKind::CXXConstructor;
      Info.Lang = SymbolLanguage::CXX;
      break;
    case Decl::CXXDestructor:
      Info.Kind = SymbolKind::CXXDestructor;
      Info.Lang = SymbolLanguage::CXX;
      break;
    case Decl::CXXConversion:
      Info.Kind = SymbolKind::CXXConversionFunction;
      Info.Lang = SymbolLanguage::CXX;
      break;
    case Decl::CXXMethod: {
      const CXXMethodDecl *MD = cast<CXXMethodDecl>(D);
      if (MD->isStatic())
        Info.Kind = SymbolKind::CXXStaticMethod;
      else
        Info.Kind = SymbolKind::CXXInstanceMethod;
      Info.Lang = SymbolLanguage::CXX;
      break;
    }
    case Decl::ClassTemplate:
      Info.Kind = SymbolKind::CXXClass;
      Info.TemplateKind = SymbolCXXTemplateKind::Template;
      break;
    case Decl::FunctionTemplate:
      Info.Kind = SymbolKind::Function;
      Info.TemplateKind = SymbolCXXTemplateKind::Template;
      if (const CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(
                           cast<FunctionTemplateDecl>(D)->getTemplatedDecl())) {
        if (isa<CXXConstructorDecl>(MD))
          Info.Kind = SymbolKind::CXXConstructor;
        else if (isa<CXXDestructorDecl>(MD))
          Info.Kind = SymbolKind::CXXDestructor;
        else if (isa<CXXConversionDecl>(MD))
          Info.Kind = SymbolKind::CXXConversionFunction;
        else {
          if (MD->isStatic())
            Info.Kind = SymbolKind::CXXStaticMethod;
          else
            Info.Kind = SymbolKind::CXXInstanceMethod;
        }
      }
      break;
    case Decl::TypeAliasTemplate:
      Info.Kind = SymbolKind::CXXTypeAlias;
      Info.TemplateKind = SymbolCXXTemplateKind::Template;
      break;
    case Decl::TypeAlias:
      Info.Kind = SymbolKind::CXXTypeAlias;
      Info.Lang = SymbolLanguage::CXX;
      break;
    default:
      break;
    }
  }

  if (Info.Kind == SymbolKind::Unknown)
    return Info;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (FD->getTemplatedKind() ==
          FunctionDecl::TK_FunctionTemplateSpecialization)
      Info.TemplateKind = SymbolCXXTemplateKind::TemplateSpecialization;
  }

  if (Info.TemplateKind != SymbolCXXTemplateKind::NonTemplate)
    Info.Lang = SymbolLanguage::CXX;

  return Info;
}
