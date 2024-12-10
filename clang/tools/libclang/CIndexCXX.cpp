//===- CIndexCXX.cpp - Clang-C Source Indexing Library --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the libclang support for C++ cursors.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"
#include "CXCursor.h"
#include "CXType.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"

using namespace clang;
using namespace clang::cxcursor;

unsigned clang_isVirtualBase(CXCursor C) {
  if (C.kind != CXCursor_CXXBaseSpecifier)
    return 0;
  
  const CXXBaseSpecifier *B = getCursorCXXBaseSpecifier(C);
  return B->isVirtual();
}

unsigned clang_visitCXXBaseClasses(CXType PT, CXFieldVisitor visitor,
                                   CXClientData client_data) {
  CXCursor PC = clang_getTypeDeclaration(PT);
  if (clang_isInvalid(PC.kind))
    return false;
  const CXXRecordDecl *RD =
      dyn_cast_if_present<CXXRecordDecl>(cxcursor::getCursorDecl(PC));
  if (!RD || RD->isInvalidDecl())
    return false;
  RD = RD->getDefinition();
  if (!RD || RD->isInvalidDecl())
    return false;

  for (auto &Base : RD->bases()) {
    // Callback to the client.
    switch (
        visitor(cxcursor::MakeCursorCXXBaseSpecifier(&Base, getCursorTU(PC)),
                client_data)) {
    case CXVisit_Break:
      return true;
    case CXVisit_Continue:
      break;
    }
  }
  return true;
}

enum CX_CXXAccessSpecifier clang_getCXXAccessSpecifier(CXCursor C) {
  AccessSpecifier spec = AS_none;

  if (C.kind == CXCursor_CXXAccessSpecifier || clang_isDeclaration(C.kind))
    spec = getCursorDecl(C)->getAccess();
  else if (C.kind == CXCursor_CXXBaseSpecifier)
    spec = getCursorCXXBaseSpecifier(C)->getAccessSpecifier();
  else
    return CX_CXXInvalidAccessSpecifier;
  
  switch (spec) {
    case AS_public: return CX_CXXPublic;
    case AS_protected: return CX_CXXProtected;
    case AS_private: return CX_CXXPrivate;
    case AS_none: return CX_CXXInvalidAccessSpecifier;
  }

  llvm_unreachable("Invalid AccessSpecifier!");
}

enum CXCursorKind clang_getTemplateCursorKind(CXCursor C) {
  using namespace clang::cxcursor;
  
  switch (C.kind) {
  case CXCursor_ClassTemplate: 
  case CXCursor_FunctionTemplate:
    if (const TemplateDecl *Template
                           = dyn_cast_or_null<TemplateDecl>(getCursorDecl(C)))
      return MakeCXCursor(Template->getTemplatedDecl(), getCursorTU(C)).kind;
    break;
      
  case CXCursor_ClassTemplatePartialSpecialization:
    if (const ClassTemplateSpecializationDecl *PartialSpec
          = dyn_cast_or_null<ClassTemplatePartialSpecializationDecl>(
                                                            getCursorDecl(C))) {
      switch (PartialSpec->getTagKind()) {
      case TagTypeKind::Interface:
      case TagTypeKind::Struct:
        return CXCursor_StructDecl;
      case TagTypeKind::Class:
        return CXCursor_ClassDecl;
      case TagTypeKind::Union:
        return CXCursor_UnionDecl;
      case TagTypeKind::Enum:
        return CXCursor_NoDeclFound;
      }
    }
    break;
      
  default:
    break;
  }
  
  return CXCursor_NoDeclFound;
}

CXCursor clang_getSpecializedCursorTemplate(CXCursor C) {
  if (!clang_isDeclaration(C.kind))
    return clang_getNullCursor();
    
  const Decl *D = getCursorDecl(C);
  if (!D)
    return clang_getNullCursor();

  Decl *Template = nullptr;
  if (const CXXRecordDecl *CXXRecord = dyn_cast<CXXRecordDecl>(D)) {
    if (const ClassTemplatePartialSpecializationDecl *PartialSpec
          = dyn_cast<ClassTemplatePartialSpecializationDecl>(CXXRecord))
      Template = PartialSpec->getSpecializedTemplate();
    else if (const ClassTemplateSpecializationDecl *ClassSpec 
               = dyn_cast<ClassTemplateSpecializationDecl>(CXXRecord)) {
      llvm::PointerUnion<ClassTemplateDecl *,
                         ClassTemplatePartialSpecializationDecl *> Result
        = ClassSpec->getSpecializedTemplateOrPartial();
      if (isa<ClassTemplateDecl *>(Result))
        Template = cast<ClassTemplateDecl *>(Result);
      else
        Template = cast<ClassTemplatePartialSpecializationDecl *>(Result);

    } else 
      Template = CXXRecord->getInstantiatedFromMemberClass();
  } else if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
    Template = Function->getPrimaryTemplate();
    if (!Template)
      Template = Function->getInstantiatedFromMemberFunction();
  } else if (const VarDecl *Var = dyn_cast<VarDecl>(D)) {
    if (Var->isStaticDataMember())
      Template = Var->getInstantiatedFromStaticDataMember();
  } else if (const RedeclarableTemplateDecl *Tmpl
                                        = dyn_cast<RedeclarableTemplateDecl>(D))
    Template = Tmpl->getInstantiatedFromMemberTemplate();
  
  if (!Template)
    return clang_getNullCursor();
  
  return MakeCXCursor(Template, getCursorTU(C));
}
