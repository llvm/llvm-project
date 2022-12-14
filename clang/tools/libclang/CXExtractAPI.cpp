//===- CXExtractAPI.cpp - libclang APIs for manipulating CXAPISet ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all libclang APIs related to manipulation CXAPISet
//
//===----------------------------------------------------------------------===//

#include "CXCursor.h"
#include "CXString.h"
#include "CXTranslationUnit.h"
#include "clang-c/CXErrorCode.h"
#include "clang-c/Documentation.h"
#include "clang-c/Index.h"
#include "clang-c/Platform.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/ExtractAPI/API.h"
#include "clang/ExtractAPI/ExtractAPIVisitor.h"
#include "clang/ExtractAPI/Serialization/SymbolGraphSerializer.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::extractapi;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(APISet, CXAPISet)

static void WalkupFromMostDerivedType(ExtractAPIVisitor &Visitor, Decl *D);

template <typename DeclTy>
static bool WalkupParentContext(DeclContext *Parent,
                                ExtractAPIVisitor &Visitor) {
  if (auto *D = dyn_cast<DeclTy>(Parent)) {
    WalkupFromMostDerivedType(Visitor, D);
    return true;
  }
  return false;
}

static void WalkupFromMostDerivedType(ExtractAPIVisitor &Visitor, Decl *D) {
  switch (D->getKind()) {
#define ABSTRACT_DECL(DECL)
#define DECL(CLASS, BASE)                                                      \
  case Decl::CLASS:                                                            \
    Visitor.WalkUpFrom##CLASS##Decl(static_cast<CLASS##Decl *>(D));            \
    break;
#include "clang/AST/DeclNodes.inc"
  }

  for (auto *Parent = D->getDeclContext(); Parent != nullptr;
       Parent = Parent->getParent()) {
    if (WalkupParentContext<ObjCContainerDecl>(Parent, Visitor))
      return;
    if (WalkupParentContext<TagDecl>(Parent, Visitor))
      return;
  }
}

static CXString GenerateCXStringFromSymbolGraphData(llvm::json::Object Obj) {
  llvm::SmallString<0> BackingString;
  llvm::raw_svector_ostream OS(BackingString);
  OS << Value(std::move(Obj));
  return cxstring::createDup(BackingString.str());
}

enum CXErrorCode clang_createAPISet(CXTranslationUnit tu, CXAPISet *out_api) {
  if (cxtu::isNotUsableTU(tu) || !out_api)
    return CXError_InvalidArguments;

  ASTUnit *Unit = cxtu::getASTUnit(tu);

  auto &Ctx = Unit->getASTContext();
  auto Lang = Unit->getInputKind().getLanguage();
  APISet *API = new APISet(Ctx.getTargetInfo().getTriple(), Lang,
                           Unit->getMainFileName().str());
  ExtractAPIVisitor Visitor(
      Ctx, [](SourceLocation Loc) { return true; }, *API);

  for (auto It = Unit->top_level_begin(); It != Unit->top_level_end(); ++It) {
    Visitor.TraverseDecl(*It);
  }

  *out_api = wrap(API);
  return CXError_Success;
}

void clang_disposeAPISet(CXAPISet api) { delete unwrap(api); }

CXString clang_getSymbolGraphForUSR(const char *usr, CXAPISet api) {
  auto *API = unwrap(api);

  if (auto SGF = SymbolGraphSerializer::serializeSingleSymbolSGF(usr, *API))
    return GenerateCXStringFromSymbolGraphData(std::move(*SGF));

  return cxstring::createNull();
}

CXString clang_getSymbolGraphForCursor(CXCursor cursor) {
  CXCursorKind Kind = clang_getCursorKind(cursor);
  if (clang_isDeclaration(Kind)) {
    const Decl *D = cxcursor::getCursorDecl(cursor);

    if (!D)
      return cxstring::createNull();

    CXTranslationUnit TU = cxcursor::getCursorTU(cursor);
    if (!TU)
      return cxstring::createNull();

    ASTUnit *Unit = cxtu::getASTUnit(TU);

    auto &Ctx = Unit->getASTContext();
    auto Lang = Unit->getInputKind().getLanguage();
    APISet API(Ctx.getTargetInfo().getTriple(), Lang,
               Unit->getMainFileName().str());
    ExtractAPIVisitor Visitor(
        Ctx, [](SourceLocation Loc) { return true; }, API);

    SmallString<128> USR;
    if (index::generateUSRForDecl(D, USR))
      return cxstring::createNull();

    WalkupFromMostDerivedType(Visitor, const_cast<Decl *>(D));
    auto *Record = API.findRecordForUSR(USR);

    if (!Record)
      return cxstring::createNull();

    for (const auto &Fragment : Record->Declaration.getFragments()) {
      if (Fragment.Declaration)
        WalkupFromMostDerivedType(Visitor,
                                  const_cast<Decl *>(Fragment.Declaration));
    }

    if (auto SGF = SymbolGraphSerializer::serializeSingleSymbolSGF(USR, API))
      return GenerateCXStringFromSymbolGraphData(std::move(*SGF));
  }

  return cxstring::createNull();
}
