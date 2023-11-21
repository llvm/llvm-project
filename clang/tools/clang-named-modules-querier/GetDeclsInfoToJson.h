//===- GetDeclsInfoToJson.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_TOOLS_CLANG_NAMED_MODULES_QUERIER_GET_DECLS_INFO_TO_JSON_H
#define CLANG_TOOLS_CLANG_NAMED_MODULES_QUERIER_GET_DECLS_INFO_TO_JSON_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ODRHash.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/JSON.h"

namespace clang {
inline unsigned getHashValue(const NamedDecl *ND) {
  ODRHash Hasher;

  if (auto *FD = dyn_cast<FunctionDecl>(ND))
    return FD->getODRHash();
  else if (auto *ED = dyn_cast<EnumDecl>(ND))
    return const_cast<EnumDecl*>(ED)->getODRHash();
  else if (auto *CRD = dyn_cast<CXXRecordDecl>(ND); CRD && CRD->hasDefinition())
    return CRD->getODRHash();
  else {
    Hasher.AddDecl(ND);
    Hasher.AddSubDecl(ND);
  }

  return Hasher.CalculateHash();
}

inline llvm::json::Object getDeclInJson(const NamedDecl *ND, SourceManager &SMgr) {
  llvm::json::Object DeclObject;
  DeclObject.try_emplace("kind", ND->getDeclKindName());
  FullSourceLoc FSL(ND->getLocation(), SMgr);
  const FileEntry *FE = SMgr.getFileEntryForID(FSL.getFileID());
  DeclObject.try_emplace("source File Name", FE ? FE->getName() : "Unknown Source File");
  DeclObject.try_emplace("line", FSL.getSpellingLineNumber());
  DeclObject.try_emplace("col", FSL.getSpellingColumnNumber());
  DeclObject.try_emplace("Hash", getHashValue(ND));
  return llvm::json::Object({{ND->getQualifiedNameAsString(), std::move(DeclObject)}});
}
}

#endif
