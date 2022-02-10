//===- SymbolGraph/API.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines SymbolGraph API records.
///
//===----------------------------------------------------------------------===//

#include "clang/SymbolGraph/API.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentLexer.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/Support/Allocator.h"

namespace clang {
namespace symbolgraph {

APIRecord::~APIRecord() {}

GlobalRecord *
API::addGlobal(GVKind Kind, StringRef Name, StringRef USR, PresumedLoc Loc,
               const AvailabilityInfo &Availability, LinkageInfo Linkage,
               const DocComment &Comment, DeclarationFragments Fragments,
               DeclarationFragments SubHeading, FunctionSignature Signature) {
  auto Result = Globals.insert({Name, nullptr});
  if (Result.second) {
    GlobalRecord *Record = new (Allocator)
        GlobalRecord{Kind,    Name,    USR,       Loc,        Availability,
                     Linkage, Comment, Fragments, SubHeading, Signature};
    Result.first->second = Record;
  }
  return Result.first->second;
}

GlobalRecord *API::addGlobalVar(StringRef Name, StringRef USR, PresumedLoc Loc,
                                const AvailabilityInfo &Availability,
                                LinkageInfo Linkage, const DocComment &Comment,
                                DeclarationFragments Fragments,
                                DeclarationFragments SubHeading) {
  return addGlobal(GVKind::Variable, Name, USR, Loc, Availability, Linkage,
                   Comment, Fragments, SubHeading, {});
}

GlobalRecord *API::addFunction(StringRef Name, StringRef USR, PresumedLoc Loc,
                               const AvailabilityInfo &Availability,
                               LinkageInfo Linkage, const DocComment &Comment,
                               DeclarationFragments Fragments,
                               DeclarationFragments SubHeading,
                               FunctionSignature Signature) {
  return addGlobal(GVKind::Function, Name, USR, Loc, Availability, Linkage,
                   Comment, Fragments, SubHeading, Signature);
}

StringRef API::recordUSR(const Decl *D) {
  SmallString<128> USR;
  index::generateUSRForDecl(D, USR);
  return copyString(USR);
}

StringRef API::copyString(StringRef String, llvm::BumpPtrAllocator &Allocator) {
  if (String.empty())
    return {};

  if (Allocator.identifyObject(String.data()))
    return String;

  void *Ptr = Allocator.Allocate(String.size(), 1);
  memcpy(Ptr, String.data(), String.size());
  return StringRef(reinterpret_cast<const char *>(Ptr), String.size());
}

StringRef API::copyString(StringRef String) {
  return copyString(String, Allocator);
}

} // namespace symbolgraph
} // namespace clang
