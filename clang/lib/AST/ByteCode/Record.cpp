//===--- Record.cpp - struct and class metadata for the VM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Record.h"
#include "clang/AST/ASTContext.h"

using namespace clang;
using namespace clang::interp;

Record::Record(const RecordDecl *Decl, BaseList &&SrcBases,
               FieldList &&SrcFields, VirtualBaseList &&SrcVirtualBases,
               unsigned VirtualSize, unsigned BaseSize, bool HasPtrField)
    : Decl(Decl), Bases(std::move(SrcBases)), Fields(std::move(SrcFields)),
      BaseSize(BaseSize), VirtualSize(VirtualSize), IsUnion(Decl->isUnion()),
      IsAnonymousUnion(IsUnion && Decl->isAnonymousStructOrUnion()),
      HasPtrField(HasPtrField) {
  for (Base &V : SrcVirtualBases)
    VirtualBases.emplace_back(V.Decl, V.Desc, V.R, V.Offset + BaseSize);

  for (Base &B : Bases) {
    BaseMap[B.Decl] = &B;
    if (!HasPtrField)
      HasPtrField |= B.R->hasPtrField();
  }
  for (Base &V : VirtualBases) {
    VirtualBaseMap[V.Decl] = &V;
    if (!HasPtrField)
      HasPtrField |= V.R->hasPtrField();
  }
}

std::string Record::getName() const {
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  Decl->getNameForDiagnostic(OS, Decl->getASTContext().getPrintingPolicy(),
                             /*Qualified=*/true);
  return Ret;
}

bool Record::hasTrivialDtor() const {
  if (isAnonymousUnion())
    return true;
  const CXXDestructorDecl *Dtor = getDestructor();
  return !Dtor || Dtor->isTrivial();
}

const Record::Base *Record::getBase(const RecordDecl *FD) const {
  auto It = BaseMap.find(FD);
  assert(It != BaseMap.end() && "Missing base");
  return It->second;
}

const Record::Base *Record::getBase(QualType T) const {
  if (auto *RD = T->getAsCXXRecordDecl())
    return BaseMap.lookup(RD);
  return nullptr;
}

const Record::Base *Record::getVirtualBase(const RecordDecl *FD) const {
  auto It = VirtualBaseMap.find(FD);
  assert(It != VirtualBaseMap.end() && "Missing virtual base");
  return It->second;
}
