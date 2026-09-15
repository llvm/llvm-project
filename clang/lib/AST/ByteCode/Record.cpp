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

Record::Record(const RecordDecl *Decl, ArrayRef<Base> Bases,
               ArrayRef<Field> Fields, ArrayRef<Base> VirtualBases,
               unsigned VirtualSize, unsigned BaseSize, bool HasPtrField)
    : Decl(Decl), Bases(Bases), Fields(Fields), VirtualBases(VirtualBases),
      BaseSize(BaseSize), VirtualSize(VirtualSize), IsUnion(Decl->isUnion()),
      IsAnonymousUnion(IsUnion && Decl->isAnonymousStructOrUnion()),
      HasPtrField(HasPtrField) {
  for (const Base &B : this->Bases)
    BaseMap[B.Decl] = &B;
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

const Record::Field *Record::findField(unsigned Offset) const {
  if (auto It = llvm::find_if(
          Fields,
          [=](const Record::Field &F) -> bool { return F.Offset == Offset; });
      It != Fields.end())
    return &*It;
  return nullptr;
}

const Record::Base *Record::getBase(const RecordDecl *RD) const {
  auto It = BaseMap.find(RD);
  assert(It != BaseMap.end() && "Missing base");
  return It->second;
}

const Record::Base *Record::getBaseOrNull(const RecordDecl *RD) const {
  return BaseMap.lookup(RD);
}

const Record::Base *Record::getBase(QualType T) const {
  if (auto *RD = T->getAsCXXRecordDecl())
    return BaseMap.lookup(RD);
  return nullptr;
}

const Record::Base *Record::findBase(unsigned Offset) const {
  if (auto It = llvm::find_if(
          Bases,
          [=](const Record::Base &B) -> bool { return B.Offset == Offset; });
      It != Bases.end())
    return &*It;
  return nullptr;
}

const Record::Base *Record::findVirtualBase(const RecordDecl *FD) const {
  if (auto *It = llvm::find_if(
          VirtualBases,
          [=](const Record::Base &B) -> bool { return B.Decl == FD; });
      It != Bases.end())
    return &*It;
  return nullptr;
}
