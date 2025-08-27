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

Record::Record(const RecordDecl *Decl, const Base *SrcBases, unsigned NumBases,
               const Field *Fields, unsigned NumFields, Base *VBases,
               unsigned NumVBases, unsigned VirtualSize, unsigned BaseSize)
    : Decl(Decl), Bases(SrcBases), NumBases(NumBases), Fields(Fields),
      NumFields(NumFields), VBases(VBases), NumVBases(NumVBases),
      BaseSize(BaseSize), VirtualSize(VirtualSize), IsUnion(Decl->isUnion()),
      IsAnonymousUnion(IsUnion && Decl->isAnonymousStructOrUnion()) {

  for (unsigned I = 0; I != NumVBases; ++I) {
    VBases[I].Offset += BaseSize;
  }

  for (const Base &B : bases())
    BaseMap[B.Decl] = &B;
  for (const Field &F : fields())
    FieldMap[F.Decl] = &F;
  for (const Base &V : virtual_bases())
    VirtualBaseMap[V.Decl] = &V;
}

std::string Record::getName() const {
  std::string Ret;
  llvm::raw_string_ostream OS(Ret);
  Decl->getNameForDiagnostic(OS, Decl->getASTContext().getPrintingPolicy(),
                             /*Qualified=*/true);
  return Ret;
}

const Record::Field *Record::getField(const FieldDecl *FD) const {
  auto It = FieldMap.find(FD->getFirstDecl());
  assert(It != FieldMap.end() && "Missing field");
  return It->second;
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
