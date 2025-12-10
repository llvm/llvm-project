//===-- RecordOps.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Operations on records (structs, classes, and unions).
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/RecordOps.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/FlowSensitive/ASTOps.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"

#define DEBUG_TYPE "dataflow"

namespace clang::dataflow {

static void copyField(const ValueDecl &Field, StorageLocation *SrcFieldLoc,
                      StorageLocation *DstFieldLoc, RecordStorageLocation &Dst,
                      Environment &Env) {
  assert(Field.getType()->isReferenceType() ||
         (SrcFieldLoc != nullptr && DstFieldLoc != nullptr));

  if (Field.getType()->isRecordType()) {
    copyRecord(cast<RecordStorageLocation>(*SrcFieldLoc),
               cast<RecordStorageLocation>(*DstFieldLoc), Env);
  } else if (Field.getType()->isReferenceType()) {
    Dst.setChild(Field, SrcFieldLoc);
  } else {
    if (Value *Val = Env.getValue(*SrcFieldLoc))
      Env.setValue(*DstFieldLoc, *Val);
    else
      Env.clearValue(*DstFieldLoc);
  }
}

static void copySyntheticField(QualType FieldType, StorageLocation &SrcFieldLoc,
                               StorageLocation &DstFieldLoc, Environment &Env) {
  if (FieldType->isRecordType()) {
    copyRecord(cast<RecordStorageLocation>(SrcFieldLoc),
               cast<RecordStorageLocation>(DstFieldLoc), Env);
  } else {
    if (Value *Val = Env.getValue(SrcFieldLoc))
      Env.setValue(DstFieldLoc, *Val);
    else
      Env.clearValue(DstFieldLoc);
  }
}

void copyRecord(RecordStorageLocation &Src, RecordStorageLocation &Dst,
                Environment &Env, const QualType TypeToCopy) {
  auto SrcType = Src.getType().getCanonicalType().getUnqualifiedType();
  auto DstType = Dst.getType().getCanonicalType().getUnqualifiedType();

  auto SrcDecl = SrcType->getAsCXXRecordDecl();
  auto DstDecl = DstType->getAsCXXRecordDecl();

  const CXXRecordDecl *DeclToCopy =
      TypeToCopy.isNull() ? nullptr : TypeToCopy->getAsCXXRecordDecl();

  [[maybe_unused]] bool CompatibleTypes =
      SrcType == DstType ||
      (SrcDecl != nullptr && DstDecl != nullptr &&
       (SrcDecl->isDerivedFrom(DstDecl) || DstDecl->isDerivedFrom(SrcDecl) ||
        (DeclToCopy != nullptr && SrcDecl->isDerivedFrom(DeclToCopy) &&
         DstDecl->isDerivedFrom(DeclToCopy))));

  LLVM_DEBUG({
    if (!CompatibleTypes) {
      llvm::dbgs() << "Source type " << Src.getType() << "\n";
      llvm::dbgs() << "Destination type " << Dst.getType() << "\n";
    }
  });
  assert(CompatibleTypes);

  if (SrcType == DstType || (SrcDecl != nullptr && DstDecl != nullptr &&
                             SrcDecl->isDerivedFrom(DstDecl))) {
    // Dst may have children modeled from other derived types than SrcType, e.g.
    // after casts of Dst to other types derived from DstType. Only copy the
    // children and synthetic fields present in both Dst and SrcType.
    const FieldSet FieldsInSrcType =
        Env.getDataflowAnalysisContext().getModeledFields(SrcType);
    for (auto [Field, DstFieldLoc] : Dst.children())
      if (const auto *FieldAsFieldDecl = dyn_cast<FieldDecl>(Field);
          FieldAsFieldDecl && FieldsInSrcType.contains(FieldAsFieldDecl))
        copyField(*Field, Src.getChild(*Field), DstFieldLoc, Dst, Env);
    const llvm::StringMap<QualType> SyntheticFieldsForSrcType =
        Env.getDataflowAnalysisContext().getSyntheticFields(SrcType);
    for (const auto &[Name, DstFieldLoc] : Dst.synthetic_fields())
      if (SyntheticFieldsForSrcType.contains(Name))
        copySyntheticField(DstFieldLoc->getType(), Src.getSyntheticField(Name),
                           *DstFieldLoc, Env);
  } else if (SrcDecl != nullptr && DstDecl != nullptr &&
             DstDecl->isDerivedFrom(SrcDecl)) {
    // Src may have children modeled from other derived types than DstType, e.g.
    // after other casts of Src to those types (likely in different branches,
    // but without flow-condition-dependent field modeling). Only copy the
    // children and synthetic fields of Src that are present in DstType.
    const FieldSet FieldsInDstType =
        Env.getDataflowAnalysisContext().getModeledFields(DstType);
    for (auto [Field, SrcFieldLoc] : Src.children()) {
      if (const auto *FieldAsFieldDecl = dyn_cast<FieldDecl>(Field);
          FieldAsFieldDecl && FieldsInDstType.contains(FieldAsFieldDecl))
        copyField(*Field, SrcFieldLoc, Dst.getChild(*Field), Dst, Env);
    }
    const llvm::StringMap<QualType> SyntheticFieldsForDstType =
        Env.getDataflowAnalysisContext().getSyntheticFields(DstType);
    for (const auto &[Name, SrcFieldLoc] : Src.synthetic_fields()) {
      if (SyntheticFieldsForDstType.contains(Name))
        copySyntheticField(SrcFieldLoc->getType(), *SrcFieldLoc,
                           Dst.getSyntheticField(Name), Env);
    }
  } else {
    for (const FieldDecl *Field :
         Env.getDataflowAnalysisContext().getModeledFields(TypeToCopy)) {
      copyField(*Field, Src.getChild(*Field), Dst.getChild(*Field), Dst, Env);
    }
    for (const auto &[SyntheticFieldName, SyntheticFieldType] :
         Env.getDataflowAnalysisContext().getSyntheticFields(TypeToCopy)) {
      copySyntheticField(SyntheticFieldType,
                         Src.getSyntheticField(SyntheticFieldName),
                         Dst.getSyntheticField(SyntheticFieldName), Env);
    }
  }
}

bool recordsEqual(const RecordStorageLocation &Loc1, const Environment &Env1,
                  const RecordStorageLocation &Loc2, const Environment &Env2) {
  LLVM_DEBUG({
    if (Loc2.getType().getCanonicalType().getUnqualifiedType() !=
        Loc1.getType().getCanonicalType().getUnqualifiedType()) {
      llvm::dbgs() << "Loc1 type " << Loc1.getType() << "\n";
      llvm::dbgs() << "Loc2 type " << Loc2.getType() << "\n";
    }
  });
  assert(Loc2.getType().getCanonicalType().getUnqualifiedType() ==
         Loc1.getType().getCanonicalType().getUnqualifiedType());

  for (auto [Field, FieldLoc1] : Loc1.children()) {
    StorageLocation *FieldLoc2 = Loc2.getChild(*Field);

    assert(Field->getType()->isReferenceType() ||
           (FieldLoc1 != nullptr && FieldLoc2 != nullptr));

    if (Field->getType()->isRecordType()) {
      if (!recordsEqual(cast<RecordStorageLocation>(*FieldLoc1), Env1,
                        cast<RecordStorageLocation>(*FieldLoc2), Env2))
        return false;
    } else if (Field->getType()->isReferenceType()) {
      if (FieldLoc1 != FieldLoc2)
        return false;
    } else if (Env1.getValue(*FieldLoc1) != Env2.getValue(*FieldLoc2)) {
      return false;
    }
  }

  for (const auto &[Name, SynthFieldLoc1] : Loc1.synthetic_fields()) {
    if (SynthFieldLoc1->getType()->isRecordType()) {
      if (!recordsEqual(
              *cast<RecordStorageLocation>(SynthFieldLoc1), Env1,
              cast<RecordStorageLocation>(Loc2.getSyntheticField(Name)), Env2))
        return false;
    } else if (Env1.getValue(*SynthFieldLoc1) !=
               Env2.getValue(Loc2.getSyntheticField(Name))) {
      return false;
    }
  }

  return true;
}

} // namespace clang::dataflow
