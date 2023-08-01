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

#define DEBUG_TYPE "dataflow"

void clang::dataflow::copyRecord(RecordStorageLocation &Src,
                                 RecordStorageLocation &Dst, Environment &Env) {
  auto SrcType = Src.getType().getCanonicalType().getUnqualifiedType();
  auto DstType = Dst.getType().getCanonicalType().getUnqualifiedType();

  auto SrcDecl = SrcType->getAsCXXRecordDecl();
  auto DstDecl = DstType->getAsCXXRecordDecl();

  bool compatibleTypes =
      SrcType == DstType ||
      (SrcDecl && DstDecl && SrcDecl->isDerivedFrom(DstDecl));
  (void)compatibleTypes;

  LLVM_DEBUG({
    if (!compatibleTypes) {
      llvm::dbgs() << "Source type " << Src.getType() << "\n";
      llvm::dbgs() << "Destination type " << Dst.getType() << "\n";
    }
  });
  assert(compatibleTypes);

  for (auto [Field, DstFieldLoc] : Dst.children()) {
    StorageLocation *SrcFieldLoc = Src.getChild(*Field);

    assert(Field->getType()->isReferenceType() ||
           (SrcFieldLoc != nullptr && DstFieldLoc != nullptr));

    if (Field->getType()->isRecordType()) {
      copyRecord(cast<RecordStorageLocation>(*SrcFieldLoc),
                 cast<RecordStorageLocation>(*DstFieldLoc), Env);
    } else if (Field->getType()->isReferenceType()) {
      Dst.setChild(*Field, SrcFieldLoc);
    } else {
      if (Value *Val = Env.getValue(*SrcFieldLoc))
        Env.setValue(*DstFieldLoc, *Val);
      else
        Env.clearValue(*DstFieldLoc);
    }
  }

  RecordValue *SrcVal = cast_or_null<RecordValue>(Env.getValue(Src));
  RecordValue *DstVal = cast_or_null<RecordValue>(Env.getValue(Dst));

  DstVal = &Env.create<RecordValue>(Dst);
  Env.setValue(Dst, *DstVal);

  if (SrcVal == nullptr)
    return;

  for (const auto &[Name, Value] : SrcVal->properties()) {
    if (Value != nullptr)
      DstVal->setProperty(Name, *Value);
  }
}

bool clang::dataflow::recordsEqual(const RecordStorageLocation &Loc1,
                                   const Environment &Env1,
                                   const RecordStorageLocation &Loc2,
                                   const Environment &Env2) {
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

  llvm::StringMap<Value *> Props1, Props2;

  if (RecordValue *Val1 = cast_or_null<RecordValue>(Env1.getValue(Loc1)))
    for (const auto &[Name, Value] : Val1->properties())
      Props1[Name] = Value;
  if (RecordValue *Val2 = cast_or_null<RecordValue>(Env2.getValue(Loc2)))
    for (const auto &[Name, Value] : Val2->properties())
      Props2[Name] = Value;

  if (Props1.size() != Props2.size())
    return false;

  for (const auto &[Name, Value] : Props1) {
    auto It = Props2.find(Name);
    if (It == Props2.end())
      return false;
    if (Value != It->second)
      return false;
  }

  return true;
}
