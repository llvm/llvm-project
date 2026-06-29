//===- OperatorNewDeletePointers.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares data structures for analysis that identifies pointer entities in
// operator new/delete overloads that must have a 'void*' type
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_OPERATORNEWDELETE_OPERATORNEWDELETEPOINTERS_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_OPERATORNEWDELETE_OPERATORNEWDELETEPOINTERS_H

#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/EntitySummary.h"
#include "llvm/ADT/StringRef.h"
#include <set>

namespace clang::ssaf {

/// OperatorNewDeletePointersEntitySummary collects the following entities in a
/// contributor:
///  - return entities of operator new overloads;
///  - the parameter (optionally the 2nd)  of operator new overloads
///    representing the pointer to a memory area to initialize the object at;
///  - the first parameter of operator delete overloads representing the pointer
///    to a memory block to deallocate or a null pointer;
///  - the parameter (optionally the 2nd)  of operator delete overloads
///    representing the pointer used as the placement parameter in the matching
///    placement new.
struct OperatorNewDeletePointersEntitySummary final : public EntitySummary {
  static constexpr llvm::StringLiteral Name = "OperatorNewDeletePointers";

  static SummaryName summaryName() { return SummaryName(Name.str()); }

  SummaryName getSummaryName() const override { return summaryName(); }

  std::set<EntityId> Entities;

  bool operator==(const OperatorNewDeletePointersEntitySummary &Other) const {
    return Entities == Other.Entities;
  }

  bool operator==(const std::set<EntityId> &OtherEntities) const {
    return Entities == OtherEntities;
  }

  bool empty() const { return Entities.empty(); }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_OPERATORNEWDELETE_OPERATORNEWDELETEPOINTERS_H
