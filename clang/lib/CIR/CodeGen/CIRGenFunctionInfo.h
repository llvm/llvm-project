//==-- CIRGenFunctionInfo.h - Representation of fn argument/return types ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines CIRGenFunctionInfo and associated types used in representing the
// CIR source types and ABI-coerced types for function arguments and
// return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CIRGENFUNCTIONINFO_H
#define LLVM_CLANG_CIR_CIRGENFUNCTIONINFO_H

#include "clang/AST/CanonicalType.h"
#include "clang/CIR/ABIArgInfo.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/TrailingObjects.h"

namespace clang::CIRGen {

struct CIRGenFunctionInfoArgInfo {
  CanQualType type;
  cir::ABIArgInfo info;
};

class CIRGenFunctionInfo final
    : public llvm::FoldingSetNode,
      private llvm::TrailingObjects<CIRGenFunctionInfo,
                                    CIRGenFunctionInfoArgInfo> {
  using ArgInfo = CIRGenFunctionInfoArgInfo;

  ArgInfo *getArgsBuffer() { return getTrailingObjects<ArgInfo>(); }
  const ArgInfo *getArgsBuffer() const { return getTrailingObjects<ArgInfo>(); }

public:
  static CIRGenFunctionInfo *create(CanQualType resultType);

  void operator delete(void *p) { ::operator delete(p); }

  // Friending class TrailingObjects is apparantly not good enough for MSVC, so
  // these have to be public.
  friend class TrailingObjects;

  // This function has to be CamelCase because llvm::FoldingSet requires so.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static void Profile(llvm::FoldingSetNodeID &id, CanQualType resultType) {
    resultType.Profile(id);
  }

  void Profile(llvm::FoldingSetNodeID &id) { getReturnType().Profile(id); }

  CanQualType getReturnType() const { return getArgsBuffer()[0].type; }

  cir::ABIArgInfo &getReturnInfo() { return getArgsBuffer()[0].info; }
  const cir::ABIArgInfo &getReturnInfo() const {
    return getArgsBuffer()[0].info;
  }
};

} // namespace clang::CIRGen

#endif
