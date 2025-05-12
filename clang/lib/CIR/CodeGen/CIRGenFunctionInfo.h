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
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/TrailingObjects.h"

namespace clang::CIRGen {

struct CIRGenFunctionInfoArgInfo {
  CanQualType type;
};

class CIRGenFunctionInfo final
    : public llvm::FoldingSetNode,
      private llvm::TrailingObjects<CIRGenFunctionInfo,
                                    CIRGenFunctionInfoArgInfo> {
  using ArgInfo = CIRGenFunctionInfoArgInfo;

  unsigned numArgs;

  ArgInfo *getArgsBuffer() { return getTrailingObjects<ArgInfo>(); }
  const ArgInfo *getArgsBuffer() const { return getTrailingObjects<ArgInfo>(); }

public:
  static CIRGenFunctionInfo *create(CanQualType resultType,
                                    llvm::ArrayRef<CanQualType> argTypes);

  void operator delete(void *p) { ::operator delete(p); }

  // Friending class TrailingObjects is apparantly not good enough for MSVC, so
  // these have to be public.
  friend class TrailingObjects;

  using const_arg_iterator = const ArgInfo *;
  using arg_iterator = ArgInfo *;

  // This function has to be CamelCase because llvm::FoldingSet requires so.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static void Profile(llvm::FoldingSetNodeID &id, CanQualType resultType,
                      llvm::ArrayRef<clang::CanQualType> argTypes) {
    resultType.Profile(id);
    for (auto i : argTypes)
      i.Profile(id);
  }

  void Profile(llvm::FoldingSetNodeID &id) { getReturnType().Profile(id); }

  llvm::MutableArrayRef<ArgInfo> arguments() {
    return llvm::MutableArrayRef<ArgInfo>(arg_begin(), numArgs);
  }
  llvm::ArrayRef<ArgInfo> arguments() const {
    return llvm::ArrayRef<ArgInfo>(arg_begin(), numArgs);
  }

  const_arg_iterator arg_begin() const { return getArgsBuffer() + 1; }
  const_arg_iterator arg_end() const { return getArgsBuffer() + 1 + numArgs; }
  arg_iterator arg_begin() { return getArgsBuffer() + 1; }
  arg_iterator arg_end() { return getArgsBuffer() + 1 + numArgs; }

  unsigned arg_size() const { return numArgs; }

  CanQualType getReturnType() const { return getArgsBuffer()[0].type; }
};

} // namespace clang::CIRGen

#endif
