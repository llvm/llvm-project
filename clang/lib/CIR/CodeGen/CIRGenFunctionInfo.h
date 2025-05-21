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
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/TrailingObjects.h"

namespace clang::CIRGen {

/// A class for recording the number of arguments that a function signature
/// requires.
class RequiredArgs {
  /// The number of required arguments, or ~0 if the signature does not permit
  /// optional arguments.
  unsigned numRequired;

public:
  enum All_t { All };

  RequiredArgs(All_t _) : numRequired(~0U) {}
  explicit RequiredArgs(unsigned n) : numRequired(n) { assert(n != ~0U); }

  unsigned getOpaqueData() const { return numRequired; }

  bool allowsOptionalArgs() const { return numRequired != ~0U; }

  /// Compute the arguments required by the given formal prototype, given that
  /// there may be some additional, non-formal arguments in play.
  ///
  /// If FD is not null, this will consider pass_object_size params in FD.
  static RequiredArgs
  getFromProtoWithExtraSlots(const clang::FunctionProtoType *prototype,
                             unsigned additional) {
    if (!prototype->isVariadic())
      return All;

    if (prototype->hasExtParameterInfos())
      llvm_unreachable("NYI");

    return RequiredArgs(prototype->getNumParams() + additional);
  }

  static RequiredArgs
  getFromProtoWithExtraSlots(clang::CanQual<clang::FunctionProtoType> prototype,
                             unsigned additional) {
    return getFromProtoWithExtraSlots(prototype.getTypePtr(), additional);
  }

  unsigned getNumRequiredArgs() const {
    assert(allowsOptionalArgs());
    return numRequired;
  }
};

// The TrailingObjects for this class contain the function return type in the
// first CanQualType slot, followed by the argument types.
class CIRGenFunctionInfo final
    : public llvm::FoldingSetNode,
      private llvm::TrailingObjects<CIRGenFunctionInfo, CanQualType> {
  RequiredArgs required;

  unsigned numArgs;

  CanQualType *getArgTypes() { return getTrailingObjects<CanQualType>(); }
  const CanQualType *getArgTypes() const {
    return getTrailingObjects<CanQualType>();
  }

  CIRGenFunctionInfo() : required(RequiredArgs::All) {}

public:
  static CIRGenFunctionInfo *create(CanQualType resultType,
                                    llvm::ArrayRef<CanQualType> argTypes,
                                    RequiredArgs required);

  void operator delete(void *p) { ::operator delete(p); }

  // Friending class TrailingObjects is apparantly not good enough for MSVC, so
  // these have to be public.
  friend class TrailingObjects;

  using const_arg_iterator = const CanQualType *;
  using arg_iterator = CanQualType *;

  // This function has to be CamelCase because llvm::FoldingSet requires so.
  // NOLINTNEXTLINE(readability-identifier-naming)
  static void Profile(llvm::FoldingSetNodeID &id, RequiredArgs required,
                      CanQualType resultType,
                      llvm::ArrayRef<CanQualType> argTypes) {
    id.AddBoolean(required.getOpaqueData());
    resultType.Profile(id);
    for (const CanQualType &arg : argTypes)
      arg.Profile(id);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Profile(llvm::FoldingSetNodeID &id) {
    // If the Profile functions get out of sync, we can end up with incorrect
    // function signatures, so we call the static Profile function here rather
    // than duplicating the logic.
    Profile(id, required, getReturnType(), arguments());
  }

  llvm::ArrayRef<CanQualType> arguments() const {
    return llvm::ArrayRef<CanQualType>(argTypesBegin(), numArgs);
  }

  llvm::ArrayRef<CanQualType> requiredArguments() const {
    return llvm::ArrayRef<CanQualType>(argTypesBegin(), getNumRequiredArgs());
  }

  CanQualType getReturnType() const { return getArgTypes()[0]; }

  const_arg_iterator argTypesBegin() const { return getArgTypes() + 1; }
  const_arg_iterator argTypesEnd() const { return getArgTypes() + 1 + numArgs; }
  arg_iterator argTypesBegin() { return getArgTypes() + 1; }
  arg_iterator argTypesEnd() { return getArgTypes() + 1 + numArgs; }

  unsigned argTypeSize() const { return numArgs; }

  llvm::MutableArrayRef<CanQualType> argTypes() {
    return llvm::MutableArrayRef<CanQualType>(argTypesBegin(), numArgs);
  }
  llvm::ArrayRef<CanQualType> argTypes() const {
    return llvm::ArrayRef<CanQualType>(argTypesBegin(), numArgs);
  }

  bool isVariadic() const { return required.allowsOptionalArgs(); }
  RequiredArgs getRequiredArgs() const { return required; }
  unsigned getNumRequiredArgs() const {
    return isVariadic() ? getRequiredArgs().getNumRequiredArgs()
                        : argTypeSize();
  }
};

} // namespace clang::CIRGen

#endif
