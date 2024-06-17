//==-- LowerFunctionInfo.h - Represents of function argument/return types --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/inlcude/CodeGen/LowerFunctionInfo.h. The
// queries are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H

#include "mlir/IR/Types.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
namespace cir {

/// A class for recording the number of arguments that a function
/// signature requires.
class RequiredArgs {
  /// The number of required arguments, or ~0 if the signature does
  /// not permit optional arguments.
  unsigned NumRequired;

public:
  enum All_t { All };

  RequiredArgs(All_t _) : NumRequired(~0U) {}
  explicit RequiredArgs(unsigned n) : NumRequired(n) { assert(n != ~0U); }

  bool allowsOptionalArgs() const { return NumRequired != ~0U; }
};

// Implementation detail of LowerFunctionInfo, factored out so it can be
// named in the TrailingObjects base class of CGFunctionInfo.
struct LowerFunctionInfoArgInfo {
  mlir::Type type;        // Original ABI-agnostic type.
  ::cir::ABIArgInfo info; // ABI-specific information.
};

// FIXME(cir): We could likely encode this information within CIR/MLIR, allowing
// us to eliminate this class.
class LowerFunctionInfo final
    : private llvm::TrailingObjects<LowerFunctionInfo,
                                    LowerFunctionInfoArgInfo> {
  typedef LowerFunctionInfoArgInfo ArgInfo;

  /// The LLVM::CallingConv to use for this function (as specified by the
  /// user).
  unsigned CallingConvention : 8;

  /// The LLVM::CallingConv to actually use for this function, which may
  /// depend on the ABI.
  unsigned EffectiveCallingConvention : 8;

  /// Whether this is an instance method.
  unsigned InstanceMethod : 1;

  /// Whether this is a chain call.
  unsigned ChainCall : 1;

  /// Whether this function is called by forwarding arguments.
  /// This doesn't support inalloca or varargs.
  unsigned DelegateCall : 1;

  RequiredArgs Required;

  /// The struct representing all arguments passed in memory.  Only used when
  /// passing non-trivial types with inalloca.  Not part of the profile.
  StructType ArgStruct;

  unsigned NumArgs;

  const ArgInfo *getArgsBuffer() const { return getTrailingObjects<ArgInfo>(); }
  ArgInfo *getArgsBuffer() { return getTrailingObjects<ArgInfo>(); }

  LowerFunctionInfo() : Required(RequiredArgs::All) {}

public:
  static LowerFunctionInfo *create(unsigned llvmCC, bool instanceMethod,
                                   bool chainCall, bool delegateCall,
                                   Type resultType,
                                   ArrayRef<mlir::Type> argTypes,
                                   RequiredArgs required) {
    // TODO(cir): Add assertions?
    assert(!::cir::MissingFeatures::extParamInfo());
    void *buffer = operator new(totalSizeToAlloc<ArgInfo>(argTypes.size() + 1));

    LowerFunctionInfo *FI = new (buffer) LowerFunctionInfo();
    FI->CallingConvention = llvmCC;
    FI->EffectiveCallingConvention = llvmCC;
    FI->InstanceMethod = instanceMethod;
    FI->ChainCall = chainCall;
    FI->DelegateCall = delegateCall;
    FI->Required = required;
    FI->ArgStruct = nullptr;
    FI->NumArgs = argTypes.size();
    FI->getArgsBuffer()[0].type = resultType;
    for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
      FI->getArgsBuffer()[i + 1].type = argTypes[i];

    return FI;
  };

  // Friending class TrailingObjects is apparently not good enough for MSVC,
  // so these have to be public.
  friend class TrailingObjects;
  size_t numTrailingObjects(OverloadToken<ArgInfo>) const {
    return NumArgs + 1;
  }

  typedef const ArgInfo *const_arg_iterator;
  typedef ArgInfo *arg_iterator;

  MutableArrayRef<ArgInfo> arguments() {
    return MutableArrayRef<ArgInfo>(arg_begin(), NumArgs);
  }

  const_arg_iterator arg_begin() const { return getArgsBuffer() + 1; }
  const_arg_iterator arg_end() const { return getArgsBuffer() + 1 + NumArgs; }
  arg_iterator arg_begin() { return getArgsBuffer() + 1; }
  arg_iterator arg_end() { return getArgsBuffer() + 1 + NumArgs; }

  unsigned arg_size() const { return NumArgs; }

  bool isVariadic() const {
    assert(!::cir::MissingFeatures::variadicFunctions());
    return false;
  }
  unsigned getNumRequiredArgs() const {
    if (isVariadic())
      llvm_unreachable("NYI");
    return arg_size();
  }

  Type getReturnType() const { return getArgsBuffer()[0].type; }

  ::cir::ABIArgInfo &getReturnInfo() { return getArgsBuffer()[0].info; }
  const ::cir::ABIArgInfo &getReturnInfo() const {
    return getArgsBuffer()[0].info;
  }

  /// Return the user specified callingconvention, which has been translated
  /// into an LLVM CC.
  unsigned getCallingConvention() const { return CallingConvention; }

  /// Get the struct type used to represent all the arguments in memory.
  StructType getArgStruct() const { return ArgStruct; }
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H
