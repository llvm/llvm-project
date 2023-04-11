//===- ConstantInitFuture.h - "Future" constant initializers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class defines the ConstantInitFuture class.  This is split out
// from ConstantInitBuilder.h in order to allow APIs to work with it
// without having to include that entire header.  This is particularly
// important because it is often useful to be able to default-construct
// a future in, say, a default argument.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CODEGEN_CONSTANTINITFUTURE_H
#define LLVM_CLANG_CIR_CODEGEN_CONSTANTINITFUTURE_H

#include "mlir/IR/Attributes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/PointerUnion.h"

// Forward-declare ConstantInitBuilderBase and give it a
// PointerLikeTypeTraits specialization so that we can safely use it
// in a PointerUnion below.
namespace cir {
class ConstantInitBuilderBase;
} // namespace cir

namespace llvm {
template <> struct PointerLikeTypeTraits<::cir::ConstantInitBuilderBase *> {
  using T = ::cir::ConstantInitBuilderBase *;

  static inline void *getAsVoidPointer(T p) { return p; }
  static inline T getFromVoidPointer(void *p) { return static_cast<T>(p); }
  static constexpr int NumLowBitsAvailable = 2;
};
} // namespace llvm

namespace cir {

/// A "future" for a completed constant initializer, which can be passed
/// around independently of any sub-builders (but not the original parent).
class ConstantInitFuture {
  using PairTy = llvm::PointerUnion<ConstantInitBuilderBase *, mlir::Attribute>;

  PairTy Data;

  friend class ConstantInitBuilderBase;
  explicit ConstantInitFuture(ConstantInitBuilderBase *builder);

public:
  ConstantInitFuture() {}

  /// A future can be explicitly created from a fixed initializer.
  explicit ConstantInitFuture(mlir::Attribute initializer) : Data(initializer) {
    assert(initializer && "creating null future");
  }

  /// Is this future non-null?
  explicit operator bool() const { return bool(Data); }

  /// Return the type of the initializer.
  mlir::Type getType() const;

  /// Abandon this initializer.
  void abandon();

  /// Install the initializer into a global variable.  This cannot
  /// be called multiple times.
  void installInGlobal(mlir::cir::GlobalOp global);

  void *getOpaqueValue() const { return Data.getOpaqueValue(); }
  static ConstantInitFuture getFromOpaqueValue(void *value) {
    ConstantInitFuture result;
    result.Data = PairTy::getFromOpaqueValue(value);
    return result;
  }
  static constexpr int NumLowBitsAvailable =
      llvm::PointerLikeTypeTraits<PairTy>::NumLowBitsAvailable;
};

} // namespace cir

namespace llvm {

template <> struct PointerLikeTypeTraits<::cir::ConstantInitFuture> {
  using T = ::cir::ConstantInitFuture;

  static inline void *getAsVoidPointer(T future) {
    return future.getOpaqueValue();
  }
  static inline T getFromVoidPointer(void *p) {
    return T::getFromOpaqueValue(p);
  }
  static constexpr int NumLowBitsAvailable = T::NumLowBitsAvailable;
};

} // end namespace llvm

#endif
