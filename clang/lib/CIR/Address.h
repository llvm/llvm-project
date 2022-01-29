//===-- Address.h - An aligned address -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class provides a simple wrapper for a pair of a pointer and an
// alignment.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_ADDRESS_H
#define LLVM_CLANG_LIB_CIR_ADDRESS_H

#include "clang/AST/CharUnits.h"

#include "llvm/IR/Constants.h"

#include "mlir/IR/Value.h"

namespace cir {

class Address {
  mlir::Value Pointer;
  clang::CharUnits Alignment;

public:
  Address(mlir::Value pointer, clang::CharUnits alignment)
      : Pointer(pointer), Alignment(alignment) {
    assert((!alignment.isZero() || pointer == nullptr) &&
           "creating valid address with invalid alignment");
  }

  static Address invalid() { return Address(nullptr, clang::CharUnits()); }
  bool isValid() const { return Pointer != nullptr; }

  mlir::Value getPointer() const {
    // assert(isValid());
    return Pointer;
  }

  /// Return the alignment of this pointer.
  clang::CharUnits getAlignment() const {
    // assert(isValid());
    return Alignment;
  }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_ADDRESS_H
