//===--- ExecutorAddress.h - Executor Address ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Represents an address in the executing process (ORC-independent).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_EXECUTORADDRESS_H
#define LLVM_CLANG_INTERPRETER_EXECUTORADDRESS_H

#include <cstdint>

namespace clang {

/// Represents an address in the executor process.
/// This is a lightweight, ORC-independent wrapper for storing executor addresses.
class ExecutorAddress {
private:
  uint64_t Value = 0;

public:
  ExecutorAddress() = default;
  explicit constexpr ExecutorAddress(uint64_t Addr) : Value(Addr) {}

  /// Create an ExecutorAddress from a pointer.
  template <typename T>
  static ExecutorAddress fromPtr(T *Ptr) {
    return ExecutorAddress(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Ptr)));
  }

  /// Cast this address to a pointer or function pointer value.
  /// Use `toPtr<T>()` with `T` being the desired pointer or function-pointer
  /// type (e.g. `int (*)(void*)`) and it will return that value.
  template <typename T>
  T toPtr() const {
    return reinterpret_cast<T>(static_cast<uintptr_t>(Value));
  }

  uint64_t getValue() const { return Value; }
  void setValue(uint64_t Addr) { Value = Addr; }
  bool isNull() const { return Value == 0; }

  explicit operator bool() const { return Value != 0; }

  friend bool operator==(const ExecutorAddress &LHS, const ExecutorAddress &RHS) {
    return LHS.Value == RHS.Value;
  }
  friend bool operator!=(const ExecutorAddress &LHS, const ExecutorAddress &RHS) {
    return LHS.Value != RHS.Value;
  }
};

} // namespace clang

#endif // LLVM_CLANG_INTERPRETER_EXECUTORADDRESS_H
