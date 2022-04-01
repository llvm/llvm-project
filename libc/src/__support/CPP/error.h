//===-- A simple classes to manage error return vals ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_ERROR_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_ERROR_H

namespace __llvm_libc {
namespace cpp {
// Many C functions return an error val and/or the actual result of the
// evaluation/operation performed by the function. This file defines a simple
// convenience data structure to encapsulate the error and the actual val in
// a single place.

struct Error {
  int error_code;
};

// This class is implemented in a simple fashion as the intention is it add
// more generality as required. Currently, it only supports simple copyable
// types for T.
template <typename T> class ErrorOr {
  bool is_error;

  union {
    T val;
    Error error;
  };

public:
  ErrorOr(const T &value) : is_error(false), val(value) {}

  ErrorOr(const Error &error) : is_error(true), error(error) {}

  operator bool() { return !is_error; }

  operator T &() { return val; }

  T &value() { return val; }

  int error_code() { return is_error ? error.error_code : 0; }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_ERROR_H
