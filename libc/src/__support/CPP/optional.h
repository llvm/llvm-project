//===-- Standalone implementation std::optional -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H

namespace __llvm_libc {
namespace cpp {

// Trivial nullopt_t struct.
struct nullopt_t {};

// nullopt that can be used and returned
inline constexpr nullopt_t nullopt;

// This is very simple implementation of the std::optional class.  There are a
// number of guarantees in the standard that are not made here.
//
// This class will be extended as needed in future.
//
// T currently needs to be a scalar or an object with the following properties:
//  - copy constructible
//  - copy assignable
//  - destructible
template <class T> class optional {

  template <class E> class OptionalStorage {
  public:
    union {
      E StoredValue;
      char Placeholder;
    };
    bool InUse = false;

    OptionalStorage() : Placeholder(0), InUse(false) {}
    OptionalStorage(const E &t) : StoredValue(t), InUse(true) {}
    ~OptionalStorage() {
      if (InUse)
        StoredValue.~E();
    }

    void reset() {
      if (InUse)
        StoredValue.~E();
      InUse = false;
    }
  };

  OptionalStorage<T> Storage;

public:
  optional() {}
  optional(nullopt_t) {}
  optional(const T &t) : Storage(t) {}

  T value() const { return Storage.StoredValue; }

  bool has_value() const { return Storage.InUse; }

  void reset() { Storage.reset(); }

  constexpr explicit operator bool() const { return Storage.InUse; }

  constexpr optional &operator=(nullopt_t) {
    reset();
    return *this;
  }

  constexpr T &operator*() & { return Storage.StoredValue; }

  constexpr const T &operator*() const & { return Storage.StoredValue; }

  constexpr T *operator->() { return &Storage.StoredValue; }

  constexpr const T *operator->() const { return &Storage.StoredValue; }
};
} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H
