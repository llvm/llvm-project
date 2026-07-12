//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// A program that instantiates the definition of unique_ptr<T, D> is ill-formed if T* is an invalid type.

// LWG 4144

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <memory>

struct Deleter {
  typedef int* pointer;

  void operator()(pointer) const;
};

typedef void Function();
typedef void AbominableFunction() const;

void pointable_function_type() { (void)sizeof(std::unique_ptr<Function, Deleter>); }

void reference_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}unique_ptr<T, D> requires T* to be a valid type}}
  (void)sizeof(std::unique_ptr<int&, Deleter>);
}

void abominable_function_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}unique_ptr<T, D> requires T* to be a valid type}}
  (void)sizeof(std::unique_ptr<AbominableFunction, Deleter>);
}
