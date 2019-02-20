//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Without rvalue references it is impossible to detect when a rvalue deleter
// is given.
// XFAIL: c++98, c++03

// <memory>

// unique_ptr

// unique_ptr<T, const D&>(pointer, D()) should not compile

#include <memory>

struct Deleter {
  void operator()(int* p) const { delete p; }
};

int main(int, char**) {
  // expected-error@+1 {{call to deleted constructor of 'std::unique_ptr<int, const Deleter &>}}
  std::unique_ptr<int, const Deleter&> s((int*)nullptr, Deleter());

  return 0;
}
