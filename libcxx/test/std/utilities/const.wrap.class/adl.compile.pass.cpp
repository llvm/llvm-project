//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// [Note 1: The unnamed second template parameter to constant_wrapper is present
// to aid argument-dependent lookup ([basic.lookup.argdep]) in finding overloads
// for which constant_wrapper's wrapped value is a suitable argument, but for which
// the constant_wrapper itself is not. - end note]

#include <utility>

namespace MyNamespace {
struct MyType {};

void adl_function(MyType) {}

} // namespace MyNamespace

void test() {
  std::constant_wrapper<MyNamespace::MyType{}> cw_mt;
  adl_function(cw_mt);
}
