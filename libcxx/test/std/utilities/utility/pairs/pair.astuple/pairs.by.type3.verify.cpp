//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

#include <memory>
#include <utility>

void f() {
  typedef std::unique_ptr<int> Ptr;
  std::pair<Ptr, int> t(Ptr(new int(4)), 23);
  Ptr p = std::get<Ptr>(t); // expected-error {{call to implicitly-deleted copy constructor of 'Ptr'}}
}
