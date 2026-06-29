//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23, c++26

#include <embed>

#depend __FILE__

#include "test_macros.h"

consteval bool test() {
#if __cpp_lib_embed
  // make sure to test each individual overload
  (void)std::embed<signed char>(__FILE__);
  // expected-error@-1 {{embed type must be unqualified 'char', 'unsigned char', or 'byte'}}
  (void)std::embed("praise the sun!");
  // expected-error@-1 {{file not found}}
  (void)std::embed<signed char>("a.txt");
  // expected-error@-1 {{file found but not appropriately '#depend ...'ed on}}
  (void)std::embed<1234>("a.txt");
  // expected-error@-1 {{the fixed-size span extent is larger than the resource data size}}
#endif
  return true;
}

int main(int, char*[]) {
  static_assert(test());
  return 0;
}
