//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

#include <assert.h>
#include <ext/hash_map>
#include <string>

int main(int, char**) {
  assert(__gnu_cxx::hash<std::string>()(std::string()) == 0); // expected-error {{does not provide a call operator}}

  return 0;
}
