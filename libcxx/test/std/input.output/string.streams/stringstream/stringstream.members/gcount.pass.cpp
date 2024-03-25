//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: 32-bit-pointer
// REQUIRES: large_tests

// Android devices frequently don't have enough memory to run this test. Rather
// than throw std::bad_alloc, exhausting memory triggers the OOM Killer.
// UNSUPPORTED: LIBCXX-ANDROID-FIXME

// Test that tellp() does not break the stringstream after INT_MAX, due to use
// of pbump() that accept int.

#include <cassert>
#include <climits>
#include <sstream>
#include <string>

int main(int, char**) {
  std::stringstream ss;
  std::string payload(INT_MAX - 1, '\0');

  ss.write(payload.data(), payload.size());
  assert(ss.tellp() == INT_MAX - 1);

  ss.write("a", 1);
  assert(ss.tellp() == INT_MAX);

  ss.write("b", 1);
  assert(ss.tellp() == INT_MAX + 1ULL);
  // it fails only after previous tellp() corrupts the internal field with int
  // overflow
  assert(ss.tellp() == INT_MAX + 1ULL);

  return 0;
}
