//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// Test that tellp() does not break the stringstream after INT_MAX, due to use
// of pbump() that accept int.

#include <string>
#include <sstream>
#include <cassert>

int main(int, char**) {
#if __SIZE_WIDTH__ == 64
  std::stringstream ss;
  std::string payload(1 << 20, 'A');

  for (size_t i = 0; i < (2ULL << 30) - payload.size(); i += payload.size()) {
    assert(ss.tellp() != -1);
    ss.write(payload.data(), payload.size());
  }

  assert(ss.tellp() != -1);
  ss.write(payload.data(), payload.size());

  assert(ss.tellp() != -1);
  ss.write(payload.data(), payload.size());
#endif

  return 0;
}
