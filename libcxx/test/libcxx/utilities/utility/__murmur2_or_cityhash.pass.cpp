//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test the CityHash implementation is correct.

// UNSUPPORTED: c++03

#include <cassert>
#include <string>
#include <utility>

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define CHOOSE_BY_ENDIANESS(little, big) (little)
#else
#  define CHOOSE_BY_ENDIANESS(little, big) (big)
#endif

int main(int, char**) {
  const std::pair<std::string, uint64_t> TestCases[] = {
      {"abcdefgh", CHOOSE_BY_ENDIANESS(0x4382a8d0fe8edb17ULL, 0xca84e809bef16fbcULL)},
      {"abcDefgh", CHOOSE_BY_ENDIANESS(0xecefb080a6854061ULL, 0xd7feb824250272dcULL)},
      {"CityHash", CHOOSE_BY_ENDIANESS(0x169ea3aebf908d6dULL, 0xea8cef3ca6f6e368ULL)},
      {"CitYHash", CHOOSE_BY_ENDIANESS(0xe18298a2760f09faULL, 0xf33a7700bb7a94a8ULL)},
  };

  std::__murmur2_or_cityhash<uint64_t> h64;
  for (const auto& test_case : TestCases) {
    assert(h64(test_case.first.data(), test_case.first.size()) == test_case.second);
  }
  return 0;
}
