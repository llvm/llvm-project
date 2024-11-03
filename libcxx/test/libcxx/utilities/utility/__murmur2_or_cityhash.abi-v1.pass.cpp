//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test that the CityHash implementation returns the results we expect.
//
// Note that this implementation is technically incorrect, however changing it is
// an ABI break. This test ensures that we don't unintentionally break the ABI v1
// by "fixing" the hash implementation.
// REQUIRES: libcpp-abi-version=1

#include <cassert>
#include <string>
#include <utility>

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define CHOOSE_BY_ENDIANESS(little, big) (little)
#else
#  define CHOOSE_BY_ENDIANESS(little, big) (big)
#endif

std::string CityHash[] = {
    {/* "abcdefgh" */ "\x61\x62\x63\x64\x65\x66\x67\x68"},
    {/* "abcDefgh" */ "\x61\x62\x63\x44\x65\x66\x67\x68"},
    {/* "CityHash" */ "\x43\x69\x74\x79\x48\x61\x73\x68"},
    {/* "CitYHash" */ "\x43\x69\x74\x59\x48\x61\x73\x68"},
};

int main(int, char**) {
  const std::pair<std::string, uint64_t> TestCases[] = {
      {CityHash[0], CHOOSE_BY_ENDIANESS(0x87c69099911bab7eULL, 0x297621d7fa436a3ULL)},
      {CityHash[1], CHOOSE_BY_ENDIANESS(0x87c69099911bab7eULL, 0xb17be531dde56e57ULL)},
      {CityHash[2], CHOOSE_BY_ENDIANESS(0x85322632e188694aULL, 0xe14f578b688e266dULL)},
      {CityHash[3], CHOOSE_BY_ENDIANESS(0x85322632e188694aULL, 0xca5a764a0450eac6ULL)},
  };

  std::__murmur2_or_cityhash<uint64_t> h64;
  for (const auto& test_case : TestCases) {
    assert(h64(test_case.first.data(), test_case.first.size()) == test_case.second);
  }

  return 0;
}
