//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

#include <bit>
#include <cstdint>

template <typename T>
concept can_reverse = requires(T x) { std::bit_reverse(x); };

static_assert(!can_reverse<bool>, "type is not allowed to be reversed");
static_assert(!can_reverse<float>);

static_assert(!can_reverse<signed char>, "signed integers are not allowed to be reversed");
static_assert(!can_reverse<short>);
static_assert(!can_reverse<int>);
static_assert(!can_reverse<long long>);
static_assert(!can_reverse<std::int8_t>);
static_assert(!can_reverse<std::int16_t>);
static_assert(!can_reverse<std::int32_t>);
static_assert(!can_reverse<std::int64_t>);

static_assert(!can_reverse<char*>, "pointers are not allowed to be reversed");
static_assert(!can_reverse<short*>);
static_assert(!can_reverse<int*>);
static_assert(!can_reverse<long long*>);
static_assert(!can_reverse<std::int8_t*>);
static_assert(!can_reverse<std::int16_t*>);
static_assert(!can_reverse<std::int32_t*>);
static_assert(!can_reverse<std::int64_t*>);

int main(int, char**) {
  using std::uint16_t;
  using std::uint32_t;
  using std::uint64_t;
  using std::uint8_t;

  // Test the bit patterns:
  // - 1111 == 1111
  // - 1010 == 0101
  // - 0101 == 1010
  // - 1000 == 0001
  // - 0001 == 1000

  // uint8_t
  static_assert(std::bit_reverse(uint8_t{0x0}) == uint8_t{0x0});
  static_assert(std::bit_reverse(uint8_t{0xFF}) == uint8_t{0xFF});
  static_assert(std::bit_reverse(uint8_t{0xAA}) == uint8_t{0x55});
  static_assert(std::bit_reverse(uint8_t{0x55}) == uint8_t{0xAA});
  static_assert(std::bit_reverse(uint8_t{0x88}) == uint8_t{0x11});
  static_assert(std::bit_reverse(uint8_t{0x11}) == uint8_t{0x88});

  // uint16_t
  static_assert(std::bit_reverse(uint16_t{0x0}) == uint16_t{0x0});
  static_assert(std::bit_reverse(uint16_t{0xFFFF}) == uint16_t{0xFFFF});
  static_assert(std::bit_reverse(uint16_t{0xAAAA}) == uint16_t{0x5555});
  static_assert(std::bit_reverse(uint16_t{0x5555}) == uint16_t{0xAAAA});
  static_assert(std::bit_reverse(uint16_t{0x8888}) == uint16_t{0x1111});
  static_assert(std::bit_reverse(uint16_t{0x1111}) == uint16_t{0x8888});

  // uint32_t
  static_assert(std::bit_reverse(uint32_t{0x0}) == uint32_t{0x0});
  static_assert(std::bit_reverse(uint32_t{0xFFFFFFFF}) == uint32_t{0xFFFFFFFF});
  static_assert(std::bit_reverse(uint32_t{0xAAAAAAAA}) == uint32_t{0x55555555});
  static_assert(std::bit_reverse(uint32_t{0x55555555}) == uint32_t{0xAAAAAAAA});
  static_assert(std::bit_reverse(uint32_t{0x88888888}) == uint32_t{0x11111111});
  static_assert(std::bit_reverse(uint32_t{0x11111111}) == uint32_t{0x88888888});

  // uint64_t
  static_assert(std::bit_reverse(uint64_t{0x0}) == uint64_t{0x0});
  static_assert(std::bit_reverse(uint64_t{0xFFFFFFFFFFFFFFFF}) == uint64_t{0xFFFFFFFFFFFFFFFF});
  static_assert(std::bit_reverse(uint64_t{0xAAAAAAAAAAAAAAAA}) == uint64_t{0x5555555555555555});
  static_assert(std::bit_reverse(uint64_t{0x5555555555555555}) == uint64_t{0xAAAAAAAAAAAAAAAA});
  static_assert(std::bit_reverse(uint64_t{0x8888888888888888}) == uint64_t{0x1111111111111111});
  static_assert(std::bit_reverse(uint64_t{0x1111111111111111}) == uint64_t{0x8888888888888888});

  return 0;
}
