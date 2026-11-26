//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <numeric>
// [numeric.clmul]

#include <cassert>
#include <limits>
#include <numeric>

#include "test_macros.h"
#include "type_algorithms.h"

template <class T>
constexpr bool test() {
  constexpr int width = std::numeric_limits<T>::digits;
  using WT            = std::wide_result<T>;

  assert(WT(T(0), T(0)) == WT(T(0), T(0)));
  assert(WT(T(1), T(0)) > WT(T(0), T(~T(0))));
  assert(WT(T(~T(0)), T(~T(0))) > WT(T(0), T(0)));

  if constexpr (width == 8) {
    assert(std::clmul(T(0), T(0)) == T(0));
    assert(std::widening_clmul(T(0), T(0)) == WT(T(0), T(0)));
    assert(std::clmul(T(0), T(1)) == T(0));
    assert(std::widening_clmul(T(0), T(1)) == WT(T(0), T(0)));
    assert(std::clmul(T(1), T(1)) == T(1));
    assert(std::widening_clmul(T(1), T(1)) == WT(T(0), T(1)));
    assert(std::clmul(T(1), T(0xFF)) == T(0xFF));
    assert(std::widening_clmul(T(1), T(0xFF)) == WT(T(0), T(0xFF)));
    assert(std::clmul(T(0xFF), T(0xFF)) == T(0x55));
    assert(std::widening_clmul(T(0xFF), T(0xFF)) == WT(T(0x55), T(0x55)));
    assert(std::clmul(T(0xA5), T(0x3C)) == T(0x4C));
    assert(std::widening_clmul(T(0xA5), T(0x3C)) == WT(T(0x19), T(0x4C)));
    assert(std::clmul(T(0xB0), T(0xC6)) == T(0xA0));
    assert(std::widening_clmul(T(0xB0), T(0xC6)) == WT(T(0x77), T(0xA0)));
    assert(std::clmul(T(0x34), T(0x7B)) == T(0x3C));
    assert(std::widening_clmul(T(0x34), T(0x7B)) == WT(T(0x09), T(0x3C)));
    assert(std::clmul(T(0x89), T(0x10)) == T(0x90));
    assert(std::widening_clmul(T(0x89), T(0x10)) == WT(T(0x08), T(0x90)));
  } else if constexpr (width == 16) {
    assert(std::clmul(T(0), T(0)) == T(0));
    assert(std::widening_clmul(T(0), T(0)) == WT(T(0), T(0)));
    assert(std::clmul(T(0), T(1)) == T(0));
    assert(std::widening_clmul(T(0), T(1)) == WT(T(0), T(0)));
    assert(std::clmul(T(1), T(1)) == T(1));
    assert(std::widening_clmul(T(1), T(1)) == WT(T(0), T(1)));
    assert(std::clmul(T(1), T(0xFFFF)) == T(0xFFFF));
    assert(std::widening_clmul(T(1), T(0xFFFF)) == WT(T(0), T(0xFFFF)));
    assert(std::clmul(T(0xFFFF), T(0xFFFF)) == T(0x5555));
    assert(std::widening_clmul(T(0xFFFF), T(0xFFFF)) == WT(T(0x5555), T(0x5555)));
    assert(std::clmul(T(0x1234), T(0xFEDC)) == T(0xB530));
    assert(std::widening_clmul(T(0x1234), T(0xFEDC)) == WT(T(0x0E03), T(0xB530)));
    assert(std::clmul(T(0xBEEF), T(0x1357)) == T(0xACBD));
    assert(std::widening_clmul(T(0xBEEF), T(0x1357)) == WT(T(0x0A0A), T(0xACBD)));
    assert(std::clmul(T(0x6D2B), T(0x9E37)) == T(0xE601));
    assert(std::widening_clmul(T(0x6D2B), T(0x9E37)) == WT(T(0x324A), T(0xE601)));
    assert(std::clmul(T(0x7EB0), T(0x6AC6)) == T(0x13A0));
    assert(std::widening_clmul(T(0x7EB0), T(0x6AC6)) == WT(T(0x1353), T(0x13A0)));
    assert(std::clmul(T(0x8A34), T(0x557B)) == T(0x033C));
    assert(std::widening_clmul(T(0x8A34), T(0x557B)) == WT(T(0x28B2), T(0x033C)));
    assert(std::clmul(T(0x4C89), T(0x1710)) == T(0xE790));
    assert(std::widening_clmul(T(0x4C89), T(0x1710)) == WT(T(0x052B), T(0xE790)));
  } else if constexpr (width == 32) {
    assert(std::clmul(T(0), T(0)) == T(0));
    assert(std::widening_clmul(T(0), T(0)) == WT(T(0), T(0)));
    assert(std::clmul(T(0), T(1)) == T(0));
    assert(std::widening_clmul(T(0), T(1)) == WT(T(0), T(0)));
    assert(std::clmul(T(1), T(1)) == T(1));
    assert(std::widening_clmul(T(1), T(1)) == WT(T(0), T(1)));
    assert(std::clmul(T(1), T(0xFFFFFFFF)) == T(0xFFFFFFFF));
    assert(std::widening_clmul(T(1), T(0xFFFFFFFF)) == WT(T(0), T(0xFFFFFFFF)));
    assert(std::clmul(T(0xFFFFFFFF), T(0xFFFFFFFF)) == T(0x55555555));
    assert(std::widening_clmul(T(0xFFFFFFFF), T(0xFFFFFFFF)) == WT(T(0x55555555), T(0x55555555)));
    assert(std::clmul(T(0x12345678), T(0x9ABCDEF0)) == T(0x5CD25A80));
    assert(std::widening_clmul(T(0x12345678), T(0x9ABCDEF0)) == WT(T(0x08860E94), T(0x5CD25A80)));
    assert(std::clmul(T(0xBADC0FFE), T(0x13579BDF)) == T(0x08C5BD6A));
    assert(std::widening_clmul(T(0xBADC0FFE), T(0x13579BDF)) == WT(T(0x0A44C89C), T(0x08C5BD6A)));
    assert(std::clmul(T(0x6D2B79F5), T(0x9E3779B9)) == T(0x1A45002D));
    assert(std::widening_clmul(T(0x6D2B79F5), T(0x9E3779B9)) == WT(T(0x324ACCB1), T(0x1A45002D)));
    assert(std::clmul(T(0x65307EB0), T(0xB5C46AC6)) == T(0x073313A0));
    assert(std::widening_clmul(T(0x65307EB0), T(0xB5C46AC6)) == WT(T(0x39B69023), T(0x073313A0)));
    assert(std::clmul(T(0x17B08A34), T(0xCDDB557B)) == T(0x35E033C));
    assert(std::widening_clmul(T(0x17B08A34), T(0xCDDB557B)) == WT(T(0x0ECF5F86), T(0x35E033C)));
    assert(std::clmul(T(0xF32C4C89), T(0x410E1710)) == T(0x3595E790));
    assert(std::widening_clmul(T(0xF32C4C89), T(0x410E1710)) == WT(T(0x3C3D8101), T(0x3595E790)));
  } else if constexpr (width == 64) {
    assert(std::clmul(T(0), T(0)) == T(0));
    assert(std::widening_clmul(T(0), T(0)) == WT(T(0), T(0)));
    assert(std::clmul(T(0), T(1)) == T(0));
    assert(std::widening_clmul(T(0), T(1)) == WT(T(0), T(0)));
    assert(std::clmul(T(1), T(1)) == T(1));
    assert(std::widening_clmul(T(1), T(1)) == WT(T(0), T(1)));
    assert(std::clmul(T(1), T(~T(0))) == T(~T(0)));
    assert(std::widening_clmul(T(1), T(~T(0))) == WT(T(0), T(~T(0))));
    assert(std::clmul(T(~T(0)), T(~T(0))) == T(0x5555555555555555ULL));
    assert(std::widening_clmul(T(~T(0)), T(~T(0))) == WT(T(0x5555555555555555ULL), T(0x5555555555555555ULL)));
    assert(std::clmul(T(0x8C0D272E65307EB0ULL), T(0x1155F8CAB5C46AC6ULL)) == T(0x13AC0EA7073313A0ULL));
    assert(std::widening_clmul(T(0x8C0D272E65307EB0ULL), T(0x1155F8CAB5C46AC6ULL)) ==
           WT(T(0x0865D8B32F19DE39ULL), T(0x13AC0EA7073313A0ULL)));
    assert(std::clmul(T(0xD0671C2817B08A34ULL), T(0x123AEF22CDDB557BULL)) == T(0xC25921D6035E033CULL));
    assert(std::widening_clmul(T(0xD0671C2817B08A34ULL), T(0x123AEF22CDDB557BULL)) ==
           WT(T(0xCB6D4111DAA3090ULL), T(0xC25921D6035E033CULL)));
    assert(std::clmul(T(0x4179DE42F32C4C89ULL), T(0x52CE9565410E1710ULL)) == T(0xAB88FBEC3595E790ULL));
    assert(std::widening_clmul(T(0x4179DE42F32C4C89ULL), T(0x52CE9565410E1710ULL)) ==
           WT(T(0x14F8527C59EF06BCULL), T(0xAB88FBEC3595E790ULL)));
    assert(std::clmul(T(0xEA7C4C81FB69DF35ULL), T(0x31945563A5215C5BULL)) == T(0x471107844130CAE7ULL));
    assert(std::widening_clmul(T(0xEA7C4C81FB69DF35ULL), T(0x31945563A5215C5BULL)) ==
           WT(T(0x137A365AC462150DULL), T(0x471107844130CAE7ULL)));
  } else if constexpr (width == 128) {
    constexpr T all_ones     = ~T(0);
    constexpr T ones_product = T(0x5555555555555555ULL) | (T(0x5555555555555555ULL) << 64);
    assert(std::clmul(T(0), T(0)) == T(0));
    assert(std::widening_clmul(T(0), T(0)) == WT(T(0), T(0)));
    assert(std::clmul(T(0), T(1)) == T(0));
    assert(std::widening_clmul(T(0), T(1)) == WT(T(0), T(0)));
    assert(std::clmul(T(1), T(1)) == T(1));
    assert(std::widening_clmul(T(1), T(1)) == WT(T(0), T(1)));
    assert(std::clmul(T(1), all_ones) == all_ones);
    assert(std::widening_clmul(T(1), all_ones) == WT(T(0), all_ones));
    assert(std::clmul(all_ones, all_ones) == ones_product);
    assert(std::widening_clmul(all_ones, all_ones) == WT(ones_product, ones_product));

    constexpr T x0    = T(0x8C0D272E65307EB0ULL) | (T(0x1155F8CAB5C46AC6ULL) << 64);
    constexpr T y0    = T(0xD0671C2817B08A34ULL) | (T(0x123AEF22CDDB557BULL) << 64);
    constexpr T low0  = T(0x75AA8ECDC89927C0ULL) | (T(0x9F2C9B6EB4F58720ULL) << 64);
    constexpr T high0 = T(0x369A10189A8ECC56ULL) | (T(0x01346DB2A350FF43ULL) << 64);
    assert(std::clmul(x0, y0) == low0);
    assert(std::widening_clmul(x0, y0) == WT(high0, low0));

    constexpr T x1    = T(0x4179DE42F32C4C89ULL) | (T(0x52CE9565410E1710ULL) << 64);
    constexpr T y1    = T(0xEA7C4C81FB69DF35ULL) | (T(0x31945563A5215C5BULL) << 64);
    constexpr T low1  = T(0xF8573444065C801DULL) | (T(0x5BCB04C57AA2A9B2ULL) << 64);
    constexpr T high1 = T(0x42A5CC207AF75E00ULL) | (T(0x0F0AE56EAFD77584ULL) << 64);
    assert(std::clmul(x1, y1) == low1);
    assert(std::widening_clmul(x1, y1) == WT(high1, low1));
  } else if constexpr (width == 256) {
    constexpr T all_ones = ~T(0);
    constexpr T ones_product =
        T(0x5555555555555555ULL) | (T(0x5555555555555555ULL) << 64) | (T(0x5555555555555555ULL) << 128) |
        (T(0x5555555555555555ULL) << 192);
    assert(std::clmul(T(0), T(0)) == T(0));
    assert(std::widening_clmul(T(0), T(0)) == WT(T(0), T(0)));
    assert(std::clmul(T(0), T(1)) == T(0));
    assert(std::widening_clmul(T(0), T(1)) == WT(T(0), T(0)));
    assert(std::clmul(T(1), T(1)) == T(1));
    assert(std::widening_clmul(T(1), T(1)) == WT(T(0), T(1)));
    assert(std::clmul(T(1), all_ones) == all_ones);
    assert(std::widening_clmul(T(1), all_ones) == WT(T(0), all_ones));
    assert(std::clmul(all_ones, all_ones) == ones_product);
    assert(std::widening_clmul(all_ones, all_ones) == WT(ones_product, ones_product));

    constexpr T x    = T(0x8C0D272E65307EB0ULL) | (T(0x1155F8CAB5C46AC6ULL) << 64) | (T(0xD0671C2817B08A34ULL) << 128) |
                       (T(0x123AEF22CDDB557BULL) << 192);
    constexpr T y    = T(0x4179DE42F32C4C89ULL) | (T(0x52CE9565410E1710ULL) << 64) | (T(0xEA7C4C81FB69DF35ULL) << 128) |
                       (T(0x31945563A5215C5BULL) << 192);
    constexpr T low  = T(0x64B3C8F9806F9330ULL) | (T(0xD22EC9F14555C7FFULL) << 64) | (T(0xDA0CCAB7F5BF5415ULL) << 128) |
                       (T(0x93300B32A86B89E7ULL) << 192);
    constexpr T high = T(0x0D8661C84486F89BULL) | (T(0xEEDE8C5CC39C7625ULL) << 64) | (T(0xFF78263BDF8E1D35ULL) << 128) |
                       (T(0x037EBA3AB8A88333ULL) << 192);
    assert(std::clmul(x, y) == low);
    assert(std::widening_clmul(x, y) == WT(high, low));
  }

  return true;
}

int main(int, char**) {
#if TEST_HAS_BITINT
  using test_types =
      types::concatenate_t< types::unsigned_integer_types,
                            types::type_list<unsigned _BitInt(32),
                                             unsigned _BitInt(64)
#  if __BITINT_MAXWIDTH__ >= 128
                                                 ,
                                             unsigned _BitInt(128)
#  endif
#  if __BITINT_MAXWIDTH__ >= 256
                                                 ,
                                             unsigned _BitInt(256)
#  endif
                                             > >;
#else
  using test_types = types::unsigned_integer_types;
#endif

  types::for_each(test_types{}, []<class T> {
    static_assert(test<T>());
    test<T>();
  });

  return 0;
}
