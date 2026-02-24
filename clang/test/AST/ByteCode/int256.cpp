// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple x86_64-unknown-linux-gnu -std=c++20 -verify=expected,both %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -verify=ref,both %s

// Constexpr evaluation tests for __int256_t / __uint256_t.

namespace Arithmetic {
  constexpr __int256_t a = 100;
  constexpr __int256_t b = 7;
  static_assert(a + b == 107, "");
  static_assert(a - b == 93, "");
  static_assert(a * b == 700, "");
  static_assert(a / b == 14, "");
  static_assert(a % b == 2, "");

  constexpr __int256_t product = 12345 * 67890;
  static_assert(product == 838102050, "");
}

namespace Bitwise {
  constexpr __uint256_t x = 0xFF00FF;
  constexpr __uint256_t y = 0x0F0F0F;
  static_assert((x & y) == 0x0F000F, "");
  static_assert((x | y) == 0xFF0FFF, "");
  static_assert((x ^ y) == 0xF00FF0, "");
  static_assert(~(__uint256_t)0 != 0, "");
}

namespace Shifts {
  constexpr __int256_t one = 1;
  static_assert((one << 0) == 1, "");
  static_assert((one << 1) == 2, "");
  static_assert((one << 64) != 0, "");
  static_assert((one << 128) != 0, "");
  static_assert((one << 255) != 0, "");
  static_assert(((__uint256_t)one << 255) >> 255 == 1, "");

  constexpr __uint256_t large = (__uint256_t)1 << 200;
  static_assert(large != 0, "");
  static_assert(large >> 200 == 1, "");
}

namespace Comparisons {
  constexpr __int256_t a = 100;
  constexpr __int256_t b = 7;
  static_assert(a > b, "");
  static_assert(b < a, "");
  static_assert(a >= 100, "");
  static_assert(b <= 7, "");
  static_assert(a != b, "");
  static_assert(a == 100, "");
}

namespace Conversions {
  constexpr __int128_t i128 = 42;
  constexpr __int256_t from128 = i128;
  static_assert(from128 == 42, "");
  constexpr __int128_t to128 = (__int128_t)from128;
  static_assert(to128 == 42, "");

  constexpr long long ll = 99;
  constexpr __int256_t fromll = ll;
  static_assert(fromll == 99, "");
}

namespace UnaryOps {
  constexpr __int256_t a = 100;
  constexpr __int256_t neg = -a;
  static_assert(neg == -100, "");
  static_assert(-neg == 100, "");
}

namespace Wrapping {
  constexpr __uint256_t zero = 0;
  constexpr __uint256_t wrap = zero - 1;
  static_assert(wrap + 1 == 0, "");
}

namespace DivByZero {
  constexpr __int256_t divzero = __int256_t{1} / __int256_t{0}; // both-error {{must be initialized by a constant expression}} \
                                                                  // both-note {{division by zero}}
  constexpr __int256_t remzero = __int256_t{1} % __int256_t{0}; // both-error {{must be initialized by a constant expression}} \
                                                                  // both-note {{division by zero}}
}

namespace BoundaryConstants {
  // UINT256_MAX = 2^256 - 1 = ((__uint256_t)1 << 255) | (((__uint256_t)1 << 255) - 1)
  constexpr __uint256_t UINT256_MAX = ~(__uint256_t)0;
  static_assert(UINT256_MAX != 0, "");
  static_assert(UINT256_MAX + 1 == 0, ""); // wraps to zero
  static_assert((UINT256_MAX >> 255) == 1, "");

  // INT256_MAX = 2^255 - 1 (sign bit clear, all other bits set)
  constexpr __int256_t INT256_MAX = (__int256_t)(UINT256_MAX >> 1);
  static_assert(INT256_MAX > 0, "");
  constexpr __uint256_t check_max = (__uint256_t)INT256_MAX;
  static_assert((check_max >> 254) == 1, ""); // bit 254 set

  // INT256_MIN = -2^255 (sign bit set, all other bits clear)
  constexpr __int256_t INT256_MIN = -INT256_MAX - 1;
  static_assert(INT256_MIN < 0, "");
  static_assert(INT256_MIN + INT256_MAX == -1, "");

  // Full-width values using all 256 bits
  constexpr __uint256_t all_ones = ~(__uint256_t)0;
  constexpr __uint256_t alternating = all_ones / 3; // 0x5555...
  static_assert(alternating != 0, "");
  static_assert((alternating & (alternating << 1)) == 0, ""); // no adjacent bits
}

namespace OverflowDetection {
  // Signed overflow in constexpr is undefined behavior -- not a constant expression
  constexpr __int256_t INT256_MAX = (__int256_t)(~(__uint256_t)0 >> 1);
  constexpr __int256_t overflow_add = INT256_MAX + 1; // both-error {{must be initialized by a constant expression}} \
                                                       // both-note {{value 57896044618658097711785492504343953926634992332820282019728792003956564819968 is outside the range of representable values}}
}

namespace MoreConversions {
  // Bool conversions
  constexpr bool from_zero = (__int256_t)0;
  static_assert(!from_zero, "");
  constexpr bool from_one = (__int256_t)1;
  static_assert(from_one, "");
  constexpr bool from_neg = (__int256_t)-1;
  static_assert(from_neg, "");

  // Char conversions
  constexpr char c = 'A';
  constexpr __int256_t from_char = c;
  static_assert(from_char == 65, "");
  constexpr char to_char = (char)from_char;
  static_assert(to_char == 'A', "");

  // Int conversions
  constexpr int i = 42;
  constexpr __int256_t from_int = i;
  static_assert(from_int == 42, "");
  constexpr int to_int = (int)from_int;
  static_assert(to_int == 42, "");

  // Long conversions
  constexpr long l = 1000000L;
  constexpr __int256_t from_long = l;
  static_assert(from_long == 1000000, "");

  // __int256 <-> __int128 round-trip with negative
  constexpr __int128_t neg128 = -42;
  constexpr __int256_t from_neg128 = neg128;
  static_assert(from_neg128 == -42, "");
  constexpr __int128_t to_neg128 = (__int128_t)from_neg128;
  static_assert(to_neg128 == -42, "");
}

namespace CompoundAssignment {
  constexpr __int256_t test_compound() {
    __int256_t x = 100;
    x += 50;   // 150
    x -= 30;   // 120
    x *= 2;    // 240
    x /= 3;    // 80
    x %= 7;    // 3
    x <<= 4;   // 48
    x >>= 2;   // 12
    x &= 0xFF; // 12
    x |= 0x100;// 268
    x ^= 0xF;  // 263
    return x;
  }
  static_assert(test_compound() == 259, "");
}

namespace IncrementDecrement {
  constexpr __int256_t test_inc_dec() {
    __int256_t x = 0;
    ++x;      // 1
    x++;      // 2
    --x;      // 1
    x--;      // 0
    return x;
  }
  static_assert(test_inc_dec() == 0, "");

  // Unsigned wrapping with decrement
  constexpr __uint256_t test_wrap_dec() {
    __uint256_t x = 0;
    --x; // wraps to UINT256_MAX
    ++x; // wraps back to 0
    return x;
  }
  static_assert(test_wrap_dec() == 0, "");
}

namespace ConstexprFunc {
  constexpr __int256_t factorial(__int256_t n) {
    __int256_t result = 1;
    for (__int256_t i = 2; i <= n; ++i)
      result *= i;
    return result;
  }
  static_assert(factorial(10) == 3628800, "");
  static_assert(factorial(20) == 2432902008176640000LL, "");
}
