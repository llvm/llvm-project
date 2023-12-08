#define AVOID_NATIVE_UINT128_T 1
#include "flang/Common/uint128.h"
#include "testing.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>

#if (defined __GNUC__ || defined __clang__) && defined __SIZEOF_INT128__
#define HAS_NATIVE_UINT128_T 1
#else
#undef HAS_NATIVE_UINT128_T
#endif

using U128 = Fortran::common::UnsignedInt128;

static void Test(std::uint64_t x) {
  U128 n{x};
  MATCH(x, static_cast<std::uint64_t>(n));
  MATCH(~x, static_cast<std::uint64_t>(~n));
  MATCH(-x, static_cast<std::uint64_t>(-n));
  MATCH(!x, static_cast<std::uint64_t>(!n));
  TEST(n == n);
  TEST(n + n == n * static_cast<U128>(2));
  TEST(n - n == static_cast<U128>(0));
  TEST(n + n == n << static_cast<U128>(1));
  TEST(n + n == n << static_cast<U128>(1));
  TEST((n + n) - n == n);
  TEST(((n + n) >> static_cast<U128>(1)) == n);
  if (x != 0) {
    TEST(static_cast<U128>(0) / n == static_cast<U128>(0));
    TEST(static_cast<U128>(n - 1) / n == static_cast<U128>(0));
    TEST(static_cast<U128>(n) / n == static_cast<U128>(1));
    TEST(static_cast<U128>(n + n - 1) / n == static_cast<U128>(1));
    TEST(static_cast<U128>(n + n) / n == static_cast<U128>(2));
  }
}

static void Test(std::uint64_t x, std::uint64_t y) {
  U128 m{x}, n{y};
  MATCH(x, static_cast<std::uint64_t>(m));
  MATCH(y, static_cast<std::uint64_t>(n));
  MATCH(x & y, static_cast<std::uint64_t>(m & n));
  MATCH(x | y, static_cast<std::uint64_t>(m | n));
  MATCH(x ^ y, static_cast<std::uint64_t>(m ^ n));
  MATCH(x + y, static_cast<std::uint64_t>(m + n));
  MATCH(x - y, static_cast<std::uint64_t>(m - n));
  MATCH(x * y, static_cast<std::uint64_t>(m * n));
  if (n != 0) {
    MATCH(x / y, static_cast<std::uint64_t>(m / n));
  }
}

#if HAS_NATIVE_UINT128_T
static __uint128_t ToNative(U128 n) {
  return static_cast<__uint128_t>(static_cast<std::uint64_t>(n >> 64)) << 64 |
      static_cast<std::uint64_t>(n);
}

static U128 FromNative(__uint128_t n) {
  return U128{static_cast<std::uint64_t>(n >> 64)} << 64 |
      U128{static_cast<std::uint64_t>(n)};
}

static void TestVsNative(__uint128_t x, __uint128_t y) {
  U128 m{FromNative(x)}, n{FromNative(y)};
  TEST(ToNative(m) == x);
  TEST(ToNative(n) == y);
  TEST(ToNative(~m) == ~x);
  TEST(ToNative(-m) == -x);
  TEST(ToNative(!m) == !x);
  TEST(ToNative(m < n) == (x < y));
  TEST(ToNative(m <= n) == (x <= y));
  TEST(ToNative(m == n) == (x == y));
  TEST(ToNative(m != n) == (x != y));
  TEST(ToNative(m >= n) == (x >= y));
  TEST(ToNative(m > n) == (x > y));
  TEST(ToNative(m & n) == (x & y));
  TEST(ToNative(m | n) == (x | y));
  TEST(ToNative(m ^ n) == (x ^ y));
  if (y < 128) {
    TEST(ToNative(m << n) == (x << y));
    TEST(ToNative(m >> n) == (x >> y));
  }
  TEST(ToNative(m + n) == (x + y));
  TEST(ToNative(m - n) == (x - y));
  TEST(ToNative(m * n) == (x * y));
  if (y > 0) {
    TEST(ToNative(m / n) == (x / y));
    TEST(ToNative(m % n) == (x % y));
    TEST(ToNative(m - n * (m / n)) == (x % y));
  }
}

static void TestVsNative() {
  for (int j{0}; j < 128; ++j) {
    for (int k{0}; k < 128; ++k) {
      __uint128_t m{1}, n{1};
      m <<= j, n <<= k;
      TestVsNative(m, n);
      TestVsNative(~m, n);
      TestVsNative(m, ~n);
      TestVsNative(~m, ~n);
      TestVsNative(m ^ n, n);
      TestVsNative(m, m ^ n);
      TestVsNative(m ^ ~n, n);
      TestVsNative(m, ~m ^ n);
      TestVsNative(m ^ ~n, m ^ n);
      TestVsNative(m ^ n, ~m ^ n);
      TestVsNative(m ^ ~n, ~m ^ n);
      Test(m, 10000000000000000); // important case for decimal conversion
      Test(~m, 10000000000000000);
    }
  }
}
#endif

int main() {
  for (std::uint64_t j{0}; j < 64; ++j) {
    Test(j);
    Test(~j);
    Test(std::uint64_t(1) << j);
    for (std::uint64_t k{0}; k < 64; ++k) {
      Test(j, k);
    }
  }
#if HAS_NATIVE_UINT128_T
  llvm::outs() << "Environment has native __uint128_t\n";
  TestVsNative();
#else
  llvm::outs() << "Environment lacks native __uint128_t\n";
#endif
  return testing::Complete();
}
