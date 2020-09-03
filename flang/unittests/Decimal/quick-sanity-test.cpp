#include "flang/Decimal/decimal.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>
#include <cstdio>
#include <cstring>

using namespace Fortran::decimal;

static int tests{0};
static int fails{0};

union u {
  float x;
  std::uint32_t u;
};

llvm::raw_ostream &failed(float x) {
  ++fails;
  union u u;
  u.x = x;
  llvm::outs() << "FAIL: 0x";
  return llvm::outs().write_hex(u.u);
}

void testDirect(float x, const char *expect, int expectExpo, int flags = 0) {
  char buffer[1024];
  ++tests;
  auto result{ConvertFloatToDecimal(buffer, sizeof buffer,
      static_cast<enum DecimalConversionFlags>(flags), 1024, RoundNearest, x)};
  if (result.str == nullptr) {
    failed(x) << ' ' << flags << ": no result str\n";
  } else if (std::strcmp(result.str, expect) != 0 ||
      result.decimalExponent != expectExpo) {
    failed(x) << ' ' << flags << ": expect '." << expect << 'e' << expectExpo
              << "', got '." << result.str << 'e' << result.decimalExponent
              << "'\n";
  }
}

void testReadback(float x, int flags) {
  char buffer[1024];
  ++tests;
  auto result{ConvertFloatToDecimal(buffer, sizeof buffer,
      static_cast<enum DecimalConversionFlags>(flags), 1024, RoundNearest, x)};
  if (result.str == nullptr) {
    failed(x) << ' ' << flags << ": no result str\n";
  } else {
    float y{0};
    char *q{const_cast<char *>(result.str)};
    int expo{result.decimalExponent};
    expo -= result.length;
    if (*q == '-' || *q == '+') {
      ++expo;
    }
    if (q >= buffer && q < buffer + sizeof buffer) {
      std::sprintf(q + result.length, "e%d", expo);
    }
    const char *p{q};
    auto rflags{ConvertDecimalToFloat(&p, &y, RoundNearest)};
    union u u;
    if (!(x == x)) {
      if (y == y || *p != '\0' || (rflags & Invalid)) {
        u.x = y;
        failed(x) << " (NaN) " << flags << ": -> '" << result.str << "' -> 0x";
        failed(x).write_hex(u.u) << " '" << p << "' " << rflags << '\n';
      }
    } else if (x != y || *p != '\0' || (rflags & Invalid)) {
      u.x = y;
      failed(x) << ' ' << flags << ": -> '" << result.str << "' -> 0x";
      failed(x).write_hex(u.u) << " '" << p << "' " << rflags << '\n';
    }
  }
}

int main() {
  union u u;
  testDirect(-1.0, "-1", 1);
  testDirect(0.0, "0", 0);
  testDirect(0.0, "+0", 0, AlwaysSign);
  testDirect(1.0, "1", 1);
  testDirect(2.0, "2", 1);
  testDirect(-1.0, "-1", 1);
  testDirect(314159, "314159", 6);
  testDirect(0.0625, "625", -1);
  u.u = 0x80000000;
  testDirect(u.x, "-0", 0);
  u.u = 0x7f800000;
  testDirect(u.x, "Inf", 0);
  testDirect(u.x, "+Inf", 0, AlwaysSign);
  u.u = 0xff800000;
  testDirect(u.x, "-Inf", 0);
  u.u = 0xffffffff;
  testDirect(u.x, "NaN", 0);
  testDirect(u.x, "NaN", 0, AlwaysSign);
  u.u = 1;
  testDirect(u.x,
      "140129846432481707092372958328991613128026194187651577175706828388979108"
      "268586060148663818836212158203125",
      -44, 0);
  testDirect(u.x, "1", -44, Minimize);
  u.u = 0x7f777777;
  testDirect(u.x, "3289396118917826996438159226753253376", 39, 0);
  testDirect(u.x, "32893961", 39, Minimize);
  for (u.u = 0; u.u < 16; ++u.u) {
    testReadback(u.x, 0);
    testReadback(-u.x, 0);
    testReadback(u.x, Minimize);
    testReadback(-u.x, Minimize);
  }
  for (u.u = 1; u.u < 0x7f800000; u.u *= 2) {
    testReadback(u.x, 0);
    testReadback(-u.x, 0);
    testReadback(u.x, Minimize);
    testReadback(-u.x, Minimize);
  }
  for (u.u = 0x7f7ffff0; u.u < 0x7f800010; ++u.u) {
    testReadback(u.x, 0);
    testReadback(-u.x, 0);
    testReadback(u.x, Minimize);
    testReadback(-u.x, Minimize);
  }
  for (u.u = 0; u.u < 0x7f800000; u.u += 65536) {
    testReadback(u.x, 0);
    testReadback(-u.x, 0);
    testReadback(u.x, Minimize);
    testReadback(-u.x, Minimize);
  }
  for (u.u = 0; u.u < 0x7f800000; u.u += 99999) {
    testReadback(u.x, 0);
    testReadback(-u.x, 0);
    testReadback(u.x, Minimize);
    testReadback(-u.x, Minimize);
  }
  for (u.u = 0; u.u < 0x7f800000; u.u += 32767) {
    testReadback(u.x, 0);
    testReadback(-u.x, 0);
    testReadback(u.x, Minimize);
    testReadback(-u.x, Minimize);
  }
  llvm::outs() << tests << " tests run, " << fails << " tests failed\n";
  return fails > 0;
}
