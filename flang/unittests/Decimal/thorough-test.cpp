#include "flang/Decimal/decimal.h"
#include "llvm/Support/raw_ostream.h"
#include <cinttypes>
#include <cstdio>
#include <cstring>

static constexpr int incr{1}; // steps through all values
static constexpr bool doNegative{true};
static constexpr bool doMinimize{true};

using namespace Fortran::decimal;

static std::uint64_t tests{0};
static std::uint64_t fails{0};

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

void testReadback(float x, int flags) {
  char buffer[1024];
  union u u;
  u.x = x;
  if (!(tests & 0x3fffff)) {
    llvm::errs() << "\n0x";
    llvm::errs().write_hex(u.u) << ' ';
  } else if (!(tests & 0xffff)) {
    llvm::errs() << '.';
  }
  ++tests;
  auto result{ConvertFloatToDecimal(buffer, sizeof buffer,
      static_cast<enum DecimalConversionFlags>(flags), 1024, RoundNearest, x)};
  if (result.str == nullptr) {
    failed(x) << ' ' << flags << ": no result str\n";
  } else {
    float y{0};
    char *q{const_cast<char *>(result.str)};
    if ((*q >= '0' && *q <= '9') ||
        ((*q == '-' || *q == '+') && q[1] >= '0' && q[1] <= '9')) {
      int expo{result.decimalExponent};
      expo -= result.length;
      if (*q == '-' || *q == '+') {
        ++expo;
      }
      std::snprintf(q + result.length,
          buffer + sizeof buffer - (q + result.length), "e%d", expo);
    }
    const char *p{q};
    auto rflags{ConvertDecimalToFloat(&p, &y, RoundNearest)};
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
  for (u.u = 0; u.u < 0x7f800010; u.u += incr) {
    testReadback(u.x, 0);
    if constexpr (doNegative) {
      testReadback(-u.x, 0);
    }
    if constexpr (doMinimize) {
      testReadback(u.x, Minimize);
      if constexpr (doNegative) {
        testReadback(-u.x, Minimize);
      }
    }
  }
  llvm::outs() << '\n' << tests << " tests run, " << fails << " tests failed\n";
  return fails > 0;
}
