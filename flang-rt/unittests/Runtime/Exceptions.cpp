//===-- unittests/Runtime/Exceptions.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Tests for the Fortran runtime fenv-wrapping entry points
/// (feclearexcept / feraiseexcept / fetestexcept).
///
/// These checks also guard against an optimizer eliding the wrapped libm
/// calls: under clang's default `-ffp-exception-behavior=ignore`, the
/// compiler is free to drop calls to fenv functions, which would silently
/// break ieee_arithmetic flag handling.  exceptions.cpp uses
/// `FLANG_FP_TRAP_ON` to disable that optimization; if it were ever removed
/// or weakened, the round-trip assertions below would fail.
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/exceptions.h"
#include "flang/Common/fp-control.h"
#include <cfenv>
#include <cstdint>
#include <gtest/gtest.h>

using namespace Fortran::runtime;

namespace {

// Saves and restores the host floating-point environment so tests do not
// leak exception flags into each other or into the rest of the suite.
class FenvScope {
public:
  FenvScope() {
    FLANG_FP_TRAP_ON
    std::fegetenv(&saved_);
  }
  ~FenvScope() {
    FLANG_FP_TRAP_ON
    std::fesetenv(&saved_);
  }

private:
  std::fenv_t saved_;
};

// Bitmask of every standard exception this platform defines.  C99 lets
// some be absent (e.g. on musl/emscripten); the runtime itself guards
// each one, so the tests mirror that.
constexpr std::uint32_t kAllSupported = 0u
#ifdef FE_INVALID
    | FE_INVALID
#endif
#ifdef FE_DIVBYZERO
    | FE_DIVBYZERO
#endif
#ifdef FE_OVERFLOW
    | FE_OVERFLOW
#endif
#ifdef FE_UNDERFLOW
    | FE_UNDERFLOW
#endif
#ifdef FE_INEXACT
    | FE_INEXACT
#endif
    ;

} // namespace

TEST(Exceptions, ClearLeavesAllUnset) {
  FenvScope guard;
  if (kAllSupported == 0u) {
    GTEST_SKIP() << "no FE_* exceptions defined on this platform";
  }
  RTNAME(feraiseexcept)(kAllSupported);
  RTNAME(feclearexcept)(kAllSupported);
  EXPECT_EQ(RTNAME(fetestexcept)(kAllSupported), 0u);
}

TEST(Exceptions, RaiseSetsExactlyRequestedFlag) {
  FenvScope guard;
  RTNAME(feclearexcept)(kAllSupported);

  // Walk every supported exception bit individually.  After raising one,
  // fetestexcept(that bit) must report it set; fetestexcept(other bits)
  // must not spuriously light up.
  const std::uint32_t bits[] = {
#ifdef FE_INVALID
      static_cast<std::uint32_t>(FE_INVALID),
#endif
#ifdef FE_DIVBYZERO
      static_cast<std::uint32_t>(FE_DIVBYZERO),
#endif
#ifdef FE_OVERFLOW
      static_cast<std::uint32_t>(FE_OVERFLOW),
#endif
#ifdef FE_UNDERFLOW
      static_cast<std::uint32_t>(FE_UNDERFLOW),
#endif
#ifdef FE_INEXACT
      static_cast<std::uint32_t>(FE_INEXACT),
#endif
  };
  if (sizeof(bits) == 0) {
    GTEST_SKIP() << "no FE_* exceptions defined on this platform";
  }

  // C99 7.6.2.3 p2 leaves it implementation-defined whether feraiseexcept
  // additionally raises FE_INEXACT when raising FE_OVERFLOW or FE_UNDERFLOW.
  // AArch64 glibc does, so exclude FE_INEXACT from the "no spurious flag"
  // mask to keep the test portable.
  constexpr std::uint32_t kInexactTolerated =
#ifdef FE_INEXACT
      static_cast<std::uint32_t>(FE_INEXACT);
#else
      0u;
#endif

  for (std::uint32_t bit : bits) {
    RTNAME(feclearexcept)(kAllSupported);
    RTNAME(feraiseexcept)(bit);
    EXPECT_NE(RTNAME(fetestexcept)(bit), 0u) << "bit 0x" << std::hex << bit;
    const std::uint32_t forbidden = kAllSupported & ~bit & ~kInexactTolerated;
    EXPECT_EQ(RTNAME(fetestexcept)(forbidden), 0u)
        << "spurious flag set when raising 0x" << std::hex << bit;
  }
}

TEST(Exceptions, ClearOneLeavesOthersAlone) {
  FenvScope guard;
#if defined(FE_OVERFLOW) && defined(FE_INVALID)
  RTNAME(feclearexcept)(kAllSupported);
  RTNAME(feraiseexcept)(FE_OVERFLOW | FE_INVALID);
  ASSERT_NE(RTNAME(fetestexcept)(FE_OVERFLOW), 0u);
  ASSERT_NE(RTNAME(fetestexcept)(FE_INVALID), 0u);

  RTNAME(feclearexcept)(FE_OVERFLOW);
  EXPECT_EQ(RTNAME(fetestexcept)(FE_OVERFLOW), 0u);
  EXPECT_NE(RTNAME(fetestexcept)(FE_INVALID), 0u);
#else
  GTEST_SKIP() << "FE_OVERFLOW and FE_INVALID required for this test";
#endif
}
