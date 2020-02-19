#include "fp-testing.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using Fortran::common::RoundingMode;
using Fortran::evaluate::RealFlag;

ScopedHostFloatingPointEnvironment::ScopedHostFloatingPointEnvironment(
#if __x86_64__
    bool treatSubnormalOperandsAsZero, bool flushSubnormalResultsToZero
#else
    bool, bool
#endif
) {
  errno = 0;
  if (feholdexcept(&originalFenv_) != 0) {
    std::fprintf(stderr, "feholdexcept() failed: %s\n", std::strerror(errno));
    std::abort();
  }
  if (fegetenv(&currentFenv_) != 0) {
    std::fprintf(stderr, "fegetenv() failed: %s\n", std::strerror(errno));
    std::abort();
  }
#if __x86_64__
  if (treatSubnormalOperandsAsZero) {
    currentFenv_.__mxcsr |= 0x0040;
  } else {
    currentFenv_.__mxcsr &= ~0x0040;
  }
  if (flushSubnormalResultsToZero) {
    currentFenv_.__mxcsr |= 0x8000;
  } else {
    currentFenv_.__mxcsr &= ~0x8000;
  }
#else
  // TODO others
#endif
  errno = 0;
  if (fesetenv(&currentFenv_) != 0) {
    std::fprintf(stderr, "fesetenv() failed: %s\n", std::strerror(errno));
    std::abort();
  }
}

ScopedHostFloatingPointEnvironment::~ScopedHostFloatingPointEnvironment() {
  errno = 0;
  if (fesetenv(&originalFenv_) != 0) {
    std::fprintf(stderr, "fesetenv() failed: %s\n", std::strerror(errno));
    std::abort();
  }
}

void ScopedHostFloatingPointEnvironment::ClearFlags() const {
  feclearexcept(FE_ALL_EXCEPT);
}

RealFlags ScopedHostFloatingPointEnvironment::CurrentFlags() {
  int exceptions = fetestexcept(FE_ALL_EXCEPT);
  RealFlags flags;
  if (exceptions & FE_INVALID) {
    flags.set(RealFlag::InvalidArgument);
  }
  if (exceptions & FE_DIVBYZERO) {
    flags.set(RealFlag::DivideByZero);
  }
  if (exceptions & FE_OVERFLOW) {
    flags.set(RealFlag::Overflow);
  }
  if (exceptions & FE_UNDERFLOW) {
    flags.set(RealFlag::Underflow);
  }
  if (exceptions & FE_INEXACT) {
    flags.set(RealFlag::Inexact);
  }
  return flags;
}

void ScopedHostFloatingPointEnvironment::SetRounding(Rounding rounding) {
  switch (rounding.mode) {
  case RoundingMode::TiesToEven: fesetround(FE_TONEAREST); break;
  case RoundingMode::ToZero: fesetround(FE_TOWARDZERO); break;
  case RoundingMode::Up: fesetround(FE_UPWARD); break;
  case RoundingMode::Down: fesetround(FE_DOWNWARD); break;
  case RoundingMode::TiesAwayFromZero:
    std::fprintf(stderr, "SetRounding: TiesAwayFromZero not available");
    std::abort();
    break;
  }
}
