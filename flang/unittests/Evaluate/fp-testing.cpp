#include "fp-testing.h"
#include "llvm/Support/Errno.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if __x86_64__
#include <xmmintrin.h>
#endif

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
    std::fprintf(stderr, "feholdexcept() failed: %s\n",
        llvm::sys::StrError(errno).c_str());
    std::abort();
  }
  fenv_t currentFenv;
  if (fegetenv(&currentFenv) != 0) {
    std::fprintf(
        stderr, "fegetenv() failed: %s\n", llvm::sys::StrError(errno).c_str());
    std::abort();
  }

#if __x86_64__
  originalMxcsr = _mm_getcsr();
  unsigned int currentMxcsr{originalMxcsr};
  if (treatSubnormalOperandsAsZero) {
    currentMxcsr |= 0x0040;
  } else {
    currentMxcsr &= ~0x0040;
  }
  if (flushSubnormalResultsToZero) {
    currentMxcsr |= 0x8000;
  } else {
    currentMxcsr &= ~0x8000;
  }
#else
  // TODO others
#endif
  errno = 0;
  if (fesetenv(&currentFenv) != 0) {
    std::fprintf(
        stderr, "fesetenv() failed: %s\n", llvm::sys::StrError(errno).c_str());
    std::abort();
  }
#if __x86_64__
  _mm_setcsr(currentMxcsr);
#endif
}

ScopedHostFloatingPointEnvironment::~ScopedHostFloatingPointEnvironment() {
  errno = 0;
  if (fesetenv(&originalFenv_) != 0) {
    std::fprintf(
        stderr, "fesetenv() failed: %s\n", llvm::sys::StrError(errno).c_str());
    std::abort();
  }
#if __x86_64__
  _mm_setcsr(originalMxcsr);
#endif
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
  case RoundingMode::TiesToEven:
    fesetround(FE_TONEAREST);
    break;
  case RoundingMode::ToZero:
    fesetround(FE_TOWARDZERO);
    break;
  case RoundingMode::Up:
    fesetround(FE_UPWARD);
    break;
  case RoundingMode::Down:
    fesetround(FE_DOWNWARD);
    break;
  case RoundingMode::TiesAwayFromZero:
    std::fprintf(stderr, "SetRounding: TiesAwayFromZero not available");
    std::abort();
    break;
  }
}
