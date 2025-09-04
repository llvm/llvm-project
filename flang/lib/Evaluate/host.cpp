//===-- lib/Evaluate/host.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "host.h"

#include "flang/Common/idioms.h"
#include "llvm/Support/Errno.h"
#include <cfenv>
#if __x86_64__
#include <xmmintrin.h>
#endif

namespace Fortran::evaluate::host {
using namespace Fortran::parser::literals;

void HostFloatingPointEnvironment::SetUpHostFloatingPointEnvironment(
    FoldingContext &context) {
  errno = 0;
  std::fenv_t currentFenv;
  if (feholdexcept(&originalFenv_) != 0) {
    common::die("Folding with host runtime: feholdexcept() failed: %s",
        llvm::sys::StrError(errno).c_str());
    return;
  }
  if (fegetenv(&currentFenv) != 0) {
    common::die("Folding with host runtime: fegetenv() failed: %s",
        llvm::sys::StrError(errno).c_str());
    return;
  }
#if __x86_64__
  hasSubnormalFlushingHardwareControl_ = true;
  originalMxcsr = _mm_getcsr();
  unsigned int currentMxcsr{originalMxcsr};
  if (context.targetCharacteristics().areSubnormalsFlushedToZero()) {
    currentMxcsr |= 0x8000;
    currentMxcsr |= 0x0040;
  } else {
    currentMxcsr &= ~0x8000;
    currentMxcsr &= ~0x0040;
  }
#elif defined(__aarch64__)
#if defined(__GNU_LIBRARY__)
  hasSubnormalFlushingHardwareControl_ = true;
  if (context.targetCharacteristics().areSubnormalsFlushedToZero()) {
    currentFenv.__fpcr |= (1U << 24); // control register
  } else {
    currentFenv.__fpcr &= ~(1U << 24); // control register
  }
#elif defined(__BIONIC__)
  hasSubnormalFlushingHardwareControl_ = true;
  if (context.targetCharacteristics().areSubnormalsFlushedToZero()) {
    currentFenv.__control |= (1U << 24); // control register
  } else {
    currentFenv.__control &= ~(1U << 24); // control register
  }
#else
  // If F18 is built with other C libraries on AArch64, software flushing will
  // be performed around host library calls if subnormal flushing is requested
#endif
#else
  // If F18 is not built on one of the above host architecture, software
  // flushing will be performed around host library calls if needed.
#endif

#ifdef __clang__
  // clang does not ensure that floating point environment flags are meaningful.
  // It may perform optimizations that will impact the floating point
  // environment. For instance, libc++ complex float tan and tanh compilation
  // with clang -O2 introduces a division by zero on X86 in unused slots of xmm
  // registers. Therefore, fetestexcept should not be used.
  hardwareFlagsAreReliable_ = false;
#endif
  errno = 0;
  if (fesetenv(&currentFenv) != 0) {
    common::die("Folding with host runtime: fesetenv() failed: %s",
        llvm::sys::StrError(errno).c_str());
    return;
  }
#if __x86_64__
  _mm_setcsr(currentMxcsr);
#endif

  switch (context.targetCharacteristics().roundingMode().mode) {
  case common::RoundingMode::TiesToEven:
    fesetround(FE_TONEAREST);
    break;
  case common::RoundingMode::ToZero:
    fesetround(FE_TOWARDZERO);
    break;
  case common::RoundingMode::Up:
    fesetround(FE_UPWARD);
    break;
  case common::RoundingMode::Down:
    fesetround(FE_DOWNWARD);
    break;
  case common::RoundingMode::TiesAwayFromZero:
    fesetround(FE_TONEAREST);
    context.Warn(common::UsageWarning::FoldingFailure,
        "TiesAwayFromZero rounding mode is not available when folding constants with host runtime; using TiesToEven instead"_warn_en_US);
    break;
  }
  flags_.clear();
  errno = 0;
}
void HostFloatingPointEnvironment::CheckAndRestoreFloatingPointEnvironment(
    FoldingContext &context) {
  int errnoCapture{errno};
  if (hardwareFlagsAreReliable()) {
    int exceptions{fetestexcept(FE_ALL_EXCEPT)};
    if (exceptions & FE_INVALID) {
      flags_.set(RealFlag::InvalidArgument);
    }
    if (exceptions & FE_DIVBYZERO) {
      flags_.set(RealFlag::DivideByZero);
    }
    if (exceptions & FE_OVERFLOW) {
      flags_.set(RealFlag::Overflow);
    }
    if (exceptions & FE_UNDERFLOW) {
      flags_.set(RealFlag::Underflow);
    }
    if (exceptions & FE_INEXACT) {
      flags_.set(RealFlag::Inexact);
    }
  }

  if (flags_.empty()) {
    if (errnoCapture == EDOM) {
      flags_.set(RealFlag::InvalidArgument);
    }
    if (errnoCapture == ERANGE) {
      // can't distinguish over/underflow from errno
      flags_.set(RealFlag::Overflow);
    }
  }

  if (!flags_.empty()) {
    RealFlagWarnings(
        context, flags_, "evaluation of intrinsic function or operation");
  }
  errno = 0;
  if (fesetenv(&originalFenv_) != 0) {
    std::fprintf(
        stderr, "fesetenv() failed: %s\n", llvm::sys::StrError(errno).c_str());
    common::die(
        "Folding with host runtime: fesetenv() failed while restoring fenv: %s",
        llvm::sys::StrError(errno).c_str());
  }
#if __x86_64__
  _mm_setcsr(originalMxcsr);
#endif

  errno = 0;
}
} // namespace Fortran::evaluate::host
