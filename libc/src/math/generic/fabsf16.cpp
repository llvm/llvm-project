//===-- Implementation of fabsf16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fabsf16.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/macros/properties/compiler.h"
#include "src/__support/macros/properties/cpu_features.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, fabsf16, (float16 x)) {
#if defined(__LIBC_MISC_MATH_BASIC_OPS_OPT) &&                                 \
    defined(LIBC_TARGET_CPU_HAS_FAST_FLOAT16_OPS)
  return __builtin_fabsf16(x);
#elif defined(LIBC_TARGET_ARCH_IS_X86) && defined(LIBC_COMPILER_IS_CLANG)
  // Prevent Clang from generating calls to slow soft-float conversion
  // functions on x86. See https://godbolt.org/z/hvo6jbnGz.

  using FPBits = fputil::FPBits<float16>;
  using StorageType = typename FPBits::StorageType;

  static constexpr volatile StorageType ABS_MASK = FPBits::EXP_SIG_MASK;

  return FPBits(static_cast<StorageType>(FPBits(x).uintval() & ABS_MASK))
      .get_val();
#else
  return fputil::abs(x);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
