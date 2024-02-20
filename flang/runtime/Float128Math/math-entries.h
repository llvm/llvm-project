//===-- runtime/Float128Math/math-entries.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_FLOAT128MATH_MATH_ENTRIES_H_
#define FORTRAN_RUNTIME_FLOAT128MATH_MATH_ENTRIES_H_
#include "terminator.h"
#include "tools.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/entry-names.h"
#include <type_traits>

namespace Fortran::runtime {

// Define a class template to gracefully fail, when
// there is no specialized template that implements
// the required function via using the third-party
// implementation.
#define DEFINE_FALLBACK(caller) \
  template <auto F> struct caller { \
    template <typename... ATs> \
    [[noreturn]] static std::invoke_result_t<decltype(F), ATs...> invoke( \
        ATs... args) { \
      Terminator terminator{__FILE__, __LINE__}; \
      terminator.Crash("Float128 variant of '%s' is unsupported", #caller); \
    } \
  };

// Define template specialization that is calling the third-party
// implementation. The template is specialized by a function pointer
// that is the FortranFloat128Math entry point. The signatures
// of the caller and the callee must match.
//
// Defining the specialization for any target library requires
// adding the generic template via DEFINE_FALLBACK, so that
// a build with another target library that does not define
// the same alias can gracefully fail in runtime.
#define DEFINE_SIMPLE_ALIAS(caller, callee) \
  template <typename RT, typename... ATs, RT (*p)(ATs...)> struct caller<p> { \
    static RT invoke(ATs... args) { \
      static_assert(std::is_invocable_r_v<RT, decltype(callee), ATs...>); \
      if constexpr (std::is_same_v<RT, void>) { \
        callee(args...); \
      } else { \
        return callee(args...); \
      } \
    } \
  };

// Define fallback callers.
DEFINE_FALLBACK(CAbs)
DEFINE_FALLBACK(Sin)
DEFINE_FALLBACK(Sqrt)

// Define ComplexF128 type that is compatible with
// the type of results/arguments of libquadmath.
// TODO: this may need more work for other libraries/compilers.
#if !defined(_ARCH_PPC) || defined(__LONG_DOUBLE_IEEE128__)
typedef _Complex float __attribute__((mode(TC))) ComplexF128;
#else
typedef _Complex float __attribute__((mode(KC))) ComplexF128;
#endif

#if HAS_QUADMATHLIB
// Define wrapper callers for libquadmath.
#include "quadmath.h"
DEFINE_SIMPLE_ALIAS(CAbs, cabsq)
DEFINE_SIMPLE_ALIAS(Sin, sinq)
DEFINE_SIMPLE_ALIAS(Sqrt, sqrtq)
#endif
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_FLOAT128MATH_MATH_ENTRIES_H_
