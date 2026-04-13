//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: We should consolidate all the helper types used in the optional tests into here.

#include "test_macros.h"

#ifndef LIBCXX_UTILITIES_OPTIONAL_HELPER_TYPES_H
#  define LIBCXX_UTILITIES_OPTIONAL_HELPER_TYPES_H

template <typename T>
struct ReferenceConversion {
  T lvalue;
  T rvalue;

  constexpr ReferenceConversion(T lval, T rval) : lvalue(lval), rvalue(rval) {}

  constexpr operator T&() & noexcept { return lvalue; }

  constexpr operator T&() && noexcept { return rvalue; }
};

template <typename T>
struct ReferenceConversionThrows {
  T lvalue;
  T rvalue;
  bool throws{false};

  constexpr ReferenceConversionThrows(T lval, T rval, bool except = false)
      : lvalue(lval), rvalue(rval), throws(except) {}

  constexpr operator T&() & {
    if (throws) {
      TEST_THROW(1);
    }

    return lvalue;
  }

  constexpr operator T&() && {
    if (throws) {
      TEST_THROW(2);
    }

    return rvalue;
  }
};

#endif
