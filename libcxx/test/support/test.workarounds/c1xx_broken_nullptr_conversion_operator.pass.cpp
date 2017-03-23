//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// Verify TEST_WORKAROUND_C1XX_BROKEN_NULLPTR_CONVERSION_OPERATOR.

#include "test_workarounds.h"

#include <type_traits>

struct ConvertsToNullptr {
  using DestType = decltype(nullptr);
  operator DestType() const { return nullptr; }
};

int main() {
#if defined(TEST_WORKAROUND_C1XX_BROKEN_NULLPTR_CONVERSION_OPERATOR)
  static_assert(!std::is_convertible<ConvertsToNullptr, decltype(nullptr)>::value, "");
#else
  static_assert(std::is_convertible<ConvertsToNullptr, decltype(nullptr)>::value, "");
#endif
}
