//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for llvm/llvm-project#195149: u16string::find used to
// dispatch through __builtin_wmemchr when sizeof(char16_t) == sizeof(wchar_t).
// Under -fshort-wchar on a platform whose native wchar_t is 4 bytes
// (e.g., Linux/Darwin), the libc wmemchr keeps walking 4-byte elements, so the
// search returned wrong results. __find now gates the wmemchr fast path on the
// platform-native wchar_t size (via __WCHAR_NATIVE_TYPE__) so the runtime
// libcall is taken only when it is binary-compatible with what wmemchr expects.
//
// Only meaningful where the platform-native wchar_t differs from 2 bytes; on
// Windows (native 2-byte wchar_t) the optimization is always safe.

// ADDITIONAL_COMPILE_FLAGS: -fshort-wchar

#include <cassert>
#include <string>

#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool test() {
  std::u16string s = u"hello";
  std::u16string t = u"goodbye";
  assert(s.find(u'o') == 4);
  assert(t.find(u'b') == 4);
  assert(s.find(u'z') == std::u16string::npos);
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif
  return 0;
}
