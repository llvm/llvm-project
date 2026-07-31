//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can override the assertion semantic used by any hardening mode with `ignore` on a per-TU
// basis (this is valid for the `none` mode as well, though a no-op).

// UNSUPPORTED: libcpp-has-no-experimental-hardening-observe-semantic
// assertion semantics require libc++ and C++11
// UNSUPPORTED: c++03
// REQUIRES: stdlib=libc++
// ADDITIONAL_COMPILE_FLAGS: -U_LIBCPP_ASSERTION_SEMANTIC -D_LIBCPP_ASSERTION_SEMANTIC=_LIBCPP_ASSERTION_SEMANTIC_IGNORE

#include <vector> // pulls in the valid element access assertion

int main(int, char**) {
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(true, "Should not fire");
  _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(false, "Also should not fire");

  return 0;
}
