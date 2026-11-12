//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op*()

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <memory>

#include "test_macros.h"

void test() {
  std::unique_ptr<int[]> p(new int(3));
  const std::unique_ptr<int[]>& cp = p;
  TEST_IGNORE_NODISCARD(*p);  // expected-error-re {{indirection requires pointer operand ('std::unique_ptr<int{{[ ]*}}[]>' invalid)}}
  TEST_IGNORE_NODISCARD(*cp); // expected-error-re {{indirection requires pointer operand ('const std::unique_ptr<int{{[ ]*}}[]>' invalid)}}
}

void test_lwg4196() {
  struct Deleter {
    using pointer = long*;

    void operator()(pointer) const {}
  };

  std::unique_ptr<const int, Deleter> p;

  // expected-error-re@*:* {{static assertion failed due to requirement {{.+}}The returned reference must not bind to a temporary object.}}
#if TEST_STD_VER >= 26
  // expected-error@*:* {{returning reference to local temporary object}}
#endif
  (void)*p;
}
