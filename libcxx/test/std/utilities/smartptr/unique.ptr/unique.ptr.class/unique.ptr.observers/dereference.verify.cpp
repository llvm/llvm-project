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

#include <memory>

#include "test_macros.h"

void f() {
  std::unique_ptr<int[]> p(new int(3));
  const std::unique_ptr<int[]>& cp = p;
  TEST_IGNORE_NODISCARD(*p);  // expected-error-re {{indirection requires pointer operand ('std::unique_ptr<int{{[ ]*}}[]>' invalid)}}
  TEST_IGNORE_NODISCARD(*cp); // expected-error-re {{indirection requires pointer operand ('const std::unique_ptr<int{{[ ]*}}[]>' invalid)}}
}
