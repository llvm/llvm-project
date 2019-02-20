//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// <experimental/string_view>

#include <experimental/string_view>

// expected-error@experimental/string_view:* {{"<experimental/string_view> has been removed. Use <string_view> instead."}}

int main(int, char**) {
  return 0;
}
