//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// <experimental/system_error>

#include <experimental/system_error>

// expected-error@experimental/system_error:* {{"<experimental/system_error> has been removed. Use <system_error> instead."}}

int main(int, char**) {
  return 0;
}
