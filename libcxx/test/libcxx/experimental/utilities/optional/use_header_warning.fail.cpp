//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// <experimental/optional>

#include <experimental/optional>

// expected-error@experimental/optional:* {{"<experimental/optional> has been removed. Use <optional> instead."}}

int main() {}
