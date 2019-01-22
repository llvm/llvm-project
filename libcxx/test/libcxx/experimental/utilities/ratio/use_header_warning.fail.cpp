//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// <experimental/ratio>

#include <experimental/ratio>

// expected-error@experimental/ratio:* {{"<experimental/ratio> has been removed. Use <ratio> instead."}}

int main() {}
