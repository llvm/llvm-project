//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// <experimental/any>

#include <experimental/any>

// expected-error@experimental/any:* {{"<experimental/any> has been removed. Use <any> instead."}}

int main() {}
