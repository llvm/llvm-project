//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// http://wg21.link/LWG2447 gives implementors freedom to reject const or volatile types in `std::allocator`.

#include <memory>

std::allocator<const int> A1; // expected-error@*:* {{std::allocator does not support const types}}
std::allocator<volatile int> A2; // expected-error@*:* {{std::allocator does not support volatile types}}
