//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// typedef decltype(nullptr) nullptr_t;

#include <cstddef>

int main()
{
    std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nullptr);
}
