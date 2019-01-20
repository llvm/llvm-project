//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <ObjectType T> T* addressof(T&& r) = delete;

#include <memory>
#include <cassert>

#include "test_macros.h"

int main()
{
#if TEST_STD_VER > 14
    const int *p = std::addressof<const int>(0);
#else
#error
#endif
}
