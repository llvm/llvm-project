//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class T> class shared_ptr
// {
// public:
//     typedef T element_type;
//     typedef weak_ptr<T> weak_type; // C++17
//     ...
// };

#include <memory>

#include "test_macros.h"

struct A;  // purposefully incomplete

int main()
{
    static_assert((std::is_same<std::shared_ptr<A>::element_type, A>::value), "");
#if TEST_STD_VER > 14
    static_assert((std::is_same<std::shared_ptr<A>::weak_type, std::weak_ptr<A>>::value), "");
#endif
}
