//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class T> class weak_ptr
// {
// public:
//     typedef T element_type;
//     ...
// };

#include <memory>

struct A;  // purposefully incomplete

int main()
{
    static_assert((std::is_same<std::weak_ptr<A>::element_type, A>::value), "");
}
