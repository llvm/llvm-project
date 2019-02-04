//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T>
// struct pointer_traits<T*>
// {
//     typedef T element_type;
//     ...
// };

#include <memory>
#include <type_traits>

int main(int, char**)
{
    static_assert((std::is_same<std::pointer_traits<const short*>::element_type, const short>::value), "");

  return 0;
}
