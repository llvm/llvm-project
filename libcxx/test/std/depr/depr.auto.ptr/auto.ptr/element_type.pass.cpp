//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X>
// class auto_ptr
// {
// public:
//   typedef X element_type;
//   ...
// };

// REQUIRES: c++98 || c++03 || c++11 || c++14

#include <memory>
#include <type_traits>

template <class T>
void
test()
{
    static_assert((std::is_same<typename std::auto_ptr<T>::element_type, T>::value), "");
    std::auto_ptr<T> p;
    ((void)p);
}

int main()
{
    test<int>();
    test<double>();
    test<void>();
}
