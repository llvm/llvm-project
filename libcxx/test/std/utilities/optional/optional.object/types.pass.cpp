//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// template <class T>
// class optional
// {
// public:
//     typedef T value_type;
//     ...

#include <optional>
#include <type_traits>

using std::optional;

template <class Opt, class T>
void
test()
{
    static_assert(std::is_same<typename Opt::value_type, T>::value, "");
}

int main()
{
    test<optional<int>, int>();
    test<optional<const int>, const int>();
    test<optional<double>, double>();
    test<optional<const double>, const double>();
}
