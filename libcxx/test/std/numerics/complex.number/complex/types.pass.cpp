//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>
// class complex
// {
// public:
//   typedef T value_type;
//   ...
// };

#include <complex>
#include <type_traits>

template <class T>
void
test()
{
    typedef std::complex<T> C;
    static_assert((std::is_same<typename C::value_type, T>::value), "");
}

int main(int, char**)
{
    test<float>();
    test<double>();
    test<long double>();

  return 0;
}
