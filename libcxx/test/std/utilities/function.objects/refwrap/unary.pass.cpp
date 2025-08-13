//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03 || c++11 || c++14

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// reference_wrapper

// check for deriving from unary_function

#include <functional>
#include <type_traits>

#include "test_macros.h"

class functor1
    : public std::unary_function<int, char>
{
};

class functor2
    : public std::binary_function<char, int, double>
{
};

class functor3
    : public std::unary_function<int, int>,
      public std::binary_function<char, int, double>
{
public:
    typedef float result_type;
};

class functor4
    : public std::unary_function<int, int>,
      public std::binary_function<char, int, double>
{
public:
};

struct C
{
    typedef int argument_type;
    typedef int result_type;
};

int main(int, char**)
{
    static_assert((std::is_base_of<std::unary_function<int, char>,
                                   std::reference_wrapper<functor1> >::value), "");
    static_assert((!std::is_base_of<std::unary_function<char, int>,
                                    std::reference_wrapper<functor2> >::value), "");
    static_assert((std::is_base_of<std::unary_function<int, int>,
                                   std::reference_wrapper<functor3> >::value), "");
    static_assert((std::is_base_of<std::unary_function<int, int>,
                                   std::reference_wrapper<functor4> >::value), "");
    static_assert((!std::is_base_of<std::unary_function<int, int>,
                                    std::reference_wrapper<C> >::value), "");
    static_assert((!std::is_base_of<std::unary_function<int, float>,
                                    std::reference_wrapper<float(*)()> >::value), "");
    static_assert((std::is_base_of<std::unary_function<int, float>,
                                   std::reference_wrapper<float (int)> >::value), "");
    static_assert((!std::is_base_of<std::unary_function<int, float>,
                                    std::reference_wrapper<float (int, int)> >::value), "");
    static_assert((std::is_base_of<std::unary_function<int, float>,
                                   std::reference_wrapper<float(*)(int)> >::value), "");
    static_assert((!std::is_base_of<std::unary_function<int, float>,
                                    std::reference_wrapper<float(*)(int, int)> >::value), "");
    static_assert((std::is_base_of<std::unary_function<C*, float>,
                                   std::reference_wrapper<float(C::*)()> >::value), "");
    static_assert((std::is_base_of<std::unary_function<const volatile C*, float>,
                                   std::reference_wrapper<float(C::*)() const volatile> >::value), "");
    static_assert((!std::is_base_of<std::unary_function<C*, float>,
                                   std::reference_wrapper<float(C::*)(int)> >::value), "");

  return 0;
}
