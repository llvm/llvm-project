//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<Returnable R, CopyConstructible... ArgTypes>
// class function<R(ArgTypes...)> {
// public:
//   typedef R result_type;
//   typedef T1 argument_type;          // iff sizeof...(ArgTypes) == 1 and
//                                      // the type in ArgTypes is T1
//   typedef T1 first_argument_type;    // iff sizeof...(ArgTypes) == 2 and
//                                      // ArgTypes contains T1 and T2
//   typedef T2 second_argument_type;   // iff sizeof...(ArgTypes) == 2 and
//                                      // ArgTypes contains T1 and T2
//  ...
//  };

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <type_traits>

#include "test_macros.h"


template <typename T>
class has_argument_type
{
    typedef char yes;
    typedef long no;

    template <typename C> static yes check( typename C::argument_type * );
    template <typename C> static no  check(...);
public:
    enum { value = sizeof(check<T>(0)) == sizeof(yes) };
};

template <typename T>
class has_first_argument_type
{
    typedef char yes;
    typedef long no;

    template <typename C> static yes check( typename C::first_argument_type * );
    template <typename C> static no  check(...);
public:
    enum { value = sizeof(check<T>(0)) == sizeof(yes) };
};


template <typename T>
class has_second_argument_type
{
    typedef char yes;
    typedef long no;

    template <typename C> static yes check( typename C::second_argument_type *);
    template <typename C> static no  check(...);
public:
    enum { value = sizeof(check<T>(0)) == sizeof(yes) };
};

template <class F, class return_type>
void test_nullary_function ()
{
    static_assert((std::is_same<typename F::result_type, return_type>::value), "" );
    static_assert((!has_argument_type<F>::value), "" );
    static_assert((!has_first_argument_type<F>::value), "" );
    static_assert((!has_second_argument_type<F>::value), "" );
}

template <class F, class return_type, class arg_type>
void test_unary_function ()
{
    static_assert((std::is_same<typename F::result_type, return_type>::value), "" );
    static_assert((std::is_same<typename F::argument_type,  arg_type>::value), "" );
    static_assert((!has_first_argument_type<F>::value), "" );
    static_assert((!has_second_argument_type<F>::value), "" );
}

template <class F, class return_type, class arg_type1, class arg_type2>
void test_binary_function ()
{
    static_assert((std::is_same<typename F::result_type,        return_type>::value), "" );
    static_assert((std::is_same<typename F::first_argument_type,  arg_type1>::value), "" );
    static_assert((std::is_same<typename F::second_argument_type, arg_type2>::value), "" );
    static_assert((!has_argument_type<F>::value), "" );
}

template <class F, class return_type>
void test_other_function ()
{
    static_assert((std::is_same<typename F::result_type, return_type>::value), "" );
    static_assert((!has_argument_type<F>::value), "" );
    static_assert((!has_first_argument_type<F>::value), "" );
    static_assert((!has_second_argument_type<F>::value), "" );
}

int main(int, char**)
{
    test_nullary_function<std::function<int()>, int>();
    test_unary_function  <std::function<double(int)>, double, int>();
    test_binary_function <std::function<double(int, char)>, double, int, char>();
    test_other_function  <std::function<double(int, char, double)>, double>();

  return 0;
}
