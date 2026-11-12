//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++03 || c++11 || c++14 || c++17

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// reference_wrapper

// check that binder typedefs exit

#include <functional>
#include <type_traits>

struct UnaryFunction
{
    typedef long argument_type;
    typedef char result_type;
};

struct BinaryFunction
{
    typedef int first_argument_type;
    typedef char second_argument_type;
    typedef long result_type;
};

static_assert(std::is_same<std::reference_wrapper<int(UnaryFunction::*)()>::result_type, int>::value, "");
static_assert(std::is_same<std::reference_wrapper<int(UnaryFunction::*)()>::argument_type, UnaryFunction*>::value, "");

static_assert(std::is_same<std::reference_wrapper<int(BinaryFunction::*)(char)>::result_type, int>::value, "");
static_assert(std::is_same<std::reference_wrapper<int(BinaryFunction::*)(char)>::first_argument_type, BinaryFunction*>::value, "");
static_assert(std::is_same<std::reference_wrapper<int(BinaryFunction::*)(char)>::second_argument_type, char>::value, "");

static_assert(std::is_same<std::reference_wrapper<void(*)()>::result_type, void>::value, "");
