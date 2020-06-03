//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// UNSUPPORTED: clang-4.0
// UNSUPPORTED: c++03

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX17_REMOVED_BINDERS

#include <functional>
#include <cassert>
#include "test_macros.h"

int identity(int v) { return v; }
int sum(int a, int b) { return a + b; }

struct Foo {
    int const_zero() const { return 0; }
    int const_identity(int v) const { return v; }
    int zero() { return 0; }
    int identity(int v) { return v; }
};

int main(int, char**)
{
    typedef std::pointer_to_unary_function<int, int> PUF; // expected-warning {{'pointer_to_unary_function<int, int>' is deprecated}}
    typedef std::pointer_to_binary_function<int, int, int> PBF; // expected-warning {{'pointer_to_binary_function<int, int, int>' is deprecated}}
    std::ptr_fun<int, int>(identity); // expected-warning {{'ptr_fun<int, int>' is deprecated}}
    std::ptr_fun<int, int, int>(sum); // expected-warning {{'ptr_fun<int, int, int>' is deprecated}}

    typedef std::mem_fun_t<int, Foo> MFT0; // expected-warning {{'mem_fun_t<int, Foo>' is deprecated}}
    typedef std::mem_fun1_t<int, Foo, int> MFT1; // expected-warning {{'mem_fun1_t<int, Foo, int>' is deprecated}}
    typedef std::const_mem_fun_t<int, Foo> CMFT0; // expected-warning {{'const_mem_fun_t<int, Foo>' is deprecated}}
    typedef std::const_mem_fun1_t<int, Foo, int> CMFT1; // expected-warning {{'const_mem_fun1_t<int, Foo, int>' is deprecated}}
    std::mem_fun<int, Foo>(&Foo::zero); // expected-warning {{'mem_fun<int, Foo>' is deprecated}}
    std::mem_fun<int, Foo, int>(&Foo::identity); // expected-warning {{'mem_fun<int, Foo, int>' is deprecated}}
    std::mem_fun<int, Foo>(&Foo::const_zero); // expected-warning {{'mem_fun<int, Foo>' is deprecated}}
    std::mem_fun<int, Foo, int>(&Foo::const_identity); // expected-warning {{'mem_fun<int, Foo, int>' is deprecated}}

    typedef std::mem_fun_ref_t<int, Foo> MFR0; // expected-warning {{'mem_fun_ref_t<int, Foo>' is deprecated}}
    typedef std::mem_fun1_ref_t<int, Foo, int> MFR1; // expected-warning {{'mem_fun1_ref_t<int, Foo, int>' is deprecated}}
    typedef std::const_mem_fun_ref_t<int, Foo> CMFR0; // expected-warning {{'const_mem_fun_ref_t<int, Foo>' is deprecated}}
    typedef std::const_mem_fun1_ref_t<int, Foo, int> CMFR1; // expected-warning {{'const_mem_fun1_ref_t<int, Foo, int>' is deprecated}}
    std::mem_fun_ref<int, Foo>(&Foo::zero); // expected-warning {{'mem_fun_ref<int, Foo>' is deprecated}}
    std::mem_fun_ref<int, Foo, int>(&Foo::identity); // expected-warning {{'mem_fun_ref<int, Foo, int>' is deprecated}}
    std::mem_fun_ref<int, Foo>(&Foo::const_zero); // expected-warning {{'mem_fun_ref<int, Foo>' is deprecated}}
    std::mem_fun_ref<int, Foo, int>(&Foo::const_identity); // expected-warning {{'mem_fun_ref<int, Foo, int>' is deprecated}}

  return 0;
}
