//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// UNSUPPORTED: clang-4.0
// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_DEPRECATION_WARNINGS
// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS
#define _LIBCPP_ENABLE_DEPRECATION_WARNINGS
#define _LIBCPP_ENABLE_CXX17_REMOVED_BINDERS

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

int main()
{
#if TEST_STD_VER < 11
    // expected-no-diagnostics
#else
    // expected-error@* 1 {{'pointer_to_unary_function<int, int>' is deprecated}}
    // expected-error@* 1 {{'pointer_to_binary_function<int, int, int>' is deprecated}}
    // expected-error@* 1 {{'ptr_fun<int, int>' is deprecated}}
    // expected-error@* 1 {{'ptr_fun<int, int, int>' is deprecated}}

    // expected-error@* 1 {{'mem_fun_t<int, Foo>' is deprecated}}
    // expected-error@* 1 {{'mem_fun1_t<int, Foo, int>' is deprecated}}
    // expected-error@* 1 {{'const_mem_fun_t<int, Foo>' is deprecated}}
    // expected-error@* 1 {{'const_mem_fun1_t<int, Foo, int>' is deprecated}}
    // expected-error@* 2 {{'mem_fun<int, Foo>' is deprecated}}
    // expected-error@* 2 {{'mem_fun<int, Foo, int>' is deprecated}}

    // expected-error@* 1 {{'mem_fun_ref_t<int, Foo>' is deprecated}}
    // expected-error@* 1 {{'mem_fun1_ref_t<int, Foo, int>' is deprecated}}
    // expected-error@* 1 {{'const_mem_fun_ref_t<int, Foo>' is deprecated}}
    // expected-error@* 1 {{'const_mem_fun1_ref_t<int, Foo, int>' is deprecated}}
    // expected-error@* 2 {{'mem_fun_ref<int, Foo>' is deprecated}}
    // expected-error@* 2 {{'mem_fun_ref<int, Foo, int>' is deprecated}}
#endif
    typedef std::pointer_to_unary_function<int, int> PUF;
    typedef std::pointer_to_binary_function<int, int, int> PBF;
    std::ptr_fun<int, int>(identity);
    std::ptr_fun<int, int, int>(sum);

    typedef std::mem_fun_t<int, Foo> MFT0;
    typedef std::mem_fun1_t<int, Foo, int> MFT1;
    typedef std::const_mem_fun_t<int, Foo> CMFT0;
    typedef std::const_mem_fun1_t<int, Foo, int> CMFT1;
    std::mem_fun<int, Foo>(&Foo::zero);
    std::mem_fun<int, Foo, int>(&Foo::identity);
    std::mem_fun<int, Foo>(&Foo::const_zero);
    std::mem_fun<int, Foo, int>(&Foo::const_identity);

    typedef std::mem_fun_ref_t<int, Foo> MFR0;
    typedef std::mem_fun1_ref_t<int, Foo, int> MFR1;
    typedef std::const_mem_fun_ref_t<int, Foo> CMFR0;
    typedef std::const_mem_fun1_ref_t<int, Foo, int> CMFR1;
    std::mem_fun_ref<int, Foo>(&Foo::zero);
    std::mem_fun_ref<int, Foo, int>(&Foo::identity);
    std::mem_fun_ref<int, Foo>(&Foo::const_zero);
    std::mem_fun_ref<int, Foo, int>(&Foo::const_identity);
}
