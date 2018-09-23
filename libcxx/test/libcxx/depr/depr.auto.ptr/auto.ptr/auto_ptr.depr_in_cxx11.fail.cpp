//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>
//
// template <class X>
// class auto_ptr;
//
// class auto_ptr<void>;
//
// template <class X>
// class auto_ptr_ref;
//
// Deprecated in C++11

// UNSUPPORTED: clang-4.0
// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_DEPRECATION_WARNINGS
// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR
#define _LIBCPP_ENABLE_DEPRECATION_WARNINGS
#define _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR

#include <memory>
#include "test_macros.h"

int main()
{
#if TEST_STD_VER < 11
    // expected-no-diagnostics
#else
    // expected-error@* 1 {{'auto_ptr<int>' is deprecated}}
    // expected-error@* 1 {{'auto_ptr<void>' is deprecated}}
    // expected-error@* 1 {{'auto_ptr_ref<int>' is deprecated}}
#endif
    typedef std::auto_ptr<int> AP;
    typedef std::auto_ptr<void> APV;
    typedef std::auto_ptr_ref<int> APR;
}
