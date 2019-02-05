//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// UNSUPPORTED: c++98, c++03
// REQUIRES: verify-support

// MODULES_DEFINES: _LIBCPP_ENABLE_DEPRECATION_WARNINGS
// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR
#define _LIBCPP_ENABLE_DEPRECATION_WARNINGS
#define _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR

#include <memory>
#include "test_macros.h"

int main()
{
    typedef std::auto_ptr<int> AP; // expected-error{{'auto_ptr<int>' is deprecated}}
    typedef std::auto_ptr<void> APV; // expected-error{{'auto_ptr<void>' is deprecated}}
    typedef std::auto_ptr_ref<int> APR; // expected-error{{'auto_ptr_ref<int>' is deprecated}}
}
