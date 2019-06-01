//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X>
// class auto_ptr;
//
//  In C++17, auto_ptr has been removed.
//  However, for backwards compatibility, if _LIBCPP_NO_REMOVE_AUTOPTR
//  is defined before including <memory>, then auto_ptr will be restored.

// MODULES_DEFINES: _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR
#define _LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR
#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    std::auto_ptr<int> p;

  return 0;
}
