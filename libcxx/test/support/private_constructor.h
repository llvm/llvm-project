//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_PRIVATE_CONSTRUCTOR_H
#define TEST_SUPPORT_PRIVATE_CONSTRUCTOR_H

#include "test_macros.h"

struct PrivateConstructor {
  TEST_CONSTEXPR_CXX26 PrivateConstructor static make(int v) { return PrivateConstructor(v); }
  TEST_CONSTEXPR_CXX26 int get() const { return val; }

private:
  TEST_CONSTEXPR_CXX26 PrivateConstructor(int v) : val(v) {}
  int val;
    };

    TEST_CONSTEXPR_CXX26 bool operator<(const PrivateConstructor& lhs, const PrivateConstructor& rhs) {
      return lhs.get() < rhs.get();
    }

    TEST_CONSTEXPR_CXX26 bool operator<(const PrivateConstructor& lhs, int rhs) { return lhs.get() < rhs; }
    TEST_CONSTEXPR_CXX26 bool operator<(int lhs, const PrivateConstructor& rhs) { return lhs < rhs.get(); }

#endif // TEST_SUPPORT_PRIVATE_CONSTRUCTOR_H
