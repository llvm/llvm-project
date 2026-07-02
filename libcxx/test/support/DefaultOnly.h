//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DEFAULTONLY_H
#define DEFAULTONLY_H

#include <cassert>
#include <type_traits> // std::is_constant_evaluated

#include "test_macros.h"

class DefaultOnly
{
    int data_;

    DefaultOnly(const DefaultOnly&);
    DefaultOnly& operator=(const DefaultOnly&);
public:
    static int count;

    TEST_CONSTEXPR_CXX20 DefaultOnly() : data_(-1) {
      if (!TEST_IS_CONSTANT_EVALUATED)
        ++count;
    }
    TEST_CONSTEXPR_CXX20 ~DefaultOnly() {
      data_ = 0;
      if (!TEST_IS_CONSTANT_EVALUATED)
        --count;
    }

    friend TEST_CONSTEXPR_CXX20 bool operator==(const DefaultOnly& x, const DefaultOnly& y) {
      return x.data_ == y.data_;
    }
    friend TEST_CONSTEXPR_CXX20 bool operator<(const DefaultOnly& x, const DefaultOnly& y) { return x.data_ < y.data_; }
};

int DefaultOnly::count = 0;

#endif // DEFAULTONLY_H
