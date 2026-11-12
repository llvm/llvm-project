//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// template<class Engine, size_t k>
// class shuffle_order_engine

// explicit shuffle_order_engine(const Engine& e);

#include <random>

#include <cassert>
#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::minstd_rand0 Engine;
        typedef std::knuth_b Adaptor;
        Engine e;
        Engine e0 = e;
        Adaptor a(std::move(e0));
        for (unsigned k = 0; k <= Adaptor::table_size; ++k) {
            (void)e();
        }

        assert(a.base() == e);
    }

  return 0;
}
