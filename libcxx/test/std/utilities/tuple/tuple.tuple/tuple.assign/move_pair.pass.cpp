//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class U1, class U2>
//   tuple& operator=(pair<U1, U2>&& u);

// UNSUPPORTED: c++03

#include <tuple>
#include <utility>
#include <memory>
#include <cassert>

#include "test_macros.h"

struct B
{
    int id_;

    explicit B(int i = 0) : id_(i) {}

    virtual ~B() {}
};

struct D
    : B
{
    explicit D(int i) : B(i) {}
};

int main(int, char**)
{
    {
        typedef std::pair<long, std::unique_ptr<D>> T0;
        typedef std::tuple<long long, std::unique_ptr<B>> T1;
        T0 t0(2, std::unique_ptr<D>(new D(3)));
        T1 t1;
        t1 = std::move(t0);
        assert(std::get<0>(t1) == 2);
        assert(std::get<1>(t1)->id_ == 3);
    }

  return 0;
}
