//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// template <MoveConstructible R, MoveConstructible ... ArgTypes>
//   bool operator==(const function<R(ArgTypes...)>&, nullptr_t);
//
// template <MoveConstructible R, MoveConstructible ... ArgTypes>
//   bool operator==(nullptr_t, const function<R(ArgTypes...)>&);
//
// template <MoveConstructible R, MoveConstructible ... ArgTypes>
//   bool operator!=(const function<R(ArgTypes...)>&, nullptr_t);
//
// template <MoveConstructible  R, MoveConstructible ... ArgTypes>
//   bool operator!=(nullptr_t, const function<R(ArgTypes...)>&);

#include <functional>
#include <cassert>

#include "test_macros.h"

int g(int) {return 0;}

int main(int, char**)
{
    {
    std::function<int(int)> f;
    assert(f == nullptr);
    assert(nullptr == f);
    f = g;
    assert(f != nullptr);
    assert(nullptr != f);
    }

  return 0;
}
