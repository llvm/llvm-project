//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class Fn, class T>
//   binder1st<Fn>
//   bind1st(const Fn& fn, const T& x);

#include <functional>
#include <cassert>

#include "../test_func.h"

int main()
{
    assert(std::bind1st(test_func(1), 5)(10.) == -5.);
}
