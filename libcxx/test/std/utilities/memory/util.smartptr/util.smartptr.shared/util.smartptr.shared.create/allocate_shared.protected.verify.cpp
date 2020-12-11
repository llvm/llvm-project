//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// GCC 5 does not evaluate static assertions dependent on a template parameter.
// UNSUPPORTED: gcc-5

// <memory>

// shared_ptr

// template<class T, class A, class... Args>
//    shared_ptr<T> allocate_shared(const A& a, Args&&... args);

#include <memory>

#include "test_macros.h"

struct S {
protected:
   S () {};  // ctor is protected
};

int main(int, char**) {
    typedef std::allocator<S> A;
    A a;
    std::shared_ptr<S> p = std::allocate_shared<S, A>(a); // expected-error@memory:* {{static_assert failed due to requirement 'is_constructible<S>::value}}

    return 0;
}
