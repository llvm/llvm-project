//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// explicit forward_list(const allocator_type& a);

#include <forward_list>
#include <cassert>

#include "test_allocator.h"
#include "../../../NotConstructible.h"

int main()
{
    {
        typedef test_allocator<NotConstructible> A;
        typedef A::value_type T;
        typedef std::forward_list<T, A> C;
        C c = A(12);
        assert(c.get_allocator() == A(12));
        assert(c.empty());
    }
}
