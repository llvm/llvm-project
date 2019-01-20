//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class locale::id
// {
// public:
//     id();
//     void operator=(const id&) = delete;
//     id(const id&) = delete;
// };

// This test isn't portable

#include <locale>
#include <cassert>

std::locale::id id0;
std::locale::id id2;
std::locale::id id1;

int main()
{
    long id = id0.__get();
    assert(id0.__get() == id+0);
    assert(id0.__get() == id+0);
    assert(id0.__get() == id+0);
    assert(id1.__get() == id+1);
    assert(id1.__get() == id+1);
    assert(id1.__get() == id+1);
    assert(id2.__get() == id+2);
    assert(id2.__get() == id+2);
    assert(id2.__get() == id+2);
    assert(id0.__get() == id+0);
    assert(id0.__get() == id+0);
    assert(id0.__get() == id+0);
    assert(id1.__get() == id+1);
    assert(id1.__get() == id+1);
    assert(id1.__get() == id+1);
    assert(id2.__get() == id+2);
    assert(id2.__get() == id+2);
    assert(id2.__get() == id+2);
}
