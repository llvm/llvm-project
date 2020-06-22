//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cassert>
#include <unordered_set>

// <unordered_set>

// bool contains(const key_type& x) const;

template <typename T, typename V, typename B, typename... Vals>
void test(B bad, Vals... args) {
    T set;
    V vals[] = {args...};

    for (auto& v : vals) set.insert(v);
    for (auto& v : vals) assert(set.contains(v));

    assert(!set.contains(bad));
}

struct E { int a = 1; double b = 1; char c = 1; };

int main(int, char**)
{
    {
        test<std::unordered_set<int>, int>(14, 10, 11, 12, 13);
        test<std::unordered_set<char>, char>('e', 'a', 'b', 'c', 'd');
    }
    {
        test<std::unordered_multiset<int>, int>(14, 10, 11, 12, 13);
        test<std::unordered_multiset<char>, char>('e', 'a', 'b', 'c', 'd');
    }

    return 0;
}
