//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

#include <vector>

#include "test_macros.h"

struct Incomplete;
template<class T> struct Holder { T t; };

template<class T, class AdlTrap = Holder<Incomplete>>
struct MyAlloc {
    using value_type = T;
    MyAlloc() = default;
    template<class U> MyAlloc(const MyAlloc<U>&) {}
    T *allocate(int n) { return std::allocator<T>().allocate(n); }
    void deallocate(T *p, int n) { return std::allocator<T>().deallocate(p, n); }
};

int main(int, char**)
{
    std::vector<bool, MyAlloc<bool>> vb;
    std::vector<bool, MyAlloc<bool>> wb(100);

    std::vector<int, MyAlloc<int>> v;
    std::vector<int, MyAlloc<int>> w(100);
    v.push_back(1);
    v.insert(v.end(), 2);
    v.insert(v.end(), w.begin(), w.end());
    v.pop_back();
    v.erase(v.begin());
    v.erase(v.begin(), v.end());
#if TEST_STD_VER >= 14
    v.swap(w);
#endif
    return 0;
}
