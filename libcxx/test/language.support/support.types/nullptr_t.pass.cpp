//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <type_traits>
#include <cassert>

// typedef decltype(nullptr) nullptr_t;

struct A
{
    A(std::nullptr_t) {}
};

int main()
{
    static_assert(sizeof(std::nullptr_t) == sizeof(void*),
                  "sizeof(std::nullptr_t) == sizeof(void*)");
    A* p = 0;
    assert(p == nullptr);
    void (A::*pmf)() = 0;
#ifdef __clang__
    // GCC 4.2 can't handle this
    assert(pmf == nullptr);
#endif
    int A::*pmd = 0;
    assert(pmd == nullptr);
    A a1(nullptr);
    A a2(0);
    bool b = nullptr;
    assert(!b);
    assert(nullptr == nullptr);
    assert(nullptr <= nullptr);
    assert(nullptr >= nullptr);
    assert(!(nullptr != nullptr));
    assert(!(nullptr < nullptr));
    assert(!(nullptr > nullptr));
    A* a = nullptr;
    assert(a == nullptr);
    assert(a <= nullptr);
    assert(a >= nullptr);
    assert(!(a != nullptr));
    assert(!(a < nullptr));
    assert(!(a > nullptr));
    assert(nullptr == a);
    assert(nullptr <= a);
    assert(nullptr >= a);
    assert(!(nullptr != a));
    assert(!(nullptr < a));
    assert(!(nullptr > a));
    std::ptrdiff_t i = reinterpret_cast<std::ptrdiff_t>(nullptr);
    assert(i == 0);
}
