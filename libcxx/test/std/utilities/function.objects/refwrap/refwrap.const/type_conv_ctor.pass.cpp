//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&) noexcept(see below);

#include <functional>
#include <cassert>

struct convertible_to_int_ref {
    int val = 0;
    operator int&() { return val; }
    operator int const&() const { return val; }
};

template <bool IsNothrow>
struct nothrow_convertible {
    int val = 0;
    operator int&() noexcept(IsNothrow) { return val; }
};

struct convertible_from_int {
    convertible_from_int(int) {}
};

void meow(std::reference_wrapper<int>) {}
void meow(convertible_from_int) {}

int gi;
std::reference_wrapper<int> purr() { return gi; };

template <class T>
void
test(T& t)
{
    std::reference_wrapper<T> r(t);
    assert(&r.get() == &t);
}

void f() {}

int main()
{
    convertible_to_int_ref convi;
    test(convi);
    convertible_to_int_ref const convic;
    test(convic);

    {
    using Ref = std::reference_wrapper<int>;
    static_assert((std::is_nothrow_constructible<Ref, nothrow_convertible<true>>::value), "");
    static_assert((!std::is_nothrow_constructible<Ref, nothrow_convertible<false>>::value), "");
    }

    {
    meow(0);
    (true) ? purr() : 0;
    }

#ifdef __cpp_deduction_guides
    {
    int i = 0;
    std::reference_wrapper ri(i);
    static_assert((std::is_same<decltype(ri), std::reference_wrapper<int>>::value), "" );
    const int j = 0;
    std::reference_wrapper rj(j);
    static_assert((std::is_same<decltype(rj), std::reference_wrapper<const int>>::value), "" );
    }
#endif
}
