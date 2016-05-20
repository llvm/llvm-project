//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test forward

#include <utility>
#include <cassert>

struct A
{
};

A source() {return A();}
const A csource() {return A();}

typedef char one;
struct two {one _[2];};
struct four {one _[4];};
struct eight {one _[8];};

one test(A&);
two test(const A&);

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

four test(A&&);
eight test(const A&&);

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
    A a;
    const A ca = A();

    ((void)a); // Prevent unused warning
    ((void)ca); // Prevent unused warning

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert(sizeof(test(std::forward<A&>(a))) == 1, "");
    static_assert(sizeof(test(std::forward<A>(a))) == 4, "");
    static_assert(sizeof(test(std::forward<A>(source()))) == 4, "");

    static_assert(sizeof(test(std::forward<const A&>(a))) == 2, "");
//    static_assert(sizeof(test(std::forward<const A&>(source()))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(a))) == 8, "");
    static_assert(sizeof(test(std::forward<const A>(source()))) == 8, "");

    static_assert(sizeof(test(std::forward<const A&>(ca))) == 2, "");
//    static_assert(sizeof(test(std::forward<const A&>(csource()))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(ca))) == 8, "");
    static_assert(sizeof(test(std::forward<const A>(csource()))) == 8, "");

#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

    static_assert(sizeof(test(std::forward<A&>(a))) == 1, "");
    static_assert(sizeof(test(std::forward<A>(a))) == 1, "");
//    static_assert(sizeof(test(std::forward<A>(source()))) == 2, "");

    static_assert(sizeof(test(std::forward<const A&>(a))) == 2, "");
    static_assert(sizeof(test(std::forward<const A&>(source()))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(a))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(source()))) == 2, "");

    static_assert(sizeof(test(std::forward<const A&>(ca))) == 2, "");
    static_assert(sizeof(test(std::forward<const A&>(csource()))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(ca))) == 2, "");
    static_assert(sizeof(test(std::forward<const A>(csource()))) == 2, "");
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

#if _LIBCPP_STD_VER > 11
    constexpr int i1 = std::move(23);
    static_assert(i1 == 23, "" );
    constexpr int i2 = std::forward<int>(42);
    static_assert(i2 == 42, "" );
#endif
}
