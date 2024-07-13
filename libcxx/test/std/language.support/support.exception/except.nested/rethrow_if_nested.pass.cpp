//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// FIXME: This test fails in MSVC mode due to a stack overflow
// XFAIL: msvc

// <exception>

// class nested_exception;

// template <class E> void rethrow_if_nested(const E& e);

#include <exception>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <utility>

#include "test_macros.h"

class A
{
    int data_;
public:
    explicit A(int data) : data_(data) {}
    A(const A&) = default;
    A& operator=(const A&) = default;
    virtual ~A() TEST_NOEXCEPT {}

    friend bool operator==(const A& x, const A& y) {return x.data_ == y.data_;}
};

class B
    : public std::nested_exception,
      public A
{
public:
    explicit B(int data) : A(data) {}
    B(const B& b) : A(b) {}
};

class C
{
public:
    virtual ~C() {}
    C * operator&() const { assert(false); return nullptr; } // should not be called
};

class D : private std::nested_exception {};


class E1 : public std::nested_exception {};
class E2 : public std::nested_exception {};
class E : public E1, public E2 {};

#if TEST_STD_VER >= 11
template <class, class...>
struct can_rethrow_if_nested_impl {
  static constexpr bool value = false;
};

template <class... Args>
struct can_rethrow_if_nested_impl<decltype((void)std::rethrow_if_nested(std::declval<Args>()...)), Args...> {
  static constexpr bool value = true;
};

template <class... Args>
struct can_rethrow_if_nested : can_rethrow_if_nested_impl<void, Args...> {};

static_assert(!can_rethrow_if_nested<>::value, "");
static_assert(can_rethrow_if_nested<A>::value, "");
static_assert(can_rethrow_if_nested<const A&>::value, "");
static_assert(can_rethrow_if_nested<B>::value, "");
static_assert(can_rethrow_if_nested<const B&>::value, "");
static_assert(!can_rethrow_if_nested<A, int*>::value, "");
static_assert(!can_rethrow_if_nested<B, int*>::value, "");
static_assert(!can_rethrow_if_nested<A, std::nullptr_t>::value, "");
static_assert(!can_rethrow_if_nested<B, std::nullptr_t>::value, "");
#endif

int main(int, char**)
{
    {
        try
        {
            A a(3);  // not a polymorphic type --> no effect
            std::rethrow_if_nested(a);
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }
    {
        try
        {
            D s;  // inaccessible base class --> no effect
            std::rethrow_if_nested(s);
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }
    {
        try
        {
            E s;  // ambiguous base class --> no effect
            std::rethrow_if_nested(s);
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }
    {
        try
        {
            throw B(5);
        }
        catch (const B& b)
        {
            try
            {
                throw b;
            }
            catch (const A& a)
            {
                try
                {
                    std::rethrow_if_nested(a);
                    assert(false);
                }
                catch (const B& b2)
                {
                    assert(b2 == B(5));
                }
            }
        }
    }
    {
        try
        {
            std::rethrow_if_nested(C());
            assert(true);
        }
        catch (...)
        {
            assert(false);
        }
    }


  return 0;
}
