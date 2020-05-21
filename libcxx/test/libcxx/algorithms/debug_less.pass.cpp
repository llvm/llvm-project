//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <algorithm>

// template <class _Compare> struct __debug_less

// __debug_less checks that a comparator actually provides a strict-weak ordering.

#include <chrono> // Include before defining _LIBCPP_ASSERT: cannot throw in a function marked noexcept.

struct DebugException {};

#ifdef _LIBCPP_ASSERT
#undef _LIBCPP_ASSERT
#endif
#define _LIBCPP_DEBUG 0
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : throw ::DebugException())

#include <algorithm>
#include <iterator>
#include <cassert>

#include "test_macros.h"

template <int ID>
struct MyType {
    int value;
    explicit MyType(int xvalue = 0) : value(xvalue) {}
};

template <int ID1, int ID2>
bool operator<(MyType<ID1> const& LHS, MyType<ID2> const& RHS) {
    return LHS.value < RHS.value;
}

struct CompareBase {
    static int called;
    static void reset() {
        called = 0;
    }
};

int CompareBase::called = 0;

template <class ValueType>
struct GoodComparator : public CompareBase {
    bool operator()(ValueType const& lhs, ValueType const& rhs) const {
        ++CompareBase::called;
        return lhs < rhs;
    }
};

template <class ValueType>
struct BadComparator : public CompareBase {
    bool operator()(ValueType const&, ValueType const&) const {
        ++CompareBase::called;
        return true;
    }
};

template <class T1, class T2>
struct TwoWayHomoComparator : public CompareBase {
    bool operator()(T1 const& lhs, T2 const& rhs) const {
        ++CompareBase::called;
        return lhs < rhs;
    }

    bool operator()(T2 const& lhs, T1 const& rhs) const {
        ++CompareBase::called;
        return lhs < rhs;
    }
};

template <class T1, class T2>
struct OneWayHomoComparator : public CompareBase {
    bool operator()(T1 const& lhs, T2 const& rhs) const {
        ++CompareBase::called;
        return lhs < rhs;
    }
};

using std::__debug_less;

typedef MyType<0> MT0;
typedef MyType<1> MT1;

void test_passing() {
    int& called = CompareBase::called;
    called = 0;
    MT0 one(1);
    MT0 two(2);
    MT1 three(3);
    MT1 four(4);

    {
        typedef GoodComparator<MT0> C;
        typedef __debug_less<C> D;

        C c;
        D d(c);

        assert(d(one, two) == true);
        assert(called == 2);
        called = 0;

        assert(d(one, one) == false);
        assert(called == 1);
        called = 0;

        assert(d(two, one) == false);
        assert(called == 1);
        called = 0;
    }
    {
        typedef TwoWayHomoComparator<MT0, MT1> C;
        typedef __debug_less<C> D;
        C c;
        D d(c);

        assert(d(one, three) == true);
        assert(called == 2);
        called = 0;

        assert(d(three, one) == false);
        assert(called == 1);
        called = 0;
    }
    {
        typedef OneWayHomoComparator<MT0, MT1> C;
        typedef __debug_less<C> D;
        C c;
        D d(c);

        assert(d(one, three) == true);
        assert(called == 1);
        called = 0;
    }
}

void test_failing() {
    int& called = CompareBase::called;
    called = 0;
    MT0 one(1);
    MT0 two(2);

    {
        typedef BadComparator<MT0> C;
        typedef __debug_less<C> D;
        C c;
        D d(c);

        try {
            d(one, two);
            assert(false);
        } catch (DebugException const&) {
        }

        assert(called == 2);
        called = 0;
    }
}

template <int>
struct Tag {
  explicit Tag(int v) : value(v) {}
  int value;
};

template <class = void>
struct FooImp {
  explicit FooImp(int x) : x_(x) {}
  int x_;
};

template <class T>
inline bool operator<(FooImp<T> const& x, Tag<0> y) {
    return x.x_ < y.value;
}

template <class T>
inline bool operator<(Tag<0>, FooImp<T> const&) {
    static_assert(sizeof(FooImp<T>) != sizeof(FooImp<T>), "should not be instantiated");
    return false;
}

template <class T>
inline bool operator<(Tag<1> x, FooImp<T> const& y) {
    return x.value < y.x_;
}

template <class T>
inline bool operator<(FooImp<T> const&, Tag<1>) {
    static_assert(sizeof(FooImp<T>) != sizeof(FooImp<T>), "should not be instantiated");
    return false;
}

typedef FooImp<> Foo;

// Test that we don't attempt to call the comparator with the arguments reversed
// for upper_bound and lower_bound since the comparator or type is not required
// to support it, nor does it require the range to have a strict weak ordering.
// See llvm.org/PR39458
void test_upper_and_lower_bound() {
    Foo table[] = {Foo(1), Foo(2), Foo(3), Foo(4), Foo(5)};
    {
        Foo* iter = std::lower_bound(std::begin(table), std::end(table), Tag<0>(3));
        assert(iter == (table + 2));
    }
    {
        Foo* iter = std::upper_bound(std::begin(table), std::end(table), Tag<1>(3));
        assert(iter == (table + 3));
    }
}

struct NonConstArgCmp {
    bool operator()(int& x, int &y) const {
        return x < y;
    }
};

void test_non_const_arg_cmp() {
    {
        NonConstArgCmp cmp;
        __debug_less<NonConstArgCmp> dcmp(cmp);
        int x = 0, y = 1;
        assert(dcmp(x, y));
        assert(!dcmp(y, x));
    }
    {
        NonConstArgCmp cmp;
        int arr[] = {5, 4, 3, 2, 1};
        std::sort(std::begin(arr), std::end(arr), cmp);
        assert(std::is_sorted(std::begin(arr), std::end(arr)));
    }
}

struct ValueIterator {
    typedef std::input_iterator_tag iterator_category;
    typedef size_t value_type;
    typedef ptrdiff_t difference_type;
    typedef size_t reference;
    typedef size_t* pointer;

    ValueIterator() { }

    reference operator*() { return 0; }
    ValueIterator& operator++() { return *this; }

    friend bool operator==(ValueIterator, ValueIterator) { return true; }
    friend bool operator!=(ValueIterator, ValueIterator) { return false; }
};

void test_value_iterator() {
    // Ensure no build failures when iterators return values, not references.
    assert(0 == std::lexicographical_compare(ValueIterator(), ValueIterator(),
                                             ValueIterator(), ValueIterator()));
}

void test_value_categories() {
    std::less<int> l;
    std::__debug_less<std::less<int> > dl(l);
    int lvalue = 42;
    const int const_lvalue = 101;

    assert(dl(lvalue, const_lvalue));
    assert(dl(/*rvalue*/1, lvalue));
    assert(dl(static_cast<int&&>(1), static_cast<const int&&>(2)));
}

#if TEST_STD_VER > 17
constexpr bool test_constexpr() {
    std::less<> cmp{};
    __debug_less<std::less<> > dcmp(cmp);
    assert(dcmp(1, 2));
    assert(!dcmp(1, 1));
    return true;
}
#endif

int main(int, char**) {
    test_passing();
    test_failing();
    test_upper_and_lower_bound();
    test_non_const_arg_cmp();
    test_value_iterator();
    test_value_categories();
#if TEST_STD_VER > 17
    static_assert(test_constexpr(), "");
#endif
    return 0;
}
