//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// template <class ValueType>
// ValueType const any_cast(any const&);
//
// template <class ValueType>
// ValueType any_cast(any &);
//
// template <class ValueType>
// ValueType any_cast(any &&);

#include <any>
#include <type_traits>
#include <cassert>

#include "any_helpers.h"
#include "count_new.h"
#include "test_macros.h"

// Test that the operators are NOT marked noexcept.
void test_cast_is_not_noexcept() {
    std::any a;
    static_assert(!noexcept(std::any_cast<int>(static_cast<std::any&>(a))), "");
    static_assert(!noexcept(std::any_cast<int>(static_cast<std::any const&>(a))), "");
    static_assert(!noexcept(std::any_cast<int>(static_cast<std::any &&>(a))), "");
}

// Test that the return type of any_cast is correct.
void test_cast_return_type() {
    std::any a;
    static_assert(std::is_same<decltype(std::any_cast<int>(a)), int>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const>(a)), int>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int&>(a)), int&>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const&>(a)), int const&>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int&&>(a)), int&&>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const&&>(a)), int const&&>::value, "");

    static_assert(std::is_same<decltype(std::any_cast<int>(std::move(a))), int>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const>(std::move(a))), int>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int&>(std::move(a))), int&>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const&>(std::move(a))), int const&>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int&&>(std::move(a))), int&&>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const&&>(std::move(a))), int const&&>::value, "");

    const std::any& ca = a;
    static_assert(std::is_same<decltype(std::any_cast<int>(ca)), int>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const>(ca)), int>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const&>(ca)), int const&>::value, "");
    static_assert(std::is_same<decltype(std::any_cast<int const&&>(ca)), int const&&>::value, "");
}

template <class Type, class ConstT = Type>
void checkThrows(std::any& a)
{
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    try {
        TEST_IGNORE_NODISCARD std::any_cast<Type>(a);
        assert(false);
    } catch (const std::bad_any_cast&) {
        // do nothing
    } catch (...) {
        assert(false);
    }

    try {
        TEST_IGNORE_NODISCARD std::any_cast<ConstT>(static_cast<const std::any&>(a));
        assert(false);
    } catch (const std::bad_any_cast&) {
        // do nothing
    } catch (...) {
        assert(false);
    }

    try {
        using RefType = typename std::conditional<
            std::is_lvalue_reference<Type>::value,
            typename std::remove_reference<Type>::type&&,
            Type
        >::type;
        TEST_IGNORE_NODISCARD std::any_cast<RefType>(static_cast<std::any&&>(a));
        assert(false);
    } catch (const std::bad_any_cast&) {
            // do nothing
    } catch (...) {
        assert(false);
    }
#else
    (TEST_IGNORE_NODISCARD a);
#endif
}

void test_cast_empty() {
    // None of these operations should allocate.
    DisableAllocationGuard g; (TEST_IGNORE_NODISCARD g);
    std::any a;
    checkThrows<int>(a);
}

template <class Type>
void test_cast_to_reference() {
    assert(Type::count == 0);
    Type::reset();
    {
        std::any a = Type(42);
        const std::any& ca = a;
        assert(Type::count == 1);
        assert(Type::copied == 0);
        assert(Type::moved == 1);

        // Try a cast to a bad type.
        // NOTE: Type cannot be an int.
        checkThrows<int>(a);
        checkThrows<int&, int const&>(a);
        checkThrows<Type*, Type const*>(a);
        checkThrows<Type const*>(a);

        // Check getting a type by reference from a non-const lvalue any.
        {
            Type& v = std::any_cast<Type&>(a);
            assert(v.value == 42);

            Type const &cv = std::any_cast<Type const&>(a);
            assert(&cv == &v);
        }
        // Check getting a type by reference from a const lvalue any.
        {
            Type const& v = std::any_cast<Type const&>(ca);
            assert(v.value == 42);

            Type const &cv = std::any_cast<Type const&>(ca);
            assert(&cv == &v);
        }
        // Check getting a type by reference from a const rvalue any.
        {
            Type const& v = std::any_cast<Type const&>(std::move(ca));
            assert(v.value == 42);

            Type const &cv = std::any_cast<Type const&>(std::move(ca));
            assert(&cv == &v);
        }
        // Check getting a type by reference from a const rvalue any.
        {
            Type&& v = std::any_cast<Type&&>(std::move(a));
            assert(v.value == 42);
            assert(std::any_cast<Type&>(a).value == 42);

            Type&& cv = std::any_cast<Type&&>(std::move(a));
            assert(&cv == &v);
            assert(std::any_cast<Type&>(a).value == 42);
        }
        // Check getting a type by reference from a const rvalue any.
        {
            Type const&& v = std::any_cast<Type const&&>(std::move(a));
            assert(v.value == 42);
            assert(std::any_cast<Type&>(a).value == 42);

            Type const&& cv = std::any_cast<Type const&&>(std::move(a));
            assert(&cv == &v);
            assert(std::any_cast<Type&>(a).value == 42);
        }
        // Check that the original object hasn't been changed.
        assertContains<Type>(a, 42);

        // Check that no objects have been created/copied/moved.
        assert(Type::count == 1);
        assert(Type::copied == 0);
        assert(Type::moved == 1);
    }
    assert(Type::count == 0);
}

template <class Type>
void test_cast_to_value() {
    assert(Type::count == 0);
    Type::reset();
    {
        std::any a = Type(42);
        assert(Type::count == 1);
        assert(Type::copied == 0);
        assert(Type::moved == 1);

        // Try a cast to a bad type.
        // NOTE: Type cannot be an int.
        checkThrows<int>(a);
        checkThrows<int&, int const&>(a);
        checkThrows<Type*, Type const*>(a);
        checkThrows<Type const*>(a);

        Type::reset(); // NOTE: reset does not modify Type::count
        // Check getting Type by value from a non-const lvalue any.
        // This should cause the non-const copy constructor to be called.
        {
            Type t = std::any_cast<Type>(a);

            assert(Type::count == 2);
            assert(Type::copied == 1);
            assert(Type::const_copied == 0);
            assert(Type::non_const_copied == 1);
            assert(Type::moved == 0);
            assert(t.value == 42);
        }
        assert(Type::count == 1);
        Type::reset();
        // Check getting const Type by value from a non-const lvalue any.
        // This should cause the const copy constructor to be called.
        {
            Type t = std::any_cast<Type const>(a);

            assert(Type::count == 2);
            assert(Type::copied == 1);
            assert(Type::const_copied == 0);
            assert(Type::non_const_copied == 1);
            assert(Type::moved == 0);
            assert(t.value == 42);
        }
        assert(Type::count == 1);
        Type::reset();
        // Check getting Type by value from a non-const lvalue any.
        // This should cause the const copy constructor to be called.
        {
            Type t = std::any_cast<Type>(static_cast<const std::any&>(a));

            assert(Type::count == 2);
            assert(Type::copied == 1);
            assert(Type::const_copied == 1);
            assert(Type::non_const_copied == 0);
            assert(Type::moved == 0);
            assert(t.value == 42);
        }
        assert(Type::count == 1);
        Type::reset();
        // Check getting Type by value from a non-const rvalue any.
        // This should cause the non-const copy constructor to be called.
        {
            Type t = std::any_cast<Type>(static_cast<std::any&&>(a));

            assert(Type::count == 2);
            assert(Type::moved == 1);
            assert(Type::copied == 0);
            assert(Type::const_copied == 0);
            assert(Type::non_const_copied == 0);
            assert(t.value == 42);
            assert(std::any_cast<Type&>(a).value == 0);
            std::any_cast<Type&>(a).value = 42; // reset the value
        }
        assert(Type::count == 1);
        Type::reset();
        // Check getting const Type by value from a non-const rvalue any.
        // This should cause the const copy constructor to be called.
        {
            Type t = std::any_cast<Type const>(static_cast<std::any&&>(a));

            assert(Type::count == 2);
            assert(Type::copied == 0);
            assert(Type::const_copied == 0);
            assert(Type::non_const_copied == 0);
            assert(Type::moved == 1);
            assert(t.value == 42);
            assert(std::any_cast<Type&>(a).value == 0);
            std::any_cast<Type&>(a).value = 42; // reset the value
        }
        assert(Type::count == 1);
        Type::reset();
        // Check getting Type by value from a const rvalue any.
        // This should cause the const copy constructor to be called.
        {
            Type t = std::any_cast<Type>(static_cast<const std::any&&>(a));

            assert(Type::count == 2);
            assert(Type::copied == 1);
            assert(Type::const_copied == 1);
            assert(Type::non_const_copied == 0);
            assert(Type::moved == 0);
            assert(t.value == 42);
            assert(std::any_cast<Type&>(a).value == 42);
        }
        // Ensure we still only have 1 Type object alive.
        assert(Type::count == 1);

        // Check that the original object hasn't been changed.
        assertContains<Type>(a, 42);
    }
    assert(Type::count == 0);
}

int main(int, char**) {
    test_cast_is_not_noexcept();
    test_cast_return_type();
    test_cast_empty();
    test_cast_to_reference<small>();
    test_cast_to_reference<large>();
    test_cast_to_value<small>();
    test_cast_to_value<large>();

  return 0;
}
