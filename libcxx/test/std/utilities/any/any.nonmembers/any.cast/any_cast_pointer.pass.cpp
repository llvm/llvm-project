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
// ValueType const* any_cast(any const *) noexcept;
//
// template <class ValueType>
// ValueType * any_cast(any *) noexcept;

#include <any>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "any_helpers.h"

// Test that the operators are properly noexcept.
void test_cast_is_noexcept() {
    std::any a;
    ASSERT_NOEXCEPT(std::any_cast<int>(&a));

    const std::any& ca = a;
    ASSERT_NOEXCEPT(std::any_cast<int>(&ca));
}

// Test that the return type of any_cast is correct.
void test_cast_return_type() {
    std::any a;
    ASSERT_SAME_TYPE(decltype(std::any_cast<int>(&a)),       int*);
    ASSERT_SAME_TYPE(decltype(std::any_cast<int const>(&a)), int const*);

    const std::any& ca = a;
    ASSERT_SAME_TYPE(decltype(std::any_cast<int>(&ca)),       int const*);
    ASSERT_SAME_TYPE(decltype(std::any_cast<int const>(&ca)), int const*);
}

// Test that any_cast handles null pointers.
void test_cast_nullptr() {
    std::any *a = nullptr;
    assert(nullptr == std::any_cast<int>(a));
    assert(nullptr == std::any_cast<int const>(a));

    const std::any *ca = nullptr;
    assert(nullptr == std::any_cast<int>(ca));
    assert(nullptr == std::any_cast<int const>(ca));
}

// Test casting an empty object.
void test_cast_empty() {
    {
        std::any a;
        assert(nullptr == std::any_cast<int>(&a));
        assert(nullptr == std::any_cast<int const>(&a));

        const std::any& ca = a;
        assert(nullptr == std::any_cast<int>(&ca));
        assert(nullptr == std::any_cast<int const>(&ca));
    }
    // Create as non-empty, then make empty and run test.
    {
        std::any a(42);
        a.reset();
        assert(nullptr == std::any_cast<int>(&a));
        assert(nullptr == std::any_cast<int const>(&a));

        const std::any& ca = a;
        assert(nullptr == std::any_cast<int>(&ca));
        assert(nullptr == std::any_cast<int const>(&ca));
    }
}

template <class Type>
void test_cast() {
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
        assert(std::any_cast<int>(&a) == nullptr);
        assert(std::any_cast<int const>(&a) == nullptr);
        assert(std::any_cast<int const volatile>(&a) == nullptr);

        // Try a cast to the right type, but as a pointer.
        assert(std::any_cast<Type*>(&a) == nullptr);
        assert(std::any_cast<Type const*>(&a) == nullptr);

        // Check getting a unqualified type from a non-const any.
        Type* v = std::any_cast<Type>(&a);
        assert(v != nullptr);
        assert(v->value == 42);

        // change the stored value and later check for the new value.
        v->value = 999;

        // Check getting a const qualified type from a non-const any.
        Type const* cv = std::any_cast<Type const>(&a);
        assert(cv != nullptr);
        assert(cv == v);
        assert(cv->value == 999);

        // Check getting a unqualified type from a const any.
        cv = std::any_cast<Type>(&ca);
        assert(cv != nullptr);
        assert(cv == v);
        assert(cv->value == 999);

        // Check getting a const-qualified type from a const any.
        cv = std::any_cast<Type const>(&ca);
        assert(cv != nullptr);
        assert(cv == v);
        assert(cv->value == 999);

        // Check that no more objects were created, copied or moved.
        assert(Type::count == 1);
        assert(Type::copied == 0);
        assert(Type::moved == 1);
    }
    assert(Type::count == 0);
}

void test_cast_non_copyable_type()
{
    // Even though 'any' never stores non-copyable types
    // we still need to support any_cast<NoCopy>(ptr)
    struct NoCopy { NoCopy(NoCopy const&) = delete; };
    std::any a(42);
    std::any const& ca = a;
    assert(std::any_cast<NoCopy>(&a) == nullptr);
    assert(std::any_cast<NoCopy>(&ca) == nullptr);
}

void test_cast_array() {
    int arr[3];
    std::any a(arr);
    RTTI_ASSERT(a.type() == typeid(int*)); // contained value is decayed
    // We can't get an array out
    int (*p)[3] = std::any_cast<int[3]>(&a);
    assert(p == nullptr);
}

void test_fn() {}

void test_cast_function_pointer() {
    using T = void(*)();
    std::any a(test_fn);
    // An any can never store a function type, but we should at least be able
    // to ask.
    assert(std::any_cast<void()>(&a) == nullptr);
    T fn_ptr = std::any_cast<T>(a);
    assert(fn_ptr == test_fn);
}

int main(int, char**) {
    test_cast_is_noexcept();
    test_cast_return_type();
    test_cast_nullptr();
    test_cast_empty();
    test_cast<small>();
    test_cast<large>();
    test_cast_non_copyable_type();
    test_cast_array();
    test_cast_function_pointer();

  return 0;
}
