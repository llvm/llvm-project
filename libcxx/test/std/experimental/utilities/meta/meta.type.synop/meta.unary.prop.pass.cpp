//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// GCC returns true for __is_trivially_constructible(void, int)
// See gcc.gnu.org/PR80682
// XFAIL: gcc

// <experimental/type_traits>

#include <experimental/type_traits>

namespace ex = std::experimental;

struct non_literal_type { non_literal_type() {} };
struct empty_type {};

struct polymorphic_type
{
    virtual void foo() {}
};

struct abstract_type
{
    virtual void foo() = 0;
};

struct final_type final {};

struct virtual_dtor_type
{
    virtual ~virtual_dtor_type() {}
};

void type_properties_test()
{
    {
        typedef const int T;
        static_assert(ex::is_const_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_const_v<T>), const bool>::value, "");
        static_assert(ex::is_const_v<T> == std::is_const<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_const_v<T>, "");
        static_assert(ex::is_const_v<T> == std::is_const<T>::value, "");
    }
    {
        typedef volatile int T;
        static_assert(ex::is_volatile_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_volatile_v<T>), const bool>::value, "");
        static_assert(ex::is_volatile_v<T> == std::is_volatile<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_volatile_v<T>, "");
        static_assert(ex::is_volatile_v<T> == std::is_volatile<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivial_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivial_v<T>), const bool>::value, "");
        static_assert(ex::is_trivial_v<T> == std::is_trivial<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(!ex::is_trivial_v<T>, "");
        static_assert(ex::is_trivial_v<T> == std::is_trivial<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivially_copyable_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_copyable_v<T>), const bool>::value, "");
        static_assert(ex::is_trivially_copyable_v<T> == std::is_trivially_copyable<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(!ex::is_trivially_copyable_v<T>, "");
        static_assert(ex::is_trivially_copyable_v<T> == std::is_trivially_copyable<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_standard_layout_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_standard_layout_v<T>), const bool>::value, "");
        static_assert(ex::is_standard_layout_v<T> == std::is_standard_layout<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(!ex::is_standard_layout_v<T>, "");
        static_assert(ex::is_standard_layout_v<T> == std::is_standard_layout<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_pod_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_pod_v<T>), const bool>::value, "");
        static_assert(ex::is_pod_v<T> == std::is_pod<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(!ex::is_pod_v<T>, "");
        static_assert(ex::is_pod_v<T> == std::is_pod<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_literal_type_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_literal_type_v<T>), const bool>::value, "");
        static_assert(ex::is_literal_type_v<T> == std::is_literal_type<T>::value, "");
    }
    {
        typedef non_literal_type T;
        static_assert(!ex::is_literal_type_v<T>, "");
        static_assert(ex::is_literal_type_v<T> == std::is_literal_type<T>::value, "");
    }
    {
        typedef empty_type T;
        static_assert(ex::is_empty_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_empty_v<T>), const bool>::value, "");
        static_assert(ex::is_empty_v<T> == std::is_empty<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_empty_v<T>, "");
        static_assert(ex::is_empty_v<T> == std::is_empty<T>::value, "");
    }
    {
        typedef polymorphic_type T;
        static_assert(ex::is_polymorphic_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_polymorphic_v<T>), const bool>::value, "");
        static_assert(ex::is_polymorphic_v<T> == std::is_polymorphic<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_polymorphic_v<T>, "");
        static_assert(ex::is_polymorphic_v<T> == std::is_polymorphic<T>::value, "");
    }
    {
        typedef abstract_type T;
        static_assert(ex::is_abstract_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_abstract_v<T>), const bool>::value, "");
        static_assert(ex::is_abstract_v<T> == std::is_abstract<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_abstract_v<T>, "");
        static_assert(ex::is_abstract_v<T> == std::is_abstract<T>::value, "");
    }
    {
        typedef final_type T;
        static_assert(ex::is_final_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_final_v<T>), const bool>::value, "");
        static_assert(ex::is_final_v<T> == std::is_final<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_final_v<T>, "");
        static_assert(ex::is_final_v<T> == std::is_final<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_signed_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_signed_v<T>), const bool>::value, "");
        static_assert(ex::is_signed_v<T> == std::is_signed<T>::value, "");
    }
    {
        typedef unsigned T;
        static_assert(!ex::is_signed_v<T>, "");
        static_assert(ex::is_signed_v<T> == std::is_signed<T>::value, "");
    }
    {
        typedef unsigned T;
        static_assert(ex::is_unsigned_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_unsigned_v<T>), const bool>::value, "");
        static_assert(ex::is_unsigned_v<T> == std::is_unsigned<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::is_unsigned_v<T>, "");
        static_assert(ex::is_unsigned_v<T> == std::is_unsigned<T>::value, "");
    }
}

void is_constructible_and_assignable_test()
{
    {
        typedef int T;
        static_assert(ex::is_constructible_v<T, int>, "");
        static_assert(std::is_same<decltype(ex::is_constructible_v<T, int>), const bool>::value, "");
        static_assert(ex::is_constructible_v<T, int> == std::is_constructible<T, int>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_constructible_v<T, int>, "");
        static_assert(ex::is_constructible_v<T, int> == std::is_constructible<T, int>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_default_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_default_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_default_constructible_v<T> == std::is_default_constructible<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(!ex::is_default_constructible_v<T>, "");
        static_assert(ex::is_default_constructible_v<T> == std::is_default_constructible<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_copy_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_copy_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_copy_constructible_v<T> == std::is_copy_constructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_copy_constructible_v<T>, "");
        static_assert(ex::is_copy_constructible_v<T> == std::is_copy_constructible<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_move_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_move_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_move_constructible_v<T> == std::is_move_constructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_move_constructible_v<T>, "");
        static_assert(ex::is_move_constructible_v<T> == std::is_move_constructible<T>::value, "");
    }
    {
        typedef int & T;
        typedef int U;
        static_assert(ex::is_assignable_v<T, U>, "");
        static_assert(std::is_same<decltype(ex::is_assignable_v<T, U>), const bool>::value, "");
        static_assert(ex::is_assignable_v<T, U> == std::is_assignable<T, U>::value, "");
    }
    {
        typedef int & T;
        typedef void U;
        static_assert(!ex::is_assignable_v<T, U>, "");
        static_assert(ex::is_assignable_v<T, U> == std::is_assignable<T, U>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_copy_assignable_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_copy_assignable_v<T>), const bool>::value, "");
        static_assert(ex::is_copy_assignable_v<T> == std::is_copy_assignable<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_copy_assignable_v<T>, "");
        static_assert(ex::is_copy_assignable_v<T> == std::is_copy_assignable<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_move_assignable_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_move_assignable_v<T>), const bool>::value, "");
        static_assert(ex::is_move_assignable_v<T> == std::is_move_assignable<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_move_assignable_v<T>, "");
        static_assert(ex::is_move_assignable_v<T> == std::is_move_assignable<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_destructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_destructible_v<T>), const bool>::value, "");
        static_assert(ex::is_destructible_v<T> == std::is_destructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_destructible_v<T>, "");
        static_assert(ex::is_destructible_v<T> == std::is_destructible<T>::value, "");
    }
}

void is_trivially_constructible_and_assignable_test()
{
    {
        typedef int T;
        static_assert(ex::is_trivially_constructible_v<T, int>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_constructible_v<T, int>), const bool>::value, "");
        static_assert(ex::is_trivially_constructible_v<T, int> == std::is_constructible<T, int>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_trivially_constructible_v<T, int>, "");
        static_assert(ex::is_trivially_constructible_v<T, int> == std::is_constructible<T, int>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivially_default_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_default_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_trivially_default_constructible_v<T> == std::is_default_constructible<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(!ex::is_trivially_default_constructible_v<T>, "");
        static_assert(ex::is_trivially_default_constructible_v<T> == std::is_default_constructible<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivially_copy_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_copy_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_trivially_copy_constructible_v<T> == std::is_copy_constructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_trivially_copy_constructible_v<T>, "");
        static_assert(ex::is_trivially_copy_constructible_v<T> == std::is_copy_constructible<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivially_move_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_move_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_trivially_move_constructible_v<T> == std::is_move_constructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_trivially_move_constructible_v<T>, "");
        static_assert(ex::is_trivially_move_constructible_v<T> == std::is_move_constructible<T>::value, "");
    }
    {
        typedef int & T;
        typedef int U;
        static_assert(ex::is_trivially_assignable_v<T, U>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_assignable_v<T, U>), const bool>::value, "");
        static_assert(ex::is_trivially_assignable_v<T, U> == std::is_assignable<T, U>::value, "");
    }
    {
        typedef int & T;
        typedef void U;
        static_assert(!ex::is_trivially_assignable_v<T, U>, "");
        static_assert(ex::is_trivially_assignable_v<T, U> == std::is_assignable<T, U>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivially_copy_assignable_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_copy_assignable_v<T>), const bool>::value, "");
        static_assert(ex::is_trivially_copy_assignable_v<T> == std::is_copy_assignable<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_trivially_copy_assignable_v<T>, "");
        static_assert(ex::is_trivially_copy_assignable_v<T> == std::is_copy_assignable<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivially_move_assignable_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_move_assignable_v<T>), const bool>::value, "");
        static_assert(ex::is_trivially_move_assignable_v<T> == std::is_move_assignable<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_trivially_move_assignable_v<T>, "");
        static_assert(ex::is_trivially_move_assignable_v<T> == std::is_move_assignable<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_trivially_destructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_trivially_destructible_v<T>), const bool>::value, "");
        static_assert(ex::is_trivially_destructible_v<T> == std::is_destructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_trivially_destructible_v<T>, "");
        static_assert(ex::is_trivially_destructible_v<T> == std::is_destructible<T>::value, "");
    }
}



void is_nothrow_constructible_and_assignable_test()
{
    {
        typedef int T;
        static_assert(ex::is_nothrow_constructible_v<T, int>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_constructible_v<T, int>), const bool>::value, "");
        static_assert(ex::is_nothrow_constructible_v<T, int> == std::is_constructible<T, int>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_nothrow_constructible_v<T, int>, "");
        static_assert(ex::is_nothrow_constructible_v<T, int> == std::is_constructible<T, int>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_nothrow_default_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_default_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_nothrow_default_constructible_v<T> == std::is_default_constructible<T>::value, "");
    }
    {
        typedef int & T;
        static_assert(!ex::is_nothrow_default_constructible_v<T>, "");
        static_assert(ex::is_nothrow_default_constructible_v<T> == std::is_default_constructible<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_nothrow_copy_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_copy_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_nothrow_copy_constructible_v<T> == std::is_copy_constructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_nothrow_copy_constructible_v<T>, "");
        static_assert(ex::is_nothrow_copy_constructible_v<T> == std::is_copy_constructible<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_nothrow_move_constructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_move_constructible_v<T>), const bool>::value, "");
        static_assert(ex::is_nothrow_move_constructible_v<T> == std::is_move_constructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_nothrow_move_constructible_v<T>, "");
        static_assert(ex::is_nothrow_move_constructible_v<T> == std::is_move_constructible<T>::value, "");
    }
    {
        typedef int & T;
        typedef int U;
        static_assert(ex::is_nothrow_assignable_v<T, U>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_assignable_v<T, U>), const bool>::value, "");
        static_assert(ex::is_nothrow_assignable_v<T, U> == std::is_assignable<T, U>::value, "");
    }
    {
        typedef int & T;
        typedef void U;
        static_assert(!ex::is_nothrow_assignable_v<T, U>, "");
        static_assert(ex::is_nothrow_assignable_v<T, U> == std::is_assignable<T, U>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_nothrow_copy_assignable_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_copy_assignable_v<T>), const bool>::value, "");
        static_assert(ex::is_nothrow_copy_assignable_v<T> == std::is_copy_assignable<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_nothrow_copy_assignable_v<T>, "");
        static_assert(ex::is_nothrow_copy_assignable_v<T> == std::is_copy_assignable<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_nothrow_move_assignable_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_move_assignable_v<T>), const bool>::value, "");
        static_assert(ex::is_nothrow_move_assignable_v<T> == std::is_move_assignable<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_nothrow_move_assignable_v<T>, "");
        static_assert(ex::is_nothrow_move_assignable_v<T> == std::is_move_assignable<T>::value, "");
    }
    {
        typedef int T;
        static_assert(ex::is_nothrow_destructible_v<T>, "");
        static_assert(std::is_same<decltype(ex::is_nothrow_destructible_v<T>), const bool>::value, "");
        static_assert(ex::is_nothrow_destructible_v<T> == std::is_destructible<T>::value, "");
    }
    {
        typedef void T;
        static_assert(!ex::is_nothrow_destructible_v<T>, "");
        static_assert(ex::is_nothrow_destructible_v<T> == std::is_destructible<T>::value, "");
    }
}

int main()
{
    type_properties_test();
    is_constructible_and_assignable_test();
    is_trivially_constructible_and_assignable_test();
    is_nothrow_constructible_and_assignable_test();
    {
        typedef virtual_dtor_type T;
        static_assert(ex::has_virtual_destructor_v<T>, "");
        static_assert(std::is_same<decltype(ex::has_virtual_destructor_v<T>), const bool>::value, "");
        static_assert(ex::has_virtual_destructor_v<T> == std::has_virtual_destructor<T>::value, "");
    }
    {
        typedef int T;
        static_assert(!ex::has_virtual_destructor_v<T>, "");
        static_assert(ex::has_virtual_destructor_v<T> == std::has_virtual_destructor<T>::value, "");
    }
}

