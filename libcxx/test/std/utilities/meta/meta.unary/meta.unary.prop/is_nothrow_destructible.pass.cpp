//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_destructible

#include <type_traits>
#include "test_macros.h"

template <class T>
void test_is_nothrow_destructible()
{
    static_assert( std::is_nothrow_destructible<T>::value, "");
    static_assert( std::is_nothrow_destructible<const T>::value, "");
    static_assert( std::is_nothrow_destructible<volatile T>::value, "");
    static_assert( std::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert( std::is_nothrow_destructible_v<T>, "");
    static_assert( std::is_nothrow_destructible_v<const T>, "");
    static_assert( std::is_nothrow_destructible_v<volatile T>, "");
    static_assert( std::is_nothrow_destructible_v<const volatile T>, "");
#endif
}

template <class T>
void test_is_not_nothrow_destructible()
{
    static_assert(!std::is_nothrow_destructible<T>::value, "");
    static_assert(!std::is_nothrow_destructible<const T>::value, "");
    static_assert(!std::is_nothrow_destructible<volatile T>::value, "");
    static_assert(!std::is_nothrow_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 14
    static_assert(!std::is_nothrow_destructible_v<T>, "");
    static_assert(!std::is_nothrow_destructible_v<const T>, "");
    static_assert(!std::is_nothrow_destructible_v<volatile T>, "");
    static_assert(!std::is_nothrow_destructible_v<const volatile T>, "");
#endif
}


struct PublicDestructor           { public:     ~PublicDestructor() {}};
struct ProtectedDestructor        { protected:  ~ProtectedDestructor() {}};
struct PrivateDestructor          { private:    ~PrivateDestructor() {}};

struct VirtualPublicDestructor           { public:    virtual ~VirtualPublicDestructor() {}};
struct VirtualProtectedDestructor        { protected: virtual ~VirtualProtectedDestructor() {}};
struct VirtualPrivateDestructor          { private:   virtual ~VirtualPrivateDestructor() {}};

struct PurePublicDestructor              { public:    virtual ~PurePublicDestructor() = 0; };
struct PureProtectedDestructor           { protected: virtual ~PureProtectedDestructor() = 0; };
struct PurePrivateDestructor             { private:   virtual ~PurePrivateDestructor() = 0; };

class Empty
{
};


union Union {};

struct bit_zero
{
    int :  0;
};

class Abstract
{
    virtual void foo() = 0;
};


int main(int, char**)
{
    test_is_not_nothrow_destructible<void>();
    test_is_not_nothrow_destructible<char[]>();
    test_is_not_nothrow_destructible<char[][3]>();

    test_is_nothrow_destructible<int&>();
    test_is_nothrow_destructible<int>();
    test_is_nothrow_destructible<double>();
    test_is_nothrow_destructible<int*>();
    test_is_nothrow_destructible<const int*>();
    test_is_nothrow_destructible<char[3]>();

#if TEST_STD_VER >= 11
    // requires noexcept. These are all destructible.
    test_is_nothrow_destructible<PublicDestructor>();
    test_is_nothrow_destructible<VirtualPublicDestructor>();
    test_is_nothrow_destructible<PurePublicDestructor>();
    test_is_nothrow_destructible<bit_zero>();
    test_is_nothrow_destructible<Abstract>();
    test_is_nothrow_destructible<Empty>();
    test_is_nothrow_destructible<Union>();
#endif
    // requires access control
    test_is_not_nothrow_destructible<ProtectedDestructor>();
    test_is_not_nothrow_destructible<PrivateDestructor>();
    test_is_not_nothrow_destructible<VirtualProtectedDestructor>();
    test_is_not_nothrow_destructible<VirtualPrivateDestructor>();
    test_is_not_nothrow_destructible<PureProtectedDestructor>();
    test_is_not_nothrow_destructible<PurePrivateDestructor>();


  return 0;
}
