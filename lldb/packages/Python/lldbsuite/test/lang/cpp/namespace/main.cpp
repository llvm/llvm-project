//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdarg>

namespace {
    typedef unsigned int my_uint_t;
    int i; // Find the line number for anonymous namespace variable i.

    int myanonfunc (int a)
    {
        return a + a;
    }

    int
    variadic_sum (int arg_count...)
    {
        int sum = 0;
        va_list args;
        va_start(args, arg_count);

        for (int i = 0; i < arg_count; i++)
            sum += va_arg(args, int);

        va_end(args);
        return sum;
    }
}

namespace A {
    typedef unsigned int uint_t;
    namespace B {
        typedef unsigned int uint_t;
        int j; // Find the line number for named namespace variable j.
        int myfunc (int a);
        int myfunc2(int a)
        {
             return a + 2;
        }
        float myfunc (float f)
        {
            return f - 2.0;
        }
    }
}

namespace Y
{
    typedef unsigned int uint_t;
    using A::B::j;
    int foo;
}

using A::B::j;          // using declaration

namespace Foo = A::B;   // namespace alias

using Foo::myfunc;      // using declaration

using namespace Foo;    // using directive

namespace A {
    namespace B {
        using namespace Y;
        int k;
    }
}

namespace ns1 {
    int value = 100;
}

namespace ns2 {
    int value = 200;
}

#include <stdio.h>
void test_namespace_scopes() {
    do {
        using namespace ns1;
        printf("ns1::value = %d\n", value); // Evaluate ns1::value
    } while(0);
    
    do {
        using namespace ns2;
        printf("ns2::value = %d\n", value); // Evaluate ns2::value
    } while(0);
}

int Foo::myfunc(int a)
{
    test_namespace_scopes();    

    ::my_uint_t anon_uint = 0;
    A::uint_t a_uint = 1;
    B::uint_t b_uint = 2;
    Y::uint_t y_uint = 3;
    i = 3;
    j = 4;
    printf("::i=%d\n", ::i);
    printf("A::B::j=%d\n", A::B::j);
    printf("variadic_sum=%d\n", variadic_sum(3, 1, 2, 3));
    myanonfunc(3);
    return myfunc2(3) + j + i + a + 2 + anon_uint + a_uint + b_uint + y_uint; // Set break point at this line.
}

int
main (int argc, char const *argv[])
{
    return Foo::myfunc(12);
}
