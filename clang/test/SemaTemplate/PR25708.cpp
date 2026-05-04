// RUN: %clang_cc1 -std=c++11 -verify %s
// expected-no-diagnostics

struct FooAccessor
{
    template <typename T>
    using Foo = typename T::Foo;
};

class Type
{
    friend struct FooAccessor;

    using Foo = int;
};

int main()
{
    FooAccessor::Foo<Type> t;
}
