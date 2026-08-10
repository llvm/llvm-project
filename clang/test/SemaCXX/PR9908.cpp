// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

template <class _Tp, class _Up>
struct __allocator_traits_rebind
{
    typedef typename _Tp::template rebind<_Up>::other type;
};

template <class Alloc>
struct allocator_traits
{
    typedef Alloc allocator_type;
    template <class T> using rebind_alloc = typename
__allocator_traits_rebind<allocator_type, T>::type;
    template <class T> using rebind_traits = allocator_traits<rebind_alloc<T>>;
};

template <class T>
struct ReboundA {};

template <class T>
struct A
{
    typedef T value_type;

    template <class U> struct rebind {typedef ReboundA<U> other;};
};

int main()
{
    allocator_traits<A<char> >::rebind_traits<double> a;
}
