// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

template <class _Tp, class _Up, bool = false>
struct __allocator_traits_rebind
{
};

template <template <class, class...> class _Alloc, class _Tp, class ..._Args,
class _Up>
struct __allocator_traits_rebind<_Alloc<_Tp, _Args...>, _Up, false>
{
   typedef _Alloc<_Up, _Args...> type;
};

template <class Alloc>
struct allocator_traits
{
   template <class T> using rebind_alloc = typename __allocator_traits_rebind<Alloc, T>::type;
   template <class T> using rebind_traits = allocator_traits<rebind_alloc<T>>;
};

template <class T>
struct allocator {};

int main()
{
   allocator_traits<allocator<char>>::rebind_alloc<int> a;
}
