// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

// PR10087: Make sure that we don't conflate exception specifications
// from different functions in the canonical type system.
namespace std
{

template <class _Tp> _Tp&& declval() noexcept;

template <class _Tp, class... _Args>
struct __is_nothrow_constructible
{
  static const bool value = noexcept(_Tp(declval<_Args>()...));
};

template<class, class _Traits, class _Allocator>
class basic_string
{
public:
  typedef typename _Traits::char_type value_type;
  typedef _Allocator allocator_type;

  basic_string()
      noexcept(__is_nothrow_constructible<allocator_type>::value);
};

template <class, class, class _Compare>
struct __map_value_compare
{
public:
  __map_value_compare()
      noexcept(__is_nothrow_constructible<_Compare>::value);
};

struct less
{
};

struct map
{
  typedef __map_value_compare<int, short, less> __vc;
  __vc vc_;
};


template<class T, class _Traits, class _Allocator>
basic_string<T, _Traits, _Allocator>::basic_string() noexcept(__is_nothrow_constructible<allocator_type>::value) {}

template <class T, class Value, class _Compare>
__map_value_compare<T, Value, _Compare>::__map_value_compare()
  noexcept(__is_nothrow_constructible<_Compare>::value) {}

}  // std
