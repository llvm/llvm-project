// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

/// This used to cause problems because the heap-allocated array
/// is initialized by an ImplicitValueInitExpr of incomplete array type.

inline namespace {
template < class _Tp >
using __add_lvalue_reference_t = __add_lvalue_reference(_Tp);
template < class _Tp > using __remove_extent_t = __remove_extent(_Tp);
}
inline namespace {
template < class > class unique_ptr;
template < class _Tp > struct unique_ptr< _Tp[] > {
_Tp *__ptr_;

template <
      class _Tag, class _Ptr>
  constexpr unique_ptr(_Tag, _Ptr __ptr, unsigned) : __ptr_(__ptr){}
  constexpr ~unique_ptr() { delete[] __ptr_; }
  constexpr __add_lvalue_reference_t< _Tp >
  operator[](decltype(sizeof(int)) __i) {
    return __ptr_[__i];
  }};
constexpr unique_ptr< int[] > make_unique(decltype(sizeof(int)) __n) {
  return unique_ptr< int[] >(int(), new __remove_extent_t< int>[__n](), __n);
}}

constexpr bool test() {
  auto p1 = make_unique(5);
  (p1[0] == 0); // both-warning {{expression result unused}}
  return true;
}
static_assert(test());
