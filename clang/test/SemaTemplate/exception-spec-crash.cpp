// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -Wno-defaulted-function-deleted
// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -DCXX_EXCEPTIONS -fsyntax-only -verify %s -Wno-defaulted-function-deleted

template <class _Tp> struct is_nothrow_move_constructible {
  static const bool value = false;
};

template <class _Tp>
class allocator;

template <>
class allocator<char> {};

template <class _Allocator>
class basic_string {
  typedef _Allocator allocator_type;
  basic_string(basic_string &&__str)
  noexcept(is_nothrow_move_constructible<allocator_type>::value);
};

class Foo {
  Foo(Foo &&) noexcept = default;
#ifdef CXX_EXCEPTIONS
// expected-error@-2 {{does not match the calculated}}
#else
// expected-no-diagnostics
#endif
  Foo &operator=(Foo &&) noexcept = default;
  basic_string<allocator<char> > vectorFoo_;
};
