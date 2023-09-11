// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++23 -fcxx-exceptions %s

namespace std::inline __1 {
  template <class>
  class reference_wrapper {};
} // namespace std::__1

struct S {};

struct Derived : S {};

struct T {};

struct U {
  U(int) noexcept {}
};

struct V {
  V(int i) {
    if (i)
      throw int{};
  }
};

struct convertible_to_int {
  operator int();
};

struct explicitly_convertible_to_int {
  explicit operator int();
};

struct InvocableT {
  void operator()() {}
  int operator()(int, int, T) noexcept { return 0; }
  void operator()(T) && {}
};

struct StaticInvocableT {
  static void operator()() noexcept {}
};

struct Incomplete; // expected-note 2 {{forward declaration of 'Incomplete'}}

// __is_invocable_r
static_assert(__is_invocable_r()); // expected-error {{expected a type}}
static_assert(!__is_invocable_r(Incomplete, void)); // expected-error {{incomplete type 'Incomplete' used in type trait expression}}
static_assert(!__is_invocable_r(void)); // expected-error {{type trait requires 2 or more arguments; have 1 argument}}
static_assert(!__is_invocable_r(void, int));

using member_function_ptr_t = void (S::*)() noexcept;
using member_function_arg_ptr_t = void (S::*)(int);
using member_function_return_t = Derived (S::*)();

// bullet 1
static_assert(__is_invocable_r(void, member_function_ptr_t, S));
static_assert(__is_invocable_r(void, member_function_ptr_t&, S));
static_assert(!__is_invocable_r(void, member_function_ptr_t&, T));
static_assert(__is_invocable_r(void, member_function_ptr_t, S&));
static_assert(!__is_invocable_r(void, member_function_ptr_t, T&));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, S&, int));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, S&, int&));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, S&, int&));
static_assert(__is_invocable_r(void, member_function_return_t, S&));
static_assert(__is_invocable_r(Derived, member_function_return_t, S&));
static_assert(__is_invocable_r(S, member_function_return_t, S&));
static_assert(!__is_invocable_r(T, member_function_return_t, S&));

// bullet 2
static_assert(__is_invocable_r(void, member_function_ptr_t, std::reference_wrapper<S>));
static_assert(__is_invocable_r(void, member_function_ptr_t&, std::reference_wrapper<S>));
static_assert(!__is_invocable_r(void, member_function_ptr_t&, std::reference_wrapper<T>));
static_assert(__is_invocable_r(void, member_function_ptr_t, std::reference_wrapper<S>&));
static_assert(!__is_invocable_r(void, member_function_ptr_t, std::reference_wrapper<T>&));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, std::reference_wrapper<S>&, int));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, std::reference_wrapper<S>&, int&));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, std::reference_wrapper<S>&, int&));
static_assert(__is_invocable_r(void, member_function_return_t, std::reference_wrapper<S>&));
static_assert(__is_invocable_r(Derived, member_function_return_t, std::reference_wrapper<S>&));
static_assert(__is_invocable_r(S, member_function_return_t, std::reference_wrapper<S>&));
static_assert(!__is_invocable_r(T, member_function_return_t, std::reference_wrapper<S>&));

// bullet 3
static_assert(__is_invocable_r(void, member_function_ptr_t, S*));
static_assert(__is_invocable_r(void, member_function_ptr_t&, S*));
static_assert(!__is_invocable_r(void, member_function_ptr_t&, T*));
static_assert(__is_invocable_r(void, member_function_ptr_t, S*&));
static_assert(!__is_invocable_r(void, member_function_ptr_t, T*&));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, S*&, int));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, S*&, int&));
static_assert(__is_invocable_r(void, member_function_arg_ptr_t, S*&, int&));
static_assert(__is_invocable_r(void, member_function_return_t, S*&));
static_assert(__is_invocable_r(Derived, member_function_return_t, S*&));
static_assert(__is_invocable_r(S, member_function_return_t, S*&));
static_assert(!__is_invocable_r(T, member_function_return_t, S*&));

using member_ptr_t = int S::*;
// bullet 4
static_assert(!__is_invocable_r(void, member_ptr_t));
static_assert(!__is_invocable_r(void, member_ptr_t, S&, int));
static_assert(__is_invocable_r(void, member_ptr_t, S&));
static_assert(__is_invocable_r(void, member_ptr_t&, S&));
static_assert(!__is_invocable_r(void, member_ptr_t, T&));
static_assert(!__is_invocable_r(void, member_ptr_t, T&, int));
static_assert(__is_invocable_r(int, member_ptr_t, S&));
static_assert(__is_invocable_r(long, member_ptr_t, S&));
static_assert(!__is_invocable_r(T, member_ptr_t, S&));
static_assert(__is_invocable_r(void, member_ptr_t, Derived&));
static_assert(!__is_invocable_r(void, member_ptr_t, Derived&, int));

// bullet 5
static_assert(!__is_invocable_r(void, member_ptr_t, int));
static_assert(__is_invocable_r(void, member_ptr_t, std::reference_wrapper<S>));
static_assert(!__is_invocable_r(void, member_ptr_t, std::reference_wrapper<S>, int));
static_assert(__is_invocable_r(void, member_ptr_t, std::reference_wrapper<S>&));
static_assert(__is_invocable_r(void, member_ptr_t&, std::reference_wrapper<S>));
static_assert(!__is_invocable_r(void, member_ptr_t, std::reference_wrapper<T>));
static_assert(__is_invocable_r(int, member_ptr_t, std::reference_wrapper<S>));
static_assert(__is_invocable_r(long, member_ptr_t, std::reference_wrapper<S>));
static_assert(!__is_invocable_r(T, member_ptr_t, std::reference_wrapper<S>));
static_assert(__is_invocable_r(void, member_ptr_t, std::reference_wrapper<Derived>));

// bullet 6
static_assert(__is_invocable_r(void, member_ptr_t, S*));
static_assert(__is_invocable_r(void, member_ptr_t&, S*));
static_assert(!__is_invocable_r(void, member_ptr_t&, S*, int));
static_assert(!__is_invocable_r(void, member_ptr_t, T*));
static_assert(__is_invocable_r(int, member_ptr_t, S*));
static_assert(__is_invocable_r(long, member_ptr_t, S*));
static_assert(!__is_invocable_r(T, member_ptr_t, S*));
static_assert(__is_invocable_r(void, member_ptr_t, Derived*));

// Bullet 7
using func_t = void(*)() noexcept;
using func_arg_t = void(*)(int);
using func_arg_ref_t = void(*)(int&) noexcept;
using func_return_t = int(*)();
static_assert(__is_invocable_r(void, func_t));
static_assert(__is_invocable_r(void, func_t&));
static_assert(!__is_invocable_r(void, func_arg_t));
static_assert(__is_invocable_r(void, func_arg_t, int));
static_assert(__is_invocable_r(void, func_arg_t, int&));
static_assert(__is_invocable_r(void, func_arg_t, const int&));
static_assert(__is_invocable_r(void, func_arg_t, short));
static_assert(__is_invocable_r(void, func_arg_t, char));
static_assert(__is_invocable_r(void, func_arg_t, long));
static_assert(__is_invocable_r(void, func_arg_t, double));
static_assert(__is_invocable_r(void, func_arg_t, convertible_to_int));
static_assert(!__is_invocable_r(void, func_arg_t, explicitly_convertible_to_int));
static_assert(__is_invocable_r(void, func_arg_ref_t, int&));
static_assert(!__is_invocable_r(void, func_arg_ref_t, int));
static_assert(!__is_invocable_r(void, func_arg_ref_t, int&&));
static_assert(!__is_invocable_r(int, func_t));
static_assert(__is_invocable_r(int, func_return_t));
static_assert(!__is_invocable_r(void, S));
static_assert(__is_invocable_r(void, InvocableT));
static_assert(!__is_invocable_r(void, InvocableT, int));
static_assert(__is_invocable_r(void, InvocableT, int, int, T));
static_assert(__is_invocable_r(U, InvocableT, int, int, T));
static_assert(__is_invocable_r(int, InvocableT, int, int, T));
static_assert(__is_invocable_r(U, InvocableT, int, int, T));
static_assert(__is_invocable_r(V, InvocableT, int, int, T));
static_assert(__is_invocable_r(void, InvocableT&, int, int, T));
static_assert(!__is_invocable_r(T, InvocableT, int, int, T));
static_assert(!__is_invocable_r(T, InvocableT&, int, int, T));
static_assert(!__is_invocable_r(T, InvocableT&, const int&, int, T));
static_assert(__is_invocable_r(void, InvocableT&&, T));
static_assert(__is_invocable_r(void, StaticInvocableT));
static_assert(!__is_invocable_r(void, StaticInvocableT, int));
static_assert(__is_invocable_r(void, void(&)()));

// Make sure to reject abominable functions
template <class T>
inline constexpr bool is_invocable = __is_invocable_r(void, T);
static_assert(!is_invocable<void() const>);
static_assert(!is_invocable<void() volatile>);
static_assert(!is_invocable<void() const volatile>);
static_assert(!is_invocable<void() &>);
static_assert(!is_invocable<void() &&>);
static_assert(!is_invocable<void() const &>);
static_assert(!is_invocable<void() const &&>);

// but normal function prototypes are accepted
static_assert(__is_invocable_r(void, void()));

// __is_nothrow_invocable_r

static_assert(__is_nothrow_invocable_r()); // expected-error {{expected a type}}
static_assert(!__is_nothrow_invocable_r(Incomplete, void)); // expected-error {{incomplete type 'Incomplete' used in type trait expression}}
static_assert(!__is_nothrow_invocable_r(void)); // expected-error {{type trait requires 2 or more arguments; have 1 argument}}
static_assert(!__is_nothrow_invocable_r(void, int));

// bullet 1
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t, S));
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t&, S));
static_assert(!__is_nothrow_invocable_r(void, member_function_ptr_t&, T));
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t, S&));
static_assert(!__is_nothrow_invocable_r(void, member_function_ptr_t, T&));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, S&, int));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, S&, int&));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, S&, int&));
static_assert(!__is_nothrow_invocable_r(void, member_function_return_t, S&));
static_assert(!__is_nothrow_invocable_r(Derived, member_function_return_t, S&));
static_assert(!__is_nothrow_invocable_r(S, member_function_return_t, S&));
static_assert(!__is_nothrow_invocable_r(T, member_function_return_t, S&));

// bullet 2
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t, std::reference_wrapper<S>));
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t&, std::reference_wrapper<S>));
static_assert(!__is_nothrow_invocable_r(void, member_function_ptr_t&, std::reference_wrapper<T>));
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t, std::reference_wrapper<S>&));
static_assert(!__is_nothrow_invocable_r(void, member_function_ptr_t, std::reference_wrapper<T>&));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, std::reference_wrapper<S>&, int));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, std::reference_wrapper<S>&, int&));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, std::reference_wrapper<S>&, int&));
static_assert(!__is_nothrow_invocable_r(void, member_function_return_t, std::reference_wrapper<S>&));
static_assert(!__is_nothrow_invocable_r(Derived, member_function_return_t, std::reference_wrapper<S>&));
static_assert(!__is_nothrow_invocable_r(S, member_function_return_t, std::reference_wrapper<S>&));
static_assert(!__is_nothrow_invocable_r(T, member_function_return_t, std::reference_wrapper<S>&));

// bullet 3
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t, S*));
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t&, S*));
static_assert(!__is_nothrow_invocable_r(void, member_function_ptr_t&, T*));
static_assert(__is_nothrow_invocable_r(void, member_function_ptr_t, S*&));
static_assert(!__is_nothrow_invocable_r(void, member_function_ptr_t, T*&));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, S*&, int));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, S*&, int&));
static_assert(!__is_nothrow_invocable_r(void, member_function_arg_ptr_t, S*&, int&));
static_assert(!__is_nothrow_invocable_r(void, member_function_return_t, S*&));
static_assert(!__is_nothrow_invocable_r(Derived, member_function_return_t, S*&));
static_assert(!__is_nothrow_invocable_r(S, member_function_return_t, S*&));
static_assert(!__is_nothrow_invocable_r(T, member_function_return_t, S*&));

// bullet 4
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, S&, int));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t, S&));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t&, S&));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, T&));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, T&, int));
static_assert(__is_nothrow_invocable_r(int, member_ptr_t, S&));
static_assert(__is_nothrow_invocable_r(long, member_ptr_t, S&));
static_assert(!__is_nothrow_invocable_r(T, member_ptr_t, S&));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t, Derived&));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, Derived&, int));

// bullet 5
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, int));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t, std::reference_wrapper<S>));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, std::reference_wrapper<S>, int));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t, std::reference_wrapper<S>&));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t&, std::reference_wrapper<S>));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, std::reference_wrapper<T>));
static_assert(__is_nothrow_invocable_r(int, member_ptr_t, std::reference_wrapper<S>));
static_assert(__is_nothrow_invocable_r(long, member_ptr_t, std::reference_wrapper<S>));
static_assert(!__is_nothrow_invocable_r(T, member_ptr_t, std::reference_wrapper<S>));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t, std::reference_wrapper<Derived>));

// bullet 6
static_assert(__is_nothrow_invocable_r(void, member_ptr_t, S*));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t&, S*));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t&, S*, int));
static_assert(!__is_nothrow_invocable_r(void, member_ptr_t, T*));
static_assert(__is_nothrow_invocable_r(int, member_ptr_t, S*));
static_assert(__is_nothrow_invocable_r(long, member_ptr_t, S*));
static_assert(!__is_nothrow_invocable_r(T, member_ptr_t, S*));
static_assert(__is_nothrow_invocable_r(void, member_ptr_t, Derived*));

// Bullet 7
static_assert(__is_nothrow_invocable_r(void, func_t));
static_assert(__is_nothrow_invocable_r(void, func_t&));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, int));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, int&));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, const int&));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, short));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, char));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, long));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, double));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, convertible_to_int));
static_assert(!__is_nothrow_invocable_r(void, func_arg_t, explicitly_convertible_to_int));
static_assert(__is_nothrow_invocable_r(void, func_arg_ref_t, int&));
static_assert(!__is_nothrow_invocable_r(void, func_arg_ref_t, int));
static_assert(!__is_nothrow_invocable_r(void, func_arg_ref_t, int&&));
static_assert(!__is_nothrow_invocable_r(int, func_t));
static_assert(!__is_nothrow_invocable_r(int, func_return_t));
static_assert(!__is_nothrow_invocable_r(void, S));
static_assert(!__is_nothrow_invocable_r(void, InvocableT));
static_assert(!__is_nothrow_invocable_r(void, InvocableT, int));
static_assert(__is_nothrow_invocable_r(void, InvocableT, int, int, T));
static_assert(__is_nothrow_invocable_r(int, InvocableT, int, int, T));
static_assert(__is_nothrow_invocable_r(U, InvocableT, int, int, T));
static_assert(!__is_nothrow_invocable_r(V, InvocableT, int, int, T));
static_assert(__is_nothrow_invocable_r(void, InvocableT&, int, int, T));
static_assert(!__is_nothrow_invocable_r(T, InvocableT, int, int, T));
static_assert(!__is_nothrow_invocable_r(T, InvocableT&, int, int, T));
static_assert(!__is_nothrow_invocable_r(T, InvocableT&, const int&, int, T));
static_assert(!__is_nothrow_invocable_r(void, InvocableT&&, T));
static_assert(__is_nothrow_invocable_r(void, StaticInvocableT));
static_assert(!__is_nothrow_invocable_r(void, StaticInvocableT, int));

// Make sure to reject abominable functions
template <class T>
inline constexpr bool is_nothrow_invocable = __is_nothrow_invocable_r(void, T);
static_assert(!is_nothrow_invocable<void() const>);
static_assert(!is_nothrow_invocable<void() volatile>);
static_assert(!is_nothrow_invocable<void() const volatile>);
static_assert(!is_nothrow_invocable<void() &>);
static_assert(!is_nothrow_invocable<void() &&>);
static_assert(!is_nothrow_invocable<void() const &>);
static_assert(!is_nothrow_invocable<void() const &&>);
