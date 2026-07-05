// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Deductions specific to C++0x.

template<typename T>
struct member_pointer_kind {
  static const unsigned value = 0;
};

template<class C, typename R, typename ...Args>
struct member_pointer_kind<R (C::*)(Args...)> {
  static const unsigned value = 1;
};

template<class C, typename R, typename ...Args>
struct member_pointer_kind<R (C::*)(Args...) &> {
  static const unsigned value = 2;
};

template<class C, typename R, typename ...Args>
struct member_pointer_kind<R (C::*)(Args...) &&> {
  static const unsigned value = 3;
};

template<class C, typename R, typename ...Args>
struct member_pointer_kind<R (C::*)(Args...) const> {
  static const unsigned value = 4;
};

template<class C, typename R, typename ...Args>
struct member_pointer_kind<R (C::*)(Args...) const &> {
  static const unsigned value = 5;
};

template<class C, typename R, typename ...Args>
struct member_pointer_kind<R (C::*)(Args...) const &&> {
  static const unsigned value = 6;
};

struct X { };

static_assert(member_pointer_kind<int (X::*)(int)>::value == 1, "");
static_assert(member_pointer_kind<int (X::*)(int) &>::value == 2, "");
static_assert(member_pointer_kind<int (X::*)(int) &&>::value == 3, "");
static_assert(member_pointer_kind<int (X::*)(int) const>::value == 4, "");
static_assert(member_pointer_kind<int (X::*)(int) const&>::value == 5, "");
static_assert(member_pointer_kind<int (X::*)(int) const&&>::value == 6, "");
