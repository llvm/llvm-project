// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -Wno-deprecated-builtins
// expected-no-diagnostics
struct T1 {
};
static_assert(__has_trivial_destructor(T1), "T1 has trivial destructor!");

struct T2 {
  ~T2();
};
static_assert(!__has_trivial_destructor(T2), "T2 has a user-declared destructor!");

struct T3 {
  virtual void f();
};
static_assert(__has_trivial_destructor(T3), "T3 has a virtual function (but still a trivial destructor)!");

struct T4 : virtual T3 {
};
static_assert(__has_trivial_destructor(T4), "T4 has a virtual base class! (but still a trivial destructor)!");

struct T5 : T1 {
};
static_assert(__has_trivial_destructor(T5), "All the direct base classes of T5 have trivial destructors!");

struct T6 {
  T5 t5;
  T1 t1[2][2];
  static T2 t2;
};
static_assert(__has_trivial_destructor(T6), "All nonstatic data members of T6 have trivial destructors!");

struct T7 {
  T2 t2;
};
static_assert(!__has_trivial_destructor(T7), "t2 does not have a trivial destructor!");

struct T8 : T2 {
};
static_assert(!__has_trivial_destructor(T8), "The base class T2 does not have a trivial destructor!");
