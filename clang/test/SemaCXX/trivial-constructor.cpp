// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -Wno-deprecated-builtins
// expected-no-diagnostics
struct T1 {
};
static_assert(__has_trivial_constructor(T1), "T1 has trivial constructor!");

struct T2 {
  T2();
};
static_assert(!__has_trivial_constructor(T2), "T2 has a user-declared constructor!");

struct T3 {
  virtual void f();
};
static_assert(!__has_trivial_constructor(T3), "T3 has a virtual function!");

struct T4 : virtual T3 {
};
static_assert(!__has_trivial_constructor(T4), "T4 has a virtual base class!");

struct T5 : T1 {
};
static_assert(__has_trivial_constructor(T5), "All the direct base classes of T5 have trivial constructors!");

struct T6 {
  T5 t5;
  T1 t1[2][2];
  static T2 t2;
};
static_assert(__has_trivial_constructor(T6), "All nonstatic data members of T6 have trivial constructors!");

struct T7 {
  T4 t4;
};
static_assert(!__has_trivial_constructor(T7), "t4 does not have a trivial constructor!");

struct T8 : T2 {
};
static_assert(!__has_trivial_constructor(T8), "The base class T2 does not have a trivial constructor!");
