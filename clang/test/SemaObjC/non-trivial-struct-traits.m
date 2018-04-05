// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify %s

// expected-no-diagnostics

struct Trivial {
  int x;
};

struct NonTrivial {
  id x;
};

int trivial_assign[__has_trivial_assign(struct Trivial) ? 1 : -1];
int trivial_move_assign[__has_trivial_move_assign(struct Trivial) ? 1 : -1];
int trivial_copy_constructor[__has_trivial_copy(struct Trivial) ? 1 : -1];
int trivial_move_constructor[__has_trivial_move_constructor(struct Trivial) ? 1 : -1];
int trivial_constructor[__has_trivial_constructor(struct Trivial) ? 1 : -1];
int trivial_destructor[__has_trivial_destructor(struct Trivial) ? 1 : -1];

int non_trivial_assign[__has_trivial_assign(struct NonTrivial) ? -1 : 1];
int non_trivial_move_assign[__has_trivial_move_assign(struct NonTrivial) ? -1 : 1];
int non_trivial_copy_constructor[__has_trivial_copy(struct NonTrivial) ? -1 : 1];
int non_trivial_move_constructor[__has_trivial_move_constructor(struct NonTrivial) ? -1 : 1];
int non_trivial_constructor[__has_trivial_constructor(struct NonTrivial) ? -1 : 1];
int non_trivial_destructor[__has_trivial_destructor(struct NonTrivial) ? -1 : 1];

int volatile_trivial_assign[__has_trivial_assign(volatile int) ? 1 : -1];
int volatile_trivial_move_assign[__has_trivial_move_assign(volatile int) ? 1 : -1];
int volatile_trivial_copy_constructor[__has_trivial_copy(volatile int) ? 1 : -1];
int volatile_trivial_move_constructor[__has_trivial_move_constructor(volatile int) ? 1 : -1];
int volatile_trivial_copy_constructor2[__has_trivial_copy(volatile struct Trivial) ? 1 : -1];
int volatile_trivial_move_constructor2[__has_trivial_move_constructor(volatile struct Trivial) ? 1 : -1];
