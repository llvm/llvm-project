// RUN: %clang_cc1 -std=gnu++98 -fsyntax-only -verify %s
// expected-no-diagnostics

_Atomic unsigned an_atomic_uint;

enum { an_enum_value = 1 };

void enum1() { an_atomic_uint += an_enum_value; }

void enum2() { an_atomic_uint |= an_enum_value; }

volatile _Atomic unsigned an_volatile_atomic_uint;

void enum3() { an_volatile_atomic_uint += an_enum_value; }

void enum4() { an_volatile_atomic_uint |= an_enum_value; }
