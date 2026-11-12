// RUN: %clang_cc1 -Wfinal-macro %s -fsyntax-only -isystem %S/Inputs -verify

// Test warning production
#define Foo 1
// expected-note@+1 4{{macro marked 'final' here}}
#pragma clang final(Foo)

// expected-warning@+2{{macro 'Foo' has been marked as final and should not be redefined}}
// expected-note@+1{{previous definition is here}}
#define Foo 1

// expected-warning@+2{{macro 'Foo' has been marked as final and should not be redefined}}
// expected-warning@+1{{'Foo' macro redefined}}
#define Foo 2

// expected-warning@+1{{redefining builtin macro}}
#define __TIME__ 1

// expected-warning@+1{{undefining builtin macro}}
#undef __TIMESTAMP__

// expected-warning@+1{{macro 'Foo' has been marked as final and should not be undefined}}
#undef Foo
// expected-warning@+1{{macro 'Foo' has been marked as final and should not be redefined}}
#define Foo 3

// Test parse errors
// expected-error@+1{{expected (}}
#pragma clang final

// expected-error@+1{{expected )}}
#pragma clang final(Foo

// expected-error@+1{{no macro named 'Baz'}}
#pragma clang final(Baz)

// expected-error@+1{{expected identifier}}
#pragma clang final(4)

// expected-error@+1{{expected (}}
#pragma clang final Baz

// no diagnostics triggered by these pragmas.
#pragma clang deprecated(Foo)
#pragma clang restrict_expansion(Foo)

#define SYSTEM_MACRO Woah
// expected-note@+1 2{{macro marked 'final' here}}
#pragma clang final(SYSTEM_MACRO)
#include <final-macro-system.h>
