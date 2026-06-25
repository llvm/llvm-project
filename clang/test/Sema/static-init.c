// RUN: %clang_cc1 -fsyntax-only -verify -Wno-pointer-to-int-cast -Wno-bool-conversion %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-pointer-to-int-cast -Wno-bool-conversion %s -fexperimental-new-constant-interpreter

typedef __typeof((int*) 0 - (int*) 0) intptr_t;

static int f = 10;
static int b = f; // expected-error {{initializer element is not a compile-time constant}}

float r  = (float) (intptr_t) &r; // expected-error {{initializer element is not a compile-time constant}}
intptr_t s = (intptr_t) &s;
_Bool t = &t;


union bar {
  int i;
};

struct foo {
  short ptr;
};

union bar u[1];
struct foo x = {(intptr_t) u}; // expected-error {{initializer element is not a compile-time constant}}
struct foo y = {(char) u}; // expected-error {{initializer element is not a compile-time constant}}
intptr_t z = (intptr_t) u; // no-error

// [C11 6.5.3.2p3]: in '&*p' neither operator is evaluated and the result is as
// if both were omitted, so forming it is not a dereference and is a valid
// constant initializer even when 'p' is null. This must hold for a bit-field
// initializer just as it does for a plain scalar (it was previously rejected
// for bit-fields only, because they take a different evaluation path).
intptr_t deref_scalar = (intptr_t) &*(int *)0; // no-error
struct with_bitfield {
  long v : 8;
};
struct with_bitfield deref_bitfield = {.v = (long) &*(int *)0}; // no-error

// '&a[i]' is evaluated as 'a + i'.
intptr_t subscript_scalar = (intptr_t) &((int *)0)[0]; // no-error
intptr_t subscript_scalar_offset = (intptr_t) &((int *)0)[2]; // no-error
struct with_bitfield subscript_bitfield = {.v = (long) &((int *)0)[0]}; // no-error
struct with_bitfield subscript_bitfield_offset = {.v = (long) &((int *)0)[2]}; // no-error
