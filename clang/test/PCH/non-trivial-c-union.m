// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -emit-pch -o %t.pch %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -include-pch %t.pch -verify %s

#ifndef HEADER
#define HEADER

typedef union {
  id f0;
} U0;

#else

// expected-note@-6 {{'U0' has subobjects that are non-trivial to destruct}}
// expected-note@-7 {{'U0' has subobjects that are non-trivial to copy}}
// expected-note@-8 {{'U0' has subobjects that are non-trivial to default-initialize}}
// expected-note@-8 {{f0 has type '__strong id' that is non-trivial to destruct}}
// expected-note@-9 {{f0 has type '__strong id' that is non-trivial to copy}}
// expected-note@-10 {{f0 has type '__strong id' that is non-trivial to default-initialize}}

U0 foo0(void); // expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to destruct}} expected-error {{cannot use type 'U0' for function/method return since it is a union that is non-trivial to copy}}

U0 g0; // expected-error {{cannot default-initialize an object of type 'U0' since it is a union that is non-trivial to default-initialize}}

#endif
