// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

// Regression test for https://github.com/llvm/llvm-project/issues/199407

#define FOO(Ty, Name) alignas(Ty) char Name[sizeof(Ty)]

FOO(struct S { float *malloc(long) __attribute__((alloc_size(1))); }, buffer); // expected-error 2{{field 'malloc' declared as a function}}
