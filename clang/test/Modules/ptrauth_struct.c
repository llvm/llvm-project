// Test serializing a module that contains a struct
// with a ptrauth qualified field.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// The command below should not crash.
// RUN: %clang_cc1  -triple arm64e-apple-macosx -fptrauth-returns \
// RUN:   -fptrauth-intrinsics -fptrauth-calls -fptrauth-indirect-gotos \
// RUN:   -fptrauth-auth-traps -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -o %t/tu.o -x c %t/tu.c

//--- tu.c
#include "struct_with_ptrauth_field.h"

int foo(struct T *t) {
    return t->s.foo(t->s.v, t->s.v) + t->arr[12];
}

//--- struct_with_ptrauth_field.h
typedef int (* FuncTy) (int, int);

struct S{
    FuncTy __ptrauth(0, 1, 1234) foo;
    int v;
};

struct T {
  struct S s;
  char arr[64];
};

//--- module.modulemap
module mod1 {
  header "struct_with_ptrauth_field.h"
}