

// RUN: %clang_cc1 -x c++ -ast-dump -fbounds-safety -fexperimental-bounds-safety-cxx -verify %s | FileCheck %s

#include <ptrcheck.h>

typedef char * __terminated_by('\0') A;
typedef char * __terminated_by((char)0) B;

A funcA();
B funcB();

// CHECK: |-FunctionDecl {{.*}} testFunc 'char *__single __terminated_by('\x00')(int)'
auto testFunc(int pred) {
    if (pred) return funcA();
    return funcB();
}

// CHECK: |-FunctionDecl {{.*}} testFunc2 'char *__single __counted_by(10)*__single(int, char *__single __counted_by(10)*__single, char *__single __counted_by(10)*__single)'
auto testFunc2(int pred, char * __counted_by(10) * c, char * __counted_by(10) * d) {
    if (pred) return c;
    return d;
}

auto testFunc3(int pred, char * __counted_by(7) * c, char * __counted_by(10) * d) {
    if (pred) return c;
    return d; // expected-error{{'auto' in return type deduced as 'char *__single __counted_by(10)*__single' (aka 'char *__single*__single') here but deduced as 'char *__single __counted_by(7)*__single' (aka 'char *__single*__single') in earlier return statement}}
}
