// RUN: %clang_cc1 %s -E | FileCheck %s

#define FOO bar ## baz ## 123

// CHECK: A: barbaz123
A: FOO

// PR9981
#define M1(A) A
#define M2(X) X
B: M1(M2(##))

// CHECK: B: ##

