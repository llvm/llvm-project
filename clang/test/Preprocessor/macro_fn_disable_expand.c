// RUN: %clang_cc1 %s -E | FileCheck %s

#define foo(x) bar x
foo(foo) (2)
// CHECK: bar foo (2)

#define m(a) a(w)
#define w ABCD
m(m)
// CHECK: m(ABCD)



// PR4438, PR5163

// We should get '42' in the argument list for gcc compatibility.
#define A 1
#define B 2
#define C(x) (x + 1)

X: C(
#ifdef A
#if A == 1
#if B
    42
#endif
#endif
#endif
    )
// CHECK: X: (42 + 1)
