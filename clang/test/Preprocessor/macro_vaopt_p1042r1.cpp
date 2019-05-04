 RUN: %clang_cc1 -E %s -pedantic -std=c++2a | FileCheck -strict-whitespace %s

#define LPAREN() (
#define G(Q) 42
#define F1(R, X, ...)  __VA_OPT__(G R X) )
1: int x = F1(LPAREN(), 0, <:-);
// CHECK: 1: int x = 42;

#define F2(...) f(0 __VA_OPT__(,) __VA_ARGS__)
#define EMP
2: F2(EMP)
// CHECK: 2: f(0 )

#define H3(X, ...) #__VA_OPT__(X##X X##X)
3: H3(, 0)
// CHECK: 3: ""

#define H4(X, ...) __VA_OPT__(a X ## X) ## b
4: H4(, 1)
// CHECK: 4: a b

#define H4B(X, ...) a ## __VA_OPT__(X ## X b)
4B: H4B(, 1)
// CHECK: 4B: a b

#define H5A(...) __VA_OPT__()/**/__VA_OPT__()
#define H5B(X) a ## X ## b
#define H5C(X) H5B(X)
5: H5C(H5A())
// CHECK: 5: ab
