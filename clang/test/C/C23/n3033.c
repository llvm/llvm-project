// RUN: %clang_cc1 -std=c23 -E %s | FileCheck %s

/* WG14 N3033: Clang 12
 * Comma ommission and deletion (__VA_OPT__)
 */

#define F(...)           f(0 __VA_OPT__(,) __VA_ARGS__)
#define G(X, ...)        f(0, X __VA_OPT__(,) __VA_ARGS__)
#define SDEF(sname, ...) S sname __VA_OPT__(= { __VA_ARGS__ })
#define EMP

F(a, b, c)        // replaced by f(0, a, b, c)
// CHECK: f(0 , a, b, c)
F()               // replaced by f(0)
// CHECK: f(0 )
F(EMP)            // replaced by f(0)
// CHECK: f(0 )

G(a, b, c)        // replaced by f(0, a, b, c)
// CHECK: f(0, a , b, c)
G(a, )            // replaced by f(0, a)
// CHECK: f(0, a )
G(a)              // replaced by f(0, a)
// CHECK: f(0, a )

SDEF(foo);        // replaced by S foo;
// CHECK: S foo ;
SDEF(bar, 1, 2);  // replaced by S bar = { 1, 2 };
// CHECK: S bar = { 1, 2 };

//#define H1(X, ...)       X __VA_OPT__(##) __VA_ARGS__  // error: ## may not appear at the beginning of a replacement list (6.10.3.3)

#define H2(X, Y, ...)    __VA_OPT__(X ## Y,) __VA_ARGS__
H2(a, b, c, d)    // replaced by ab, c, d
// CHECK: ab, c, d

#define H3(X, ...)       #__VA_OPT__(X##X X##X)
H3(, 0)           // replaced by ""
// CHECK: ""

#define H4(X, ...)       __VA_OPT__(a X ## X) ## b
H4(, 1)           // replaced by a b
// CHECK: a b

#define H5A(...)         __VA_OPT__()/**/__VA_OPT__()
#define H5B(X)           a ## X ## b
#define H5C(X)           H5B(X)
H5C(H5A())        // replaced by ab
// CHECK: ab

