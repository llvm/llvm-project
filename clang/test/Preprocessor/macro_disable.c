// RUN: %clang_cc1 %s -E | FileCheck -strict-whitespace %s
// Check for C99 6.10.3.4p2.

#define f(a) f(x * (a)) 
#define x 2 
#define z z[0] 
f(f(z)); 
// CHECK: f(2 * (f(2 * (z[0]))));



#define A A B C 
#define B B C A 
#define C C A B 
A 
// CHECK: A B C A B A C A B C A


// PR1820
#define i(x) h(x
#define h(x) x(void) 
extern int i(i));
// CHECK: int i(void)


#define M_0(x) M_ ## x 
#define M_1(x) x + M_0(0) 
#define M_2(x) x + M_1(1) 
#define M_3(x) x + M_2(2) 
#define M_4(x) x + M_3(3) 
#define M_5(x) x + M_4(4) 

a: M_0(1)(2)(3)(4)(5);
b: M_0(5)(4)(3)(2)(1);

// CHECK: a: 2 + M_0(3)(4)(5);
// CHECK: b: 4 + 4 + 3 + 2 + 1 + M_0(3)(2)(1);

#define n(v) v
#define l m
#define m l a
c: n(m) X
// CHECK: c: m a X
