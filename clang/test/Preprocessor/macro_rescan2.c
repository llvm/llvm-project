// RUN: %clang_cc1 %s -E | grep 'a: 2\*f(9)'
// RUN: %clang_cc1 %s -E | grep 'b: 2\*9\*g'

#define f(a) a*g 
#define g f 
a: f(2)(9) 

#undef f
#undef g

#define f(a) a*g 
#define g(a) f(a) 

b: f(2)(9)

