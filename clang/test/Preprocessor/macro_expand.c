// RUN: %clang_cc1 -E %s | grep '^A: Y$'
// RUN: %clang_cc1 -E %s | grep '^B: f()$'
// RUN: %clang_cc1 -E %s | grep '^C: for()$'

#define X() Y
#define Y() X

A: X()()()

// PR3927
#define f(x) h(x
#define for(x) h(x
#define h(x) x()
B: f(f))
C: for(for))

// rdar://6880648
#define f(x,y...) y
f()
