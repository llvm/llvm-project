// __COUNTER__ support: rdar://4329310
// RUN: %clang -E %s > %t

#define PASTE2(x,y) x##y
#define PASTE1(x,y) PASTE2(x,y)
#define UNIQUE(x) PASTE1(x,__COUNTER__)

// RUN: grep "A: 0" %t
A: __COUNTER__

// RUN: grep "B: foo1" %t
B: UNIQUE(foo);
// RUN: grep "C: foo2" %t
C: UNIQUE(foo);
// RUN: grep "D: 3" %t
D: __COUNTER__
