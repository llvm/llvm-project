// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix

// RUN: cat %s | clang-repl | FileCheck %s
int *x = new int();
template <class T> struct GuardX { T *&x; GuardX(T *&x) : x(x) {}; ~GuardX(); };

// clang normally defers codegen for this out-of-line ~GuardX(), which would
// cause the JIT to report Symbols not found: [ _ZN6GuardXIiED2Ev ]
extern "C" int printf(const char *, ...);
template <class T> GuardX<T>::~GuardX() { delete x; printf("Running dtor\n"); }

// Let's make sure that the RuntimeInterfaceBuilder requests it explicitly:
(GuardX<int>(x))

// CHECK-NOT: Symbols not found
// CHECK: Running dtor
