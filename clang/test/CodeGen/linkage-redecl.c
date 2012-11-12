// RUN: %clang_cc1 -emit-llvm %s -o - |grep internal

// C99 6.2.2p3
// PR3425
static void f(int x);

void g0() {
  f(5);
}

extern void f(int x) { } // still has internal linkage
