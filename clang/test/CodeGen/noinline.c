// Make sure -fno-inline-functions is behaving correctly.
// rdar://10972766

// RUN: %clang_cc1 -O3 -fno-inline -fno-inline-functions -emit-llvm %s -o - | FileCheck -check-prefix=NOINLINE %s

inline int dont_inline_me(int a, int b) { return(a+b); }

inline __attribute__ ((__always_inline__)) int inline_me(int a, int b) { return(a*b); }

volatile int *pa = (int*) 0x1000;
void foo(void) {
// NOINLINE: @foo
// NOINLINE: dont_inline_me
// NOINLINE-NOT: inlinehint
    pa[0] = dont_inline_me(pa[1],pa[2]);	
// NOINLINE-NOT: inline_me
    pa[3] = inline_me(pa[4],pa[5]);
}

// NOINLINE: Function Attrs: noinline
// NOINLINE: @dont_inline_me
