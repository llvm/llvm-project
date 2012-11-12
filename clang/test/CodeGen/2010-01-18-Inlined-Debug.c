// PR: 6058
// RUN: %clang_cc1 -g -emit-llvm %s  -O0 -o /dev/null

static inline int foo(double) __attribute__ ((always_inline));
static inline int foo(double __x) { return __x; }

void bar(double x) {
  foo(x);
}



