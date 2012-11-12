// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm-bc -o - %s | opt --std-compile-opts | llvm-dis > %t
// RUN: grep "ret i32" %t | count 2
// RUN: grep "ret i32 0" %t | count 2
// <rdar://problem/6113085>

struct s0 {
  int x, y;
};

int f0() {
  struct s0 x = {0};
  return x.y;
}

#if 0
/* Optimizer isn't smart enough to reduce this since we use
   memset. Hrm. */
int f1() {
  struct s0 x[2] = { {0} };
  return x[1].x;
}
#endif

int f2() {
  int x[2] = { 0 };
  return x[1];
}

