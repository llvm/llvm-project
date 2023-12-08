// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o %t %s
// RUN: grep '@g0 = internal global %.truct.s0 { i32 3 }' %t | count 1

struct s0 {
  int a;
};

static struct s0 g0;

static int f0(void) {
  return g0.a;
}

static struct s0 g0 = {3};

void *g1 = f0;
