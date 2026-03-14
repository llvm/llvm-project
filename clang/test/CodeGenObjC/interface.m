// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -O3 -emit-llvm -o %t %s
// RUN: grep 'ret i32 385' %t

void *alloca(unsigned);

@interface I0 {
@public
  int iv0;
  int iv1;
  int iv2;
}
@end

static int f0(I0 *a0) {
  return (*(a0 + 2)).iv0;
}

static int f1(I0 *a0) {
  return a0[2].iv1;
}

static int f2(I0 *a0) {
  return (*(a0 - 1)).iv2;
}

int g0(void) {
  I0 *a = alloca(sizeof(*a) * 4);
  a[2].iv0 = 5;
  a[2].iv1 = 7;
  a[2].iv2 = 11;
  return f0(a) * f1(a) * f2(&a[3]);
}


