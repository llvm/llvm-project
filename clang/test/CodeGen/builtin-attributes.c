// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple arm-unknown-linux-gnueabi -emit-llvm -o - %s | FileCheck %s

int printf(const char *, ...);
void exit(int);

float frexpf(float, int*);
double frexp(double, int*);
long double frexpl(long double, int*);

// CHECK: declare i32 @printf(ptr noundef, ...)
void f0() {
  printf("a\n");
}

// CHECK: call void @exit
// CHECK: unreachable
void f1() {
  exit(1);
}

// CHECK: call ptr @strstr{{.*}} [[NUW:#[0-9]+]]
char* f2(char* a, char* b) {
  return __builtin_strstr(a, b);
}

// frexp is NOT readnone. It writes to its pointer argument.
//
// CHECK: f3
// CHECK: call double @frexp(double noundef %
// CHECK-NOT: readnone
// CHECK: call float @frexpf(float noundef %
// CHECK-NOT: readnone
// CHECK: call double @frexpl(double noundef %
// CHECK-NOT: readnone
//
// Same thing for modf and friends.
//
// CHECK: call double @modf(double noundef %
// CHECK-NOT: readnone
// CHECK: call float @modff(float noundef %
// CHECK-NOT: readnone
// CHECK: call double @modfl(double noundef %
// CHECK-NOT: readnone
//
// CHECK: call double @remquo(double noundef %
// CHECK-NOT: readnone
// CHECK: call float @remquof(float noundef %
// CHECK-NOT: readnone
// CHECK: call double @remquol(double noundef %
// CHECK-NOT: readnone
// CHECK: ret
int f3(double x) {
  int e;
  frexp(x, &e);
  frexpf(x, &e);
  frexpl(x, &e);
  __builtin_modf(x, &e);
  __builtin_modff(x, &e);
  __builtin_modfl(x, &e);
  __builtin_remquo(x, x, &e);
  __builtin_remquof(x, x, &e);
  __builtin_remquol(x, x, &e);
  return e;
}

// CHECK: attributes [[NUW]] = { nounwind{{.*}} }
