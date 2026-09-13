// RUN: %clang_cc1 -triple sparc-unknown-none -O1 -emit-llvm -o - %s | FileCheck %s

void sink(long double);
void vararg(int, ...);
long double source(void);

long double test(long double x) {
  sink(x);
  vararg(0, x);
  return source();
}

// CHECK-LABEL: define{{.*}} double @test(double noundef %x)
// CHECK: {{.*}}call void @sink(double noundef %x)
// CHECK: {{.*}}call void (i32, ...) @vararg(i32 noundef 0, double noundef %x)
// CHECK: [[RESULT:%.*]] = {{.*}}call double @source()
// CHECK: ret double [[RESULT]]
