// RUN: %clang_cc1 -triple sparc-unknown-none -O1 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple sparc-unknown-linux-gnu -O1 -emit-llvm -o - %s | FileCheck %s --check-prefix=LINUX

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

// Linux retains the SPARC V8 System V 128-bit long double ABI.
// LINUX-LABEL: define{{.*}} void @test(ptr {{.*}}sret(fp128) align 8{{.*}}, ptr {{.*}}byval(fp128) align 8{{.*}})
// LINUX: {{.*}}call void @sink(ptr {{.*}}byval(fp128) align 8{{.*}})
// LINUX: {{.*}}call void (i32, ...) @vararg(i32 noundef 0, ptr {{.*}}byval(fp128) align 8{{.*}})
// LINUX: {{.*}}call void @source(ptr {{.*}}sret(fp128) align 8{{.*}})
// LINUX: ret void
