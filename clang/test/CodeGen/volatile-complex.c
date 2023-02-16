// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// Validate that volatile _Complex loads and stores are generated
// properly, including their alignment (even when overaligned).
//
// This test assumes that floats are 32-bit aligned and doubles are
// 64-bit aligned, and uses x86-64 as a target that should have this
// property.

volatile _Complex float cf;
volatile _Complex double cd;
volatile _Complex float cf32 __attribute__((aligned(32)));
volatile _Complex double cd32 __attribute__((aligned(32)));

// CHECK-LABEL: define{{.*}} void @test_cf()
void test_cf(void) {
  // CHECK:      load volatile float, ptr @cf, align 4
  // CHECK-NEXT: load volatile float, ptr getelementptr inbounds ({ float, float }, ptr @cf, i32 0, i32 1), align 4
  (void)(cf);
  // CHECK-NEXT: [[R:%.*]] = load volatile float, ptr @cf, align 4
  // CHECK-NEXT: [[I:%.*]] = load volatile float, ptr getelementptr inbounds ({ float, float }, ptr @cf, i32 0, i32 1), align 4
  // CHECK-NEXT: store volatile float [[R]], ptr @cf, align 4
  // CHECK-NEXT: store volatile float [[I]], ptr getelementptr inbounds ({ float, float }, ptr @cf, i32 0, i32 1), align 4
  (void)(cf=cf);
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test_cd()
void test_cd(void) {
  // CHECK:      load volatile double, ptr @cd, align 8
  // CHECK-NEXT: load volatile double, ptr getelementptr inbounds ({ double, double }, ptr @cd, i32 0, i32 1), align 8
  (void)(cd);
  // CHECK-NEXT: [[R:%.*]] = load volatile double, ptr @cd, align 8
  // CHECK-NEXT: [[I:%.*]] = load volatile double, ptr getelementptr inbounds ({ double, double }, ptr @cd, i32 0, i32 1), align 8
  // CHECK-NEXT: store volatile double [[R]], ptr @cd, align 8
  // CHECK-NEXT: store volatile double [[I]], ptr getelementptr inbounds ({ double, double }, ptr @cd, i32 0, i32 1), align 8
  (void)(cd=cd);
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test_cf32()
void test_cf32(void) {
  // CHECK:      load volatile float, ptr @cf32, align 32
  // CHECK-NEXT: load volatile float, ptr getelementptr inbounds ({ float, float }, ptr @cf32, i32 0, i32 1), align 4
  (void)(cf32);
  // CHECK-NEXT: [[R:%.*]] = load volatile float, ptr @cf32, align 32
  // CHECK-NEXT: [[I:%.*]] = load volatile float, ptr getelementptr inbounds ({ float, float }, ptr @cf32, i32 0, i32 1), align 4
  // CHECK-NEXT: store volatile float [[R]], ptr @cf32, align 32
  // CHECK-NEXT: store volatile float [[I]], ptr getelementptr inbounds ({ float, float }, ptr @cf32, i32 0, i32 1), align 4
  (void)(cf32=cf32);
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @test_cd32()
void test_cd32(void) {
  // CHECK:      load volatile double, ptr @cd32, align 32
  // CHECK-NEXT: load volatile double, ptr getelementptr inbounds ({ double, double }, ptr @cd32, i32 0, i32 1), align 8
  (void)(cd32);
  // CHECK-NEXT: [[R:%.*]] = load volatile double, ptr @cd32, align 32
  // CHECK-NEXT: [[I:%.*]] = load volatile double, ptr getelementptr inbounds ({ double, double }, ptr @cd32, i32 0, i32 1), align 8
  // CHECK-NEXT: store volatile double [[R]], ptr @cd32, align 32
  // CHECK-NEXT: store volatile double [[I]], ptr getelementptr inbounds ({ double, double }, ptr @cd32, i32 0, i32 1), align 8
  (void)(cd32=cd32);
  // CHECK-NEXT: ret void
}
