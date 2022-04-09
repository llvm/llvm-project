// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

void test0(void) {
  // CHECK-LABEL: define{{.*}} void @test0()
  // CHECK:      [[F:%.*]] = alloca float
  // CHECK-NEXT: [[REAL:%.*]] = load volatile float, float* getelementptr inbounds ({ float, float }, { float, float }* @test0_v, i32 0, i32 0), align 4
  // CHECK-NEXT: load volatile float, float* getelementptr inbounds ({{.*}} @test0_v, i32 0, i32 1), align 4
  // CHECK-NEXT: store float [[REAL]], float* [[F]], align 4
  // CHECK-NEXT: ret void
  extern volatile _Complex float test0_v;
  float f = (float) test0_v;
}

void test1(void) {
  // CHECK-LABEL: define{{.*}} void @test1()
  // CHECK:      [[REAL:%.*]] = load volatile float, float* getelementptr inbounds ({{.*}} @test1_v, i32 0, i32 0), align 4
  // CHECK-NEXT: [[IMAG:%.*]] = load volatile float, float* getelementptr inbounds ({{.*}} @test1_v, i32 0, i32 1), align 4
  // CHECK-NEXT: store volatile float [[REAL]], float* getelementptr inbounds ({{.*}} @test1_v, i32 0, i32 0), align 4
  // CHECK-NEXT: store volatile float [[IMAG]], float* getelementptr inbounds ({{.*}} @test1_v, i32 0, i32 1), align 4
  // CHECK-NEXT: ret void
  extern volatile _Complex float test1_v;
  test1_v = test1_v;
}
