// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

void test0(void) {
  // CHECK-LABEL: define{{.*}} void @test0()
  // CHECK:      [[F:%.*]] = alloca float
  // CHECK-NEXT: [[REAL:%.*]] = load volatile float, ptr @test0_v, align 4
  // CHECK-NEXT: load volatile float, ptr getelementptr inbounds nuw (i8, ptr @test0_v, i64 4), align 4
  // CHECK-NEXT: store float [[REAL]], ptr [[F]], align 4
  // CHECK-NEXT: ret void
  extern volatile _Complex float test0_v;
  float f = (float) test0_v;
}

void test1(void) {
  // CHECK-LABEL: define{{.*}} void @test1()
  // CHECK:      [[REAL:%.*]] = load volatile float, ptr @test1_v, align 4
  // CHECK-NEXT: [[IMAG:%.*]] = load volatile float, ptr getelementptr inbounds nuw (i8, ptr @test1_v, i64 4), align 4
  // CHECK-NEXT: store volatile float [[REAL]], ptr @test1_v, align 4
  // CHECK-NEXT: store volatile float [[IMAG]], ptr getelementptr inbounds nuw (i8, ptr @test1_v, i64 4), align 4
  // CHECK-NEXT: ret void
  extern volatile _Complex float test1_v;
  test1_v = test1_v;
}
