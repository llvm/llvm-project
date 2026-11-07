// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

void f(_Complex float a, _Complex float b) {
  // CHECK-LABEL: @f(
  // CHECK: [[REAL:%.*]] = fadd float
  // CHECK-NEXT: [[IMAG:%.*]] = fadd float
  // CHECK-NEXT: [[REAL_PTR:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP:%.*]], i32 0, i32 0
  // CHECK-NEXT: [[IMAG_PTR:%.*]] = getelementptr inbounds nuw { float, float }, ptr [[TMP]], i32 0, i32 1
  // CHECK-NEXT: store float [[REAL]], ptr [[REAL_PTR]], align 4
  // CHECK-NEXT: store float [[IMAG]], ptr [[IMAG_PTR]], align 4
  // CHECK-NEXT: [[INPUT:%.*]] = load i64, ptr [[TMP]], align 4
  // CHECK-NEXT: call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"(i64 [[INPUT]])
  asm volatile("" :: "r"(a + b));
}

void double_complex(_Complex double a, _Complex double b) {
  // CHECK-LABEL: @double_complex(
  // CHECK: [[REAL:%.*]] = fadd double
  // CHECK-NEXT: [[IMAG:%.*]] = fadd double
  // CHECK-NEXT: [[REAL_PTR:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[TMP:%.*]], i32 0, i32 0
  // CHECK-NEXT: [[IMAG_PTR:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[TMP]], i32 0, i32 1
  // CHECK-NEXT: store double [[REAL]], ptr [[REAL_PTR]], align 8
  // CHECK-NEXT: store double [[IMAG]], ptr [[IMAG_PTR]], align 8
  // CHECK-NEXT: call void asm sideeffect "", "*rm,~{dirflag},~{fpsr},~{flags}"(ptr elementtype({ double, double }) [[TMP]])
  asm volatile("" :: "rm"(a + b));
}

void memory_complex(_Complex float *a) {
  // CHECK-LABEL: @memory_complex(
  // CHECK: [[ADDR:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: store ptr %{{.*}}, ptr [[ADDR]], align 8
  // CHECK-NEXT: [[INPUT:%.*]] = load ptr, ptr [[ADDR]], align 8
  // CHECK-NEXT: call void asm sideeffect "", "*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype({ float, float }) [[INPUT]])
  asm volatile("" :: "m"(*a));
}

_Atomic(_Complex float) get_atomic_complex(void);

void atomic_complex(void) {
  // CHECK-LABEL: @atomic_complex(
  // CHECK: call void @get_atomic_complex(
  // CHECK: load atomic i64
  // CHECK: call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"(i64 %{{.*}})
  asm volatile("" :: "r"(get_atomic_complex()));
}
