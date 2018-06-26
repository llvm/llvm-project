// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-mfcall -fsanitize-trap=cfi-mfcall -fvisibility hidden -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-mfcall -fsanitize-trap=cfi-mfcall -fvisibility default -emit-llvm -o - %s | FileCheck --check-prefix=DEFAULT %s

struct B1 {};
struct B2 {};
struct B3 : B2 {};
struct S : B1, B3 {};

// DEFAULT-NOT: llvm.type.test

void f(S *s, void (S::*p)()) {
  // CHECK: [[OFFSET:%.*]] = sub i64 {{.*}}, 1
  // CHECK: [[VFPTR:%.*]] = getelementptr i8, i8* %{{.*}}, i64 [[OFFSET]]
  // CHECK: [[TT:%.*]] = call i1 @llvm.type.test(i8* [[VFPTR]], metadata !"_ZTSM1SFvvE.virtual")
  // CHECK: br i1 [[TT]], label {{.*}}, label %[[TRAP1:[^,]*]]

  // CHECK: [[TRAP1]]:
  // CHECK-NEXT: llvm.trap

  // CHECK: [[NVFPTR:%.*]] = bitcast void (%struct.S*)* {{.*}} to i8*
  // CHECK: [[TT1:%.*]] = call i1 @llvm.type.test(i8* [[NVFPTR]], metadata !"_ZTSM2B1FvvE")
  // CHECK: [[OR1:%.*]] = or i1 false, [[TT1]]
  // CHECK: [[TT2:%.*]] = call i1 @llvm.type.test(i8* [[NVFPTR]], metadata !"_ZTSM2B2FvvE")
  // CHECK: [[OR2:%.*]] = or i1 [[OR1]], [[TT2]]
  // CHECK: br i1 [[OR2]], label {{.*}}, label %[[TRAP2:[^,]*]]

  // CHECK: [[TRAP2]]:
  // CHECK-NEXT: llvm.trap
  (s->*p)();
}
