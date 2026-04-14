// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

void conditionalResultIimplicitCast(int a, int b, float f) {
  // Should implicit cast back to int.
  int x = a && b;
  // CIR: %[[#INT:]] = cir.ternary
  // CIR: %{{.+}} = cir.cast bool_to_int %[[#INT]] : !cir.bool -> !s32i
  float y = f && f;
  // CIR: %[[#BOOL:]] = cir.ternary
  // CIR: %[[#INT:]] = cir.cast bool_to_int %[[#BOOL]] : !cir.bool -> !s32i
  // CIR: %{{.+}} = cir.cast int_to_float %[[#INT]] : !s32i -> !cir.float
}

// LLVM: define {{.*}}void @conditionalResultIimplicitCast(i32 {{.*}}, i32 {{.*}}, float {{.*}})
// LLVM:   zext i1 %{{.*}} to i32
// LLVM:   zext i1 %{{.*}} to i32
// LLVM:   sitofp i32 %{{.*}} to float

// OGCG: define {{.*}}void @conditionalResultIimplicitCast(i32 {{.*}}, i32 {{.*}}, float {{.*}})
// OGCG:   zext i1 %{{.*}} to i32
// OGCG:   zext i1 %{{.*}} to i32
// OGCG:   sitofp i32 %{{.*}} to float
