// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void f() {
  int result = 0;

  if consteval {
    result = 10; 
    // CIR-NOT: cir.const #cir.int<10> : !s32i
    // LLVM-NOT: store i32 10
    // OGCG-NOT: store i32 10
  } else {
    result = 20; 
    // CHECK: cir.const #cir.int<20> : !s32i
    // LLVM: store i32 20, ptr %1, align 4
    // OGCG: store i32 20, ptr %result, align 4

  }

  if !consteval {
    result = 30; 
    // CIR: cir.const #cir.int<30> : !s32i
    // LLVM: store i32 30, ptr %1, align 4
    // OGCG: store i32 30, ptr %result, align 4
  } else {
    result = 40; 
    // CIR-NOT: cir.const #cir.int<40> : !s32i
    // LLVM-NOT: store i32 40
    // OGCG-NOT: store i32 40
  }

  if consteval {
    result = 50; 
    // CIR-NOT: cir.const #cir.int<50> : !s32i
    // LLVM-NOT: store i32 50
    // OGCG-NOT: store i32 50
  }

  if !consteval {
    result = 60; 
    // CIR: cir.const #cir.int<60> : !s32i
    // LLVM: store i32 60, ptr %1, align 4
    // OGCG: store i32 60, ptr %result, align 4
  }

}
