// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c89 -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c89 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c89 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Implicit int return type.
test = 0;
func (void) {
  return 0;
}

// CIR: cir.global external @test = #cir.int<0> : !s32i
// CIR: cir.func {{.*}} @func() -> !s32i

// LLVM: @test = global i32 0, align 4
// LLVM: define dso_local i32 @func()
// LLVM:   ret i32

// OGCG: @test = global i32 0, align 4
// OGCG: define dso_local i32 @func()
// OGCG:   ret i32 0
