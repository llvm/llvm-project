// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM_OGCG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM_OGCG

void f(char* fmt, ...);

void g() {
  f("test\0");
}
// CIR: cir.global {{.*}} @".str" = #cir.const_array<"test" : !cir.array<!s8i x 4>, trailing_zeros> : !cir.array<!s8i x 6>
// LLVM_OGCG: @.str = {{.*}} [6 x i8] c"test\00\00"
