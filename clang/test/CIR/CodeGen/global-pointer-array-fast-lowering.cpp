// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

const char *foo = "asdf";
const char *names[] = { "a", "b", "c" };
const int table[] = { 0, 1, 2, 3 };
const int matrix[2][2] = { { 1, 2 }, { 3, 4 } };
const bool flags[] = { true, false, true };
int len() {
  return sizeof(names) / sizeof(*names) + table[0] + matrix[1][1] +
         flags[2];
}

// CIR: cir.global {{.*}}@".str" = #cir.const_array<"asdf" : !cir.array<!s8i x 4>, trailing_zeros>

// LLVM:       @.str = {{.*}}constant [5 x i8]
// LLVM-NOT:   insertvalue

// CIR: cir.global {{.*}}@names = #cir.const_array<[#cir.global_view<@".str.1"> : !cir.ptr<!s8i>, #cir.global_view<@".str.2"> : !cir.ptr<!s8i>, #cir.global_view<@".str.3"> : !cir.ptr<!s8i>]>

// LLVM:       @names = {{.*}}global [3 x ptr] [ptr @.str{{.*}}, ptr @.str{{.*}}, ptr @.str{{.*}}]
// LLVM-NOT:   insertvalue

// CIR: cir.global {{.*}}table = #cir.const_array<[#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]>
// LLVM:       {{.*}}table{{.*}} = {{.*}}constant [4 x i32] [i32 0, i32 1, i32 2, i32 3]
// LLVM-NOT:   insertvalue

// CIR: cir.global {{.*}}matrix = #cir.const_array<
// LLVM:       {{.*}}matrix{{.*}} = {{.*}}constant [2 x [2 x i32]]
// LLVM-NOT:   insertvalue

// CIR: cir.global {{.*}}flags = #cir.const_array<[#true, #false, #true]> : !cir.array<!cir.bool x 3>
// LLVM:       {{.*}}flags{{.*}} = {{.*}}constant [3 x i8]
// LLVM-NOT:   insertvalue
