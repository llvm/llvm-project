// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int x;
int arr[4];
int f(void);

unsigned long gx = (unsigned long)&x;
// CIR: cir.global external @gx = #cir.global_view<@x> : !u64i
// LLVM: @gx = global i64 ptrtoint (ptr @x to i64), align 8

unsigned long garr2 = (unsigned long)&arr[2];
// CIR: cir.global external @garr2 = #cir.global_view<@arr, [2 : i32]> : !u64i
// LLVM: @garr2 = global i64 ptrtoint (ptr getelementptr {{.*}}(i8, ptr @arr, i64 8) to i64), align 8

unsigned long gf = (unsigned long)&f;
// CIR: cir.global external @gf = #cir.global_view<@f> : !u64i
// LLVM: @gf = global i64 ptrtoint (ptr @f to i64), align 8
