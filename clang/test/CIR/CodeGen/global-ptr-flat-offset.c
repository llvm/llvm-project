// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fclangir -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o %t.ll %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t.ll %s

struct S {
  int x;
  int y;
};

struct S gS;

struct S *past_end = &gS + 1;

struct S *before = &gS - 1;

char *interior = (char *)&gS + 3;

int *member = &gS.y;

// CIR-DAG: cir.global external @past_end = #cir.global_offset<@gS, 8> : !cir.ptr<!rec_S>
// CIR-DAG: cir.global external @before = #cir.global_offset<@gS, -8> : !cir.ptr<!rec_S>
// CIR-DAG: cir.global external @interior = #cir.global_offset<@gS, 3> : !cir.ptr<!s8i>
// CIR-DAG: cir.global external @member = #cir.global_view<@gS, [1 : i32]> : !cir.ptr<!s32i>

// LLVM-DAG: @past_end = global ptr getelementptr {{.*}}(i8, ptr @gS, i64 8)
// LLVM-DAG: @before = global ptr getelementptr {{.*}}(i8, ptr @gS, i64 -8)
// LLVM-DAG: @interior = global ptr getelementptr {{.*}}(i8, ptr @gS, i64 3)
// LLVM-DAG: @member = global ptr getelementptr {{.*}}(i8, ptr @gS, i64 4)
