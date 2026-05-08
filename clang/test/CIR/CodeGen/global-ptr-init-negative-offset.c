// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fclangir -emit-llvm -o %t-cir.ll %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o %t.ll %s
// RUN: FileCheck -check-prefix=LLVM --input-file=%t.ll %s

// Constant initializers that point to an element before the start of an
// array produce a negative flat offset (e.g. flex-generated scanner tables
// that do `yytop = &yywork[-1]`). Although forming such a pointer is UB per
// the C standard, Clang accepts it as a constant expression.

struct Entry {
  int verify, advance;
};

extern struct Entry arr[];

struct Entry *before = &arr[-1];
int *before_field = &arr[-1].advance;

// CIR: cir.global external @before = #cir.global_view<@arr, [-1 : i32]> : !cir.ptr<!rec_Entry>
// CIR: cir.global external @before_field = #cir.global_view<@arr, [-1 : i32, 1 : i32]> : !cir.ptr<!s32i>

// LLVM: @before = global ptr getelementptr (i8, ptr @arr, i64 -8)
// LLVM: @before_field = global ptr getelementptr (i8, ptr @arr, i64 -4)
