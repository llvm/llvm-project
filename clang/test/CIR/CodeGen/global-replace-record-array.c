// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct T {
  int i;
  char c[6];
  double d;
};

extern struct T arr[];

// Member 'c' is followed by padding, so the offset of arr[N].c is not a
// multiple of the alignment of struct T. When the incomplete array type of
// @arr is replaced with the complete type, the views below must still
// designate the same byte offsets.
char *p0 = &arr[0].c[2];
char *p1 = &arr[1].c[0];
double *p2 = &arr[1].d;

struct T arr[2] = {{1, "ab", 2.0}, {3, "cd", 4.0}};

// CIR-DAG: cir.global external @p0 = #cir.global_view<@arr, [0, 1, 2]> : !cir.ptr<!s8i>
// CIR-DAG: cir.global external @p1 = #cir.global_view<@arr, [1, 1]> : !cir.ptr<!s8i>
// CIR-DAG: cir.global external @p2 = #cir.global_view<@arr, [1, 2]> : !cir.ptr<!cir.double>

// LLVM-DAG: @p0 = global ptr getelementptr {{.*}}(i8, ptr @arr, i64 6)
// LLVM-DAG: @p1 = global ptr getelementptr {{.*}}(i8, ptr @arr, i64 28)
// LLVM-DAG: @p2 = global ptr getelementptr {{.*}}(i8, ptr @arr, i64 40)
