// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// LLVM: %struct.S1 = type { [3200 x double], [3200 x double] }
// LLVM: %struct.S2 = type { [10 x ptr] }
// LLVM: %struct.S3 = type { [2000 x i32], [2000 x i32], [2000 x i32] }
// LLVM: %struct.S4 = type { i32, i32, i32 }
// LLVM: %union.U1 = type { [2000 x i32] }

// LLVM: @s1 = global %struct.S1 zeroinitializer, align 8
// LLVM: @b1 = global ptr getelementptr inbounds (%struct.S1, ptr @s1, i32 0, i32 1), align 8
// LLVM: @s2 = global %struct.S2 zeroinitializer, align 8
// LLVM: @b2 = global ptr @s2, align 8
// LLVM: @s3 = global %struct.S3 zeroinitializer, align 4
// LLVM: @b3 = global ptr getelementptr inbounds (%struct.S3, ptr @s3, i32 0, i32 2), align 8
// LLVM: @s4 = global %struct.S4 zeroinitializer, align 4
// LLVM: @b4 = global ptr getelementptr inbounds (%struct.S4, ptr @s4, i32 0, i32 2), align 8
// LLVM: @u1 = global %union.U1 zeroinitializer, align 4
// LLVM: @b5 = global ptr @u1, align 8

struct S1 {
  double a[3200];
  double b[3200];
} s1;

double *b1 = s1.b;

struct S2 {
  double* a[10];
} s2;

double **b2 = s2.a;

struct S3 {
  int a[2000];
  int b[2000];
  int c[2000];
} s3;

double *b3 = s3.c;

struct S4 {
    int a, b, c;
} s4;

int* b4 = &s4.c;

union U1 {
  int a[2000];
  int b[2000];
  int c[2000];
} u1;

double *b5 = u1.a;