// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-extensions -emit-llvm < %s | FileCheck %s

// -------------
// Scalar integer
// -------------
__unaligned int x;
void test1(void) {
  // CHECK: {{%.*}} = load i32, ptr @x, align 1
  // CHECK: store i32 {{%.*}}, ptr @x, align 1
  x++;
}

void test2(void) {
  // CHECK: %y = alloca i32, align 1
  // CHECK: {{%.*}} = load i32, ptr %y, align 1
  // CHECK: store i32 {{%.*}}, ptr %y, align 1
  __unaligned int y;
  y++;
}

void test2_1(void) {
  // CHECK: %y = alloca i32, align 1
  // CHECK: store i32 1, ptr %y, align 1
  __unaligned int y = 1;
}

// -------------
// Global pointer
// -------------
int *__unaligned p1;
void test3(void) {

  // CHECK: {{%.*}} = load ptr, ptr @p1, align 1
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 4
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 4
  (*p1)++;
}

int __unaligned *p2;
void test4(void) {
  // CHECK: {{%.*}} = load ptr, ptr @p2, align 8
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  (*p2)++;
}

int __unaligned *__unaligned p3;
void test5(void) {
  // CHECK: {{%.*}} = load ptr, ptr @p3, align 1
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  (*p3)++;
}

// -------------
// Local pointer
// -------------
void test6(void) {
  // CHECK: %lp1 = alloca ptr, align 1
  // CHECK: {{%.*}} = load ptr, ptr %lp1, align 1
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 4
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 4
  int *__unaligned lp1;
  (*lp1)++;
}

void test7(void) {
  // CHECK: %lp2 = alloca ptr, align 8
  // CHECK: {{%.*}} = load ptr, ptr %lp2, align 8
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  int __unaligned *lp2;
  (*lp2)++;
}

void test8(void) {
  // CHECK: %lp3 = alloca ptr, align 1
  // CHECK: {{%.*}} = load ptr, ptr %lp3, align 1
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  int __unaligned *__unaligned lp3;
  (*lp3)++;
}

// -------------
// Global array
// -------------
__unaligned int a[10];
void test9(void) {
  // CHECK: {{%.*}} = load i32, ptr getelementptr inbounds ([10 x i32], ptr @a, i64 0, i64 3), align 1
  // CHECK: store i32 {{%.*}}, ptr getelementptr inbounds ([10 x i32], ptr @a, i64 0, i64 3), align 1
  (a[3])++;
}

// -------------
// Local array
// -------------
void test10(void) {
  // CHECK: %la = alloca [10 x i32], align 1
  // CHECK: {{%.*}} = getelementptr inbounds [10 x i32], ptr %la, i64 0, i64 3
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  __unaligned int la[10];
  (la[3])++;
}

// --------
// Typedefs
// --------

typedef __unaligned int UnalignedInt;
void test13(void) {
  // CHECK: %i = alloca i32, align 1
  // CHECK: {{%.*}} = load i32, ptr %i, align 1
  // CHECK: store i32 {{%.*}}, ptr %i, align 1
  UnalignedInt i;
  i++;
}

typedef int Aligned;
typedef __unaligned Aligned UnalignedInt2;
void test14(void) {
  // CHECK: %i = alloca i32, align 1
  // CHECK: {{%.*}} = load i32, ptr %i, align 1
  // CHECK: store i32 {{%.*}}, ptr %i, align 1
  UnalignedInt2 i;
  i++;
}

typedef UnalignedInt UnalignedInt3;
void test15(void) {
  // CHECK: %i = alloca i32, align 1
  // CHECK: {{%.*}} = load i32, ptr %i, align 1
  // CHECK: store i32 {{%.*}}, ptr %i, align 1
  UnalignedInt3 i;
  i++;
}

// -------------
// Decayed types
// -------------
void test16(__unaligned int c[10]) {
  // CHECK: {{%.*}} = alloca ptr, align 8
  // CHECK: store ptr %c, ptr {{%.*}}, align 8
  // CHECK: {{%.*}} = load ptr, ptr {{%.*}}, align 8
  // CHECK: {{%.*}} = getelementptr inbounds i32, ptr {{%.*}}, i64 3
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  c[3]++;
}

// -----------
// __alignof__
// -----------
int test17(void) {
  // CHECK: ret i32 1
  return __alignof__(__unaligned int);
}

int test18(void) {
  // CHECK: ret i32 1
  __unaligned int a;
  return __alignof__(a);
}

int test19(void) {
  // CHECK: ret i32 1
  __unaligned int a[10];
  return __alignof__(a);
}

// -----------
// structs
// -----------
typedef
struct S1 {
    char c;
    int x;
} S1;

__unaligned S1 s1;
void test20(void) {
    // CHECK: {{%.*}} = load i32, ptr getelementptr inbounds nuw (%struct.S1, ptr @s1, i32 0, i32 1), align 1
    // CHECK: store i32 {{%.*}}, ptr getelementptr inbounds nuw (%struct.S1, ptr @s1, i32 0, i32 1), align 1
    s1.x++;
}

void test21(void) {
  // CHECK: {{%.*}} = alloca %struct.S1, align 1
  // CHECK: {{%.*}} = getelementptr inbounds nuw %struct.S1, ptr {{%.*}}, i32 0, i32 1
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  __unaligned S1 s1_2;
  s1_2.x++;
}

typedef
struct __attribute__((packed)) S2 {
    char c;
    int x;
} S2;

__unaligned S2 s2;
void test22(void) {
    // CHECK: {{%.*}} = load i32, ptr getelementptr inbounds nuw (%struct.S2, ptr @s2, i32 0, i32 1), align 1
    // CHECK: store i32 {{%.*}}, ptr getelementptr inbounds nuw (%struct.S2, ptr @s2, i32 0, i32 1), align 1
    s2.x++;
}

void test23(void) {
  // CHECK: {{%.*}} = alloca %struct.S2, align 1
  // CHECK: {{%.*}} = getelementptr inbounds nuw %struct.S2, ptr {{%.*}}, i32 0, i32 1
  // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 1
  // CHECK: store i32 {{%.*}}, ptr {{%.*}}, align 1
  __unaligned S2 s2_2;
  s2_2.x++;
}
