// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

struct I { int k[3]; };
struct M { struct I o[2]; };
struct M v1[1] = { [0].o[0 ... 1].k[0 ... 1] = 4, 5 };
unsigned v2[2][3] = {[0 ... 1][0 ... 1] = 2222, 3333};

// CHECK-DAG: %struct.M = type { [2 x %struct.I] }
// CHECK-DAG: %struct.I = type { [3 x i32] }

// CHECK-DAG: [1 x %struct.M] [%struct.M { [2 x %struct.I] [%struct.I { [3 x i32] [i32 4, i32 4, i32 0] }, %struct.I { [3 x i32] [i32 4, i32 4, i32 5] }] }],
// CHECK-DAG: [2 x [3 x i32]] {{[[][[]}}3 x i32] [i32 2222, i32 2222, i32 0], [3 x i32] [i32 2222, i32 2222, i32 3333]],
// CHECK-DAG: [[INIT14:.*]] = private constant [16 x i32] [i32 0, i32 0, i32 0, i32 0, i32 0, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 0, i32 0, i32 0, i32 0], align 4

void f1(void) {
  // Scalars in braces.
  int a = { 1 };
}

void f2(void) {
  int a[2][2] = { { 1, 2 }, { 3, 4 } };
  int b[3][3] = { { 1, 2 }, { 3, 4 } };
  int *c[2] = { &a[1][1], &b[2][2] };
  int *d[2][2] = { {&a[1][1], &b[2][2]}, {&a[0][0], &b[1][1]} };
  int *e[3][3] = { {&a[1][1], &b[2][2]}, {&a[0][0], &b[1][1]} };
  char ext[3][3] = {".Y",".U",".V"};
}

typedef void (* F)(void);
extern void foo(void);
struct S { F f; };
void f3(void) {
  struct S a[1] = { { foo } };
}

// Constants
// CHECK-DAG: @g3 ={{.*}} constant i32 10
// CHECK-DAG: @f4.g4 = internal constant i32 12
const int g3 = 10;
int f4(void) {
  static const int g4 = 12;
  return g4;
}

// PR6537
typedef union vec3 {
  struct { double x, y, z; };
  double component[3];
} vec3;
vec3 f5(vec3 value) {
  return (vec3) {{
    .x = value.x
  }};
}

void f6(void) {
  int x;
  long ids[] = { (long) &x };  
}




// CHECK-DAG: @test7 ={{.*}} global{{.*}}{ i32 0, [4 x i8] c"bar\00" }
// PR8217
struct a7 {
  int  b;
  char v[];
};

struct a7 test7 = { .b = 0, .v = "bar" };


// CHECK-DAG: @huge_array ={{.*}} global {{.*}} <{ i32 1, i32 0, i32 2, i32 0, i32 3, [999999995 x i32] zeroinitializer }>
int huge_array[1000000000] = {1, 0, 2, 0, 3, 0, 0, 0};

// CHECK-DAG: @huge_struct ={{.*}} global {{.*}} { i32 1, <{ i32, [999999999 x i32] }> <{ i32 2, [999999999 x i32] zeroinitializer }> }
struct Huge {
  int a;
  int arr[1000 * 1000 * 1000];
} huge_struct = {1, {2, 0, 0, 0}};

// CHECK-DAG: @large_array_with_zeroes ={{.*}} constant <{ [21 x i8], [979 x i8] }> <{ [21 x i8] c"abc\01\02\03xyzzy\00\00\00\00\00\00\00\00\00q", [979 x i8] zeroinitializer }>
const char large_array_with_zeroes[1000] = {
  'a', 'b', 'c', 1, 2, 3, 'x', 'y', 'z', 'z', 'y', [20] = 'q'
};

char global;

// CHECK-DAG: @large_array_with_zeroes_2 ={{.*}} global <{ [10 x ptr], [90 x ptr] }> <{ [10 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @global], [90 x ptr] zeroinitializer }>
const void *large_array_with_zeroes_2[100] = {
  [9] = &global
};
// CHECK-DAG: @large_array_with_zeroes_3 ={{.*}} global <{ [10 x ptr], [990 x ptr] }> <{ [10 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr @global], [990 x ptr] zeroinitializer }>
const void *large_array_with_zeroes_3[1000] = {
  [9] = &global
};

// PR279 comment #3
char test8(int X) {
  char str[100000] = "abc"; // tail should be memset.
  return str[X];
  // CHECK-LABEL: @test8(
  // CHECK: call void @llvm.memset
  // CHECK: store i8 97, ptr %{{[0-9]*}}, align 1
  // CHECK: store i8 98, ptr %{{[0-9]*}}, align 1
  // CHECK: store i8 99, ptr %{{[0-9]*}}, align 1
  // CHECK-NOT: getelementptr
  // CHECK: load
}

void bar(void*);

// PR279
void test9(int X) {
  int Arr[100] = { X };     // Should use memset
  bar(Arr);
  // CHECK-LABEL: @test9(
  // CHECK: call void @llvm.memset
  // CHECK-NOT: store i32 0
  // CHECK: call void @bar
}

struct a {
  int a, b, c, d, e, f, g, h, i, j, k, *p;
};

struct b {
  struct a a,b,c,d,e,f,g;
};

void test10(int X) {
  struct b S = { .a.a = X, .d.e = X, .f.e = 0, .f.f = 0, .f.p = 0 };
  bar(&S);

  // CHECK-LABEL: @test10(
  // CHECK: call void @llvm.memset
  // CHECK-NOT: store i32 0
  // CHECK: call void @bar
}

void nonzeroMemseti8(void) {
  char arr[33] = { 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, };
  // CHECK-LABEL: @nonzeroMemseti8(
  // CHECK-NOT: store
  // CHECK-NOT: memcpy
  // CHECK: call void @llvm.memset.p0.i32(ptr {{.*}}, i8 42, i32 33, i1 false)
}

void nonzeroMemseti16(void) {
  unsigned short arr[17] = { 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, 0x4242, };
  // CHECK-LABEL: @nonzeroMemseti16(
  // CHECK-NOT: store
  // CHECK-NOT: memcpy
  // CHECK: call void @llvm.memset.p0.i32(ptr {{.*}}, i8 66, i32 34, i1 false)
}

void nonzeroMemseti32(void) {
  unsigned arr[9] = { 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, 0xF0F0F0F0, };
  // CHECK-LABEL: @nonzeroMemseti32(
  // CHECK-NOT: store
  // CHECK-NOT: memcpy
  // CHECK: call void @llvm.memset.p0.i32(ptr {{.*}}, i8 -16, i32 36, i1 false)
}

void nonzeroMemseti64(void) {
  unsigned long long arr[7] = { 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,  0xAAAAAAAAAAAAAAAA,  0xAAAAAAAAAAAAAAAA,  };
  // CHECK-LABEL: @nonzeroMemseti64(
  // CHECK-NOT: store
  // CHECK-NOT: memcpy
  // CHECK: call void @llvm.memset.p0.i32(ptr {{.*}}, i8 -86, i32 56, i1 false)
}

void nonzeroMemsetf32(void) {
  float arr[9] = { 0x1.cacacap+75, 0x1.cacacap+75, 0x1.cacacap+75, 0x1.cacacap+75, 0x1.cacacap+75, 0x1.cacacap+75, 0x1.cacacap+75, 0x1.cacacap+75, 0x1.cacacap+75, };
  // CHECK-LABEL: @nonzeroMemsetf32(
  // CHECK-NOT: store
  // CHECK-NOT: memcpy
  // CHECK: call void @llvm.memset.p0.i32(ptr {{.*}}, i8 101, i32 36, i1 false)
}

void nonzeroMemsetf64(void) {
  double arr[7] = { 0x1.4444444444444p+69, 0x1.4444444444444p+69, 0x1.4444444444444p+69, 0x1.4444444444444p+69, 0x1.4444444444444p+69, 0x1.4444444444444p+69, 0x1.4444444444444p+69, };
  // CHECK-LABEL: @nonzeroMemsetf64(
  // CHECK-NOT: store
  // CHECK-NOT: memcpy
  // CHECK: call void @llvm.memset.p0.i32(ptr {{.*}}, i8 68, i32 56, i1 false)
}

// PR9257
struct test11S {
  int A[10];
};
void test11(struct test11S *P) {
  *P = (struct test11S) { .A = { [0 ... 3] = 4 } };
  // CHECK-LABEL: @test11(
  // CHECK: store i32 4, ptr %{{.*}}, align 4
  // CHECK: store i32 4, ptr %{{.*}}, align 4
  // CHECK: store i32 4, ptr %{{.*}}, align 4
  // CHECK: store i32 4, ptr %{{.*}}, align 4
  // CHECK: ret void
}


// Verify that we can convert a recursive struct with a memory that returns
// an instance of the struct we're converting.
struct test12 {
  struct test12 (*p)(void);
} test12g;


void test13(int x) {
  struct X { int a; int b : 10; int c; };
  struct X y = {.c = x};
  // CHECK-LABEL: @test13(
  // CHECK: and i16 {{.*}}, -1024
}

// CHECK-LABEL: @PR20473(
void PR20473(void) {
  // CHECK: memcpy{{.*}}
  bar((char[2]) {""});
  // CHECK: memcpy{{.*}}
  bar((char[3]) {""});
}

// Test that we initialize large member arrays by copying from a global and not
// with a series of stores.
struct S14 { int a[16]; };

void test14(struct S14 *s14) {
  // CHECK-LABEL: @test14(
  // CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 {{.*}}, ptr align 4 [[INIT14]], i32 64, i1 false)
  // CHECK-NOT: store
  // CHECK: ret void
  *s14 = (struct S14) { { [5 ... 11] = 17 } };
}
