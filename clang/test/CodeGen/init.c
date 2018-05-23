// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

struct I { int k[3]; };
struct M { struct I o[2]; };
struct M v1[1] = { [0].o[0 ... 1].k[0 ... 1] = 4, 5 };
unsigned v2[2][3] = {[0 ... 1][0 ... 1] = 2222, 3333};

// CHECK-DAG: %struct.M = type { [2 x %struct.I] }
// CHECK-DAG: %struct.I = type { [3 x i32] }

// CHECK-DAG: [1 x %struct.M] [%struct.M { [2 x %struct.I] [%struct.I { [3 x i32] [i32 4, i32 4, i32 0] }, %struct.I { [3 x i32] [i32 4, i32 4, i32 5] }] }],
// CHECK-DAG: [2 x [3 x i32]] {{[[][[]}}3 x i32] [i32 2222, i32 2222, i32 0], [3 x i32] [i32 2222, i32 2222, i32 3333]],
// CHECK-DAG: [[INIT14:.*]] = private global [16 x i32] [i32 0, i32 0, i32 0, i32 0, i32 0, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 17, i32 0, i32 0, i32 0, i32 0], align 4

void f1() {
  // Scalars in braces.
  int a = { 1 };
}

void f2() {
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
void f3() {
  struct S a[1] = { { foo } };
}

// Constants
// CHECK-DAG: @g3 = constant i32 10
// CHECK-DAG: @f4.g4 = internal constant i32 12
const int g3 = 10;
int f4() {
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

// rdar://problem/8154689
void f6() {
  int x;
  long ids[] = { (long) &x };  
}




// CHECK-DAG: @test7 = global{{.*}}{ i32 0, [4 x i8] c"bar\00" }
// PR8217
struct a7 {
  int  b;
  char v[];
};

struct a7 test7 = { .b = 0, .v = "bar" };


// CHECK-DAG: @huge_array = global {{.*}} <{ i32 1, i32 0, i32 2, i32 0, i32 3, [999999995 x i32] zeroinitializer }>
int huge_array[1000000000] = {1, 0, 2, 0, 3, 0, 0, 0};

// CHECK-DAG: @huge_struct = global {{.*}} { i32 1, <{ i32, [999999999 x i32] }> <{ i32 2, [999999999 x i32] zeroinitializer }> }
struct Huge {
  int a;
  int arr[1000 * 1000 * 1000];
} huge_struct = {1, {2, 0, 0, 0}};


// PR279 comment #3
char test8(int X) {
  char str[100000] = "abc"; // tail should be memset.
  return str[X];
// CHECK: @test8(
// CHECK: call void @llvm.memset
// CHECK: store i8 97
// CHECK: store i8 98
// CHECK: store i8 99
// CHECK-NOT: getelementptr
// CHECK: load
}

void bar(void*);

// PR279
int test9(int X) {
  int Arr[100] = { X };     // Should use memset
  bar(Arr);
// CHECK: @test9
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

int test10(int X) {
  struct b S = { .a.a = X, .d.e = X, .f.e = 0, .f.f = 0, .f.p = 0 };
  bar(&S);

  // CHECK: @test10
  // CHECK: call void @llvm.memset
  // CHECK-NOT: store i32 0
  // CHECK: call void @bar
}


// PR9257
struct test11S {
  int A[10];
};
void test11(struct test11S *P) {
  *P = (struct test11S) { .A = { [0 ... 3] = 4 } };
  // CHECK: @test11
  // CHECK: store i32 4
  // CHECK: store i32 4
  // CHECK: store i32 4
  // CHECK: store i32 4
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
  // CHECK: @test13
  // CHECK: and i16 {{.*}}, -1024
}

// CHECK-LABEL: @PR20473
void PR20473() {
  // CHECK: memcpy{{.*}}getelementptr inbounds ([2 x i8], [2 x i8]* @
  bar((char[2]) {""});
  // CHECK: memcpy{{.*}}getelementptr inbounds ([3 x i8], [3 x i8]* @
  bar((char[3]) {""});
}

// Test that we initialize large member arrays by copying from a global and not
// with a series of stores.
struct S14 { int a[16]; };

void test14(struct S14 *s14) {
// CHECK-LABEL: @test14
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 {{.*}}, i8* align 4 {{.*}} [[INIT14]] {{.*}}, i32 64, i1 false)
// CHECK-NOT: store
// CHECK: ret void
  *s14 = (struct S14) { { [5 ... 11] = 17 } };
}
