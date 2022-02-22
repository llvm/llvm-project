// RUN: %clang_cc1 -w -fmerge-all-constants -emit-llvm < %s | FileCheck %s

// CHECK: @test1.x = internal constant [12 x i32] [i32 1
// CHECK: @__const.test2.x = private unnamed_addr constant [13 x i32] [i32 1,
// CHECK: @test5w = {{(dso_local )?}}global { i32, [4 x i8] } { i32 2, [4 x i8] undef }
// CHECK: @test5y = {{(dso_local )?}}global { double } { double 7.300000e+0{{[0]*}}1 }

// CHECK: @__const.test6.x = private unnamed_addr constant %struct.SelectDest { i8 1, i8 2, i32 3, i32 0 }

// CHECK: @test7 = {{(dso_local )?}}global [2 x %struct.test7s] [%struct.test7s { i32 1, i32 2 }, %struct.test7s { i32 4, i32 0 }]

void test1(void) {
  // This should codegen as a "@test1.x" global.
  const int x[] = { 1, 2, 3, 4, 6, 8, 9, 10, 123, 231, 123,23 };
  foo(x);

// CHECK: @test1()
// CHECK: {{call.*@foo.*@test1.x}}
}


// rdar://7346691
void test2(void) {
  // This should codegen as a "@test2.x" global + memcpy.
  int x[] = { 1, 2, 3, 4, 6, 8, 9, 10, 123, 231, 123,23, 24 };
  foo(x);

  // CHECK: @test2()
  // CHECK: %x = alloca [13 x i32]
  // CHECK: call void @llvm.memcpy
  // CHECK: call{{.*}}@foo{{.*}}i32* noundef %
}


void test3(void) {
  // This should codegen as a memset.
  int x[100] = { 0 };
  foo(x);

  // CHECK: @test3()
  // CHECK: %x = alloca [100 x i32]
  // CHECK: call void @llvm.memset
}

void test4(void) {
  char a[10] = "asdf";
  char b[10] = { "asdf" };
  // CHECK: @test4()
  // CHECK: %a = alloca [10 x i8]
  // CHECK: %b = alloca [10 x i8]
  // CHECK: call void @llvm.memcpy
  // CHECK: call void @llvm.memcpy
}


union test5u { int i; double d; };

void test5(void) {
  union test5u ola = (union test5u) 351;
  union test5u olb = (union test5u) 1.0;
}

union test5u test5w = (union test5u)2;
union test5u test5y = (union test5u)73.0;



// PR6660 - sqlite miscompile
struct SelectDest {
  unsigned char eDest;
  unsigned char affinity;
  int iParm;
  int iMem;
};

void test6(void) {
  struct SelectDest x = {1, 2, 3};
  test6f(&x);
}

// rdar://7657600
struct test7s { int a; int b; } test7[] = {
  {1, 2},
  {4},
};

// rdar://7872531
#pragma pack(push, 2)
struct test8s { int f0; char f1; } test8g = {};


// PR7519

struct S {
  void (*x) (struct S *);
};

extern struct S *global_dc;
void cp_diagnostic_starter(struct S *);

void init_error(void) {
  global_dc->x = cp_diagnostic_starter;
}



// rdar://8147692 - ABI crash in recursive struct-through-function-pointer.
typedef struct {
  int x5a;
} x5;

typedef struct x2 *x0;
typedef long (*x1)(x0 x0a, x5 x6);
struct x2 {
  x1 x4;
};
long x3(x0 x0a, x5 a) {
  return x0a->x4(x0a, a);
}
