// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -Wno-strict-prototypes -emit-llvm -o - | FileCheck %s

// PR1895
// sizeof function
int zxcv(void);
int x=sizeof(zxcv);
int y=__alignof__(zxcv);


void *test(int *i) {
 short a = 1;
 i += a;
 i + a;
 a + i;
}

_Bool test2b; 
int test2(void) { if (test2b); return 0; }

// PR1921
int test3(void) {
  const unsigned char *bp;
  bp -= (short)1;
}

// PR2080 - sizeof void
int t1 = sizeof(void);
int t2 = __alignof__(void);
void test4(void) {
  t1 = sizeof(void);
  t2 = __alignof__(void);
  
  t1 = sizeof(test4());
  t2 = __alignof__(test4());
}

// 'const float' promotes to double in varargs.
int test5(const float x, float float_number) {
  return __builtin_isless(x, float_number);
}

// this one shouldn't fold
int ola(void) {
  int a=2;
  if ((0, (int)a) & 2) { return 1; }
  return 2;
}

// this one shouldn't fold as well
void eMaisUma(void) {
  double t[1];
  if (*t)
    return;
}

// rdar://6520707
void f0(void (*fp)(void), void (*fp2)(void)) {
  int x = fp - fp2;
}

// noop casts as lvalues.
struct X {
  int Y;
};
struct X foo();
int bar(void) {
  return ((struct X)foo()).Y + 1;
}

// PR3809: INC/DEC of function pointers.
void f2(void);
unsigned f1(void) {
  void (*fp)(void) = f2;
  
  ++fp;
  fp++;
  --fp;
  fp--;
  return (unsigned) fp;
}  

union f3_x {int x; float y;};
int f3(void) {return ((union f3_x)2).x;}

union f4_y {int x; _Complex float y;};
_Complex float f4(void) {return ((union f4_y)(_Complex float)2.0).y;}

struct f5_a { int a; } f5_a;
union f5_z {int x; struct f5_a y;};
struct f5_a f5(void) {return ((union f5_z)f5_a).y;}

// ?: in "lvalue"
struct s6 { int f0; };
int f6(int a0, struct s6 a1, struct s6 a2) {
  return (a0 ? a1 : a2).f0;
}

// PR4026
void f7(void) {
  __func__;
}

// PR4067
int f8(void) {
  return ({ foo(); }).Y;
}

// rdar://6880558
struct S;
struct C {
  int i;
  struct S *tab[];
};
struct S { struct C c; };
void f9(struct S *x) {
  foo(((void)1, x->c).tab[0]);
}

void f10(void) {
  __builtin_sin(0);
}

// rdar://7530813
// CHECK-LABEL: define{{.*}} i32 @f11
int f11(long X) {
  int A[100];
  return A[X];

// CHECK: [[Xaddr:%[^ ]+]] = alloca i64, align 8
// CHECK: [[A:%.*]] = alloca [100 x i32], align
// CHECK: [[X:%.*]] = load {{.*}}, ptr [[Xaddr]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [100 x i32], ptr [[A]], i64 0, i64 [[X]]
// CHECK-NEXT: load i32, ptr [[T0]], align 4
}

int f12(void) {
  // PR3150
  // CHECK-LABEL: define{{.*}} i32 @f12
  // CHECK: ret i32 1
  return 1||1;
}

// Make sure negate of fp uses -0.0 for proper -0 handling.
double f13(double X) {
  // CHECK-LABEL: define{{.*}} double @f13
  // CHECK: fneg double
  return -X;
}

// Check operations on incomplete types.
void f14(struct s14 *a) {
  (void) &*a;
}

// CHECK-LABEL: define{{.*}} void @f15
void f15(void) {
  extern void f15_start(void);
  f15_start();
  // CHECK: call void @f15_start()

  extern void *f15_v(void);
  extern const void *f15_cv(void);
  extern volatile void *f15_vv(void);
  *f15_v(); *f15_v(), *f15_v(); f15_v() ? *f15_v() : *f15_v();
  *f15_cv(); *f15_cv(), *f15_cv(); f15_cv() ? *f15_cv() : *f15_cv();
  *f15_vv(); *f15_vv(), *f15_vv(); f15_vv() ? *f15_vv() : *f15_vv();
  // CHECK-NOT: load
  // CHECK: ret void
}

// PR8967: this was crashing
// CHECK-LABEL: define{{.*}} void @f16()
void f16(void) {
  __extension__({ goto lbl; });
 lbl:
  ;
}

// PR13704: negative increment in i128 is not preserved.
// CHECK-LABEL: define{{.*}} void @f17()
void f17(void) {
  extern void extfunc(__int128);
  __int128 x = 2;
  x--;
  extfunc(x);
// CHECK: add nsw i128 %{{.}}, -1
}

// PR23597: We should evaluate union cast operands even if the cast is unused.
typedef union u {
    int i;
} strct;
int returns_int(void);
void f18(void) {
  (strct)returns_int();
}
// CHECK-LABEL: define{{.*}} void @f18()
// CHECK: call i32 @returns_int()

// Ensure the right stmt is returned
int f19(void) {
  return ({ 3;;4;; });
}
// CHECK-LABEL: define{{.*}} i32 @f19()
// CHECK: [[T:%.*]] = alloca i32
// CHECK: store i32 4, ptr [[T]]
// CHECK: [[L:%.*]] = load i32, ptr [[T]]
// CHECK: ret i32 [[L]]
