// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-optzns -o - %s -O2 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-optzns -o - %s -O0 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-optzns -o - %s -O2 -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-optzns -o - %s -O0 -fexperimental-new-constant-interpreter | FileCheck %s


int a = 42;

/* --- Compound literals */

struct foo { int x, y; };

int y;
struct foo f = (struct foo){ __builtin_constant_p(y), 42 };

struct foo test0(int expr) {
  // CHECK-LABEL: test0
  // CHECK: call i1 @llvm.is.constant.i32
  struct foo f = (struct foo){ __builtin_constant_p(expr), 42 };
  return f;
}

/* --- Pointer types */

int test1(void) {
  // CHECK-LABEL: test1
  // CHECK: ret i32 0
  return __builtin_constant_p(&a - 13);
}

/* --- Aggregate types */

int b[] = {1, 2, 3};

int test2(void) {
  // CHECK-LABEL: test2
  // CHECK: ret i32 0
  return __builtin_constant_p(b);
}

const char test3_c[] = {1, 2, 3, 0};

int test3(void) {
  // CHECK-LABEL: test3
  // CHECK: ret i32 0
  return __builtin_constant_p(test3_c);
}

inline char test4_i(const char *x) {
  return x[1];
}

int test4(void) {
  // CHECK: define{{.*}} i32 @test4
  // CHECK: ret i32 0
  return __builtin_constant_p(test4_i(test3_c));
}

/* --- Constant global variables */

const int c = 42;

int test5(void) {
  // CHECK-LABEL: test5
  // CHECK: ret i32 1
  return __builtin_constant_p(c);
}

/* --- Array types */

int arr[] = { 1, 2, 3 };
const int c_arr[] = { 1, 2, 3 };

int test6(void) {
  // CHECK-LABEL: test6
  // CHECK: call i1 @llvm.is.constant.i32
  return __builtin_constant_p(arr[2]);
}

int test7(void) {
  // CHECK-LABEL: test7
  // CHECK: ret i32 1
  return __builtin_constant_p(c_arr[2]);
}

int test8(void) {
  // CHECK-LABEL: test8
  // CHECK: ret i32 0
  return __builtin_constant_p(c_arr);
}

/* --- Function pointers */

int test9(void) {
  // CHECK-LABEL: test9
  // CHECK: ret i32 0
  return __builtin_constant_p(&test9);
}

int test10(void) {
  // CHECK-LABEL: test10
  // CHECK: ret i32 1
  return __builtin_constant_p(&test10 != 0);
}

typedef unsigned long uintptr_t;
#define assign(p, v) ({ \
  uintptr_t _r_a_p__v = (uintptr_t)(v);                           \
  if (__builtin_constant_p(v) && _r_a_p__v == (uintptr_t)0) {     \
    union {                                                       \
      uintptr_t __val;                                            \
      char __c[1];                                                \
    } __u = {                                                     \
      .__val = (uintptr_t)_r_a_p__v                               \
    };                                                            \
    *(volatile unsigned int*)&p = *(unsigned int*)(__u.__c);      \
    __u.__val;                                                    \
  }                                                               \
  _r_a_p__v;                                                      \
})

typedef void fn_p(void);
extern fn_p *dest_p;

static void src_fn(void) {
}

void test11(void) {
  assign(dest_p, src_fn);
}

extern int test12_v;

struct { const char *t; int a; } test12[] = {
    { "tag", __builtin_constant_p(test12_v) && !test12_v ? 1 : 0 }
};

extern char test13_v;
struct { int a; } test13 = { __builtin_constant_p(test13_v) };

extern unsigned long long test14_v;

void test14(void) {
  // CHECK-LABEL: test14
  // CHECK: call void asm sideeffect "", {{.*}}(i32 -1) 
  __asm__ __volatile__("" :: "n"( (__builtin_constant_p(test14_v) || 0) ? 1 : -1));
}

int test15_f(void);
// CHECK-LABEL: define{{.*}} void @test15
// CHECK-NOT: call {{.*}}test15_f
void test15(void) {
  int a, b;
  (void)__builtin_constant_p((a = b, test15_f()));
}

/* --- Complex types (GH119126) */

__complex double test_complex_cond(void) {
  // CHECK-LABEL: test_complex_cond
  // CHECK: [[R:%.*]] = call i1 @llvm.is.constant.f64(double %f.real)
  // CHECK: [[I:%.*]] = call i1 @llvm.is.constant.f64(double %f.imag)
  // CHECK: and i1 [[R]], [[I]]
  __complex double f;
  return (__builtin_constant_p(f) ? f : 0);
}

int test_complex_float(_Complex float x) {
  // CHECK-LABEL: test_complex_float
  // CHECK: [[R:%.*]] = call i1 @llvm.is.constant.f32(float %x.real)
  // CHECK: [[I:%.*]] = call i1 @llvm.is.constant.f32(float %x.imag)
  // CHECK: [[AND:%.*]] = and i1 [[R]], [[I]]
  // CHECK: zext i1 [[AND]] to i32
  int r = __builtin_constant_p(x);
  return r;
}

int test_complex_long_double(_Complex long double x) {
  // CHECK-LABEL: test_complex_long_double
  // CHECK: [[R:%.*]] = call i1 @llvm.is.constant.f80(x86_fp80 %x.real)
  // CHECK: [[I:%.*]] = call i1 @llvm.is.constant.f80(x86_fp80 %x.imag)
  // CHECK: and i1 [[R]], [[I]]
  return __builtin_constant_p(x);
}

int test_complex_float16(_Complex _Float16 x) {
  // CHECK-LABEL: test_complex_float16
  // CHECK: [[R:%.*]] = call i1 @llvm.is.constant.f16(half %x.real)
  // CHECK: [[I:%.*]] = call i1 @llvm.is.constant.f16(half %x.imag)
  // CHECK: and i1 [[R]], [[I]]
  return __builtin_constant_p(x);
}

int test_complex_int(_Complex int x) {
  // CHECK-LABEL: test_complex_int
  // CHECK: [[R:%.*]] = call i1 @llvm.is.constant.i32(i32 %x.real)
  // CHECK: [[I:%.*]] = call i1 @llvm.is.constant.i32(i32 %x.imag)
  // CHECK: and i1 [[R]], [[I]]
  return __builtin_constant_p(x);
}

int test_complex_const1(void) {
  // CHECK-LABEL: test_complex_const1
  // CHECK: ret i32 1
  return __builtin_constant_p(1.0i);
}

int test_complex_const2(void) {
  // CHECK-LABEL: test_complex_const2
  // CHECK: ret i32 1
  return __builtin_constant_p(__builtin_complex(1.0f, 2.0f));
}

const _Complex long double c_cld = __builtin_complex(1.0L, 2.0L);

int test_complex_const3(void) {
  // CHECK-LABEL: test_complex_const3
  // CHECK: ret i32 1
  return __builtin_constant_p(c_cld);
}

_Complex double test_complex_f(void);

int test_complex_side_effect(void) {
  // CHECK-LABEL: test_complex_side_effect
  // CHECK-NOT: test_complex_f
  // CHECK-NOT: llvm.is.constant
  // CHECK: ret i32 0
  return __builtin_constant_p(test_complex_f());
}

/* --- Scalar controls */

int test_double(double x) {
  // CHECK-LABEL: test_double
  // CHECK: call i1 @llvm.is.constant.f64(double %
  // CHECK-NOT: and i1
  // CHECK: ret i32
  return __builtin_constant_p(x);
}

enum test_enum_t { TEST_ENUM_A };

int test_enum(enum test_enum_t x) {
  // CHECK-LABEL: test_enum
  // CHECK: call i1 @llvm.is.constant.i32(i32 %
  return __builtin_constant_p(x);
}

int test_bitint(_BitInt(37) x) {
  // CHECK-LABEL: test_bitint
  // CHECK: call i1 @llvm.is.constant.i37(i37 %
  return __builtin_constant_p(x);
}

int test_pointer(int *p) {
  // CHECK-LABEL: test_pointer
  // CHECK: ret i32 0
  return __builtin_constant_p(p);
}
