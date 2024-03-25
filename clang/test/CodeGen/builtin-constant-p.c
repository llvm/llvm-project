// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-optzns -o - %s -O2 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -disable-llvm-optzns -o - %s -O0 | FileCheck %s

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
