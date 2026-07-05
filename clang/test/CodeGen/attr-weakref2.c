// RUN: %clang_cc1 -emit-llvm -triple i386-linux-gnu -o %t %s
// RUN: FileCheck --input-file=%t %s

// CHECK: @test1_f = extern_weak global i32
extern int test1_f;
static int test1_g __attribute__((weakref("test1_f")));
int test1_h(void) {
  return test1_g;
}

// CHECK: @test2_f = global i32 0, align 4
int test2_f;
static int test2_g __attribute__((weakref("test2_f")));
int test2_h(void) {
  return test2_g;
}

// CHECK: @test3_f = external global i32
extern int test3_f;
static int test3_g __attribute__((weakref("test3_f")));
int test3_foo(void) {
  return test3_f;
}
int test3_h(void) {
  return test3_g;
}

// CHECK: @test4_f = global i32 0, align 4
extern int test4_f;
static int test4_g __attribute__((weakref("test4_f")));
int test4_h(void) {
  return test4_g;
}
int test4_f;

// CHECK: @test5_f = external global i32
extern int test5_f;
static int test5_g __attribute__((weakref("test5_f")));
int test5_h(void) {
  return test5_g;
}
int test5_foo(void) {
  return test5_f;
}

// CHECK: @test6_f = extern_weak global i32
extern int test6_f __attribute__((weak));
static int test6_g __attribute__((weakref("test6_f")));
int test6_h(void) {
  return test6_g;
}
int test6_foo(void) {
  return test6_f;
}
