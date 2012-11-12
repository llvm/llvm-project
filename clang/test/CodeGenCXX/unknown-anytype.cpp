// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -funknown-anytype -emit-llvm -o - %s | FileCheck %s

int test0() {
  extern __unknown_anytype test0_any;
  // CHECK: load i32* @test0_any
  return (int) test0_any;
}

int test1() {
  extern __unknown_anytype test1_any();
  // CHECK: call i32 @_Z9test1_anyv()
  return (int) test1_any();
}

extern "C" __unknown_anytype test2_any(...);
float test2() {
  // CHECK: call float (...)* @test2_any(double {{[^,]+}})
  return (float) test2_any(0.5f);
}

extern "C" __unknown_anytype test2a_any(...);
float test2a() {
  // CHECK: call float (...)* @test2a_any(float {{[^,]+}})
  return (float) test2a_any((float) 0.5f);
}

float test3() {
  extern __unknown_anytype test3_any;
  // CHECK: [[FN:%.*]] = load float (i32)** @test3_any,
  // CHECK: call float [[FN]](i32 5)
  return ((float(*)(int)) test3_any)(5);
}

namespace test4 {
  extern __unknown_anytype test4_any1;
  extern __unknown_anytype test4_any2;

  int test() {
    // CHECK: load i32* @_ZN5test410test4_any1E
    // CHECK: load i8* @_ZN5test410test4_any2E
    return (int) test4_any1 + (char) test4_any2;
  }
}

extern "C" __unknown_anytype test5_any();
void test5() {
  // CHECK: call void @test5_any()
  return (void) test5_any();
}

extern "C" __unknown_anytype test6_any(float *);
long test6() {
  // CHECK: call i64 @test6_any(float* null)
  return (long) test6_any(0);
}

struct Test7 {
  ~Test7();
};
extern "C" __unknown_anytype test7_any(int);
Test7 test7() {
  // CHECK: call void @test7_any({{%.*}}* sret {{%.*}}, i32 5)
  return (Test7) test7_any(5);
}

struct Test8 {
  __unknown_anytype foo();
  __unknown_anytype foo(int);

  void test();
};
void Test8::test() {
  float f;
  // CHECK: call i32 @_ZN5Test83fooEv(
  f = (int) foo();
  // CHECK: call i32 @_ZN5Test83fooEi(
  f = (int) foo(5);
  // CHECK: call i32 @_ZN5Test83fooEv(
  f = (float) this->foo();
  // CHECK: call i32 @_ZN5Test83fooEi(
  f = (float) this->foo(5);
}
void test8(Test8 *p) {
  double d;
  // CHECK: call i32 @_ZN5Test83fooEv(
  d = (double) p->foo();
  // CHECK: call i32 @_ZN5Test83fooEi(
  d = (double) p->foo(5);
  // CHECK: call i32 @_ZN5Test83fooEv(
  d = (bool) (*p).foo();
  // CHECK: call i32 @_ZN5Test83fooEi(
  d = (bool) (*p).foo(5);
}

extern "C" __unknown_anytype test9_foo;
void *test9() {
  // CHECK: ret i8* bitcast (i32* @test9_foo to i8*)
  return (int*) &test9_foo;
}
