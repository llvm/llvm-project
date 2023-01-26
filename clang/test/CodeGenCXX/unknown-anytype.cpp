// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -funknown-anytype -emit-llvm -o %t %s
// RUN: FileCheck -check-prefix COMMON %s < %t
// RUN: FileCheck -check-prefix X86_64 %s < %t
// RUN: %clang_cc1 -triple i386-apple-darwin10 -funknown-anytype -emit-llvm -o %t %s
// RUN: FileCheck -check-prefix COMMON %s < %t
// RUN: FileCheck -check-prefix I386 %s < %t

// x86-64 is the special case here because of its variadic convention.
// We want to ensure that it always uses a variadic convention even if
// other platforms do not.
// rdar://13731520

int test0() {
  extern __unknown_anytype test0_any;
  // COMMON: load i32, ptr @test0_any
  return (int) test0_any;
}

int test1() {
  extern __unknown_anytype test1_any();
  // COMMON: call noundef i32 @_Z9test1_anyv()
  return (int) test1_any();
}

extern "C" __unknown_anytype test2_any(...);
float test2() {
  // X86_64: call float (double, ...) @test2_any(double {{[^,]+}})
  // I386: call float (double, ...) @test2_any(double {{[^,]+}})
  return (float) test2_any(0.5f);
}

extern "C" __unknown_anytype test2a_any(...);
float test2a() {
  // X86_64: call float (float, ...) @test2a_any(float {{[^,]+}})
  // I386: call float (float, ...) @test2a_any(float {{[^,]+}})
  return (float) test2a_any((float) 0.5f);
}

float test3() {
  extern __unknown_anytype test3_any;
  // COMMON: [[FN:%.*]] = load ptr, ptr @test3_any,
  // COMMON: call noundef float [[FN]](i32 noundef 5)
  return ((float(*)(int)) test3_any)(5);
}

namespace test4 {
  extern __unknown_anytype test4_any1;
  extern __unknown_anytype test4_any2;

  int test() {
    // COMMON: load i32, ptr @_ZN5test410test4_any1E
    // COMMON: load i8, ptr @_ZN5test410test4_any2E
    return (int) test4_any1 + (char) test4_any2;
  }
}

extern "C" __unknown_anytype test5_any();
void test5() {
  // COMMON: call void @test5_any()
  return (void) test5_any();
}

extern "C" __unknown_anytype test6_any(float *);
long test6() {
  // COMMON: call i64 @test6_any(ptr noundef null)
  return (long long) test6_any(0);
}

struct Test7 {
  ~Test7();
};
extern "C" __unknown_anytype test7_any(int);
Test7 test7() {
  // COMMON: call void @test7_any(ptr sret({{%.*}}) align 1 {{%.*}}, i32 noundef 5)
  return (Test7) test7_any(5);
}

struct Test8 {
  __unknown_anytype foo();
  __unknown_anytype foo(int);

  void test();
};
void Test8::test() {
  float f;
  // COMMON: call noundef i32 @_ZN5Test83fooEv(
  f = (int) foo();
  // COMMON: call noundef i32 @_ZN5Test83fooEi(
  f = (int) foo(5);
  // COMMON: call noundef i32 @_ZN5Test83fooEv(
  f = (float) this->foo();
  // COMMON: call noundef i32 @_ZN5Test83fooEi(
  f = (float) this->foo(5);
}
void test8(Test8 *p) {
  double d;
  // COMMON: call noundef i32 @_ZN5Test83fooEv(
  d = (double) p->foo();
  // COMMON: call noundef i32 @_ZN5Test83fooEi(
  d = (double) p->foo(5);
  // COMMON: call noundef i32 @_ZN5Test83fooEv(
  d = (bool) (*p).foo();
  // COMMON: call noundef i32 @_ZN5Test83fooEi(
  d = (bool) (*p).foo(5);
}

extern "C" __unknown_anytype test9_foo;
void *test9() {
  // COMMON: ret ptr @test9_foo
  return (int*) &test9_foo;
}

// Don't explode on this.
extern "C" __unknown_anytype test10_any(...);
void test10() {
  (void) test10_any(), (void) test10_any();
}

extern "C" __unknown_anytype malloc(...);
void test11() {
  void *s = (void*)malloc(12);
  // COMMON: call ptr (i32, ...) @malloc(i32 noundef 12)
  void *d = (void*)malloc(435);
  // COMMON: call ptr (i32, ...) @malloc(i32 noundef 435)
}
