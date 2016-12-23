// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

// PR14355: don't crash
// Keep this test in its own file because CodeGenTypes has global state.
// CHECK: define void @test10_foo({}* %p1.coerce) [[NUW:#[0-9]+]] {
struct test10_B;
typedef struct test10_B test10_F3(double);
void test10_foo(test10_F3 p1);
struct test10_B test10_b(double);
void test10_bar() {
  test10_foo(test10_b);
}
struct test10_B {};
void test10_foo(test10_F3 p1)
{
  p1(0.0);
}

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
