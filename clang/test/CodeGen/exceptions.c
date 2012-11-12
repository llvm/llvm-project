// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fexceptions -fblocks | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-unknown -emit-llvm -o - %s -fexceptions -fsjlj-exceptions -fblocks | FileCheck %s -check-prefix=CHECK-ARM

// rdar://problem/8621849
void test1() {
  extern void test1_helper(void (^)(int));

  // CHECK:     define void @test1()
  // CHECK-ARM: define arm_aapcscc void @test1()

  __block int x = 10;

  // CHECK:     invoke void @test1_helper(
  // CHECK-ARM: invoke arm_aapcscc void @test1_helper(
  test1_helper(^(int v) { x = v; });

  // CHECK:          landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
  // CHECK-NEXT:       cleanup
  // CHECK-ARM:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gcc_personality_sj0 to i8*)
  // CHECK-ARM-NEXT:   cleanup
}
