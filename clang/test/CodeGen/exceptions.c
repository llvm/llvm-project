// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-strict-prototypes -emit-llvm -o - %s -fexceptions -fblocks | FileCheck %s
// RUN: %clang_cc1 -triple armv7-apple-unknown -Wno-strict-prototypes -emit-llvm -o - %s -fexceptions -exception-model=sjlj -fblocks | FileCheck %s -check-prefix=CHECK-ARM

void test1(void) {
  extern void test1_helper(void (^)(int));

  // CHECK-LABEL:     define{{.*}} void @test1() {{.*}} personality ptr @__gcc_personality_v0
  // CHECK-ARM-LABEL: define{{.*}} arm_aapcscc void @test1() {{.*}} personality ptr @__gcc_personality_sj0

  __block int x = 10;

  // CHECK:     invoke void @test1_helper(
  // CHECK-ARM: invoke arm_aapcscc void @test1_helper(
  test1_helper(^(int v) { x = v; });

  // CHECK:          landingpad { ptr, i32 }
  // CHECK-NEXT:       cleanup
  // CHECK-ARM:      landingpad { ptr, i32 }
  // CHECK-ARM-NEXT:   cleanup
}

void test2_helper();
void test2(void) {
  __block int x = 10;
  ^{ (void)x; };
  test2_helper(5, 6, 7);
}
void test2_helper(int x, int y) {
}
// CHECK: invoke void @test2_helper(i32 noundef 5, i32 noundef 6)
