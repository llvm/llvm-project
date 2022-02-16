// RUN: %clang_cc1 -triple armv7-apple-unknown -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s -fexceptions -exception-model=sjlj -fblocks | FileCheck %s

// Verify strictfp attributes on invoke calls (and therefore also on
// function definitions).

// rdar://problem/8621849
void test1(void) {
  extern void test1_helper(void (^)(int));

  // CHECK: define{{.*}} arm_aapcscc void @test1() [[STRICTFP0:#[0-9]+]] personality i8* bitcast (i32 (...)* @__gcc_personality_sj0 to i8*)

  __block int x = 10;

  // CHECK: invoke arm_aapcscc void @test1_helper({{.*}}) [[STRICTFP1:#[0-9]+]]
  test1_helper(^(int v) { x = v; });

  // CHECK:      landingpad { i8*, i32 }
  // CHECK-NEXT:   cleanup
}

void test2_helper();
void test2(void) {
  // CHECK: define{{.*}} arm_aapcscc void @test2() [[STRICTFP0]] personality i8* bitcast (i32 (...)* @__gcc_personality_sj0 to i8*) {
  __block int x = 10;
  ^{ (void)x; };

  // CHECK: invoke arm_aapcscc void @test2_helper({{.*}}) [[STRICTFP1:#[0-9]+]]
  test2_helper(5, 6, 7);

  // CHECK:      landingpad { i8*, i32 }
  // CHECK-NEXT:   cleanup
}
void test2_helper(int x, int y) {
}

// CHECK: attributes [[STRICTFP0]] = { {{.*}}strictfp{{.*}} }
// CHECK: attributes [[STRICTFP1]] = { strictfp }
