// RUN: %clang_cc1 -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.8 -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s -check-prefix=CHECK-WITH
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.7 -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s -check-prefix=CHECK-WITHOUT

void destroy(void**);

void test0(void) {
  void test0_helper(void);
  void *ptr __attribute__((cleanup(destroy)));
  test0_helper();

  // CHECK-WITH-LABEL:       define{{.*}} void @test0()
  // CHECK-WITH-SAME:    personality ptr @__gcc_personality_v0
  // CHECK-WITH:         [[PTR:%.*]] = alloca ptr,
  // CHECK-WITH:         call void @destroy(ptr noundef [[PTR]])
  // CHECK-WITH-NEXT:    ret void
  // CHECK-WITH:         invoke void @destroy(ptr noundef [[PTR]])
  // CHECK-WITH:         landingpad { ptr, i32 }
  // CHECK-WITH-NEXT:      catch ptr null
  // CHECK-WITH-NEXT:    call void @objc_terminate()

  // CHECK-WITHOUT-LABEL:    define{{.*}} void @test0()
  // CHECK-WITHOUT-SAME: personality ptr @__gcc_personality_v0
  // CHECK-WITHOUT:      [[PTR:%.*]] = alloca ptr,
  // CHECK-WITHOUT:      call void @destroy(ptr noundef [[PTR]])
  // CHECK-WITHOUT-NEXT: ret void
  // CHECK-WITHOUT:      invoke void @destroy(ptr noundef [[PTR]])
  // CHECK-WITHOUT:      landingpad { ptr, i32 }
  // CHECK-WITHOUT-NEXT:   catch ptr null
  // CHECK-WITHOUT-NEXT: call void @abort()
}
