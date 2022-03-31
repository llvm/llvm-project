// RUN: %clang_cc1 -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.8 -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s -check-prefix=CHECK-WITH
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.7 -emit-llvm -fexceptions -fobjc-exceptions -o - %s | FileCheck %s -check-prefix=CHECK-WITHOUT

void destroy(void**);

// rdar://problem/9519113
void test0(void) {
  void test0_helper(void);
  void *ptr __attribute__((cleanup(destroy)));
  test0_helper();

  // CHECK-WITH-LABEL:       define{{.*}} void @test0()
  // CHECK-WITH-SAME:    personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
  // CHECK-WITH:         [[PTR:%.*]] = alloca i8*,
  // CHECK-WITH:         call void @destroy(i8** noundef [[PTR]])
  // CHECK-WITH-NEXT:    ret void
  // CHECK-WITH:         invoke void @destroy(i8** noundef [[PTR]])
  // CHECK-WITH:         landingpad { i8*, i32 }
  // CHECK-WITH-NEXT:      catch i8* null
  // CHECK-WITH-NEXT:    call void @objc_terminate()

  // CHECK-WITHOUT-LABEL:    define{{.*}} void @test0()
  // CHECK-WITHOUT-SAME: personality i8* bitcast (i32 (...)* @__gcc_personality_v0 to i8*)
  // CHECK-WITHOUT:      [[PTR:%.*]] = alloca i8*,
  // CHECK-WITHOUT:      call void @destroy(i8** noundef [[PTR]])
  // CHECK-WITHOUT-NEXT: ret void
  // CHECK-WITHOUT:      invoke void @destroy(i8** noundef [[PTR]])
  // CHECK-WITHOUT:      landingpad { i8*, i32 }
  // CHECK-WITHOUT-NEXT:   catch i8* null
  // CHECK-WITHOUT-NEXT: call void @abort()
}
