// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-exceptions -o - %s | FileCheck %s
// pr10411

id make(void);
void test(void) { 
  @throw make();
}

// TODO: We should probably emit this specific pattern without the reclaim.

// CHECK-LABEL:    define{{.*}} void @test()
// CHECK:      [[T0:%.*]] = call ptr @make()
// CHECK-NEXT: [[T1:%.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.autorelease(ptr [[T1]])
// CHECK-NEXT: call void @objc_exception_throw(ptr [[T2]]) [[NR:#[0-9]+]]
// CHECK-NEXT: unreachable

// CHECK: attributes [[NR]] = { noreturn }
