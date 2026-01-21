// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm -fobjc-arc -fobjc-exceptions -o - %s | FileCheck %s
// pr10411

id make(void);
void test(void) { 
  @throw make();
}

// TODO: We should probably emit this specific pattern without the reclaim.

// CHECK-LABEL:    define{{.*}} void @test()
// CHECK:      %call1 = call ptr @make() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK:      call void (...) @llvm.objc.clang.arc.noop.use(ptr %call1) #2
// CHECK:      %0 = call ptr @llvm.objc.autorelease(ptr %call1) #2
// CHECK:      call void @objc_exception_throw(ptr %0) [[NR:#[0-9]+]]
// CHECK-NEXT: unreachable

// CHECK: attributes [[NR]] = { noreturn }
