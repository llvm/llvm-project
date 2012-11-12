// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-exceptions -o - %s | FileCheck %s
// pr10411
// rdar://10042689

id make(void);
void test() { 
  @throw make();
}

// TODO: We should probably emit this specific pattern without the reclaim.

// CHECK:    define void @test()
// CHECK:      [[T0:%.*]] = call i8* @make()
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[T0]])
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_autorelease(i8* [[T1]])
// CHECK-NEXT: call void @objc_exception_throw(i8* [[T2]]) noreturn
// CHECK-NEXT: unreachable
