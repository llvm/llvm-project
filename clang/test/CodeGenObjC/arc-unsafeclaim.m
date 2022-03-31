//   Make sure it works on x86-64.
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime=macosx-10.11 -fobjc-arc -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-UNOPTIMIZED -check-prefix=NOTAIL-CALL

// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime=macosx-10.11 -fobjc-arc -emit-llvm -O2 -disable-llvm-passes -o - %s | FileCheck %s -check-prefix=ATTACHED-CALL

//   Make sure it works on x86-32.
// RUN: %clang_cc1 -triple i386-apple-darwin11 -fobjc-runtime=macosx-fragile-10.11 -fobjc-arc -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-UNOPTIMIZED -check-prefix=CHECK-MARKED -check-prefix=CALL

//   Make sure it works on ARM64.
// RUN: %clang_cc1 -triple arm64-apple-ios9 -fobjc-runtime=ios-9.0 -fobjc-arc -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-UNOPTIMIZED -check-prefix=CHECK-MARKED -check-prefix=CALL

//   Make sure it works on ARM.
// RUN: %clang_cc1 -triple armv7-apple-ios9 -fobjc-runtime=ios-9.0 -fobjc-arc -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-UNOPTIMIZED -check-prefix=CHECK-MARKED -check-prefix=CALL
// RUN: %clang_cc1 -triple armv7-apple-ios9 -fobjc-runtime=ios-9.0 -fobjc-arc -O -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-OPTIMIZED -check-prefix=CALL

//   Make sure that it's implicitly disabled if the runtime version isn't high enough.
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.10 -fobjc-arc -emit-llvm -o - %s | FileCheck %s -check-prefix=DISABLED
// RUN: %clang_cc1 -triple arm64-apple-ios8 -fobjc-runtime=ios-8 -fobjc-arc -emit-llvm -o - %s | FileCheck %s -check-prefix=DISABLED -check-prefix=DISABLED-MARKED

@class A;

A *makeA(void);

void test_assign(void) {
  __unsafe_unretained id x;
  x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_assign()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A:.*]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CALL-NEXT:            [[T2:%.*]] = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// DISABLED-LABEL:     define{{.*}} void @test_assign()
// DISABLED:             [[T0:%.*]] = call [[A:.*]]* @makeA()
// DISABLED-MARKED-NEXT: call void asm sideeffect
// DISABLED-NEXT:        [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// DISABLED-NEXT:        [[T2:%.*]] = {{.*}}call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T1]])

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_assign_assign(void) {
  __unsafe_unretained id x, y;
  x = y = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_assign_assign()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[Y:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CALL-NEXT:            [[T2:%.*]] = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           store i8* [[T4]], i8** [[Y]]
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_assign_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_strong_assign_assign(void) {
  __strong id x;
  __unsafe_unretained id y;
  x = y = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_strong_assign_assign()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[Y:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           store i8* [[T4]], i8** [[Y]]
// CHECK-NEXT:           [[OLD:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-NEXT:           call void @llvm.objc.release(i8* [[OLD]]
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(i8** [[X]], i8* null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(i8* [[T0]])
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_strong_assign_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_assign_strong_assign(void) {
  __unsafe_unretained id x;
  __strong id y;
  x = y = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_assign_strong_assign()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[Y:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           [[OLD:%.*]] = load i8*, i8** [[Y]]
// CHECK-NEXT:           store i8* [[T4]], i8** [[Y]]
// CHECK-NEXT:           call void @llvm.objc.release(i8* [[OLD]]
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(i8** [[Y]], i8* null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(i8* [[T0]])
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_assign_strong_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_init(void) {
  __unsafe_unretained id x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_init()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CALL-NEXT:            [[T2:%.*]] = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_init()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_init_assignment(void) {
  __unsafe_unretained id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_init_assignment()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[Y:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CALL-NEXT:            [[T2:%.*]] = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-NEXT:           store i8* [[T4]], i8** [[Y]]
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT: ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_init_assignment()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_strong_init_assignment(void) {
  __unsafe_unretained id x;
  __strong id y = x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_strong_init_assignment()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[Y:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-NEXT:           store i8* [[T4]], i8** [[Y]]
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(i8** [[Y]], i8* null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(i8* [[T0]])
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT: ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_strong_init_assignment()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_init_strong_assignment(void) {
  __strong id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_init_strong_assignment()
// CHECK:                [[X:%.*]] = alloca i8*
// CHECK:                [[Y:%.*]] = alloca i8*
// CHECK:                [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:           [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:           [[T4:%.*]] = bitcast [[A]]* [[T3]] to i8*
// CHECK-NEXT:           [[OLD:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT:           store i8* [[T4]], i8** [[X]]
// CHECK-NEXT:           call void @llvm.objc.release(i8* [[OLD]])
// CHECK-NEXT:           store i8* [[T4]], i8** [[Y]]
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(i8** [[X]], i8* null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(i8* [[T0]])
// CHECK-OPTIMIZED-NEXT: bitcast
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT: ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_init_strong_assignment()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_ignored(void) {
  makeA();
}
// CHECK-LABEL:     define{{.*}} void @test_ignored()
// CHECK:             [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT: call void asm sideeffect
// CHECK-NEXT:        [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// NOTAIL-CALL-NEXT:  [[T2:%.*]] = notail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CALL-NEXT:         [[T2:%.*]] = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:        bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:        ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_ignored()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])

void test_cast_to_void(void) {
  (void) makeA();
}
// CHECK-LABEL:     define{{.*}} void @test_cast_to_void()
// CHECK:             [[T0:%.*]] = call [[A]]* @makeA()
// CHECK-MARKED-NEXT: call void asm sideeffect
// CHECK-NEXT:        [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// NOTAIL-CALL-NEXT:  [[T2:%.*]] = notail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CALL-NEXT:         [[T2:%.*]] = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* [[T1]])
// CHECK-NEXT:        bitcast i8* [[T2]] to [[A]]*
// CHECK-NEXT:        ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_cast_to_void()
// ATTACHED-CALL:              [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use([[A]]* [[T0]])


// This is always at the end of the module.

// CHECK-OPTIMIZED: !llvm.module.flags = !{!0,
// CHECK-OPTIMIZED: !0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov{{.*}}marker for objc_retainAutoreleaseReturnValue"}
