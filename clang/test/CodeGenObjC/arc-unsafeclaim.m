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
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CALL-NEXT:            [[T2:%.*]] = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// DISABLED-LABEL:     define{{.*}} void @test_assign()
// DISABLED:             [[T0:%.*]] = call ptr @makeA()
// DISABLED-MARKED-NEXT: call void asm sideeffect
// DISABLED-NEXT:        [[T2:%.*]] = {{.*}}call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_assign_assign(void) {
  __unsafe_unretained id x, y;
  x = y = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_assign_assign()
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[Y:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CALL-NEXT:            [[T2:%.*]] = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           store ptr [[T2]], ptr [[Y]]
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_assign_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_strong_assign_assign(void) {
  __strong id x;
  __unsafe_unretained id y;
  x = y = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_strong_assign_assign()
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[Y:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           store ptr [[T2]], ptr [[Y]]
// CHECK-NEXT:           [[OLD:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-NEXT:           call void @llvm.objc.release(ptr [[OLD]]
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_strong_assign_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_assign_strong_assign(void) {
  __unsafe_unretained id x;
  __strong id y;
  x = y = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_assign_strong_assign()
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[Y:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           [[OLD:%.*]] = load ptr, ptr [[Y]]
// CHECK-NEXT:           store ptr [[T2]], ptr [[Y]]
// CHECK-NEXT:           call void @llvm.objc.release(ptr [[OLD]]
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(ptr [[Y]], ptr null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_assign_strong_assign()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_init(void) {
  __unsafe_unretained id x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_init()
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CALL-NEXT:            [[T2:%.*]] = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT:           ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_init()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_init_assignment(void) {
  __unsafe_unretained id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_init_assignment()
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[Y:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// NOTAIL-CALL-NEXT:     [[T2:%.*]] = notail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CALL-NEXT:            [[T2:%.*]] = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-NEXT:           store ptr [[T2]], ptr [[Y]]
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT: ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_init_assignment()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_strong_init_assignment(void) {
  __unsafe_unretained id x;
  __strong id y = x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_strong_init_assignment()
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[Y:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-NEXT:           store ptr [[T2]], ptr [[Y]]
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(ptr [[Y]], ptr null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT: ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_strong_init_assignment()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_init_strong_assignment(void) {
  __strong id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL:        define{{.*}} void @test_init_strong_assignment()
// CHECK:                [[X:%.*]] = alloca ptr
// CHECK:                [[Y:%.*]] = alloca ptr
// CHECK:                [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT:    call void asm sideeffect
// CHECK-NEXT:           [[T2:%.*]] = {{.*}}call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:           [[OLD:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT:           store ptr [[T2]], ptr [[X]]
// CHECK-NEXT:           call void @llvm.objc.release(ptr [[OLD]])
// CHECK-NEXT:           store ptr [[T2]], ptr [[Y]]
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-UNOPTIMIZED-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr null)
// CHECK-OPTIMIZED-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-OPTIMIZED-NEXT: call void @llvm.objc.release(ptr [[T0]])
// CHECK-OPTIMIZED-NEXT: lifetime.end
// CHECK-NEXT: ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_init_strong_assignment()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_ignored(void) {
  makeA();
}
// CHECK-LABEL:     define{{.*}} void @test_ignored()
// CHECK:             [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT: call void asm sideeffect
// NOTAIL-CALL-NEXT:  [[T2:%.*]] = notail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CALL-NEXT:         [[T2:%.*]] = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:        ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_ignored()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])

void test_cast_to_void(void) {
  (void) makeA();
}
// CHECK-LABEL:     define{{.*}} void @test_cast_to_void()
// CHECK:             [[T0:%.*]] = call ptr @makeA()
// CHECK-MARKED-NEXT: call void asm sideeffect
// NOTAIL-CALL-NEXT:  [[T2:%.*]] = notail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CALL-NEXT:         [[T2:%.*]] = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT:        ret void

// ATTACHED-CALL-LABEL:      define{{.*}} void @test_cast_to_void()
// ATTACHED-CALL:              [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ],
// ATTACHED-CALL:              call void (...) @llvm.objc.clang.arc.noop.use(ptr [[T0]])


// This is always at the end of the module.

// CHECK-OPTIMIZED: !llvm.module.flags = !{!0,
// CHECK-OPTIMIZED: !0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov{{.*}}marker for objc_retainAutoreleaseReturnValue"}
