// RUN: %clang_cc1 -triple arm64-apple-ios9 -fobjc-runtime=ios-9.0 -fobjc-arc -O -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-apple-macosx10 -fobjc-runtime=ios-9.0 -fobjc-arc -O -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK

@class A;

A *makeA(void);

void test_assign(void) {
  __unsafe_unretained id x;
  x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_assign()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_assign_assign(void) {
  __unsafe_unretained id x, y;
  x = y = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_assign_assign()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[Y:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    store ptr [[T0]], ptr [[Y]]
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_strong_assign_assign(void) {
  __strong id x;
  __unsafe_unretained id y;
  x = y = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_strong_assign_assign()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[Y:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    store ptr [[T0]], ptr [[Y]]
// CHECK-NEXT:    [[OLD:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[OLD]]
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_assign_strong_assign(void) {
  __unsafe_unretained id x;
  __strong id y;
  x = y = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_assign_strong_assign()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[Y:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[OLD:%.*]] = load ptr, ptr [[Y]]
// CHECK-NEXT:    store ptr [[T0]], ptr [[Y]]
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[OLD]]
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    [[T0:%.*]] = load ptr, ptr [[Y]]
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_init(void) {
  __unsafe_unretained id x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_init()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_init_assignment(void) {
  __unsafe_unretained id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_init_assignment()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[Y:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    store ptr [[T0]], ptr [[Y]]
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_strong_init_assignment(void) {
  __unsafe_unretained id x;
  __strong id y = x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_strong_init_assignment()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[Y:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    store ptr [[T0]], ptr [[Y]]
// CHECK-NEXT:    [[T0:%.*]] = load ptr, ptr [[Y]]
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_init_strong_assignment(void) {
  __strong id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_init_strong_assignment()
// CHECK:         [[X:%.*]] = alloca ptr
// CHECK:         [[Y:%.*]] = alloca ptr
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[OLD:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT:    store ptr [[T0]], ptr [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[OLD]])
// CHECK-NEXT:    store ptr [[T0]], ptr [[Y]]
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    [[T0:%.*]] = load ptr, ptr [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[T0]])
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_ignored(void) {
  makeA();
}
// CHECK-LABEL: define{{.*}} void @test_ignored()
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    ret void

void test_cast_to_void(void) {
  (void) makeA();
}
// CHECK-LABEL: define{{.*}} void @test_cast_to_void()
// CHECK:         [[T0:%.*]] = call ptr @makeA() [ "clang.arc.attachedcall"(ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    ret void

// This is always at the end of the module.

// CHECK-OPTIMIZED: !llvm.module.flags = !{!0,
// CHECK-OPTIMIZED: !0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov{{.*}}marker for objc_retainAutoreleaseReturnValue"}
