// RUN: %clang_cc1 -triple arm64-apple-ios9 -fobjc-runtime=ios-9.0 -fobjc-arc -O -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK

@class A;

A *makeA(void);

void test_assign() {
  __unsafe_unretained id x;
  x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_assign()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A:.*]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_assign_assign() {
  __unsafe_unretained id x, y;
  x = y = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_assign_assign()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[Y:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_strong_assign_assign() {
  __strong id x;
  __unsafe_unretained id y;
  x = y = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_strong_assign_assign()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[Y:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT:    [[OLD:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(i8* [[OLD]]
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    [[T0:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(i8* [[T0]])
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_assign_strong_assign() {
  __unsafe_unretained id x;
  __strong id y;
  x = y = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_assign_strong_assign()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[Y:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    [[OLD:%.*]] = load i8*, i8** [[Y]]
// CHECK-NEXT:    store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT:    call void @llvm.objc.release(i8* [[OLD]]
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    [[T0:%.*]] = load i8*, i8** [[Y]]
// CHECK-NEXT:    call void @llvm.objc.release(i8* [[T0]])
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_init() {
  __unsafe_unretained id x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_init()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_init_assignment() {
  __unsafe_unretained id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_init_assignment()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[Y:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_strong_init_assignment() {
  __unsafe_unretained id x;
  __strong id y = x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_strong_init_assignment()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[Y:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT:    [[T0:%.*]] = load i8*, i8** [[Y]]
// CHECK-NEXT:    call void @llvm.objc.release(i8* [[T0]])
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_init_strong_assignment() {
  __strong id x;
  __unsafe_unretained id y = x = makeA();
}
// CHECK-LABEL: define{{.*}} void @test_init_strong_assignment()
// CHECK:         [[X:%.*]] = alloca i8*
// CHECK:         [[Y:%.*]] = alloca i8*
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK-NEXT:    [[OLD:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT:    store i8* [[T1]], i8** [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(i8* [[OLD]])
// CHECK-NEXT:    store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    [[T0:%.*]] = load i8*, i8** [[X]]
// CHECK-NEXT:    call void @llvm.objc.release(i8* [[T0]])
// CHECK-NEXT:    bitcast
// CHECK-NEXT:    lifetime.end
// CHECK-NEXT:    ret void

void test_ignored() {
  makeA();
}
// CHECK-LABEL: define{{.*}} void @test_ignored()
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    ret void

void test_cast_to_void() {
  (void) makeA();
}
// CHECK-LABEL: define{{.*}} void @test_cast_to_void()
// CHECK:         [[T0:%.*]] = call [[A]]* @makeA() [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.unsafeClaimAutoreleasedReturnValue) ]
// CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use({{.*}} [[T0]])
// CHECK-NEXT:    ret void

// This is always at the end of the module.

// CHECK-OPTIMIZED: !llvm.module.flags = !{!0,
// CHECK-OPTIMIZED: !0 = !{i32 1, !"clang.arc.retainAutoreleasedReturnValueMarker", !"mov{{.*}}marker for objc_retainAutoreleaseReturnValue"}
