// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 | FileCheck %s
// rdar://8893785

void MYFUNC(void) {
// CHECK-LABEL:    define{{.*}} void @MYFUNC()
// CHECK:      [[OBSERVER_SLOT:%.*]] = alloca [[OBSERVER_T:%.*]], align 8

// CHECK:      [[T0:%.*]] = getelementptr inbounds [[OBSERVER_T]], [[OBSERVER_T]]* [[OBSERVER_SLOT]], i32 0, i32 1
// CHECK:      store [[OBSERVER_T]]* [[OBSERVER_SLOT]], [[OBSERVER_T]]** [[T0]]

// CHECK:      [[T1:%.*]] = bitcast i8* ()*
// CHECK:      [[FORWARDING:%.*]] = getelementptr inbounds [[OBSERVER_T]], [[OBSERVER_T]]* [[OBSERVER_SLOT]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load [[OBSERVER_T]]*, [[OBSERVER_T]]** [[FORWARDING]]
// CHECK-NEXT: [[OBSERVER:%.*]] = getelementptr inbounds [[OBSERVER_T]], [[OBSERVER_T]]* [[T0]], i32 0, i32 6
// CHECK-NEXT: store i8* [[T1]], i8** [[OBSERVER]]
  __block id observer = ^{ return observer; };
}

