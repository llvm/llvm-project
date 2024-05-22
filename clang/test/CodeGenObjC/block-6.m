// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 | FileCheck %s

void MYFUNC(void) {
// CHECK-LABEL:    define{{.*}} void @MYFUNC()
// CHECK:      [[OBSERVER_SLOT:%.*]] = alloca [[OBSERVER_T:%.*]], align 8
// CHECK:      [[BLOCK:%.*]] = alloca <{

// CHECK:      [[T0:%.*]] = getelementptr inbounds [[OBSERVER_T]], ptr [[OBSERVER_SLOT]], i32 0, i32 1
// CHECK:      store ptr [[OBSERVER_SLOT]], ptr [[T0]]

// CHECK:      [[FORWARDING:%.*]] = getelementptr inbounds [[OBSERVER_T]], ptr [[OBSERVER_SLOT]], i32 0, i32 1
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[FORWARDING]]
// CHECK-NEXT: [[OBSERVER:%.*]] = getelementptr inbounds [[OBSERVER_T]], ptr [[T0]], i32 0, i32 6
// CHECK-NEXT: store ptr [[BLOCK]], ptr [[OBSERVER]]
  __block id observer = ^{ return observer; };
}

