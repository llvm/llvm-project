// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -o - %s | FileCheck %s

// CHECK: [[A:.*]] = type { i64, [10 x i8*] }

// CHECK: [[LAYOUT0:@.*]] = private unnamed_addr constant [3 x i8] c" 9\00"

// rdar://13045269
// If a __block variable requires extended layout information *and*
// a copy/dispose helper, be sure to adjust the offsets used in copy/dispose.
namespace test0 {
  struct A {
    unsigned long count;
    id data[10];
  };

  void foo() {
    __block A v;
  }
  // CHECK-LABEL:    define void @_ZN5test03fooEv() 
  // CHECK:      [[V:%.*]] = alloca [[BYREF_A:%.*]], align 8
  // CHECK:      [[T0:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[V]], i32 0, i32 4
  // CHECK-NEXT: store i8* bitcast (void (i8*, i8*)* [[COPY_HELPER:@.*]] to i8*), i8** [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[V]], i32 0, i32 5
  // CHECK-NEXT: store i8* bitcast (void (i8*)* [[DISPOSE_HELPER:@.*]] to i8*), i8** [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[V]], i32 0, i32 6
  // CHECK-NEXT: store i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[LAYOUT0]], i32 0, i32 0), i8** [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[V]], i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AC1Ev([[A]]* [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[V]], i32 0, i32 7
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[BYREF_A]]* [[V]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T1]], i32 8)
  // CHECK-NEXT: call void @_ZN5test01AD1Ev([[A]]* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void [[COPY_HELPER]](
  // CHECK:      [[T0:%.*]] = bitcast i8* {{.*}} to [[BYREF_A]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[T0]], i32 0, i32 7
  // CHECK-NEXT: load
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* {{.*}} to [[BYREF_A]]*
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[T2]], i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AC1ERKS0_([[A]]* [[T1]], [[A]]* dereferenceable({{[0-9]+}}) [[T3]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void [[DISPOSE_HELPER]](
  // CHECK:      [[T0:%.*]] = bitcast i8* {{.*}} to [[BYREF_A]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[T0]], i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AD1Ev([[A]]* [[T1]])
  // CHECK-NEXT: ret void
}
