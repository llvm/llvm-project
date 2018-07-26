// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -fexceptions -fobjc-arc-exceptions -o - %s | FileCheck -check-prefix CHECK %s
// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -fexceptions -fobjc-arc-exceptions -O1 -o - %s | FileCheck -check-prefix CHECK-O1 %s

// CHECK: [[A:.*]] = type { i64, [10 x i8*] }
// CHECK: %[[STRUCT_TEST1_S0:.*]] = type { i32 }
// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }

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

namespace test1 {

// Check that copy/dispose helper functions are exception safe.

// CHECK-LABEL: define internal void @__copy_helper_block_(
// CHECK: %[[BLOCK_SOURCE:.*]] = bitcast i8* %{{.*}} to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>*
// CHECK: %[[BLOCK_DEST:.*]] = bitcast i8* %{{.*}} to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>*

// CHECK: %[[V4:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_SOURCE]], i32 0, i32 6
// CHECK: %[[V5:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_DEST]], i32 0, i32 6
// CHECK: %[[BLOCKCOPY_SRC:.*]] = load i8*, i8** %[[V4]], align 8
// CHECK: %[[V6:.*]] = bitcast i8** %[[V5]] to i8*
// CHECK: call void @_Block_object_assign(i8* %[[V6]], i8* %[[BLOCKCOPY_SRC]], i32 8)

// CHECK: %[[V7:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_SOURCE]], i32 0, i32 7
// CHECK: %[[V8:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_DEST]], i32 0, i32 7
// CHECK: call void @objc_copyWeak(i8** %[[V8]], i8** %[[V7]])

// CHECK: %[[V9:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_SOURCE]], i32 0, i32 5
// CHECK: %[[V10:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_DEST]], i32 0, i32 5
// CHECK: %[[BLOCKCOPY_SRC2:.*]] = load i8*, i8** %[[V9]], align 8
// CHECK: store i8* null, i8** %[[V10]], align 8
// CHECK: call void @objc_storeStrong(i8** %[[V10]], i8* %[[BLOCKCOPY_SRC2]])

// CHECK: %[[V11:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_SOURCE]], i32 0, i32 8
// CHECK: %[[V12:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_DEST]], i32 0, i32 8
// CHECK: invoke void @_ZN5test12S0C1ERKS0_(%[[STRUCT_TEST1_S0]]* %[[V12]], %[[STRUCT_TEST1_S0]]* dereferenceable(4) %[[V11]])
// CHECK: to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]

// CHECK: [[INVOKE_CONT]]:
// CHECK: %[[V13:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_SOURCE]], i32 0, i32 9
// CHECK: %[[V14:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK_DEST]], i32 0, i32 9
// CHECK: invoke void @_ZN5test12S0C1ERKS0_(%[[STRUCT_TEST1_S0]]* %[[V14]], %[[STRUCT_TEST1_S0]]* dereferenceable(4) %[[V13]])
// CHECK: to label %[[INVOKE_CONT4:.*]] unwind label %[[LPAD3:.*]]

// CHECK: [[INVOKE_CONT4]]:
// CHECK: ret void

// CHECK: [[LPAD]]:
// CHECK: br label %[[EHCLEANUP:.*]]

// CHECK: [[LPAD3]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(%[[STRUCT_TEST1_S0]]* %[[V12]])
// CHECK: to label %[[INVOKE_CONT5:.*]] unwind label %[[TERMINATE_LPAD:.*]]

// CHECK: [[INVOKE_CONT5]]:
// CHECK: br label %[[EHCLEANUP]]

// CHECK: [[EHCLEANUP]]:
// CHECK: call void @objc_storeStrong(i8** %[[V10]], i8* null)
// CHECK: call void @objc_destroyWeak(i8** %[[V8]])
// CHECK: %[[V21:.*]] = load i8*, i8** %[[V5]], align 8
// CHECK: call void @_Block_object_dispose(i8* %[[V21]], i32 8)
// CHECK: br label %[[EH_RESUME:.*]]

// CHECK: [[EH_RESUME]]:
// CHECK: resume { i8*, i32 }

// CHECK: [[TERMINATE_LPAD]]:
// CHECK: call void @__clang_call_terminate(

// CHECK-O1-LABEL: define internal void @__copy_helper_block_(
// CHECK-O1: tail call void @objc_release({{.*}}) {{.*}} !clang.imprecise_release

// CHECK: define internal void @__destroy_helper_block_(
// CHECK: %[[BLOCK:.*]] = bitcast i8* %{{.*}} to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>*
// CHECK: %[[V2:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK]], i32 0, i32 6
// CHECK: %[[V3:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK]], i32 0, i32 7
// CHECK: %[[V4:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V5:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK]], i32 0, i32 8
// CHECK: %[[V6:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]] }>* %[[BLOCK]], i32 0, i32 9
// CHECK: invoke void @_ZN5test12S0D1Ev(%[[STRUCT_TEST1_S0]]* %[[V6]])
// CHECK: to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]

// CHECK: [[INVOKE_CONT]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(%[[STRUCT_TEST1_S0]]* %[[V5]])
// CHECK: to label %[[INVOKE_CONT2:.*]] unwind label %[[LPAD1:.*]]

// CHECK: [[INVOKE_CONT2]]:
// CHECK: call void @objc_storeStrong(i8** %[[V4]], i8* null)
// CHECK: call void @objc_destroyWeak(i8** %[[V3]])
// CHECK: %[[V7:.*]] = load i8*, i8** %[[V2]], align 8
// CHECK: call void @_Block_object_dispose(i8* %[[V7]], i32 8)
// CHECK: ret void

// CHECK: [[LPAD]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(%[[STRUCT_TEST1_S0]]* %[[V5]])
// CHECK: to label %[[INVOKE_CONT3:.*]] unwind label %[[TERMINATE_LPAD]]

// CHECK: [[LPAD1]]
// CHECK: br label %[[EHCLEANUP:.*]]

// CHECK: [[INVOKE_CONT3]]:
// CHECK: br label %[[EHCLEANUP]]

// CHECK: [[EHCLEANUP]]:
// CHECK: call void @objc_storeStrong(i8** %[[V4]], i8* null)
// CHECK: call void @objc_destroyWeak(i8** %[[V3]])
// CHECK: %[[V14:.*]] = load i8*, i8** %[[V2]], align 8
// CHECK: call void @_Block_object_dispose(i8* %[[V14]], i32 8)
// CHECK: br label %[[EH_RESUME]]

// CHECK: [[EH_RESUME]]:
// CHECK: resume { i8*, i32 }

// CHECK: [[TERMINATE_LPAD]]:
// CHECK: call void @__clang_call_terminate(

// CHECK-O1-LABEL: define internal void @__destroy_helper_block_(
// CHECK-O1: tail call void @objc_release({{.*}}) {{.*}} !clang.imprecise_release
// CHECK-O1: tail call void @objc_release({{.*}}) {{.*}} !clang.imprecise_release

struct S0 {
  S0();
  S0(const S0 &);
  ~S0();
  int f0;
};

id getObj();

void foo1() {
  __block id t0 = getObj();
  __weak id t1 = getObj();
  id t2 = getObj();
  S0 t3, t4;
  ^{ (void)t0; (void)t1; (void)t2; (void)t3; (void)t4; };
}
}
