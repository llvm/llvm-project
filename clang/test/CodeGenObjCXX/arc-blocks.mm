// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -fexceptions -fobjc-arc-exceptions -o - %s | FileCheck -check-prefix CHECK %s
// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -fexceptions -fobjc-arc-exceptions -O1 -o - %s | FileCheck -check-prefix CHECK-O1 %s
// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -o - %s | FileCheck -check-prefix CHECK-NOEXCP %s

// CHECK: [[A:.*]] = type { i64, [10 x i8*] }
// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }
// CHECK: %[[STRUCT_TEST1_S0:.*]] = type { i32 }
// CHECK: %[[STRUCT_TRIVIAL_INTERNAL:.*]] = type { i32 }

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
    ^{ (void)v; };
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
  // CHECK: bitcast [[BYREF_A]]* [[V]] to i8*
  // CHECK: [[T1:%.*]] = bitcast [[BYREF_A]]* [[V]] to i8*
  // CHECK-NEXT: call void @_Block_object_dispose(i8* [[T1]], i32 8)
  // CHECK-NEXT: call void @_ZN5test01AD1Ev([[A]]* [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void [[COPY_HELPER]](
  // CHECK:      [[T0:%.*]] = bitcast i8* {{.*}} to [[BYREF_A]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[T0]], i32 0, i32 7
  // CHECK-NEXT: load
  // CHECK-NEXT: [[T2:%.*]] = bitcast i8* {{.*}} to [[BYREF_A]]*
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[T2]], i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AC1ERKS0_([[A]]* [[T1]], [[A]]* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[T3]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void [[DISPOSE_HELPER]](
  // CHECK:      [[T0:%.*]] = bitcast i8* {{.*}} to [[BYREF_A]]*
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[BYREF_A]], [[BYREF_A]]* [[T0]], i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AD1Ev([[A]]* [[T1]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_
// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_
// CHECK-LABEL-O1: define linkonce_odr hidden void @__copy_helper_block_
// CHECK-LABEL-O1: define linkonce_odr hidden void @__destroy_helper_block_

namespace test1 {

// Check that copy/dispose helper functions are exception safe.

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_ea8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(
// CHECK: %[[BLOCK_SOURCE:.*]] = bitcast i8* %{{.*}} to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>*
// CHECK: %[[BLOCK_DEST:.*]] = bitcast i8* %{{.*}} to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>*

// CHECK: %[[V9:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_SOURCE]], i32 0, i32 5
// CHECK: %[[V10:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_DEST]], i32 0, i32 5
// CHECK: %[[BLOCKCOPY_SRC2:.*]] = load i8*, i8** %[[V9]], align 8
// CHECK: store i8* null, i8** %[[V10]], align 8
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V10]], i8* %[[BLOCKCOPY_SRC2]])

// CHECK: %[[V4:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_SOURCE]], i32 0, i32 6
// CHECK: %[[V5:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_DEST]], i32 0, i32 6
// CHECK: %[[BLOCKCOPY_SRC:.*]] = load i8*, i8** %[[V4]], align 8
// CHECK: %[[V6:.*]] = bitcast i8** %[[V5]] to i8*
// CHECK: call void @_Block_object_assign(i8* %[[V6]], i8* %[[BLOCKCOPY_SRC]], i32 8)

// CHECK: %[[V7:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_SOURCE]], i32 0, i32 7
// CHECK: %[[V8:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_DEST]], i32 0, i32 7
// CHECK: call void @llvm.objc.copyWeak(i8** %[[V8]], i8** %[[V7]])

// CHECK: %[[V11:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_SOURCE]], i32 0, i32 8
// CHECK: %[[V12:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_DEST]], i32 0, i32 8
// CHECK: invoke void @_ZN5test12S0C1ERKS0_(%[[STRUCT_TEST1_S0]]* %[[V12]], %[[STRUCT_TEST1_S0]]* nonnull align 4 dereferenceable(4) %[[V11]])
// CHECK: to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]

// CHECK: [[INVOKE_CONT]]:
// CHECK: %[[V13:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_SOURCE]], i32 0, i32 9
// CHECK: %[[V14:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK_DEST]], i32 0, i32 9
// CHECK: invoke void @_ZN5test12S0C1ERKS0_(%[[STRUCT_TEST1_S0]]* %[[V14]], %[[STRUCT_TEST1_S0]]* nonnull align 4 dereferenceable(4) %[[V13]])
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
// CHECK: call void @llvm.objc.destroyWeak(i8** %[[V8]])
// CHECK: %[[V21:.*]] = load i8*, i8** %[[V5]], align 8
// CHECK: call void @_Block_object_dispose(i8* %[[V21]], i32 8)
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V10]], i8* null)
// CHECK: br label %[[EH_RESUME:.*]]

// CHECK: [[EH_RESUME]]:
// CHECK: resume { i8*, i32 }

// CHECK: [[TERMINATE_LPAD]]:
// CHECK: call void @__clang_call_terminate(

// CHECK-O1-LABEL: define linkonce_odr hidden void @__copy_helper_block_ea8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(
// CHECK-O1: call void @llvm.objc.release({{.*}}) {{.*}} !clang.imprecise_release
// CHECK-NOEXCP: define linkonce_odr hidden void @__copy_helper_block_8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(

// CHECK: define linkonce_odr hidden void @__destroy_helper_block_ea8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(
// CHECK: %[[BLOCK:.*]] = bitcast i8* %{{.*}} to <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>*
// CHECK: %[[V4:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V2:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK]], i32 0, i32 6
// CHECK: %[[V3:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK]], i32 0, i32 7
// CHECK: %[[V5:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK]], i32 0, i32 8
// CHECK: %[[V6:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8*, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>* %[[BLOCK]], i32 0, i32 9
// CHECK: invoke void @_ZN5test12S0D1Ev(%[[STRUCT_TEST1_S0]]* %[[V6]])
// CHECK: to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]

// CHECK: [[INVOKE_CONT]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(%[[STRUCT_TEST1_S0]]* %[[V5]])
// CHECK: to label %[[INVOKE_CONT2:.*]] unwind label %[[LPAD1:.*]]

// CHECK: [[INVOKE_CONT2]]:
// CHECK: call void @llvm.objc.destroyWeak(i8** %[[V3]])
// CHECK: %[[V7:.*]] = load i8*, i8** %[[V2]], align 8
// CHECK: call void @_Block_object_dispose(i8* %[[V7]], i32 8)
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V4]], i8* null)
// CHECK: ret void

// CHECK: [[LPAD]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(%[[STRUCT_TEST1_S0]]* %[[V5]])
// CHECK: to label %[[INVOKE_CONT3:.*]] unwind label %[[TERMINATE_LPAD:.*]]

// CHECK: [[LPAD1]]
// CHECK: br label %[[EHCLEANUP:.*]]

// CHECK: [[INVOKE_CONT3]]:
// CHECK: br label %[[EHCLEANUP]]

// CHECK: [[EHCLEANUP]]:
// CHECK: call void @llvm.objc.destroyWeak(i8** %[[V3]])
// CHECK: %[[V14:.*]] = load i8*, i8** %[[V2]], align 8
// CHECK: call void @_Block_object_dispose(i8* %[[V14]], i32 8)
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V4]], i8* null)
// CHECK: br label %[[EH_RESUME:.*]]

// CHECK: [[EH_RESUME]]:
// CHECK: resume { i8*, i32 }

// CHECK: [[TERMINATE_LPAD]]:
// CHECK: call void @__clang_call_terminate(

// CHECK-O1-LABEL: define linkonce_odr hidden void @__destroy_helper_block_ea8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(
// CHECK-O1: call void @llvm.objc.release({{.*}}) {{.*}} !clang.imprecise_release
// CHECK-O1: call void @llvm.objc.release({{.*}}) {{.*}} !clang.imprecise_release
// CHECK-NOEXCP: define linkonce_odr hidden void @__destroy_helper_block_8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(

namespace {
struct TrivialInternal {
  int a;
};
}

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
  // Capturing a non-external type doesn't cause the copy/dispose helpers to be
  // internal unless the captured type has a non-trivial copy constructor or
  // destructor.
  TrivialInternal t5;
  ^{ (void)t0; (void)t1; (void)t2; (void)t3; (void)t4; (void)t5; };
}
}

// Test that calls to @llvm.objc.retainBlock aren't emitted in some cases.

namespace test_block_retain {
  typedef void (^BlockTy)();

  void foo1(id);

// CHECK-LABEL: define void @_ZN17test_block_retain14initializationEP11objc_object(
// CHECK-NOT: @llvm.objc.retainBlock(
  void initialization(id a) {
    BlockTy b0 = ^{ foo1(a); };
    BlockTy b1 = (^{ foo1(a); });
    b0();
    b1();
  }

// CHECK-LABEL: define void @_ZN17test_block_retain20initializationStaticEP11objc_object(
// CHECK: @llvm.objc.retainBlock(
  void initializationStatic(id a) {
    static BlockTy b0 = ^{ foo1(a); };
    b0();
  }

// CHECK-LABEL: define void @_ZN17test_block_retain15initialization2EP11objc_object
// CHECK: %[[B0:.*]] = alloca void ()*, align 8
// CHECK: %[[B1:.*]] = alloca void ()*, align 8
// CHECK: load void ()*, void ()** %[[B0]], align 8
// CHECK-NOT: @llvm.objc.retainBlock
// CHECK: %[[V9:.*]] = load void ()*, void ()** %[[B0]], align 8
// CHECK: %[[V10:.*]] = bitcast void ()* %[[V9]] to i8*
// CHECK: %[[V11:.*]] = call i8* @llvm.objc.retainBlock(i8* %[[V10]])
// CHECK: %[[V12:.*]] = bitcast i8* %[[V11]] to void ()*
// CHECK: store void ()* %[[V12]], void ()** %[[B1]], align 8
  void initialization2(id a) {
    BlockTy b0 = ^{ foo1(a); };
    b0();
    BlockTy b1 = b0; // can't optimize this yet.
    b1();
  }

// CHECK-LABEL: define void @_ZN17test_block_retain10assignmentEP11objc_object(
// CHECK-NOT: @llvm.objc.retainBlock(
  void assignment(id a) {
    BlockTy b0;
    (b0) = ^{ foo1(a); };
    b0();
    b0 = (^{ foo1(a); });
    b0();
  }

// CHECK-LABEL: define void @_ZN17test_block_retain16assignmentStaticEP11objc_object(
// CHECK: @llvm.objc.retainBlock(
  void assignmentStatic(id a) {
    static BlockTy b0;
    b0 = ^{ foo1(a); };
    b0();
  }

// CHECK-LABEL: define void @_ZN17test_block_retain21assignmentConditionalEP11objc_objectb(
// CHECK: @llvm.objc.retainBlock(
  void assignmentConditional(id a, bool c) {
    BlockTy b0;
    if (c)
      // can't optimize this since 'b0' is declared in the outer scope.
      b0 = ^{ foo1(a); };
    b0();
  }

// CHECK-LABEL: define void @_ZN17test_block_retain11assignment2EP11objc_object(
// CHECK: %[[B0:.*]] = alloca void ()*, align 8
// CHECK: %[[B1:.*]] = alloca void ()*, align 8
// CHECK-NOT: @llvm.objc.retainBlock
// CHECK: store void ()* null, void ()** %[[B1]], align 8
// CHECK: %[[V9:.*]] = load void ()*, void ()** %[[B0]], align 8
// CHECK: %[[V10:.*]] = bitcast void ()* %[[V9]] to i8*
// CHECK: %[[V11:.*]] = call i8* @llvm.objc.retainBlock(i8* %[[V10]]
// CHECK: %[[V12:.*]] = bitcast i8* %[[V11]] to void ()*
// CHECK: store void ()* %[[V12]], void ()** %[[B1]], align 8
  void assignment2(id a) {
    BlockTy b0 = ^{ foo1(a); };
    b0();
    BlockTy b1;
    b1 = b0; // can't optimize this yet.
    b1();
  }

// We cannot remove the call to @llvm.objc.retainBlock if the variable is of type id.

// CHECK: define void @_ZN17test_block_retain21initializationObjCPtrEP11objc_object(
// CHECK: alloca i8*, align 8
// CHECK: %[[B0:.*]] = alloca i8*, align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, align 8
// CHECK: %[[V3:.*]] = bitcast <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]] to void ()*
// CHECK: %[[V4:.*]] = bitcast void ()* %[[V3]] to i8*
// CHECK: %[[V5:.*]] = call i8* @llvm.objc.retainBlock(i8* %[[V4]])
// CHECK: %[[V6:.*]] = bitcast i8* %[[V5]] to void ()*
// CHECK: %[[V7:.*]] = bitcast void ()* %[[V6]] to i8*
// CHECK: store i8* %[[V7]], i8** %[[B0]], align 8
  void initializationObjCPtr(id a) {
    id b0 = ^{ foo1(a); };
    ((BlockTy)b0)();
  }

// CHECK: define void @_ZN17test_block_retain17assignmentObjCPtrEP11objc_object(
// CHECK: %[[B0:.*]] = alloca void ()*, align 8
// CHECK: %[[B1:.*]] = alloca i8*, align 8
// CHECK: %[[V4:.*]] = load void ()*, void ()** %[[B0]], align 8
// CHECK: %[[V5:.*]] = bitcast void ()* %[[V4]] to i8*
// CHECK: %[[V6:.*]] = call i8* @llvm.objc.retainBlock(i8* %[[V5]])
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to void ()*
// CHECK: %[[V8:.*]] = bitcast void ()* %[[V7]] to i8*
// CHECK: store i8* %[[V8]], i8** %[[B1]], align 8
  void assignmentObjCPtr(id a) {
    BlockTy b0 = ^{ foo1(a); };
    id b1;
    b1 = b0;
    ((BlockTy)b1)();
  }
}
