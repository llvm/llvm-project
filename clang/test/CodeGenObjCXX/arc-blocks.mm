// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -fexceptions -fobjc-arc-exceptions -o - %s | FileCheck -check-prefix CHECK %s
// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -fexceptions -fobjc-arc-exceptions -O1 -o - %s | FileCheck -check-prefix CHECK-O1 %s
// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -o - %s | FileCheck -check-prefix CHECK-NOEXCP %s
// RUN: %clang_cc1 -std=gnu++98 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -fexceptions -fobjc-arc-exceptions -fobjc-avoid-heapify-local-blocks -o - %s | FileCheck -check-prefix CHECK-NOHEAP %s

// CHECK: [[A:.*]] = type { i64, [10 x ptr] }
// CHECK: %[[STRUCT_TEST1_S0:.*]] = type { i32 }
// CHECK: %[[STRUCT_TRIVIAL_INTERNAL:.*]] = type { i32 }

// CHECK: [[LAYOUT0:@.*]] = private unnamed_addr constant [3 x i8] c" 9\00"

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
  // CHECK-LABEL:    define{{.*}} void @_ZN5test03fooEv() 
  // CHECK:      [[V:%.*]] = alloca [[BYREF_A:%.*]], align 8
  // CHECK:      [[T0:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr [[V]], i32 0, i32 4
  // CHECK-NEXT: store ptr [[COPY_HELPER:@.*]], ptr [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr [[V]], i32 0, i32 5
  // CHECK-NEXT: store ptr [[DISPOSE_HELPER:@.*]], ptr [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr [[V]], i32 0, i32 6
  // CHECK-NEXT: store ptr [[LAYOUT0]], ptr [[T0]]
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr [[V]], i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AC1Ev(ptr {{[^,]*}} [[T0]])
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr [[V]], i32 0, i32 7
  // CHECK: call void @_Block_object_dispose(ptr [[V]], i32 8)
  // CHECK-NEXT: call void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[T0]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void [[COPY_HELPER]](
  // CHECK: [[T1:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr {{.*}}, i32 0, i32 7
  // CHECK-NEXT: load
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr {{.*}}, i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AC1ERKS0_(ptr {{[^,]*}} [[T1]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[T3]])
  // CHECK-NEXT: ret void

  // CHECK:    define internal void [[DISPOSE_HELPER]](
  // CHECK: [[T1:%.*]] = getelementptr inbounds nuw [[BYREF_A]], ptr {{.*}}, i32 0, i32 7
  // CHECK-NEXT: call void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[T1]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_
// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_
// CHECK-LABEL-O1: define linkonce_odr hidden void @__copy_helper_block_
// CHECK-LABEL-O1: define linkonce_odr hidden void @__destroy_helper_block_

namespace test1 {

// Check that copy/dispose helper functions are exception safe.

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_ea8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(

// CHECK: %[[V9:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 5
// CHECK: %[[V10:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 5
// CHECK: %[[BLOCKCOPY_SRC2:.*]] = load ptr, ptr %[[V9]], align 8
// CHECK: store ptr null, ptr %[[V10]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V10]], ptr %[[BLOCKCOPY_SRC2]])

// CHECK: %[[V4:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 6
// CHECK: %[[V5:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 6
// CHECK: %[[BLOCKCOPY_SRC:.*]] = load ptr, ptr %[[V4]], align 8
// CHECK: call void @_Block_object_assign(ptr %[[V5]], ptr %[[BLOCKCOPY_SRC]], i32 8)

// CHECK: %[[V7:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 7
// CHECK: %[[V8:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 7
// CHECK: call void @llvm.objc.copyWeak(ptr %[[V8]], ptr %[[V7]])

// CHECK: %[[V11:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 8
// CHECK: %[[V12:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 8
// CHECK: invoke void @_ZN5test12S0C1ERKS0_(ptr {{[^,]*}} %[[V12]], ptr noundef nonnull align 4 dereferenceable(4) %[[V11]])
// CHECK: to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]

// CHECK: [[INVOKE_CONT]]:
// CHECK: %[[V13:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 9
// CHECK: %[[V14:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 9
// CHECK: invoke void @_ZN5test12S0C1ERKS0_(ptr {{[^,]*}} %[[V14]], ptr noundef nonnull align 4 dereferenceable(4) %[[V13]])
// CHECK: to label %[[INVOKE_CONT4:.*]] unwind label %[[LPAD3:.*]]

// CHECK: [[INVOKE_CONT4]]:
// CHECK: ret void

// CHECK: [[LPAD]]:
// CHECK: br label %[[EHCLEANUP:.*]]

// CHECK: [[LPAD3]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(ptr {{[^,]*}} %[[V12]])
// CHECK: to label %[[INVOKE_CONT5:.*]] unwind label %[[TERMINATE_LPAD:.*]]

// CHECK: [[INVOKE_CONT5]]:
// CHECK: br label %[[EHCLEANUP]]

// CHECK: [[EHCLEANUP]]:
// CHECK: call void @llvm.objc.destroyWeak(ptr %[[V8]])
// CHECK: %[[V21:.*]] = load ptr, ptr %[[V5]], align 8
// CHECK: call void @_Block_object_dispose(ptr %[[V21]], i32 8)
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V10]], ptr null)
// CHECK: br label %[[EH_RESUME:.*]]

// CHECK: [[EH_RESUME]]:
// CHECK: resume { ptr, i32 }

// CHECK: [[TERMINATE_LPAD]]:
// CHECK: call void @__clang_call_terminate(

// CHECK-O1-LABEL: define linkonce_odr hidden void @__copy_helper_block_ea8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(
// CHECK-O1: call void @llvm.objc.release({{.*}}) {{.*}} !clang.imprecise_release
// CHECK-NOEXCP: define linkonce_odr hidden void @__copy_helper_block_8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(

// CHECK: define linkonce_odr hidden void @__destroy_helper_block_ea8_32s40r48w56c15_ZTSN5test12S0E60c15_ZTSN5test12S0E(
// CHECK: %[[V4:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 5
// CHECK: %[[V2:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 6
// CHECK: %[[V3:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 7
// CHECK: %[[V5:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 8
// CHECK: %[[V6:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, %[[STRUCT_TEST1_S0]], %[[STRUCT_TEST1_S0]], %[[STRUCT_TRIVIAL_INTERNAL]] }>, ptr %{{.*}}, i32 0, i32 9
// CHECK: invoke void @_ZN5test12S0D1Ev(ptr {{[^,]*}} %[[V6]])
// CHECK: to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]

// CHECK: [[INVOKE_CONT]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(ptr {{[^,]*}} %[[V5]])
// CHECK: to label %[[INVOKE_CONT2:.*]] unwind label %[[LPAD1:.*]]

// CHECK: [[INVOKE_CONT2]]:
// CHECK: call void @llvm.objc.destroyWeak(ptr %[[V3]])
// CHECK: %[[V7:.*]] = load ptr, ptr %[[V2]], align 8
// CHECK: call void @_Block_object_dispose(ptr %[[V7]], i32 8)
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V4]], ptr null)
// CHECK: ret void

// CHECK: [[LPAD]]:
// CHECK: invoke void @_ZN5test12S0D1Ev(ptr {{[^,]*}} %[[V5]])
// CHECK: to label %[[INVOKE_CONT3:.*]] unwind label %[[TERMINATE_LPAD:.*]]

// CHECK: [[LPAD1]]
// CHECK: br label %[[EHCLEANUP:.*]]

// CHECK: [[INVOKE_CONT3]]:
// CHECK: br label %[[EHCLEANUP]]

// CHECK: [[EHCLEANUP]]:
// CHECK: call void @llvm.objc.destroyWeak(ptr %[[V3]])
// CHECK: %[[V14:.*]] = load ptr, ptr %[[V2]], align 8
// CHECK: call void @_Block_object_dispose(ptr %[[V14]], i32 8)
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V4]], ptr null)
// CHECK: br label %[[EH_RESUME:.*]]

// CHECK: [[EH_RESUME]]:
// CHECK: resume { ptr, i32 }

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

typedef void (^BlockTy)();
void foo1(id);

namespace test_block_retain {

// CHECK-NOHEAP-LABEL: define{{.*}} void @_ZN17test_block_retain14initializationEP11objc_object(
// CHECK-NOHEAP-NOT: @llvm.objc.retainBlock(
  void initialization(id a) {
    BlockTy b0 = ^{ foo1(a); };
    BlockTy b1 = (^{ foo1(a); });
    b0();
    b1();
  }

// CHECK-NOHEAP-LABEL: define{{.*}} void @_ZN17test_block_retain20initializationStaticEP11objc_object(
// CHECK-NOHEAP: @llvm.objc.retainBlock(
  void initializationStatic(id a) {
    static BlockTy b0 = ^{ foo1(a); };
    b0();
  }

// CHECK-NOHEAP-LABEL: define{{.*}} void @_ZN17test_block_retain15initialization2EP11objc_object
// CHECK-NOHEAP: %b0 = alloca ptr, align 8
// CHECK-NOHEAP: %b1 = alloca ptr, align 8
// CHECK-NOHEAP: load ptr, ptr %b0, align 8
// CHECK-NOHEAP-NOT: @llvm.objc.retainBlock
// CHECK-NOHEAP: %[[V9:.*]] = load ptr, ptr %b0, align 8
// CHECK-NOHEAP: %[[V11:.*]] = call ptr @llvm.objc.retainBlock(ptr %[[V9]])
// CHECK-NOHEAP: store ptr %[[V11]], ptr %b1, align 8
  void initialization2(id a) {
    BlockTy b0 = ^{ foo1(a); };
    b0();
    BlockTy b1 = b0; // can't optimize this yet.
    b1();
  }

// CHECK-NOHEAP-LABEL: define{{.*}} void @_ZN17test_block_retain10assignmentEP11objc_object(
// CHECK-NOHEAP-NOT: @llvm.objc.retainBlock(
  void assignment(id a) {
    BlockTy b0;
    (b0) = ^{ foo1(a); };
    b0();
    b0 = (^{ foo1(a); });
    b0();
  }

// CHECK-NOHEAP-LABEL: define{{.*}} void @_ZN17test_block_retain16assignmentStaticEP11objc_object(
// CHECK-NOHEAP: @llvm.objc.retainBlock(
  void assignmentStatic(id a) {
    static BlockTy b0;
    b0 = ^{ foo1(a); };
    b0();
  }

// CHECK-NOHEAP-LABEL: define{{.*}} void @_ZN17test_block_retain21assignmentConditionalEP11objc_objectb(
// CHECK-NOHEAP: @llvm.objc.retainBlock(
  void assignmentConditional(id a, bool c) {
    BlockTy b0;
    if (c)
      // can't optimize this since 'b0' is declared in the outer scope.
      b0 = ^{ foo1(a); };
    b0();
  }

// CHECK-NOHEAP-LABEL: define{{.*}} void @_ZN17test_block_retain11assignment2EP11objc_object(
// CHECK-NOHEAP: %b0 = alloca ptr, align 8
// CHECK-NOHEAP: %b1 = alloca ptr, align 8
// CHECK-NOHEAP-NOT: @llvm.objc.retainBlock
// CHECK-NOHEAP: store ptr null, ptr %b1, align 8
// CHECK-NOHEAP: %[[V9:.*]] = load ptr, ptr %b0, align 8
// CHECK-NOHEAP: %[[V11:.*]] = call ptr @llvm.objc.retainBlock(ptr %[[V9]]
// CHECK-NOHEAP: store ptr %[[V11]], ptr %b1, align 8
  void assignment2(id a) {
    BlockTy b0 = ^{ foo1(a); };
    b0();
    BlockTy b1;
    b1 = b0; // can't optimize this yet.
    b1();
  }

// We cannot remove the call to @llvm.objc.retainBlock if the variable is of type id.

// CHECK-NOHEAP: define{{.*}} void @_ZN17test_block_retain21initializationObjCPtrEP11objc_object(
// CHECK-NOHEAP: alloca ptr, align 8
// CHECK-NOHEAP: %[[B0:.*]] = alloca ptr, align 8
// CHECK-NOHEAP: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK-NOHEAP: %[[V5:.*]] = call ptr @llvm.objc.retainBlock(ptr %[[BLOCK]])
// CHECK-NOHEAP: store ptr %[[V5]], ptr %[[B0]], align 8
  void initializationObjCPtr(id a) {
    id b0 = ^{ foo1(a); };
    ((BlockTy)b0)();
  }

// CHECK-NOHEAP: define{{.*}} void @_ZN17test_block_retain17assignmentObjCPtrEP11objc_object(
// CHECK-NOHEAP: %b0 = alloca ptr, align 8
// CHECK-NOHEAP: %b1 = alloca ptr, align 8
// CHECK-NOHEAP: %[[V4:.*]] = load ptr, ptr %b0, align 8
// CHECK-NOHEAP: %[[V6:.*]] = call ptr @llvm.objc.retainBlock(ptr %[[V4]])
// CHECK-NOHEAP: store ptr %[[V6]], ptr %b1, align 8
  void assignmentObjCPtr(id a) {
    BlockTy b0 = ^{ foo1(a); };
    id b1;
    b1 = b0;
    ((BlockTy)b1)();
  }
}

// Check that the block capture is released after the full expression.

// CHECK-LABEL: define void @_ZN13test_rval_ref4testEP11objc_object(
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V1:.*]] = call ptr @llvm.objc.retain(
// CHECK: store ptr %[[V1]], ptr %[[BLOCK_CAPTURED]], align 8
// CHECK: invoke void @_ZN13test_rval_ref17callTemplateBlockEOU15__autoreleasingU13block_pointerFvvE(

// CHECK: call void @llvm.objc.storeStrong(ptr %[[BLOCK_CAPTURED]], ptr null)

namespace test_rval_ref {
  void callTemplateBlock(BlockTy &&func);

  void test(id str) {
    return callTemplateBlock(^void() {
      foo1(str);
    });
  }
}
