// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -fobjc-arc -fblocks -std=c++1y -emit-pch %s -o %t
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -fobjc-arc -fblocks -std=c++1y -include-pch %t -fobjc-avoid-heapify-local-blocks -emit-llvm -o - %s | FileCheck %s

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

namespace test_block_retain {
  typedef void (^BlockTy)();
  void foo1(id);

  inline void initialization(id a) {
    // Call to @llvm.objc.retainBlock isn't needed.
    BlockTy b0 = ^{ foo1(a); };
    b0();
  }

  inline void assignmentConditional(id a, bool c) {
    BlockTy b0;
    if (c)
      // @llvm.objc.retainBlock is called since 'b0' is declared in the outer scope.
      b0 = ^{ foo1(a); };
    b0();
  }
}

#else

// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }

namespace test_block_retain {
// CHECK-LABEL: define linkonce_odr void @_ZN17test_block_retain14initializationEP11objc_object(
// CHECK-NOT: call i8* @llvm.objc.retainBlock(

  void test_initialization(id a) {
    initialization(a);
  }

// CHECK-LABEL: define{{.*}} void @_ZN17test_block_retain26test_assignmentConditionalEP11objc_objectb(
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>, align 8
// CHECK: %[[V4:.*]] = bitcast <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8* }>* %[[BLOCK]] to void ()*
// CHECK: %[[V5:.*]] = bitcast void ()* %[[V4]] to i8*
// CHECK: call i8* @llvm.objc.retainBlock(i8* %[[V5]])

  void test_assignmentConditional(id a, bool c) {
    assignmentConditional(a, c);
  }
}

#endif
