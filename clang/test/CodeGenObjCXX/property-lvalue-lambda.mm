// RUN: %clang_cc1 -no-opaque-pointers -no-enable-noundef-analysis -fblocks -disable-llvm-passes -triple x86_64-apple-darwin10 -std=c++17 -emit-llvm -o - %s | FileCheck %s

typedef void (^blk_t)();
typedef void (*fnptr_t)();

@interface X
@property blk_t blk;
@property fnptr_t fnptr;
@end

template <class T>
blk_t operator+(blk_t lhs, T) { return lhs; }

template <class T>
fnptr_t operator+(fnptr_t lhs, T) { return lhs; }

// CHECK-LABEL: define{{.*}} void @_Z2t1P1X
void t1(X *x) {
  // Check that we call lambda.operator blk_t(), and that we send that result to
  // the setter.

  // CHECK: [[CALL:%.*]] = call void ()* @"_ZZ2t1P1XENK3$_0cvU13block_pointerFvvEEv"
  // CHECK: call void{{.*}}@objc_msgSend{{.*}}({{.*}} void ()* [[CALL]])
  x.blk = [] {};

  // CHECK: [[CALL2:%.*]] = call void ()* @"_ZZ2t1P1XENK3$_1cvPFvvEEv"
  // CHECK: call void{{.*}}@objc_msgSend{{.*}}({{.*}} void ()* [[CALL2]])
  x.fnptr = [] {};
}

// CHECK-LABEL: define{{.*}} void @_Z2t2P1X
void t2(X *x) {
  // Test the case when the lambda isn't unique. (see OpaqueValueExpr::isUnique)
  // FIXME: This asserts if the lambda isn't trivially copy/movable.

  // [x setBlk: operator+([x blk], [] {})]

  // CHECK: call void{{.*}}@objc_msgSend{{.*}}
  // CHECK: [[PLUS:%.*]] = call void ()* @"_ZplIZ2t2P1XE3$_2EU13block_pointerFvvES4_T_"
  // CHECK: call void{{.*}}@objc_msgSend{{.*}}({{.*}} [[PLUS]])
  x.blk += [] {};

  // CHECK: call void{{.*}}@objc_msgSend{{.*}}
  // CHECK: [[PLUS:%.*]] = call void ()* @"_ZplIZ2t2P1XE3$_3EPFvvES4_T_"
  // CHECK: call void{{.*}}@objc_msgSend{{.*}}({{.*}} [[PLUS]])
  x.fnptr += [] {};
}
