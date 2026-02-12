// RUN: %clang_cc1 -std=c++17 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -O0 -disable-llvm-passes -o - %s | FileCheck %s

// Test for correct IR generation when binding ObjC ARC __strong rvalues
// to const __autoreleasing references. Previously, this caused an assertion
// failure in Qualifiers::addConsistentQualifiers.

// The const id& parameter has implicit __autoreleasing lifetime.
void take(const id&);

// CHECK-LABEL: define{{.*}} void @_Z19test_rvalue_bindingv()
// CHECK: [[OBJ:%.*]] = alloca ptr, align 8
// CHECK: store ptr null, ptr [[OBJ]], align 8
// CHECK: call void @_Z4takeRU15__autoreleasingKP11objc_object(ptr noundef nonnull align 8 dereferenceable(8) [[OBJ]])
// CHECK: call void @llvm.objc.storeStrong(ptr [[OBJ]], ptr null)
// CHECK: ret void
void test_rvalue_binding() {
  id obj = nullptr;
  take(static_cast<id&&>(obj));
}

// CHECK-LABEL: define{{.*}} void @_Z19test_lvalue_bindingv()
// CHECK: [[OBJ:%.*]] = alloca ptr, align 8
// CHECK: store ptr null, ptr [[OBJ]], align 8
// CHECK: call void @_Z4takeRU15__autoreleasingKP11objc_object(ptr noundef nonnull align 8 dereferenceable(8) [[OBJ]])
// CHECK: call void @llvm.objc.storeStrong(ptr [[OBJ]], ptr null)
// CHECK: ret void
void test_lvalue_binding() {
  id obj = nullptr;
  take(obj);
}

// Test with fold expressions and perfect forwarding (original crash case).
template <typename... Args>
void call(Args... args) {
  (take(static_cast<Args&&>(args)), ...);
}

// CHECK-LABEL: define{{.*}} void @_Z20test_fold_expressionv()
// CHECK: call void @_Z4callIJU8__strongP11objc_objectEEvDpT_(ptr noundef null)
void test_fold_expression() {
  call<id>(nullptr);
}

// CHECK-LABEL: define{{.*}} void @_Z4callIJU8__strongP11objc_objectEEvDpT_(ptr noundef %args)
// CHECK: [[ARGS_ADDR:%.*]] = alloca ptr, align 8
// CHECK: store ptr null, ptr [[ARGS_ADDR]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr [[ARGS_ADDR]], ptr %args)
// CHECK: call void @_Z4takeRU15__autoreleasingKP11objc_object(ptr noundef nonnull align 8 dereferenceable(8) [[ARGS_ADDR]])
// CHECK: call void @llvm.objc.storeStrong(ptr [[ARGS_ADDR]], ptr null)
// CHECK: ret void
