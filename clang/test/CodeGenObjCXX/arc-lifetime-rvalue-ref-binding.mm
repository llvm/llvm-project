// RUN: %clang_cc1 -std=c++17 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -O0 -disable-llvm-passes -o - %s | FileCheck %s

// Test for correct IR generation when binding ObjC ARC __strong rvalues
// to const __autoreleasing references. Previously, this caused an assertion
// failure in Qualifiers::addConsistentQualifiers.

// The const id& parameter has implicit __autoreleasing lifetime.
void take(const id&);

// CHECK-LABEL: define{{.*}} void @_Z19test_rvalue_bindingv()
// CHECK: [[OBJ:%.*]] = alloca ptr, align 8
// CHECK: [[REF_TMP:%.*]] = alloca ptr, align 8
// CHECK: store ptr null, ptr [[OBJ]], align 8
// CHECK: [[LOAD:%.*]] = load ptr, ptr [[OBJ]], align 8
// CHECK: [[RETAIN:%.*]] = call ptr @llvm.objc.retain(ptr [[LOAD]])
// CHECK: store ptr [[RETAIN]], ptr [[REF_TMP]], align 8
// CHECK: [[AUTORELEASE:%.*]] = call ptr @llvm.objc.autorelease(ptr [[RETAIN]])
// CHECK: call void @_Z4takeRKU15__autoreleasingP11objc_object(ptr noundef nonnull align 8 dereferenceable(8) [[REF_TMP]])
void test_rvalue_binding() {
  id obj = nullptr;
  take(static_cast<id&&>(obj));
}

// CHECK-LABEL: define{{.*}} void @_Z18test_lvalue_bindingv()
// CHECK: [[OBJ:%.*]] = alloca ptr, align 8
// CHECK: [[REF_TMP:%.*]] = alloca ptr, align 8
// CHECK: store ptr null, ptr [[OBJ]], align 8
// CHECK: [[LOAD:%.*]] = load ptr, ptr [[OBJ]], align 8
// CHECK: store ptr [[LOAD]], ptr [[REF_TMP]], align 8
// CHECK: [[AUTORELEASE:%.*]] = call ptr @llvm.objc.autorelease(ptr [[LOAD]])
// CHECK: call void @_Z4takeRKU15__autoreleasingP11objc_object(ptr noundef nonnull align 8 dereferenceable(8) [[REF_TMP]])
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
// CHECK: call void @_Z4callIJP11objc_objectEEvDpT_(ptr noundef null)
void test_fold_expression() {
  call<id>(nullptr);
}

// CHECK-LABEL: define{{.*}} void @_Z4callIJP11objc_objectEEvDpT_(ptr noundef %args)
// CHECK: [[ARGS_ADDR:%.*]] = alloca ptr, align 8
// CHECK: [[REF_TMP:%.*]] = alloca ptr, align 8
// CHECK: store ptr %args, ptr [[ARGS_ADDR]], align 8
// CHECK: [[LOAD:%.*]] = load ptr, ptr [[ARGS_ADDR]], align 8
// CHECK: [[RETAIN:%.*]] = call ptr @llvm.objc.retain(ptr [[LOAD]])
// CHECK: store ptr [[RETAIN]], ptr [[REF_TMP]], align 8
// CHECK: [[AUTORELEASE:%.*]] = call ptr @llvm.objc.autorelease(ptr [[RETAIN]])
// CHECK: call void @_Z4takeRKU15__autoreleasingP11objc_object(ptr noundef nonnull align 8 dereferenceable(8) [[REF_TMP]])
