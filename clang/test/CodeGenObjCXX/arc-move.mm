// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -O2 -std=c++11 -disable-llvm-passes -o - %s | FileCheck %s

// define{{.*}} void @_Z11simple_moveRU8__strongP11objc_objectS2_
void simple_move(__strong id &x, __strong id &y) {
  // CHECK: = load ptr, ptr
  // CHECK: store ptr null
  // CHECK: = load ptr, ptr
  // CHECK: store ptr
  // CHECK-NEXT: call void @llvm.objc.release
  x = static_cast<__strong id&&>(y);
  // CHECK-NEXT: ret void
}

template<typename T>
struct remove_reference {
  typedef T type;
};

template<typename T>
struct remove_reference<T&> {
  typedef T type;
};

template<typename T>
struct remove_reference<T&&> {
  typedef T type;
};

template<typename T> 
typename remove_reference<T>::type&& move(T &&x) { 
  return static_cast<typename remove_reference<T>::type&&>(x); 
}

// CHECK-LABEL: define{{.*}} void @_Z12library_moveRU8__strongP11objc_objectS2_
void library_move(__strong id &x, __strong id &y) {
  // CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z4moveIRU8__strongP11objc_objectEON16remove_referenceIT_E4typeEOS5_
  // CHECK: load ptr, ptr
  // CHECK: store ptr null, ptr
  // CHECK: load ptr, ptr
  // CHECK-NEXT: load ptr, ptr
  // CHECK-NEXT: store ptr
  // CHECK-NEXT: call void @llvm.objc.release
  // CHECK-NEXT: ret void
  x = move(y);
}

// CHECK-LABEL: define{{.*}} void @_Z12library_moveRU8__strongP11objc_object
void library_move(__strong id &y) {
  // CHECK: [[X:%x]] = alloca ptr, align 8
  // CHECK: [[I:%.*]] = alloca i32, align 4
  // CHECK: call void @llvm.lifetime.start.p0(i64 8, ptr [[X]])
  // CHECK: [[Y:%[a-zA-Z0-9]+]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z4moveIRU8__strongP11objc_objectEON16remove_referenceIT_E4typeEOS5_
  // Load the object
  // CHECK-NEXT: [[OBJ:%[a-zA-Z0-9]+]] = load ptr, ptr [[Y]]
  // Null out y
  // CHECK-NEXT: store ptr null, ptr [[Y]]
  // Initialize x with the object
  // CHECK-NEXT: store ptr [[OBJ]], ptr [[X:%[a-zA-Z0-9]+]]
  id x = move(y);

  // CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 4, ptr [[I]])
  // CHECK-NEXT: store i32 17
  int i = 17;
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4, ptr [[I]])
  // CHECK-NEXT: [[OBJ:%[a-zA-Z0-9]+]] = load ptr, ptr [[X]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[OBJ]])
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[X]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z10const_moveRU8__strongKP11objc_object(
void const_move(const __strong id &x) {
  // CHECK:      [[Y:%y]] = alloca ptr,
  // CHECK:      [[X:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_Z4moveIRU8__strongKP11objc_objectEON16remove_referenceIT_E4typeEOS5_(
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[X]]
  // CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.retain(ptr [[T0]])
  // CHECK-NEXT: store ptr [[T1]], ptr [[Y]]
  // CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[T0]])
  id y = move(x);
}
