// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -O2 -std=c++11 -disable-llvm-optzns -o - %s | FileCheck %s

// define void @_Z11simple_moveRU8__strongP11objc_objectS2_
void simple_move(__strong id &x, __strong id &y) {
  // CHECK: = load i8*, i8**
  // CHECK: store i8* null
  // CHECK: = load i8*, i8**
  // CHECK: store i8*
  // CHECK-NEXT: call void @objc_release
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

// CHECK-LABEL: define void @_Z12library_moveRU8__strongP11objc_objectS2_
void library_move(__strong id &x, __strong id &y) {
  // CHECK: call dereferenceable({{[0-9]+}}) i8** @_Z4moveIRU8__strongP11objc_objectEON16remove_referenceIT_E4typeEOS5_
  // CHECK: load i8*, i8**
  // CHECK: store i8* null, i8**
  // CHECK: load i8**, i8***
  // CHECK-NEXT: load i8*, i8**
  // CHECK-NEXT: store i8*
  // CHECK-NEXT: call void @objc_release
  // CHECK-NEXT: ret void
  x = move(y);
}

// CHECK-LABEL: define void @_Z12library_moveRU8__strongP11objc_object
void library_move(__strong id &y) {
  // CHECK: [[X:%.*]] = alloca i8*, align 8
  // CHECK: [[I:%.*]] = alloca i32, align 4
  // CHECK:      [[XPTR1:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 8, i8* [[XPTR1]])
  // CHECK: [[Y:%[a-zA-Z0-9]+]] = call dereferenceable({{[0-9]+}}) i8** @_Z4moveIRU8__strongP11objc_objectEON16remove_referenceIT_E4typeEOS5_
  // Load the object
  // CHECK-NEXT: [[OBJ:%[a-zA-Z0-9]+]] = load i8*, i8** [[Y]]
  // Null out y
  // CHECK-NEXT: store i8* null, i8** [[Y]]
  // Initialize x with the object
  // CHECK-NEXT: store i8* [[OBJ]], i8** [[X:%[a-zA-Z0-9]+]]
  id x = move(y);

  // CHECK-NEXT: [[IPTR1:%.*]] = bitcast i32* [[I]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start(i64 4, i8* [[IPTR1]])
  // CHECK-NEXT: store i32 17
  int i = 17;
  // CHECK-NEXT: [[IPTR2:%.*]] = bitcast i32* [[I]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 4, i8* [[IPTR2]])
  // CHECK-NEXT: [[OBJ:%[a-zA-Z0-9]+]] = load i8*, i8** [[X]]
  // CHECK-NEXT: call void @objc_release(i8* [[OBJ]])
  // CHECK-NEXT: [[XPTR2:%.*]] = bitcast i8** [[X]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end(i64 8, i8* [[XPTR2]])
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @_Z10const_moveRU8__strongKP11objc_object(
void const_move(const __strong id &x) {
  // CHECK:      [[Y:%.*]] = alloca i8*,
  // CHECK:      [[X:%.*]] = call dereferenceable({{[0-9]+}}) i8** @_Z4moveIRU8__strongKP11objc_objectEON16remove_referenceIT_E4typeEOS5_(
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[X]]
  // CHECK-NEXT: [[T1:%.*]] = call i8* @objc_retain(i8* [[T0]])
  // CHECK-NEXT: store i8* [[T1]], i8** [[Y]]
  // CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
  // CHECK-NEXT: call void @objc_release(i8* [[T0]])
  id y = move(x);
}
