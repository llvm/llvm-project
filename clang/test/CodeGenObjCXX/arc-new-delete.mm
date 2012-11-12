// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -fblocks -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s | FileCheck %s

typedef __strong id strong_id;
typedef __weak id weak_id;

// CHECK: define void @_Z8test_newP11objc_object
void test_new(id invalue) {
  // CHECK: alloca i8*
  // CHECK-NEXT: call i8* @objc_retain

  // CHECK: call noalias i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // CHECK-NEXT: store i8* null, i8**
  new strong_id;
  // CHECK: call noalias i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // CHECK-NEXT: store i8* null, i8**
  new weak_id;

  // CHECK: call noalias i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // CHECK-NEXT: store i8* null, i8**
  new __strong id;
  // CHECK: call noalias i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // CHECK-NEXT: store i8* null, i8**
  new __weak id;

  // CHECK: call noalias i8* @_Znwm
  // CHECK: call i8* @objc_retain
  // CHECK: store i8*
  new __strong id(invalue);

  // CHECK: call noalias i8* @_Znwm
  // CHECK: call i8* @objc_initWeak
  new __weak id(invalue);

  // CHECK: call void @objc_storeStrong
  // CHECK: ret void
}

// CHECK: define void @_Z14test_array_new
void test_array_new() {
  // CHECK: call noalias i8* @_Znam
  // CHECK: store i64 17, i64*
  // CHECK: call void @llvm.memset.p0i8.i64
  new strong_id[17];

  // CHECK: call noalias i8* @_Znam
  // CHECK: store i64 17, i64*
  // CHECK: call void @llvm.memset.p0i8.i64
  new weak_id[17];
  // CHECK: ret void
}

// CHECK: define void @_Z11test_deletePU8__strongP11objc_objectPU6__weakS0_
void test_delete(__strong id *sptr, __weak id *wptr) {
  // CHECK: br i1
  // CHECK: load i8**
  // CHECK-NEXT: call void @objc_release
  // CHECK: call void @_ZdlPv
  delete sptr;

  // CHECK: call void @objc_destroyWeak
  // CHECK: call void @_ZdlPv
  delete wptr;

  // CHECK: ret void
}

// CHECK: define void @_Z17test_array_deletePU8__strongP11objc_objectPU6__weakS0_
void test_array_delete(__strong id *sptr, __weak id *wptr) {
  // CHECK: icmp eq i8** [[BEGIN:%.*]], null
  // CHECK: [[LEN:%.*]] = load i64* {{%.*}}
  // CHECK: [[END:%.*]] = getelementptr inbounds i8** [[BEGIN]], i64 [[LEN]]
  // CHECK-NEXT: icmp eq i8** [[BEGIN]], [[END]]
  // CHECK: [[PAST:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]],
  // CHECK-NEXT: [[CUR]] = getelementptr inbounds i8** [[PAST]], i64 -1
  // CHECK-NEXT: call void @objc_storeStrong(i8** [[CUR]], i8* null)
  // CHECK-NEXT: icmp eq i8** [[CUR]], [[BEGIN]]
  // CHECK: call void @_ZdaPv
  delete [] sptr;

  // CHECK: icmp eq i8** [[BEGIN:%.*]], null
  // CHECK: [[LEN:%.*]] = load i64* {{%.*}}
  // CHECK: [[END:%.*]] = getelementptr inbounds i8** [[BEGIN]], i64 [[LEN]]
  // CHECK-NEXT: icmp eq i8** [[BEGIN]], [[END]]
  // CHECK: [[PAST:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]],
  // CHECK-NEXT: [[CUR]] = getelementptr inbounds i8** [[PAST]], i64 -1
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[CUR]])
  // CHECK-NEXT: icmp eq i8** [[CUR]], [[BEGIN]]
  // CHECK: call void @_ZdaPv
  delete [] wptr;
}
