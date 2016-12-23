// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -fblocks -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=UNOPT
// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -fblocks -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -O -disable-llvm-passes | FileCheck %s -check-prefix=CHECK -check-prefix=OPT

typedef __strong id strong_id;
typedef __weak id weak_id;

// CHECK-LABEL: define void @_Z8test_newP11objc_object
void test_new(id invalue) {
  // CHECK: [[INVALUEADDR:%.*]] = alloca i8*
  // UNOPT-NEXT: store i8* null, i8** [[INVALUEADDR]]
  // UNOPT-NEXT: call void @objc_storeStrong(i8** [[INVALUEADDR]], i8* [[INVALUE:%.*]])
  // OPT-NEXT: [[T0:%.*]] = call i8* @objc_retain(i8* [[INVALUE:%.*]])
  // OPT-NEXT: store i8* [[T0]], i8** [[INVALUEADDR]]

  // CHECK: call i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // CHECK-NEXT: store i8* null, i8**
  new strong_id;
  // CHECK: call i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // UNOPT-NEXT: store i8* null, i8**
  // OPT-NEXT: call i8* @objc_initWeak(i8** {{.*}}, i8* null)
  new weak_id;

  // CHECK: call i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // CHECK-NEXT: store i8* null, i8**
  new __strong id;
  // CHECK: call i8* @_Znwm
  // CHECK-NEXT: {{bitcast i8\*.*to i8\*\*}}
  // UNOPT-NEXT: store i8* null, i8**
  // OPT-NEXT: call i8* @objc_initWeak(i8** {{.*}}, i8* null)
  new __weak id;

  // CHECK: call i8* @_Znwm
  // CHECK: call i8* @objc_retain
  // CHECK: store i8*
  new __strong id(invalue);

  // CHECK: call i8* @_Znwm
  // CHECK: call i8* @objc_initWeak
  new __weak id(invalue);

  // UNOPT: call void @objc_storeStrong
  // OPT: call void @objc_release
  // CHECK: ret void
}

// CHECK-LABEL: define void @_Z14test_array_new
void test_array_new() {
  // CHECK: call i8* @_Znam
  // CHECK: store i64 17, i64*
  // CHECK: call void @llvm.memset.p0i8.i64
  new strong_id[17];

  // CHECK: call i8* @_Znam
  // CHECK: store i64 17, i64*
  // CHECK: call void @llvm.memset.p0i8.i64
  new weak_id[17];
  // CHECK: ret void
}

// CHECK-LABEL: define void @_Z11test_deletePU8__strongP11objc_objectPU6__weakS0_
void test_delete(__strong id *sptr, __weak id *wptr) {
  // CHECK: br i1
  // UNOPT: call void @objc_storeStrong(i8** {{.*}}, i8* null)
  // OPT: load i8*, i8**
  // OPT-NEXT: call void @objc_release
  // CHECK: call void @_ZdlPv
  delete sptr;

  // CHECK: call void @objc_destroyWeak
  // CHECK: call void @_ZdlPv
  delete wptr;

  // CHECK: ret void
}

// CHECK-LABEL: define void @_Z17test_array_deletePU8__strongP11objc_objectPU6__weakS0_
void test_array_delete(__strong id *sptr, __weak id *wptr) {
  // CHECK: icmp eq i8** [[BEGIN:%.*]], null
  // CHECK: [[LEN:%.*]] = load i64, i64* {{%.*}}
  // CHECK: [[END:%.*]] = getelementptr inbounds i8*, i8** [[BEGIN]], i64 [[LEN]]
  // CHECK-NEXT: icmp eq i8** [[BEGIN]], [[END]]
  // CHECK: [[PAST:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]],
  // CHECK-NEXT: [[CUR]] = getelementptr inbounds i8*, i8** [[PAST]], i64 -1
  // UNOPT-NEXT: call void @objc_storeStrong(i8** [[CUR]], i8* null)
  // OPT-NEXT: [[T0:%.*]] = load i8*, i8** [[CUR]]
  // OPT-NEXT: objc_release(i8* [[T0]])
  // CHECK-NEXT: icmp eq i8** [[CUR]], [[BEGIN]]
  // CHECK: call void @_ZdaPv
  delete [] sptr;

  // CHECK: icmp eq i8** [[BEGIN:%.*]], null
  // CHECK: [[LEN:%.*]] = load i64, i64* {{%.*}}
  // CHECK: [[END:%.*]] = getelementptr inbounds i8*, i8** [[BEGIN]], i64 [[LEN]]
  // CHECK-NEXT: icmp eq i8** [[BEGIN]], [[END]]
  // CHECK: [[PAST:%.*]] = phi i8** [ [[END]], {{%.*}} ], [ [[CUR:%.*]],
  // CHECK-NEXT: [[CUR]] = getelementptr inbounds i8*, i8** [[PAST]], i64 -1
  // CHECK-NEXT: call void @objc_destroyWeak(i8** [[CUR]])
  // CHECK-NEXT: icmp eq i8** [[CUR]], [[BEGIN]]
  // CHECK: call void @_ZdaPv
  delete [] wptr;
}
