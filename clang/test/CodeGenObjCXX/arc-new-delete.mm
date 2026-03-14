// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -fblocks -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=UNOPT
// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -fblocks -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -O -disable-llvm-passes | FileCheck %s -check-prefix=CHECK -check-prefix=OPT

typedef __strong id strong_id;
typedef __weak id weak_id;

// CHECK-LABEL: define{{.*}} void @_Z8test_newP11objc_object
void test_new(id invalue) {
  // CHECK: [[INVALUEADDR:%.*]] = alloca ptr
  // UNOPT-NEXT: store ptr null, ptr [[INVALUEADDR]]
  // UNOPT-NEXT: call void @llvm.objc.storeStrong(ptr [[INVALUEADDR]], ptr [[INVALUE:%.*]])
  // OPT-NEXT: [[T0:%.*]] = call ptr @llvm.objc.retain(ptr [[INVALUE:%.*]])
  // OPT-NEXT: store ptr [[T0]], ptr [[INVALUEADDR]]

  // CHECK: [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znwm
  // CHECK-NEXT: store ptr null, ptr
  new strong_id;
  // CHECK: [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znwm
  // UNOPT-NEXT: store ptr null, ptr
  // OPT-NEXT: call ptr @llvm.objc.initWeak(ptr {{.*}}, ptr null)
  new weak_id;

  // CHECK: [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znwm
  // CHECK-NEXT: store ptr null, ptr
  new __strong id;
  // CHECK: [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znwm
  // UNOPT-NEXT: store ptr null, ptr
  // OPT-NEXT: call ptr @llvm.objc.initWeak(ptr {{.*}}, ptr null)
  new __weak id;

  // CHECK: [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znwm
  // CHECK: call ptr @llvm.objc.retain
  // CHECK: store ptr
  new __strong id(invalue);

  // CHECK: [[CALL:%.*]] = call noalias noundef nonnull ptr @_Znwm
  // CHECK: call ptr @llvm.objc.initWeak
  new __weak id(invalue);

  // UNOPT: call void @llvm.objc.storeStrong
  // OPT: call void @llvm.objc.release
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z14test_array_new
void test_array_new() {
  // CHECK: call noalias noundef nonnull ptr @_Znam
  // CHECK: store i64 17, ptr
  // CHECK: call void @llvm.memset.p0.i64
  new strong_id[17];

  // CHECK: call noalias noundef nonnull ptr @_Znam
  // CHECK: store i64 17, ptr
  // CHECK: call void @llvm.memset.p0.i64
  new weak_id[17];
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z11test_deletePU8__strongP11objc_objectPU6__weakS0_
void test_delete(__strong id *sptr, __weak id *wptr) {
  // CHECK: br i1
  // UNOPT: call void @llvm.objc.storeStrong(ptr {{.*}}, ptr null)
  // OPT: load ptr, ptr
  // OPT-NEXT: call void @llvm.objc.release
  // CHECK: call void @_ZdlPv
  delete sptr;

  // CHECK: call void @llvm.objc.destroyWeak
  // CHECK: call void @_ZdlPv
  delete wptr;

  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z17test_array_deletePU8__strongP11objc_objectPU6__weakS0_
void test_array_delete(__strong id *sptr, __weak id *wptr) {
  // CHECK: icmp eq ptr [[BEGIN:%.*]], null
  // CHECK: [[LEN:%.*]] = load i64, ptr {{%.*}}
  // CHECK: [[END:%.*]] = getelementptr inbounds ptr, ptr [[BEGIN]], i64 [[LEN]]
  // CHECK-NEXT: icmp eq ptr [[BEGIN]], [[END]]
  // CHECK: [[PAST:%.*]] = phi ptr [ [[END]], {{%.*}} ], [ [[CUR:%.*]],
  // CHECK-NEXT: [[CUR]] = getelementptr inbounds ptr, ptr [[PAST]], i64 -1
  // UNOPT-NEXT: call void @llvm.objc.storeStrong(ptr [[CUR]], ptr null)
  // OPT-NEXT: [[T0:%.*]] = load ptr, ptr [[CUR]]
  // OPT-NEXT: llvm.objc.release(ptr [[T0]])
  // CHECK-NEXT: icmp eq ptr [[CUR]], [[BEGIN]]
  // CHECK: call void @_ZdaPv
  delete [] sptr;

  // CHECK: icmp eq ptr [[BEGIN:%.*]], null
  // CHECK: [[LEN:%.*]] = load i64, ptr {{%.*}}
  // CHECK: [[END:%.*]] = getelementptr inbounds ptr, ptr [[BEGIN]], i64 [[LEN]]
  // CHECK-NEXT: icmp eq ptr [[BEGIN]], [[END]]
  // CHECK: [[PAST:%.*]] = phi ptr [ [[END]], {{%.*}} ], [ [[CUR:%.*]],
  // CHECK-NEXT: [[CUR]] = getelementptr inbounds ptr, ptr [[PAST]], i64 -1
  // CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[CUR]])
  // CHECK-NEXT: icmp eq ptr [[CUR]], [[BEGIN]]
  // CHECK: call void @_ZdaPv
  delete [] wptr;
}
