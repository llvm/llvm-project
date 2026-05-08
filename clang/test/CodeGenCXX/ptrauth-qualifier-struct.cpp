// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -std=c++11 -emit-llvm %s  -o - | FileCheck -check-prefixes=CHECK,IOS %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls -fptrauth-intrinsics -std=c++11 -emit-llvm %s  -o - | FileCheck %s

#define AQ __ptrauth(1,1,50)
#define IQ __ptrauth(1,0,50)

// CHECK: %[[STRUCT_SA:.*]] = type { ptr, ptr }
// CHECK: %[[STRUCT_SI:.*]] = type { ptr }

struct SA {
  int * AQ m0; // Signed using address discrimination.
  int * AQ m1; // Signed using address discrimination.
};

struct SI {
  int * IQ m; // No address discrimination.
};

struct __attribute__((trivial_abi)) TrivialSA {
  int * AQ m0; // Signed using address discrimination.
  int * AQ m1; // Signed using address discrimination.
};

// Check that TrivialSA is passed indirectly despite being annotated with
// 'trivial_abi'.

// CHECK: define {{.*}}void @_Z18testParamTrivialSA9TrivialSA(ptr noundef dead_on_return %{{.*}})

void testParamTrivialSA(TrivialSA a) {
}

// CHECK: define {{.*}}void @_Z19testCopyConstructor2SA(ptr
// CHECK: call {{.*}}@_ZN2SAC1ERKS_(

// CHECK: define linkonce_odr {{.*}}@_ZN2SAC1ERKS_(
// CHECK: call {{.*}}@_ZN2SAC2ERKS_(

void testCopyConstructor(SA a) {
  SA t = a;
}

// CHECK: define {{.*}}void @_Z19testMoveConstructor2SA(ptr
// CHECK: call {{.*}}@_ZN2SAC1EOS_(

// CHECK: define linkonce_odr {{.*}}@_ZN2SAC1EOS_(
// CHECK: call {{.*}}@_ZN2SAC2EOS_(

void testMoveConstructor(SA a) {
  SA t = static_cast<SA &&>(a);
}

// CHECK: define {{.*}}void @_Z18testCopyAssignment2SA(ptr
// CHECK: call noundef nonnull align 8 dereferenceable(16) ptr @_ZN2SAaSERKS_(

// CHECK: define {{.*}}linkonce_odr noundef nonnull align 8 dereferenceable(16) ptr @_ZN2SAaSERKS_(ptr noundef nonnull align 8 dereferenceable(16) %[[THIS:.*]], ptr noundef nonnull align 8 dereferenceable(16) %0)
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: store ptr %[[V0:.*]], ptr %[[_ADDR]], align 8
// CHECK: %[[THISI:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[THISI]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load ptr, ptr %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint ptr %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint ptr %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint ptr %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])

void testCopyAssignment(SA a) {
  SA t;
  t = a;
}

// CHECK: define {{.*}}void @_Z18testMoveAssignment2SA(ptr
// CHECK: call noundef nonnull align 8 dereferenceable(16) ptr @_ZN2SAaSEOS_(

// CHECK: define {{.*}}linkonce_odr noundef nonnull align 8 dereferenceable(16) ptr @_ZN2SAaSEOS_(ptr noundef nonnull align 8 dereferenceable(16) %[[THIS:.*]], ptr noundef nonnull align 8 dereferenceable(16) %0)
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: store ptr %[[V0:.*]], ptr %[[_ADDR]], align 8
// CHECK: %[[THISI:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[THISI]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load ptr, ptr %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint ptr %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint ptr %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint ptr %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])

void testMoveAssignment(SA a) {
  SA t;
  t = static_cast<SA &&>(a);
}

// CHECK: define {{.*}}void @_Z19testCopyConstructor2SI(
// CHECK: call void @llvm.memcpy.p0.p0.i64(

void testCopyConstructor(SI a) {
  SI t = a;
}

// CHECK: define {{.*}}void @_Z19testMoveConstructor2SI(
// CHECK: call void @llvm.memcpy.p0.p0.i64(

void testMoveConstructor(SI a) {
  SI t = static_cast<SI &&>(a);
}

// CHECK: define {{.*}}void @_Z18testCopyAssignment2SI(
// CHECK: call void @llvm.memcpy.p0.p0.i64(

void testCopyAssignment(SI a) {
  SI t;
  t = a;
}

// CHECK: define {{.*}}void @_Z18testMoveAssignment2SI(
// CHECK: call void @llvm.memcpy.p0.p0.i64(

void testMoveAssignment(SI a) {
  SI t;
  t = static_cast<SI &&>(a);
}

// CHECK: define linkonce_odr {{.*}}@_ZN2SAC2ERKS_(ptr noundef nonnull align 8 dereferenceable(16) %[[THIS:.*]], ptr noundef nonnull align 8 dereferenceable(16) %0)
// IOS: %[[RETVAL:.*]] = alloca ptr, align 8
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: store ptr %[[V0:.*]], ptr %[[_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// IOS: store ptr %[[THIS1]], ptr %[[RETVAL]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load ptr, ptr %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint ptr %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint ptr %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint ptr %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])

// CHECK: define linkonce_odr {{.*}}@_ZN2SAC2EOS_(ptr noundef nonnull align 8 dereferenceable(16) %[[THIS:.*]], ptr noundef nonnull align 8 dereferenceable(16) %0)
// IOS: %[[RETVAL:.*]] = alloca ptr, align 8
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[THIS]], ptr %[[THIS_ADDR]], align 8
// CHECK: store ptr %[[V0:.*]], ptr %[[_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]], align 8
// IOS: store ptr %[[THIS1]], ptr %[[RETVAL]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load ptr, ptr %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint ptr %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint ptr %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint ptr %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])
