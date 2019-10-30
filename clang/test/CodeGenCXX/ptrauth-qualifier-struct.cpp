// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -std=c++11 -emit-llvm %s  -o - | FileCheck %s

#define AQ __ptrauth(1,1,50)
#define IQ __ptrauth(1,0,50)

// CHECK: %[[STRUCT_TRIVIALSA:.*]] = type { i32*, i32* }
// CHECK: %[[STRUCT_SA:.*]] = type { i32*, i32* }
// CHECK: %[[STRUCT_SI:.*]] = type { i32* }

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

// CHECK: define void @_Z18testParamTrivialSA9TrivialSA(%[[STRUCT_TRIVIALSA]]* %{{.*}})

void testParamTrivialSA(TrivialSA a) {
}

// CHECK: define void @_Z19testCopyConstructor2SA(%[[STRUCT_SA]]*
// CHECK: call %[[STRUCT_SA]]* @_ZN2SAC1ERKS_(

// CHECK: define linkonce_odr %[[STRUCT_SA]]* @_ZN2SAC1ERKS_(
// CHECK: call %[[STRUCT_SA]]* @_ZN2SAC2ERKS_(

void testCopyConstructor(SA a) {
  SA t = a;
}

// CHECK: define void @_Z19testMoveConstructor2SA(%[[STRUCT_SA]]*
// CHECK: call %[[STRUCT_SA]]* @_ZN2SAC1EOS_(

// CHECK: define linkonce_odr %[[STRUCT_SA]]* @_ZN2SAC1EOS_(
// CHECK: call %[[STRUCT_SA]]* @_ZN2SAC2EOS_(

void testMoveConstructor(SA a) {
  SA t = static_cast<SA &&>(a);
}

// CHECK: define void @_Z18testCopyAssignment2SA(%[[STRUCT_SA]]*
// CHECK: call dereferenceable(16) %[[STRUCT_SA]]* @_ZN2SAaSERKS_(

// CHECK: define linkonce_odr dereferenceable(16) %[[STRUCT_SA:.*]]* @_ZN2SAaSERKS_(%[[STRUCT_SA]]* %[[THIS:.*]], %[[STRUCT_SA]]* dereferenceable(16) %0)
// CHECK: %[[THIS_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: store %[[STRUCT_SA]]* %[[THIS]], %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[V0:.*]], %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[THISI:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[THISI]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load i32*, i32** %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint i32** %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint i32** %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint i32* %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign.i64(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])

void testCopyAssignment(SA a) {
  SA t;
  t = a;
}

// CHECK: define void @_Z18testMoveAssignment2SA(%[[STRUCT_SA]]*
// CHECK: call dereferenceable(16) %[[STRUCT_SA]]* @_ZN2SAaSEOS_(

// CHECK: define linkonce_odr dereferenceable(16) %[[STRUCT_SA:.*]]* @_ZN2SAaSEOS_(%[[STRUCT_SA]]* %[[THIS:.*]], %[[STRUCT_SA]]* dereferenceable(16) %0)
// CHECK: %[[THIS_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: store %[[STRUCT_SA]]* %[[THIS]], %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[V0:.*]], %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[THISI:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[THISI]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load i32*, i32** %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint i32** %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint i32** %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint i32* %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign.i64(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])

void testMoveAssignment(SA a) {
  SA t;
  t = static_cast<SA &&>(a);
}

// CHECK: define void @_Z19testCopyConstructor2SI(i
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(

void testCopyConstructor(SI a) {
  SI t = a;
}

// CHECK: define void @_Z19testMoveConstructor2SI(
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(

void testMoveConstructor(SI a) {
  SI t = static_cast<SI &&>(a);
}

// CHECK: define void @_Z18testCopyAssignment2SI(
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(

void testCopyAssignment(SI a) {
  SI t;
  t = a;
}

// CHECK: define void @_Z18testMoveAssignment2SI(
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(

void testMoveAssignment(SI a) {
  SI t;
  t = static_cast<SI &&>(a);
}

// CHECK: define linkonce_odr %[[STRUCT_SA:.*]]* @_ZN2SAC2ERKS_(%[[STRUCT_SA]]* %[[THIS:.*]], %[[STRUCT_SA]]* dereferenceable(16) %0)
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[THIS_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: store %[[STRUCT_SA]]* %[[THIS]], %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[V0:.*]], %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[THIS1]], %[[STRUCT_SA]]** %[[RETVAL]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[THIS1]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load i32*, i32** %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint i32** %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint i32** %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint i32* %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign.i64(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])

// CHECK: define linkonce_odr %[[STRUCT_SA:.*]]* @_ZN2SAC2EOS_(%[[STRUCT_SA]]* %[[THIS:.*]], %[[STRUCT_SA]]* dereferenceable(16) %0)
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[THIS_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: store %[[STRUCT_SA]]* %[[THIS]], %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[V0:.*]], %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[THIS1:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[THIS_ADDR]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[THIS1]], %[[STRUCT_SA]]** %[[RETVAL]], align 8
// CHECK: %[[M0:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[THIS1]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[_ADDR]], align 8
// CHECK: %[[M02:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %[[V1]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load i32*, i32** %[[M02]], align 8
// CHECK: %[[V3:.*]] = ptrtoint i32** %[[M02]] to i64
// CHECK: %[[V4:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V3]], i64 50)
// CHECK: %[[V5:.*]] = ptrtoint i32** %[[M0]] to i64
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V5]], i64 50)
// CHECK: %[[V8:.*]] = ptrtoint i32* %[[V2]] to i64
// CHECK: %[[V9:.*]] = call i64 @llvm.ptrauth.resign.i64(i64 %[[V8]], i32 1, i64 %[[V4]], i32 1, i64 %[[V6]])
