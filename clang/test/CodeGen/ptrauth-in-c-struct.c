// RUN: %clang_cc1 -triple arm64-apple-ios -fblocks -fptrauth-calls -fptrauth-returns -fptrauth-intrinsics -emit-llvm -o - %s | FileCheck %s

#define AQ __ptrauth(1,1,50)
#define IQ __ptrauth(1,0,50)

typedef void (^BlockTy)(void);

// CHECK: %[[STRUCT_SA:.*]] = type { i32, i32* }
// CHECK: %[[STRUCT_SI:.*]] = type { i32* }

typedef struct {
  int f0;
  int * AQ f1; // Signed using address discrimination.
} SA;

typedef struct {
  int * IQ f; // No address discrimination.
} SI;

SA getSA(void);
void calleeSA(SA);

// CHECK: define void @test_copy_constructor_SA(%[[STRUCT_SA]]* %{{.*}})
// CHECK: call void @__copy_constructor_8_8_t0w4_pa8(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_pa8(i8** %[[DST:.*]], i8** %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca i8**, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca i8**, align 8
// CHECK: store i8** %[[DST]], i8*** %[[DST_ADDR]], align 8
// CHECK: store i8** %[[SRC]], i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load i8**, i8*** %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load i8**, i8*** %[[SRC_ADDR]], align 8
// CHECK: %[[V5:.*]] = bitcast i8** %[[V0]] to i8*
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, i8* %[[V5]], i64 8
// CHECK: %[[V7:.*]] = bitcast i8* %[[V6]] to i8**
// CHECK: %[[V8:.*]] = bitcast i8** %[[V1]] to i8*
// CHECK: %[[V9:.*]] = getelementptr inbounds i8, i8* %[[V8]], i64 8
// CHECK: %[[V10:.*]] = bitcast i8* %[[V9]] to i8**
// CHECK: %[[V11:.*]] = load i8*, i8** %[[V10]], align 8
// CHECK: %[[V12:.*]] = ptrtoint i8** %[[V10]] to i64
// CHECK: %[[V13:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V12]], i64 50)
// CHECK: %[[V14:.*]] = ptrtoint i8** %[[V7]] to i64
// CHECK: %[[V15:.*]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[V14]], i64 50)
// CHECK: %[[V17:.*]] = ptrtoint i8* %[[V11]] to i64
// CHECK: %[[V18:.*]] = call i64 @llvm.ptrauth.resign.i64(i64 %[[V17]], i32 1, i64 %[[V13]], i32 1, i64 %[[V15]])

void test_copy_constructor_SA(SA *s) {
  SA t = *s;
}

// CHECK: define void @test_copy_assignment_SA(
// CHECK: call void @__copy_assignment_8_8_t0w4_pa8(

// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_t0w4_pa8(

void test_copy_assignment_SA(SA *d, SA *s) {
  *d = *s;
}

// CHECK: define void @test_move_constructor_SA(
// CHECK: define internal void @__Block_byref_object_copy_(
// CHECK: define linkonce_odr hidden void @__move_constructor_8_8_t0w4_pa8(

void test_move_constructor_SA(void) {
  __block SA t;
  BlockTy b = ^{ (void)t; };
}

// CHECK: define void @test_move_assignment_SA(
// CHECK: call void @__move_assignment_8_8_t0w4_pa8(
// CHECK: define linkonce_odr hidden void @__move_assignment_8_8_t0w4_pa8(

void test_move_assignment_SA(SA *p) {
  *p = getSA();
}

// CHECK: define void @test_parameter_SA(%[[STRUCT_SA]]* %{{.*}})
// CHECK-NOT: call
// CHECK: ret void

void test_parameter_SA(SA a) {
}

// CHECK: define void @test_argument_SA(%[[STRUCT_SA]]* %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_SA]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[A]], %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_SA]]* %[[AGG_TMP]] to i8**
// CHECK: %[[V2:.*]] = bitcast %[[STRUCT_SA]]* %[[V0]] to i8**
// CHECK: call void @__copy_constructor_8_8_t0w4_pa8(i8** %[[V1]], i8** %[[V2]]) #5
// CHECK: call void @calleeSA(%[[STRUCT_SA]]* %[[AGG_TMP]])
// CHECK-NOT: call
// CHECK: ret void

void test_argument_SA(SA *a) {
  calleeSA(*a);
}

// CHECK: define void @test_return_SA(%[[STRUCT_SA]]* noalias sret %[[AGG_RESULT:.*]], %[[STRUCT_SA]]* %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: store %[[STRUCT_SA]]* %[[A]], %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_SA]]* %[[AGG_RESULT]] to i8**
// CHECK: %[[V2:.*]] = bitcast %[[STRUCT_SA]]* %[[V0]] to i8**
// CHECK: call void @__copy_constructor_8_8_t0w4_pa8(i8** %[[V1]], i8** %[[V2]]) #5
// CHECK-NOT: call
// CHECK: ret void

SA test_return_SA(SA *a) {
  return *a;
}

// CHECK: define void @test_copy_constructor_SI(
// CHECK-NOT: call
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(
// CHECK-NOT: call
// CHECK: ret void

void test_copy_constructor_SI(SI *s) {
  SI t = *s;
}

// CHECK: define void @test_parameter_SI(i64 %{{.*}})
// CHECK-NOT: call
// CHECK: ret void

void test_parameter_SI(SI a) {
}
