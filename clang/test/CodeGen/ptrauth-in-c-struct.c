// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-ios -fblocks -fptrauth-calls -fptrauth-returns -fptrauth-intrinsics -emit-llvm -o - %s | FileCheck %s

#define AQ1_50 __ptrauth(1,1,50)
#define AQ2_30 __ptrauth(2,1,30)
#define IQ __ptrauth(1,0,50)

typedef void (^BlockTy)(void);

// CHECK: %[[STRUCT_SA:.*]] = type { i32, i32* }
// CHECK: %[[STRUCT_SA2:.*]] = type { i32, i32* }
// CHECK: %[[STRUCT_SI:.*]] = type { i32* }

typedef struct {
  int f0;
  int * AQ1_50 f1; // Signed using address discrimination.
} SA;

typedef struct {
  int f0;
  int * AQ2_30 f1; // Signed using address discrimination.
} SA2;

typedef struct {
  int * IQ f; // No address discrimination.
} SI;

SA getSA(void);
void calleeSA(SA);

int g0;

// CHECK: define void @test_copy_constructor_SA(%[[STRUCT_SA]]* noundef %{{.*}})
// CHECK: call void @__copy_constructor_8_8_t0w4_pa1_50_8(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_pa1_50_8(i8** noundef %[[DST:.*]], i8** noundef %[[SRC:.*]])
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
// CHECK: %[[V13:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V12]], i64 50)
// CHECK: %[[V14:.*]] = ptrtoint i8** %[[V7]] to i64
// CHECK: %[[V15:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V14]], i64 50)
// CHECK: %[[V17:.*]] = ptrtoint i8* %[[V11]] to i64
// CHECK: %[[V18:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V17]], i32 1, i64 %[[V13]], i32 1, i64 %[[V15]])

void test_copy_constructor_SA(SA *s) {
  SA t = *s;
}

// CHECK: define void @test_copy_constructor_SA2(%[[STRUCT_SA2]]* noundef %{{.*}})
// CHECK: call void @__copy_constructor_8_8_t0w4_pa2_30_8(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_pa2_30_8(i8** noundef %[[DST:.*]], i8** noundef %[[SRC:.*]])
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
// CHECK: %[[V13:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V12]], i64 30)
// CHECK: %[[V14:.*]] = ptrtoint i8** %[[V7]] to i64
// CHECK: %[[V15:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V14]], i64 30)
// CHECK: %[[V17:.*]] = ptrtoint i8* %[[V11]] to i64
// CHECK: %[[V18:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V17]], i32 2, i64 %[[V13]], i32 2, i64 %[[V15]])

void test_copy_constructor_SA2(SA2 *s) {
  SA2 t = *s;
}

// CHECK: define void @test_copy_assignment_SA(
// CHECK: call void @__copy_assignment_8_8_t0w4_pa1_50_8(

// CHECK: define linkonce_odr hidden void @__copy_assignment_8_8_t0w4_pa1_50_8(

void test_copy_assignment_SA(SA *d, SA *s) {
  *d = *s;
}

// CHECK: define void @test_move_constructor_SA(
// CHECK: define internal void @__Block_byref_object_copy_(
// CHECK: define linkonce_odr hidden void @__move_constructor_8_8_t0w4_pa1_50_8(

void test_move_constructor_SA(void) {
  __block SA t;
  BlockTy b = ^{ (void)t; };
}

// CHECK: define void @test_move_assignment_SA(
// CHECK: call void @__move_assignment_8_8_t0w4_pa1_50_8(
// CHECK: define linkonce_odr hidden void @__move_assignment_8_8_t0w4_pa1_50_8(

void test_move_assignment_SA(SA *p) {
  *p = getSA();
}

// CHECK: define void @test_parameter_SA(%[[STRUCT_SA]]* noundef %{{.*}})
// CHECK-NOT: call
// CHECK: ret void

void test_parameter_SA(SA a) {
}

// CHECK: define void @test_argument_SA(%[[STRUCT_SA]]* noundef %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_SA]], align 8
// CHECK: store %[[STRUCT_SA]]* %[[A]], %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_SA]]* %[[AGG_TMP]] to i8**
// CHECK: %[[V2:.*]] = bitcast %[[STRUCT_SA]]* %[[V0]] to i8**
// CHECK: call void @__copy_constructor_8_8_t0w4_pa1_50_8(i8** %[[V1]], i8** %[[V2]]) #5
// CHECK: call void @calleeSA(%[[STRUCT_SA]]* noundef %[[AGG_TMP]])
// CHECK-NOT: call
// CHECK: ret void

void test_argument_SA(SA *a) {
  calleeSA(*a);
}

// CHECK: define void @test_return_SA(%[[STRUCT_SA]]* noalias sret(%struct.SA) align 8 %[[AGG_RESULT:.*]], %[[STRUCT_SA]]* noundef %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca %[[STRUCT_SA]]*, align 8
// CHECK: store %[[STRUCT_SA]]* %[[A]], %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load %[[STRUCT_SA]]*, %[[STRUCT_SA]]** %[[A_ADDR]], align 8
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_SA]]* %[[AGG_RESULT]] to i8**
// CHECK: %[[V2:.*]] = bitcast %[[STRUCT_SA]]* %[[V0]] to i8**
// CHECK: call void @__copy_constructor_8_8_t0w4_pa1_50_8(i8** %[[V1]], i8** %[[V2]]) #5
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

// CHECK-LABEL: define void @test_array(
// CHECK: %[[F1:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %{{.*}}, i32 0, i32 1
// CHECK: %[[V0:.*]] = ptrtoint i32** %[[F1]] to i64
// CHECK: %[[V1:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V0]], i64 50)
// CHECK: %[[V2:.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (i32* @g0 to i64), i32 1, i64 %[[V1]])
// CHECK: %[[V3:.*]] = inttoptr i64 %[[V2]] to i32*
// CHECK: store i32* %[[V3]], i32** %[[F1]], align 8
// CHECK: %[[F12:.*]] = getelementptr inbounds %[[STRUCT_SA]], %[[STRUCT_SA]]* %{{.*}}, i32 0, i32 1
// CHECK: %[[V4:.*]] = ptrtoint i32** %[[F12]] to i64
// CHECK: %[[V5:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V4]], i64 50)
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (i32* @g0 to i64), i32 1, i64 %[[V5]])
// CHECK: %[[V7:.*]] = inttoptr i64 %[[V6]] to i32*
// CHECK: store i32* %[[V7]], i32** %[[F12]], align 8

void test_array(void) {
  const SA a[] = {{0, &g0}, {1, &g0}};
}
