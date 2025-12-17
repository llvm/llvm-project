// RUN: %clang_cc1 -triple arm64-apple-ios -fblocks -fptrauth-calls -fptrauth-returns -fptrauth-intrinsics -emit-llvm -o - %s | FileCheck %s

#define AQ1_50 __ptrauth(1,1,50)
#define AQ2_30 __ptrauth(2,1,30)
#define IQ __ptrauth(1,0,50)

typedef void (^BlockTy)(void);

// CHECK: %[[STRUCT_SA:.*]] = type { i32, ptr }
// CHECK: %[[STRUCT_SA2:.*]] = type { i32, ptr }
// CHECK: %[[STRUCT_SI:.*]] = type { ptr }

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

typedef struct {
  // Transitively includes an address discriminated value
  SA nested;
} Nested_AddrDiscrimination;

typedef struct {
  // Transitively includes a pointer to a struct containing
  // an address discriminated value, which means that this
  // does not actually contain an address discriminated value
  SA *nestedPtr;
} Nested_PtrAddrDiscrimination;

SA getSA(void);
void calleeSA(SA);

int g0;

// CHECK: define void @test_copy_constructor_SA(ptr noundef %{{.*}})
// CHECK: call void @__copy_constructor_8_8_t0w4_pa1_50_8(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_pa1_50_8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: %[[V9:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// CHECK: %[[V11:.*]] = load ptr, ptr %[[V9]], align 8
// CHECK: %[[V12:.*]] = ptrtoint ptr %[[V9]] to i64
// CHECK: %[[V13:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V12]], i64 50)
// CHECK: %[[V14:.*]] = ptrtoint ptr %[[V6]] to i64
// CHECK: %[[V15:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V14]], i64 50)
// CHECK: %[[V17:.*]] = ptrtoint ptr %[[V11]] to i64
// CHECK: %[[V18:.*]] = call i64 @llvm.ptrauth.resign(i64 %[[V17]], i32 1, i64 %[[V13]], i32 1, i64 %[[V15]])

void test_copy_constructor_SA(SA *s) {
  SA t = *s;
}

// CHECK: define void @test_copy_constructor_SA2(ptr noundef %{{.*}})
// CHECK: call void @__copy_constructor_8_8_t0w4_pa2_30_8(

// CHECK: define linkonce_odr hidden void @__copy_constructor_8_8_t0w4_pa2_30_8(ptr noundef %[[DST:.*]], ptr noundef %[[SRC:.*]])
// CHECK: %[[DST_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[SRC_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[DST]], ptr %[[DST_ADDR]], align 8
// CHECK: store ptr %[[SRC]], ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[DST_ADDR]], align 8
// CHECK: %[[V1:.*]] = load ptr, ptr %[[SRC_ADDR]], align 8
// CHECK: %[[V6:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 8
// CHECK: %[[V9:.*]] = getelementptr inbounds i8, ptr %[[V1]], i64 8
// CHECK: %[[V11:.*]] = load ptr, ptr %[[V9]], align 8
// CHECK: %[[V12:.*]] = ptrtoint ptr %[[V9]] to i64
// CHECK: %[[V13:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V12]], i64 30)
// CHECK: %[[V14:.*]] = ptrtoint ptr %[[V6]] to i64
// CHECK: %[[V15:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V14]], i64 30)
// CHECK: %[[V17:.*]] = ptrtoint ptr %[[V11]] to i64
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

// CHECK: define void @test_parameter_SA(ptr dead_on_return noundef %{{.*}})
// CHECK-NOT: call
// CHECK: ret void

void test_parameter_SA(SA a) {
}

// CHECK: define void @test_argument_SA(ptr noundef %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[AGG_TMP:.*]] = alloca %[[STRUCT_SA]], align 8
// CHECK: store ptr %[[A]], ptr %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: call void @__copy_constructor_8_8_t0w4_pa1_50_8(ptr %[[AGG_TMP]], ptr %[[V0]])
// CHECK: call void @calleeSA(ptr dead_on_return noundef %[[AGG_TMP]])
// CHECK-NOT: call
// CHECK: ret void

void test_argument_SA(SA *a) {
  calleeSA(*a);
}

// CHECK: define void @test_return_SA(ptr dead_on_unwind noalias writable sret(%struct.SA) align 8 %[[AGG_RESULT:.*]], ptr noundef %[[A:.*]])
// CHECK: %[[A_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[A]], ptr %[[A_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[A_ADDR]], align 8
// CHECK: call void @__copy_constructor_8_8_t0w4_pa1_50_8(ptr %[[AGG_RESULT]], ptr %[[V0]])
// CHECK-NOT: call
// CHECK: ret void

SA test_return_SA(SA *a) {
  return *a;
}

// CHECK: define void @test_copy_constructor_SI(
// CHECK-NOT: call
// CHECK: call void @llvm.memcpy.p0.p0.i64(
// CHECK-NOT: call
// CHECK: ret void

void test_copy_constructor_SI(SI *s) {
  SI t = *s;
}

// CHECK: define void @test_parameter_SI(ptr %{{.*}})
// CHECK-NOT: call
// CHECK: ret void

void test_parameter_SI(SI a) {
}

// CHECK-LABEL: define void @test_array(
// CHECK: %[[F1:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[V0:.*]] = ptrtoint ptr %[[F1]] to i64
// CHECK: %[[V1:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V0]], i64 50)
// CHECK: %[[V2:.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @g0 to i64), i32 1, i64 %[[V1]])
// CHECK: %[[V3:.*]] = inttoptr i64 %[[V2]] to ptr
// CHECK: store ptr %[[V3]], ptr %[[F1]], align 8
// CHECK: %[[F12:.*]] = getelementptr inbounds nuw %[[STRUCT_SA]], ptr %{{.*}}, i32 0, i32 1
// CHECK: %[[V4:.*]] = ptrtoint ptr %[[F12]] to i64
// CHECK: %[[V5:.*]] = call i64 @llvm.ptrauth.blend(i64 %[[V4]], i64 50)
// CHECK: %[[V6:.*]] = call i64 @llvm.ptrauth.sign(i64 ptrtoint (ptr @g0 to i64), i32 1, i64 %[[V5]])
// CHECK: %[[V7:.*]] = inttoptr i64 %[[V6]] to ptr
// CHECK: store ptr %[[V7]], ptr %[[F12]], align 8

void test_array(void) {
  const SA a[] = {{0, &g0}, {1, &g0}};
}


void test_nested_struct(Nested_AddrDiscrimination* Src) {
  Nested_AddrDiscrimination Dst = *Src;
}
// CHECK-LABEL: define void @test_nested_struct
// CHECK: [[DST:%.*]]  = alloca %struct.Nested_AddrDiscrimination
// CHECK: [[SRC_ADDR:%.*]] = load ptr, ptr %Src.addr
// CHECK: call void @__copy_constructor_8_8_S_t0w4_pa1_50_8(ptr [[DST]], ptr [[SRC_ADDR]])

// CHECK-LABEL: define linkonce_odr hidden void @__copy_constructor_8_8_S_t0w4_pa1_50_8(
// CHECK: call void @__copy_constructor_8_8_t0w4_pa1_50_8


void test_nested_struct_ptr(Nested_PtrAddrDiscrimination* Src) {
  Nested_PtrAddrDiscrimination Dst = *Src;
}
// CHECK-LABEL: define void @test_nested_struct_ptr
// CHECK: [[DST:%.*]]  = alloca %struct.Nested_PtrAddrDiscrimination
// CHECK: [[SRC_ADDR:%.*]] = load ptr, ptr %Src.addr
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[DST]], ptr align 8 [[SRC_ADDR]], i64 8, i1 false)
