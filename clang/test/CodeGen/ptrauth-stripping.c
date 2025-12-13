// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s -o - | FileCheck %s 

typedef void *NonePointer;
typedef void *__ptrauth(1, 1, 101, "strip") StripPointer;
typedef void *__ptrauth(1, 1, 102, "sign-and-strip") SignAndStripPointer;
typedef void *__ptrauth(1, 1, 103, "sign-and-auth") SignAndAuthPointer;
typedef __UINT64_TYPE__ NoneIntptr;
typedef __UINT64_TYPE__ __ptrauth(1, 0, 105, "strip") StripIntptr;
typedef __UINT64_TYPE__ __ptrauth(1, 0, 106, "sign-and-strip") SignAndStripIntptr;
typedef __UINT64_TYPE__ __ptrauth(1, 0, 107, "sign-and-auth") SignAndAuthIntptr;

NonePointer globalNonePointer = "foo0";
StripPointer globalStripPointer = "foo1";
SignAndStripPointer globalSignAndStripPointer = "foo2";
SignAndAuthPointer globalSignAndAuthPointer = "foo3";
NoneIntptr globalNoneIntptr = (__UINT64_TYPE__)&globalNonePointer;
StripIntptr globalStripIntptr = (__UINT64_TYPE__)&globalStripPointer;
SignAndStripIntptr globalSignAndStripIntptr = (__UINT64_TYPE__)&globalSignAndStripPointer;
SignAndAuthIntptr globalSignAndAuthIntptr = (__UINT64_TYPE__)&globalSignAndAuthPointer;

// CHECK: @.str = private unnamed_addr constant [5 x i8] c"foo0\00", align 1
// CHECK: @globalNonePointer = global ptr @.str, align 8
// CHECK: @.str.1 = private unnamed_addr constant [5 x i8] c"foo1\00", align 1
// CHECK: @globalStripPointer = global ptr @.str.1, align 8
// CHECK: @.str.2 = private unnamed_addr constant [5 x i8] c"foo2\00", align 1
// CHECK: @globalSignAndStripPointer = global ptr ptrauth (ptr @.str.2, i32 1, i64 102, ptr @globalSignAndStripPointer), align 8
// CHECK: @.str.3 = private unnamed_addr constant [5 x i8] c"foo3\00", align 1
// CHECK: @globalSignAndAuthPointer = global ptr ptrauth (ptr @.str.3, i32 1, i64 103, ptr @globalSignAndAuthPointer), align 8
// CHECK: @globalNoneIntptr = global i64 ptrtoint (ptr @globalNonePointer to i64), align 8
// CHECK: @globalStripIntptr = global i64 ptrtoint (ptr @globalStripPointer to i64), align 8
// CHECK: @globalSignAndStripIntptr = global i64 ptrtoint (ptr ptrauth (ptr @globalSignAndStripPointer, i32 1, i64 106) to i64), align 8
// CHECK: @globalSignAndAuthIntptr = global i64 ptrtoint (ptr ptrauth (ptr @globalSignAndAuthPointer, i32 1, i64 107) to i64), align 8

typedef struct {
  NonePointer ptr;
  NoneIntptr i;
} NoneStruct;
typedef struct {
  StripPointer ptr;
  StripIntptr i;
} StripStruct;
typedef struct {
  SignAndStripPointer ptr;
  SignAndStripIntptr i;
} SignAndStripStruct;
typedef struct {
  SignAndAuthPointer ptr;
  SignAndAuthIntptr i;
} SignAndAuthStruct;

// CHECK-LABEL: @testNone
NoneStruct testNone(NoneStruct *a, NoneStruct *b, NoneStruct c) {
  globalNonePointer += 1;
  // CHECK: [[GLOBALP:%.*]] = load ptr, ptr @globalNonePointer
  // CHECK: [[GLOBALPP:%.*]] = getelementptr inbounds i8, ptr [[GLOBALP]], i64 1
  // CHECK: store ptr [[GLOBALPP]], ptr @globalNonePointer
  globalNoneIntptr += 1;
  // CHECK: [[GLOBALI:%.*]] = load i64, ptr @globalNoneIntptr
  // CHECK: [[GLOBALIP:%.*]] = add i64 [[GLOBALI]], 1
  // CHECK: store i64 [[GLOBALIP]], ptr @globalNoneIntptr
  a->ptr += 1;
  // CHECK: [[PTR:%.*]] = load ptr, ptr %a.addr, align 8
  // CHECK: [[PTR_PTR:%.*]] = getelementptr inbounds nuw %struct.NoneStruct, ptr [[PTR]], i32 0, i32 0
  // CHECK: [[PTR:%.*]] = load ptr, ptr [[PTR_PTR]], align 8
  // CHECK: [[AP:%.*]] = getelementptr inbounds i8, ptr [[PTR]], i64 1
  // CHECK: store ptr [[AP]], ptr [[PTR_PTR]], align 8
  a->i += 1;
  // CHECK: [[PTR:%.*]] = load ptr, ptr %a.addr, align 8
  // CHECK: [[I_PTR:%.*]] = getelementptr inbounds nuw %struct.NoneStruct, ptr [[PTR]], i32 0, i32 1
  // CHECK: [[I:%.*]] = load i64, ptr [[I_PTR]], align 8
  // CHECK: [[IP:%.*]] = add i64 [[I]], 1
  // CHECK: store i64 [[IP]], ptr [[I_PTR]], align 8
  *b = *a;
  // CHECK: [[B_ADDR:%.*]] = load ptr, ptr %b.addr, align 8
  // CHECK: [[A_ADDR:%.*]] = load ptr, ptr %a.addr, align 8
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[B_ADDR]], ptr align 8 [[A_ADDR]], i64 16, i1 false)
  return c;
}

// CHECK-LABEL: @testStrip1
void testStrip1() {
  globalStripPointer += 1;
  // CHECK: [[GLOBALP:%.*]] = load ptr, ptr @globalStripPointer
  // CHECK: [[GLOBALPI:%.*]] = ptrtoint ptr [[GLOBALP]] to i64
  // CHECK: {{%.*}} = call i64 @llvm.ptrauth.strip(i64 [[GLOBALPI]], i32 1)
}
// CHECK-LABEL: @testStrip2
void testStrip2(StripStruct *a) {
  a->ptr += 1;
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: [[PTR:%.*]] = getelementptr inbounds nuw %struct.StripStruct, ptr [[A]], i32 0, i32 0
  // CHECK: [[APTR:%.*]] = load ptr, ptr [[PTR]]
  // CHECK: [[APTRI:%.*]] = ptrtoint ptr [[APTR]] to i64
  // CHECK: {{%.*}} = call i64 @llvm.ptrauth.strip(i64 [[APTRI]], i32 1)
}
// CHECK-LABEL: @testStrip3
void testStrip3(StripStruct *a) {
  a->i += 1;
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: [[I:%.*]] = getelementptr inbounds nuw %struct.StripStruct, ptr [[A]], i32 0, i32 1
  // CHECK: [[I64:%.*]] = load i64, ptr [[I]]
  // CHECK: {{%.*}} = call i64 @llvm.ptrauth.strip(i64 [[I64]], i32 1)
}
// CHECK-LABEL: @testStrip4
void testStrip4(StripStruct *a, StripStruct *b) {
  *b = *a;
  // CHECK: call void @__copy_assignment_8_8_pa1_101_0_t8w8(ptr %0, ptr %1)
}

// CHECK-LABEL: @testStrip5
StripStruct testStrip5(StripStruct a) {
  return a;
  // CHECK: call void @__copy_constructor_8_8_pa1_101_0_t8w8(ptr %agg.result, ptr %a)
}

// CHECK-LABEL: @testSignAndStrip1
void testSignAndStrip1(void) {
  globalSignAndStripPointer += 1;
  // CHECK: [[GP:%.*]] = load ptr, ptr @globalSignAndStripPointer
  // CHECK: [[GPI:%.*]] = ptrtoint ptr [[GP]] to i64
  // CHECK: [[STRIPPED:%.*]] = call i64 @llvm.ptrauth.strip(i64 [[GPI]], i32 1)
  // CHECK: [[STRIPPEDP:%.*]] = inttoptr i64 [[STRIPPED]] to ptr
  // CHECK: [[PHI:%.*]] = phi ptr [ null, %entry ], [ [[STRIPPEDP]], %resign.nonnull ]
  // CHECK: [[ADDPTR:%.*]] = getelementptr inbounds i8, ptr [[PHI]], i64 1
  // CHECK: [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @globalSignAndStripPointer to i64), i64 102)
  // CHECK: [[ADDPTRI:%.*]] = ptrtoint ptr [[ADDPTR]] to i64
  // CHECK: {{%.*}} = call i64 @llvm.ptrauth.sign(i64 [[ADDPTRI]], i32 1, i64 [[DISC]])
}

// CHECK-LABEL: @testSignAndStrip2
void testSignAndStrip2(SignAndStripStruct *a) {
  a->ptr += 1;
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: %ptr = getelementptr inbounds nuw %struct.SignAndStripStruct, ptr [[A]], i32 0, i32 0
  // CHECK: [[APTR:%.*]] = load ptr, ptr %ptr
  // CHECK: [[APTRI:%.*]] = ptrtoint ptr [[APTR]] to i64
  // CHECK: [[STRIPPED:%.*]] = call i64 @llvm.ptrauth.strip(i64 [[APTRI]], i32 1)
  // CHECK: [[STRIPPEDP:%.*]] = inttoptr i64 [[STRIPPED]] to ptr
  // CHECK: [[PHI:%.*]] = phi ptr [ null, %entry ], [ [[STRIPPEDP]], %resign.nonnull ]
  // CHECK: %add.ptr = getelementptr inbounds i8, ptr [[PHI]], i64 1
  // CHECK: [[PTRI:%.*]] = ptrtoint ptr %ptr to i64
  // CHECK: [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[PTRI]], i64 102)
  // CHECK: [[APTRI:%.*]] = ptrtoint ptr %add.ptr to i64
  // CHECK: call i64 @llvm.ptrauth.sign(i64 [[APTRI]], i32 1, i64 [[DISC]])
}

// CHECK-LABEL: @testSignAndStrip3
void testSignAndStrip3(SignAndStripStruct *a) {
  a->i += 1;
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: [[I:%.*]] = getelementptr inbounds nuw %struct.SignAndStripStruct, ptr [[A]], i32 0, i32 1
  // CHECK: [[I64:%.*]] = load i64, ptr [[I]]
  // CHECK: [[STRIPPED:%.*]] = call i64 @llvm.ptrauth.strip(i64 [[I64]], i32 1)
  // CHECK: [[PHI:%.*]] = phi i64 [ 0, %entry ], [ [[STRIPPED]], %resign.nonnull ]
  // CHECK: %add = add i64 [[PHI]], 1
  // CHECK: {{%.*}} = call i64 @llvm.ptrauth.sign(i64 %add, i32 1, i64 106)
}

// CHECK-LABEL: @testSignAndStrip4
void testSignAndStrip4(SignAndStripStruct *a, SignAndStripStruct *b) {
  *b = *a;
  // CHECK: call void @__copy_assignment_8_8_pa1_102_0_t8w8(ptr %0, ptr %1)
}

// CHECK-LABEL: @testSignAndStrip5
SignAndStripStruct testSignAndStrip5(SignAndStripStruct a) {
  return a;
  // CHECK: call void @__copy_constructor_8_8_pa1_102_0_t8w8(ptr %agg.result, ptr %a)
}

// CHECK-LABEL: @testSignAndAuth1
void testSignAndAuth1() {
  globalSignAndAuthPointer += 1;
  // CHECK: %0 = load ptr, ptr @globalSignAndAuthPointer
  // CHECK: %1 = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @globalSignAndAuthPointer to i64), i64 103)
  // CHECK: %3 = ptrtoint ptr %0 to i64
  // CHECK: %4 = call i64 @llvm.ptrauth.auth(i64 %3, i32 1, i64 %1)
  // CHECK: %5 = inttoptr i64 %4 to ptr
  // CHECK: %6 = phi ptr [ null, %entry ], [ %5, %resign.nonnull ]
  // CHECK: %add.ptr = getelementptr inbounds i8, ptr %6, i64 1
  // CHECK: %7 = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @globalSignAndAuthPointer to i64), i64 103)
  // CHECK: %8 = ptrtoint ptr %add.ptr to i64
  // CHECK: %9 = call i64 @llvm.ptrauth.sign(i64 %8, i32 1, i64 %7)
}

// CHECK-LABEL: @testSignAndAuth2
void testSignAndAuth2(SignAndAuthStruct *a) {
  a->i += 1;
  // CHECK: %0 = load ptr, ptr %a.addr
  // CHECK: %i = getelementptr inbounds nuw %struct.SignAndAuthStruct, ptr %0, i32 0, i32 1
  // CHECK: %1 = load i64, ptr %i
  // CHECK: %3 = call i64 @llvm.ptrauth.auth(i64 %1, i32 1, i64 107)
  // CHECK: %4 = phi i64 [ 0, %entry ], [ %3, %resign.nonnull ]
  // CHECK: %add = add i64 %4, 1
  // CHECK: %6 = call i64 @llvm.ptrauth.sign(i64 %add, i32 1, i64 107)
  // CHECK: %7 = phi i64 [ 0, %resign.cont ], [ %6, %resign.nonnull1 ]
}

// CHECK-LABEL: @testSignAndAuth3
void testSignAndAuth3(SignAndAuthStruct *a) {
  a->ptr += 1;
  // CHECK: %0 = load ptr, ptr %a.addr
  // CHECK: %ptr = getelementptr inbounds nuw %struct.SignAndAuthStruct, ptr %0, i32 0, i32 0
  // CHECK: %1 = load ptr, ptr %ptr
  // CHECK: %2 = ptrtoint ptr %ptr to i64
  // CHECK: %3 = call i64 @llvm.ptrauth.blend(i64 %2, i64 103)
  // CHECK: %5 = ptrtoint ptr %1 to i64
  // CHECK: %6 = call i64 @llvm.ptrauth.auth(i64 %5, i32 1, i64 %3)
  // CHECK: %7 = inttoptr i64 %6 to ptr
  // CHECK: %8 = phi ptr [ null, %entry ], [ %7, %resign.nonnull ]
  // CHECK: %add.ptr = getelementptr inbounds i8, ptr %8, i64 1
  // CHECK: %9 = ptrtoint ptr %ptr to i64
  // CHECK: %10 = call i64 @llvm.ptrauth.blend(i64 %9, i64 103)
  // CHECK: %11 = ptrtoint ptr %add.ptr to i64
  // CHECK: %12 = call i64 @llvm.ptrauth.sign(i64 %11, i32 1, i64 %10)
}

// CHECK-LABEL: @testSignAndAuth4
void testSignAndAuth4(SignAndAuthStruct *a, SignAndAuthStruct *b) {
  *b = *a;
  // CHECK: call void @__copy_assignment_8_8_pa1_103_0_t8w8(ptr %0, ptr %1)
}

// CHECK-LABEL: @testSignAndAuth5
SignAndAuthStruct testSignAndAuth5(SignAndAuthStruct a) {
  return a;
  // CHECK: call void @__copy_constructor_8_8_pa1_103_0_t8w8(ptr %agg.result, ptr %a)
}

// CHECK-LABEL: @testCoercions1
void testCoercions1(StripStruct *a, SignAndStripStruct *b) {
  a->ptr = b->ptr;
  // CHECK: %0 = load ptr, ptr %a.addr
  // CHECK: %ptr = getelementptr inbounds nuw %struct.StripStruct, ptr %0, i32 0, i32 0
  // CHECK: %1 = load ptr, ptr %b.addr
  // CHECK: %ptr1 = getelementptr inbounds nuw %struct.SignAndStripStruct, ptr %1, i32 0, i32 0
  // CHECK: %2 = load ptr, ptr %ptr1
  // CHECK: %3 = ptrtoint ptr %ptr1 to i64
  // CHECK: %4 = call i64 @llvm.ptrauth.blend(i64 %3, i64 102)
  // CHECK: %8 = ptrtoint ptr %2 to i64
  // CHECK: %9 = call i64 @llvm.ptrauth.strip(i64 %8, i32 1)
}

// CHECK-LABEL: @testCoercions2
void testCoercions2(StripStruct *a, SignAndAuthStruct *b) {
  b->ptr = a->ptr;
  // CHECK: store ptr %a, ptr %a.addr
  // CHECK: store ptr %b, ptr %b.addr
  // CHECK: %0 = load ptr, ptr %b.addr
  // CHECK: %ptr = getelementptr inbounds nuw %struct.SignAndAuthStruct, ptr %0, i32 0, i32 0
  // CHECK: %1 = load ptr, ptr %a.addr
  // CHECK: %ptr1 = getelementptr inbounds nuw %struct.StripStruct, ptr %1, i32 0, i32 0
  // CHECK: %2 = load ptr, ptr %ptr1
  // CHECK: %3 = ptrtoint ptr %ptr1 to i64
  // CHECK: %4 = call i64 @llvm.ptrauth.blend(i64 %3, i64 101)
  // CHECK: %5 = ptrtoint ptr %ptr to i64
  // CHECK: %6 = call i64 @llvm.ptrauth.blend(i64 %5, i64 103)
  // CHECK: %7 = icmp ne ptr %2, null
  // CHECK: %8 = ptrtoint ptr %2 to i64
  // CHECK: %9 = call i64 @llvm.ptrauth.strip(i64 %8, i32 1)
  // CHECK: %10 = inttoptr i64 %9 to ptr
  // CHECK: %11 = ptrtoint ptr %10 to i64
  // CHECK: %12 = call i64 @llvm.ptrauth.sign(i64 %11, i32 1, i64 %6)
}

// CHECK-LABEL: @testCoercions3
void testCoercions3(SignAndStripStruct *a, SignAndAuthStruct *b) {
  a->ptr = b->ptr;
  // CHECK: %0 = load ptr, ptr %a.addr
  // CHECK: %ptr = getelementptr inbounds nuw %struct.SignAndStripStruct, ptr %0, i32 0, i32 0
  // CHECK: %1 = load ptr, ptr %b.addr
  // CHECK: %ptr1 = getelementptr inbounds nuw %struct.SignAndAuthStruct, ptr %1, i32 0, i32 0
  // CHECK: %2 = load ptr, ptr %ptr1
  // CHECK: %3 = ptrtoint ptr %ptr1 to i64
  // CHECK: %4 = call i64 @llvm.ptrauth.blend(i64 %3, i64 103)
  // CHECK: %5 = ptrtoint ptr %ptr to i64
  // CHECK: %6 = call i64 @llvm.ptrauth.blend(i64 %5, i64 102)
  // CHECK: %8 = ptrtoint ptr %2 to i64
  // CHECK: %9 = call i64 @llvm.ptrauth.auth(i64 %8, i32 1, i64 %4)
  // CHECK: %10 = inttoptr i64 %9 to ptr
  // CHECK: %11 = ptrtoint ptr %10 to i64
  // CHECK: %12 = call i64 @llvm.ptrauth.sign(i64 %11, i32 1, i64 %6)
  // CHECK: %13 = inttoptr i64 %12 to ptr
  // CHECK: %14 = phi ptr [ null, %entry ], [ %13, %resign.nonnull ]
}

// CHECK-LABEL: @testCoercions4
void testCoercions4(StripStruct *a, SignAndStripStruct *b) {
  a->i = b->i;
  // CHECK: store ptr %a, ptr %a.addr
  // CHECK: store ptr %b, ptr %b.addr
  // CHECK: %0 = load ptr, ptr %a.addr
  // CHECK: %i = getelementptr inbounds nuw %struct.StripStruct, ptr %0, i32 0, i32 1
  // CHECK: %1 = load ptr, ptr %b.addr
  // CHECK: %i1 = getelementptr inbounds nuw %struct.SignAndStripStruct, ptr %1, i32 0, i32 1
  // CHECK: %2 = load i64, ptr %i1
  // CHECK: %4 = call i64 @llvm.ptrauth.strip(i64 %2, i32 1)
  // CHECK: %5 = phi i64 [ 0, %entry ], [ %4, %resign.nonnull ]
}

// CHECK-LABEL: @testCoercions5
void testCoercions5(StripStruct *a, SignAndAuthStruct *b) {
  b->i = a->i;
  // CHECK: %0 = load ptr, ptr %b.addr
  // CHECK: %i = getelementptr inbounds nuw %struct.SignAndAuthStruct, ptr %0, i32 0, i32 1
  // CHECK: %1 = load ptr, ptr %a.addr
  // CHECK: %i1 = getelementptr inbounds nuw %struct.StripStruct, ptr %1, i32 0, i32 1
  // CHECK: %2 = load i64, ptr %i1
  // CHECK: %4 = call i64 @llvm.ptrauth.strip(i64 %2, i32 1)
  // CHECK: %5 = call i64 @llvm.ptrauth.sign(i64 %4, i32 1, i64 107)
  // CHECK: %6 = phi i64 [ 0, %entry ], [ %5, %resign.nonnull ]
  // CHECK: store i64 %6, ptr %i
}

// CHECK-LABEL: @testCoercions6
void testCoercions6(SignAndStripStruct *a, SignAndAuthStruct *b) {
  a->i = b->i;
  // CHECK: %0 = load ptr, ptr %a.addr
  // CHECK: %i = getelementptr inbounds nuw %struct.SignAndStripStruct, ptr %0, i32 0, i32 1
  // CHECK: %1 = load ptr, ptr %b.addr
  // CHECK: %i1 = getelementptr inbounds nuw %struct.SignAndAuthStruct, ptr %1, i32 0, i32 1
  // CHECK: %2 = load i64, ptr %i1
  // CHECK: %3 = icmp ne i64 %2, 0
  // CHECK: %4 = call i64 @llvm.ptrauth.auth(i64 %2, i32 1, i64 107)
  // CHECK: %5 = call i64 @llvm.ptrauth.sign(i64 %4, i32 1, i64 106)
  // CHECK: %6 = phi i64 [ 0, %entry ], [ %5, %resign.nonnull ]
}
