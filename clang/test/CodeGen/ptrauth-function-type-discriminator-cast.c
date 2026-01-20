// RUN: %clang_cc1 %s -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -o-       | FileCheck %s --check-prefixes=CHECK,TYPE

// RUN: %clang_cc1 %s -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -o-       | FileCheck %s --check-prefixes=CHECK,TYPE

// RUN: %clang_cc1 %s                                                -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -o-       | FileCheck %s --check-prefixes=CHECK,ZERO

// RUN: %clang_cc1 %s                                                -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -o-       | FileCheck %s --check-prefixes=CHECK,ZERO

// RUN: %clang_cc1 %s -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -xc++ -o- | FileCheck %s --check-prefixes=CHECK,CHECKCXX,TYPE,TYPECXX

// RUN: %clang_cc1 %s -fptrauth-function-pointer-type-discrimination -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics \
// RUN:   -disable-llvm-passes -emit-llvm -xc++ -o- | FileCheck %s --check-prefixes=CHECK,CHECKCXX,TYPE,TYPECXX

#ifdef __cplusplus
extern "C" {
#endif

void f(void);
void f2(int);
void (*fptr)(void);
void *opaque;
unsigned long uintptr;

#ifdef __cplusplus
struct ptr_member {
  void (*fptr_)(int) = 0;
};
ptr_member pm;
void (*test_member)() = (void (*)())pm.fptr_;

// CHECKCXX-LABEL: define{{.*}} internal void @__cxx_global_var_init
// TYPECXX: call i64 @llvm.ptrauth.resign(i64 {{.*}}, i32 0, i64 2712, i32 0, i64 18983)
#endif


// CHECK-LABEL: define{{.*}} void @test_cast_to_opaque
void test_cast_to_opaque() {
  opaque = (void *)f;

  // TYPE: [[RESIGN_VAL:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 0)
  // TYPE: [[RESIGN_PTR:%.*]] = inttoptr i64 [[RESIGN_VAL]] to ptr
  // ZERO-NOT: @llvm.ptrauth.resign
}

// CHECK-LABEL: define{{.*}} void @test_cast_from_opaque
void test_cast_from_opaque() {
  fptr = (void (*)(void))opaque;

  // TYPE: [[LOAD:%.*]] = load ptr, ptr @opaque
  // TYPE: [[CMP:%.*]] = icmp ne ptr [[LOAD]], null
  // TYPE: br i1 [[CMP]], label %[[RESIGN_LAB:.*]], label

  // TYPE: [[RESIGN_LAB]]:
  // TYPE: [[INT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // TYPE: [[RESIGN_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT]], i32 0, i64 0, i32 0, i64 18983)

  // ZERO-NOT: @llvm.ptrauth.resign
}

// CHECK-LABEL: define{{.*}} void @test_cast_to_intptr
void test_cast_to_intptr() {
  uintptr = (unsigned long)fptr;

  // TYPE: [[ENTRY:.*]]:
  // TYPE: [[LOAD:%.*]] = load ptr, ptr @fptr
  // TYPE: [[CMP:%.*]] = icmp ne ptr [[LOAD]], null
  // TYPE: br i1 [[CMP]], label %[[RESIGN_LAB:.*]], label %[[RESIGN_CONT:.*]]

  // TYPE: [[RESIGN_LAB]]:
  // TYPE: [[INT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // TYPE: [[RESIGN_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT]], i32 0, i64 18983, i32 0, i64 0)
  // TYPE: [[RESIGN:%.*]] = inttoptr i64 [[RESIGN_INT]] to ptr
  // TYPE: br label %[[RESIGN_CONT]]

  // TYPE: [[RESIGN_CONT]]:
  // TYPE: phi ptr [ null, %[[ENTRY]] ], [ [[RESIGN]], %[[RESIGN_LAB]] ]

  // ZERO-NOT: @llvm.ptrauth.resign
}

// CHECK-LABEL: define{{.*}} void @test_function_to_function_cast
void test_function_to_function_cast() {
  void (*fptr2)(int) = (void (*)(int))fptr;
  // TYPE: call i64 @llvm.ptrauth.resign(i64 {{.*}}, i32 0, i64 18983, i32 0, i64 2712)
  // ZERO-NOT: @llvm.ptrauth.resign
}

// CHECK-LABEL: define{{.*}} void @test_call_lvalue_cast
void test_call_lvalue_cast() {
  (*(void (*)(int))f)(42);

  // TYPE: entry:
  // TYPE-NEXT: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 2712)
  // TYPE-NEXT: [[RESIGN_INT:%.*]] = inttoptr i64 [[RESIGN]] to ptr
  // TYPE-NEXT: call void [[RESIGN_INT]](i32 noundef 42) [ "ptrauth"(i32 0, i64 2712) ]
  // ZERO-NOT: @llvm.ptrauth.resign
  // ZERO: call void ptrauth (ptr @f, i32 0)(i32 noundef 42) [ "ptrauth"(i32 0, i64 0) ]
}


#ifdef __cplusplus
}
#endif
