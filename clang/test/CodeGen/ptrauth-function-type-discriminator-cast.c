// RUN: %clang_cc1 %s       -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s
// RUN: %clang_cc1 -xc++ %s -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s --check-prefixes=CHECK,CHECKCXX

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

// CHECKCXX-LABEL: define internal void @__cxx_global_var_init
// CHECKCXX: call i64 @llvm.ptrauth.resign(i64 {{.*}}, i32 0, i64 2712, i32 0, i64 18983)
#endif


// CHECK-LABEL: define void @test_cast_to_opaque
void test_cast_to_opaque() {
  opaque = (void *)f;

  // CHECK: [[RESIGN_VAL:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 0)
  // CHECK: [[RESIGN_PTR:%.*]] = inttoptr i64 [[RESIGN_VAL]] to ptr
}

// CHECK-LABEL: define void @test_cast_from_opaque
void test_cast_from_opaque() {
  fptr = (void (*)(void))opaque;

  // CHECK: [[LOAD:%.*]] = load ptr, ptr @opaque
  // CHECK: [[CMP:%.*]] = icmp ne ptr [[LOAD]], null
  // CHECK: br i1 [[CMP]], label %[[RESIGN_LAB:.*]], label

  // CHECK: [[RESIGN_LAB]]:
  // CHECK: [[INT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // CHECK: [[RESIGN_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT]], i32 0, i64 0, i32 0, i64 18983)
}

// CHECK-LABEL: define void @test_cast_to_intptr
void test_cast_to_intptr() {
  uintptr = (unsigned long)fptr;

  // CHECK: [[ENTRY:.*]]:
  // CHECK: [[LOAD:%.*]] = load ptr, ptr @fptr
  // CHECK: [[CMP:%.*]] = icmp ne ptr [[LOAD]], null
  // CHECK: br i1 [[CMP]], label %[[RESIGN_LAB:.*]], label %[[RESIGN_CONT:.*]]

  // CHECK: [[RESIGN_LAB]]:
  // CHECK: [[INT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // CHECK: [[RESIGN_INT:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT]], i32 0, i64 18983, i32 0, i64 0)
  // CHECK: [[RESIGN:%.*]] = inttoptr i64 [[RESIGN_INT]] to ptr
  // CHECK: br label %[[RESIGN_CONT]]

  // CHECK: [[RESIGN_CONT]]:
  // CHECK: phi ptr [ null, %[[ENTRY]] ], [ [[RESIGN]], %[[RESIGN_LAB]] ]
}

// CHECK-LABEL: define void @test_function_to_function_cast
void test_function_to_function_cast() {
  void (*fptr2)(int) = (void (*)(int))fptr;
  // CHECK: call i64 @llvm.ptrauth.resign(i64 {{.*}}, i32 0, i64 18983, i32 0, i64 2712)
}

// CHECK-LABEL: define void @test_call_lvalue_cast
void test_call_lvalue_cast() {
  (*(void (*)(int))f)(42);

  // CHECK: entry:
  // CHECK-NEXT: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 2712)
  // CHECK-NEXT: [[RESIGN_INT:%.*]] = inttoptr i64 [[RESIGN]] to ptr
  // CHECK-NEXT: call void [[RESIGN_INT]](i32 noundef 42) [ "ptrauth"(i32 0, i64 2712) ]
}


#ifdef __cplusplus
}
#endif
