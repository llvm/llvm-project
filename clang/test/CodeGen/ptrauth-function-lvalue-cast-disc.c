// RUN: %clang_cc1 %s -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -emit-llvm -o-  -fptrauth-function-pointer-type-discrimination | FileCheck %s

typedef void (*fptr_t)(void);

char *cptr;
void (*fptr)(void);

// CHECK-LABEL: define void @test1
void test1() {
  // CHECK: [[LOAD:%.*]] = load ptr, ptr @cptr
  // CHECK: [[TOINT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // CHECK: call i64 @llvm.ptrauth.resign(i64 [[TOINT]], i32 0, i64 0, i32 0, i64 18983)
  // CHECK: call void {{.*}}() [ "ptrauth"(i32 0, i64 18983) ]

  (*(fptr_t)cptr)();
}

// CHECK-LABEL: define i8 @test2
char test2() {
  return *(char *)fptr;

  // CHECK: [[LOAD:%.*]] = load ptr, ptr @fptr
  // CHECK: [[CMP:%.*]] = icmp ne ptr [[LOAD]], null
  // CHECK-NEXT: br i1 [[CMP]], label %[[NONNULL:.*]], label %[[CONT:.*]]

  // CHECK: [[NONNULL]]:
  // CHECK: [[TOINT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // CHECK: [[CALL:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[TOINT]], i32 0, i64 18983, i32 0, i64 0)
  // CHECK: [[TOPTR:%.*]] = inttoptr i64 [[CALL]] to ptr

  // CHECK: [[CONT]]:
  // CHECK: phi ptr [ null, {{.*}} ], [ [[TOPTR]], %[[NONNULL]] ]
}

// CHECK-LABEL: define void @test4
void test4() {
  (*((fptr_t)(&*((char *)(&*(fptr_t)cptr)))))();

  // CHECK: [[LOAD:%.*]] = load ptr, ptr @cptr
  // CHECK-NEXT: [[CAST4:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // CHECK-NEXT: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[CAST4]], i32 0, i64 0, i32 0, i64 18983)
  // CHECK-NEXT: [[CAST5:%.*]] = inttoptr i64 [[RESIGN]] to ptr
  // CHECK-NEXT: call void [[CAST5]]() [ "ptrauth"(i32 0, i64 18983) ]
}

void *vptr;
// CHECK-LABEL: define void @test5
void test5() {
  vptr = &*(char *)fptr;

  // CHECK: [[LOAD:%.*]] = load ptr, ptr @fptr
  // CHECK-NEXT: [[CMP]] = icmp ne ptr [[LOAD]], null
  // CHECK-NEXT: br i1 [[CMP]], label %[[NONNULL:.*]], label %[[CONT:.*]]

  // CHECK: [[NONNULL]]:
  // CHECK: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 {{.*}}, i32 0, i64 18983, i32 0, i64 0)
  // CHECK: [[CAST:%.*]] = inttoptr i64 [[RESIGN]] to ptr

  // CHECK: [[CONT]]:
  // CHECK: [[PHI:%.*]] = phi ptr [ null, {{.*}} ], [ [[CAST]], %[[NONNULL]] ]
  // CHECK: store ptr [[PHI]], ptr @vptr
}
