// RUN: %clang_cc1 %s -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -emit-llvm -o- -fptrauth-function-pointer-type-discrimination | FileCheck -check-prefixes CHECK,TYPE %s
// RUN: %clang_cc1 %s -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics -emit-llvm -o- -fptrauth-function-pointer-type-discrimination | FileCheck -check-prefixes CHECK,TYPE %s
// RUN: %clang_cc1 %s -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -emit-llvm -o- | FileCheck -check-prefixes CHECK,ZERO %s
// RUN: %clang_cc1 %s -triple aarch64-linux-gnu  -fptrauth-calls -fptrauth-intrinsics -emit-llvm -o- | FileCheck -check-prefixes CHECK,ZERO %s

typedef void (*fptr_t)(void);

char *cptr;
void (*fptr)(void);

// CHECK-LABEL: define{{.*}} void @test1
void test1() {
  // TYPE: [[LOAD:%.*]] = load ptr, ptr @cptr
  // TYPE: [[TOINT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // TYPE: call i64 @llvm.ptrauth.resign(i64 [[TOINT]], i32 0, i64 0, i32 0, i64 18983)
  // TYPE: call void {{.*}}() [ "ptrauth"(i32 0, i64 18983) ]
  // ZERO-NOT: @llvm.ptrauth.resign

  (*(fptr_t)cptr)();
}

// CHECK-LABEL: define{{.*}} i8 @test2
char test2() {
  return *(char *)fptr;

  // TYPE: [[LOAD:%.*]] = load ptr, ptr @fptr
  // TYPE: [[CMP:%.*]] = icmp ne ptr [[LOAD]], null
  // TYPE-NEXT: br i1 [[CMP]], label %[[NONNULL:.*]], label %[[CONT:.*]]

  // TYPE: [[NONNULL]]:
  // TYPE: [[TOINT:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // TYPE: [[CALL:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[TOINT]], i32 0, i64 18983, i32 0, i64 0)
  // TYPE: [[TOPTR:%.*]] = inttoptr i64 [[CALL]] to ptr

  // TYPE: [[CONT]]:
  // TYPE: phi ptr [ null, {{.*}} ], [ [[TOPTR]], %[[NONNULL]] ]
  // ZERO-NOT: @llvm.ptrauth.resign
}

// CHECK-LABEL: define{{.*}} void @test4
void test4() {
  (*((fptr_t)(&*((char *)(&*(fptr_t)cptr)))))();

  // CHECK: [[LOAD:%.*]] = load ptr, ptr @cptr
  // TYPE-NEXT: [[CAST4:%.*]] = ptrtoint ptr [[LOAD]] to i64
  // TYPE-NEXT: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[CAST4]], i32 0, i64 0, i32 0, i64 18983)
  // TYPE-NEXT: [[CAST5:%.*]] = inttoptr i64 [[RESIGN]] to ptr
  // TYPE-NEXT: call void [[CAST5]]() [ "ptrauth"(i32 0, i64 18983) ]
  // ZERO-NOT: @llvm.ptrauth.resign
  // ZERO: call void [[LOAD]]() [ "ptrauth"(i32 0, i64 0) ]
}

void *vptr;
// CHECK-LABEL: define{{.*}} void @test5
void test5() {
  vptr = &*(char *)fptr;

  // TYPE: [[LOAD:%.*]] = load ptr, ptr @fptr
  // TYPE-NEXT: [[CMP]] = icmp ne ptr [[LOAD]], null
  // TYPE-NEXT: br i1 [[CMP]], label %[[NONNULL:.*]], label %[[CONT:.*]]

  // TYPE: [[NONNULL]]:
  // TYPE: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 {{.*}}, i32 0, i64 18983, i32 0, i64 0)
  // TYPE: [[CAST:%.*]] = inttoptr i64 [[RESIGN]] to ptr

  // TYPE: [[CONT]]:
  // TYPE: [[PHI:%.*]] = phi ptr [ null, {{.*}} ], [ [[CAST]], %[[NONNULL]] ]
  // TYPE: store ptr [[PHI]], ptr @vptr
  // ZERO-NOT: @llvm.ptrauth.resign
}
