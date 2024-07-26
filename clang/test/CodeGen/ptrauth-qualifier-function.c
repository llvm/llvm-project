// RUN: %clang_cc1 %s       -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s
// RUN: %clang_cc1 -xc++ %s -fptrauth-function-pointer-type-discrimination -triple arm64e-apple-ios13 -fptrauth-calls -fptrauth-intrinsics -disable-llvm-passes -emit-llvm -o- | FileCheck %s

#ifdef __cplusplus
extern "C" {
#endif

void (*fptr)(void);
void (* __ptrauth(0, 0, 42) f2ptr_42_discm)(int);

// CHECK-LABEL: define void @test_assign_to_qualified
void test_assign_to_qualified() {
  f2ptr_42_discm = (void (*)(int))fptr;

  // CHECK: [[ENTRY:.*]]:{{$}}
  // CHECK: [[FPTR:%.*]] = load ptr, ptr @fptr
  // CHECK-NEXT: [[CMP:%.*]] = icmp ne ptr [[FPTR]], null
  // CHECK-NEXT: br i1 [[CMP]], label %[[RESIGN1:.*]], label %[[JOIN1:.*]]

  // CHECK: [[RESIGN1]]:
  // CHECK-NEXT: [[FPTR2:%.*]] = ptrtoint ptr [[FPTR]] to i64
  // CHECK-NEXT: [[FPTR4:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[FPTR2]], i32 0, i64 18983, i32 0, i64 2712)
  // CHECK-NEXT: [[FPTR5:%.*]] = inttoptr i64 [[FPTR4]] to ptr
  // CHECK-NEXT: br label %[[JOIN1]]

  // CHECK: [[JOIN1]]:
  // CHECK-NEXT: [[FPTR6:%.*]] = phi ptr [ null, %[[ENTRY]] ], [ [[FPTR5]], %[[RESIGN1]] ]
  // CHECK-NEXT: [[CMP:%.*]] = icmp ne ptr [[FPTR6]], null
  // CHECK-NEXT: br i1 [[CMP]], label %[[RESIGN2:.*]], label %[[JOIN2:.*]]

  // CHECK: [[RESIGN2]]:
  // CHECK-NEXT: [[FPTR7:%.*]] = ptrtoint ptr [[FPTR6]] to i64
  // CHECK-NEXT: [[FPTR8:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[FPTR7]], i32 0, i64 2712, i32 0, i64 42)
  // CHECK-NEXT: [[FPTR9:%.*]] = inttoptr i64 [[FPTR8]] to ptr
  // CHECK-NEXT: br label %[[JOIN2]]

  // CHECK: [[JOIN2]]
  // CHECK-NEXT: [[FPTR10:%.*]] = phi ptr [ null, %[[JOIN1]] ], [ [[FPTR9]], %[[RESIGN2]] ]
  // CHECK-NEXT store void (i32)* [[FPTR10]], void (i32)** @f2ptr_42_discm
}

// CHECK-LABEL: define void @test_assign_from_qualified
void test_assign_from_qualified() {
  fptr = (void (*)(void))f2ptr_42_discm;

  // CHECK: [[ENTRY:.*]]:{{$}}
  // CHECK: [[FPTR:%.*]] = load ptr, ptr @f2ptr_42_discm
  // CHECK-NEXT: [[CMP:%.*]] = icmp ne ptr [[FPTR]], null
  // CHECK-NEXT: br i1 [[CMP]], label %[[RESIGN1:.*]], label %[[JOIN1:.*]]

  // CHECK: [[RESIGN1]]:
  // CHECK-NEXT: [[FPTR1:%.*]] = ptrtoint ptr [[FPTR]] to i64
  // CHECK-NEXT: [[FPTR2:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[FPTR1]], i32 0, i64 42, i32 0, i64 2712)
  // CHECK-NEXT: [[FPTR3:%.*]] = inttoptr i64 [[FPTR2]] to ptr
  // CHECK-NEXT: br label %[[JOIN1]]

  // CHECK: [[JOIN1]]:
  // CHECK-NEXT: [[FPTR4:%.*]] = phi ptr [ null, %[[ENTRY]] ], [ [[FPTR3]], %[[RESIGN1]] ]
  // CHECK-NEXT: [[CMP:%.*]] = icmp ne ptr [[FPTR4]], null
  // CHECK-NEXT: br i1 [[CMP]], label %[[RESIGN2:.*]], label %[[JOIN2:.*]]

  // CHECK: [[RESIGN2]]:
  // CHECK-NEXT: [[FPTR6:%.*]] = ptrtoint ptr [[FPTR4]] to i64
  // CHECK-NEXT: [[FPTR7:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[FPTR6]], i32 0, i64 2712, i32 0, i64 18983)
  // CHECK-NEXT: [[FPTR8:%.*]] = inttoptr i64 [[FPTR7]] to ptr
  // CHECK-NEXT: br label %[[JOIN2]]

  // CHECK: [[JOIN2]]
  // CHECK-NEXT: [[FPTR9:%.*]] = phi ptr [ null, %[[JOIN1]] ], [ [[FPTR8]], %[[RESIGN2]] ]
  // CHECK-NEXT store void ()* [[FPTR10]], void ()** @f2ptr_42_discm
}

#ifdef __cplusplus
}
#endif
