// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-elf     -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

extern int external;

// CHECK: @ptr1 = global ptr ptrauth (ptr @external, i32 0)
void *ptr1 = __builtin_ptrauth_sign_constant(&external, 0, 0);

// CHECK: @ptr2 = global ptr ptrauth (ptr @external, i32 0, i64 1234)
void *ptr2 = __builtin_ptrauth_sign_constant(&external, 0, 1234);

// CHECK: @ptr3 = global ptr ptrauth (ptr @external, i32 2, i64 0, ptr @ptr3)
void *ptr3 = __builtin_ptrauth_sign_constant(&external, 2, &ptr3);

// CHECK: @ptr4 = global ptr ptrauth (ptr @external, i32 2, i64 26, ptr @ptr4)
void *ptr4 = __builtin_ptrauth_sign_constant(&external, 2, __builtin_ptrauth_blend_discriminator(&ptr4, 26));

// CHECK: @ptr5 = global ptr null
void *ptr5;

void test_sign_constant_code() {
// CHECK-LABEL: define {{.*}}void @test_sign_constant_code()
// CHECK-NEXT:  entry:
// CHECK-NEXT:    store ptr ptrauth (ptr @external, i32 0), ptr @ptr1, align 8
// CHECK-NEXT:    store ptr ptrauth (ptr @external, i32 2, i64 1234), ptr @ptr2, align 8
// CHECK-NEXT:    store ptr ptrauth (ptr @external, i32 2, i64 0, ptr @ptr3), ptr @ptr3, align 8
// CHECK-NEXT:    store ptr ptrauth (ptr @external, i32 2, i64 1234, ptr @ptr4), ptr @ptr4, align 8
// CHECK-NEXT:    ret void
  ptr1 = __builtin_ptrauth_sign_constant(&external, 0, 0);
  ptr2 = __builtin_ptrauth_sign_constant(&external, 2, 1234);
  ptr3 = __builtin_ptrauth_sign_constant(&external, 2, &ptr3);
  ptr4 = __builtin_ptrauth_sign_constant(&external, 2, __builtin_ptrauth_blend_discriminator(&ptr4, 1234));
}
