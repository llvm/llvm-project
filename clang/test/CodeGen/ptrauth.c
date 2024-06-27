// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck -check-prefix=CHECK -check-prefix=NOPCH %s
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-pch %s -o %t.ast
// RUN: %clang_cc1 -fptrauth-function-pointer-type-discrimination -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm -x ast -o - %t.ast | FileCheck -check-prefix=CHECK -check-prefix=PCH %s

#define FNPTRKEY 0

void (*fnptr)(void);
long discriminator;

extern void external_function(void);
// CHECK: @fptr1 = global ptr ptrauth (ptr @external_function, i32 0, i64 18983)
void (*fptr1)(void) = external_function;
// CHECK: @fptr2 = global ptr ptrauth (ptr @external_function, i32 0, i64 18983)
void (*fptr2)(void) = &external_function;

// CHECK: @fptr3 = global ptr ptrauth (ptr @external_function, i32 2, i64 26)
void (*fptr3)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, 26);

// CHECK: @fptr4 = global ptr ptrauth (ptr @external_function, i32 2, i64 26, ptr @fptr4)
void (*fptr4)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, __builtin_ptrauth_blend_discriminator(&fptr4, 26));

// CHECK-LABEL: define void @test_call()
void test_call() {
  // CHECK:      [[T0:%.*]] = load ptr, ptr @fnptr,
  // CHECK-NEXT: call void [[T0]]() [ "ptrauth"(i32 0, i64 18983) ]
  fnptr();
}

// CHECK-LABEL: define void @test_direct_call()
void test_direct_call() {
  // CHECK: call void @test_call(){{$}}
  test_call();
}

void abort();
// CHECK-LABEL: define void @test_direct_builtin_call()
void test_direct_builtin_call() {
  // CHECK: call void @abort() {{#[0-9]+$}}
  abort();
}

// CHECK-LABEL: define ptr @test_function_pointer()
// CHECK:  ret ptr ptrauth (ptr @external_function, i32 0, i64 18983)
void (*test_function_pointer())(void) {
  return external_function;
}

struct InitiallyIncomplete;
extern struct InitiallyIncomplete returns_initially_incomplete(void);
// CHECK-LABEL: define void @use_while_incomplete()
void use_while_incomplete() {
  // NOPCH:      [[VAR:%.*]] = alloca ptr,
  // NOPCH-NEXT: store ptr ptrauth (ptr @returns_initially_incomplete, i32 0, i64 25106), ptr [[VAR]]
  // PCH:        [[VAR:%.*]] = alloca ptr,
  // PCH-NEXT:   store ptr ptrauth (ptr @returns_initially_incomplete, i32 0, i64 25106), ptr [[VAR]]
  struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
}
struct InitiallyIncomplete { int x; };
// CHECK-LABEL: define void @use_while_complete()
void use_while_complete() {
  // CHECK:      [[VAR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr ptrauth (ptr @returns_initially_incomplete, i32 0, i64 25106), ptr [[VAR]]
  // CHECK-NEXT: ret void
  struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
}

// CHECK-LABEL: define void @test_memcpy_inline(
// CHECK-NOT: call{{.*}}memcpy

extern inline __attribute__((__always_inline__))
void *memcpy(void *d, const void *s, unsigned long) {
  return 0;
}

void test_memcpy_inline(char *d, char *s) {
  memcpy(d, s, 4);
}
