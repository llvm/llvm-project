// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

#define FNPTRKEY 0

void (*fnptr)(void);
long discriminator;

extern void external_function(void);
// CHECK: [[EXTERNAL_FUNCTION:@.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 0, i64 0, i64 0 }, section "llvm.ptrauth", align 8
// CHECK: @fptr1 = global void ()* bitcast ({ i8*, i32, i64, i64 }* [[EXTERNAL_FUNCTION]] to void ()*)
void (*fptr1)(void) = external_function;
// CHECK: @fptr2 = global void ()* bitcast ({ i8*, i32, i64, i64 }* [[EXTERNAL_FUNCTION]] to void ()*)
void (*fptr2)(void) = &external_function;

// CHECK: [[SIGNED:@.*]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 2, i64 0, i64 26 }, section "llvm.ptrauth", align 8
// CHECK: @fptr3 = global void ()* bitcast ({ i8*, i32, i64, i64 }* [[SIGNED]] to void ()*)
void (*fptr3)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, 26);

// CHECK: @fptr4 = global void ()* bitcast ({ i8*, i32, i64, i64 }* [[SIGNED:@.*]] to void ()*)
// CHECK: [[SIGNED]] = private constant { i8*, i32, i64, i64 } { i8* bitcast (void ()* @external_function to i8*), i32 2, i64 ptrtoint (void ()** @fptr4 to i64), i64 26 }, section "llvm.ptrauth", align 8
void (*fptr4)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, __builtin_ptrauth_blend_discriminator(&fptr4, 26));

// CHECK-LABEL: define void @test_call()
void test_call() {
  // CHECK:      [[T0:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: call void [[T0]]() [ "ptrauth"(i32 0, i64 0) ]
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

// CHECK-LABEL: define void @test_sign_unauthenticated_peephole()
void test_sign_unauthenticated_peephole() {
  // CHECK:      [[T0:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: call void [[T0]](){{$}}
  // CHECK-NEXT: ret void
  __builtin_ptrauth_sign_unauthenticated(fnptr, FNPTRKEY, 0)();
}

// This peephole doesn't kick in because it's incorrect when ABI pointer
// authentication is enabled.
// CHECK-LABEL: define void @test_auth_peephole()
void test_auth_peephole() {
  // CHECK:      [[T0:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[T1:%.*]] = load i64, i64* @discriminator,
  // CHECK-NEXT: [[T2:%.*]] = ptrtoint void ()* [[T0]] to i64
  // CHECK-NEXT: [[T3:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T2]], i32 0, i64 [[T1]])
  // CHECK-NEXT: [[T4:%.*]] = inttoptr  i64 [[T3]] to void ()*
  // CHECK-NEXT: call void [[T4]]() [ "ptrauth"(i32 0, i64 0) ]
  // CHECK-NEXT: ret void
  __builtin_ptrauth_auth(fnptr, 0, discriminator)();
}

// CHECK-LABEL: define void @test_auth_and_resign_peephole()
void test_auth_and_resign_peephole() {
  // CHECK:      [[T0:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[T1:%.*]] = load i64, i64* @discriminator,
  // CHECK-NEXT: call void [[T0]]() [ "ptrauth"(i32 2, i64 [[T1]]) ]
  // CHECK-NEXT: ret void
  __builtin_ptrauth_auth_and_resign(fnptr, 2, discriminator, FNPTRKEY, 0)();
}

// CHECK-LABEL: define void ()* @test_function_pointer()
// CHECK:        [[EXTERNAL_FUNCTION]]
void (*test_function_pointer())(void) {
  return external_function;
}

// rdar://34562484 - Handle IR types changing in the caching mechanism.
struct InitiallyIncomplete;
extern struct InitiallyIncomplete returns_initially_incomplete(void);
// CHECK-LABEL: define void @use_while_incomplete()
void use_while_incomplete() {
  // CHECK:      [[VAR:%.*]] = alloca {}*,
  // CHECK-NEXT: store {}*  bitcast ({ i8*, i32, i64, i64 }* @returns_initially_incomplete.ptrauth to {}*), {}** [[VAR]],
  // CHECK-NEXT: ret void
  struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
}
struct InitiallyIncomplete { int x; };
// CHECK-LABEL: define void @use_while_complete()
void use_while_complete() {
  // CHECK:      [[VAR:%.*]] = alloca i64 ()*,
  // CHECK-NEXT: store i64 ()*  bitcast ({ i8*, i32, i64, i64 }* @returns_initially_incomplete.ptrauth to i64 ()*), i64 ()** [[VAR]],
  // CHECK-NEXT: ret void
  struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
}
