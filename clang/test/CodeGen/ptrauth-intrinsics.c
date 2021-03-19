// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

void (*fnptr)(void);
long int_discriminator;
void *ptr_discriminator;
long signature;

// CHECK-LABEL: define void @test_auth()
void test_auth() {
  // CHECK:      [[PTR:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[DISC0:%.*]] = load i8*, i8** @ptr_discriminator,
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint void ()* [[PTR]] to i64
  // CHECK-NEXT: [[DISC:%.*]] = ptrtoint i8* [[DISC0]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.auth.i64(i64 [[T0]], i32 0, i64 [[DISC]])
  // CHECK-NEXT: [[RESULT:%.*]] = inttoptr  i64 [[T1]] to void ()*
  // CHECK-NEXT: store void ()* [[RESULT]], void ()** @fnptr,
  fnptr = __builtin_ptrauth_auth(fnptr, 0, ptr_discriminator);
}

// CHECK-LABEL: define void @test_auth_peephole()
void test_auth_peephole() {
  // CHECK:      [[PTR:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[DISC0:%.*]] = load i8*, i8** @ptr_discriminator,
  // CHECK-NEXT: [[DISC:%.*]] = ptrtoint i8* [[DISC0]] to i64
  // CHECK-NEXT: call void [[PTR]]() [ "ptrauth"(i32 0, i64 [[DISC]]) ]
  // CHECK-NEXT: ret void
  __builtin_ptrauth_auth(fnptr, 0, ptr_discriminator)();
}

// CHECK-LABEL: define void @test_strip()
void test_strip() {
  // CHECK:      [[PTR:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint void ()* [[PTR]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.strip.i64(i64 [[T0]], i32 0)
  // CHECK-NEXT: [[RESULT:%.*]] = inttoptr  i64 [[T1]] to void ()*
  // CHECK-NEXT: store void ()* [[RESULT]], void ()** @fnptr,
  fnptr = __builtin_ptrauth_strip(fnptr, 0);
}

// CHECK-LABEL: define void @test_sign_unauthenticated()
void test_sign_unauthenticated() {
  // CHECK:      [[PTR:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[DISC0:%.*]] = load i8*, i8** @ptr_discriminator,
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint void ()* [[PTR]] to i64
  // CHECK-NEXT: [[DISC:%.*]] = ptrtoint i8* [[DISC0]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.sign.i64(i64 [[T0]], i32 0, i64 [[DISC]])
  // CHECK-NEXT: [[RESULT:%.*]] = inttoptr  i64 [[T1]] to void ()*
  // CHECK-NEXT: store void ()* [[RESULT]], void ()** @fnptr,
  fnptr = __builtin_ptrauth_sign_unauthenticated(fnptr, 0, ptr_discriminator);
}

// CHECK-LABEL: define void @test_auth_and_resign()
void test_auth_and_resign() {
  // CHECK:      [[PTR:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[DISC0:%.*]] = load i8*, i8** @ptr_discriminator,
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint void ()* [[PTR]] to i64
  // CHECK-NEXT: [[DISC:%.*]] = ptrtoint i8* [[DISC0]] to i64
  // CHECK-NEXT: [[T1:%.*]] = call i64 @llvm.ptrauth.resign.i64(i64 [[T0]], i32 0, i64 [[DISC]], i32 3, i64 15)
  // CHECK-NEXT: [[RESULT:%.*]] = inttoptr  i64 [[T1]] to void ()*
  // CHECK-NEXT: store void ()* [[RESULT]], void ()** @fnptr,
  fnptr = __builtin_ptrauth_auth_and_resign(fnptr, 0, ptr_discriminator, 3, 15);
}

// CHECK-LABEL: define void @test_blend_discriminator()
void test_blend_discriminator() {
  // CHECK:      [[PTR:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[DISC:%.*]] = load i64, i64* @int_discriminator,
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint void ()* [[PTR]] to i64
  // CHECK-NEXT: [[RESULT:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 [[T0]], i64 [[DISC]])
  // CHECK-NEXT: store i64 [[RESULT]], i64* @int_discriminator,
  int_discriminator = __builtin_ptrauth_blend_discriminator(fnptr, int_discriminator);
}

// CHECK-LABEL: define void @test_sign_generic_data()
void test_sign_generic_data() {
  // CHECK:      [[PTR:%.*]] = load void ()*, void ()** @fnptr,
  // CHECK-NEXT: [[DISC0:%.*]] = load i8*, i8** @ptr_discriminator,
  // CHECK-NEXT: [[T0:%.*]] = ptrtoint void ()* [[PTR]] to i64
  // CHECK-NEXT: [[DISC:%.*]] = ptrtoint i8* [[DISC0]] to i64
  // CHECK-NEXT: [[RESULT:%.*]] = call i64 @llvm.ptrauth.sign.generic.i64(i64 [[T0]], i64 [[DISC]])
  // CHECK-NEXT: store i64 [[RESULT]], i64* @signature,
  signature = __builtin_ptrauth_sign_generic_data(fnptr, ptr_discriminator);
}

// CHECK-LABEL: define void @test_string_discriminator()
void test_string_discriminator() {
  // CHECK:      [[X:%.*]] = alloca i32

  // Check a couple of random discriminators used by Swift.

  // CHECK:      store i32 58298, i32* [[X]],
  int x = __builtin_ptrauth_string_discriminator("InitializeWithCopy");

  // CHECK:      store i32 9112, i32* [[X]],
  x = __builtin_ptrauth_string_discriminator("DestroyArray");
}
