// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s  -o - | FileCheck %s

#include <ptrauth.h>

typedef void* __ptrauth(2,1,1234,"strip") TestPtr;

// CHECK: @strip = private unnamed_addr constant [6 x i8] c"strip\00", align 1

int check_has_auth() {
  return __builtin_ptrauth_has_authentication(TestPtr);
}

// CHECK-LABEL: check_has_auth
// CHECK: ret i32 1

int check_has_no_auth() {
  return __builtin_ptrauth_has_authentication(float);
}
// CHECK-LABEL: check_has_no_auth
// CHECK: ret i32 0

int check_ptrauth_schema_key() {
  return __builtin_ptrauth_schema_key(TestPtr);
}
// CHECK-LABEL: check_ptrauth_schema_key
// ret i32 2

int check_ptrauth_schema_is_address_discriminated() {
  return __builtin_ptrauth_schema_is_address_discriminated(TestPtr);
}
// CHECK-LABEL: check_ptrauth_schema_is_address_discriminated
// CHECK: ret i32 1

int check_ptrauth_schema_extra_discriminator() {
  return __builtin_ptrauth_schema_extra_discriminator(TestPtr);
}
// CHECK-LABEL: check_ptrauth_schema_extra_discriminator
// CHECK: ret i32 1234

const char* check_ptrauth_schema_options() {
  return __builtin_ptrauth_schema_options(TestPtr);
}
// CHECK-LABEL: check_ptrauth_schema_options
// CHECK: ret ptr @strip
