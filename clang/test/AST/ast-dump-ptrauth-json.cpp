// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -std=c++11 -ast-dump=json %s | FileCheck %s

// CHECK: "name": "__builtin_ptrauth_type_discriminator",
// CHECK: "qualType": "int *__ptrauth(1,1,123)"

int d = __builtin_ptrauth_type_discriminator(int());
int * __ptrauth(1,1,123) p;
