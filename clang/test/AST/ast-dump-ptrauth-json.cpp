// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -std=c++11 -ast-dump=json %s | FileCheck %s

// CHECK: "name": "__builtin_ptrauth_type_discriminator",

int d = __builtin_ptrauth_type_discriminator(int());
