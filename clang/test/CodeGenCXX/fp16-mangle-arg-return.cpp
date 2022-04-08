// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple arm-arm-none-eabi -fallow-half-arguments-and-returns %s | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple aarch64-arm-none-eabi -fallow-half-arguments-and-returns %s | FileCheck %s

// Test name-mangling of __fp16 passed directly as a function argument
// (when that is permitted).

// CHECK: define {{.*}}void @_Z13fp16_argumentDh(half noundef %{{.*}})
void fp16_argument(__fp16 arg) {}

// Test name-mangling of __fp16 as a return type. The return type of
// fp16_return itself isn't mentioned in the mangled name, so to test
// this, we have to pass it a function pointer and make __fp16 the
// return type of that.

// CHECK: define {{.*}}void @_Z11fp16_returnPFDhvE(half ()* noundef %{{.*}})
void fp16_return(__fp16 (*func)(void)) {}
