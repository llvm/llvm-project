// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -O0 -S -emit-llvm -disable-llvm-optzns -o - %s | FileCheck %s --check-prefixes=CHECK,O0
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -O2 -S -emit-llvm -disable-llvm-optzns -o - %s | FileCheck %s --check-prefixes=CHECK,O2


// Check that we make a direct call from direct_caller._Msimd to
// direct_callee._Msimd when there is no better option.
__attribute__((target_version("simd"))) int direct_callee(void) { return 1; }
__attribute__((target_version("default"))) int direct_callee(void) { return 2; }
__attribute__((target_version("simd"))) int direct_caller(void) { return direct_callee(); }
__attribute__((target_version("default"))) int direct_caller(void) { return direct_callee(); }
// O0-LABEL: @direct_caller._Msimd(
// O0:    = call i32 @direct_callee.ifunc()
// O2-LABEL: @direct_caller._Msimd(
// O2:    = call i32 @direct_callee._Msimd()


__attribute__((target_version("simd"), optnone)) int optnone_caller(void) { return direct_callee(); }
__attribute__((target_version("default"), optnone)) int optnone_caller(void) { return direct_callee(); }
// CHECK-LABEL: @optnone_caller._Msimd(
// CHECK: = call i32 @direct_callee.ifunc()


// ... and that we go through the ifunc+resolver when there is a better option
// that might be chosen at runtime.
__attribute__((target_version("simd"))) int resolved_callee1(void) { return 3; }
__attribute__((target_version("fcma"))) int resolved_callee1(void) { return 4; }
__attribute__((target_version("default"))) int resolved_callee1(void) { return 5; }
__attribute__((target_version("simd"))) int resolved_caller1(void) { return resolved_callee1(); }
__attribute__((target_version("default"))) int resolved_caller1(void) { return resolved_callee1(); }
// CHECK-LABEL: @resolved_caller1._Msimd(
// CHECK: = call i32 @resolved_callee1.ifunc()


// FIXME: we could direct call in cases like this:
__attribute__((target_version("fp"))) int resolved_callee2(void) { return 6; }
__attribute__((target_version("default"))) int resolved_callee2(void) { return 7; }
__attribute__((target_version("simd+fp"))) int resolved_caller2(void) { return resolved_callee2(); }
__attribute__((target_version("default"))) int resolved_caller2(void) { return resolved_callee2(); }
// CHECK-LABEL: @resolved_caller2._MfpMsimd(
// CHECK: = call i32 @resolved_callee2.ifunc()


// CHECK: @direct_caller.default(
// CHECK  = call i32 @direct_callee.ifunc()
// CHECK-LABEL: @optnone_caller.default(
// CHECK: = call i32 @direct_callee.ifunc()
// CHECK-LABEL: @resolved_caller1.default(
// CHECK: = call i32 @resolved_callee1.ifunc()
// CHECK-LABEL: @resolved_caller2.default(
// CHECK: = call i32 @resolved_callee2.ifunc()

int source() {
    return direct_caller() +
           optnone_caller() +
           resolved_caller1() +
           resolved_caller2();
}
