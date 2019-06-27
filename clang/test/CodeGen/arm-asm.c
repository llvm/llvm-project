// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumb %s -emit-llvm -o - | FileCheck %s
int t1() {
    static float k = 1.0f;
    // CHECK: flds s15
    __asm__ volatile ("flds s15, %[k] \n" :: [k] "Uv" (k) : "s15");
    return 0;
}
