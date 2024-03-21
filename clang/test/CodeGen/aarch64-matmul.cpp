// RUN: %clang_cc1             -triple aarch64 -target-feature +neon -target-feature +i8mm -S -emit-llvm %s -o - | FileCheck %s

#ifdef __ARM_FEATURE_MATMUL_INT8
extern "C" void arm_feature_matmulint8_defined() {}
#endif
// CHECK: define{{.*}} void @arm_feature_matmulint8_defined()


