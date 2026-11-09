// RUN: %clang_cc1 -triple thumbv7-windows-msvc -fms-extensions -Werror -emit-llvm -o - %s | FileCheck %s

// pcs("aapcs-vfp") is the default on Windows on ARM: no warning, and the
// functions mangle as __cdecl.

#define ABI __attribute__((pcs("aapcs-vfp")))
using StageFn = void ABI (*)(float *, float);
static void ABI stage(float *p, float v) { *p = v; }
StageFn get() { return stage; }
void (*plain)(float *, float) = get();

// CHECK-DAG: define dso_local arm_aapcs_vfpcc {{.*}}ptr @"?get@@YAP6AXPAMM@ZXZ"()
// CHECK-DAG: define internal arm_aapcs_vfpcc void @"?stage@@YAXPAMM@Z"(
