// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-cir -fclangir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-llvm -fclangir -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM

#define as_int(x) __builtin_astype((x), int)
#define as_float(x) __builtin_astype((x), float)

int float_to_int(float x)
{
  return as_int(x);
}

// CIR: cir.cast bitcast %{{.*}} : !cir.float -> !s32i
// LLVM: bitcast float %{{.*}} to i32
// OG-LLVM: bitcast float %{{.*}} to i32

float int_to_float(int x)
{
  return as_float(x);
}

// CIR: cir.cast bitcast %{{.*}} : !s32i -> !cir.float
// LLVM: bitcast i32 %{{.*}} to float
// OG-LLVM: bitcast i32 %{{.*}} to float