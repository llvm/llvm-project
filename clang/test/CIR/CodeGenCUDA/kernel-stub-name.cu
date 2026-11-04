// Based on clang/test/CodeGenCUDA/kernel-stub-name.cu.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -x cuda -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --implicit-check-not=cu.kernel_name

#include "Inputs/cuda.h"

// CHECK: cir.func {{.*}} @[[CSTUB:__device_stub__ckernel]]()
// CIR-BEFORE-LPP: cir.func {{.*}} @[[CSTUB:__device_stub__ckernel]]() attributes {{{.*}}cu.kernel_name = #cir.cu.kernel_name<"ckernel">{{.*}}}
// CHECK: cir.return
// CHECK-NEXT: }
extern "C" __global__ void ckernel() {}

// CHECK: cir.func {{.*}} @_ZN2ns23__device_stub__nskernelEv()
// CIR-BEFORE-LPP: cir.func {{.*}} @_ZN2ns23__device_stub__nskernelEv() attributes {{{.*}}cu.kernel_name = #cir.cu.kernel_name<"_ZN2ns8nskernelEv">{{.*}}}
namespace ns {
__global__ void nskernel() {}
} // namespace ns

// CHECK: cir.func {{.*}} @_Z25__device_stub__kernelfuncIiEvv()
// CIR-BEFORE-LPP: cir.func {{.*}} @_Z25__device_stub__kernelfuncIiEvv() attributes {{{.*}}cu.kernel_name = #cir.cu.kernel_name<"_Z10kernelfuncIiEvv">{{.*}}}
template <class T>
__global__ void kernelfunc() {}
template __global__ void kernelfunc<int>();
