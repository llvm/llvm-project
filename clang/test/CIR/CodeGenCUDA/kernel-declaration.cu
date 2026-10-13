// A launch of a kernel defined in another TU must not register that kernel or
// embed this TU's fat binary. Match classic codegen for both declaration-only
// TUs and TUs that also define a different kernel.

// RUN: echo -n "GPU binary would be here." > %t.fatbin

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefix=CIR --implicit-check-not=fatbin \
// RUN:     --implicit-check-not=Register --implicit-check-not=module_ctor

// With no entities to register, the binary must not even be read.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.nonexistent -o - \
// RUN:   | FileCheck %s --check-prefix=CIR --implicit-check-not=fatbin \
// RUN:     --implicit-check-not=Register --implicit-check-not=module_ctor

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefix=LAUNCH --implicit-check-not=fatbin \
// RUN:     --implicit-check-not=Register --implicit-check-not=module_ctor \
// RUN:     --implicit-check-not=llvm.global_ctors
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefix=LAUNCH --implicit-check-not=fatbin \
// RUN:     --implicit-check-not=Register --implicit-check-not=module_ctor \
// RUN:     --implicit-check-not=llvm.global_ctors

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x hip \
// RUN:   -fhip-new-launch-api -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefix=LAUNCH --implicit-check-not=fatbin \
// RUN:     --implicit-check-not=Register --implicit-check-not=module_ctor \
// RUN:     --implicit-check-not=llvm.global_ctors

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x cuda \
// RUN:   -DDEFINE_KERNEL -target-sdk-version=12.3 -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefixes=LAUNCH,MIXED \
// RUN:     --implicit-check-not='call{{.*}} @__{{cuda|hip}}RegisterFunction({{.*}}ptr @{{[^,]*}}k0'
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x cuda \
// RUN:   -DDEFINE_KERNEL -target-sdk-version=12.3 -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefixes=LAUNCH,MIXED \
// RUN:     --implicit-check-not='call{{.*}} @__{{cuda|hip}}RegisterFunction({{.*}}ptr @{{[^,]*}}k0'
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x hip \
// RUN:   -DDEFINE_KERNEL -fhip-new-launch-api -fcuda-include-gpubinary %t.fatbin -o - \
// RUN:   | FileCheck %s --check-prefixes=LAUNCH,MIXED \
// RUN:     --implicit-check-not='call{{.*}} @__{{cuda|hip}}RegisterFunction({{.*}}ptr @{{[^,]*}}k0'

#include "Inputs/cuda.h"

__global__ void k0(int);
#ifdef DEFINE_KERNEL
__global__ void k1(int);
#endif

void launch() {
  k0<<<1, 1>>>(42);
#ifdef DEFINE_KERNEL
  k1<<<1, 1>>>(42);
#endif
}

// Define after the launch to exercise upgrading an existing stub declaration.
#ifdef DEFINE_KERNEL
__global__ void k1(int) {}
#endif

// CIR: cir.call @_Z17__device_stub__k0i(
// LAUNCH-DAG: call void @_Z17__device_stub__k0i(i32{{.*}} 42)
// LAUNCH-DAG: declare void @_Z17__device_stub__k0i(
// MIXED-DAG: define{{.*}} void @_Z17__device_stub__k1i(
// MIXED-DAG: call{{.*}} @__{{cuda|hip}}RegisterFunction({{.*}}ptr @{{[^,]*}}k1{{[^,]*}},
// MIXED-DAG: define internal void @__{{cuda|hip}}_module_ctor()
