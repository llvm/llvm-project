// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s

#define __device__ __attribute__((device))

struct A { __device__ virtual void f(); };
struct B { __device__ virtual void g(); int b; };
struct C : A, B { __device__ void g() override; };
__device__ void C::g() {}

// CIR-LABEL: cir.func {{.*}} @_ZThn8_N1C1gEv
// CIR:         %[[THIS:.*]] = cir.load {{.*}}, !cir.ptr<!rec_C, target_address_space(4)>
// CIR:         %[[RAW:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<!rec_C, target_address_space(4)> -> !cir.ptr<!u8i, target_address_space(4)>
// CIR:         %[[ADJ:.*]] = cir.ptr_stride %[[RAW]], %{{.*}} : (!cir.ptr<!u8i, target_address_space(4)>, !s64i) -> !cir.ptr<!u8i, target_address_space(4)>
// CIR:         cir.cast bitcast %[[ADJ]] : !cir.ptr<!u8i, target_address_space(4)> -> !cir.ptr<!rec_C, target_address_space(4)>

// LLVM-LABEL: define spir_func void @_ZThn8_N1C1gEv(
// LLVM:         %[[THIS:.*]] = load ptr addrspace(4), ptr addrspace(4)
// LLVM:         getelementptr {{.*}}i8, ptr addrspace(4) %[[THIS]], i64 -8
