// Based on clang/test/CodeGenCUDA/device-stub.cu (incubator).

// Create a dummy GPU binary file for registration.
// RUN: echo -n "GPU binary would be here." > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -o %t.nogpu.cir
// RUN: FileCheck --input-file=%t.nogpu.cir %s --check-prefix=NOGPUBIN

#include "Inputs/cuda.h"

__global__ void kernelfunc(int i, int j, int k) {}

void hostfunc(void) { kernelfunc<<<1, 1>>>(1, 1, 1); }

// CIR: cir.global "private" constant cir_private @__cuda_fatbin_str = #cir.const_array<"GPU binary would be here."> : !cir.array<!u8i x 25> {alignment = 8 : i64, section = ".nv_fatbin"}

// Check the fatbin wrapper struct: { magic, version, ptr to fatbin, null }, with section.
// CIR: cir.global constant cir_private @__cuda_fatbin_wrapper = #cir.const_record<{
// CIR-SAME: #cir.int<1180844977> : !s32i,
// CIR-SAME: #cir.int<1> : !s32i,
// CIR-SAME: #cir.global_view<@__cuda_fatbin_str> : !cir.ptr<!void>,
// CIR-SAME: #cir.ptr<null> : !cir.ptr<!void>
// CIR-SAME: }> : !rec_anon_struct {section = ".nvFatBinSegment"}

// Check the GPU binary handle global.
// CIR: cir.global "private" internal @__cuda_gpubin_handle = #cir.ptr<null> : !cir.ptr<!cir.ptr<!void>>

// OGCG: constant [25 x i8] c"GPU binary would be here.", section ".nv_fatbin", align 8
// OGCG: @__cuda_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1180844977, i32 1, ptr @{{.*}}, ptr null }, section ".nvFatBinSegment"
// OGCG: @__cuda_gpubin_handle = internal global ptr null

// No GPU binary — no registration infrastructure at all.
// NOGPUBIN-NOT: fatbin
// NOGPUBIN-NOT: gpubin
// NOGPUBIN-NOT: __cuda_register_globals
// NOGPUBIN-NOT: __cuda_module_ctor
// NOGPUBIN-NOT: __cuda_module_dtor
