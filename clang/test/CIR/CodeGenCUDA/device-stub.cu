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

// Check module constructor is registered in module attributes.
// CIR: cir.global_ctors = [#cir.global_ctor<"__cuda_module_ctor", 65535>]

// Check runtime function declarations (appear before dtor in output).
// CIR: cir.func private @atexit(!cir.ptr<!cir.func<()>>) -> !s32i
// CIR: cir.func private @__cudaUnregisterFatBinary(!cir.ptr<!cir.ptr<!void>>)

// Check the module destructor body: load handle and call UnregisterFatBinary.
// CIR: cir.func internal private @__cuda_module_dtor()
// CIR-NEXT: %[[HANDLE_ADDR:.*]] = cir.get_global @__cuda_gpubin_handle
// CIR-NEXT: %[[HANDLE:.*]] = cir.load %[[HANDLE_ADDR]]
// CIR-NEXT: cir.call @__cudaUnregisterFatBinary(%[[HANDLE]])
// CIR-NEXT: cir.return

// CIR: cir.func private @__cudaRegisterFatBinaryEnd(!cir.ptr<!cir.ptr<!void>>)

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

// CIR: cir.func private @__cudaRegisterFatBinary(!cir.ptr<!void>) -> !cir.ptr<!cir.ptr<!void>>

// Check the module constructor body: register fatbin, store handle,
// call RegisterFatBinaryEnd (CUDA >= 10.1), then register dtor with atexit.
// CIR: cir.func internal private @__cuda_module_ctor()
// CIR-NEXT: %[[WRAPPER:.*]] = cir.get_global @__cuda_fatbin_wrapper
// CIR-NEXT: %[[VOID_PTR:.*]] = cir.cast bitcast %[[WRAPPER]]
// CIR-NEXT: %[[RET:.*]] = cir.call @__cudaRegisterFatBinary(%[[VOID_PTR]])
// CIR-NEXT: %[[HANDLE_ADDR:.*]] = cir.get_global @__cuda_gpubin_handle
// CIR-NEXT: cir.store %[[RET]], %[[HANDLE_ADDR]]
// CIR-NEXT: cir.call @__cudaRegisterFatBinaryEnd(%[[RET]])
// CIR-NEXT: %[[DTOR_PTR:.*]] = cir.get_global @__cuda_module_dtor
// CIR-NEXT: {{.*}} = cir.call @atexit(%[[DTOR_PTR]])
// CIR-NEXT: cir.return

// OGCG: constant [25 x i8] c"GPU binary would be here.", section ".nv_fatbin", align 8
// OGCG: @__cuda_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1180844977, i32 1, ptr @{{.*}}, ptr null }, section ".nvFatBinSegment"
// OGCG: @__cuda_gpubin_handle = internal global ptr null
// OGCG: @llvm.global_ctors = appending global {{.*}}@__cuda_module_ctor

// OGCG: define internal void @__cuda_module_ctor
// OGCG: call{{.*}}__cudaRegisterFatBinary(ptr @__cuda_fatbin_wrapper)
// OGCG: store ptr %{{.*}}, ptr @__cuda_gpubin_handle
// OGCG: call i32 @atexit(ptr @__cuda_module_dtor)

// OGCG: define internal void @__cuda_module_dtor
// OGCG: load ptr, ptr @__cuda_gpubin_handle
// OGCG: call void @__cudaUnregisterFatBinary

// No GPU binary — no registration infrastructure at all.
// NOGPUBIN-NOT: fatbin
// NOGPUBIN-NOT: gpubin
// NOGPUBIN-NOT: __cuda_register_globals
// NOGPUBIN-NOT: __cuda_module_ctor
// NOGPUBIN-NOT: __cuda_module_dtor
