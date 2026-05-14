// Based on clang/test/CodeGenCUDA/device-stub.cu (incubator).

// Create a dummy GPU binary file for registration.
// RUN: echo -n "GPU binary would be here." > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x cuda \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x cuda \
// RUN:   -target-sdk-version=12.3 -o %t.nogpu.cir
// RUN: FileCheck --input-file=%t.nogpu.cir %s --check-prefix=NOGPUBIN

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x hip \
// RUN:   -fhip-new-launch-api -fcuda-include-gpubinary %t -o %t.hip.cir
// RUN: FileCheck --input-file=%t.hip.cir %s --check-prefix=HIP-CIR

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x hip \
// RUN:   -fhip-new-launch-api -fcuda-include-gpubinary %t -o %t.hip-cir.ll
// RUN: FileCheck --input-file=%t.hip-cir.ll %s --check-prefix=HIP-LLVM
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x hip \
// RUN:   -fhip-new-launch-api -fcuda-include-gpubinary %t -o %t.hip.ll
// RUN: FileCheck --input-file=%t.hip.ll %s --check-prefix=HIP-OGCG

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-cir %s -x hip \
// RUN:   -fhip-new-launch-api -o %t.nogpu.hip.cir
// RUN: FileCheck --input-file=%t.nogpu.hip.cir %s --check-prefix=HIP-NOGPUBIN

#include "Inputs/cuda.h"

__global__ void kernelfunc(int i, int j, int k) {}

void hostfunc(void) { kernelfunc<<<1, 1>>>(1, 1, 1); }

// Check module constructor is registered in module attributes.
// CIR: cir.global_ctors = [#cir.global_ctor<"__cuda_module_ctor", 65535>]

// Check runtime function declarations.
// CIR: cir.func private @atexit(!cir.ptr<!cir.func<()>>) -> !s32i
// CIR: cir.func private @__cudaUnregisterFatBinary(!cir.ptr<!cir.ptr<!void>>)

// Check the module destructor body: load handle and call UnregisterFatBinary.
// CIR: cir.func internal private @__cuda_module_dtor()
// CIR-NEXT: %[[HANDLE_ADDR:.*]] = cir.get_global @__cuda_gpubin_handle
// CIR-NEXT: %[[HANDLE:.*]] = cir.load %[[HANDLE_ADDR]]
// CIR-NEXT: cir.call @__cudaUnregisterFatBinary(%[[HANDLE]])
// CIR-NEXT: cir.return

// CIR: cir.func private @__cudaRegisterFatBinaryEnd(!cir.ptr<!cir.ptr<!void>>)

// Check the __cudaRegisterFunction runtime declaration:
//   int __cudaRegisterFunction(void**, void*, void*, void*, int,
//                              void*, void*, void*, void*, void*)
// CIR: cir.func private @__cudaRegisterFunction(!cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>) -> !s32i

// Check the device-side name string for kernelfunc (mangled, null-terminated).
// CIR: cir.global "private" constant cir_private @".str_Z10kernelfunciii" = #cir.const_array<"_Z10kernelfunciii" : !cir.array<!u8i x 18>, trailing_zeros> : !cir.array<!u8i x 18>

// Check __cuda_register_globals body: one __cudaRegisterFunction call per kernel.
// CIR: cir.func internal private @__cuda_register_globals(%arg0: !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-NEXT: %[[STR_ADDR:.*]] = cir.get_global @".str_Z10kernelfunciii"
// CIR-NEXT: %[[DEVICE_FUNC:.*]] = cir.cast bitcast %[[STR_ADDR]]
// CIR-NEXT: %[[HOST_FUNC_RAW:.*]] = cir.get_global @{{.*}}kernelfunc{{.*}}
// CIR-NEXT: %[[HOST_FUNC:.*]] = cir.cast bitcast %[[HOST_FUNC_RAW]]
// CIR-NEXT: %[[THREAD_LIMIT:.*]] = cir.const #cir.int<-1> : !s32i
// CIR-NEXT: cir.call @__cudaRegisterFunction(%{{.*}}, %[[HOST_FUNC]], %[[DEVICE_FUNC]], %[[DEVICE_FUNC]], %[[THREAD_LIMIT]], %[[NULL]], %[[NULL]], %[[NULL]], %[[NULL]], %[[NULL]])
// CIR-NEXT: cir.return

// CIR: cir.global "private" constant cir_private @__cuda_fatbin_str = #cir.const_array<"GPU binary would be here." : !cir.array<!u8i x 25>> : !cir.array<!u8i x 25> {alignment = 8 : i64, section = ".nv_fatbin"}

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
// call __cuda_register_globals, call RegisterFatBinaryEnd (CUDA >= 10.1),
// then register dtor with atexit.
// CIR: cir.func internal private @__cuda_module_ctor()
// CIR-NEXT: %[[WRAPPER:.*]] = cir.get_global @__cuda_fatbin_wrapper
// CIR-NEXT: %[[VOID_PTR:.*]] = cir.cast bitcast %[[WRAPPER]]
// CIR-NEXT: %[[RET:.*]] = cir.call @__cudaRegisterFatBinary(%[[VOID_PTR]])
// CIR-NEXT: %[[HANDLE_ADDR:.*]] = cir.get_global @__cuda_gpubin_handle
// CIR-NEXT: cir.store %[[RET]], %[[HANDLE_ADDR]]
// CIR-NEXT: cir.call @__cuda_register_globals(%[[RET]])
// CIR-NEXT: cir.call @__cudaRegisterFatBinaryEnd(%[[RET]])
// CIR-NEXT: %[[DTOR_PTR:.*]] = cir.get_global @__cuda_module_dtor
// CIR-NEXT: {{.*}} = cir.call @atexit(%[[DTOR_PTR]])
// CIR-NEXT: cir.return

// OGCG: constant [25 x i8] c"GPU binary would be here.", section ".nv_fatbin", align 8
// OGCG: @__cuda_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1180844977, i32 1, ptr @{{.*}}, ptr null }, section ".nvFatBinSegment"
// OGCG: @__cuda_gpubin_handle = internal global ptr null
// OGCG: @llvm.global_ctors = appending global {{.*}}@__cuda_module_ctor

// OGCG: define internal void @__cuda_register_globals
// OGCG: call{{.*}}__cudaRegisterFunction(ptr %0, {{.*}}kernelfunc{{.*}}, ptr @0
// OGCG: ret void

// OGCG: define internal void @__cuda_module_ctor
// OGCG: call{{.*}}__cudaRegisterFatBinary(ptr @__cuda_fatbin_wrapper)
// OGCG: store ptr %{{.*}}, ptr @__cuda_gpubin_handle
// OGCG-NEXT: call void @__cuda_register_globals
// OGCG: call i32 @atexit(ptr @__cuda_module_dtor)

// OGCG: define internal void @__cuda_module_dtor
// OGCG: load ptr, ptr @__cuda_gpubin_handle
// OGCG: call void @__cudaUnregisterFatBinary

// LLVM: constant [25 x i8] c"GPU binary would be here.", section ".nv_fatbin", align 8
// LLVM: @__cuda_fatbin_wrapper = {{.*}}constant { i32, i32, ptr, ptr } { i32 1180844977, i32 1, ptr @{{.*}}, ptr null }, section ".nvFatBinSegment"
// LLVM: @__cuda_gpubin_handle = internal global ptr null
// LLVM: @llvm.global_ctors = appending global {{.*}}@__cuda_module_ctor

// LLVM: define internal void @__cuda_module_dtor
// LLVM: load ptr, ptr @__cuda_gpubin_handle
// LLVM: call void @__cudaUnregisterFatBinary

// LLVM: define internal void @__cuda_register_globals
// LLVM: call{{.*}}@__cudaRegisterFunction(ptr %{{.*}}, ptr @{{.*}}kernelfunc{{.*}}, ptr @{{.*}}, ptr @{{.*}}, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// LLVM: ret void

// LLVM: define internal void @__cuda_module_ctor
// LLVM: call{{.*}}@__cudaRegisterFatBinary(ptr @__cuda_fatbin_wrapper)
// LLVM: store ptr %{{.*}}, ptr @__cuda_gpubin_handle
// LLVM-NEXT: call void @__cuda_register_globals
// LLVM: call i32 @atexit(ptr @__cuda_module_dtor)

// No GPU binary — no registration infrastructure at all.
// NOGPUBIN-NOT: fatbin
// NOGPUBIN-NOT: gpubin
// NOGPUBIN-NOT: __cuda_register_globals
// NOGPUBIN-NOT: __cuda_module_ctor
// NOGPUBIN-NOT: __cuda_module_dtor

// =============================================================================
// HIP host-side registration (`buildCUDAModuleCtor` / `buildHIPModuleDtor` /
// `buildCUDARegisterGlobalFunctions` HIP arms in CIR LoweringPrepare).
// =============================================================================

// HIP module ctor is registered with the default global-ctor priority.
// HIP-CIR: cir.global_ctors = [#cir.global_ctor<"__hip_module_ctor", 65535>]

// Runtime function decls.
// HIP-CIR: cir.func private @atexit(!cir.ptr<!cir.func<()>>) -> !s32i
// HIP-CIR: cir.func private @__hipUnregisterFatBinary(!cir.ptr<!cir.ptr<!void>>)

// Module dtor: only unregister when the handle is non-null, then null it out.
// Reuses the SSA value loaded in the entry block for the unregister call.
// HIP-CIR: cir.func internal private @__hip_module_dtor()
// HIP-CIR:   %[[DH0:.*]] = cir.get_global @__hip_gpubin_handle
// HIP-CIR:   %[[H0:.*]] = cir.load %[[DH0]]
// HIP-CIR:   %[[NULL0:.*]] = cir.const #cir.ptr<null>
// HIP-CIR:   %[[NE:.*]] = cir.cmp ne %[[H0]], %[[NULL0]]
// HIP-CIR:   cir.brcond %[[NE]] ^bb1, ^bb2
// HIP-CIR: ^bb1:
// HIP-CIR:   cir.call @__hipUnregisterFatBinary(%[[H0]])
// HIP-CIR:   %[[DH1:.*]] = cir.get_global @__hip_gpubin_handle
// HIP-CIR:   cir.store %[[NULL0]], %[[DH1]]
// HIP-CIR:   cir.br ^bb2
// HIP-CIR: ^bb2:
// HIP-CIR:   cir.return

// __hipRegisterFunction runtime declaration.
// HIP-CIR: cir.func private @__hipRegisterFunction(!cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>) -> !s32i

// __hip_register_globals: under -fhip-new-launch-api the host-side argument is
// the kernel-handle GlobalOp shadow (e.g. @_Z10kernelfunciii) — not the
// device-stub function pointer that the CUDA arm uses.
// HIP-CIR: cir.global "private" constant cir_private @".str_Z10kernelfunciii" = #cir.const_array<"_Z10kernelfunciii" : !cir.array<!u8i x 18>, trailing_zeros> : !cir.array<!u8i x 18>
// HIP-CIR: cir.func internal private @__hip_register_globals(%[[FATBIN:.*]]: !cir.ptr<!cir.ptr<!void>>
// HIP-CIR:   %[[NULL1:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// HIP-CIR:   %[[STR_ADDR:.*]] = cir.get_global @".str_Z10kernelfunciii"
// HIP-CIR:   %[[DEVICE_FUNC:.*]] = cir.cast bitcast %[[STR_ADDR]]
// HIP-CIR:   %[[KH:.*]] = cir.get_global @_Z10kernelfunciii : !cir.ptr<!cir.ptr<!cir.func<(!s32i, !s32i, !s32i)>>>
// HIP-CIR:   %[[HOST_FUNC:.*]] = cir.cast bitcast %[[KH]]
// HIP-CIR:   %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s32i
// HIP-CIR:   cir.call @__hipRegisterFunction(%[[FATBIN]], %[[HOST_FUNC]], %[[DEVICE_FUNC]], %[[DEVICE_FUNC]], %[[MINUS_ONE]], %[[NULL1]], %[[NULL1]], %[[NULL1]], %[[NULL1]], %[[NULL1]])
// HIP-CIR:   cir.return

// Fatbin string + wrapper live in the HIP-specific sections; magic
// 0x48495046 = 1212764230.
// HIP-CIR: cir.global "private" constant cir_private @__hip_fatbin_str = #cir.const_array<"GPU binary would be here." : !cir.array<!u8i x 25>> : !cir.array<!u8i x 25> {alignment = 8 : i64, section = ".hip_fatbin"}
// HIP-CIR: cir.global constant cir_private @__hip_fatbin_wrapper = #cir.const_record<{
// HIP-CIR-SAME: #cir.int<1212764230> : !s32i,
// HIP-CIR-SAME: #cir.int<1> : !s32i,
// HIP-CIR-SAME: #cir.global_view<@__hip_fatbin_str> : !cir.ptr<!void>,
// HIP-CIR-SAME: #cir.ptr<null> : !cir.ptr<!void>
// HIP-CIR-SAME: }> : !rec_anon_struct {section = ".hipFatBinSegment"}

// HIP-CIR: cir.global "private" internal @__hip_gpubin_handle = #cir.ptr<null> : !cir.ptr<!cir.ptr<!void>>
// HIP-CIR: cir.func private @__hipRegisterFatBinary(!cir.ptr<!void>) -> !cir.ptr<!cir.ptr<!void>>

// Module ctor: guard registration on a null handle, register globals from the
// (possibly newly-stored) handle, then atexit(__hip_module_dtor).
// HIP-CIR: cir.func internal private @__hip_module_ctor()
// HIP-CIR:   %[[GHA:.*]] = cir.get_global @__hip_gpubin_handle
// HIP-CIR:   %[[H:.*]] = cir.load %[[GHA]]
// HIP-CIR:   %[[NULLPTR:.*]] = cir.const #cir.ptr<null>
// HIP-CIR:   %[[EQ:.*]] = cir.cmp eq %[[H]], %[[NULLPTR]]
// HIP-CIR:   cir.brcond %[[EQ]] ^bb1, ^bb2
// HIP-CIR: ^bb1:
// HIP-CIR:   %[[WRAPPER:.*]] = cir.get_global @__hip_fatbin_wrapper
// HIP-CIR:   %[[VOID_PTR:.*]] = cir.cast bitcast %[[WRAPPER]]
// HIP-CIR:   %[[REG:.*]] = cir.call @__hipRegisterFatBinary(%[[VOID_PTR]])
// HIP-CIR:   %[[GHA2:.*]] = cir.get_global @__hip_gpubin_handle
// HIP-CIR:   cir.store %[[REG]], %[[GHA2]]
// HIP-CIR:   cir.br ^bb2
// HIP-CIR: ^bb2:
// HIP-CIR:   %[[GHA3:.*]] = cir.get_global @__hip_gpubin_handle
// HIP-CIR:   %[[H2:.*]] = cir.load %[[GHA3]]
// HIP-CIR:   cir.call @__hip_register_globals(%[[H2]])
// HIP-CIR:   %[[DTOR_PTR:.*]] = cir.get_global @__hip_module_dtor
// HIP-CIR:   {{.*}} = cir.call @atexit(%[[DTOR_PTR]])
// HIP-CIR:   cir.return

// HIP-CIR: cir.global constant external @_Z10kernelfunciii = #cir.global_view<@_Z25__device_stub__kernelfunciii> : !cir.ptr<!cir.func<(!s32i, !s32i, !s32i)>> {alignment = 8 : i64}

// HIP OGCG cross-check (LLVM IR matches what OG codegen emits for HIP).
// HIP-OGCG: @{{.*}} = private constant [25 x i8] c"GPU binary would be here.", section ".hip_fatbin"
// HIP-OGCG: @__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @{{.*}}, ptr null }, section ".hipFatBinSegment"
// HIP-OGCG: @__hip_gpubin_handle = internal global ptr null
// HIP-OGCG: @llvm.global_ctors = appending global {{.*}}@__hip_module_ctor

// HIP-OGCG: define internal void @__hip_module_ctor()
// HIP-OGCG:   load ptr, ptr @__hip_gpubin_handle
// HIP-OGCG:   icmp eq ptr {{.*}}, null
// HIP-OGCG:   call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
// HIP-OGCG:   store ptr {{.*}}, ptr @__hip_gpubin_handle
// HIP-OGCG:   call void @__hip_register_globals(
// HIP-OGCG:   call i32 @atexit(ptr @__hip_module_dtor)
// HIP-OGCG:   ret void

// HIP-OGCG: define internal void @__hip_module_dtor()
// HIP-OGCG:   load ptr, ptr @__hip_gpubin_handle
// HIP-OGCG:   icmp ne ptr {{.*}}, null
// HIP-OGCG:   call void @__hipUnregisterFatBinary
// HIP-OGCG:   store ptr null, ptr @__hip_gpubin_handle

// HIP LLVM lowering cross-check.
// HIP-LLVM: @{{.*}} = private constant [25 x i8] c"GPU binary would be here.", section ".hip_fatbin", align 8
// HIP-LLVM: @__hip_fatbin_wrapper = {{.*}}constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @{{.*}}, ptr null }, section ".hipFatBinSegment"
// HIP-LLVM: @__hip_gpubin_handle = internal global ptr null
// HIP-LLVM: @_Z10kernelfunciii = constant ptr @_Z25__device_stub__kernelfunciii, align 8
// HIP-LLVM: @llvm.global_ctors = appending global {{.*}}@__hip_module_ctor

// HIP-LLVM: define internal void @__hip_module_dtor()
// HIP-LLVM: load ptr, ptr @__hip_gpubin_handle
// HIP-LLVM: icmp ne ptr {{.*}}, null
// HIP-LLVM: br i1 {{.*}}, label %{{.*}}, label %{{.*}}
// HIP-LLVM: call void @__hipUnregisterFatBinary(ptr {{.*}})
// HIP-LLVM: store ptr null, ptr @__hip_gpubin_handle
// HIP-LLVM: ret void

// HIP-LLVM: define internal void @__hip_register_globals(ptr %[[FATBIN:.*]])
// HIP-LLVM: call{{.*}}@__hipRegisterFunction(ptr %[[FATBIN]], ptr @_Z10kernelfunciii, ptr @{{.*}}, ptr @{{.*}}, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// HIP-LLVM: ret void

// HIP-LLVM: define internal void @__hip_module_ctor()
// HIP-LLVM: load ptr, ptr @__hip_gpubin_handle
// HIP-LLVM: icmp eq ptr {{.*}}, null
// HIP-LLVM: br i1 {{.*}}, label %{{.*}}, label %{{.*}}
// HIP-LLVM: call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
// HIP-LLVM: store ptr {{.*}}, ptr @__hip_gpubin_handle
// HIP-LLVM: load ptr, ptr @__hip_gpubin_handle
// HIP-LLVM: call void @__hip_register_globals(ptr {{.*}})
// HIP-LLVM: call i32 @atexit(ptr @__hip_module_dtor)
// HIP-LLVM: ret void

// No GPU binary: no fatbin, no handle, no registration scaffolding.
// HIP-NOGPUBIN-NOT: __hip_fatbin
// HIP-NOGPUBIN-NOT: __hip_gpubin_handle
// HIP-NOGPUBIN-NOT: __hip_register_globals
// HIP-NOGPUBIN-NOT: __hip_module_ctor
// HIP-NOGPUBIN-NOT: __hip_module_dtor
