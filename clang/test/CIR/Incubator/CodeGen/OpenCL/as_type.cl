// RUN: %clang_cc1 %s -cl-std=CL2.0 -fclangir -emit-cir -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=CIR

// RUN: %clang_cc1 %s -cl-std=CL2.0 -fclangir -emit-llvm -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -triple spirv64-unknown-unknown -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OG-LLVM

typedef __attribute__(( ext_vector_type(4) )) char char4;

// CIR: cir.func @f4(%{{.*}}: !s32i loc({{.*}})) -> !cir.vector<!s8i x 4>
// CIR: %[[x:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR: cir.cast bitcast %[[x]] : !s32i -> !cir.vector<!s8i x 4>
// LLVM: define spir_func <4 x i8> @f4(i32 %[[x:.*]])
// LLVM: %[[astype:.*]] = bitcast i32 %[[x]]  to <4 x i8>
// LLVM-NOT: shufflevector
// LLVM: ret <4 x i8> %[[astype]]
// OG-LLVM: define spir_func noundef <4 x i8> @f4(i32 noundef %[[x:.*]])
// OG-LLVM: %[[astype:.*]] = bitcast i32 %[[x]] to <4 x i8>
// OG-LLVM-NOT: shufflevector
// OG-LLVM: ret <4 x i8> %[[astype]]
char4 f4(int x) {
  return __builtin_astype(x, char4);
}

// CIR: cir.func @f6(%{{.*}}: !cir.vector<!s8i x 4> loc({{.*}})) -> !s32i
// CIR: %[[x:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!cir.vector<!s8i x 4>, lang_address_space(offload_private)>, !cir.vector<!s8i x 4>
// CIR: cir.cast bitcast %[[x]] : !cir.vector<!s8i x 4> -> !s32i
// LLVM: define{{.*}} spir_func i32 @f6(<4 x i8> %[[x:.*]])
// LLVM: %[[astype:.*]] = bitcast <4 x i8> %[[x]] to i32
// LLVM-NOT: shufflevector
// LLVM: ret i32 %[[astype]]
// OG-LLVM: define{{.*}} spir_func noundef i32 @f6(<4 x i8> noundef %[[x:.*]])
// OG-LLVM: %[[astype:.*]] = bitcast <4 x i8> %[[x]] to i32
// OG-LLVM-NOT: shufflevector
// OG-LLVM: ret i32 %[[astype]]
int f6(char4 x) {
  return __builtin_astype(x, int);
}

// CIR: cir.func @f4_ptr(%{{.*}}: !cir.ptr<!s32i, lang_address_space(offload_global)> loc({{.*}})) -> !cir.ptr<!cir.vector<!s8i x 4>, lang_address_space(offload_local)>
// CIR: %[[x:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>, lang_address_space(offload_private)>, !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR: cir.cast address_space %[[x]] : !cir.ptr<!s32i, lang_address_space(offload_global)> -> !cir.ptr<!cir.vector<!s8i x 4>, lang_address_space(offload_local)>
// LLVM: define spir_func ptr addrspace(3) @f4_ptr(ptr addrspace(1) readnone captures(ret: address, provenance) %[[x:.*]])
// LLVM: %[[astype:.*]] = addrspacecast ptr addrspace(1) %[[x]] to ptr addrspace(3)
// LLVM-NOT: shufflevector
// LLVM: ret ptr addrspace(3) %[[astype]]
// OG-LLVM: define spir_func ptr addrspace(3) @f4_ptr(ptr addrspace(1) noundef readnone captures(ret: address, provenance) %[[x:.*]])
// OG-LLVM: %[[astype:.*]] = addrspacecast ptr addrspace(1) %[[x]] to ptr addrspace(3)
// OG-LLVM-NOT: shufflevector
// OG-LLVM: ret ptr addrspace(3) %[[astype]]
__local char4* f4_ptr(__global int* x) {
  return __builtin_astype(x, __local char4*);
}