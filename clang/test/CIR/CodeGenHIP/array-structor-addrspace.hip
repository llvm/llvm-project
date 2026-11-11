// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -fcxx-exceptions -fexceptions -DEXCEPTIONS -emit-cir %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s -check-prefix=CIR-EH

// Array elements and array cookies are in the address space of the array.

#define __device__ __attribute__((device))

typedef __SIZE_TYPE__ size_t;
__device__ void *operator new[](size_t);
__device__ void operator delete[](void *);

struct S {
  __device__ S();
  __device__ S(int);
  __device__ ~S();
  int x;
};

#ifndef EXCEPTIONS

__device__ void local_array() { S arr[4]; }

// CIR-LABEL: cir.func {{.*}}@_Z11local_arrayv
// CIR: %[[ARR:.*]] = cir.cast address_space %{{.*}} : !cir.ptr<!cir.array<!rec_S x 4>> -> !cir.ptr<!cir.array<!rec_S x 4>, target_address_space(4)>
// CIR: cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!rec_S x 4>, target_address_space(4)> -> !cir.ptr<!rec_S, target_address_space(4)>
// CIR: cir.call @_ZN1SC1Ev(%{{.*}}) {{.*}} : (!cir.ptr<!rec_S, target_address_space(4)> {{.*}}) -> ()
// CIR: cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!rec_S x 4>, target_address_space(4)> -> !cir.ptr<!rec_S, target_address_space(4)>
// CIR: cir.call @_ZN1SD1Ev(%{{.*}}) {{.*}} : (!cir.ptr<!rec_S, target_address_space(4)> {{.*}}) -> ()

// LLVM-LABEL: define {{.*}}@_Z11local_arrayv
// LLVM: %[[ARR:.*]] = addrspacecast ptr %{{.*}} to ptr addrspace(4)
// LLVM: getelementptr %struct.S, ptr addrspace(4) %[[ARR]], i32 0
// LLVM: call spir_func addrspace(4) void @_ZN1SC1Ev(ptr addrspace(4) {{.*}})
// LLVM: getelementptr %struct.S, ptr addrspace(4) %[[ARR]], i32 0
// LLVM: call spir_func addrspace(4) void @_ZN1SD1Ev(ptr addrspace(4) {{.*}})

// OGCG-LABEL: define {{.*}}@_Z11local_arrayv
// OGCG: %[[ARR:.*]] = addrspacecast ptr %{{.*}} to ptr addrspace(4)
// OGCG: getelementptr inbounds [4 x %struct.S], ptr addrspace(4) %[[ARR]], i32 0, i32 0
// OGCG: call spir_func addrspace(4) void @_ZN1SC1Ev(ptr addrspace(4) {{.*}})
// OGCG: getelementptr inbounds [4 x %struct.S], ptr addrspace(4) %[[ARR]], i32 0, i32 0
// OGCG: call spir_func addrspace(4) void @_ZN1SD1Ev(ptr addrspace(4) {{.*}})

__device__ S *new_array(int n) { return new S[n]; }

// CIR-LABEL: cir.func {{.*}}@_Z9new_arrayi
// CIR: %[[ALLOC:.*]] = cir.call @_Znam(%{{.*}}) {{.*}} -> (!cir.ptr<!void, target_address_space(4)> {{.*}})
// CIR: %[[BYTES:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void, target_address_space(4)> -> !cir.ptr<!u8i, target_address_space(4)>
// CIR: %[[COOKIE:.*]] = cir.cast bitcast %[[BYTES]] : !cir.ptr<!u8i, target_address_space(4)> -> !cir.ptr<!u64i, target_address_space(4)>
// CIR: cir.store align(8) %{{.*}}, %[[COOKIE]] : !u64i, !cir.ptr<!u64i, target_address_space(4)>
// CIR: cir.ptr_stride %[[BYTES]], %{{.*}} : (!cir.ptr<!u8i, target_address_space(4)>, !s32i) -> !cir.ptr<!u8i, target_address_space(4)>
// CIR: cir.call @_ZN1SC1Ev(%{{.*}}) {{.*}} : (!cir.ptr<!rec_S, target_address_space(4)> {{.*}}) -> ()

// LLVM-LABEL: define {{.*}}@_Z9new_arrayi
// LLVM: %[[ALLOC:.*]] = call spir_func {{.*}} ptr addrspace(4) @_Znam(
// LLVM: store i64 %{{.*}}, ptr addrspace(4) %[[ALLOC]], align 8
// LLVM: getelementptr i8, ptr addrspace(4) %[[ALLOC]], i64 8
// LLVM: call spir_func addrspace(4) void @_ZN1SC1Ev(ptr addrspace(4) {{.*}})

// OGCG-LABEL: define {{.*}}@_Z9new_arrayi
// OGCG: %[[ALLOC:.*]] = call spir_func {{.*}} ptr addrspace(4) @_Znam(
// OGCG: store i64 %{{.*}}, ptr addrspace(4) %[[ALLOC]], align 8
// OGCG: getelementptr inbounds i8, ptr addrspace(4) %[[ALLOC]], i64 8
// OGCG: call spir_func addrspace(4) void @_ZN1SC1Ev(ptr addrspace(4) {{.*}})

__device__ void delete_array(S *p) { delete[] p; }

// CIR-LABEL: cir.func {{.*}}@_Z12delete_arrayP1S
// CIR: %[[BYTES:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_S, target_address_space(4)> -> !cir.ptr<!u8i, target_address_space(4)>
// CIR: %[[ALLOC:.*]] = cir.ptr_stride %[[BYTES]], %{{.*}} : (!cir.ptr<!u8i, target_address_space(4)>, !s64i) -> !cir.ptr<!u8i, target_address_space(4)>
// CIR: %[[VOID:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!u8i, target_address_space(4)> -> !cir.ptr<!void, target_address_space(4)>
// CIR: %[[COOKIE:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!u8i, target_address_space(4)> -> !cir.ptr<!u64i, target_address_space(4)>
// CIR: cir.load align(4) %[[COOKIE]] : !cir.ptr<!u64i, target_address_space(4)>, !u64i
// CIR: cir.call @_ZN1SD1Ev(%{{.*}}) {{.*}} : (!cir.ptr<!rec_S, target_address_space(4)>) -> ()
// CIR: cir.call @_ZdaPvm(%[[VOID]], %{{.*}}) {{.*}} : (!cir.ptr<!void, target_address_space(4)>, !u64i) -> ()

// LLVM-LABEL: define {{.*}}@_Z12delete_arrayP1S
// LLVM: %[[ALLOC:.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i64 -8
// LLVM: load i64, ptr addrspace(4) %[[ALLOC]], align 4
// LLVM: call spir_func addrspace(4) void @_ZN1SD1Ev(ptr addrspace(4) %{{.*}})
// LLVM: call spir_func addrspace(4) void @_ZdaPvm(ptr addrspace(4) %[[ALLOC]], i64 %{{.*}})

// OGCG-LABEL: define {{.*}}@_Z12delete_arrayP1S
// OGCG: %[[ALLOC:.*]] = getelementptr inbounds i8, ptr addrspace(4) %{{.*}}, i64 -8
// OGCG: load i64, ptr addrspace(4) %[[ALLOC]], align 4
// OGCG: call spir_func addrspace(4) void @_ZN1SD1Ev(ptr addrspace(4) {{.*}})
// OGCG: call spir_func addrspace(4) void @_ZdaPvm(ptr addrspace(4) noundef %[[ALLOC]], i64 noundef %{{.*}})

#else

__device__ void partial_init() { S arr[4] = {1}; }

// CIR-EH-LABEL: cir.func {{.*}}@_Z12partial_initv
// CIR-EH: %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %{{.*}} : !cir.ptr<!cir.array<!rec_S x 4>, target_address_space(4)> -> !cir.ptr<!rec_S, target_address_space(4)>
// CIR-EH: } cleanup eh {
// CIR-EH:   %[[END:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_S, target_address_space(4)>, target_address_space(4)>, !cir.ptr<!rec_S, target_address_space(4)>
// CIR-EH:   cir.cmp ne %[[END]], %[[BEGIN]] : !cir.ptr<!rec_S, target_address_space(4)>
// CIR-EH:     cir.ptr_stride %{{.*}}, %{{.*}} : (!cir.ptr<!rec_S, target_address_space(4)>, !s64i) -> !cir.ptr<!rec_S, target_address_space(4)>
// CIR-EH:     cir.call @_ZN1SD1Ev(%{{.*}}) {{.*}} : (!cir.ptr<!rec_S, target_address_space(4)> {{.*}}) -> ()

#endif
