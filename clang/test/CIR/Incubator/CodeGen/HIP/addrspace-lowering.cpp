#include "cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api -emit-cir \
// RUN:            -I%S/../Inputs/ %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip  \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s



// ------------------------------------------------------------
//  CHECK POINTER ARGUMENT LOWERING (bitcast or addrspacecast)
// ------------------------------------------------------------

__shared__ int a;
// LLVM-DEVICE: @a = addrspace(3) global i32 undef, align 4
// OGCG-DEVICE: @a = addrspace(3) global i32 undef, align 4

__device__ int b;
// LLVM-DEVICE: @b = addrspace(1) externally_initialized global i32 0, align 4
// OGCG-DEVICE: @b = addrspace(1) externally_initialized global i32 0, align 4

__constant__ int c;
// LLVM-DEVICE: @c = addrspace(4) externally_initialized constant i32 0, align 4
// OGCG-DEVICE: @c = addrspace(4) externally_initialized constant i32 0, align 4

// Forward decls in various address spaces.
extern "C" __device__ void bar(const char *p);
extern "C" __device__ void takes_global(float *p);
extern "C" __device__ void takes_shared(int *p);
extern "C" __device__ void takes_void(void *p);
extern "C" __device__ void nullfun(int *p);
extern "C" __device__ void takeS(struct S s);
extern "C" __device__ void call_fp(void (*f)(int));

__constant__ int CC[12];
__device__ float GArr[8];
__device__ void fp_target(int);

// A struct that contains a pointer
struct S { int *p; };

// ------------------------------------------------------------
// 1. local → generic: expected bitcast or AS0 match
// ------------------------------------------------------------
__device__ void test_local() {
  int x = 42;
  bar((const char*)&x);
}
// CIR-DEVICE-LABEL: @_Z10test_localv
// CIR-DEVICE: cir.alloca
// CIR-DEVICE: cir.store
// CIR-DEVICE: cir.cast bitcast {{.*}} -> !cir.ptr<!s8i>
// CIR-DEVICE: cir.call @bar
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z10test_localv
// LLVM-DEVICE: alloca i32
// LLVM-DEVICE: addrspacecast ptr addrspace(5) {{.*}} to ptr
// LLVM-DEVICE: store i32 42
// LLVM-DEVICE: call void @bar(ptr {{.*}})
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z10test_localv
// OGCG-DEVICE: alloca i32, align 4, addrspace(5)
// OGCG-DEVICE: addrspacecast ptr addrspace(5) {{.*}} to ptr
// OGCG-DEVICE: store i32 42
// OGCG-DEVICE: call void @bar(ptr noundef {{.*}})
// OGCG-DEVICE: ret void

// ------------------------------------------------------------
// 2. global AS → generic param
// ------------------------------------------------------------
__device__ void test_global() {
  takes_global(GArr);
}
// CIR-DEVICE-LABEL: @_Z11test_globalv
// CIR-DEVICE: cir.get_global @GArr
// CIR-DEVICE: cir.cast array_to_ptrdecay
// CIR-DEVICE: cir.cast address_space
// CIR-DEVICE: cir.call @takes_global
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z11test_globalv
// LLVM-DEVICE: call void @takes_global(ptr addrspacecast (ptr addrspace(1) @GArr to ptr))
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z11test_globalv
// OGCG-DEVICE: call void @takes_global(ptr noundef addrspacecast (ptr addrspace(1) @GArr to ptr))
// OGCG-DEVICE: ret void

// ------------------------------------------------------------
// 3. shared AS(3) → generic param (requires addrspacecast)
// ------------------------------------------------------------
__device__ void test_shared() {
  __shared__ int s[2];
  takes_shared(s);
}
// CIR-DEVICE-LABEL: @_Z11test_sharedv
// CIR-DEVICE: cir.get_global @_ZZ11test_sharedvE1s
// CIR-DEVICE: cir.cast array_to_ptrdecay
// CIR-DEVICE: cir.cast address_space
// CIR-DEVICE: cir.call @takes_shared
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z11test_sharedv
// LLVM-DEVICE: call void @takes_shared(ptr addrspacecast (ptr addrspace(3) @_ZZ11test_sharedvE1s to ptr))
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z11test_sharedv
// OGCG-DEVICE: call void @takes_shared(ptr noundef addrspacecast (ptr addrspace(3) @_ZZ11test_sharedvE1s to ptr))
// OGCG-DEVICE: ret void

// ------------------------------------------------------------
// 4. mismatched pointee types but same AS: bitcast only
// ------------------------------------------------------------
__device__ void test_void_bitcast() {
  int x = 7;
  takes_void((void*)&x);
}
// CIR-DEVICE-LABEL: @_Z17test_void_bitcastv
// CIR-DEVICE: cir.alloca
// CIR-DEVICE: cir.store
// CIR-DEVICE: cir.cast bitcast {{.*}} -> !cir.ptr<!void>
// CIR-DEVICE: cir.call @takes_void
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z17test_void_bitcastv
// LLVM-DEVICE: alloca i32
// LLVM-DEVICE: addrspacecast ptr addrspace(5) {{.*}} to ptr
// LLVM-DEVICE: store i32 7
// LLVM-DEVICE: call void @takes_void(ptr {{.*}})
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z17test_void_bitcastv
// OGCG-DEVICE: alloca i32, align 4, addrspace(5)
// OGCG-DEVICE: addrspacecast ptr addrspace(5) {{.*}} to ptr
// OGCG-DEVICE: store i32 7
// OGCG-DEVICE: call void @takes_void(ptr noundef {{.*}})
// OGCG-DEVICE: ret void

// ------------------------------------------------------------
// 5. nullptr: ensure correct null pointer cast is emitted
// ------------------------------------------------------------
__device__ void test_null() {
  nullfun(nullptr);
}
// CIR-DEVICE-LABEL: @_Z9test_nullv
// CIR-DEVICE: cir.const #cir.ptr<null>
// CIR-DEVICE: cir.call @nullfun
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z9test_nullv
// LLVM-DEVICE: call void @nullfun(ptr null)
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z9test_nullv
// OGCG-DEVICE: call void @nullfun(ptr noundef null)
// OGCG-DEVICE: ret void

// ------------------------------------------------------------
// 6. Struct containing a pointer
// ------------------------------------------------------------
__device__ void test_struct() {
  int x = 5;
  S s{&x};
  takeS(s);
}
// CIR-DEVICE-LABEL: @_Z11test_structv
// CIR-DEVICE: cir.alloca !s32i
// CIR-DEVICE: cir.alloca !rec_S
// CIR-DEVICE: cir.get_member {{.*}} "p"
// CIR-DEVICE: cir.store {{.*}} : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-DEVICE: cir.copy
// CIR-DEVICE: cir.call @takeS
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z11test_structv
// LLVM-DEVICE: alloca i32
// LLVM-DEVICE: alloca %struct.S
// LLVM-DEVICE: getelementptr %struct.S
// LLVM-DEVICE: store ptr {{.*}}, ptr {{.*}}
// LLVM-DEVICE: call void @llvm.memcpy
// LLVM-DEVICE: load %struct.S
// LLVM-DEVICE: call void @takeS(%struct.S {{.*}})
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z11test_structv
// OGCG-DEVICE: alloca i32, align 4, addrspace(5)
// OGCG-DEVICE: alloca %struct.S, align 8, addrspace(5)
// OGCG-DEVICE: alloca %struct.S, align 8, addrspace(5)
// OGCG-DEVICE: addrspacecast ptr addrspace(5) {{.*}} to ptr
// OGCG-DEVICE: store i32 5
// OGCG-DEVICE: getelementptr inbounds nuw %struct.S
// OGCG-DEVICE: store ptr {{.*}}, ptr {{.*}}
// OGCG-DEVICE: call void @llvm.memcpy.p0.p0.i64
// OGCG-DEVICE: load ptr
// OGCG-DEVICE: call void @takeS(ptr {{.*}})
// OGCG-DEVICE: ret void

// ------------------------------------------------------------
// 7. Function pointer argument
// ------------------------------------------------------------
__device__ void test_fp() {
  call_fp(fp_target);
}
// CIR-DEVICE-LABEL: @_Z7test_fpv
// CIR-DEVICE: cir.get_global @_Z9fp_targeti
// CIR-DEVICE: cir.call @call_fp
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z7test_fpv
// LLVM-DEVICE: call void @call_fp(ptr @_Z9fp_targeti)
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z7test_fpv
// OGCG-DEVICE: call void @call_fp(ptr noundef @_Z9fp_targeti)
// OGCG-DEVICE: ret void

// ------------------------------------------------------------
// 8. Original test from previous patch: string literal → char*
// ------------------------------------------------------------
__device__ void foo() {
  char cchar[] = "const char.\n";
  bar(cchar);
}
// CIR-DEVICE-LABEL: @_Z3foov
// CIR-DEVICE: cir.alloca !cir.array<!s8i x 13>, !cir.ptr<!cir.array<!s8i x 13>, lang_address_space(offload_private)>
// CIR-DEVICE: cir.cast address_space
// CIR-DEVICE: cir.get_global @__const._Z3foov
// CIR-DEVICE: cir.copy
// CIR-DEVICE: cir.cast array_to_ptrdecay
// CIR-DEVICE: cir.call @bar
// CIR-DEVICE: cir.return

// LLVM-DEVICE-LABEL: @_Z3foov
// LLVM-DEVICE: alloca [13 x i8]
// LLVM-DEVICE: addrspacecast ptr addrspace(5) {{.*}} to ptr
// LLVM-DEVICE: call void @llvm.memcpy.p0.p0.i32
// LLVM-DEVICE: getelementptr i8
// LLVM-DEVICE: call void @bar(ptr {{.*}})
// LLVM-DEVICE: ret void

// OGCG-DEVICE-LABEL: @_Z3foov
// OGCG-DEVICE: alloca [13 x i8], align 1, addrspace(5)
// OGCG-DEVICE: addrspacecast ptr addrspace(5) {{.*}} to ptr
// OGCG-DEVICE: call void @llvm.memcpy.p0.p4.i64
// OGCG-DEVICE: getelementptr inbounds [13 x i8]
// OGCG-DEVICE: call void @bar(ptr noundef {{.*}})
// OGCG-DEVICE: ret void
