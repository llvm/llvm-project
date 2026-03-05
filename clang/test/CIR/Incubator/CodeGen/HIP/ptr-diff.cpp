#include "cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-cir %s -o %t.ll
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip  \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../Inputs/ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

__device__ int ptr_diff() {
  const char c_str[] = "c-string"; 
  const char* len =  c_str;  
  return c_str - len;
}


// CIR-DEVICE: %[[#LenLocalAlloca:]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>, lang_address_space(offload_private)>, ["len", init]
// CIR-DEVICE: %[[#LenLocalAddr:]] = cir.cast address_space %[[#LenLocalAlloca]] : !cir.ptr<!cir.ptr<!s8i>, lang_address_space(offload_private)> -> !cir.ptr<!cir.ptr<!s8i>>
// CIR-DEVICE: %[[#GlobalPtr:]] = cir.get_global @_ZZ8ptr_diffvE5c_str : !cir.ptr<!cir.array<!s8i x 9>, lang_address_space(offload_constant)>
// CIR-DEVICE: %[[#CastDecay:]] = cir.cast array_to_ptrdecay %[[#GlobalPtr]] : !cir.ptr<!cir.array<!s8i x 9>, lang_address_space(offload_constant)>
// CIR-DEVICE: %[[#LenLocalAddrCast:]] = cir.cast bitcast %[[#LenLocalAddr]] : !cir.ptr<!cir.ptr<!s8i>> -> !cir.ptr<!cir.ptr<!s8i, lang_address_space(offload_constant)>>
// CIR-DEVICE: cir.store align(8) %[[#CastDecay]], %[[#LenLocalAddrCast]] : !cir.ptr<!s8i, lang_address_space(offload_constant)>, !cir.ptr<!cir.ptr<!s8i, lang_address_space(offload_constant)>>
// CIR-DEVICE: %[[#CStr:]] = cir.cast array_to_ptrdecay %[[#GlobalPtr]] : !cir.ptr<!cir.array<!s8i x 9>, lang_address_space(offload_constant)> -> !cir.ptr<!s8i, lang_address_space(offload_constant)>
// CIR-DEVICE: %[[#LoadedLenAddr:]] = cir.load align(8) %[[#LenLocalAddr]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CIR-DEVICE: %[[#AddrCast:]] = cir.cast address_space %[[#LoadedLenAddr]] : !cir.ptr<!s8i> -> !cir.ptr<!s8i, lang_address_space(offload_constant)>
// CIR-DEVICE: %[[#DIFF:]] = cir.ptr_diff %[[#CStr]], %[[#AddrCast]] : !cir.ptr<!s8i, lang_address_space(offload_constant)>

// LLVM-DEVICE: define dso_local i32 @_Z8ptr_diffv()
// LLVM-DEVICE: %[[#RetvalAddr:]] = alloca i32, i64 1, align 4, addrspace(5)
// LLVM-DEVICE: %[[#LenLocalAddr:]] = alloca ptr, i64 1, align 8, addrspace(5)
// LLVM-DEVICE: %[[#LenLocalAddrCast:]] = addrspacecast ptr addrspace(5) %[[#LenLocalAddr]] to ptr
// LLVM-DEVICE: store ptr addrspace(4) @_ZZ8ptr_diffvE5c_str, ptr %[[#LenLocalAddrCast]], align 8
// LLVM-DEVICE: %[[#LoadedAddr:]] = load ptr, ptr %[[#LenLocalAddrCast]], align 8
// LLVM-DEVICE: %[[#CastedVal:]] = addrspacecast ptr %[[#LoadedAddr]] to ptr addrspace(4)
// LLVM-DEVICE: %[[#IntVal:]] = ptrtoint ptr addrspace(4) %[[#CastedVal]] to i64
// LLVM-DEVICE: %[[#SubVal:]] = sub i64 ptrtoint (ptr addrspace(4) @_ZZ8ptr_diffvE5c_str to i64), %[[#IntVal]]

// OGCG-DEVICE: define dso_local noundef i32 @_Z8ptr_diffv() #0
// OGCG-DEVICE: %[[RETVAL:.*]] = alloca i32, align 4, addrspace(5)
// OGCG-DEVICE: %[[C_STR:.*]] = alloca [9 x i8], align 1, addrspace(5)
// OGCG-DEVICE: %[[LEN:.*]] = alloca ptr, align 8, addrspace(5)
// OGCG-DEVICE: %[[RETVAL_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[RETVAL]] to ptr
// OGCG-DEVICE: %[[C_STR_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[C_STR]] to ptr
// OGCG-DEVICE: %[[LEN_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[LEN]] to ptr
// OGCG-DEVICE: %[[ARRAYDECAY:.*]] = getelementptr inbounds [9 x i8], ptr %[[C_STR_ASCAST]], i64 0, i64 0
// OGCG-DEVICE: store ptr %[[ARRAYDECAY]], ptr %[[LEN_ASCAST]], align 8
// OGCG-DEVICE: %[[ARRAYDECAY1:.*]] = getelementptr inbounds [9 x i8], ptr %[[C_STR_ASCAST]], i64 0, i64 0
// OGCG-DEVICE: %[[LOADED:.*]] = load ptr, ptr %[[LEN_ASCAST]], align 8
// OGCG-DEVICE: %[[LHS:.*]] = ptrtoint ptr %[[ARRAYDECAY1]] to i64
// OGCG-DEVICE: %[[RHS:.*]] = ptrtoint ptr %[[LOADED]] to i64
// OGCG-DEVICE: %[[SUB:.*]] = sub i64 %[[LHS]], %[[RHS]]
// OGCG-DEVICE: %[[CONV:.*]] = trunc i64 %[[SUB]] to i32
