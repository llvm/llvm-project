// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -cl-std=CL2.0 -O0 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -cl-std=CL2.0 -O0 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -cl-std=CL2.0 -O0 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

void private_array(int i) {
  int a[4];
  a[i] = 1;
}

// CIR-LABEL: cir.func {{.*}} @private_array(
// CIR:         %[[A:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!cir.array<!s32i x 4>, target_address_space(5)>
// CIR:         %[[IDX:.*]] = cir.cast integral %{{.+}} : !s32i -> !s64i
// CIR-NEXT:    cir.get_element inbounds %[[A]][%[[IDX]] : !s64i] : !cir.ptr<!cir.array<!s32i x 4>, target_address_space(5)> -> !cir.ptr<!s32i, target_address_space(5)>

// LLVM-LABEL: define {{.*}} void @private_array(i32 noundef %{{.+}})
// LLVM:         %[[A:.*]] = alloca [4 x i32], align 4, addrspace(5)
// LLVM:         %[[SEXT:.*]] = sext i32 %{{.+}} to i64
// LLVMCIR-NEXT: %[[TRUNC:.*]] = trunc i64 %[[SEXT]] to i32
// LLVMCIR-NEXT: getelementptr inbounds [4 x i32], ptr addrspace(5) %[[A]], i32 0, i32 %[[TRUNC]]
// OGCG-NEXT:    getelementptr inbounds [4 x i32], ptr addrspace(5) %[[A]], i64 0, i64 %[[SEXT]]

int local_array(local int (*p)[4], int i) {
  return (*p)[i];
}

// CIR-LABEL: cir.func {{.*}} @local_array(
// CIR:         %[[P:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!cir.array<!s32i x 4>, target_address_space(3)>, target_address_space(5)>, !cir.ptr<!cir.array<!s32i x 4>, target_address_space(3)>
// CIR:         %[[IDX:.*]] = cir.cast integral %{{.+}} : !s32i -> !s64i
// CIR-NEXT:    cir.get_element inbounds %[[P]][%[[IDX]] : !s64i] : !cir.ptr<!cir.array<!s32i x 4>, target_address_space(3)> -> !cir.ptr<!s32i, target_address_space(3)>

// LLVM-LABEL: define {{.*}} i32 @local_array(ptr addrspace(3) noundef %{{.+}}, i32 noundef %{{.+}})
// LLVM:         %[[P:.*]] = load ptr addrspace(3), ptr addrspace(5) %{{.+}}, align 4
// LLVM:         %[[SEXT:.*]] = sext i32 %{{.+}} to i64
// LLVMCIR-NEXT: %[[TRUNC:.*]] = trunc i64 %[[SEXT]] to i32
// LLVMCIR-NEXT: getelementptr inbounds [4 x i32], ptr addrspace(3) %[[P]], i32 0, i32 %[[TRUNC]]
// OGCG-NEXT:    getelementptr inbounds [4 x i32], ptr addrspace(3) %[[P]], i64 0, i64 %[[SEXT]]
