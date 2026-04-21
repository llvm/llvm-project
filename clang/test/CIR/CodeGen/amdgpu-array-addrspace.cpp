// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test that address spaces are preserved through array-to-pointer decay
// and array element access on AMDGPU, where globals are in AS 1.

int globalArr[10] = {0};

void takes_ptr(int *p);

// The array_to_ptrdecay cast must preserve the address space of the base
// pointer, followed by an address_space cast.

// CIR-LABEL: cir.func{{.*}} @_Z17pass_global_arrayv()
// CIR:         %[[ARR:.*]] = cir.get_global @globalArr : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)>
// CIR-NEXT:    %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(1)>
// CIR-NEXT:    %[[FLAT:.*]] = cir.cast address_space %[[DECAY]] : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i>
// CIR-NEXT:    cir.call @_Z9takes_ptrPi(%[[FLAT]])

// LLVM-LABEL: define{{.*}} void @_Z17pass_global_arrayv()
// LLVM:         call void @_Z9takes_ptrPi(ptr noundef addrspacecast (ptr addrspace(1) @globalArr to ptr))

// OGCG-LABEL: define{{.*}} void @_Z17pass_global_arrayv()
// OGCG:         call void @_Z9takes_ptrPi(ptr noundef addrspacecast (ptr addrspace(1) @globalArr to ptr))
void pass_global_array() {
  takes_ptr(globalArr);
}

// The get_element op must preserve the address space of the base pointer
// so that the subsequent load uses the correct address space.

// CIR-LABEL: cir.func{{.*}} @_Z18index_global_arrayi
// CIR:         %[[ARR:.*]] = cir.get_global @globalArr : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)>
// CIR-NEXT:    %[[ELEM:.*]] = cir.get_element %[[ARR]][%{{.*}} : !s32i] : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)> -> !cir.ptr<!s32i, target_address_space(1)>
// CIR-NEXT:    %{{.*}} = cir.load align(4) %[[ELEM]] : !cir.ptr<!s32i, target_address_space(1)>, !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z18index_global_arrayi
// LLVM:         %[[GEP:.*]] = getelementptr [10 x i32], ptr addrspace(1) @globalArr, i32 0, i64 %{{.*}}
// LLVM-NEXT:    %{{.*}} = load i32, ptr addrspace(1) %[[GEP]], align 4

// OGCG-LABEL: define{{.*}} i32 @_Z18index_global_arrayi
// OGCG:         getelementptr inbounds [10 x i32], ptr addrspacecast (ptr addrspace(1) @globalArr to ptr)
// OGCG:         load i32, ptr %{{.*}}, align 4
int index_global_array(int i) {
  return globalArr[i];
}
