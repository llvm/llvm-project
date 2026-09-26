// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test that address spaces are preserved through array-to-pointer decay
// and array element access on AMDGPU, where globals are in AS 1.

int globalArr[10] = {0};

// A dynamic initializer stores through the flat address of the global, as in
// classic CodeGen.

int f();
int dyn = f();

// CIR:       cir.func {{.*}}@__cxx_global_var_init
// CIR:         %[[DYN:.*]] = cir.get_global @dyn : !cir.ptr<!s32i, target_address_space(1)>
// CIR-NEXT:    %[[FLAT:.*]] = cir.cast address_space %[[DYN]] : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i>
// CIR:         cir.store align(4) %{{.*}}, %[[FLAT]] : !s32i, !cir.ptr<!s32i>

// LLVM:        store i32 %{{.*}}, ptr addrspacecast (ptr addrspace(1) @dyn to ptr), align 4
// OGCG:        store i32 %{{.*}}, ptr addrspacecast (ptr addrspace(1) @dyn to ptr), align 4

void takes_ptr(int *p);

// The address of the global is cast to the declared (flat) address space
// before the array decays.

// CIR-LABEL: cir.func{{.*}} @_Z17pass_global_arrayv()
// CIR:         %[[ARR:.*]] = cir.get_global @globalArr : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)>
// CIR-NEXT:    %[[FLAT:.*]] = cir.cast address_space %[[ARR]] : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)> -> !cir.ptr<!cir.array<!s32i x 10>>
// CIR-NEXT:    %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[FLAT]] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!s32i>
// CIR-NEXT:    cir.call @_Z9takes_ptrPi(%[[DECAY]])

// LLVM-LABEL: define{{.*}} void @_Z17pass_global_arrayv()
// LLVM:         call void @_Z9takes_ptrPi(ptr noundef addrspacecast (ptr addrspace(1) @globalArr to ptr))

// OGCG-LABEL: define{{.*}} void @_Z17pass_global_arrayv()
// OGCG:         call void @_Z9takes_ptrPi(ptr noundef addrspacecast (ptr addrspace(1) @globalArr to ptr))
void pass_global_array() {
  takes_ptr(globalArr);
}

// Indexing goes through the flat address, as in classic CodeGen.

// CIR-LABEL: cir.func{{.*}} @_Z18index_global_arrayi
// CIR:         %[[ARR:.*]] = cir.get_global @globalArr : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)>
// CIR-NEXT:    %[[FLAT:.*]] = cir.cast address_space %[[ARR]] : !cir.ptr<!cir.array<!s32i x 10>, target_address_space(1)> -> !cir.ptr<!cir.array<!s32i x 10>>
// CIR-NEXT:    %[[ELEM:.*]] = cir.get_element %[[FLAT]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!s32i>
// CIR-NEXT:    %{{.*}} = cir.load align(4) %[[ELEM]] : !cir.ptr<!s32i>, !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z18index_global_arrayi
// LLVM:         %[[GEP:.*]] = getelementptr [10 x i32], ptr addrspacecast (ptr addrspace(1) @globalArr to ptr), i32 0, i64 %{{.*}}
// LLVM-NEXT:    %{{.*}} = load i32, ptr %[[GEP]], align 4

// OGCG-LABEL: define{{.*}} i32 @_Z18index_global_arrayi
// OGCG:         getelementptr inbounds [10 x i32], ptr addrspacecast (ptr addrspace(1) @globalArr to ptr)
// OGCG:         load i32, ptr %{{.*}}, align 4
int index_global_array(int i) {
  return globalArr[i];
}

