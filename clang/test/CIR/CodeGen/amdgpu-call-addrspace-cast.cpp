// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test that address space casts are inserted when passing pointers in
// non-default address spaces (global AS 1, private AS 5 on AMDGPU) to
// functions expecting flat (generic) pointers.

int globalVar = 42;

void takes_ptr(int *p);

// CIR-LABEL: cir.func{{.*}} @_Z20call_with_global_ptrv()
// CIR:         %[[GPTR:.*]] = cir.get_global @globalVar : !cir.ptr<!s32i, target_address_space(1)>
// CIR-NEXT:    %[[CAST:.*]] = cir.cast address_space %[[GPTR]] : !cir.ptr<!s32i, target_address_space(1)> -> !cir.ptr<!s32i>
// CIR-NEXT:    cir.call @_Z9takes_ptrPi(%[[CAST]])

// LLVM-LABEL: define{{.*}} void @_Z20call_with_global_ptrv()
// LLVM:         call void @_Z9takes_ptrPi(ptr noundef addrspacecast (ptr addrspace(1) @globalVar to ptr))

// OGCG-LABEL: define{{.*}} void @_Z20call_with_global_ptrv()
// OGCG:         call void @_Z9takes_ptrPi(ptr noundef addrspacecast (ptr addrspace(1) @globalVar to ptr))
void call_with_global_ptr() {
  takes_ptr(&globalVar);
}

// CIR-LABEL: cir.func{{.*}} @_Z19call_with_local_ptrv()
// CIR:         %[[ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i, target_address_space(5)>
// CIR:         %[[CAST:.*]] = cir.cast address_space %[[ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-NEXT:    cir.call @_Z9takes_ptrPi(%[[CAST]])

// LLVM-LABEL: define{{.*}} void @_Z19call_with_local_ptrv()
// LLVM:         %[[ALLOCA:.*]] = alloca i32, i64 1, align 4, addrspace(5)
// LLVM:         %[[CAST:.*]] = addrspacecast ptr addrspace(5) %[[ALLOCA]] to ptr
// LLVM-NEXT:    call void @_Z9takes_ptrPi(ptr noundef %[[CAST]])

// OGCG-LABEL: define{{.*}} void @_Z19call_with_local_ptrv()
// OGCG:         %[[ALLOCA:.*]] = alloca i32, align 4, addrspace(5)
// OGCG-NEXT:    %[[CAST:.*]] = addrspacecast ptr addrspace(5) %[[ALLOCA]] to ptr
// OGCG:         call void @_Z9takes_ptrPi(ptr noundef %[[CAST]])
void call_with_local_ptr() {
  int x = 1;
  takes_ptr(&x);
}
