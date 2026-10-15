// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x cl -triple spirv64-unknown-unknown -cl-std=CL2.0 -O0 -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR-SPV %s \
// RUN:   --implicit-check-not='cir.cast address_space'
// RUN: %clang_cc1 -x cl -triple spirv64-unknown-unknown -cl-std=CL2.0 -O0 -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM-SPV %s
// RUN: %clang_cc1 -x cl -triple spirv64-unknown-unknown -cl-std=CL2.0 -O0 -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=OGCG-SPV %s

// RUN: %clang_cc1 -x cl -triple amdgpu9.00-amd-amdhsa -cl-std=CL2.0 -O0 -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR-GCN %s \
// RUN:   --implicit-check-not='cir.cast address_space'
// RUN: %clang_cc1 -x cl -triple amdgpu9.00-amd-amdhsa -cl-std=CL2.0 -O0 -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM-GCN %s
// RUN: %clang_cc1 -x cl -triple amdgpu9.00-amd-amdhsa -cl-std=CL2.0 -O0 -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=OGCG-GCN %s

// OpenCL string literals are __constant in the AST already, so the global is
// emitted in the constant address space and used without an address space
// cast.

void take(const __constant char *);

// CIR-SPV: cir.global "private" constant cir_private dso_local target_address_space(2) @".str" = #cir.const_array<"hi" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3>
// CIR-SPV: cir.global "private" constant cir_private dso_local target_address_space(2) @".str.1" = #cir.const_array<"abc" : !cir.array<!s8i x 3>, trailing_zeros> : !cir.array<!s8i x 4>
// CIR-SPV: cir.global "private" constant cir_private dso_local target_address_space(2) @__func__.func_name = #cir.const_array<"func_name" : !cir.array<!s8i x 9>, trailing_zeros> : !cir.array<!s8i x 10>

// CIR-GCN: cir.global "private" constant cir_private dso_local target_address_space(4) @".str" = #cir.const_array<"hi" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3>
// CIR-GCN: cir.global "private" constant cir_private dso_local target_address_space(4) @".str.1" = #cir.const_array<"abc" : !cir.array<!s8i x 3>, trailing_zeros> : !cir.array<!s8i x 4>
// CIR-GCN: cir.global "private" constant cir_private dso_local target_address_space(4) @__func__.func_name = #cir.const_array<"func_name" : !cir.array<!s8i x 9>, trailing_zeros> : !cir.array<!s8i x 10>

// LLVM-SPV: @.str = private addrspace(2) constant [3 x i8] c"hi\00"
// LLVM-SPV: @.str.1 = private addrspace(2) constant [4 x i8] c"abc\00"
// LLVM-SPV: @__func__.func_name = private addrspace(2) constant [10 x i8] c"func_name\00"

// OGCG-SPV: @.str = private unnamed_addr addrspace(2) constant [3 x i8] c"hi\00"
// OGCG-SPV: @.str.1 = private unnamed_addr addrspace(2) constant [4 x i8] c"abc\00"
// OGCG-SPV: @__func__.func_name = private unnamed_addr addrspace(2) constant [10 x i8] c"func_name\00"

// LLVM-GCN: @.str = private addrspace(4) constant [3 x i8] c"hi\00"
// LLVM-GCN: @.str.1 = private addrspace(4) constant [4 x i8] c"abc\00"
// LLVM-GCN: @__func__.func_name = private addrspace(4) constant [10 x i8] c"func_name\00"

// OGCG-GCN: @.str = private unnamed_addr addrspace(4) constant [3 x i8] c"hi\00"
// OGCG-GCN: @.str.1 = private unnamed_addr addrspace(4) constant [4 x i8] c"abc\00"
// OGCG-GCN: @__func__.func_name = private unnamed_addr addrspace(4) constant [10 x i8] c"func_name\00"

// CIR-SPV-LABEL: cir.func{{.*}} @call_take
// CIR-SPV: %[[G:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(2)>
// CIR-SPV: %[[D:.*]] = cir.cast array_to_ptrdecay %[[G]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(2)> -> !cir.ptr<!s8i, target_address_space(2)>
// CIR-SPV: cir.call @take(%[[D]])

// CIR-GCN-LABEL: cir.func{{.*}} @call_take
// CIR-GCN: %[[G:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)>
// CIR-GCN: %[[D:.*]] = cir.cast array_to_ptrdecay %[[G]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)> -> !cir.ptr<!s8i, target_address_space(4)>
// CIR-GCN: cir.call @take(%[[D]])

// LLVM-SPV-LABEL: define{{.*}} void @call_take
// LLVM-SPV: call{{.*}} void @take(ptr addrspace(2) noundef @.str)

// OGCG-SPV-LABEL: define{{.*}} void @call_take
// OGCG-SPV: call{{.*}} void @take(ptr addrspace(2) noundef @.str)

// LLVM-GCN-LABEL: define{{.*}} void @call_take
// LLVM-GCN: call void @take(ptr addrspace(4) noundef @.str)

// OGCG-GCN-LABEL: define{{.*}} void @call_take
// OGCG-GCN: call void @take(ptr addrspace(4) noundef @.str)
void call_take(void) { take("hi"); }

// CIR-SPV-LABEL: cir.func{{.*}} @subscript
// CIR-SPV: %[[G:.*]] = cir.get_global @".str.1" : !cir.ptr<!cir.array<!s8i x 4>, target_address_space(2)>
// CIR-SPV: cir.get_element %[[G]][{{.*}}] : !cir.ptr<!cir.array<!s8i x 4>, target_address_space(2)> -> !cir.ptr<!s8i, target_address_space(2)>

// CIR-GCN-LABEL: cir.func{{.*}} @subscript
// CIR-GCN: %[[G:.*]] = cir.get_global @".str.1" : !cir.ptr<!cir.array<!s8i x 4>, target_address_space(4)>
// CIR-GCN: cir.get_element %[[G]][{{.*}}] : !cir.ptr<!cir.array<!s8i x 4>, target_address_space(4)> -> !cir.ptr<!s8i, target_address_space(4)>

// LLVM-SPV-LABEL: define{{.*}} i8 @subscript
// LLVM-SPV: getelementptr {{.*}}[4 x i8], ptr addrspace(2) @.str.1

// OGCG-SPV-LABEL: define{{.*}} i8 @subscript
// OGCG-SPV: getelementptr {{.*}}[4 x i8], ptr addrspace(2) @.str.1

// LLVM-GCN-LABEL: define{{.*}} i8 @subscript
// LLVM-GCN: getelementptr {{.*}}[4 x i8], ptr addrspace(4) @.str.1

// OGCG-GCN-LABEL: define{{.*}} i8 @subscript
// OGCG-GCN: getelementptr {{.*}}[4 x i8], ptr addrspace(4) @.str.1
char subscript(int i) { return "abc"[i]; }

// CIR-SPV-LABEL: cir.func{{.*}} @func_name
// CIR-SPV: %[[G:.*]] = cir.get_global @__func__.func_name : !cir.ptr<!cir.array<!s8i x 10>, target_address_space(2)>
// CIR-SPV: cir.cast array_to_ptrdecay %[[G]]

// CIR-GCN-LABEL: cir.func{{.*}} @func_name
// CIR-GCN: %[[G:.*]] = cir.get_global @__func__.func_name : !cir.ptr<!cir.array<!s8i x 10>, target_address_space(4)>
// CIR-GCN: cir.cast array_to_ptrdecay %[[G]]

// LLVM-SPV-LABEL: define{{.*}} void @func_name
// LLVM-SPV: call{{.*}} void @take(ptr addrspace(2) noundef @__func__.func_name)

// OGCG-SPV-LABEL: define{{.*}} void @func_name
// OGCG-SPV: call{{.*}} void @take(ptr addrspace(2) noundef @__func__.func_name)

// LLVM-GCN-LABEL: define{{.*}} void @func_name
// LLVM-GCN: call void @take(ptr addrspace(4) noundef @__func__.func_name)

// OGCG-GCN-LABEL: define{{.*}} void @func_name
// OGCG-GCN: call void @take(ptr addrspace(4) noundef @__func__.func_name)
void func_name(void) { take(__func__); }
