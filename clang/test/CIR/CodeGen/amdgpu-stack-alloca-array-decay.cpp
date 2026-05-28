// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void foo() {
  const char fmt[] = "hello";
  const char *tmp = fmt;
  while (*tmp++)
    ;
}

// CIR-LABEL: cir.func{{.*}} @_Z3foov()
// CIR:         %[[FMT_RAW:.*]] = cir.alloca "fmt" {{.*}} !cir.array<!s8i x 6> -> !cir.ptr<!cir.array<!s8i x 6>, target_address_space(5)>
// CIR-NEXT:    %[[TMP_RAW:.*]] = cir.alloca "tmp" {{.*}} !cir.ptr<!s8i> -> !cir.ptr<!cir.ptr<!s8i>, target_address_space(5)>
// CIR-DAG:     %[[FMT:.*]] = cir.cast address_space %[[FMT_RAW]] : !cir.ptr<!cir.array<!s8i x 6>, target_address_space(5)> -> !cir.ptr<!cir.array<!s8i x 6>>
// CIR-DAG:     %[[TMP:.*]] = cir.cast address_space %[[TMP_RAW]] : !cir.ptr<!cir.ptr<!s8i>, target_address_space(5)> -> !cir.ptr<!cir.ptr<!s8i>>
// CIR:         %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[FMT]] : !cir.ptr<!cir.array<!s8i x 6>> -> !cir.ptr<!s8i>
// CIR:         cir.store{{.*}} %[[DECAY]], %[[TMP]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>

// LLVM-LABEL: define{{.*}} void @_Z3foov()
// LLVM:         %[[FMT:.*]] = alloca [6 x i8], i64 1, align 1, addrspace(5)
// LLVM-NEXT:    %[[TMP:.*]] = alloca ptr, i64 1, align 8, addrspace(5)
// LLVM-DAG:     %[[FMT_CAST:.*]] = addrspacecast ptr addrspace(5) %[[FMT]] to ptr
// LLVM-DAG:     %[[TMP_CAST:.*]] = addrspacecast ptr addrspace(5) %[[TMP]] to ptr
// LLVM:         %[[DECAY:.*]] = getelementptr {{.*}} ptr %[[FMT_CAST]],
// LLVM:         store ptr %[[DECAY]], ptr %[[TMP_CAST]], align 8

// OGCG-LABEL: define{{.*}} void @_Z3foov()
// OGCG:         %[[FMT:.*]] = alloca [6 x i8], align 1, addrspace(5)
// OGCG-NEXT:    %[[TMP:.*]] = alloca ptr, align 8, addrspace(5)
// OGCG-NEXT:    %[[FMT_CAST:.*]] = addrspacecast ptr addrspace(5) %[[FMT]] to ptr
// OGCG-NEXT:    %[[TMP_CAST:.*]] = addrspacecast ptr addrspace(5) %[[TMP]] to ptr
// OGCG:         %[[DECAY:.*]] = getelementptr inbounds [6 x i8], ptr %[[FMT_CAST]], i64 0, i64 0
// OGCG:         store ptr %[[DECAY]], ptr %[[TMP_CAST]], align 8
