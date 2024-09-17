// REQUIRES: amdgpu-registered-target
// Test without serialization:
// RUN: %clang_cc1 -triple amdgcn -ast-dump -ast-dump-filter __amdgpu_buffer_rsrc_t %s | FileCheck %s -check-prefix=BUFFER-RSRC
// RUN: %clang_cc1 -triple amdgcn -ast-dump -ast-dump-filter __amdgpu_named_workgroup_barrier %s | FileCheck %s -check-prefix=WORKGROUP-BARRIER
// RUN: %clang_cc1 -triple amdgcn -ast-dump -ast-dump-filter __amdgpu_named_cluster_barrier %s | FileCheck %s -check-prefix=CLUSTER-BARRIER
// RUN: %clang_cc1 -triple amdgcn -ast-dump -ast-dump-filter __amdgpu_semaphore %s | FileCheck %s -check-prefix=SEMAPHORE
//
// Test with serialization:
// RUN: %clang_cc1 -triple amdgcn -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple amdgcn -include-pch %t -ast-dump-all -ast-dump-filter __amdgpu_buffer_rsrc_t /dev/null | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" | FileCheck %s -check-prefix=BUFFER-RSRC
// RUN: %clang_cc1 -x c -triple amdgcn -include-pch %t -ast-dump-all -ast-dump-filter __amdgpu_named_workgroup_barrier /dev/null | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" | FileCheck %s -check-prefix=WORKGROUP-BARRIER
// RUN: %clang_cc1 -x c -triple amdgcn -include-pch %t -ast-dump-all -ast-dump-filter __amdgpu_named_cluster_barrier /dev/null | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" | FileCheck %s -check-prefix=CLUSTER-BARRIER
// RUN: %clang_cc1 -x c -triple amdgcn -include-pch %t -ast-dump-all -ast-dump-filter __amdgpu_semaphore /dev/null | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" | FileCheck %s -check-prefix=SEMAPHORE

// BUFFER-RSRC: TypedefDecl {{.*}} implicit __amdgpu_buffer_rsrc_t
// BUFFER-RSRC-NEXT: -BuiltinType {{.*}} '__amdgpu_buffer_rsrc_t'

// WORKGROUP-BARRIER: TypedefDecl {{.*}} implicit __amdgpu_named_workgroup_barrier_t
// WORKGROUP-BARRIER-NEXT: -BuiltinType {{.*}} '__amdgpu_named_workgroup_barrier_t'

// CLUSTER-BARRIER: TypedefDecl {{.*}} implicit __amdgpu_named_cluster_barrier_t
// CLUSTER-BARRIER-NEXT: -BuiltinType {{.*}} '__amdgpu_named_cluster_barrier_t'

// SEMAPHORE: TypedefDecl {{.*}} implicit __amdgpu_semaphore0_t
// SEMAPHORE-NEXT: -BuiltinType {{.*}} '__amdgpu_semaphore0_t'
// SEMAPHORE: TypedefDecl {{.*}} implicit __amdgpu_semaphore7_t
// SEMAPHORE-NEXT: -BuiltinType {{.*}} '__amdgpu_semaphore7_t'
