// REQUIRES: amdgpu-registered-target
// Test without serialization:
// RUN: %clang_cc1 -triple amdgcn -ast-dump -ast-dump-filter __amdgpu_buffer_rsrc_t %s | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple amdgcn -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple amdgcn -include-pch %t -ast-dump-all -ast-dump-filter __amdgpu_buffer_rsrc_t /dev/null | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" | FileCheck %s

// CHECK: TypedefDecl {{.*}} implicit __amdgpu_buffer_rsrc_t
// CHECK-NEXT: -BuiltinType {{.*}} '__amdgpu_buffer_rsrc_t'
