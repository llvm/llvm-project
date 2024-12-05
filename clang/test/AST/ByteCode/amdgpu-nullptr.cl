// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -triple amdgcn -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-enable-noundef-analysis %s -cl-std=CL2.0 -triple amdgcn -emit-llvm -fexperimental-new-constant-interpreter -o - | FileCheck %s


// CHECK: @fold_priv ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(5) addrspacecast (ptr addrspace(1) null to ptr addrspace(5)), align 4
private short *fold_priv = (private short*)(generic int*)(global void*)0;

// CHECK: @fold_priv_arith ={{.*}} local_unnamed_addr addrspace(1) global ptr addrspace(5) inttoptr (i32 9 to ptr addrspace(5)), align 4
private char *fold_priv_arith = (private char*)0 + 10;


