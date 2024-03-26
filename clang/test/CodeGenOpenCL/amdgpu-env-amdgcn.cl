// RUN: %clang_cc1 %s -O0 -triple amdgcn -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -O0 -triple amdgcn---opencl -emit-llvm -o - | FileCheck %s

// CHECK: target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
void foo(void) {}
