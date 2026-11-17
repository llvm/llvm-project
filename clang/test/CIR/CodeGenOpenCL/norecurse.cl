// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-llvm -O0 %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm -O0 %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

kernel void kernel1(int a) {}

// CIR: cir.func{{.*}} @kernel1({{.*}} attributes {{.*}}norecurse

// LLVM: define {{.*}} @kernel1({{.*}}) #[[ATTR:[0-9]+]]
// LLVM: attributes #[[ATTR]] = {{.*}}norecurse
