// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

export void f() {}

// CIR: cir.func {{.*}} @_Z1fv() {{.*}} attributes {{.*}}norecurse

// LLVM: define {{.*}} @_Z1fv() #[[ATTR:[0-9]+]]
// LLVM: attributes #[[ATTR]] = {{.*}}norecurse
