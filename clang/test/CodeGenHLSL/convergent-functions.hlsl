// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

void fn() {
};

// CHECK: define{{.*| }}void {{.*}}fn{{.*}}()
// CHECK-SAME: #[[Attr:[0-9]+]]
// CHECK: attributes #[[Attr]] = { {{[^}]*}}convergent{{[^}]*}} }
