// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -fspv-enable-maximal-reconvergence -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -hlsl-entry test -fspv-enable-maximal-reconvergence -emit-llvm -o - -O0 %s | FileCheck %s --check-prefixes=CHECK-ENTRY

[numthreads(1,1,1)]
void main() {
// CHECK: define void @main() [[attributeNumber:#[0-9]+]] {
}

// CHECK: attributes [[attributeNumber]] = {{.*}} "enable-maximal-reconvergence"="true" {{.*}}


[numthreads(1,1,1)]
void test() {
// CHECK-ENTRY: define void @test() [[attributeNumber:#[0-9]+]] {
}
    
// CHECK-ENTRY: attributes [[attributeNumber]] = {{.*}} "enable-maximal-reconvergence"="true" {{.*}}
