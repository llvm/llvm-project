// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: define internal{{.*}} void @__cxx_global_var_init()
// CHECK: [[entry_token:%.*]] = call token @llvm.experimental.convergence.entry()
// CHECK: br label %[[loop_entry:.*]]

// CHECK: [[loop_entry]]:
// CHECK: [[loop_token:%.*]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[entry_token]]) ]
// CHECK: call{{.*}} void {{.*}} [ "convergencectrl"(token [[loop_token]]) ]
// CHECK: br i1 {{%.*}} label {{%.*}} label %[[loop_entry]]

struct S {
    int i;
    S() { i = 10; }
};

static S s[2];

[numthreads(4,1,1)]
void main() {
}

