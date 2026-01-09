// RUN: %clang_cc1 -triple=x86_64-linux-gnu -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s

// CHECK: define{{.*}} void @g() [[FLATTEN_ATTR:#[0-9]+]]
__attribute__((flatten))
void g(void) {
}

// CHECK: attributes [[FLATTEN_ATTR]] = {{{.*}}flatten{{.*}}}
