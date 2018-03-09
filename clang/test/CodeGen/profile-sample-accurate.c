// Test to ensure -emit-llvm profile-sample-accurate is honored by clang.
// RUN: %clang -S -emit-llvm %s -fprofile-sample-accurate -o - | FileCheck %s

// CHECK: define{{.*}} void @foo()
// CHECK: attributes{{.*}} "profile-sample-accurate"
void foo() {
}
