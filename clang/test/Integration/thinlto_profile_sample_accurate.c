// Test to ensure -emit-llvm profile-sample-accurate is honored in ThinLTO.
// RUN: %clang -O2 %s -flto=thin -fprofile-sample-accurate -c -o %t.o
// RUN: llvm-lto -thinlto -o %t %t.o
// RUN: %clang_cc1 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -emit-llvm -o - | FileCheck %s

// CHECK: define{{.*}} void @foo()
// CHECK: attributes{{.*}} "profile-sample-accurate"
void foo(void) {
}
