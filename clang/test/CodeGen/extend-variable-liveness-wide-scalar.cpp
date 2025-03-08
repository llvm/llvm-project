// RUN: %clang_cc1 %s -emit-llvm -fextend-variable-liveness -triple x86_64-unknown-linux -o - | FileCheck %s
// REQUIRES: x86-registered-target
// This test checks that the fake uses can be generated in exception handling
// blocks and that we can emit fake uses for the __int128 data type.

void bar();

// CHECK: call void (...) @llvm.fake.use(i128 %
void foo(__int128 wide_int) {
  bar();
}
