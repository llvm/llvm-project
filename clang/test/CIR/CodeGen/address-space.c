// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test address space 1
// CIR: cir.func dso_local @foo(%arg0: !cir.ptr<!s32i, target_address_space(1)>
// LLVM: define dso_local void @foo(ptr addrspace(1) %0)
// OGCG: define dso_local void @foo(ptr addrspace(1) noundef %arg)
void foo(int __attribute__((address_space(1))) *arg) {
  return;
}

// Test explicit address space 0 (should be same as default)
// CIR: cir.func dso_local @bar(%arg0: !cir.ptr<!s32i, target_address_space(0)>
// LLVM: define dso_local void @bar(ptr %0)
// OGCG: define dso_local void @bar(ptr noundef %arg)
void bar(int __attribute__((address_space(0))) *arg) {
  return;
}

// Test default address space (no attribute)
// CIR: cir.func dso_local @baz(%arg0: !cir.ptr<!s32i>
// LLVM: define dso_local void @baz(ptr %0)
// OGCG: define dso_local void @baz(ptr noundef %arg)
void baz(int *arg) {
  return;
}
