// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// CIR: cir.func {{@.*foo.*}}(%arg0: !cir.ptr<!s32i, addrspace(target<1>)>
// LLVM: define dso_local void @foo(ptr addrspace(1) %0)
void foo(int __attribute__((address_space(1))) *arg) {
  return;
}

// CIR: cir.func {{@.*bar.*}}(%arg0: !cir.ptr<!s32i, addrspace(target<0>)>
// LLVM: define dso_local void @bar(ptr %0)
void bar(int __attribute__((address_space(0))) *arg) {
  return;
}

// CIR: cir.func {{@.*baz.*}}(%arg0: !cir.ptr<!s32i>
// LLVM: define dso_local void @baz(ptr %0)
void baz(int *arg) {
  return;
}
