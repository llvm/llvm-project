// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test global variable with address space
// CIR: cir.global external @gvar = #cir.ptr<null> : !cir.ptr<!s32i, target_address_space(1)>
// LLVM: @gvar = global ptr addrspace(1) null
// OGCG: @gvar = global ptr addrspace(1) null
int __attribute__((address_space(1))) *gvar;

// Test address space 1
// CIR: cir.func {{.*}} @foo(%arg0: !cir.ptr<!s32i, target_address_space(1)>
// LLVM: define dso_local void @foo(ptr addrspace(1) noundef %0)
// OGCG: define dso_local void @foo(ptr addrspace(1) noundef %arg)
void foo(int __attribute__((address_space(1))) *arg) {
  return;
}

// Test explicit address space 0 (should be same as default)
// CIR: cir.func {{.*}} @bar(%arg0: !cir.ptr<!s32i, target_address_space(0)>
// LLVM: define dso_local void @bar(ptr noundef %0)
// OGCG: define dso_local void @bar(ptr noundef %arg)
void bar(int __attribute__((address_space(0))) *arg) {
  return;
}

// Test default address space (no attribute)
// CIR: cir.func {{.*}} @baz(%arg0: !cir.ptr<!s32i>
// LLVM: define dso_local void @baz(ptr noundef %0)
// OGCG: define dso_local void @baz(ptr noundef %arg)
void baz(int *arg) {
  return;
}

// End to end function returning pointer to address space global
// CIR: cir.func {{.*}} @get_gvar()
// CIR:   cir.get_global @gvar : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
// LLVM: define dso_local ptr addrspace(1) @get_gvar()
// LLVM:   load ptr addrspace(1), ptr @gvar
// OGCG: define dso_local ptr addrspace(1) @get_gvar()
// OGCG:   load ptr addrspace(1), ptr @gvar
int __attribute__((address_space(1)))* get_gvar() {
  return gvar;
}
