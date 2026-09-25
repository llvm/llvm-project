// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void test_basic() {
  __builtin_verbose_trap("Category", "Reason");
}

// CIR: cir.func {{.*}}@_Z10test_basicv()
// CIR:   cir.trap
// LLVM: define{{.*}} void @_Z10test_basicv()
// LLVM:   call void @llvm.trap()
// OGCG: define{{.*}} void @_Z10test_basicv()
// OGCG:   call void @llvm.trap()

void test_multiple(bool cond) {
  if (cond) {
    __builtin_verbose_trap("Cat1", "Reason1");
  } else {
    __builtin_verbose_trap("Cat2", "Reason2");
  }
}

// CIR: cir.func {{.*}}@_Z13test_multipleb
// CIR:   cir.trap
// CIR:   cir.trap
// LLVM: define{{.*}} void @_Z13test_multipleb
// LLVM:   call void @llvm.trap()
// LLVM:   call void @llvm.trap()
// OGCG: define{{.*}} void @_Z13test_multipleb
// OGCG:   call void @llvm.trap()
// OGCG:   call void @llvm.trap()
