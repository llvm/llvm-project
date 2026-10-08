// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fstrict-vtable-pointers -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-STRICT --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-plain.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t-plain.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fstrict-vtable-pointers -disable-llvm-passes -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fstrict-vtable-pointers -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct A {
  virtual void f(char);
};

void f1(A *a) {
  a->f('c');
}

// Under -O1 -fstrict-vtable-pointers the virtual function pointer load is
// marked invariant; the vptr load preceding it is not.
// CIR-STRICT-LABEL: cir.func{{.*}}@_Z2f1P1A
// CIR-STRICT:   %[[VPTR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-STRICT:   %[[FN_PTR_PTR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][0]
// CIR-STRICT:   %[[FN_PTR:.*]] = cir.load invariant {{.*}} %[[FN_PTR_PTR]]

// Without the flags the same load must not be invariant.
// CIR-LABEL: cir.func{{.*}}@_Z2f1P1A
// CIR-NOT: cir.load invariant

// LLVM-LABEL: define {{.*}}@_Z2f1P1A
// LLVM: load ptr, ptr %{{.*}}!invariant.load ![[INV:[0-9]+]]
// LLVM: ![[INV]] = !{}
