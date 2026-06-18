// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

extern enum X x;
void f(void) {
  x;
}

enum X {
  One,
  Two
};

// CIR: cir.global "private" external @x : !u32i
// CIR: cir.func{{.*}} @f
// CIR:   cir.get_global @x : !cir.ptr<!u32i>

// LLVM: @x = external global i32
// LLVM: define {{.*}}void @f()

// OGCG: @x = external global i32
// OGCG: define {{.*}}void @f()
