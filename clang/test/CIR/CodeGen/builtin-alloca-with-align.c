// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void *with_align(unsigned long n) {
  return __builtin_alloca_with_align(n, 256);
}

// CIR-LABEL: cir.func {{.*}} @with_align(
// CIR:         %{{.+}} = cir.alloca "bi_alloca" align(32) size(%{{.+}}) : !cir.ptr<!u8i>

// LLVM-LABEL: define {{.*}} ptr @with_align(
// LLVM:         alloca i8, i64 %{{.+}}, align 32

// OGCG-LABEL: define {{.*}} ptr @with_align(
// OGCG:         alloca i8, i64 %{{.+}}, align 32

void *with_align_uninitialized(unsigned long n) {
  return __builtin_alloca_with_align_uninitialized(n, 8);
}

// CIR-LABEL: cir.func {{.*}} @with_align_uninitialized(
// CIR:         %{{.+}} = cir.alloca "bi_alloca" align(1) size(%{{.+}}) : !cir.ptr<!u8i>

// LLVM-LABEL: define {{.*}} ptr @with_align_uninitialized(
// LLVM:         alloca i8, i64 %{{.+}}, align 1

// OGCG-LABEL: define {{.*}} ptr @with_align_uninitialized(
// OGCG:         alloca i8, i64 %{{.+}}, align 1
