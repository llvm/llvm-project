// This chunk is based on Clang :: Frontend/ast-main.c,
// but targeted for z/OS.  (Ensure we have no fatal error.)
// We also duplicate for C++.
// RUN: env SDKROOT="/" %clang -Xclang -source-date-epoch -Xclang 1783944000 -target s390x-none-zos -emit-llvm -S -o - -x c - < %s | grep -v DIFile > %t1.ll
// RUN: env SDKROOT="/" %clang -target s390x-none-zos -emit-ast -o %t.ast %s
// RUN: env SDKROOT="/" %clang -Xclang -source-date-epoch -Xclang 1783944000 -target s390x-none-zos -emit-llvm -S -o - -x ast - < %t.ast | grep -v DIFile > %t2.ll
// make sure that the IR from the two runs are the same
// RUN: diff %t1.ll %t2.ll

// check that the langage attribute has come through
// RUN: FileCheck --check-prefix CHECK-C %s < %t1.ll
// CHECK-C: !"zos_cu_language", !"C"}


// RUN: env SDKROOT="/" %clang -Xclang -source-date-epoch -Xclang 1783944000 -target s390x-none-zos -emit-llvm -S -o - -x c++ - < %s | grep -v DIFile > %t1.ll
// RUN: env SDKROOT="/" %clang -target s390x-none-zos -emit-ast -o %t.ast -x c++ %s
// RUN: env SDKROOT="/" %clang -Xclang -source-date-epoch -Xclang 1783944000 -target s390x-none-zos -emit-llvm -S -o - -x ast - < %t.ast | grep -v DIFile > %t2.ll
// make sure that the IR from the two runs are the same
// RUN: diff %t1.ll %t2.ll

// check that the langage attribute has come through
// RUN: FileCheck --check-prefix CHECK-CPP %s < %t1.ll
// CHECK-CPP: !"zos_cu_language", !"C++"}



int main(void) {
  return 0;
}
