// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

double fabs(double x) {
  return __builtin_fabs(x);
}

// CIR: {{.*}} = cir.fabs {{.*}} : !cir.double
// LLVM: {{.*}} = call double @llvm.fabs.f64(double {{.*}})
// OGCG: {{.*}} = call double @llvm.fabs.f64(double {{.*}})
