// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -fopenmp-enable-irbuilder -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -fopenmp-enable-irbuilder -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=60 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp flush
#pragma omp flush seq_cst
#pragma omp flush acq_rel
#pragma omp flush acquire
#pragma omp flush release
#pragma omp flush(a)
  return a + argc;
}

// CHECK-LABEL: @main
int main() {
  static int a;
#pragma omp flush
#pragma omp flush seq_cst
#pragma omp flush acq_rel
#pragma omp flush acquire
#pragma omp flush release
#pragma omp flush(a)
  // CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
  // CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
  // CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
  // CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
  // CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
  // CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
  return tmain(a);
  // CHECK: call {{.*}} [[TMAIN:@.+]](
  // CHECK: ret
}

// CHECK: [[TMAIN]]
// CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
// CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
// CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
// CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
// CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
// CHECK: call {{.*}}void @__kmpc_flush(ptr {{(@|%).+}})
// CHECK: ret

// CHECK-NOT: line: 0,

#endif
