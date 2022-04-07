// Test host code gen

// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// Test host code gen

// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers  -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER


template <typename T>
T tmain(T &r) {
  int n = 1000;  
  // schedule: dynamic chunk
  #pragma omp target map(tofrom:r)
  #pragma omp teams
  #pragma omp distribute parallel for reduction(+:r)
  for (int i = 0; i < n; ++i)
    r += (T)i;  

  return r;
}

int main() {
  int n = 1000;
  int r = 0;
  #pragma omp target map(tofrom:r)
  #pragma omp teams
  #pragma omp distribute parallel for reduction(+:r)
  for (int i = 0; i < n; ++i)
    r += i;

  return tmain<int>(r);
}

// CHECK-LABEL: main
// CHECK: call{{.+}} @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}},
// CHECK: call void [[OFFL:@.+]](
// CHECK: call{{.+}} [[TMAIN:@.+]](i{{32|64}}
// CHECK: ret

// CHECK: define{{.+}} [[OFFL]](
// CHECK: call{{.+}} @__kmpc_fork_teams({{.+}}, {{.+}}, {{.+}} [[TEOUTL:@.+]] to{{.+}}
// CHECK: ret void

// CHECK: define{{.+}} [[TEOUTL]](
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} [[PAROUTL:@.+]] to{{.+}}
// CHECK: ret void

// CHECK: define{{.+}} [[PAROUTL]](
// CHECK: call{{.+}} @__kmpc_reduce_nowait(
// CHECK: call{{.+}} @__kmpc_end_reduce_nowait(
// CHECK: ret void

// CHECK: define{{.+}} [[TMAIN]](i{{32|64}}
// CHECK: call{{.+}} @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}},
// CHECK: call void [[TOFFL:@.+]](
// CHECK: ret

// CHECK: define{{.+}} [[TOFFL]](
// CHECK: call{{.+}} @__kmpc_fork_teams({{.+}}, {{.+}}, {{.+}} [[TEMPLTEOUTL:@.+]] to{{.+}}
// CHECK: ret void

// CHECK: define{{.+}} [[TEMPLTEOUTL]](
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} [[TPAROUTL:@.+]] to{{.+}}
// CHECK: ret void

// CHECK: define{{.+}} [[TPAROUTL]](
// CHECK: call{{.+}} @__kmpc_reduce_nowait(
// CHECK: call{{.+}} @__kmpc_end_reduce_nowait(
// CHECK: ret void

#endif // HEADER
