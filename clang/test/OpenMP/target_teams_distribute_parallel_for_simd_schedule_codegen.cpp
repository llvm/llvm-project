// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Test host codegen.
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64 --check-prefix CK1-OMP45
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64 --check-prefix CK1-OMP45
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix CK1-OMP45
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix CK1-OMP45

// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64 --check-prefix CK1-OMP50
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64  --check-prefix CK1-OMP50
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix CK1-OMP50
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32 --check-prefix CK1-OMP50

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

#ifdef CK1

template <typename T, int X, long long Y>
struct SS{
  T a[X];
  float b;
  // CK1: define {{.*}}i32 @{{.+}}foo{{.+}}(
  int foo(void) {

  // CK1: call i32 @__tgt_target_teams(
  // CK1: call void @[[OFFL1:.+]](
    #pragma omp target teams distribute parallel for simd
    for(int i = 0; i < X; i++) {
      a[i] = (T)0;
    }
  // CK1: call i32 @__tgt_target_teams(
  // CK1: call void @[[OFFL2:.+]](
    #pragma omp target teams distribute parallel for simd schedule(static)
    for(int i = 0; i < X; i++) {
      a[i] = (T)0;
    }
  // CK1: call i32 @__tgt_target_teams(
  // CK1: call void @[[OFFL3:.+]](
    #pragma omp target teams distribute parallel for simd schedule(static, X/2)
    for(int i = 0; i < X; i++) {
      a[i] = (T)0;
    }

  // CK1: call i32 @__tgt_target_teams(
  // CK1: call void @[[OFFL4:.+]](
    #pragma omp target teams distribute parallel for simd schedule(dynamic)
    for(int i = 0; i < X; i++) {
      a[i] = (T)0;
    }

  // CK1: call i32 @__tgt_target_teams(
  // CK1: call void @[[OFFL5:.+]](
    #pragma omp target teams distribute parallel for simd schedule(dynamic, X/2)
    for(int i = 0; i < X; i++) {
      a[i] = (T)0;
    }

  // CK1: define internal void @[[OFFL1]](
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL1:.+]] to {{.+}},
  // CK1: ret void

  // CK1: define internal void @[[OUTL1]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL1:.+]] to
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[PAR_OUTL1]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34,
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[OFFL2]](
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL2:.+]] to {{.+}},
  // CK1: ret void

  // CK1: define internal void @[[OUTL2]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL2:.+]] to
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[PAR_OUTL2]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34,
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[OFFL3]](
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL3:.+]] to {{.+}},
  // CK1: ret void

  // CK1: define internal void @[[OUTL3]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL3:.+]] to
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[PAR_OUTL3]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 33,
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[OFFL4]](
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL4:.+]] to {{.+}},
  // CK1: ret void

  // CK1: define internal void @[[OUTL4]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL4:.+]] to
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[PAR_OUTL4]]({{.+}})
  // CK1-OMP45: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 35,
  // CK1-OMP50: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 1073741859,
  // CK1: call {{.+}} @__kmpc_dispatch_next_4(
  // CK1: ret void

  // CK1: define internal void @[[OFFL5]](
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL5:.+]] to {{.+}},
  // CK1: ret void

  // CK1: define internal void @[[OUTL5]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL5:.+]] to
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[PAR_OUTL5]]({{.+}})
  // CK1-OMP45: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 35,
  // CK1-OMP50: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 1073741859,
  // CK1: call {{.+}} @__kmpc_dispatch_next_4(
  // CK1: ret void

    return a[0];
  }
};

int teams_template_struct(void) {
  SS<int, 123, 456> V;
  return V.foo();

}
#endif // CK1

// Test host codegen.
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64 --check-prefix CK2-OMP45
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64 --check-prefix CK2-OMP45
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32 --check-prefix CK2-OMP45
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32 --check-prefix CK2-OMP45

// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64 --check-prefix CK2-OMP50
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64 --check-prefix CK2-OMP50
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32 --check-prefix CK2-OMP50
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32 --check-prefix CK2-OMP50

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// SIMD-ONLY-NOT: {{__kmpc|__tgt}}
#ifdef CK2

template <typename T, int n>
int tmain(T argc) {
  T a[n];
  int m = 10;
#pragma omp target teams distribute parallel for simd
  for(int i = 0; i < n; i++) {
    a[i] = (T)0;
  }
#pragma omp target teams distribute parallel for simd schedule(static)
  for(int i = 0; i < n; i++) {
    a[i] = (T)0;
  }
#pragma omp target teams distribute parallel for simd schedule(static, m)
  for(int i = 0; i < n; i++) {
    a[i] = (T)0;
  }
#pragma omp target teams distribute parallel for simd schedule(dynamic)
  for(int i = 0; i < n; i++) {
    a[i] = (T)0;
  }
#pragma omp target teams distribute parallel for simd schedule(dynamic, m)
  for(int i = 0; i < n; i++) {
    a[i] = (T)0;
  }
  return 0;
}

int main (int argc, char **argv) {
  int n = 100;
  int a[n];
  int m = 10;
#pragma omp target teams distribute parallel for simd
  for(int i = 0; i < n; i++) {
    a[i] = 0;
  }
#pragma omp target teams distribute parallel for simd dist_schedule(static)
  for(int i = 0; i < n; i++) {
    a[i] = 0;
  }
#pragma omp target teams distribute parallel for simd dist_schedule(static, m)
  for(int i = 0; i < n; i++) {
    a[i] = 0;
  }
#pragma omp target teams distribute parallel for simd schedule(dynamic)
  for(int i = 0; i < n; i++) {
    a[i] = 0;
  }
#pragma omp target teams distribute parallel for simd schedule(dynamic, m)
  for(int i = 0; i < n; i++) {
    a[i] = 0;
  }
  return tmain<int, 10>(argc);
}

// CK2: define {{.*}}i32 @{{[^,]+}}(i{{.+}}{{.+}} %[[ARGC:.+]], {{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFL1:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFL2:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFL3:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFL4:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFL5:.+]]({{.+}})
// CK2: {{%.+}} = call{{.*}} i32 @[[TMAIN:.+]]({{.+}})
// CK2: ret

// CK2:  define {{.*}}void @[[OFFL1]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 3, {{.+}} @[[OUTL1:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTL1]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL1:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTL1]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34,
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define {{.*}}void @[[OFFL2]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 3, {{.+}} @[[OUTL2:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTL2]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL2:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTL2]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34,
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2:  define {{.*}}void @[[OFFL3]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}} @[[OUTL3:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTL3]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL3:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTL3]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2:  define {{.*}}void @[[OFFL4]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[OUTL4:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTL4]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL4:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTL4]]({{.+}})
// CK2-OMP45: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 35,
// CK2-OMP50: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 1073741859,
// CK2: call {{.+}} @__kmpc_dispatch_next_4(
// CK2: ret void


// CK2: define {{.*}}void @[[OFFL5]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[OUTL5:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTL5]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL5:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTL5]]({{.+}})
// CK2-OMP45: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 35,
// CK2-OMP50: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 1073741859,
// CK2: call {{.+}} @__kmpc_dispatch_next_4(
// CK2: ret void

// CK2: define {{.*}}i32 @[[TMAIN]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFLT1:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFLT2:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFLT3:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFLT4:.+]]({{.+}})
// CK2: call i32 @__tgt_target_teams(
// CK2: call void @[[OFFLT5:.+]]({{.+}})
// CK2:  ret
// CK2-NEXT: }

// CK2:  define {{.*}}void @[[OFFLT1]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTLT1:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTLT1]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTLT1:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTLT1]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34,
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2:  define {{.*}}void @[[OFFLT2]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTLT2:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTLT2]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTLT2:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTLT2]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34,
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2:  define {{.*}}void @[[OFFLT3]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[OUTLT3:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTLT3]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTLT3:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTLT3]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 33,
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2:  define {{.*}}void @[[OFFLT4]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[OUTLT4:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTLT4]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTLT4:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTLT4]]({{.+}})
// CK2-OMP45: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 35,
// CK2-OMP50: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 1073741859,
// CK2: call {{.+}} @__kmpc_dispatch_next_4(
// CK2: ret void

// CK2:  define {{.*}}void @[[OFFLT5]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 {{.+}}, {{.+}} @[[OUTLT5:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTLT5]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4(
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTLT5:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTLT5]]({{.+}})
// CK2-OMP45: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 35,
// CK2-OMP50: call void @__kmpc_dispatch_init_4({{.+}}, {{.+}}, i32 1073741859,
// CK2: call {{.+}} @__kmpc_dispatch_next_4(
// CK2: ret void

#endif // CK2
#endif // #ifndef HEADER
