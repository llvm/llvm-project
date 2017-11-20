// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Test host codegen.
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-32
#ifdef CK1

template <typename T, int X, long long Y>
struct SS{
  T a[X][Y];

  // CK1: define {{.*}}i32 @{{.+}}foo{{.+}}(
  int foo(void) {

    // CK1: call i32 @__tgt_target(
    // CK1: call void @[[OFFL1:.+]](
    #pragma omp target
    #pragma omp teams distribute parallel for collapse(2)
    for(int i = 0; i < X; i++) {
      for(int j = 0; j < Y; j++) {
	a[i][j] = (T)0;
      }
    }
    // CK1: define internal void @[[OFFL1]](
    // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTL1:.+]] to {{.+}},
    // CK1: ret void

    // CK1: define internal void @[[OUTL1]]({{.+}})
    // discard loop variables not needed here
    // CK1: [[OMP_UB:%.omp.comb.ub]] = alloca i32,
    // CK1: store i32 56087, i32* [[OMP_UB]],
    // CK1: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92, {{.+}}, {{.+}}, i32* [[OMP_UB]],
    // CK1: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL1:.+]] to
    // CK1: call void @__kmpc_for_static_fini(
    // CK1: ret void

    // CK1: define internal void @[[PAR_OUTL1]]({{.+}})
    // CK1: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34, {{.+}}, {{.+}},
    // CK1: call void @__kmpc_for_static_fini(
    // CK1: ret void

    return a[0][0];
  }
};

int teams_template_struct(void) {
  SS<int, 123, 456> V;
  return V.foo();

}
#endif // CK1

// Test host codegen.
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32
#ifdef CK2

template <typename T, int n, int m>
int tmain(T argc) {
  T a[n][m];
  #pragma omp target
  #pragma omp teams distribute parallel for collapse(2)
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      a[i][j] = (T)0;
    }
  }
  return 0;
}

int main (int argc, char **argv) {
  int n = 100;
  int m = 2;
  int a[n][m];
  #pragma omp target
  #pragma omp teams distribute parallel for collapse(2)
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      a[i][j] = 0;
    }
  }
  return tmain<int, 10, 2>(argc);
}

// CK2: define {{.*}}i32 @{{[^,]+}}(i{{.+}}{{.+}} %[[ARGC:.+]], {{.+}})
// CK2: call i32 @__tgt_target(
// CK2: call void @[[OFFL1:.+]]({{.+}})
// CK2: {{%.+}} = call{{.*}} i32 @[[TMAIN:.+]]({{.+}})
// CK2: ret

// CK2:  define {{.*}}void @[[OFFL1]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}} @[[OUTL1:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTL1]]({{.+}})
// CK2: [[OMP_UB:%.omp.comb.ub]] = alloca i64,
// CK2: store i64 {{.+}}, i64* [[OMP_UB]],
// CK2: call void @__kmpc_for_static_init_8({{.+}}, {{.+}}, i32 92, {{.+}}, {{.+}}, i64* [[OMP_UB]],
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL1:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[PAR_OUTL1]]({{.+}})
// CK2: call void @__kmpc_for_static_init_{{[4|8]}}({{.+}}, {{.+}}, i32 34, {{.+}}, {{.+}},
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void


// CK2: define {{.*}}i32 @[[TMAIN]]({{.+}})
// CK2: call i32 @__tgt_target(
// CK2: call void @[[OFFLT1:.+]]({{.+}})
// CK2:  ret
// CK2-NEXT: }

// CK2:  define {{.*}}void @[[OFFLT1]]({{.+}})
// CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTLT1:.+]] to {{.+}},
// CK2: ret void

// CK2: define internal void @[[OUTLT1]]({{.+}})
// discard loop variables not needed here
// CK2: [[OMP_UB:%.omp.comb.ub]] = alloca i32,
// CK2: store i32 {{.+}}, i32* [[OMP_UB]],
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 92, {{.+}}, {{.+}}, i32* [[OMP_UB]],
// CK2: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[TPAR_OUTL1:.+]] to
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

// CK2: define internal void @[[TPAR_OUTL1]]({{.+}})
// CK2: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, i32 34, {{.+}}, {{.+}},
// CK2: call void @__kmpc_for_static_fini(
// CK2: ret void

#endif // CK2
#endif // #ifndef HEADER
