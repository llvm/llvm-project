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

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1

int a[100];

// CK1: define {{.*}}i32 @{{.+}}teams_argument_globali(
int teams_argument_global(int n){
  int te = n / 128;
  int th = 128;
  // discard n_addr
  // CK1: alloca i32,
  // CK1: [[TE:%.+]] = alloca i32,
  // CK1: [[TH:%.+]] = alloca i32,
  // CK1: [[TE_CAST:%.+]] = alloca i{{32|64}},
  // CK1: [[TH_CAST:%.+]] = alloca i{{32|64}},
  // CK1: [[TE_PAR:%.+]] = load{{.+}}, {{.+}} [[TE_CAST]],
  // CK1: [[TH_PAR:%.+]] = load{{.+}}, {{.+}} [[TH_CAST]],

  // CK1: call void @__kmpc_push_target_tripcount_mapper(%struct.ident_t* @{{.+}}, i64 -1, i64 %{{.+}})
  // CK1: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 4, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null

  // CK1: call void @[[OFFL1:.+]](i{{32|64}} [[TE_PAR]], i{{32|64}} [[TH_PAR]],
  #pragma omp target
  #pragma omp teams distribute parallel for simd num_teams(te), thread_limit(th) simdlen(64)
  for(int i = 0; i < n; i++) {
    a[i] = 0;
  }

  int i;
  // CK1: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null
  // CK1: call void @[[OFFL2:.+]](
  #pragma omp target
  {{{
  #pragma omp teams distribute parallel for simd safelen(4) aligned(a) linear(i) 
  for(i = 0; i < n; i++) {
    a[i] = 0;
  }
  }}}
  // outlined target regions
  // CK1: define internal void @[[OFFL1]](i{{32|64}} [[TE_ARG:%.+]], i{{32|64}} [[TH_ARG:%.+]], i{{32|64}} {{.+}}, {{.+}})
  // CK1: [[TE_ADDR:%.+]] = alloca i{{32|64}},
  // CK1: [[TH_ADDR:%.+]] = alloca i{{32|64}},
  // CK1: store{{.+}} [[TE_ARG]], {{.+}} [[TE_ADDR]],
  // CK1: store{{.+}} [[TH_ARG]], {{.+}} [[TH_ADDR]],
  // CK1-64: [[TE_CONV:%.+]] = bitcast{{.+}} [[TE_ADDR]] to
  // CK1-64: [[TH_CONV:%.+]] = bitcast{{.+}} [[TH_ADDR]] to
  // CK1-64: [[TE_VAL:%.+]] = load i32, i32* [[TE_CONV]],
  // CK1-64: [[TH_VAL:%.+]] = load i32, i32* [[TH_CONV]],
  // CK1-32: [[TE_VAL:%.+]] = load i32, i32* [[TE_ADDR]],
  // CK1-32: [[TH_VAL:%.+]] = load i32, i32* [[TH_ADDR]],
  // CK1: call void @__kmpc_push_num_teams({{.+}}, {{.+}}, i32 [[TE_VAL]], i32 [[TH_VAL]])
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 2, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}}, {{.+}})
  // CK1: ret void

  // CK1: define internal void @[[OUTL1]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.+}} @__kmpc_fork_call(
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  // CK1: define internal void @[[OFFL2]]({{.+}}, {{.+}})
  // CK1: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 3, {{.+}} @[[OUTL2:.+]] to {{.+}}, {{.+}}, {{.+}})
  // CK1: ret void

  // CK1: define internal void @[[OUTL2]]({{.+}})
  // CK1: call void @__kmpc_for_static_init_4(
  // CK1: call void {{.+}} @__kmpc_fork_call(
  // CK1: call void @__kmpc_for_static_fini(
  // CK1: ret void

  return a[0];
}

// CK1-DAG: !{!"llvm.loop.vectorize.enable", i1 true}
// CK1-DAG: !{!"llvm.loop.vectorize.width", i32 4}
// CK1-DAG: !{!"llvm.loop.vectorize.width", i32 64}

#endif // CK1

// Test host codegen.
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2: define {{.*}}i32 @{{.+}}teams_local_argv(
int teams_local_arg(void) {
  int n = 100;
  int a[n], i;

  // CK2: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 4, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}, i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null
  // CK2: call void @[[OFFL1:.+]](
  #pragma omp target
  #pragma omp teams distribute parallel for simd safelen(4) aligned(a) linear(i)
  for(i = 0; i < n; i++) {
    a[i] = 0;
  }

  // outlined target region
  // CK2: define internal void @[[OFFL1]]({{.+}}, {{.+}})
  // CK2: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}}, {{.+}})
  // CK2: ret void

  // CK2: define internal void @[[OUTL1]]({{.+}})
  // CK2: call void @__kmpc_for_static_init_4(
  // CK2: call void {{.+}} @__kmpc_fork_call(
  // CK2: call void @__kmpc_for_static_fini(
  // CK2: ret void

  return a[0];
}

// CK2-DAG: !{!"llvm.loop.vectorize.enable", i1 true}
// CK2-DAG: !{!"llvm.loop.vectorize.width", i32 4}

#endif // CK2

// Test host codegen.
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3: [[SSI:%.+]] = type { [{{.+}} x i32], float }

template <typename T, int X, long long Y>
struct SS{
  T a[X];
  float b;
  // CK3: define {{.*}}i32 @{{.+}}foo{{.+}}(
  int foo(void) {
    int i;
  // CK3: call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 2, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null
  // CK3: call void @[[OFFL1:.+]]([[SSI]]* %{{.+}})
    #pragma omp target
    #pragma omp teams distribute parallel for simd safelen(4) aligned(a) linear(i)
    for(i = 0; i < X; i++) {
      a[i] = (T)0;
    }

      // outlined target region
  // CK3: define internal void @[[OFFL1]]([[SSI]]* {{.+}})
  // CK3: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 2, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}}, {{.+}})
  // CK3: ret void

  // CK3: define internal void @[[OUTL1]]({{.+}})
  // CK3: call void @__kmpc_for_static_init_4(
  // CK3: call void {{.+}} @__kmpc_fork_call(
  // CK3: call void @__kmpc_for_static_fini(
  // CK3: ret void

    return a[0];
  }
};

int teams_template_struct(void) {
  SS<int, 123, 456> V;
  return V.foo();

}

// CK3-DAG: !{!"llvm.loop.vectorize.enable", i1 true}
// CK3-DAG: !{!"llvm.loop.vectorize.width", i32 4}
#endif // CK3

// Test host codegen.
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// SIMD-ONLY3-NOT: {{__kmpc|__tgt}}

#ifdef CK4

template <typename T, int n>
int tmain(T argc) {
  T a[n];
  int te = n/128;
  int th = 128;
#pragma omp target
#pragma omp teams distribute parallel for simd num_teams(te) thread_limit(th) simdlen(64)
  for(int i = 0; i < n; i++) {
    a[i] = (T)0;
  }
  return 0;
}

int main (int argc, char **argv) {
  int n = 100;
  int a[n], i;
#pragma omp target
#pragma omp teams distribute parallel for simd safelen(4) aligned(a) linear(i)
  for(i = 0; i < n; i++) {
    a[i] = 0;
  }
  return tmain<int, 10>(argc);
}

// CK4:  define {{.*}}i32 @{{[^,]+}}(i{{.+}}{{.+}} %[[ARGC:.+]], {{.+}})
// CK4:   call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 4, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}, i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null
// CK4: call void @[[OFFL1:.+]]({{.+}})
// CK4: {{%.+}} = call{{.*}} i32 @[[TMAIN:.+]]({{.+}})
// CK4:  ret

// CK4:  define {{.*}}void @[[OFFL1]]({{.+}})
// CK4: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}} @[[OUTL1:.+]] to {{.+}}, {{.+}}, {{.+}})
// CK4: ret void

// CK4: define internal void @[[OUTL1]]({{.+}})
// CK4: call void @__kmpc_for_static_init_4(
// CK4: call void {{.+}} @__kmpc_fork_call(
// CK4: call void @__kmpc_for_static_fini(
// CK4: ret void

// CK4:  define {{.*}}i32 @[[TMAIN]]({{.+}})
// CK4:   call i32 @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i8** null
// CK4: call void @[[OFFLT:.+]]({{.+}})
// CK4:  ret
// CK4-NEXT: }

// CK4: define {{.*}}void @[[OFFLT]](i{{32|64}} [[TE_ARG:%.+]], i{{32|64}} [[TH_ARG:%.+]], {{.+}})
// CK4: [[TE_ADDR:%.+]] = alloca i{{32|64}},
// CK4: [[TH_ADDR:%.+]] = alloca i{{32|64}},
// CK4: store{{.+}} [[TE_ARG]], {{.+}} [[TE_ADDR]],
// CK4: store{{.+}} [[TH_ARG]], {{.+}} [[TH_ADDR]],
// CK4-64: [[TE_CONV:%.+]] = bitcast{{.+}} [[TE_ADDR]] to
// CK4-64: [[TH_CONV:%.+]] = bitcast{{.+}} [[TH_ADDR]] to
// CK4-64: [[TE_VAL:%.+]] = load i32, i32* [[TE_CONV]],
// CK4-64: [[TH_VAL:%.+]] = load i32, i32* [[TH_CONV]],
// CK4-32: [[TE_VAL:%.+]] = load i32, i32* [[TE_ADDR]],
// CK4-32: [[TH_VAL:%.+]] = load i32, i32* [[TH_ADDR]],
// CK4: call void @__kmpc_push_num_teams({{.+}}, {{.+}}, i32 [[TE_VAL]], i32 [[TH_VAL]])
// CK4: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}} @[[OUTLT:.+]] to {{.+}}, {{.+}}, {{.+}})
// CK4: ret void

// CK4: define internal void @[[OUTLT]]({{.+}})
// CK4: call void @__kmpc_for_static_init_4(
// CK4: call void {{.+}} @__kmpc_fork_call(
// CK4: call void @__kmpc_for_static_fini(
// CK4: ret void

// CK4-DAG: !{!"llvm.loop.vectorize.enable", i1 true}
// CK4-DAG: !{!"llvm.loop.vectorize.width", i32 4}
// CK4-DAG: !{!"llvm.loop.vectorize.width", i32 64}

#endif // CK4
#endif

