// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// RUN: %clang_cc1 -DCK1 -verify -fopenmp -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix=CK1
// RUN: %clang_cc1 -DCK1 -fopenmp -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --check-prefix=CK1
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix=CK1
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=CK1
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix=CK1

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"

#ifdef CK1

#define N 100

void p_vxv(int *v1, int *v2, int *v3, int n);
void t_vxv(int *v1, int *v2, int *v3, int n);

#pragma omp declare variant(t_vxv) match(construct={target})
#pragma omp declare variant(p_vxv) match(construct={parallel})
void vxv(int *v1, int *v2, int *v3, int n) {
    for (int i = 0; i < n; i++) v3[i] = v1[i] * v2[i];
}
// CK1: define dso_local void @vxv

void p_vxv(int *v1, int *v2, int *v3, int n) {
#pragma omp for
    for (int i = 0; i < n; i++) v3[i] = v1[i] * v2[i] * 3;
}
// CK1: define dso_local void @p_vxv

#pragma omp declare target
void t_vxv(int *v1, int *v2, int *v3, int n) {
#pragma distribute simd
    for (int i = 0; i < n; i++) v3[i] = v1[i] * v2[i] * 2;
}
#pragma omp end declare target
// CK1: define dso_local void @t_vxv


// CK1-LABEL: define {{[^@]+}}@test
int test(void) {
  int v1[N], v2[N], v3[N];

  // init
  for (int i = 0; i < N; i++) {
    v1[i] = (i + 1);
    v2[i] = -(i + 1);
    v3[i] = 0;
  }

#pragma omp target teams map(to: v1[:N],v2[:N]) map(from: v3[:N])
  {
    vxv(v1, v2, v3, N);
  }
// CK1: call void @__omp_offloading_[[OFFLOAD:.+]]({{.+}})

  vxv(v1, v2, v3, N);
// CK1: call void @vxv

#pragma omp parallel
  {
    vxv(v1, v2, v3, N);
  }
// CK1: call void ({{.+}}) @__kmpc_fork_call(%struct.ident_t* {{.+}}, i32 3, void ({{.+}})* bitcast (void (i32*, i32*, [100 x i32]*, [100 x i32]*, [100 x i32]*)* [[PARALLEL_REGION:@.+]] to void

  return 0;
}

// CK1: define internal void @__omp_offloading_[[OFFLOAD]]({{.+}})
// CK1: call void ({{.+}}) @__kmpc_fork_teams(%struct.ident_t* {{.+}}, i32 3, void ({{.+}})* bitcast (void (i32*, i32*, [100 x i32]*, [100 x i32]*, [100 x i32]*)* [[TARGET_REGION:@.+]] to void
// CK1: define internal void [[TARGET_REGION]](
// CK1: call void @t_vxv

// CK1: define internal void [[PARALLEL_REGION]](
// CK1: call void @p_vxv
#endif // CK1

// RUN: %clang_cc1 -DCK2 -verify -fopenmp -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix=CK2
// RUN: %clang_cc1 -DCK2 -fopenmp -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --check-prefix=CK2
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix=CK2
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=CK2
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix=CK2

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"

#ifdef CK2

void test_teams(int ***v1, int ***v2, int ***v3, int n);
void test_target(int ***v1, int ***v2, int ***v3, int n);
void test_parallel(int ***v1, int ***v2, int ***v3, int n);

#pragma omp declare variant(test_teams) match(construct = {teams})
#pragma omp declare variant(test_target) match(construct = {target})
#pragma omp declare variant(test_parallel) match(construct = {parallel})
void test_base(int ***v1, int ***v2, int ***v3, int n) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        v3[i][j][k] = v1[i][j][k] * v2[i][j][k];
}

#pragma omp declare target
void test_teams(int ***v1, int ***v2, int ***v3, int n) {
#pragma omp distribute parallel for simd collapse(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        v3[i][j][k] = v1[i][j][k] * v2[i][j][k];
}
#pragma omp end declare target

#pragma omp declare target
void test_target(int ***v1, int ***v2, int ***v3, int n) {
#pragma omp parallel for simd collapse(3)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        v3[i][j][k] = v1[i][j][k] * v2[i][j][k];
}
#pragma omp end declare target

void test_parallel(int ***v1, int ***v2, int ***v3, int n) {
#pragma omp for collapse(3)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        v3[i][j][k] = v1[i][j][k] * v2[i][j][k];
}

// CK2-LABEL: define {{[^@]+}}@test
void test(int ***v1, int ***v2, int ***v3, int n) {
  int i;

#pragma omp target
#pragma omp teams
  {
    test_base(v1, v2, v3, 0);
  }
// CK2: call void @__omp_offloading_[[OFFLOAD_1:.+]]({{.+}})

#pragma omp target
  {
    test_base(v1, v2, v3, 0);
  }
// CK2: call void @__omp_offloading_[[OFFLOAD_2:.+]]({{.+}})

#pragma omp parallel
  {
    test_base(v1, v2, v3, 0);
  }
// CK2: call void ({{.+}}) @__kmpc_fork_call(%struct.ident_t* {{.+}}, i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32****, i32****, i32****)* [[PARALLEL_REGION:@.+]] to void
}

// CK2: define internal void @__omp_offloading_[[OFFLOAD_1]]({{.+}})
// CK2: call void ({{.+}}) @__kmpc_fork_teams(%struct.ident_t* {{.+}}, i32 3, void ({{.+}})* bitcast (void (i32*, i32*, i32****, i32****, i32****)* [[TARGET_REGION_1:@.+]] to void
// CK2: define internal void [[TARGET_REGION_1]](
// CK2: call void @test_teams

// CK2: define internal void @__omp_offloading_[[OFFLOAD_2]]({{.+}})
// CK2: call void @test_target

// CK2: define internal void [[PARALLEL_REGION]](
// CK2: call void @test_parallel

#endif // CK2

// RUN: %clang_cc1 -DCK3 -verify -fopenmp -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix=CK3
// RUN: %clang_cc1 -DCK3 -fopenmp -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --check-prefix=CK3
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix=CK3
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=CK3
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix=CK3

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"

#ifdef CK3

#define N 100

int t_for(int *v1, int *v2, int *v3, int n);
int t_simd(int *v1, int *v2, int *v3, int n);

#pragma omp declare variant(t_simd) match(construct = {simd})
#pragma omp declare variant(t_for) match(construct = {for})
int t(int *v1, int *v2, int *v3, int idx) {
  return v1[idx] * v2[idx];
}

int t_for(int *v1, int *v2, int *v3, int idx) {
  return v1[idx] * v2[idx];
}

#pragma omp declare simd
int t_simd(int *v1, int *v2, int *v3, int idx) {
  return v1[idx] * v2[idx];
}

// CK3-LABEL: define {{[^@]+}}@test
void test(void) {
  int v1[N], v2[N], v3[N];

  // init
  for (int i = 0; i < N; i++) {
    v1[i] = (i + 1);
    v2[i] = -(i + 1);
    v3[i] = 0;
  }

#pragma omp simd
  for (int i = 0; i < N; i++) {
    v3[i] = t(v1, v2, v3, i);
  }
// CK3: call = call i32 @t_simd


#pragma omp for
  for (int i = 0; i < N; i++) {
    v3[i] = t(v1, v2, v3, i);
  }
// CK3: call{{.+}} = call i32 @t_for
}

#endif // CK3

// RUN: %clang_cc1 -DCK4 -verify -fopenmp -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --check-prefix=CK4
// RUN: %clang_cc1 -DCK4 -fopenmp -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK4 -fopenmp -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --check-prefix=CK4
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix=CK4
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=CK4
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix=CK4

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c -triple x86_64-unknown-linux -emit-pch -o %t -fopenmp-version=45 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c -triple x86_64-unknown-linux -include-pch %t -verify %s -emit-llvm -o - -fopenmp-version=45 | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-unknown-linux -fopenmp-targets=amdgcn-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --implicit-check-not="{{__kmpc|__tgt}}"

#ifdef CK4

#define N 100

void not_selected_vxv(int *v1, int *v2, int *v3, int n);
void combined_vxv(int *v1, int *v2, int *v3, int n);
void all_vxv(int *v1, int *v2, int *v3, int n);

#pragma omp declare variant(all_vxv) match(construct={target,teams,parallel,for,simd})
#pragma omp declare variant(combined_vxv) match(construct={target,teams,parallel,for})
#pragma omp declare variant(not_selected_vxv) match(construct={parallel,for})
void vxv(int *v1, int *v2, int *v3, int n) {
    for (int i = 0; i < n; i++) v3[i] = v1[i] * v2[i];
}

void not_selected_vxv(int *v1, int *v2, int *v3, int n) {
    for (int i = 0; i < n; i++) v3[i] = v1[i] * v2[i] * 3;
}

#pragma omp declare target
void combined_vxv(int *v1, int *v2, int *v3, int n) {
    for (int i = 0; i < n; i++) v3[i] = v1[i] * v2[i] * 2;
}
#pragma omp end declare target

#pragma omp declare target
void all_vxv(int *v1, int *v2, int *v3, int n) {
    for (int i = 0; i < n; i++) v3[i] = v1[i] * v2[i] * 4;
}
#pragma omp end declare target

// CK4-LABEL: define {{[^@]+}}@test
void test(void) {
    int v1[N], v2[N], v3[N];

    //init
    for (int i = 0; i < N; i++) {
      v1[i] = (i + 1);
      v2[i] = -(i + 1);
      v3[i] = 0;
    }

#pragma omp target teams map(to: v1[:N],v2[:N]) map(from: v3[:N])
    {
#pragma omp parallel for
      for (int i = 0; i < N; i++)
        vxv(v1, v2, v3, N);
    }
// CK4: call void @__omp_offloading_[[OFFLOAD_1:.+]]({{.+}})

#pragma omp simd
    for (int i = 0; i < N; i++)
      vxv(v1, v2, v3, N);
// CK4: call void @vxv

#pragma omp target teams distribute parallel for simd map(from: v3[:N])
    for (int i = 0; i < N; i++)
      for (int i = 0; i < N; i++)
        for (int i = 0; i < N; i++)
          vxv(v1, v2, v3, N);
// CK4: call void @__omp_offloading_[[OFFLOAD_2:.+]]({{.+}})
}
// CK4-DAG: call void @all_vxv
// CK4-DAG: call void @combined_vxv

#endif // CK4

#endif // HEADER
