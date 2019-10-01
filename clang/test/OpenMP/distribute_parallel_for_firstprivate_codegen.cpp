// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-target | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-target | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32

// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-target | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-target | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-target | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-target | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-target | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-target | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  operator T() { return T(); }
  ~S() {}
};

// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
template <typename T>
T tmain() {
  S<T> test;
  T t_var = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> &var = test;
  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for firstprivate(t_var, vec, s_arr, s_arr, var, var)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  return T();
}

int main() {
  static int svar;
  volatile double g;
  volatile double &g1 = g;

  #ifdef LAMBDA
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
    static float sfvar;
    // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams(
    // LAMBDA: call void [[OFFLOADING_FUN:@.+]](

    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED:@.+]] to {{.+}})
    #pragma omp target
    #pragma omp teams
    #pragma omp distribute parallel for firstprivate(g, g1, svar, sfvar)
    for (int i = 0; i < 2; ++i) {
      // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_OUTLINED]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, double* {{.+}} [[G_IN:%.+]], double*{{.+}} [[G1_IN:%.+]], i{{[0-9]+}}*{{.+}} [[SVAR_IN:%.+]], float*{{.+}} [[SFVAR_IN:%.+]])

      // addr alloca's
      // LAMBDA: [[G_ADDR:%.+]] = alloca double*,
      // LAMBDA: [[G1_ADDR:%.+]] = alloca double*,
      // LAMBDA: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
      // LAMBDA: [[SFVAR_ADDR:%.+]] = alloca float*,
      // LAMBDA: [[G1_REF:%.+]] = alloca double*,
      // LAMBDA: [[G1_REF1:%.+]] = alloca double*,

      // private alloca's
      // LAMBDA: [[G_PRIV:%.+]] = alloca double,
      // LAMBDA: [[G1_PRIV:%.+]] = alloca double,
      // LAMBDA: [[TMP_PRIV:%.+]] = alloca double*,
      // LAMBDA: [[SVAR_PRIV:%.+]] = alloca i{{[0-9]+}},
      // LAMBDA: [[SFVAR_PRIV:%.+]] = alloca float,

      // transfer input parameters into addr alloca's
      // LAMBDA-DAG: store {{.+}} [[G_IN]], {{.+}} [[G_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[G1_IN]], {{.+}} [[G1_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SVAR_IN]], {{.+}} [[SVAR_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SFVAR_IN]], {{.+}} [[SFVAR_ADDR]],

      // LAMBDA-DAG: [[G_CONV:%.+]] = load {{.+}}*, {{.+}}** [[G_ADDR]]
      // LAMBDA-DAG: [[G1_CONV:%.+]] = load {{.+}}*, {{.+}}** [[G1_ADDR]]
      // LAMBDA-DAG: [[SVAR_CONV:%.+]] = load {{.+}}*, {{.+}}** [[SVAR_ADDR]]
      // LAMBDA-DAG: [[SFVAR_CONV:%.+]] = load {{.+}}*, {{.+}}** [[SFVAR_ADDR]]

      // init private alloca's with addr alloca's
      // g
      // LAMBDA-DAG: [[G_ADDR_VAL:%.+]] = load {{.+}}, {{.+}}* [[G_CONV]],
      // LAMBDA-DAG: store {{.+}} [[G_ADDR_VAL]], {{.+}}* [[G_PRIV]],

      // g1
      // LAMBDA-DAG: [[TMP_REF:%.+]] = load {{.+}}*, {{.+}}** [[G1_REF1]],
      // LAMBDA-DAG: [[TMP_VAL:%.+]] = load {{.+}}, {{.+}}* [[TMP_REF]],
      // LAMBDA-DAG: store {{.+}} [[TMP_VAL]], {{.+}}* [[G1_PRIV]]
      // LAMBDA-DAG: store {{.+}}* [[G1_PRIV]], {{.+}}** [[TMP_PRIV]],

      // svar
      // LAMBDA-DAG: [[SVAR_VAL:%.+]] = load {{.+}}, {{.+}}* [[SVAR_CONV]],
      // LAMBDA-DAG: store {{.+}} [[SVAR_VAL]], {{.+}}* [[SVAR_PRIV]],

      // sfvar
      // LAMBDA-DAG: [[SFVAR_VAL:%.+]] = load {{.+}}, {{.+}}* [[SFVAR_CONV]],
      // LAMBDA-DAG: store {{.+}} [[SFVAR_VAL]], {{.+}}* [[SFVAR_PRIV]],

      // LAMBDA: call {{.*}}void @__kmpc_for_static_init_4(
      // pass firstprivate parameters to parallel outlined function
      // g
      // LAMBDA-64-DAG: [[G_PRIV_VAL:%.+]] = load {{.+}}, {{.+}}* [[G_PRIV]],
      // LAMBDA-64: [[G_CAST_CONV:%.+]] = bitcast {{.+}}* [[G_CAST:%.+]] to
      // LAMBDA-64-DAG: store {{.+}} [[G_PRIV_VAL]], {{.+}}* [[G_CAST_CONV]],
      // LAMBDA-64-DAG: [[G_PAR:%.+]] = load {{.+}}, {{.+}}* [[G_CAST]],

      // g1
      // LAMBDA-DAG: [[TMP_PRIV_VAL:%.+]] = load {{.+}}, {{.+}}* [[TMP_PRIV]],
      // LAMBDA-DAG: [[G1_PRIV_VAL:%.+]] = load {{.+}}, {{.+}}* [[TMP_PRIV_VAL]],
      // LAMBDA: [[G1_CAST_CONV:%.+]] = bitcast {{.+}}* [[G1_CAST:%.+]] to
      // LAMBDA-DAG: store {{.+}} [[G1_PRIV_VAL]], {{.+}}* [[G1_CAST_CONV]],
      // LAMBDA-DAG: [[G1_PAR:%.+]] = load {{.+}}, {{.+}}* [[G1_CAST]],

      // svar
      // LAMBDA: [[SVAR_VAL:%.+]] = load {{.+}}, {{.+}}* [[SVAR_PRIV]],
      // LAMBDA-64-DAG: [[SVAR_CAST_CONV:%.+]] = bitcast {{.+}}* [[SVAR_CAST:%.+]] to
      // LAMBDA-64-DAG: store {{.+}} [[SVAR_VAL]], {{.+}}* [[SVAR_CAST_CONV]],
      // LAMBDA-32-DAG: store {{.+}} [[SVAR_VAL]], {{.+}}* [[SVAR_CAST:%.+]],
      // LAMBDA-DAG: [[SVAR_PAR:%.+]] = load {{.+}}, {{.+}}* [[SVAR_CAST]],

      // sfvar
      // LAMBDA: [[SFVAR_VAL:%.+]] = load {{.+}}, {{.+}}* [[SFVAR_PRIV]],
      // LAMBDA-DAG: [[SFVAR_CAST_CONV:%.+]] = bitcast {{.+}}* [[SFVAR_CAST:%.+]] to
      // LAMBDA-DAG: store {{.+}} [[SFVAR_VAL]], {{.+}}* [[SFVAR_CAST_CONV]],
      // LAMBDA-DAG: [[SFVAR_PAR:%.+]] = load {{.+}}, {{.+}}* [[SFVAR_CAST]],

      // LAMBDA-64: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED:@.+]] to void ({{.+}})*), {{.+}}, {{.+}}, {{.+}} [[G_PAR]], {{.+}} [[G1_PAR]], {{.+}} [[SVAR_PAR]], {{.+}} [[SFVAR_PAR]])
      // LAMBDA-32: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED:@.+]] to void ({{.+}})*), {{.+}}, {{.+}}, {{.+}} [[G_PRIV]], {{.+}} [[G1_PAR]], {{.+}} [[SVAR_PAR]], {{.+}} [[SFVAR_PAR]])
      // LAMBDA: call {{.*}}void @__kmpc_for_static_fini(
      // LAMBDA: ret void


      // LAMBDA-64: define{{.+}} void [[OMP_PARFOR_OUTLINED]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, {{.+}}, {{.+}}, i{{[0-9]+}} [[G_IN:%.+]], i{{[0-9]+}} [[G1_IN:%.+]], i{{[0-9]+}} [[SVAR_IN:%.+]], i{{[0-9]+}} [[SFVAR_IN:%.+]])
      // LAMBDA-32: define{{.+}} void [[OMP_PARFOR_OUTLINED]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, {{.+}}, {{.+}}, double* {{.+}} [[G_IN:%.+]], i{{[0-9]+}} [[G1_IN:%.+]], i{{[0-9]+}} [[SVAR_IN:%.+]], i{{[0-9]+}} [[SFVAR_IN:%.+]])
      // skip initial params
      // LAMBDA: {{.+}} = alloca{{.+}},
      // LAMBDA: {{.+}} = alloca{{.+}},
      // LAMBDA: {{.+}} = alloca{{.+}},
      // LAMBDA: {{.+}} = alloca{{.+}},

      // addr alloca's
      // LAMBDA-64: [[G_ADDR:%.+]] = alloca i{{[0-9]+}},
      // LAMBDA-32: [[G_ADDR:%.+]] = alloca double*,
      // LAMBDA: [[G1_ADDR:%.+]] = alloca i{{[0-9]+}},
      // LAMBDA: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}},
      // LAMBDA: [[SFVAR_ADDR:%.+]] = alloca i{{[0-9]+}},
      // LAMBDA: [[G1_REF:%.+]] = alloca double*,

      // private alloca's (only for 32-bit)
      // LAMBDA-32: [[G_PRIV:%.+]] = alloca double,

      // transfer input parameters into addr alloca's
      // LAMBDA-DAG: store {{.+}} [[G_IN]], {{.+}} [[G_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[G1_IN]], {{.+}} [[G1_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SVAR_IN]], {{.+}} [[SVAR_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SFVAR_IN]], {{.+}} [[SFVAR_ADDR]],

      // prepare parameters for lambda
      // g
      // LAMBDA-64-DAG: [[G_CONV:%.+]] = bitcast {{.+}}* [[G_ADDR]] to
      // LAMBDA-32-DAG: [[G_ADDR_REF:%.+]] = load {{.+}}*, {{.+}}** [[G_ADDR]]
      // LAMBDA-32-DAG: [[G_ADDR_VAL:%.+]] = load {{.+}}, {{.+}}* [[G_ADDR_REF]],
      // LAMBDA-32-DAG: store {{.+}} [[G_ADDR_VAL]], {{.+}}* [[G_PRIV]],

      // g1
      // LAMBDA-DAG: [[G1_CONV:%.+]] = bitcast {{.+}}* [[G1_ADDR]] to
      // LAMBDA-DAG: store {{.+}}* [[G1_CONV]], {{.+}}* [[G1_REF]],

      // svar
      // LAMBDA-64-DAG: [[SVAR_CONV:%.+]] = bitcast {{.+}}* [[SVAR_ADDR]] to

      // sfvar
      // LAMBDA-DAG: [[SFVAR_CONV:%.+]] = bitcast {{.+}}* [[SFVAR_ADDR]] to

      // LAMBDA: call {{.*}}void @__kmpc_for_static_init_4(
      g = 1;
      g1 = 1;
      svar = 3;
      sfvar = 4.0;
      // LAMBDA-64: store double 1.0{{.+}}, double* [[G_CONV]],
      // LAMBDA-32: store double 1.0{{.+}}, double* [[G_PRIV]],
      // LAMBDA: [[G1_REF_REF:%.+]] = load {{.+}}*, {{.+}}** [[G1_REF]],
      // LAMBDA: store {{.+}} 1.0{{.+}}, {{.+}}* [[G1_REF_REF]],
      // LAMBDA-64: store {{.+}} 3, {{.+}}* [[SVAR_CONV]],
      // LAMBDA-32: store {{.+}} 3, {{.+}}* [[SVAR_ADDR]],
      // LAMBDA: store {{.+}} 4.0{{.+}}, {{.+}}* [[SFVAR_CONV]],

      // pass params to inner lambda
      // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA-64: store double* [[G_CONV]], double** [[G_PRIVATE_ADDR_REF]],
      // LAMBDA-32: store double* [[G_PRIV]], double** [[G_PRIVATE_ADDR_REF]],
      // LAMBDA: [[G1_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[G1_REF_REF:%.+]] = load double*, double** [[G1_REF]],
      // LAMBDA: store double* [[G1_REF_REF]], double** [[G1_PRIVATE_ADDR_REF]],
      // LAMBDA: [[SVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
      // LAMBDA-64: store i{{[0-9]+}}* [[SVAR_CONV]], i{{[0-9]+}}** [[SVAR_PRIVATE_ADDR_REF]]
      // LAMBDA-32: store i{{[0-9]+}}* [[SVAR_ADDR]], i{{[0-9]+}}** [[SVAR_PRIVATE_ADDR_REF]]
      // LAMBDA: [[SFVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
      // LAMBDA: store float* [[SFVAR_CONV]], float** [[SFVAR_PRIVATE_ADDR_REF]]
      // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
      // LAMBDA: call {{.*}}void @__kmpc_for_static_fini(
      // LAMBDA: ret void
      [&]() {
	// LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
	// LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
	g = 2;
	g1 = 2;
	svar = 4;
	sfvar = 8.0;
	// LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
	// LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
	// LAMBDA: [[G_REF:%.+]] = load double*, double** [[G_PTR_REF]]
	// LAMBDA: store double 2.0{{.+}}, double* [[G_REF]]

	// LAMBDA: [[TMP_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
	// LAMBDA: [[G1_REF:%.+]] = load double*, double** [[TMP_PTR_REF]]
	// LAMBDA: store double 2.0{{.+}}, double* [[G1_REF]],
	// LAMBDA: [[SVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
	// LAMBDA: [[SVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SVAR_PTR_REF]]
	// LAMBDA: store i{{[0-9]+}} 4, i{{[0-9]+}}* [[SVAR_REF]]
	// LAMBDA: [[SFVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
	// LAMBDA: [[SFVAR_REF:%.+]] = load float*, float** [[SFVAR_PTR_REF]]
	// LAMBDA: store float 8.0{{.+}}, float* [[SFVAR_REF]]
      }();
    }
  }();
  return 0;
  #else
  S<float> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> &var = test;

  #pragma omp target
  #pragma omp teams
  #pragma omp distribute parallel for firstprivate(t_var, vec, s_arr, s_arr, var, var, svar)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  return tmain<int>();
  #endif
}

// CHECK-LABEL: define{{.*}} i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOAD_FUN_0:@.+]](
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_DESTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])

// CHECK: define{{.+}} [[OFFLOAD_FUN_0]](i{{[0-9]+}} [[T_VAR_IN:%.+]], [2 x i{{[0-9]+}}]* {{.+}} [[VEC_IN:%.+]], [2 x [[S_FLOAT_TY]]]* {{.+}} [[S_ARR_IN:%.+]], [[S_FLOAT_TY]]* {{.+}} [[VAR_IN:%.+]], i{{[0-9]+}} [[SVAR_IN:%.+]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 5, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i{{[0-9]+}}]*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]*, i{{[0-9]+}}*)* [[OMP_OUTLINED_0:@.+]] to void
// CHECK: ret

// CHECK: define internal void [[OMP_OUTLINED_0]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i{{[0-9]+}}*{{.+}} [[T_VAR_IN:%.+]], [2 x i{{[0-9]+}}]* {{.+}} [[VEC_IN:%.+]], [2 x [[S_FLOAT_TY]]]* {{.+}} [[S_ARR_IN:%.+]], [[S_FLOAT_TY]]* {{.+}} [[VAR_IN:%.+]], i{{[0-9]+}}*{{.+}} [[SVAR_IN:%.+]])

// CHECK: alloca i{{[0-9]+}}*,
// CHECK: alloca i{{[0-9]+}}*,
// addr alloca's
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// CHECK: [[TMP:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[TMP1:%.+]] = alloca [[S_FLOAT_TY]]*,

// skip loop alloca's
// CHECK: [[OMP_IV:.omp.iv+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_LB:.omp.comb.lb+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_UB:.omp.comb.ub+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_ST:.omp.stride+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST:.omp.is_last+]] = alloca i{{[0-9]+}},

// private alloca's
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[TMP_PRIV:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SVAR_PRIV:%.+]] = alloca i{{[0-9]+}},

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// init addr alloca's with input values
// CHECK-DAG: store {{.+}} [[T_VAR_IN]], {{.+}}* [[T_VAR_ADDR]],
// CHECK-DAG: store {{.+}} [[VEC_IN]], {{.+}} [[VEC_ADDR]],
// CHECK-DAG: store {{.+}} [[S_ARR_IN]], {{.+}} [[S_ARR_ADDR]],
// CHECK-DAG: store {{.+}} [[VAR_IN]], {{.+}} [[VAR_ADDR]],
// CHECK-DAG: store {{.+}} [[SVAR_IN]], {{.+}} [[SVAR_ADDR]],

// init private alloca's with addr alloca's
// t-var
// CHECK-DAG: [[T_VAR_REF:%.+]] = load {{.+}}, {{.+}}** [[T_VAR_ADDR]],
// CHECK-DAG: [[T_VAR_ADDR_VAL:%.+]] = load {{.+}}, {{.+}}* [[T_VAR_REF]],
// CHECK-DAG: store {{.+}} [[T_VAR_ADDR_VAL]], {{.+}} [[T_VAR_PRIV]],

// vec
// CHECK-DAG: [[VEC_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[VEC_ADDR]],
// CHECK-DAG: [[VEC_PRIV_BCAST:%.+]] = bitcast {{.+}} [[VEC_PRIV]] to
// CHECK-DAG: [[VEC_ADDR_BCAST:%.+]] = bitcast {{.+}} [[VEC_ADDR_VAL]] to
// CHECK-DAG: call void @llvm.memcpy{{.+}}({{.+}}* align {{[0-9]+}} [[VEC_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[VEC_ADDR_BCAST]],

// s_arr
// CHECK-DAG: [[S_ARR_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[S_ARR_ADDR]],
// CHECK-DAG: [[S_ARR_BGN:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_ADDR_BCAST:%.+]] = bitcast {{.+}}* [[S_ARR_ADDR_VAL]] to
// CHECK-DAG: [[S_ARR_BGN_GEP:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_BGN]],
// CHECK-DAG: [[S_ARR_EMPTY:%.+]] = icmp {{.+}} [[S_ARR_BGN]], [[S_ARR_BGN_GEP]]
// CHECK-DAG: br {{.+}} [[S_ARR_EMPTY]], label %[[CPY_DONE:.+]], label %[[CPY_BODY:.+]]
// CHECK-DAG: [[CPY_BODY]]:
// CHECK-DAG: call void @llvm.memcpy{{.+}}(
// CHECK-DAG: [[CPY_DONE]]:

// var
// CHECK-DAG: [[TMP_REF:%.+]] = load {{.+}}*, {{.+}}* [[TMP1]],
// CHECK-DAG: [[VAR_PRIV_BCAST:%.+]] = bitcast {{.+}}* [[VAR_PRIV]] to
// CHECK-DAG: [[TMP_REF_BCAST:%.+]] = bitcast {{.+}}* [[TMP_REF]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[VAR_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[TMP_REF_BCAST]],
// CHECK-DAG: store {{.+}}* [[VAR_PRIV]], {{.+}}** [[TMP_PRIV]],

// svar
// CHECK-DAG: [[SVAR_REF:%.+]] = load {{.+}}*, {{.+}}** [[SVAR_ADDR]],
// CHECK-DAG: [[SVAR:%.+]] = load {{.+}}, {{.+}}* [[SVAR_REF]],
// CHECK-DAG: store {{.+}} [[SVAR]], {{.+}}* [[SVAR_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// pass private alloca's to fork
// CHECK-DAG: [[T_VAR_PRIV_VAL:%.+]] = load {{.+}}, {{.+}}* [[T_VAR_PRIV]],
// not dag to distinguish with S_VAR_CAST
// CHECK-64: [[T_VAR_CAST_CONV:%.+]] = bitcast {{.+}}* [[T_VAR_CAST:%.+]] to
// CHECK-64-DAG: store {{.+}} [[T_VAR_PRIV_VAL]], {{.+}} [[T_VAR_CAST_CONV]],
// CHECK-32: store {{.+}} [[T_VAR_PRIV_VAL]], {{.+}} [[T_VAR_CAST:%.+]],
// CHECK-DAG: [[T_VAR_CAST_VAL:%.+]] = load {{.+}}, {{.+}}* [[T_VAR_CAST]],
// CHECK-DAG: [[TMP_PRIV_VAL:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[TMP_PRIV]],
// CHECK-DAG: [[SVAR_PRIV_VAL:%.+]] = load {{.+}}, {{.+}}* [[SVAR_PRIV]],
// CHECK-64-DAG: [[SVAR_CAST_CONV:%.+]] = bitcast {{.+}}* [[SVAR_CAST:%.+]] to
// CHECK-64-DAG: store {{.+}} [[SVAR_PRIV_VAL]], {{.+}}* [[SVAR_CAST_CONV]],
// CHECK-32-DAG: store {{.+}} [[SVAR_PRIV_VAL]], {{.+}}* [[SVAR_CAST:%.+]],
// CHECK-DAG: [[SVAR_CAST_VAL:%.+]] = load {{.+}}, {{.+}}* [[SVAR_CAST]],
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_0:@.+]] to void ({{.+}})*), {{.+}}, {{.+}}, [2 x i{{[0-9]+}}]* [[VEC_PRIV]], i{{[0-9]+}} [[T_VAR_CAST_VAL]], [2 x [[S_FLOAT_TY]]]* [[S_ARR_PRIV]], [[S_FLOAT_TY]]* [[TMP_PRIV_VAL]], i{{[0-9]+}} [[SVAR_CAST_VAL]])
// CHECK: call void @__kmpc_for_static_fini(

// call destructors: var..
// CHECK-DAG: call {{.+}} [[S_FLOAT_TY_DEF_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])

// ..and s_arr
// CHECK: {{.+}}:
// CHECK: [[S_ARR_EL_PAST:%.+]] = phi [[S_FLOAT_TY]]*
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = getelementptr {{.+}}, {{.+}} [[S_ARR_EL_PAST]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_DESTR]]([[S_FLOAT_TY]]* [[S_ARR_PRIV_ITEM]])

// CHECK: ret void

// By OpenMP specifications, 'firstprivate' applies to both distribute and parallel for.
// However, the support for 'firstprivate' of 'parallel' is only used when 'parallel'
// is found alone. Therefore we only have one 'firstprivate' support for 'parallel for'
// in combination
// CHECK: define internal void [[OMP_PARFOR_OUTLINED_0]]({{.+}}, {{.+}}, {{.+}}, {{.+}}, [2 x i{{[0-9]+}}]* {{.+}} [[VEC_IN:%.+]], i{{[0-9]+}} [[T_VAR_IN:%.+]], [2 x [[S_FLOAT_TY]]]* {{.+}} [[S_ARR_IN:%.+]], [[S_FLOAT_TY]]* {{.+}} [[VAR_IN:%.+]], i{{[0-9]+}} [[SVAR_IN:%.+]])

// addr alloca's
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[TMP:%.+]] = alloca [[S_FLOAT_TY]]*,

// skip loop alloca's
// CHECK: [[OMP_IV:.omp.iv+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_LB:.omp.lb+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_UB:.omp.ub+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_ST:.omp.stride+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST:.omp.is_last+]] = alloca i{{[0-9]+}},

// private alloca's
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[TMP_PRIV:%.+]] = alloca [[S_FLOAT_TY]]*,

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// init addr alloca's with input values
// CHECK-DAG: store {{.+}} [[VEC_IN]], {{.+}} [[VEC_ADDR]],
// CHECK-DAG: store {{.+}} [[T_VAR_IN]], {{.+}}* [[T_VAR_ADDR]],
// CHECK-DAG: store {{.+}} [[S_ARR_IN]], {{.+}} [[S_ARR_ADDR]],
// CHECK-DAG: store {{.+}} [[VAR_IN]], {{.+}} [[VAR_ADDR]],
// CHECK-DAG: store {{.+}} [[SVAR_IN]], {{.+}} [[SVAR_ADDR]],

// init private alloca's with addr alloca's
// vec
// CHECK-DAG: [[VEC_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[VEC_ADDR]],
// CHECK-DAG: [[VEC_PRIV_BCAST:%.+]] = bitcast {{.+}} [[VEC_PRIV]] to
// CHECK-DAG: [[VEC_ADDR_BCAST:%.+]] = bitcast {{.+}} [[VEC_ADDR_VAL]] to
// CHECK-DAG: call void @llvm.memcpy{{.+}}({{.+}}* align {{[0-9]+}} [[VEC_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[VEC_ADDR_BCAST]],

// s_arr
// CHECK-DAG: [[S_ARR_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[S_ARR_ADDR]],
// CHECK-DAG: [[S_ARR_BGN:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_ADDR_BCAST:%.+]] = bitcast {{.+}}* [[S_ARR_ADDR_VAL]] to
// CHECK-DAG: [[S_ARR_BGN_GEP:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_BGN]],
// CHECK-DAG: [[S_ARR_EMPTY:%.+]] = icmp {{.+}} [[S_ARR_BGN]], [[S_ARR_BGN_GEP]]
// CHECK-DAG: br {{.+}} [[S_ARR_EMPTY]], label %[[CPY_DONE:.+]], label %[[CPY_BODY:.+]]
// CHECK-DAG: [[CPY_BODY]]:
// CHECK-DAG: call void @llvm.memcpy{{.+}}(
// CHECK-DAG: [[CPY_DONE]]:

// var
// CHECK-DAG: [[VAR_ADDR_REF:%.+]] = load {{.+}}*, {{.+}}* [[TMP]],
// CHECK-DAG: [[VAR_PRIV_BCAST:%.+]] = bitcast {{.+}}* [[VAR_PRIV]] to
// CHECK-DAG: [[VAR_ADDR_BCAST:%.+]] = bitcast {{.+}}* [[VAR_ADDR_REF]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[VAR_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[VAR_ADDR_BCAST]],
// CHECK-DAG: store {{.+}}* [[VAR_PRIV]], {{.+}}** [[TMP_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(

// call destructors: var..
// CHECK-DAG: call {{.+}} [[S_FLOAT_TY_DEF_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])

// ..and s_arr
// CHECK: {{.+}}:
// CHECK: [[S_ARR_EL_PAST:%.+]] = phi [[S_FLOAT_TY]]*
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = getelementptr {{.+}}, {{.+}} [[S_ARR_EL_PAST]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_DESTR]]([[S_FLOAT_TY]]* [[S_ARR_PRIV_ITEM]])

// CHECK: ret void

// template tmain with S_INT_TY
// CHECK-LABEL: define{{.*}} i{{[0-9]+}} @{{.+}}tmain{{.+}}()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOAD_FUN_0:@.+]](
// CHECK: call {{.*}} [[S_INT_TY_DEF_DESTR:@.+]]([[S_INT_TY]]* [[TEST]])

// CHECK: define{{.+}} [[OFFLOAD_FUN_0]](i{{[0-9]+}} [[T_VAR_IN:%.+]], [2 x i{{[0-9]+}}]* {{.+}} [[VEC_IN:%.+]], [2 x [[S_INT_TY]]]* {{.+}} [[S_ARR_IN:%.+]], [[S_INT_TY]]* {{.+}} [[VAR_IN:%.+]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i{{[0-9]+}}]*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]*)* [[OMP_OUTLINED_0:@.+]] to void
// CHECK: ret

// CHECK: define internal void [[OMP_OUTLINED_0]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i{{[0-9]+}}*{{.+}} [[T_VAR_IN:%.+]], [2 x i{{[0-9]+}}]* {{.+}} [[VEC_IN:%.+]], [2 x [[S_INT_TY]]]* {{.+}} [[S_ARR_IN:%.+]], [[S_INT_TY]]* {{.+}} [[VAR_IN:%.+]])

// addr alloca's
// CHECK: alloca i{{[0-9]+}}*,
// CHECK: alloca i{{[0-9]+}}*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: [[TMP:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: [[TMP1:%.+]] = alloca [[S_INT_TY]]*,

// skip loop alloca's
// CHECK: [[OMP_IV:.omp.iv+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_LB:.omp.comb.lb+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_UB:.omp.comb.ub+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_ST:.omp.stride+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST:.omp.is_last+]] = alloca i{{[0-9]+}},

// private alloca's
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[TMP_PRIV:%.+]] = alloca [[S_INT_TY]]*,

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// init addr alloca's with input values
// CHECK-DAG: store {{.+}} [[T_VAR_IN]], {{.+}}* [[T_VAR_ADDR]],
// CHECK-DAG: store {{.+}} [[VEC_IN]], {{.+}} [[VEC_ADDR]],
// CHECK-DAG: store {{.+}} [[S_ARR_IN]], {{.+}} [[S_ARR_ADDR]],
// CHECK-DAG: store {{.+}} [[VAR_IN]], {{.+}} [[VAR_ADDR]],

// init private alloca's with addr alloca's
// t-var
// CHECK-DAG: [[T_VAR_ADDR_REF:%.+]] = load {{.+}}*, {{.+}}** [[T_VAR_ADDR]],
// CHECK-DAG: [[T_VAR_ADDR_VAL:%.+]] = load {{.+}}, {{.+}}* [[T_VAR_ADDR_REF]],
// CHECK-DAG: store {{.+}} [[T_VAR_ADDR_VAL]], {{.+}} [[T_VAR_PRIV]],

// vec
// CHECK-DAG: [[VEC_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[VEC_ADDR]],
// CHECK-DAG: [[VEC_PRIV_BCAST:%.+]] = bitcast {{.+}} [[VEC_PRIV]] to
// CHECK-DAG: [[VEC_ADDR_BCAST:%.+]] = bitcast {{.+}} [[VEC_ADDR_VAL]] to
// CHECK-DAG: call void @llvm.memcpy{{.+}}({{.+}}* align {{[0-9]+}} [[VEC_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[VEC_ADDR_BCAST]],

// s_arr
// CHECK-DAG: [[S_ARR_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[S_ARR_ADDR]],
// CHECK-DAG: [[S_ARR_BGN:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_ADDR_BCAST:%.+]] = bitcast {{.+}}* [[S_ARR_ADDR_VAL]] to
// CHECK-DAG: [[S_ARR_BGN_GEP:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_BGN]],
// CHECK-DAG: [[S_ARR_EMPTY:%.+]] = icmp {{.+}} [[S_ARR_BGN]], [[S_ARR_BGN_GEP]]
// CHECK-DAG: br {{.+}} [[S_ARR_EMPTY]], label %[[CPY_DONE:.+]], label %[[CPY_BODY:.+]]
// CHECK-DAG: [[CPY_BODY]]:
// CHECK-DAG: call void @llvm.memcpy{{.+}}(
// CHECK-DAG: [[CPY_DONE]]:

// var
// CHECK-DAG: [[TMP_REF:%.+]] = load {{.+}}*, {{.+}}* [[TMP1]],
// CHECK-DAG: [[VAR_PRIV_BCAST:%.+]] = bitcast {{.+}}* [[VAR_PRIV]] to
// CHECK-DAG: [[TMP_REF_BCAST:%.+]] = bitcast {{.+}}* [[TMP_REF]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[VAR_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[TMP_REF_BCAST]],
// CHECK-DAG: store {{.+}}* [[VAR_PRIV]], {{.+}}** [[TMP_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// pass private alloca's to fork
// CHECK-DAG: [[T_VAR_PRIV_VAL:%.+]] = load {{.+}}, {{.+}}* [[T_VAR_PRIV]],
// not dag to distinguish with S_VAR_CAST
// CHECK-64: [[T_VAR_CAST_CONV:%.+]] = bitcast {{.+}}* [[T_VAR_CAST:%.+]] to
// CHECK-64-DAG: store {{.+}} [[T_VAR_PRIV_VAL]], {{.+}} [[T_VAR_CAST_CONV]],
// CHECK-32: store {{.+}} [[T_VAR_PRIV_VAL]], {{.+}} [[T_VAR_CAST:%.+]],
// CHECK-DAG: [[T_VAR_CAST_VAL:%.+]] = load {{.+}}, {{.+}}* [[T_VAR_CAST]],
// CHECK-DAG: [[TMP_PRIV_VAL:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** [[TMP_PRIV]],
// CHECK: call{{.+}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}}[[OMP_PARFOR_OUTLINED_0:@.+]] to void ({{.+}})*), {{.+}}, {{.+}}, [2 x i{{[0-9]+}}]* [[VEC_PRIV]], i{{[0-9]+}} [[T_VAR_CAST_VAL]], [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]], [[S_INT_TY]]* [[TMP_PRIV_VAL]])
// CHECK: call void @__kmpc_for_static_fini(

// call destructors: var..
// CHECK-DAG: call {{.+}} [[S_INT_TY_DEF_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])

// ..and s_arr
// CHECK: {{.+}}:
// CHECK: [[S_ARR_EL_PAST:%.+]] = phi [[S_INT_TY]]*
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = getelementptr {{.+}}, {{.+}} [[S_ARR_EL_PAST]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_DESTR]]([[S_INT_TY]]* [[S_ARR_PRIV_ITEM]])

// CHECK: ret void

// By OpenMP specifications, 'firstprivate' applies to both distribute and parallel for.
// However, the support for 'firstprivate' of 'parallel' is only used when 'parallel'
// is found alone. Therefore we only have one 'firstprivate' support for 'parallel for'
// in combination
// CHECK: define internal void [[OMP_PARFOR_OUTLINED_0]]({{.+}}, {{.+}}, {{.+}}, {{.+}}, [2 x i{{[0-9]+}}]* {{.+}} [[VEC_IN:%.+]], i{{[0-9]+}} [[T_VAR_IN:%.+]], [2 x [[S_INT_TY]]]* {{.+}} [[S_ARR_IN:%.+]], [[S_INT_TY]]* {{.+}} [[VAR_IN:%.+]])

// addr alloca's
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: [[TMP:%.+]] = alloca [[S_INT_TY]]*,

// skip loop alloca's
// CHECK: [[OMP_IV:.omp.iv+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_LB:.omp.lb+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_UB:.omp.ub+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_ST:.omp.stride+]] = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST:.omp.is_last+]] = alloca i{{[0-9]+}},

// private alloca's
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[TMP_PRIV:%.+]] = alloca [[S_INT_TY]]*,

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]

// init addr alloca's with input values
// CHECK-DAG: store {{.+}} [[VEC_IN]], {{.+}} [[VEC_ADDR]],
// CHECK-DAG: store {{.+}} [[T_VAR_IN]], {{.+}}* [[T_VAR_ADDR]],
// CHECK-DAG: store {{.+}} [[S_ARR_IN]], {{.+}} [[S_ARR_ADDR]],
// CHECK-DAG: store {{.+}} [[VAR_IN]], {{.+}} [[VAR_ADDR]],

// init private alloca's with addr alloca's
// vec
// CHECK-DAG: [[VEC_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[VEC_ADDR]],
// CHECK-DAG: [[VEC_PRIV_BCAST:%.+]] = bitcast {{.+}} [[VEC_PRIV]] to
// CHECK-DAG: [[VEC_ADDR_BCAST:%.+]] = bitcast {{.+}} [[VEC_ADDR_VAL]] to
// CHECK-DAG: call void @llvm.memcpy{{.+}}({{.+}}* align {{[0-9]+}} [[VEC_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[VEC_ADDR_BCAST]],

// s_arr
// CHECK-DAG: [[S_ARR_ADDR_VAL:%.+]] = load {{.+}}*, {{.+}}** [[S_ARR_ADDR]],
// CHECK-DAG: [[S_ARR_BGN:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_ADDR_BCAST:%.+]] = bitcast {{.+}}* [[S_ARR_ADDR_VAL]] to
// CHECK-DAG: [[S_ARR_BGN_GEP:%.+]] = getelementptr {{.+}}, {{.+}}* [[S_ARR_BGN]],
// CHECK-DAG: [[S_ARR_EMPTY:%.+]] = icmp {{.+}} [[S_ARR_BGN]], [[S_ARR_BGN_GEP]]
// CHECK-DAG: br {{.+}} [[S_ARR_EMPTY]], label %[[CPY_DONE:.+]], label %[[CPY_BODY:.+]]
// CHECK-DAG: [[CPY_BODY]]:
// CHECK-DAG: call void @llvm.memcpy{{.+}}(
// CHECK-DAG: [[CPY_DONE]]:

// var
// CHECK-DAG: [[VAR_ADDR_REF:%.+]] = load {{.+}}*, {{.+}}* [[TMP]],
// CHECK-DAG: [[VAR_PRIV_BCAST:%.+]] = bitcast {{.+}}* [[VAR_PRIV]] to
// CHECK-DAG: [[VAR_ADDR_BCAST:%.+]] = bitcast {{.+}}* [[VAR_ADDR_REF]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[VAR_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[VAR_ADDR_BCAST]],
// CHECK-DAG: store {{.+}}* [[VAR_PRIV]], {{.+}}** [[TMP_PRIV]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(

// call destructors: var..
// CHECK-DAG: call {{.+}} [[S_INT_TY_DEF_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])

// ..and s_arr
// CHECK: {{.+}}:
// CHECK: [[S_ARR_EL_PAST:%.+]] = phi [[S_INT_TY]]*
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = getelementptr {{.+}}, {{.+}} [[S_ARR_EL_PAST]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_DESTR]]([[S_INT_TY]]* [[S_ARR_PRIV_ITEM]])

// CHECK: ret void

#endif
