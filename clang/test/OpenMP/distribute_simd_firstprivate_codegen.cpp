// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
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
#pragma omp distribute simd firstprivate(t_var, vec, s_arr, s_arr, var, var)
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
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, {{.+}}, {{.+}}* [[OMP_OUTLINED:@.+]] to {{.+}})
    #pragma omp target
    #pragma omp teams
#pragma omp distribute simd firstprivate(g, g1, svar, sfvar)
    for (int i = 0; i < 2; ++i) {
      // LAMBDA: define internal{{.*}} void [[OMP_OUTLINED]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, double*{{.*}} [[G_IN:%.+]], double*{{.+}} [[G1_IN:%.+]], i{{[0-9]+}}*{{.+}} [[SVAR_IN:%.+]], float*{{.+}} [[SFVAR_IN:%.+]])
      // Private alloca's for conversion
      // LAMBDA: [[G_ADDR:%.+]] = alloca double*,
      // LAMBDA: [[G1_ADDR:%.+]] = alloca double*,
      // LAMBDA: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
      // LAMBDA: [[SFVAR_ADDR:%.+]] = alloca float*,
      // LAMBDA: [[G1_REF:%.+]] = alloca double*,

      // Actual private variables to be used in the body (tmp is used for the reference type)
      // LAMBDA: [[G_PRIVATE:%.+]] = alloca double,
      // LAMBDA: [[G1_PRIVATE:%.+]] = alloca double,
      // LAMBDA: [[TMP_PRIVATE:%.+]] = alloca double*,
      // LAMBDA: [[SVAR_PRIVATE:%.+]] = alloca i{{[0-9]+}},
      // LAMBDA: [[SFVAR_PRIVATE:%.+]] = alloca float,

      // Store input parameter addresses into private alloca's for conversion
      // LAMBDA: store double* [[G_IN]], double** [[G_ADDR]],
      // LAMBDA: store double* [[G1_IN]], double** [[G1_ADDR]],
      // LAMBDA: store i{{[0-9]+}}* [[SVAR_IN]], i{{[0-9]+}}** [[SVAR_ADDR]],
      // LAMBDA: store float* [[SFVAR_IN]], float** [[SFVAR_ADDR]],

      // LAMBDA-DAG: [[G_ADDR_VAL:%.+]] = load double*, double** [[G_ADDR]],
      // LAMBDA-DAG: [[G1_ADDR_VAL:%.+]] = load double*, double** [[G1_ADDR]],
      // LAMBDA-DAG: [[SVAR_ADDR_VAL:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SVAR_ADDR]],
      // LAMBDA-DAG: [[SFVAR_ADDR_VAL:%.+]] = load float*, float** [[SFVAR_ADDR]],
      // LAMBDA-DAG: store double* [[G1_ADDR_VAL]], double** [[G1_REF]],

      // LAMBDA-DAG: [[G_VAL:%.+]] = load{{.+}} double, double* [[G_ADDR_VAL]],
      // LAMBDA-DAG: store double [[G_VAL]], double* [[G_PRIVATE]],
      // LAMBDA-DAG: [[G1_VAL_REF:%.+]] = load double*, double** [[G1_REF]],
      // LAMBDA-DAG: [[G1_VAL:%.+]] = load{{.+}} double, double* %
      // LAMBDA-DAG: store double [[G1_VAL]], double* [[G1_PRIVATE]],
      // LAMBDA-DAG: store double* [[G1_PRIVATE]], double** [[TMP_PRIVATE]],
      // LAMBDA-DAG: [[SVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SVAR_ADDR_VAL]],
      // LAMBDA-DAG: store i{{[0-9]+}} [[SVAR_VAL]], i{{[0-9]+}}* [[SVAR_PRIVATE]],
      // LAMBDA-DAG: [[SFVAR_VAL:%.+]] = load float, float* [[SFVAR_ADDR_VAL]],
      // LAMBDA-DAG: store float [[SFVAR_VAL]], float* [[SFVAR_PRIVATE]],

      // LAMBDA: call {{.*}}void @__kmpc_for_static_init_4(
      g += 1;
      g1 += 1;
      svar += 3;
      sfvar += 4.0;
      // LAMBDA-DAG: [[G_VAL:%.+]] = load double, double* [[G_PRIVATE]],
      // LAMBDA-DAG: [[G_NEXT:%.+]] = fadd double [[G_VAL]], 1.{{.+}}
      // LAMBDA-DAG: store double [[G_NEXT]], double* [[G_PRIVATE]],
      // LAMBDA-DAG: [[TMP_VAL1:%.+]] = load double*, double** [[TMP_PRIVATE]],
      // LAMBDA-DAG: [[TMP_VAL_VAL1:%.+]] = load{{.*}} double, double* [[TMP_VAL1]],
      // LAMBDA-DAG: [[TMP_ADD:%.+]] = fadd double [[TMP_VAL_VAL1]], 1.{{.+}}
      // LAMBDA-DAG: store{{.*}} double [[TMP_ADD]], double* [[TMP_VAL1]],
      // LAMBDA-DAG: [[SVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SVAR_PRIVATE]],
      // LAMBDA-DAG: [[SVAR_ADD:%.+]] = add{{.*}} i{{[0-9]+}} [[SVAR_VAL]], 3
      // LAMBDA-DAG: store i{{[0-9]+}} [[SVAR_ADD]], i{{[0-9]+}}* [[SVAR_PRIVATE]],
      // LAMBDA-DAG: [[SFVAR_VAL:%.+]] = load float, float* [[SFVAR_PRIVATE]],
      // LAMBDA-DAG: [[SFVAR_CONV_VAL1:%.+]] = fpext float [[SFVAR_VAL]] to double
      // LAMBDA-DAG: [[SFVAR_ADD:%.+]] = fadd double [[SFVAR_CONV_VAL1]], 4.{{.+}}
      // LAMBDA-DAG: [[SFVAR_CONV_VAL2:%.+]] = fptrunc double [[SFVAR_ADD]] to float
      // LAMBDA-DAG: store float [[SFVAR_CONV_VAL2:%.+]], float* [[SFVAR_PRIVATE]],

      // call inner lambda (use refs to private alloca's)
      // LAMBDA: [[GEP_0:%.+]] = getelementptr{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: store double* [[G_PRIVATE]], double** [[GEP_0]],
      // LAMBDA: [[GEP_1:%.+]] = getelementptr{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[TMP_PAR:%.+]] = load double*, double** [[TMP_PRIVATE]],
      // LAMBDA: store double* [[TMP_PAR]], double** [[GEP_1]],
      // LAMBDA: [[GEP_2:%.+]] = getelementptr{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
      // LAMBDA: store i{{[0-9]+}}* [[SVAR_PRIVATE]], i{{[0-9]+}}** [[GEP_2]],
      // LAMBDA: [[GEP_3:%.+]] = getelementptr{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
      // LAMBDA: store float* [[SFVAR_PRIVATE]], float** [[GEP_3]],
      // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}* {{.+}})
      // LAMBDA: call {{.*}}void @__kmpc_for_static_fini(
      [&]() {
	// LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
	// LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
	g += 2;
	g1 += 2;
	svar += 4;
	sfvar += 8.0;
	// LAMBDA-DAG: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
	// LAMBDA-DAG: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
	// LAMBDA-DAG: [[G_REF:%.+]] = load double*, double** [[G_PTR_REF]],
	// LAMBDA-DAG: [[G_REF_VAL:%.+]] = load double, double* [[G_REF]],
	// LAMBDA-DAG: [[G_REF_ADD:%.+]] = fadd double [[G_REF_VAL]], 2.{{.+}}
	// LAMBDA-DAG: store double [[G_REF_ADD]], double* [[G_REF]]

	// LAMBDA-DAG: [[TMP_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
	// LAMBDA-DAG: [[G1_REF:%.+]] = load double*, double** [[TMP_PTR_REF]]
	// LAMBDA-DAG: [[G1_REF_VAL:%.+]] = load double, double* [[G1_REF]],
	// LAMBDA-DAG: [[G1_ADD:%.+]] = fadd double [[G1_REF_VAL]], 2.{{.+}}
	// LAMBDA-DAG: store double [[G1_ADD]], double* [[G1_REF]],

	// LAMBDA-DAG: [[SVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
	// LAMBDA-DAG: [[SVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SVAR_PTR_REF]]
	// LAMBDA-DAG: [[SVAR_REF_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SVAR_REF]]
	// LAMBDA-DAG: [[SVAR_ADD:%.+]] = add{{.*}} i{{[0-9]+}} [[SVAR_REF_VAL]], 4
	// LAMBDA-DAG: store i{{[0-9]+}} [[SVAR_ADD]], i{{[0-9]+}}* [[SVAR_REF]]

	// LAMBDA-DAG: [[SFVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
	// LAMBDA-DAG: [[SFVAR_REF:%.+]] = load float*, float** [[SFVAR_PTR_REF]]
	// LAMBDA-DAG: [[SFVAR_REF_VAL:%.+]] = load float, float* [[SFVAR_REF]]
	// LAMBDA-DAG: [[SFVAR_REF_CONV:%.+]] = fpext float [[SFVAR_REF_VAL]] to double
	// LAMBDA-DAG: [[SFVAR_ADD:%.+]] = fadd double [[SFVAR_REF_CONV]], 8.{{.+}}
	// LAMBDA-DAG: [[SFVAR_ADD_CONV:%.+]] = fptrunc double [[SFVAR_ADD]] to float
	// LAMBDA-DAG: store float [[SFVAR_ADD_CONV]], float* [[SFVAR_REF]],
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
  #pragma omp distribute simd firstprivate(t_var, vec, s_arr, s_arr, var, var, svar)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  return tmain<int>();
  #endif
}

// CHECK: define{{.*}} i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOAD_FUN:@.+]](
// CHECK: ret

// CHECK: define{{.+}} [[OFFLOAD_FUN]](
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 5, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i{{[0-9]+}}]*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]*, i{{[0-9]+}}*)* [[OMP_OUTLINED:@.+]] to void
// CHECK: ret
//
// CHECK: define internal void [[OMP_OUTLINED]](i{{[0-9]+}}*{{.+}}, i{{[0-9]+}}*{{.+}}, i{{[0-9]+}}*{{.+}} [[T_VAR_IN:%.+]], [2 x i{{[0-9]+}}]*{{.*}} [[VEC_IN:%.+]], [2 x [[S_FLOAT_TY]]]*{{.*}} [[S_ARR_IN:%.+]], [[S_FLOAT_TY]]*{{.*}} [[VAR_IN:%.+]], i{{[0-9]+}}*{{.+}} [[SVAR_IN:%.+]])

// CHECK: alloca i{{[0-9]+}}*,
// CHECK: alloca i{{[0-9]+}}*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// CHECK: [[TMP:%.+]] = alloca [[S_FLOAT_TY]]*,

// discard omp loop variables
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},

// CHECK-DAG: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK-DAG: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK-DAG: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK-DAG: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK-DAG: [[TMP_PRIV:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SVAR_PRIV:%.+]] = alloca i{{[0-9]+}},

// CHECK: store i{{[0-9]+}}* [[T_VAR_IN]], i{{[0-9]+}}** [[T_VAR_ADDR]],
// CHECK: store [2 x i{{[0-9]+}}]* [[VEC_IN]], [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK: store [2 x [[S_FLOAT_TY]]]* [[S_ARR_IN]], [2 x [[S_FLOAT_TY]]]** [[S_ARR_ADDR]],
// CHECK: store [[S_FLOAT_TY]]* [[VAR_IN]], [[S_FLOAT_TY]]** [[VAR_ADDR]],
// CHECK: store i{{[0-9]+}}* [[SVAR_IN]], i{{[0-9]+}}** [[SVAR_ADDR]],

// init t_var
// CHECK-DAG: [[T_VAR_ADDR_CONV_VAL_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[T_VAR_ADDR]],
// CHECK-DAG: [[T_VAR_ADDR_CONV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_ADDR_CONV_VAL_REF]],
// CHECK-DAG: store i{{[0-9]+}} [[T_VAR_ADDR_CONV_VAL]], i{{[0-9]+}}* [[T_VAR_PRIV]],

// init vec
// CHECK-DAG: [[VEC_ADDR_VAL:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK-DAG: [[VEC_ADDR_VAL_BCAST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_ADDR_VAL]] to i{{[0-9]+}}*
// CHECK-DAG: [[VEC_PRIV_BCAST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i{{[0-9]+}}*
// CHECK-DAG: call void @llvm.memcpy.{{.*}}(i{{[0-9]+}}* align {{[0-9]+}} [[VEC_PRIV_BCAST]], i{{[0-9]+}}* align {{[0-9]+}} [[VEC_ADDR_VAL_BCAST]],{{.+}})

// init s_arr
// CHECK-DAG: [[S_ARR_ADDR_VAL:%.+]] = load [2 x [[S_FLOAT_TY]]]*, [2 x [[S_FLOAT_TY]]]** [[S_ARR_ADDR]],
// CHECK-DAG: [[S_ARR_ADDR_BCAST:%.+]] = bitcast [2 x [[S_FLOAT_TY]]]* [[S_ARR_ADDR_VAL]] to [[S_FLOAT_TY]]*
// CHECK-DAG: [[S_ARR_PRIV_BGN:%.+]] = getelementptr{{.+}} [2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* [[S_ARR_PRIV]]{{.+}}
// CHECK-DAG: [[S_ARR_PRIV_NEXT:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[S_ARR_PRIV_BGN]]{{.+}}
// CHECK-DAG: [[S_ARR_IS_EMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[S_ARR_PRIV_BGN]], [[S_ARR_PRIV_NEXT]]
// CHECK-DAG: br i1 [[S_ARR_IS_EMPTY]], label %[[S_ARR_CPY_DONE:.+]], label %[[S_ARR_CPY_BODY:.+]]

// CHECK-DAG: [[S_ARR_CPY_BODY]]:
// CHECK-DAG: [[S_ARR_SRC_PAST:%.+]] = phi{{.+}} [ [[S_ARR_ADDR_BCAST]],{{.+}} ], [ [[S_ARR_SRC:%.+]],{{.+}} ]
// CHECK-DAG: [[S_ARR_DST_PAST:%.+]] = phi{{.+}} [ [[S_ARR_PRIV_BGN]],{{.+}} ], [ [[S_ARR_DST:%.+]],{{.+}} ]
// CHECK-DAG: [[S_ARR_SRC_BCAST:%.+]] = bitcast{{.+}} [[S_ARR_SRC_PAST]] to{{.+}}
// CHECK-DAG: [[S_ARR_DST_BCAST:%.+]] = bitcast{{.+}} [[S_ARR_DST_PAST]] to{{.+}}
// CHECK-DAG: call{{.+}} @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[S_ARR_DST_BCAST]], {{.+}}* align {{[0-9]+}} [[S_ARR_SRC_BCAST]]{{.+}})
// CHECK-DAG: [[S_ARR_SRC]] = getelementptr{{.+}}
// CHECK-DAG: [[S_ARR_DST]] = getelementptr{{.+}}
// CHECK-DAG: [[S_ARR_CPY_FIN:%.+]] = icmp{{.+}} [[S_ARR_DST]], [[S_ARR_PRIV_NEXT]]
// CHECK-DAG: br i1 [[S_ARR_CPY_FIN]], label %[[S_ARR_CPY_DONE]], label %[[S_ARR_CPY_BODY]]
// CHECK-DAG: [[S_ARR_CPY_DONE]]:

// init var
// CHECK-DAG: [[VAR_ADDR_VAL:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[VAR_ADDR]],
// CHECK-DAG: store{{.+}} [[VAR_ADDR_VAL]],{{.+}} [[TMP]],
// CHECK-DAG: [[TMP_VAL:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[TMP]],
// CHECK-DAG: [[VAR_PRIV_BCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR_PRIV]] to{{.+}}
// CHECK-DAG: [[TMP_BCAST:%.+]] = bitcast [[S_FLOAT_TY]]* %{{.+}} to{{.+}}
// CHECK-DAG: call{{.+}} @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[VAR_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[TMP_BCAST]],{{.+}})
// CHECK-DAG: store [[S_FLOAT_TY]]* [[VAR_PRIV]], [[S_FLOAT_TY]]** [[TMP_PRIV]],

// init svar
// CHECK-DAG: [[SVAR_CONV_VAL_REF:%.+]] = load{{.+}},{{.+}} [[SVAR_ADDR]],
// CHECK-DAG: [[SVAR_CONV_VAL:%.+]] = load{{.+}},{{.+}} [[SVAR_CONV_VAL_REF]],
// CHECK-DAG: store{{.+}} [[SVAR_CONV_VAL]],{{.+}} [[SVAR_PRIV]],

// CHECK-DAG: store i{{[0-9]+}} 0, i{{[0-9]+}}* %.omp{{.+}},
// CHECK-DAG: store i{{[0-9]+}} 1, i{{[0-9]+}}* %.omp{{.+}},
// CHECK-DAG: store i{{[0-9]+}} 1, i{{[0-9]+}}* %.omp{{.+}},
// CHECK-DAG: store i{{[0-9]+}} 0, i{{[0-9]+}}* %.omp{{.+}},

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// Template
// CHECK: define{{.*}} i{{[0-9]+}} [[TMAIN_INT:@.+]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOAD_FUN_1:@.+]](
// CHECK: ret

// CHECK: define{{.+}} [[OFFLOAD_FUN_1]](
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i{{[0-9]+}}]*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]*)* [[OMP_OUTLINED_1:@.+]] to void
// CHECK: ret
//
// CHECK: define internal void [[OMP_OUTLINED_1]](i{{[0-9]+}}*{{.+}}, i{{[0-9]+}}*{{.+}}, i{{[0-9]+}}*{{.+}} [[T_VAR_IN:%.+]], [2 x i{{[0-9]+}}]*{{.*}} [[VEC_IN:%.+]], [2 x [[S_INT_TY]]]*{{.*}} [[S_ARR_IN:%.+]], [[S_INT_TY]]*{{.*}} [[VAR_IN:%.+]])

// CHECK: alloca i{{[0-9]+}}*,
// CHECK: alloca i{{[0-9]+}}*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}}*,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: [[TMP:%.+]] = alloca [[S_INT_TY]]*,

// discard omp loop variables
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},
// CHECK: {{.*}} = alloca i{{[0-9]+}},

// CHECK-DAG: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK-DAG: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK-DAG: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK-DAG: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK-DAG: [[TMP_PRIV:%.+]] = alloca [[S_INT_TY]]*,

// CHECK: store i{{[0-9]+}}* [[T_VAR_IN]], i{{[0-9]+}}** [[T_VAR_ADDR]],
// CHECK: store [2 x i{{[0-9]+}}]* [[VEC_IN]], [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK: store [2 x [[S_INT_TY]]]* [[S_ARR_IN]], [2 x [[S_INT_TY]]]** [[S_ARR_ADDR]],
// CHECK: store [[S_INT_TY]]* [[VAR_IN]], [[S_INT_TY]]** [[VAR_ADDR]],

// init t_var
// CHECK-DAG: [[T_VAR_ADDR_CONV_VAL_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[T_VAR_ADDR]],
// CHECK-DAG: [[T_VAR_ADDR_CONV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_ADDR_CONV_VAL_REF]],
// CHECK-DAG: store i{{[0-9]+}} [[T_VAR_ADDR_CONV_VAL]], i{{[0-9]+}}* [[T_VAR_PRIV]],

// init vec
// CHECK-DAG: [[VEC_ADDR_VAL:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK-DAG: [[VEC_ADDR_VAL_BCAST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_ADDR_VAL]] to i{{[0-9]+}}*
// CHECK-DAG: [[VEC_PRIV_BCAST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i{{[0-9]+}}*
// CHECK-DAG: call void @llvm.memcpy.{{.*}}(i{{[0-9]+}}* align {{[0-9]+}} [[VEC_PRIV_BCAST]], i{{[0-9]+}}* align {{[0-9]+}} [[VEC_ADDR_VAL_BCAST]],{{.+}})

// init s_arr
// CHECK-DAG: [[S_ARR_ADDR_VAL:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** [[S_ARR_ADDR]],
// CHECK-DAG: [[S_ARR_ADDR_BCAST:%.+]] = bitcast [2 x [[S_INT_TY]]]* [[S_ARR_ADDR_VAL]] to [[S_INT_TY]]*
// CHECK-DAG: [[S_ARR_PRIV_BGN:%.+]] = getelementptr{{.+}} [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]]{{.+}}
// CHECK-DAG: [[S_ARR_PRIV_NEXT:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_PRIV_BGN]]{{.+}}
// CHECK-DAG: [[S_ARR_IS_EMPTY:%.+]] = icmp eq [[S_INT_TY]]* [[S_ARR_PRIV_BGN]], [[S_ARR_PRIV_NEXT]]
// CHECK-DAG: br i1 [[S_ARR_IS_EMPTY]], label %[[S_ARR_CPY_DONE:.+]], label %[[S_ARR_CPY_BODY:.+]]

// CHECK-DAG: [[S_ARR_CPY_BODY]]:
// CHECK-DAG: [[S_ARR_SRC_PAST:%.+]] = phi{{.+}} [ [[S_ARR_ADDR_BCAST]],{{.+}} ], [ [[S_ARR_SRC:%.+]],{{.+}} ]
// CHECK-DAG: [[S_ARR_DST_PAST:%.+]] = phi{{.+}} [ [[S_ARR_PRIV_BGN]],{{.+}} ], [ [[S_ARR_DST:%.+]],{{.+}} ]
// CHECK-DAG: [[S_ARR_SRC_BCAST:%.+]] = bitcast{{.+}} [[S_ARR_SRC_PAST]] to{{.+}}
// CHECK-DAG: [[S_ARR_DST_BCAST:%.+]] = bitcast{{.+}} [[S_ARR_DST_PAST]] to{{.+}}
// CHECK-DAG: call{{.+}} @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[S_ARR_DST_BCAST]], {{.+}}* align {{[0-9]+}} [[S_ARR_SRC_BCAST]]{{.+}})
// CHECK-DAG: [[S_ARR_SRC]] = getelementptr{{.+}}
// CHECK-DAG: [[S_ARR_DST]] = getelementptr{{.+}}
// CHECK-DAG: [[S_ARR_CPY_FIN:%.+]] = icmp{{.+}} [[S_ARR_DST]], [[S_ARR_PRIV_NEXT]]
// CHECK-DAG: br i1 [[S_ARR_CPY_FIN]], label %[[S_ARR_CPY_DONE]], label %[[S_ARR_CPY_BODY]]
// CHECK-DAG: [[S_ARR_CPY_DONE]]:

// init var
// CHECK-DAG: [[VAR_ADDR_VAL:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** [[VAR_ADDR]],
// CHECK-DAG: store{{.+}} [[VAR_ADDR_VAL]],{{.+}} [[TMP]],
// CHECK-DAG: [[TMP_VAL:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** [[TMP]],
// CHECK-DAG: [[VAR_PRIV_BCAST:%.+]] = bitcast [[S_INT_TY]]* [[VAR_PRIV]] to{{.+}}
// CHECK-DAG: [[TMP_BCAST:%.+]] = bitcast [[S_INT_TY]]* %{{.+}} to{{.+}}
// CHECK-DAG: call{{.+}} @llvm.memcpy.{{.+}}({{.+}}* align {{[0-9]+}} [[VAR_PRIV_BCAST]], {{.+}}* align {{[0-9]+}} [[TMP_BCAST]],{{.+}})
// CHECK-DAG: store [[S_INT_TY]]* [[VAR_PRIV]], [[S_INT_TY]]** [[TMP_PRIV]],

// CHECK-DAG: store i{{[0-9]+}} 0, i{{[0-9]+}}* %.omp{{.+}},
// CHECK-DAG: store i{{[0-9]+}} 1, i{{[0-9]+}}* %.omp{{.+}},
// CHECK-DAG: store i{{[0-9]+}} 1, i{{[0-9]+}}* %.omp{{.+}},
// CHECK-DAG: store i{{[0-9]+}} 0, i{{[0-9]+}}* %.omp{{.+}},

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: !{!"llvm.loop.vectorize.enable", i1 true}
#endif
