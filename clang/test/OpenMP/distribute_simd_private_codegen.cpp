// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32

// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
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
#pragma omp distribute simd private(t_var, vec, s_arr, s_arr, var, var)
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

    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN]]()
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[OMP_OUTLINED:@.+]] to {{.+}})
    #pragma omp target
    #pragma omp teams
#pragma omp distribute simd private(g, g1, svar, sfvar)
    for (int i = 0; i < 2; ++i) {
      // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_OUTLINED]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
      // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca double,
      // LAMBDA: [[G1_PRIVATE_ADDR:%.+]] = alloca double,
      // LAMBDA: [[TMP_PRIVATE_ADDR:%.+]] = alloca double*,
      // LAMBDA: [[SVAR_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
      // LAMBDA: [[SFVAR_PRIVATE_ADDR:%.+]] = alloca float,
      // LAMBDA: store double* [[G1_PRIVATE_ADDR]], double** [[TMP_PRIVATE_ADDR]],
      g = 1;
      g1 = 1;
      svar = 3;
      sfvar = 4.0;
      // LAMBDA: call {{.*}}void @__kmpc_for_static_init_4(
      // LAMBDA: store double 1.0{{.+}}, double* [[G_PRIVATE_ADDR]],
      // LAMBDA: store i{{[0-9]+}} 3, i{{[0-9]+}}* [[SVAR_PRIVATE_ADDR]],
      // LAMBDA: store float 4.0{{.+}}, float* [[SFVAR_PRIVATE_ADDR]],
      // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: store double* [[G_PRIVATE_ADDR]], double** [[G_PRIVATE_ADDR_REF]],
      // LAMBDA: [[TMP_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[G1_PRIVATE_ADDR_FROM_TMP:%.+]] = load double*, double** [[TMP_PRIVATE_ADDR]],
      // LAMBDA: store double* [[G1_PRIVATE_ADDR_FROM_TMP]], double** [[TMP_PRIVATE_ADDR_REF]],
      // LAMBDA: [[SVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
      // LAMBDA: store i{{[0-9]+}}* [[SVAR_PRIVATE_ADDR]], i{{[0-9]+}}** [[SVAR_PRIVATE_ADDR_REF]]
      // LAMBDA: [[SFVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
      // LAMBDA: store float* [[SFVAR_PRIVATE_ADDR]], float** [[SFVAR_PRIVATE_ADDR_REF]]
      // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
      // LAMBDA: call {{.*}}void @__kmpc_for_static_fini(
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
#pragma omp distribute simd private(t_var, vec, s_arr, s_arr, var, var, svar)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  int i;

  #pragma omp target
  #pragma omp teams
#pragma omp distribute simd
  for (i = 0; i < 2; ++i) {
    ;
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

// CHECK: define{{.+}} [[OFFLOAD_FUN]]()
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[OMP_OUTLINED:@.+]] to void
// CHECK: ret
//
// CHECK: define internal void [[OMP_OUTLINED]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK-NOT: alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK-NOT: alloca [[S_FLOAT_TY]],
// CHECK: [[S_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_FLOAT_TY]]*
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define{{.*}} i{{[0-9]+}} [[TMAIN_INT:@.+]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOAD_FUN_1:@.+]](
// CHECK: ret


// CHECK: define internal void [[OFFLOAD_FUN_1]]()
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[OMP_OUTLINED_1:@.+]] to void
// CHECK: ret
//
// CHECK: define internal void [[OMP_OUTLINED_1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK-NOT: alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK-NOT: alloca [[S_INT_TY]],
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_INT_TY]]*
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: !{!"llvm.loop.vectorize.enable", i1 true}
#endif
