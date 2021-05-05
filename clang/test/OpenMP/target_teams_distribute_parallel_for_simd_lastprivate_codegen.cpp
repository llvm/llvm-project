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

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY

// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix SIMD-ONLY
// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

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
  #pragma omp target teams distribute parallel for simd lastprivate(t_var, vec, s_arr, s_arr, var, var)
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
    // LAMBDA: call i{{[0-9]+}} @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}},
    // LAMBDA: call void [[OFFLOADING_FUN:@.+]](

    // LAMBDA: define{{.+}} void [[OFFLOADING_FUN]](
    // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}}* [[OMP_OUTLINED:@.+]] to {{.+}})
    #pragma omp target teams distribute parallel for simd lastprivate(g, g1, svar, sfvar)
    for (int i = 0; i < 2; ++i) {
      // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_OUTLINED]](i32* {{.+}}, i32* {{.+}}, {{.+}} [[G1_IN:%.+]], {{.+}} [[SVAR_IN:%.+]], {{.+}} [[SFVAR_IN:%.+]], {{.+}} [[G_IN:%.+]])
      // skip gbl and bound tid
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: [[G1_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[SVAR_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[SFVAR_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[G_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[G1_REF:%.+]] = alloca double*,
      // loop variables
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: [[OMP_IS_LAST:%.+]] = alloca i{{[0-9]+}},

      // LAMBDA-DAG: store {{.+}} [[G_IN]], {{.+}} [[G_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[G1_IN]], {{.+}} [[G1_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SVAR_IN]], {{.+}} [[SVAR_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SFVAR_IN]], {{.+}} [[SFVAR_ADDR]],

      // LAMBDA-64-DAG: [[G_TGT:%.+]] = bitcast {{.+}} [[G_ADDR]] to
      // LAMBDA-32-DAG: [[G_TGT:%.+]] = load {{.+}}, {{.+}} [[G_ADDR]],
      // LAMBDA-DAG: [[G1_TGT:%.+]] = load {{.+}}, {{.+}} [[G1_REF]],
      // LAMBDA-64-DAG: [[SVAR_TGT:%.+]] = bitcast {{.+}} [[SVAR_ADDR]] to
      // LAMBDA-DAG: [[SFVAR_TGT:%.+]] = bitcast {{.+}} [[SFVAR_ADDR]] to

      g1 = 1;
      svar = 3;
      sfvar = 4.0;
      // LAMBDA: call {{.*}}void @__kmpc_for_static_init_4(
      // LAMBDA: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[LPAR_OUTL:.+]] to
      // LAMBDA: call {{.*}}void @__kmpc_for_static_fini(

      // LAMBDA: store i32 2, i32* %
      // LAMBDA: [[OMP_IS_LAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IS_LAST]],
      // LAMBDA: [[IS_LAST_IT:%.+]] = icmp ne i{{[0-9]+}} [[OMP_IS_LAST_VAL]], 0
      // LAMBDA: br i1 [[IS_LAST_IT]], label %[[OMP_LASTPRIV_BLOCK:.+]], label %[[OMP_LASTPRIV_DONE:.+]]

      // LAMBDA: [[OMP_LASTPRIV_BLOCK]]:
      // LAMBDA-DAG: store {{.+}}, {{.+}} [[G_TGT]],
      // LAMBDA-DAG: store {{.+}}, {{.+}} [[G1_TGT]],
      // LAMBDA-64-DAG: store {{.+}}, {{.+}} [[SVAR_TGT]],
      // LAMBDA-32-DAG: store {{.+}}, {{.+}} [[SVAR_ADDR]],
      // LAMBDA-DAG: store {{.+}}, {{.+}} [[SFVAR_TGT]],
      // LAMBDA: br label %[[OMP_LASTPRIV_DONE]]
      // LAMBDA: [[OMP_LASTPRIV_DONE]]:
      // LAMBDA: ret

      // LAMBDA: define{{.*}} internal{{.*}} void @[[LPAR_OUTL]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, {{.+}}, {{.+}}, {{.+}} [[G1_IN:%.+]], {{.+}} [[SVAR_IN:%.+]], {{.+}} [[SFVAR_IN:%.+]], {{.+}} [[G_IN:%.+]])
      // skip tid and prev variables
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: alloca
      // LAMBDA: [[G1_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[SVAR_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[SFVAR_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[G_ADDR:%.+]] = alloca {{.+}},
      // LAMBDA: [[G1_REF:%.+]] = alloca double*,
      // loop variables
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: {{.+}} = alloca i{{[0-9]+}},
      // LAMBDA: [[OMP_IS_LAST:%.+]] = alloca i{{[0-9]+}},

      // LAMBDA-DAG: store {{.+}} [[G_IN]], {{.+}} [[G_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[G1_IN]], {{.+}} [[G1_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SVAR_IN]], {{.+}} [[SVAR_ADDR]],
      // LAMBDA-DAG: store {{.+}} [[SFVAR_IN]], {{.+}} [[SFVAR_ADDR]],

      // LAMBDA-64-DAG: [[G_TGT:%.+]] = bitcast {{.+}} [[G_ADDR]] to
      // LAMBDA-32-DAG: [[G_TGT:%.+]] = load {{.+}}, {{.+}} [[G_ADDR]],
      // LAMBDA-DAG: [[G1_TGT:%.+]] = load {{.+}}, {{.+}} [[G1_REF]],
      // LAMBDA-64-DAG: [[SVAR_TGT:%.+]] = bitcast {{.+}} [[SVAR_ADDR]] to
      // LAMBDA-DAG: [[SFVAR_TGT:%.+]] = bitcast {{.+}} [[SFVAR_ADDR]] to


      // LAMBDA: call {{.*}}void @__kmpc_for_static_init_4(
      // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](
      // LAMBDA: call {{.*}}void @__kmpc_for_static_fini(

      // LAMBDA: store i32 2, i32* %
      // LAMBDA: [[OMP_IS_LAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IS_LAST]],
      // LAMBDA: [[IS_LAST_IT:%.+]] = icmp ne i{{[0-9]+}} [[OMP_IS_LAST_VAL]], 0
      // LAMBDA: br i1 [[IS_LAST_IT]], label %[[OMP_LASTPRIV_BLOCK:.+]], label %[[OMP_LASTPRIV_DONE:.+]]

      // LAMBDA: [[OMP_LASTPRIV_BLOCK]]:
      // LAMBDA-DAG: store {{.+}}, {{.+}} [[G_TGT]],
      // LAMBDA-DAG: store {{.+}}, {{.+}} [[G1_TGT]],
      // LAMBDA-64-DAG: store {{.+}}, {{.+}} [[SVAR_TGT]],
      // LAMBDA-32-DAG: store {{.+}}, {{.+}} [[SVAR_ADDR]],
      // LAMBDA-DAG: store {{.+}}, {{.+}} [[SFVAR_TGT]],
      // LAMBDA: br label %[[OMP_LASTPRIV_DONE]]
      // LAMBDA: [[OMP_LASTPRIV_DONE]]:
      // LAMBDA: ret

      [&]() {
        // LAMBDA: define {{.+}} void [[INNER_LAMBDA]]({{.+}} [[ARG_PTR:%.+]])
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

  #pragma omp target teams distribute parallel for simd lastprivate(t_var, vec, s_arr, s_arr, var, var, svar)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  int i;

  return tmain<int>();
  #endif
}

// CHECK: define{{.*}} i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* {{[^,]*}} [[TEST]])
// CHECK: call i{{[0-9]+}} @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}},
// CHECK: call void [[OFFLOAD_FUN:@.+]](
// CHECK: ret

// CHECK: define{{.+}} [[OFFLOAD_FUN]](
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(
// CHECK: ret
//
// CHECK: define internal void [[OMP_OUTLINED:@.+]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [2 x i{{[0-9]+}}]*{{.+}} [[VEC_IN:%.+]], i{{[0-9]+}}{{.+}} [[T_VAR_IN:%.+]], [2 x [[S_FLOAT_TY]]]*{{.+}} [[S_ARR_IN:%.+]], [[S_FLOAT_TY]]*{{.+}} [[VAR_IN:%.+]], i{{[0-9]+}}{{.*}} [[S_VAR_IN:%.+]])
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[TMP_VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// skip loop variables
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST:%.+]] = alloca i{{[0-9]+}},

// copy from parameters to local address variables
// CHECK: store {{.+}} [[VEC_IN]], {{.+}} [[VEC_ADDR]],
// CHECK: store {{.+}} [[T_VAR_IN]], {{.+}} [[T_VAR_ADDR]],
// CHECK: store {{.+}} [[S_ARR_IN]], {{.+}} [[S_ARR_ADDR]],
// CHECK: store {{.+}} [[VAR_IN]], {{.+}} [[VAR_ADDR]],
// CHECK: store {{.+}} [[S_VAR_IN]], {{.+}} [[SVAR_ADDR]],

// prepare lastprivate targets
// CHECK-64-DAG: [[TVAR_TGT:%.+]] = bitcast {{.+}} [[T_VAR_ADDR]] to
// CHECK-DAG: [[VEC_TGT:%.+]] = load {{.+}}, {{.+}} [[VEC_ADDR]],
// CHECK-DAG: [[S_ARR_TGT:%.+]] = load {{.+}}, {{.+}} [[S_ARR_ADDR]],
// CHECK-DAG: [[VAR_TGT:%.+]] = load {{.+}}, {{.+}} [[TMP_VAR_ADDR]],
// CHECK-64-DAG: [[SVAR_TGT:%.+]] = bitcast {{.+}} [[SVAR_ADDR]] to

// the distribute loop
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[PAR_OUTL:.+]] to
// CHECK: call void @__kmpc_for_static_fini(

// lastprivates
// CHECK: [[OMP_IS_LAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IS_LAST]],
// CHECK: [[IS_LAST_IT:%.+]] = icmp ne i{{[0-9]+}} [[OMP_IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_IT]], label %[[OMP_LASTPRIV_BLOCK:.+]], label %[[OMP_LASTPRIV_DONE:.+]]

// CHECK: [[OMP_LASTPRIV_BLOCK]]:
// CHECK-64-DAG: store {{.+}}, {{.+}} [[TVAR_TGT]],
// CHECK-32-DAG: store {{.+}}, {{.+}} [[T_VAR_ADDR]],
// CHECK-DAG: [[VEC_TGT_REF:%.+]] = bitcast {{.+}} [[VEC_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VEC_TGT_REF]],
// CHECK-DAG: [[S_ARR_BEGIN:%.+]] = getelementptr {{.+}} [[S_ARR_TGT]],
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(
// CHECK-DAG: [[VAR_TGT_BCAST:%.+]] = bitcast {{.+}} [[VAR_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VAR_TGT_BCAST]],
// CHECK-64-DAG: store {{.+}}, {{.+}} [[SVAR_TGT]],
// CHECK-32-DAG: store {{.+}}, {{.+}} [[SVAR_ADDR]],
// CHECK: ret void

// CHECK: define internal void [[OMP_OUTLINED:@.+]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, {{.+}}, {{.+}}, [2 x i{{[0-9]+}}]*{{.+}} [[VEC_IN:%.+]], i{{[0-9]+}}{{.+}} [[T_VAR_IN:%.+]], [2 x [[S_FLOAT_TY]]]*{{.+}} [[S_ARR_IN:%.+]], [[S_FLOAT_TY]]*{{.+}} [[VAR_IN:%.+]], i{{[0-9]+}}{{.*}} [[S_VAR_IN:%.+]])

// gbl and bound tid vars, prev lb and ub vars
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},

// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SVAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[TMP_VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// skip loop variables
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST:%.+]] = alloca i{{[0-9]+}},

// copy from parameters to local address variables
// CHECK: store {{.+}} [[VEC_IN]], {{.+}} [[VEC_ADDR]],
// CHECK: store {{.+}} [[T_VAR_IN]], {{.+}} [[T_VAR_ADDR]],
// CHECK: store {{.+}} [[S_ARR_IN]], {{.+}} [[S_ARR_ADDR]],
// CHECK: store {{.+}} [[VAR_IN]], {{.+}} [[VAR_ADDR]],
// CHECK: store {{.+}} [[S_VAR_IN]], {{.+}} [[SVAR_ADDR]],

// prepare lastprivate targets
// CHECK-64-DAG: [[TVAR_TGT:%.+]] = bitcast {{.+}} [[T_VAR_ADDR]] to
// CHECK-DAG: [[VEC_TGT:%.+]] = load {{.+}}, {{.+}} [[VEC_ADDR]],
// CHECK-DAG: [[S_ARR_TGT:%.+]] = load {{.+}}, {{.+}} [[S_ARR_ADDR]],
// CHECK-DAG: [[VAR_TGT:%.+]] = load {{.+}}, {{.+}} [[TMP_VAR_ADDR]],
// CHECK-64-DAG: [[SVAR_TGT:%.+]] = bitcast {{.+}} [[SVAR_ADDR]] to

// the distribute loop
// CHECK: call void @__kmpc_for_static_init_4(
// skip body: code generation routine is same as distribute parallel for lastprivate
// CHECK: call void @__kmpc_for_static_fini(

// lastprivates
// CHECK: [[OMP_IS_LAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IS_LAST]],
// CHECK: [[IS_LAST_IT:%.+]] = icmp ne i{{[0-9]+}} [[OMP_IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_IT]], label %[[OMP_LASTPRIV_BLOCK:.+]], label %[[OMP_LASTPRIV_DONE:.+]]

// CHECK: [[OMP_LASTPRIV_BLOCK]]:
// CHECK-64-DAG: store {{.+}}, {{.+}} [[TVAR_TGT]],
// CHECK-32-DAG: store {{.+}}, {{.+}} [[T_VAR_ADDR]],
// CHECK-DAG: [[VEC_TGT_REF:%.+]] = bitcast {{.+}} [[VEC_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VEC_TGT_REF]],
// CHECK-DAG: [[S_ARR_BEGIN:%.+]] = getelementptr {{.+}} [[S_ARR_TGT]],
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(
// CHECK-DAG: [[VAR_TGT_BCAST:%.+]] = bitcast {{.+}} [[VAR_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VAR_TGT_BCAST]],
// CHECK-64-DAG: store {{.+}}, {{.+}} [[SVAR_TGT]],
// CHECK-32-DAG: store {{.+}}, {{.+}} [[SVAR_ADDR]],
// CHECK: ret void

// template tmain
// CHECK: define{{.*}} i{{[0-9]+}} [[TMAIN_INT:@.+]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* {{[^,]*}} [[TEST]])
// CHECK: call i{{[0-9]+}} @__tgt_target_teams_mapper(%struct.ident_t* @{{.+}},
// CHECK: call void [[OFFLOAD_FUN_1:@.+]](
// CHECK: ret

// CHECK: define internal void [[OFFLOAD_FUN_1]](
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4,
// CHECK: ret

// CHECK: define internal void [[OMP_OUTLINED_1:@.+]](i{{[0-9]+}}* noalias [[GTID_ADDR1:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [2 x i{{[0-9]+}}]*{{.+}} [[VEC_IN1:%.+]], i{{[0-9]+}}{{.+}} [[T_VAR_IN1:%.+]], [2 x [[S_INT_TY]]]*{{.+}} [[S_ARR_IN1:%.+]], [[S_INT_TY]]*{{.+}} [[VAR_IN1:%.+]])
// skip alloca of global_tid and bound_tid
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: [[VEC_ADDR1:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR1:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR1:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[VAR_ADDR1:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: [[TMP_VAR_ADDR1:%.+]] = alloca [[S_INT_TY]]*,
// skip loop variables
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST1:%.+]] = alloca i{{[0-9]+}},

// copy from parameters to local address variables
// CHECK: store {{.+}} [[VEC_IN1]], {{.+}} [[VEC_ADDR1]],
// CHECK: store {{.+}} [[T_VAR_IN1]], {{.+}} [[T_VAR_ADDR1]],
// CHECK: store {{.+}} [[S_ARR_IN1]], {{.+}} [[S_ARR_ADDR1]],
// CHECK: store {{.+}} [[VAR_IN1]], {{.+}} [[VAR_ADDR1]],

// prepare lastprivate targets
// CHECK-64-DAG: [[T_VAR_TGT:%.+]] = bitcast {{.+}} [[T_VAR_ADDR1]] to
// CHECK-DAG: [[VEC_TGT:%.+]] = load {{.+}}, {{.+}} [[VEC_ADDR1]],
// CHECK-DAG: [[S_ARR_TGT:%.+]] = load {{.+}}, {{.+}} [[S_ARR_ADDR1]],
// CHECK-DAG: [[VAR_TGT:%.+]] = load {{.+}}, {{.+}} [[TMP_VAR_ADDR1]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void {{.*}} @__kmpc_fork_call({{.+}}, {{.+}}, {{.+}} @[[TPAR_OUTL:.+]] to
// CHECK: call void @__kmpc_for_static_fini(

// lastprivates
// CHECK: [[OMP_IS_LAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IS_LAST1]],
// CHECK: [[IS_LAST_IT:%.+]] = icmp ne i{{[0-9]+}} [[OMP_IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_IT]], label %[[OMP_LASTPRIV_BLOCK:.+]], label %[[OMP_LASTPRIV_DONE:.+]]

// CHECK: [[OMP_LASTPRIV_BLOCK]]:
// CHECK-64-DAG: store {{.+}}, {{.+}} [[T_VAR_TGT]],
// CHECK-32-DAG: store {{.+}}, {{.+}} [[T_VAR_ADDR1]],
// CHECK-DAG: [[VEC_TGT_BCAST:%.+]] = bitcast {{.+}} [[VEC_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VEC_TGT_BCAST]],
// CHECK-DAG: {{.+}} = getelementptr {{.+}} [[S_ARR_TGT]],
// CHECK: call void @llvm.memcpy.{{.+}}(
// CHECK-DAG: [[VAR_ADDR_BCAST:%.+]] = bitcast {{.+}} [[VAR_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VAR_ADDR_BCAST]],
// CHECK: ret void

// CHECK: define internal void [[TPAR_OUTL:@.+]](i{{[0-9]+}}* noalias [[GTID_ADDR1:%.+]], i{{[0-9]+}}* noalias %{{.+}}, {{.+}}, {{.+}}, [2 x i{{[0-9]+}}]*{{.+}} [[VEC_IN1:%.+]], i{{[0-9]+}}{{.+}} [[T_VAR_IN1:%.+]], [2 x [[S_INT_TY]]]*{{.+}} [[S_ARR_IN1:%.+]], [[S_INT_TY]]*{{.+}} [[VAR_IN1:%.+]])
// skip alloca of global_tid and bound_tid, and prev lb and ub vars
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: {{.+}} = alloca i{{[0-9]+}}*,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},

// CHECK: [[VEC_ADDR1:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR1:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR1:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[VAR_ADDR1:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: [[TMP_VAR_ADDR1:%.+]] = alloca [[S_INT_TY]]*,
// skip loop variables
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: {{.+}} = alloca i{{[0-9]+}},
// CHECK: [[OMP_IS_LAST1:%.+]] = alloca i{{[0-9]+}},

// copy from parameters to local address variables
// CHECK: store {{.+}} [[VEC_IN1]], {{.+}} [[VEC_ADDR1]],
// CHECK: store {{.+}} [[T_VAR_IN1]], {{.+}} [[T_VAR_ADDR1]],
// CHECK: store {{.+}} [[S_ARR_IN1]], {{.+}} [[S_ARR_ADDR1]],
// CHECK: store {{.+}} [[VAR_IN1]], {{.+}} [[VAR_ADDR1]],

// prepare lastprivate targets
// CHECK-64-DAG: [[T_VAR_TGT:%.+]] = bitcast {{.+}} [[T_VAR_ADDR1]] to
// CHECK-DAG: [[VEC_TGT:%.+]] = load {{.+}}, {{.+}} [[VEC_ADDR1]],
// CHECK-DAG: [[S_ARR_TGT:%.+]] = load {{.+}}, {{.+}} [[S_ARR_ADDR1]],
// CHECK-DAG: [[VAR_TGT:%.+]] = load {{.+}}, {{.+}} [[TMP_VAR_ADDR1]],

// CHECK: call void @__kmpc_for_static_init_4(
// skip body: code generation routine is same as distribute parallel for lastprivate
// CHECK: call void @__kmpc_for_static_fini(

// lastprivates
// CHECK: [[OMP_IS_LAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[OMP_IS_LAST1]],
// CHECK: [[IS_LAST_IT:%.+]] = icmp ne i{{[0-9]+}} [[OMP_IS_LAST_VAL]], 0
// CHECK: br i1 [[IS_LAST_IT]], label %[[OMP_LASTPRIV_BLOCK:.+]], label %[[OMP_LASTPRIV_DONE:.+]]

// CHECK: [[OMP_LASTPRIV_BLOCK]]:
// CHECK-64-DAG: store {{.+}}, {{.+}} [[T_VAR_TGT]],
// CHECK-32-DAG: store {{.+}}, {{.+}} [[T_VAR_ADDR1]],
// CHECK-DAG: [[VEC_TGT_BCAST:%.+]] = bitcast {{.+}} [[VEC_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VEC_TGT_BCAST]],
// CHECK-DAG: {{.+}} = getelementptr {{.+}} [[S_ARR_TGT]],
// CHECK: call void @llvm.memcpy.{{.+}}(
// CHECK-DAG: [[VAR_ADDR_BCAST:%.+]] = bitcast {{.+}} [[VAR_TGT]] to
// CHECK-DAG: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VAR_ADDR_BCAST]],
// CHECK: ret void

#endif
