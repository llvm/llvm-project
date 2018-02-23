// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
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

volatile double g;

// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
template <typename T>
T tmain() {
  S<T> test;
  T t_var = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> var(3);
#pragma omp parallel
#pragma omp single private(t_var, vec, s_arr, s_arr, var, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
  static int sivar;
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = {{(dso_local )?}}global double
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel
#pragma omp single private(g, sivar)
  {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca double,
    // LAMBDA: [[SIVAR_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    g = 1;
    sivar = 101;
    // LAMBDA: call {{.*}}i32 @__kmpc_single(
    // LAMBDA: store double 1.0{{.+}}, double* [[G_PRIVATE_ADDR]],
    // LAMBDA: store i{{[0-9]+}} 101, i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]],
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store double* [[G_PRIVATE_ADDR]], double** [[G_PRIVATE_ADDR_REF]]
    // LAMBDA: [[SIVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
    // LAMBDA: store i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]], i{{[0-9]+}}** [[SIVAR_PRIVATE_ADDR_REF]]
    // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
    // LAMBDA: call {{.*}}void @__kmpc_end_single(
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      sivar = 211;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load double*, double** [[G_PTR_REF]]
      // LAMBDA: store double 2.0{{.+}}, double* [[G_REF]]
      // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 211, i{{[0-9]+}}* [[SIVAR_REF]]
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = {{(dso_local )?}}global double
  // BLOCKS-LABEL: @main
  // BLOCKS: call {{.*}}void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call {{.*}}void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel
#pragma omp single private(g, sivar)
  {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca double,
    // BLOCKS: [[SIVAR_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    g = 1;
    sivar = 101;
    // BLOCKS: call {{.*}}i32 @__kmpc_single(
    // BLOCKS: store double 1.0{{.+}}, double* [[G_PRIVATE_ADDR]],
    // BLOCKS: store i{{[0-9]+}} 101, i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]],
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: double* [[G_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: call {{.*}}void {{%.+}}(i8
    // BLOCKS: call {{.*}}void @__kmpc_end_single(
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      sivar = 203;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store double 2.0{{.+}}, double*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 203, i{{[0-9]+}}*
      // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
#else
  S<float> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> var(3);
#pragma omp parallel
#pragma omp single private(t_var, vec, s_arr, s_arr, var, var, sivar)
  {
    vec[0] = t_var;
    s_arr[0] = var;
    sivar = 303;
  }
  return tmain<int>();
#endif
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: = call i{{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call void [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK-NOT: alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK-NOT: alloca [[S_FLOAT_TY]],
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK: call i32 @__kmpc_single(
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK-NOT: [[SIVAR_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_FLOAT_TY]]*
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK-NOT: [[SIVAR_PRIV]]
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
// CHECK: call void @__kmpc_end_single(
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK-NOT: alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK-NOT: alloca [[S_INT_TY]],
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK: call i32 @__kmpc_single(
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_INT_TY]]*
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: call void @__kmpc_end_single(
// CHECK: ret void
#endif

