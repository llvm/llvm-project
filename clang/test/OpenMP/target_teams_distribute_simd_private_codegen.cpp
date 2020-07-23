// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix LAMBDA --check-prefix LAMBDA-64

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

struct St {
  int a, b;
  St() : a(0), b(0) {}
  St(const St &st) : a(st.a + st.b), b(0) {}
  ~St() {}
};

volatile int g = 1212;
volatile int &g1 = g;

template <class T>
struct S {
  T f;
  S(T a) : f(a + g) {}
  S() : f(g) {}
  S(const S &s, St t = St()) : f(s.f + t.a) {}
  operator T() { return T(); }
  ~S() {}
};

// CHECK-DAG: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }

template <typename T>
T tmain() {
  S<T> test;
  T t_var = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> &var = test;
#pragma omp target teams distribute simd private(t_var, vec, s_arr, var)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  return T();
}

// CHECK-DAG: [[TEST:@.+]] = global [[S_FLOAT_TY]] zeroinitializer,
S<float> test;
// CHECK-DAG: [[T_VAR:@.+]] = global i{{[0-9]+}} 333,
int t_var = 333;
// CHECK-DAG: [[VEC:@.+]] = global [2 x i{{[0-9]+}}] [i{{[0-9]+}} 1, i{{[0-9]+}} 2],
int vec[] = {1, 2};
// CHECK-DAG: [[S_ARR:@.+]] = global [2 x [[S_FLOAT_TY]]] zeroinitializer,
S<float> s_arr[] = {1, 2};
// CHECK-DAG: [[VAR:@.+]] = global [[S_FLOAT_TY]] zeroinitializer,
S<float> var(3);
// CHECK-DAG: [[SIVAR:@.+]] = internal global i{{[0-9]+}} 0,

int main() {
  static int sivar;
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]](
  [&]() {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
    // LAMBDA: call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i64* null, i8** null, i32 0, i32 1)
    // LAMBDA: call void @[[LOFFL1:.+]]()
    // LAMBDA:  ret
#pragma omp target teams distribute simd private(g, g1, sivar)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define{{.*}} internal{{.*}} void @[[LOFFL1]]()
    // LAMBDA: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}} @[[LOUTL1:.+]] to {{.+}})
    // LAMBDA: ret void

    // LAMBDA: define internal void @[[LOUTL1]]({{.+}})
    // Skip global, bound tid and loop vars
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: alloca i32,
    // LAMBDA: [[G_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[G1_PRIV:%.+]] = alloca i{{[0-9]+}}
    // LAMBDA: [[TMP:%.+]] = alloca i{{[0-9]+}}*,
    // LAMBDA: [[SIVAR_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: store{{.+}} [[G1_PRIV]], {{.+}} [[TMP]],
    g = 1;
    g1 = 1;
    sivar = 2;
    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA-DAG: store{{.+}} 1, {{.+}} [[G_PRIV]],
    // LAMBDA-DAG: store{{.+}} 2, {{.+}} [[SIVAR_PRIV]],
    // LAMBDA-DAG: [[G1_REF:%.+]] = load{{.+}}, {{.+}} [[TMP]],
    // LAMBDA-DAG: store{{.+}} 1, {{.+}} [[G1_REF]],
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](
    // LAMBDA: call void @__kmpc_for_static_fini(
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      g1 = 2;
      sivar = 4;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]

      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_REF]]
      // LAMBDA: [[G1_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[G1_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G1_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[G1_REF]]
      // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
      // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 4, i{{[0-9]+}}* [[SIVAR_REF]]
    }();
  }
  }();
  return 0;
#else
#pragma omp target teams distribute simd private(t_var, vec, s_arr, var, sivar)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
    sivar += i;
  }
  return tmain<int>();
#endif
}

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i64* null, i8** null, i32 0, i32 1)
// CHECK: call void @[[OFFL1:.+]]()
// CHECK: {{%.+}} = call{{.*}} i32 @[[TMAIN_INT:.+]]()
// CHECK:  ret

// CHECK: define{{.*}} void @[[OFFL1]]()
// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}} @[[OUTL1:.+]] to {{.+}})
// CHECK: ret void

// CHECK: define internal void @[[OUTL1]]({{.+}})
// Skip global, bound tid and loop vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK: {{.+}} = alloca i32,
// CHECK-DAG: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK-DAG: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK-DAG: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK-DAG: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK-DAG: [[SIVAR_PRIV:%.+]] = alloca i{{[0-9]+}},

// private(s_arr)
// CHECK-DAG: [[S_ARR_PRIV_BGN:%.+]] = getelementptr{{.*}} [2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_PTR_ALLOC:%.+]] = phi{{.+}} [ [[S_ARR_PRIV_BGN]], {{.+}} ], [ [[S_ARR_NEXT:%.+]], {{.+}} ]
// CHECK-DAG: call void @{{.+}}({{.+}} [[S_ARR_PTR_ALLOC]])
// CHECK-DAG: [[S_ARR_NEXT]] = getelementptr {{.+}} [[S_ARR_PTR_ALLOC]],

// private(var)
// CHECK-DAG: call void @{{.+}}({{.+}} [[VAR_PRIV]])

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-DAG: {{.+}} = {{.+}} [[T_VAR_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[VEC_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[S_ARR_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[VAR_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[SIVAR_PRIV]]
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void


// CHECK: define{{.*}} i{{[0-9]+}} @[[TMAIN_INT]]()
// CHECK: call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 0,
// CHECK: call void @[[TOFFL1:.+]]()
// CHECK:  ret

// CHECK: define {{.*}}void @[[TOFFL1]]()
// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}} @[[TOUTL1:.+]] to {{.+}})
// CHECK: ret void

// CHECK: define internal void @[[TOUTL1]]({{.+}})
// Skip global, bound tid and loop vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[TMP:%.+]] = alloca [[S_INT_TY]]*,

// private(s_arr)
// CHECK-DAG: [[S_ARR_PRIV_BGN:%.+]] = getelementptr{{.*}} [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_PTR_ALLOC:%.+]] = phi{{.+}} [ [[S_ARR_PRIV_BGN]], {{.+}} ], [ [[S_ARR_NEXT:%.+]], {{.+}} ]
// CHECK-DAG: call void @{{.+}}({{.+}} [[S_ARR_PTR_ALLOC]])
// CHECK-DAG: [[S_ARR_NEXT]] = getelementptr {{.+}} [[S_ARR_PTR_ALLOC]],

// CHECK-DAG: [[S_ARR_PRIV_BGN:%.+]] = getelementptr{{.*}} [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_PTR_ALLOC:%.+]] = phi{{.+}} [ [[S_ARR_PRIV_BGN]], {{.+}} ], [ [[S_ARR_NEXT:%.+]], {{.+}} ]
// CHECK-DAG: call void @{{.+}}({{.+}} [[S_ARR_PTR_ALLOC]])
// CHECK-DAG: [[S_ARR_NEXT]] = getelementptr {{.+}} [[S_ARR_PTR_ALLOC]],

// private(var)
// CHECK-DAG: call void @{{.+}}({{.+}} [[VAR_PRIV]])
// CHECK-DAG: store{{.+}} [[VAR_PRIV]], {{.+}} [[TMP]]

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-DAG: {{.+}} = {{.+}} [[T_VAR_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[VEC_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[S_ARR_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[TMP]]
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

#endif
