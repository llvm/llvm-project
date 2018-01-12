// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -DCHECK -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DCHECK -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++  -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
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
// CHECK-DAG: [[ST_TY:%.+]] = type { i{{[0-9]+}}, i{{[0-9]+}} }

template <typename T>
T tmain() {
  S<T> test;
  T t_var = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> &var = test;
#pragma omp target teams distribute simd firstprivate(t_var, vec, s_arr, var)
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
    // LAMBDA: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 3, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 0)
    // LAMBDA: call void @[[LOFFL1:.+]](i{{64|32}} %{{.+}})
    // LAMBDA:  ret
#pragma omp target teams distribute simd firstprivate(g, g1, sivar)
  for (int i = 0; i < 2; ++i) {
    // LAMBDA: define{{.*}} internal{{.*}} void @[[LOFFL1]](i{{64|32}} {{%.+}}, i{{64|32}} {{%.+}})
    // LAMBDA: {{%.+}} = alloca i{{[0-9]+}},
    // LAMBDA: {{%.+}} = alloca i{{[0-9]+}},
    // LAMBDA: {{%.+}} = alloca i{{[0-9]+}},
    // LAMBDA: [[G_CAST:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[G1_CAST:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[SIVAR_CAST:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA-DAG: [[G_CAST_VAL:%.+]] = load{{.+}} [[G_CAST]],
    // LAMBDA-DAG: [[G1_CAST_VAL:%.+]] = load{{.+}} [[G1_CAST]],
    // LAMBDA-DAG: [[SIVAR_CAST_VAL:%.+]] = load{{.+}} [[SIVAR_CAST]],
    // LAMBDA: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 3, {{.+}} @[[LOUTL1:.+]] to {{.+}}, {{.+}} [[G_CAST_VAL]], {{.+}} [[G1_CAST_VAL]], {{.+}} [[SIVAR_CAST_VAL]])
    // LAMBDA: ret void

    // LAMBDA: define internal void @[[LOUTL1]]({{.+}})
    // Skip global and bound tid vars
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: {{.+}} = alloca i32*,
    // LAMBDA: [[G_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[G1_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[SIVAR_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[G1_TMP:%.+]] = alloca i32*,
    // skip loop vars
    // LAMBDA-DAG: store {{.+}}, {{.+}} [[G_ADDR]],
    // LAMBDA-DAG: store {{.+}}, {{.+}} [[G1_ADDR]],
    // LAMBDA-DAG: store {{.+}}, {{.+}} [[SIVAR_ADDR]],
    // LAMBDA-DAG: [[G_CONV:%.+]] = bitcast {{.+}} [[G_ADDR]] to
    // LAMBDA-DAG: [[G1_CONV:%.+]] = bitcast {{.+}} [[G1_ADDR]] to
    // LAMBDA-DAG: [[SIVAR_CONV:%.+]] = bitcast {{.+}} [[SIVAR_ADDR]] to
    // LAMBDA-DAG: store{{.+}} [[G1_CONV]], {{.+}} [[G1_TMP]],
    g = 1;
    g1 = 1;
    sivar = 2;
    // LAMBDA: call void @__kmpc_for_static_init_4(
    // LAMBDA-DAG: store{{.+}} 1, {{.+}} [[G_CONV]],
    // LAMBDA-DAG: [[G1:%.+]] = load{{.+}}, {{.+}}* [[G1_TMP]]
    // LAMBDA-DAG: store{{.+}} 1, {{.+}} [[G1]],
    // LAMBDA-DAG: store{{.+}} 2, {{.+}} [[SIVAR_CONV]],
    // LAMBDA-DAG: [[G1_REF:%.+]] = load{{.+}}, {{.+}} [[G1_TMP]],
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
#pragma omp target teams distribute simd firstprivate(t_var, vec, s_arr, var, sivar)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
    sivar += i;
  }
  return tmain<int>();
#endif
}

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 5, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 0)
// CHECK: call void @[[OFFL1:.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK: {{%.+}} = call{{.*}} i32 @[[TMAIN_INT:.+]]()
// CHECK:  ret

// CHECK: define{{.*}} void @[[OFFL1]]({{.+}})
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[T_VAR_CAST:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[SIVAR_CAST:%.+]] = alloca i{{[0-9]+}},

// CHECK-DAG: [[VEC_TE_PAR:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** [[VEC_PRIV]],
// CHECK-DAG: [[T_VAR_TE_PAR:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_CAST]],
// CHECK-DAG: [[S_ARR_TE_PAR:%.+]] = load [2 x [[S_FLOAT_TY]]]*, [2 x [[S_FLOAT_TY]]]** [[S_ARR_PRIV]],
// CHECK-DAG: [[VAR_TE_PAR:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[VAR_PRIV]],
// CHECK-DAG: [[SIVAR_TE_PAR:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIVAR_CAST]],

// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 5, {{.+}} @[[OUTL1:.+]] to {{.+}}, [2 x i{{[0-9]+}}]* [[VEC_TE_PAR]], i{{[0-9]+}} [[T_VAR_TE_PAR]], [2 x [[S_FLOAT_TY]]]* [[S_ARR_TE_PAR]], [[S_FLOAT_TY]]* [[VAR_TE_PAR]], i{{[0-9]+}} [[SIVAR_TE_PAR]])
// CHECK: ret void

// CHECK: define internal void @[[OUTL1]]({{.+}})
// Skip global and bound tid vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_FLOAT_TY]]*,
// CHECK: [[SIVAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// Skip temp vars for loop
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[AGG_TMP1:%.+]] = alloca [[ST_TY]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[AGG_TMP2:%.+]] = alloca [[ST_TY]],

// param copy
// CHECK: store [2 x i{{[0-9]+}}]* {{.+}}, [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK: store i{{[0-9]+}} {{.+}}, i{{[0-9]+}}* [[T_VAR_ADDR]],
// CHECK: store [2 x [[S_FLOAT_TY]]]* {{.+}}, [2 x [[S_FLOAT_TY]]]** [[S_ARR_ADDR]],
// CHECK: store [[S_FLOAT_TY]]* {{.+}}, [[S_FLOAT_TY]]** [[VAR_ADDR]],
// CHECK: store i{{[0-9]+}} {{.+}}, i{{[0-9]+}}* [[SIVAR_ADDR]],

// T_VAR and SIVAR
// CHECK-DAG-64: [[CONV_TVAR:%.+]] = bitcast i64* [[T_VAR_ADDR]] to i32*
// CHECK-DAG-64: [[CONV_SIVAR:%.+]] = bitcast i64* [[SIVAR_ADDR]] to i32*

// preparation vars
// CHECK-DAG: [[VEC_ADDR_VAL:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK-DAG: [[S_ARR_ADDR_REF:%.+]] = load [2 x [[S_FLOAT_TY]]]*, [2 x [[S_FLOAT_TY]]]** [[S_ARR_ADDR]],
// CHECK-DAG: [[VAR_ADDR_REF:%.+]] = load{{.+}} [[VAR_ADDR]],

// firstprivate vec(vec): copy from *_addr into priv1 and then from priv1 into priv2
// CHECK-DAG: [[VEC_DEST_PRIV:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8* 
// CHECK-DAG: [[VEC_SRC:%.+]] =  bitcast [2 x i{{[0-9]+}}]* [[VEC_ADDR_VAL]] to i8* 
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[VEC_DEST_PRIV]], i8* [[VEC_SRC]], {{.+}})

// firstprivate(s_arr)
// CHECK-DAG: [[S_ARR_PRIV_BGN:%.+]] = getelementptr{{.*}} [2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_ADDR_BGN:%.+]] = bitcast [2 x [[S_FLOAT_TY]]]* [[S_ARR_ADDR_REF]] to
// CHECK-DAG: [[S_ARR_FIN:%.+]] = icmp{{.+}} [[S_ARR_PRIV_BGN]],
// CHECK-DAG: [[S_ARR_SRC_COPY:%.+]] = phi{{.+}} [ [[S_ARR_ADDR_BGN]], {{.+}} ], [ [[S_ARR_SRC:%.+]], {{.+}} ]
// CHECK-DAG: [[S_ARR_DST_COPY:%.+]] = phi{{.+}} [ [[S_ARR_PRIV_BGN]], {{.+}}], [ [[S_ARR_DST:%.+]], {{.+}} ]
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP1]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[S_ARR_DST_COPY]], {{.+}} [[S_ARR_SRC_COPY]], {{.+}} [[AGG_TMP1]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP1]])
// CHECK-DAG: [[S_ARR_DST]] = getelementptr {{.+}} [[S_ARR_DST_COPY]],
// CHECK-DAG: [[S_ARR_SRC]] = getelementptr {{.+}} [[S_ARR_SRC_COPY]],

// firstprivate(var)
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP2]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[VAR_PRIV]], {{.+}} [[VAR_ADDR_REF]], {{.+}} [[AGG_TMP2]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP2]])

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-DAG-32: {{.+}} = {{.+}} [[T_VAR_ADDR]]
// CHECK-DAG-64: {{.+}} = {{.+}} [[CONV_TVAR]]
// CHECK-DAG: {{.+}} = {{.+}} [[VEC_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[S_ARR_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[VAR_PRIV]]
// CHECK-DAG-32: {{.+}} = {{.+}} [[SIVAR_ADDR]]
// CHECK-DAG-64: {{.+}} = {{.+}} [[CONV_SIVAR]]
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define{{.*}} i{{[0-9]+}} @[[TMAIN_INT]]()
// CHECK: call i32 @__tgt_target_teams(i64 -1, i8* @{{[^,]+}}, i32 4, i8** %{{[^,]+}}, i8** %{{[^,]+}}, i{{64|32}}* {{.+}}@{{[^,]+}}, i32 0, i32 0), i64* {{.+}}@{{[^,]+}}, i32 0, i32 0), i32 0, i32 0)
// CHECK: call void @[[TOFFL1:.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK:  ret

// CHECK: define {{.*}}void @[[TOFFL1]]({{.+}})
// CHECK: [[TVEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[TT_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[TS_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[TVAR_PRIV:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: [[TT_VAR_CAST:%.+]] = alloca i{{[0-9]+}},

// CHECK-DAG: [[TVEC_TE_PAR:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** [[TVEC_PRIV]],
// CHECK-DAG: [[TT_VAR_TE_PAR:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[TT_VAR_CAST]],
// CHECK-DAG: [[TS_ARR_TE_PAR:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** [[TS_ARR_PRIV]],
// CHECK-DAG: [[TVAR_TE_PAR:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** [[TVAR_PRIV]],

// CHECK: call void {{.+}} @__kmpc_fork_teams({{.+}}, i32 4, {{.+}} @[[TOUTL1:.+]] to {{.+}}, [2 x i{{[0-9]+}}]* [[TVEC_TE_PAR]], i{{[0-9]+}} [[TT_VAR_TE_PAR]], [2 x [[S_INT_TY]]]* [[TS_ARR_TE_PAR]], [[S_INT_TY]]* [[TVAR_TE_PAR]])
// CHECK: ret void

// CHECK: define internal void @[[TOUTL1]]({{.+}})
// Skip global and bound tid vars
// CHECK: {{.+}} = alloca i32*,
// CHECK: {{.+}} = alloca i32*,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i{{[0-9]+}}]*,
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]]*,
// Skip temp vars for loop
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[AGG_TMP1:%.+]] = alloca [[ST_TY]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[AGG_TMP2:%.+]] = alloca [[ST_TY]],
// CHECK: [[TMP:%.+]] = alloca [[S_INT_TY]]*,

// param copy
// CHECK: store [2 x i{{[0-9]+}}]* {{.+}}, [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK: store i{{[0-9]+}} {{.+}}, i{{[0-9]+}}* [[T_VAR_ADDR]],
// CHECK: store [2 x [[S_INT_TY]]]* {{.+}}, [2 x [[S_INT_TY]]]** [[S_ARR_ADDR]],
// CHECK: store [[S_INT_TY]]* {{.+}}, [[S_INT_TY]]** [[VAR_ADDR]],


// T_VAR and preparation variables
// CHECK: [[VEC_ADDR_VAL:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** [[VEC_ADDR]],
// CHECK-64: [[CONV_TVAR:%.+]] = bitcast i64* [[T_VAR_ADDR]] to i32*
// CHECK: [[S_ARR_ADDR_REF:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** [[S_ARR_ADDR]],

// firstprivate vec(vec): copy from *_addr into priv1 and then from priv1 into priv2
// CHECK-DAG: [[VEC_DEST_PRIV:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8* 
// CHECK-DAG: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_ADDR_VAL]] to i8* 
// CHECK: call void @llvm.memcpy.{{.+}}(i8* [[VEC_DEST_PRIV]], i8* [[VEC_SRC]], {{.+}})

// firstprivate(s_arr)
// CHECK-DAG: [[S_ARR_PRIV_BGN:%.+]] = getelementptr{{.*}} [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]],
// CHECK-DAG: [[S_ARR_ADDR_BGN:%.+]] = bitcast [2 x [[S_INT_TY]]]* [[S_ARR_ADDR_REF]] to
// CHECK-DAG: [[S_ARR_FIN:%.+]] = icmp{{.+}} [[S_ARR_PRIV_BGN]],
// CHECK-DAG: [[S_ARR_SRC_COPY:%.+]] = phi{{.+}} [ [[S_ARR_ADDR_BGN]], {{.+}} ], [ [[S_ARR_SRC:%.+]], {{.+}} ]
// CHECK-DAG: [[S_ARR_DST_COPY:%.+]] = phi{{.+}} [ [[S_ARR_PRIV_BGN]], {{.+}} ], [ [[S_ARR_DST:%.+]], {{.+}} ]
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP1]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[S_ARR_DST_COPY]], {{.+}} [[S_ARR_SRC_COPY]], {{.+}} [[AGG_TMP1]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP1]])
// CHECK-DAG: [[S_ARR_DST]] = getelementptr {{.+}} [[S_ARR_DST_COPY]],
// CHECK-DAG: [[S_ARR_SRC]] = getelementptr {{.+}} [[S_ARR_SRC_COPY]],

// firstprivate(var)
// CHECK-DAG: [[VAR_ADDR_REF:%.+]] = load{{.+}} [[VAR_ADDR]],
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP2]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[VAR_PRIV]], {{.+}} [[VAR_ADDR_REF]], {{.+}} [[AGG_TMP2]])
// CHECK-DAG: call void @{{.+}}({{.+}} [[AGG_TMP2]])
// CHECK-DAG: store [[S_INT_TY]]* [[VAR_PRIV]], [[S_INT_TY]]** [[TMP]],

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-DAG-32: {{.+}} = {{.+}} [[T_VAR_ADDR]]
// CHECK-DAG-64: {{.+}} = {{.+}} [[CONV_TVAR]]
// CHECK-DAG: {{.+}} = {{.+}} [[VEC_PRIV]]
// CHECK-DAG: {{.+}} = {{.+}} [[TMP]]
// CHECK-DAG: {{.+}} = {{.+}} [[S_ARR_PRIV]]
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

#endif
