// Test host codegen.
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-64
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DLAMBDA -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix LAMBDA --check-prefix LAMBDA-32

// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DLAMBDA -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -DARRAY  -verify -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix ARRAY --check-prefix ARRAY-64
// RUN: %clang_cc1 -DARRAY  -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DARRAY  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix ARRAY --check-prefix ARRAY-64
// RUN: %clang_cc1 -DARRAY  -verify -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck %s --check-prefix ARRAY --check-prefix ARRAY-32
// RUN: %clang_cc1 -DARRAY  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DARRAY  -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck %s --check-prefix ARRAY --check-prefix ARRAY-32

// RUN: %clang_cc1 -DARRAY  -verify -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DARRAY  -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DARRAY  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DARRAY  -verify -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DARRAY  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s -Wno-openmp-mapping
// RUN: %clang_cc1 -DARRAY  -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -include-pch %t -verify %s -emit-llvm -o - -Wno-openmp-mapping | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER
#ifndef ARRAY
struct St {
  int a, b;
  St() : a(0), b(0) {}
  St(const St &st) : a(st.a + st.b), b(0) {}
  ~St() {}
};

volatile int g __attribute__((aligned(128))) = 1212;

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
  T t_var __attribute__((aligned(128))) = T();
  T vec[] __attribute__((aligned(128))) = {1, 2};
  S<T> s_arr[] __attribute__((aligned(128))) = {1, 2};
  S<T> var __attribute__((aligned(128))) (3);
  #pragma omp target
  #pragma omp teams firstprivate(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
#pragma omp target
#pragma omp teams firstprivate(t_var)
  {}
  return T();
}

int main() {
  static int sivar;
#ifdef LAMBDA
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@.+]](
  [&]() {    
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 2, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* {{.+}}, {{.+}})    
  #pragma omp target
  #pragma omp teams firstprivate(g, sivar)
  {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* nonnull align 4 dereferenceable(4) [[G_IN:%.+]], i{{64|32}} {{.*}}[[SIVAR_IN:%.+]])
    // LAMBDA: store i{{[0-9]+}}* [[G_IN]], i{{[0-9]+}}** [[G_ADDR:%.+]],
    // LAMBDA: store i{{[0-9]+}} [[SIVAR_IN]], i{{[0-9]+}}* [[SIVAR_ADDR:%.+]],
    // LAMBDA: [[G_ADDR_VAL:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_ADDR]],
    // LAMBDA-64: [[SIVAR_CONV:%.+]] = bitcast i64*  [[SIVAR_ADDR]] to i32*
    // LAMBDA: [[G_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[G_ADDR_VAL]],
    // LAMBDA: store i{{[0-9]+}} [[G_VAL]], i{{[0-9]+}}* [[G_LOCAL:%.+]],
    g = 1;
    sivar = 2;
    // LAMBDA: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_LOCAL]],
    // LAMBDA-64: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[SIVAR_CONV]],
    // LAMBDA-32: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[SIVAR_ADDR]],
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store i{{[0-9]+}}* [[G_LOCAL]], i{{[0-9]+}}** [[G_PRIVATE_ADDR_REF]]
    // LAMBDA: [[SIVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
    // LAMBDA-64: store i{{[0-9]+}}* [[SIVAR_CONV]], i{{[0-9]+}}** [[SIVAR_PRIVATE_ADDR_REF]]
    // LAMBDA-32: store i{{[0-9]+}}* [[SIVAR_ADDR]], i{{[0-9]+}}** [[SIVAR_PRIVATE_ADDR_REF]]
    // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      sivar = 4;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_PTR_REF]]
      // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 4, i{{[0-9]+}}* [[SIVAR_REF]]
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
  #pragma omp target
  #pragma omp teams firstprivate(t_var, vec, s_arr, var, sivar)
  {
    vec[0] = t_var;
    s_arr[0] = var;
    sivar = 2;
  }
  #pragma omp target
  #pragma omp teams firstprivate(t_var)
  {}
  return tmain<int>();
#endif
}

// CHECK: define internal {{.*}}void [[OMP_OFFLOADING:@.+]](
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 5, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i32]*, i{{32|64}}, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]*, i{{[0-9]+}})* [[OMP_OUTLINED:@.+]] to void
// CHECK: ret
//
// CHECK: define internal {{.*}}void [[OMP_OUTLINED]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [2 x i32]* nonnull align 4 dereferenceable(8) %{{.+}}, i{{32|64}} {{.*}}%{{.+}}, [2 x [[S_FLOAT_TY]]]* nonnull align 4 dereferenceable(8) %{{.+}}, [[S_FLOAT_TY]]* nonnull align 4 dereferenceable(4) %{{.+}}, i{{32|64}} {{.*}}[[SIVAR:%.+]])
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[SIVAR7_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}],
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[VEC_REF:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** %
// CHECK-64: [[T_VAR_CONV:%.+]] = bitcast i64* [[T_VAR_PRIV]] to i32*
// CHECK: [[S_ARR_REF:%.+]] = load [2 x [[S_FLOAT_TY]]]*, [2 x [[S_FLOAT_TY]]]** %
// CHECK: [[VAR_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %
// CHECK-64: [[SIVAR7_CONV:%.+]] = bitcast i64* [[SIVAR7_PRIV]] to i32*
// CHECK: [[VEC_DEST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8*
// CHECK: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_REF]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[VEC_DEST]], i8* align {{[0-9]+}} [[VEC_SRC]],
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* [[S_ARR_PRIV]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_BEGIN:%.+]] = bitcast [2 x [[S_FLOAT_TY]]]* [[S_ARR_REF]] to [[S_FLOAT_TY]]*
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR:@.+]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_CONSTR:@.+]]([[S_FLOAT_TY]]* {{.+}}, [[S_FLOAT_TY]]* {{.+}}, [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR:@.+]]([[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_CONSTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]], [[S_FLOAT_TY]]* {{.*}} [[VAR_REF]], [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR]]([[ST_TY]]* [[ST_TY_TEMP]])

// CHECK-64: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[SIVAR7_CONV]],
// CHECK-32: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[SIVAR7_PRIV]],

// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
// CHECK: ret void

// CHECK: define internal {{.*}}void [[OMP_OFFLOADING_1:@.+]](
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i{{[0-9]+}})* [[OMP_OUTLINED_1:@.+]] to void
// CHECK: ret

// CHECK: define internal {{.*}}void [[OMP_OUTLINED_1]](i{{[0-9]+}}* noalias {{%.+}}, i{{[0-9]+}}* noalias {{%.+}}, i{{32|64}} {{.*}}[[T_VAR:%.+]])
// CHECK: [[T_VAR_LOC:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}} [[T_VAR]], i{{[0-9]+}}* [[T_VAR_LOC]],
// CHECK: ret

// CHECK: define internal {{.*}}void [[OMP_OFFLOADING_2:@.+]](i{{[0-9]+}}* {{.+}} {{%.+}}, [2 x i32]* {{.+}} {{%.+}}, [2 x [[S_INT_TY]]]* {{.+}} {{%.+}}, [[S_INT_TY]]* {{.+}} {{%.+}})
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i32]*, i32*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]*)* [[OMP_OUTLINED_2:@.+]] to void
// CHECK: ret

//
// CHECK: define internal {{.*}}void [[OMP_OUTLINED_2]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [2 x i32]* nonnull align 4 dereferenceable(8) %{{.+}}, i32* nonnull align 4 dereferenceable(4) %{{.+}}, [2 x [[S_INT_TY]]]* nonnull align 4 dereferenceable(8) %{{.+}}, [[S_INT_TY]]* nonnull align 4 dereferenceable(4) %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}}, align 128
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}], align 128
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]], align 128
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]], align 128
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[VEC_REF:%.+]] = load [2 x i{{[0-9]+}}]*, [2 x i{{[0-9]+}}]** %
// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %
// CHECK: [[S_ARR_REF:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** %
// CHECK: [[VAR_REF:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** %

// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_REF]], align 128
// CHECK: store i{{[0-9]+}} [[T_VAR_VAL]], i{{[0-9]+}}* [[T_VAR_PRIV]], align 128
// CHECK: [[VEC_DEST:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_PRIV]] to i8*
// CHECK: [[VEC_SRC:%.+]] = bitcast [2 x i{{[0-9]+}}]* [[VEC_REF]] to i8*
// CHECK: call void @llvm.memcpy.{{.+}}(i8* align 128 [[VEC_DEST]], i8* align 128 [[VEC_SRC]], i{{[0-9]+}} {{[0-9]+}},
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[S_ARR_PRIV]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_BEGIN:%.+]] = bitcast [2 x [[S_INT_TY]]]* [[S_ARR_REF]] to [[S_INT_TY]]*
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR:@.+]]([[S_INT_TY]]* {{.+}}, [[S_INT_TY]]* {{.+}}, [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR:@.+]]([[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]
// CHECK: call {{.*}} [[ST_TY_DEFAULT_CONSTR]]([[ST_TY]]* [[ST_TY_TEMP:%.+]])
// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]* [[VAR_PRIV]], [[S_INT_TY]]* {{.*}} [[VAR_REF]], [[ST_TY]]* [[ST_TY_TEMP]])
// CHECK: call {{.*}} [[ST_TY_DESTR]]([[ST_TY]]* [[ST_TY_TEMP]])
// CHECK-DAG: call {{.*}} [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call {{.*}} [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: ret void

// CHECK: define internal {{.*}}void [[OMP_OFFLOADING_3:@.+]](
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_teams(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i{{[0-9]+}}*)* [[OMP_OUTLINED_3:@.+]] to void
// CHECK: ret

// CHECK: define internal {{.*}}void [[OMP_OUTLINED_3]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i32* nonnull align 4 dereferenceable(4) [[T_VAR:%.+]])
// CHECK: [[T_VAR_LOC:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[T_VAR]], i{{[0-9]+}}** [[T_VAR_ADDR:%.+]],
// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[T_VAR_ADDR]],
// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_REF]],
// CHECK: store i{{[0-9]+}} [[T_VAR_VAL]], i{{[0-9]+}}* [[T_VAR_LOC]],
// CHECK: ret

#else
struct St {
  int a, b;
  St() : a(0), b(0) {}
  St(const St &) { }
  ~St() {}
  void St_func(St s[2], int n, long double vla1[n]) {
    double vla2[n][n] __attribute__((aligned(128)));
    a = b;
    #pragma omp target
    #pragma omp teams firstprivate(s, vla1, vla2)
    vla1[b] = vla2[1][n - 1] = a = b;
  }
};

void array_func(float a[3], St s[2], int n, long double vla1[n]) {
  double vla2[n][n] __attribute__((aligned(128)));
// ARRAY: call {{.+}} @__kmpc_fork_teams(
// ARRAY-DAG: [[PRIV_S:%.+]] = alloca %struct.St*,
// ARRAY-64-DAG: [[PRIV_VLA1:%.+]] = alloca ppc_fp128*,
// ARRAY-32-DAG: [[PRIV_VLA1:%.+]] = alloca x86_fp80*,
// ARRAY-DAG: [[PRIV_A:%.+]] = alloca float*,
// ARRAY-DAG: [[PRIV_VLA2:%.+]] = alloca double*,
// ARRAY-DAG: store float* %{{.+}}, float** [[PRIV_A]],
// ARRAY-DAG: store %struct.St* %{{.+}}, %struct.St** [[PRIV_S]],
// ARRAY-64-DAG: store ppc_fp128* %{{.+}}, ppc_fp128** [[PRIV_VLA1]],
// ARRAY-32-DAG: store x86_fp80* %{{.+}}, x86_fp80** [[PRIV_VLA1]],
// ARRAY-DAG: store double* %{{.+}}, double** [[PRIV_VLA2]],
// ARRAY: call i8* @llvm.stacksave()
// ARRAY: [[SIZE:%.+]] = mul nuw i{{[0-9]+}} %{{.+}}, 8
// ARRAY: call void @llvm.memcpy.p0i8.p0i8.i{{[0-9]+}}(i8* align 128 %{{.+}}, i8* align 128 %{{.+}}, i{{[0-9]+}} [[SIZE]], i1 false)
  #pragma omp target
  #pragma omp teams firstprivate(a, s, vla1, vla2)
  s[0].St_func(s, n, vla1);
  ;
}

// ARRAY: @__kmpc_fork_teams(
// ARRAY-DAG: [[PRIV_S:%.+]] = alloca %struct.St*,
// ARRAY-64-DAG: [[PRIV_VLA1:%.+]] = alloca ppc_fp128*,
// ARRAY-32-DAG: [[PRIV_VLA1:%.+]] = alloca x86_fp80*,
// ARRAY-DAG: [[PRIV_VLA2:%.+]] = alloca double*,
// ARRAY-DAG: store %struct.St* %{{.+}}, %struct.St** [[PRIV_S]],
// ARRAY-64-DAG: store ppc_fp128* %{{.+}}, ppc_fp128** [[PRIV_VLA1]],
// ARRAY-32-DAG: store x86_fp80* %{{.+}}, x86_fp80** [[PRIV_VLA1]],
// ARRAY-DAG: store double* %{{.+}}, double** [[PRIV_VLA2]],
// ARRAY: call i8* @llvm.stacksave()
// ARRAY: [[SIZE:%.+]] = mul nuw i{{[0-9]+}} %{{.+}}, 8
// ARRAY: call void @llvm.memcpy.p0i8.p0i8.i{{[0-9]+}}(i8* align 128 %{{.+}}, i8* align 128 %{{.+}}, i{{[0-9]+}} [[SIZE]], i1 false)
#endif
#endif
