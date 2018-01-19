// RUN: %clang_cc1 -verify -fopenmp -fnoopenmp-use-tls -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -fnoopenmp-use-tls -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// RUN: %clang_cc1 -verify -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=ARRAY %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fnoopenmp-use-tls -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fnoopenmp-use-tls -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s -check-prefix=TLS-CHECK
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefix=TLS-CHECK
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=TLS-LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=TLS-BLOCKS %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DARRAY -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck -check-prefix=TLS-ARRAY %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -std=c++11 -DARRAY -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef ARRAY
#ifndef HEADER
#define HEADER

volatile int g __attribute__((aligned(128))) = 1212;
#pragma omp threadprivate(g)

template <class T>
struct S {
  T f;
  S(T a) : f(a + g) {}
  S() : f(g) {}
  S &operator=(const S &) { return *this; };
  operator T() { return T(); }
  ~S() {}
};

// CHECK-DAG: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// TLS-CHECK-DAG: [[S_FLOAT_TY:%.+]] = type { float }
// TLS-CHECK-DAG: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// TLS-CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*

// CHECK-DAG: [[T_VAR:@.+]] = internal global i{{[0-9]+}} 1122,
// CHECK-DAG: [[VEC:@.+]] = internal global [2 x i{{[0-9]+}}] [i{{[0-9]+}} 1, i{{[0-9]+}} 2],
// CHECK-DAG: [[S_ARR:@.+]] = internal global [2 x [[S_FLOAT_TY]]] zeroinitializer,
// CHECK-DAG: [[VAR:@.+]] = internal global [[S_FLOAT_TY]] zeroinitializer,
// CHECK-DAG: [[TMAIN_T_VAR:@.+]] = linkonce_odr global i{{[0-9]+}} 333,
// CHECK-DAG: [[TMAIN_VEC:@.+]] = linkonce_odr global [2 x i{{[0-9]+}}] [i{{[0-9]+}} 3, i{{[0-9]+}} 3],
// CHECK-DAG: [[TMAIN_S_ARR:@.+]] = linkonce_odr global [2 x [[S_INT_TY]]] zeroinitializer,
// CHECK-DAG: [[TMAIN_VAR:@.+]] = linkonce_odr global [[S_INT_TY]] zeroinitializer,
// TLS-CHECK-DAG: [[T_VAR:@.+]] = internal thread_local global i{{[0-9]+}} 1122,
// TLS-CHECK-DAG: [[VEC:@.+]] = internal thread_local global [2 x i{{[0-9]+}}] [i{{[0-9]+}} 1, i{{[0-9]+}} 2],
// TLS-CHECK-DAG: [[S_ARR:@.+]] = internal thread_local global [2 x [[S_FLOAT_TY]]] zeroinitializer,
// TLS-CHECK-DAG: [[VAR:@.+]] = internal thread_local global [[S_FLOAT_TY]] zeroinitializer,
// TLS-CHECK-DAG: [[TMAIN_T_VAR:@.+]] = linkonce_odr thread_local global i{{[0-9]+}} 333,
// TLS-CHECK-DAG: [[TMAIN_VEC:@.+]] = linkonce_odr thread_local global [2 x i{{[0-9]+}}] [i{{[0-9]+}} 3, i{{[0-9]+}} 3],
// TLS-CHECK-DAG: [[TMAIN_S_ARR:@.+]] = linkonce_odr thread_local global [2 x [[S_INT_TY]]] zeroinitializer,
// TLS-CHECK-DAG: [[TMAIN_VAR:@.+]] = linkonce_odr thread_local global [[S_INT_TY]] zeroinitializer,

template <typename T>
T tmain() {
  S<T> test;
  test = S<T>();
  static T t_var __attribute__((aligned(128))) = 333;
  static T vec[] __attribute__((aligned(128))) = {3, 3};
  static S<T> s_arr[] __attribute__((aligned(128))) = {1, 2};
  static S<T> var __attribute__((aligned(128))) (3);
#pragma omp threadprivate(t_var, vec, s_arr, var)
#pragma omp parallel copyin(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
#pragma omp parallel copyin(t_var)
  {}
  return T();
}

int main() {
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@.+]](
  // TLS-LAMBDA: [[G:@.+]] = {{.*}}thread_local {{.*}}global i{{[0-9]+}} 1212,
  // TLS-LAMBDA-LABEL: @main
  // TLS-LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call {{.*}}void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})

  // TLS-LAMBDA:     [[G_CPY_VAL:%.+]] = call{{( cxx_fast_tlscc)?}} i{{[0-9]+}}* [[G_CTOR:@.+]]()
  // TLS-LAMBDA:     call {{.*}}void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* [[G_CPY_VAL]])

#pragma omp parallel copyin(g)
  {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // TLS-LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}})

    // threadprivate_g = g;
    // LAMBDA: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[G]]
    // LAMBDA: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
    // LAMBDA: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[G]] to i{{[0-9]+}}), %{{.+}}
    // LAMBDA: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
    // LAMBDA: [[NOT_MASTER]]
    // LAMBDA: load i{{[0-9]+}}, i{{[0-9]+}}* [[G]], align 128
    // LAMBDA: store volatile i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}}, align 128
    // LAMBDA: [[DONE]]

    // TLS-LAMBDA-DAG: [[G_CAPTURE_SRC:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %
    // TLS-LAMBDA-DAG: [[G_CAPTURE_DST:%.+]] = call{{( cxx_fast_tlscc)?}} i{{[0-9]+}}* [[G_CTOR]]()
    // TLS-LAMBDA-DAG: [[G_CAPTURE_SRCC:%.+]] = ptrtoint i{{[0-9]+}}* [[G_CAPTURE_SRC]] to i{{[0-9]+}}
    // TLS-LAMBDA-DAG: [[G_CAPTURE_DSTC:%.+]] = ptrtoint i{{[0-9]+}}* [[G_CAPTURE_DST]] to i{{[0-9]+}}
    // TLS-LAMBDA: icmp ne i{{[0-9]+}} {{%.+}}, {{%.+}}
    // TLS-LAMBDA: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
    // TLS-LAMBDA: [[NOT_MASTER]]
    // TLS-LAMBDA: load i{{[0-9]+}}, i{{[0-9]+}}* [[G_CAPTURE_SRC]],
    // TLS-LAMBDA: store volatile i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* [[G_CAPTURE_DST]], align 128
    // TLS-LAMBDA: [[DONE]]

    // LAMBDA: call {{.*}}void @__kmpc_barrier(
    // TLS-LAMBDA: call {{.*}}void @__kmpc_barrier(
    g = 1;
    // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}*
    // TLS-LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}*

    // TLS-LAMBDA:     define {{.*}}i{{[0-9]+}}* [[G_CTOR]]()
    // TLS-LAMBDA:     ret i{{[0-9]+}}* [[G]]
    // TLS-LAMBDA:     }

    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]

      // TLS-LAMBDA: [[G_CAPTURE_DST:%.+]] = call{{( cxx_fast_tlscc)?}} i{{[0-9]+}}* [[G_CTOR]]()
      // TLS-LAMBDA: store volatile i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_CAPTURE_DST]], align 128
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // BLOCKS-LABEL: @main
  // BLOCKS: call {{.*}}void {{%.+}}(i8

  // TLS-BLOCKS: [[G:@.+]] = {{.*}}thread_local {{.*}}global i{{[0-9]+}} 1212,
  // TLS-BLOCKS-LABEL: @main
  // TLS-BLOCKS: call {{.*}}void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call {{.*}}void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})

  // TLS-BLOCKS:     [[G_CPY_VAL:%.+]] = call{{( cxx_fast_tlscc)?}} i{{[0-9]+}}* [[G_CTOR:@.+]]()
  // TLS-BLOCKS:     call {{.*}}void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* [[G_CPY_VAL]])

#pragma omp parallel copyin(g)
  {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // TLS-BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}})

    // threadprivate_g = g;
    // BLOCKS: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[G]]
    // BLOCKS: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
    // BLOCKS: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[G]] to i{{[0-9]+}}), %{{.+}}
    // BLOCKS: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
    // BLOCKS: [[NOT_MASTER]]
    // BLOCKS: load i{{[0-9]+}}, i{{[0-9]+}}* [[G]], align 128
    // BLOCKS: store volatile i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}}, align 128
    // BLOCKS: [[DONE]]

    // TLS-BLOCKS-DAG: [[G_CAPTURE_SRC:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %
    // TLS-BLOCKS-DAG: [[G_CAPTURE_DST:%.+]] = call{{( cxx_fast_tlscc)?}} i{{[0-9]+}}* [[G_CTOR]]()
    // TLS-BLOCKS-DAG: [[G_CAPTURE_SRCC:%.+]] = ptrtoint i{{[0-9]+}}* [[G_CAPTURE_SRC]] to i{{[0-9]+}}
    // TLS-BLOCKS-DAG: [[G_CAPTURE_DSTC:%.+]] = ptrtoint i{{[0-9]+}}* [[G_CAPTURE_DST]] to i{{[0-9]+}}
    // TLS-BLOCKS: icmp ne i{{[0-9]+}} {{%.+}}, {{%.+}}
    // TLS-BLOCKS: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
    // TLS-BLOCKS: [[NOT_MASTER]]
    // TLS-BLOCKS: load i{{[0-9]+}}, i{{[0-9]+}}* [[G_CAPTURE_SRC]],
    // TLS-BLOCKS: store volatile i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* [[G_CAPTURE_DST]], align 128
    // TLS-BLOCKS: [[DONE]]

    // BLOCKS: call {{.*}}void @__kmpc_barrier(
    // TLS-BLOCKS: call {{.*}}void @__kmpc_barrier(
    g = 1;
    // BLOCKS: store volatile i{{[0-9]+}} 1, i{{[0-9]+}}*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call {{.*}}void {{%.+}}(i8

    // TLS-BLOCKS: [[G_CAPTURE_DST:%.+]] = call{{( cxx_fast_tlscc)?}} i{{[0-9]+}}* [[G_CTOR]]()
    // TLS-BLOCKS: store volatile i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_CAPTURE_DST]]
    // TLS-BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // TLS-BLOCKS: call {{.*}}void {{%.+}}(i8

    // TLS-BLOCKS:     define {{.*}}i{{[0-9]+}}* [[G_CTOR]]()
    // TLS-BLOCKS:     ret i{{[0-9]+}}* [[G]]
    // TLS-BLOCKS:     }
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      // TLS-BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[G]]
      // BLOCKS: store volatile i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: ret

      // TLS-BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // TLS-BLOCKS: [[G_CAPTURE_DST:%.+]] = call{{( cxx_fast_tlscc)?}} i{{[0-9]+}}* [[G_CTOR]]()
      // TLS-BLOCKS: store volatile i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_CAPTURE_DST]]
      // TLS-BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // TLS-BLOCKS: ret
    }();
  }
  }();
  return 0;
#else
  S<float> test;
  test = S<float>();
  static int t_var = 1122;
  static int vec[] = {1, 2};
  static S<float> s_arr[] = {1, 2};
  static S<float> var(3);
#pragma omp threadprivate(t_var, vec, s_arr, var)
#pragma omp parallel copyin(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
#pragma omp parallel copyin(t_var)
  {}
  return tmain<int>();
#endif
}

// CHECK-LABEL: @main
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN:@.+]]([[S_FLOAT_TY]]* [[TEST]], [[S_FLOAT_TY]]*
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[MAIN_MICROTASK:@.+]] to void (i32*, i32*, ...)*))
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[MAIN_MICROTASK1:@.+]] to void (i32*, i32*, ...)*))
// CHECK: = call {{.*}}i{{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call {{.*}} [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret

// TLS-CHECK-LABEL: @main
// TLS-CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// TLS-CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN:@.+]]([[S_FLOAT_TY]]* [[TEST]], [[S_FLOAT_TY]]*
// TLS-CHECK:     call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32*, [2 x i32]*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]*)* [[MAIN_MICROTASK:@.+]] to void (i32*, i32*, ...)*),
// TLS-CHECK:     call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32*)* [[MAIN_MICROTASK1:@.+]] to void (i32*, i32*, ...)*),
// TLS-CHECK: = call {{.*}}i{{.+}} [[TMAIN_INT:@.+]]()
// TLS-CHECK: call {{.*}} [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// TLS-CHECK: ret

// CHECK: define internal {{.*}}void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],

// TLS-CHECK: define internal {{.*}}void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}},
// TLS-CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// threadprivate_t_var = t_var;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR]],
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},

// TLS-CHECK: [[MASTER_REF:%.+]] = load i32*, i32** %
// TLS-CHECK: [[MASTER_REF2:%.+]] = load [2 x i32]*, [2 x i32]** %
// TLS-CHECK: [[MASTER_REF3:%.+]] = load [2 x [[S_FLOAT_TY]]]*, [2 x [[S_FLOAT_TY]]]** %
// TLS-CHECK: [[MASTER_REF4:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %

// TLS-CHECK: [[MASTER_LONG:%.+]] = ptrtoint i32* [[MASTER_REF]] to i{{[0-9]+}}
// TLS-CHECK: icmp ne i{{[0-9]+}} [[MASTER_LONG]], ptrtoint (i{{[0-9]+}}* [[T_VAR]] to i{{[0-9]+}})
// TLS-CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// TLS-CHECK: [[NOT_MASTER]]
// TLS-CHECK: [[MASTER_VAL:%.+]] = load i32, i32* [[MASTER_REF]]
// TLS-CHECK: store i32 [[MASTER_VAL]], i32* [[T_VAR]]

// threadprivate_vec = vec;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[VEC]]
// CHECK: call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}}  %{{.+}}, i8* align {{[0-9]+}} bitcast ([2 x i{{[0-9]+}}]* [[VEC]] to i8*),

// TLS-CHECK: [[MASTER_CAST:%.+]] = bitcast [2 x i32]* [[MASTER_REF2]] to i8*
// TLS-CHECK: call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} bitcast ([2 x i{{[0-9]+}}]* [[VEC]] to i8*), i8* align {{[0-9]+}} [[MASTER_CAST]]

// threadprivate_s_arr = s_arr;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[S_ARR]]
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* {{%.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN]]([[S_FLOAT_TY]]* {{.+}}, [[S_FLOAT_TY]]* {{.+}})
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]

// TLS-CHECK: [[MASTER_CAST:%.+]] = bitcast [2 x [[S_FLOAT_TY]]]* [[MASTER_REF3]] to [[S_FLOAT_TY]]*
// TLS-CHECK-DAG: [[S_ARR_SRC_BEGIN:%.+]] = phi [[S_FLOAT_TY]]* {{.*}}[[MASTER_CAST]]
// TLS-CHECK-DAG: [[S_ARR_DST_BEGIN:%.+]] = phi [[S_FLOAT_TY]]* {{.*}}getelementptr inbounds ([2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* [[S_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0)
// TLS-CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN]]([[S_FLOAT_TY]]* {{.+}}, [[S_FLOAT_TY]]* {{.+}})
// TLS-CHECK-DAG: [[S_ARR_SRC_END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[S_ARR_SRC_BEGIN]], i{{[0-9]+}} 1
// TLS-CHECK-DAG: [[S_ARR_DST_END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[S_ARR_DST_BEGIN]], i{{[0-9]+}} 1
// TLS-CHECK: icmp eq [[S_FLOAT_TY]]* [[S_ARR_DST_END]], getelementptr ([[S_FLOAT_TY]], [[S_FLOAT_TY]]* getelementptr inbounds ([2 x [[S_FLOAT_TY]]], [2 x [[S_FLOAT_TY]]]* [[S_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0), i{{[0-9]+}} 2)
// TLS-CHECK: br i1 %{{.*}}, label %[[ARR_DONE:.+]], label {{.*}}
// TLS-CHECK: [[ARR_DONE]]

// threadprivate_var = var;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[VAR]]
// CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN]]([[S_FLOAT_TY]]* {{%.+}}, [[S_FLOAT_TY]]* {{.*}}[[VAR]])
// CHECK: [[DONE]]

// TLS-CHECK: call {{.*}} [[S_FLOAT_TY_COPY_ASSIGN]]([[S_FLOAT_TY]]* {{.*}}[[VAR]], [[S_FLOAT_TY]]* {{.*}}[[MASTER_REF4]])

// CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

// TLS-CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// TLS-CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],
// TLS-CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// TLS-CHECK: ret void

// CHECK: define internal {{.*}}void [[MAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],

// TLS-CHECK: define internal {{.*}}void [[MAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// TLS-CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// threadprivate_t_var = t_var;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR]],
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},
// CHECK: [[DONE]]

// TLS-CHECK: [[MASTER_REF:%.+]] = load i32*, i32** %

// TLS-CHECK: [[MASTER_LONG:%.+]] = ptrtoint i32* [[MASTER_REF]] to i{{[0-9]+}}
// TLS-CHECK: icmp ne i{{[0-9]+}} [[MASTER_LONG]], ptrtoint (i{{[0-9]+}}* [[T_VAR]] to i{{[0-9]+}})
// TLS-CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// TLS-CHECK: [[NOT_MASTER]]
// TLS-CHECK: [[MASTER_VAL:%.+]] = load i32, i32* [[MASTER_REF]]
// TLS-CHECK: store i32 [[MASTER_VAL]], i32* [[T_VAR]]
// TLS-CHECK: [[DONE]]

// CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

// TLS-CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// TLS-CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],
// TLS-CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// TLS-CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN:@.+]]([[S_INT_TY]]* [[TEST]], [[S_INT_TY]]*
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[TMAIN_MICROTASK:@.+]] to void (i32*, i32*, ...)*))
// CHECK: call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[TMAIN_MICROTASK1:@.+]] to void (i32*, i32*, ...)*))
// CHECK: call {{.*}} [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret

// TLS-CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// TLS-CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// TLS-CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN:@.+]]([[S_INT_TY]]* [[TEST]], [[S_INT_TY]]*
// TLS-CHECK:     call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32*, [2 x i32]*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]*)* [[TMAIN_MICROTASK:@.+]] to void (i32*, i32*, ...)*),
// TLS-CHECK:     call {{.*}}void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32*)* [[TMAIN_MICROTASK1:@.+]] to void (i32*, i32*, ...)*),
//
// CHECK: define internal {{.*}}void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],
//
// TLS-CHECK: define internal {{.*}}void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// TLS-CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// threadprivate_t_var = t_var;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[TMAIN_T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[TMAIN_T_VAR]], align 128
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}}, align 128

// TLS-CHECK: [[MASTER_REF:%.+]] = load i32*, i32** %
// TLS-CHECK: [[MASTER_REF1:%.+]] = load [2 x i32]*, [2 x i32]** %
// TLS-CHECK: [[MASTER_REF2:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** %
// TLS-CHECK: [[MASTER_REF3:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** %

// TLS-CHECK: [[MASTER_LONG:%.+]] = ptrtoint i32* [[MASTER_REF]] to i{{[0-9]+}}
// TLS-CHECK: icmp ne i{{[0-9]+}} [[MASTER_LONG]], ptrtoint (i{{[0-9]+}}* [[TMAIN_T_VAR]] to i{{[0-9]+}})
// TLS-CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// TLS-CHECK: [[NOT_MASTER]]
// TLS-CHECK: [[MASTER_VAL:%.+]] = load i32, i32* [[MASTER_REF]],
// TLS-CHECK: store i32 [[MASTER_VAL]], i32* [[TMAIN_T_VAR]], align 128

// threadprivate_vec = vec;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_VEC]]
// CHECK: call {{.*}}void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} %{{.+}}, i8* align {{[0-9]+}} bitcast ([2 x i{{[0-9]+}}]* [[TMAIN_VEC]] to i8*),

// TLS-CHECK: [[MASTER_CAST:%.+]] = bitcast [2 x i32]* [[MASTER_REF1]] to i8*
// TLS-CHECK: call void @llvm.memcpy{{.*}}(i8* align {{[0-9]+}} bitcast ([2 x i{{[0-9]+}}]* [[TMAIN_VEC]] to i8*), i8* align {{[0-9]+}} [[MASTER_CAST]]

// threadprivate_s_arr = s_arr;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_S_ARR]]
// CHECK: [[S_ARR_PRIV_BEGIN:%.+]] = getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* {{%.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[S_ARR_PRIV_END:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], i{{[0-9]+}} 2
// CHECK: [[IS_EMPTY:%.+]] = icmp eq [[S_INT_TY]]* [[S_ARR_PRIV_BEGIN]], [[S_ARR_PRIV_END]]
// CHECK: br i1 [[IS_EMPTY]], label %[[S_ARR_BODY_DONE:.+]], label %[[S_ARR_BODY:.+]]
// CHECK: [[S_ARR_BODY]]
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN]]([[S_INT_TY]]* {{.+}}, [[S_INT_TY]]* {{.+}})
// CHECK: br i1 {{.+}}, label %{{.+}}, label %[[S_ARR_BODY]]

// TLS-CHECK: [[MASTER_CAST:%.+]] = bitcast [2 x [[S_INT_TY]]]* [[MASTER_REF2]] to [[S_INT_TY]]*
// TLS-CHECK-DAG: [[S_ARR_SRC_BEGIN:%.+]] = phi [[S_INT_TY]]* {{.*}}[[MASTER_CAST]]
// TLS-CHECK-DAG: [[S_ARR_DST_BEGIN:%.+]] = phi [[S_INT_TY]]* {{.*}}getelementptr inbounds ([2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[TMAIN_S_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0)
// TLS-CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN]]([[S_INT_TY]]* {{.+}}, [[S_INT_TY]]* {{.+}})
// TLS-CHECK-DAG: [[S_ARR_SRC_END:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_SRC_BEGIN]], i{{[0-9]+}} 1
// TLS-CHECK-DAG: [[S_ARR_DST_END:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_DST_BEGIN]], i{{[0-9]+}} 1
// TLS-CHECK: icmp eq [[S_INT_TY]]* [[S_ARR_DST_END]], getelementptr ([[S_INT_TY]], [[S_INT_TY]]* getelementptr inbounds ([2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[TMAIN_S_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0), i{{[0-9]+}} 2)
// TLS-CHECK: br i1 %{{.*}}, label %[[ARR_DONE:.+]], label {{.*}}
// TLS-CHECK: [[ARR_DONE]]

// threadprivate_var = var;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_VAR]]
// CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN]]([[S_INT_TY]]* {{%.+}}, [[S_INT_TY]]* {{.*}}[[TMAIN_VAR]])
// CHECK: [[DONE]]

// TLS-CHECK: call {{.*}} [[S_INT_TY_COPY_ASSIGN]]([[S_INT_TY]]* {{.*}}[[TMAIN_VAR]], [[S_INT_TY]]* {{.*}}[[MASTER_REF3]])

// CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

// TLS-CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// TLS-CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],
// TLS-CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// TLS-CHECK: ret void

// CHECK: define internal {{.*}}void [[TMAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],

// TLS-CHECK: define internal {{.*}}void [[TMAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}},
// TLS-CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// threadprivate_t_var = t_var;
// CHECK: call {{.*}}i8* @__kmpc_threadprivate_cached({{.+}} [[TMAIN_T_VAR]]
// CHECK: ptrtoint i{{[0-9]+}}* %{{.+}} to i{{[0-9]+}}
// CHECK: icmp ne i{{[0-9]+}} ptrtoint (i{{[0-9]+}}* [[TMAIN_T_VAR]] to i{{[0-9]+}}), %{{.+}}
// CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// CHECK: [[NOT_MASTER]]
// CHECK: load i{{[0-9]+}}, i{{[0-9]+}}* [[TMAIN_T_VAR]],
// CHECK: store i{{[0-9]+}} %{{.+}}, i{{[0-9]+}}* %{{.+}},
// CHECK: [[DONE]]

// TLS-CHECK: [[MASTER_REF:%.+]] = load i32*, i32** %

// TLS-CHECK: [[MASTER_LONG:%.+]] = ptrtoint i32* [[MASTER_REF]] to i{{[0-9]+}}
// TLS-CHECK: icmp ne i{{[0-9]+}} [[MASTER_LONG]], ptrtoint (i{{[0-9]+}}* [[TMAIN_T_VAR]] to i{{[0-9]+}})
// TLS-CHECK: br i1 %{{.+}}, label %[[NOT_MASTER:.+]], label %[[DONE:.+]]
// TLS-CHECK: [[NOT_MASTER]]
// TLS-CHECK: [[MASTER_VAL:%.+]] = load i32, i32* [[MASTER_REF]]
// TLS-CHECK: store i32 [[MASTER_VAL]], i32* [[TMAIN_T_VAR]]
// TLS-CHECK: [[DONE]]

// CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void

// TLS-CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_ADDR]],
// TLS-CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]],
// TLS-CHECK: call {{.*}}void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// TLS-CHECK: ret void

#endif
#else
// ARRAY-LABEL: array_func
// TLS-ARRAY-LABEL: array_func

struct St {
  int a, b;
  St() : a(0), b(0) {}
  St &operator=(const St &) { return *this; };
  ~St() {}
};

void array_func() {
  static int a[2];
  static St s[2];
// ARRAY: @__kmpc_fork_call(
// ARRAY: call i8* @__kmpc_threadprivate_cached(
// ARRAY: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.+}}, i8* align 4 bitcast ([2 x i32]* @{{.+}} to i8*), i64 8, i1 false)
// ARRAY: call dereferenceable(8) %struct.St* @{{.+}}(%struct.St* %{{.+}}, %struct.St* dereferenceable(8) %{{.+}})

// TLS-ARRAY: @__kmpc_fork_call(
// TLS-ARRAY: [[REFT:%.+]] = load [2 x i32]*, [2 x i32]** [[ADDR:%.+]],
// TLS-ARRAY: [[REF:%.+]] = bitcast [2 x i32]* [[REFT]] to i8*
// TLS-ARRAY: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 bitcast ([2 x i32]* @{{.+}} to i8*), i8* align 4 [[REF]], i64 8, i1 false)
// TLS-ARRAY: call dereferenceable(8) %struct.St* @{{.+}}(%struct.St* %{{.+}}, %struct.St* dereferenceable(8) %{{.+}})

#pragma omp threadprivate(a, s)
#pragma omp parallel copyin(a, s)
  ;
}
#endif


