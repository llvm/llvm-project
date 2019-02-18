// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap -check-prefix=BLOCKS %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

volatile double g, g_orig;
volatile double &g1 = g_orig;

template <class T>
struct S {
  T f;
  S(T a) : f(a + g) {}
  S() : f(g) {}
  operator T() { return T(); }
  S &operator&(const S &) { return *this; }
  ~S() {}
};

// CHECK-DAG: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK-DAG: [[ATOMIC_REDUCE_BARRIER_LOC:@.+]] = private unnamed_addr global %{{.+}} { i32 0, i32 18, i32 0, i32 0, i8*
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr global %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// CHECK-DAG: [[REDUCTION_LOC:@.+]] = private unnamed_addr global %{{.+}} { i32 0, i32 18, i32 0, i32 0, i8*
// CHECK-DAG: [[REDUCTION_LOCK:@.+]] = common global [8 x i32] zeroinitializer

template <typename T, int length>
T tmain() {
  T t;
  S<T> test;
  T t_var = T(), t_var1;
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> &var = test;
  S<T> var1;
  S<T> arr[length];
#pragma omp parallel
#pragma omp for reduction(+:t_var) reduction(&:var) reduction(&& : var1) reduction(min: t_var1) nowait
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
#pragma omp parallel
#pragma omp for reduction(&& : t_var)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
#pragma omp parallel
#pragma omp for reduction(+ : arr[1:length-2])
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  return T();
}

extern S<float> **foo();

int main() {
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global double
  // LAMBDA-LABEL: @main
  // LAMBDA: call void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel
#pragma omp for reduction(+:g, g1)
    for (int i = 0; i < 2; ++i) {
    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca double,

    // Reduction list for runtime.
    // LAMBDA: [[RED_LIST:%.+]] = alloca [2 x i8*],

    // LAMBDA: store double 0.0{{.+}}, double* [[G_PRIVATE_ADDR]]
    // LAMBDA: call void @__kmpc_for_static_init_4(
    g = 1;
    g1 = 1;
    // LAMBDA: store double 1.0{{.+}}, double* [[G_PRIVATE_ADDR]],
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store double* [[G_PRIVATE_ADDR]], double** [[G_PRIVATE_ADDR_REF]]
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
    // LAMBDA: call void @__kmpc_for_static_fini(

    // LAMBDA: [[G_PRIV_REF:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[RED_LIST]], i64 0, i64 0
    // LAMBDA: [[BITCAST:%.+]] = bitcast double* [[G_PRIVATE_ADDR]] to i8*
    // LAMBDA: store i8* [[BITCAST]], i8** [[G_PRIV_REF]],
    // LAMBDA: call i32 @__kmpc_reduce(
    // LAMBDA: switch i32 %{{.+}}, label %[[REDUCTION_DONE:.+]] [
    // LAMBDA: i32 1, label %[[CASE1:.+]]
    // LAMBDA: i32 2, label %[[CASE2:.+]]
    // LAMBDA: [[CASE1]]
    // LAMBDA: [[G_VAL:%.+]] = load double, double* [[G]]
    // LAMBDA: [[G_PRIV_VAL:%.+]] = load double, double* [[G_PRIVATE_ADDR]]
    // LAMBDA: [[ADD:%.+]] = fadd double [[G_VAL]], [[G_PRIV_VAL]]
    // LAMBDA: store double [[ADD]], double* [[G]]
    // LAMBDA: call void @__kmpc_end_reduce(
    // LAMBDA: br label %[[REDUCTION_DONE]]
    // LAMBDA: [[CASE2]]
    // LAMBDA: [[G_PRIV_VAL:%.+]] = load double, double* [[G_PRIVATE_ADDR]]
    // LAMBDA: fadd double
    // LAMBDA: cmpxchg i64*
    // LAMBDA: call void @__kmpc_end_reduce(
    // LAMBDA: br label %[[REDUCTION_DONE]]
    // LAMBDA: [[REDUCTION_DONE]]
    // LAMBDA: ret void
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      g1 = 2;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load double*, double** [[G_PTR_REF]]
      // LAMBDA: store double 2.0{{.+}}, double* [[G_REF]]
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global double
  // BLOCKS-LABEL: @main
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel
#pragma omp for reduction(-:g, g1)
    for (int i = 0; i < 2; ++i)  {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca double,

    // Reduction list for runtime.
    // BLOCKS: [[RED_LIST:%.+]] = alloca [2 x i8*],

    // BLOCKS: store double 0.0{{.+}}, double* [[G_PRIVATE_ADDR]]
    g = 1;
    g1 = 1;
    // BLOCKS: call void @__kmpc_for_static_init_4(
    // BLOCKS: store double 1.0{{.+}}, double* [[G_PRIVATE_ADDR]],
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: double* [[G_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
    // BLOCKS: call void @__kmpc_for_static_fini(

    // BLOCKS: [[G_PRIV_REF:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[RED_LIST]], i64 0, i64 0
    // BLOCKS: [[BITCAST:%.+]] = bitcast double* [[G_PRIVATE_ADDR]] to i8*
    // BLOCKS: store i8* [[BITCAST]], i8** [[G_PRIV_REF]],
    // BLOCKS: call i32 @__kmpc_reduce(
    // BLOCKS: switch i32 %{{.+}}, label %[[REDUCTION_DONE:.+]] [
    // BLOCKS: i32 1, label %[[CASE1:.+]]
    // BLOCKS: i32 2, label %[[CASE2:.+]]
    // BLOCKS: [[CASE1]]
    // BLOCKS: [[G_VAL:%.+]] = load double, double* [[G]]
    // BLOCKS: [[G_PRIV_VAL:%.+]] = load double, double* [[G_PRIVATE_ADDR]]
    // BLOCKS: [[ADD:%.+]] = fadd double [[G_VAL]], [[G_PRIV_VAL]]
    // BLOCKS: store double [[ADD]], double* [[G]]
    // BLOCKS: call void @__kmpc_end_reduce(
    // BLOCKS: br label %[[REDUCTION_DONE]]
    // BLOCKS: [[CASE2]]
    // BLOCKS: [[G_PRIV_VAL:%.+]] = load double, double* [[G_PRIVATE_ADDR]]
    // BLOCKS: fadd double
    // BLOCKS: cmpxchg i64*
    // BLOCKS: call void @__kmpc_end_reduce(
    // BLOCKS: br label %[[REDUCTION_DONE]]
    // BLOCKS: [[REDUCTION_DONE]]
    // BLOCKS: ret void
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      g1 = 2;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store double 2.0{{.+}}, double*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
#else
  S<float> test;
  float t_var = 0, t_var1;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2, 3, 4};
  S<float> &var = test;
  S<float> var1, arrs[10][4];
  S<float> **var2 = foo();
  S<float> vvar2[5];
  S<float> (&var3)[4] = s_arr;
#pragma omp parallel
#pragma omp for reduction(+:t_var) reduction(&:var) reduction(&& : var1) reduction(min: t_var1)
  for (int i = 0; i < 2; ++i) {
    vec[i] = t_var;
    s_arr[i] = var;
  }
  int arr[10][vec[1]];
#pragma omp parallel for reduction(+:arr[1][:vec[1]]) reduction(&:arrs[1:vec[1]][1:2])
  for (int i = 0; i < 10; ++i)
    ++arr[1][i];
#pragma omp parallel
#pragma omp for reduction(+:arr) reduction(&:arrs)
  for (int i = 0; i < 10; ++i)
    ++arr[1][i];
  // arr is a VLA, but the array section has constant length so we can generate a constant sized array!
#pragma omp parallel
#pragma omp for reduction(+:arr[1][0:2])
  for (int i = 0; i < 10; ++i)
    ++arr[1][i];
#pragma omp parallel
#pragma omp for reduction(& : var2[0 : 5][1 : 6])
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for reduction(& : var2[1][1 : 6])
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for reduction(& : var2[1 : 1][1 : 6])
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for reduction(& : var2[1 : 1][1])
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for reduction(& : vvar2[0 : 5])
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for reduction(& : var3[1 : 2])
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for reduction(& : var3[ : 2])
  for (int i = 0; i < 10; ++i)
    ;
  // TODO: The compiler should also be able to generate a constant sized array in this case!
#pragma omp parallel
#pragma omp for reduction(& : var3[2 : ])
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp parallel
#pragma omp for reduction(& : var3)
  for (int i = 0; i < 10; ++i)
    ;
  return tmain<int, 42>();
#endif
}

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 6, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, float*, [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]*, float*, [2 x i32]*, [4 x [[S_FLOAT_TY]]]*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 5, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i64, i64, i32*, [2 x i32]*, [10 x [4 x [[S_FLOAT_TY]]]]*)* [[MAIN_MICROTASK1:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i64, i64, i32*, [10 x [4 x [[S_FLOAT_TY]]]]*)* [[MAIN_MICROTASK2:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 3, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i64, i64, i32*)* [[MAIN_MICROTASK3:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[S_FLOAT_TY]]***)* [[MAIN_MICROTASK4:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[S_FLOAT_TY]]***)* [[MAIN_MICROTASK5:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[S_FLOAT_TY]]***)* [[MAIN_MICROTASK6:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[S_FLOAT_TY]]***)* [[MAIN_MICROTASK7:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [5 x [[S_FLOAT_TY]]]*)* [[MAIN_MICROTASK8:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [4 x [[S_FLOAT_TY]]]*)* [[MAIN_MICROTASK9:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [4 x [[S_FLOAT_TY]]]*)* [[MAIN_MICROTASK10:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [4 x [[S_FLOAT_TY]]]*)* [[MAIN_MICROTASK11:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [4 x [[S_FLOAT_TY]]]*)* [[MAIN_MICROTASK12:@.+]] to void
// CHECK: = call {{.*}}i{{.+}} [[TMAIN_INT_42:@.+]]()
// CHECK: call {{.*}} [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, float* dereferenceable(4) %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}}, float* dereferenceable(4) %{{.+}}, [2 x i32]* dereferenceable(8) %vec, [4 x [[S_FLOAT_TY]]]* dereferenceable(16) %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca float,
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[VAR1_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[T_VAR1_PRIV:%.+]] = alloca float,

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [4 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[T_VAR_REF:%.+]] = load float*, float** %
// CHECK: [[VAR1_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %
// CHECK: [[T_VAR1_REF:%.+]] = load float*, float** %

// For + reduction operation initial value of private variable is 0.
// CHECK: store float 0.0{{.+}}, float* [[T_VAR_PRIV]],

// For & reduction operation initial value of private variable is ones in all bits.
// CHECK: [[VAR_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[VAR_PRIV]])

// For && reduction operation initial value of private variable is 1.0.
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[VAR1_PRIV]])

// For min reduction operation initial value of private variable is largest repesentable value.
// CHECK: store float 0x47EFFFFFE0000000, float* [[T_VAR1_PRIV]],


// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_for_static_init_4(
// Skip checks for internal operations.
// CHECK: call void @__kmpc_for_static_fini(

// void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};

// CHECK: [[T_VAR_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 0
// CHECK: [[BITCAST:%.+]] = bitcast float* [[T_VAR_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[T_VAR_PRIV_REF]],
// CHECK: [[VAR_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 1
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[VAR_PRIV_REF]],
// CHECK: [[VAR1_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 2
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR1_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[VAR1_PRIV_REF]],
// CHECK: [[T_VAR1_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 3
// CHECK: [[BITCAST:%.+]] = bitcast float* [[T_VAR1_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[T_VAR1_PRIV_REF]],

// res = __kmpc_reduce(<loc>, <gtid>, <n>, sizeof(RedList), RedList, reduce_func, &<lock>);

// CHECK: [[BITCAST:%.+]] = bitcast [4 x i8*]* [[RED_LIST]] to i8*
// CHECK: [[RES:%.+]] = call i32 @__kmpc_reduce(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], i32 4, i64 32, i8* [[BITCAST]], void (i8*, i8*)* [[REDUCTION_FUNC:@.+]], [8 x i32]* [[REDUCTION_LOCK]])

// switch(res)
// CHECK: switch i32 [[RES]], label %[[RED_DONE:.+]] [
// CHECK: i32 1, label %[[CASE1:.+]]
// CHECK: i32 2, label %[[CASE2:.+]]
// CHECK: ]

// case 1:
// t_var += t_var_reduction;
// CHECK: [[T_VAR_VAL:%.+]] = load float, float* [[T_VAR_REF]],
// CHECK: [[T_VAR_PRIV_VAL:%.+]] = load float, float* [[T_VAR_PRIV]],
// CHECK: [[UP:%.+]] = fadd float [[T_VAR_VAL]], [[T_VAR_PRIV_VAL]]
// CHECK: store float [[UP]], float* [[T_VAR_REF]],

// var = var.operator &(var_reduction);
// CHECK: [[UP:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @{{.+}}([[S_FLOAT_TY]]* [[VAR_REF]], [[S_FLOAT_TY]]* dereferenceable(4) [[VAR_PRIV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_FLOAT_TY]]* [[UP]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// var1 = var1.operator &&(var1_reduction);
// CHECK: [[TO_FLOAT:%.+]] = call float @{{.+}}([[S_FLOAT_TY]]* [[VAR1_REF]])
// CHECK: [[VAR1_BOOL:%.+]] = fcmp une float [[TO_FLOAT]], 0.0
// CHECK: br i1 [[VAR1_BOOL]], label %[[TRUE:.+]], label %[[END2:.+]]
// CHECK: [[TRUE]]
// CHECK: [[TO_FLOAT:%.+]] = call float @{{.+}}([[S_FLOAT_TY]]* [[VAR1_PRIV]])
// CHECK: [[VAR1_REDUCTION_BOOL:%.+]] = fcmp une float [[TO_FLOAT]], 0.0
// CHECK: br label %[[END2]]
// CHECK: [[END2]]
// CHECK: [[COND_LVALUE:%.+]] = phi i1 [ false, %{{.+}} ], [ [[VAR1_REDUCTION_BOOL]], %[[TRUE]] ]
// CHECK: [[CONV:%.+]] = uitofp i1 [[COND_LVALUE]] to float
// CHECK:  call void @{{.+}}([[S_FLOAT_TY]]* [[COND_LVALUE:%.+]], float [[CONV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR1_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_FLOAT_TY]]* [[COND_LVALUE]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// t_var1 = min(t_var1, t_var1_reduction);
// CHECK: [[T_VAR1_VAL:%.+]] = load float, float* [[T_VAR1_REF]],
// CHECK: [[T_VAR1_PRIV_VAL:%.+]] = load float, float* [[T_VAR1_PRIV]],
// CHECK: [[CMP:%.+]] = fcmp olt float [[T_VAR1_VAL]], [[T_VAR1_PRIV_VAL]]
// CHECK: br i1 [[CMP]]
// CHECK: [[UP:%.+]] = phi float
// CHECK: store float [[UP]], float* [[T_VAR1_REF]],

// __kmpc_end_reduce(<loc>, <gtid>, &<lock>);
// CHECK: call void @__kmpc_end_reduce(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], [8 x i32]* [[REDUCTION_LOCK]])

// break;
// CHECK: br label %[[RED_DONE]]

// case 2:
// t_var += t_var_reduction;
// CHECK: load float, float* [[T_VAR_PRIV]]
// CHECK: [[T_VAR_REF_INT:%.+]] = bitcast float* [[T_VAR_REF]] to i32*
// CHECK: [[OLD1:%.+]] = load atomic i32, i32* [[T_VAR_REF_INT]] monotonic,
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[ORIG_OLD_INT:%.+]] = phi i32 [ [[OLD1]], %{{.+}} ], [ [[OLD2:%.+]], %[[CONT]] ]
// CHECK: fadd float
// CHECK: [[UP_INT:%.+]] = load i32, i32*
// CHECK: [[T_VAR_REF_INT:%.+]] = bitcast float* [[T_VAR_REF]] to i32*
// CHECK: [[RES:%.+]] = cmpxchg i32* [[T_VAR_REF_INT]], i32 [[ORIG_OLD_INT]], i32 [[UP_INT]] monotonic monotonic
// CHECK: [[OLD2:%.+]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[ATOMIC_DONE:.+]], label %[[CONT]]
// CHECK: [[ATOMIC_DONE]]

// var = var.operator &(var_reduction);
// CHECK: call void @__kmpc_critical(
// CHECK: [[UP:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @{{.+}}([[S_FLOAT_TY]]* [[VAR_REF]], [[S_FLOAT_TY]]* dereferenceable(4) [[VAR_PRIV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_FLOAT_TY]]* [[UP]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)
// CHECK: call void @__kmpc_end_critical(

// var1 = var1.operator &&(var1_reduction);
// CHECK: call void @__kmpc_critical(
// CHECK: [[TO_FLOAT:%.+]] = call float @{{.+}}([[S_FLOAT_TY]]* [[VAR1_REF]])
// CHECK: [[VAR1_BOOL:%.+]] = fcmp une float [[TO_FLOAT]], 0.0
// CHECK: br i1 [[VAR1_BOOL]], label %[[TRUE:.+]], label %[[END2:.+]]
// CHECK: [[TRUE]]
// CHECK: [[TO_FLOAT:%.+]] = call float @{{.+}}([[S_FLOAT_TY]]* [[VAR1_PRIV]])
// CHECK: [[VAR1_REDUCTION_BOOL:%.+]] = fcmp une float [[TO_FLOAT]], 0.0
// CHECK: br label %[[END2]]
// CHECK: [[END2]]
// CHECK: [[COND_LVALUE:%.+]] = phi i1 [ false, %{{.+}} ], [ [[VAR1_REDUCTION_BOOL]], %[[TRUE]] ]
// CHECK: [[CONV:%.+]] = uitofp i1 [[COND_LVALUE]] to float
// CHECK:  call void @{{.+}}([[S_FLOAT_TY]]* [[COND_LVALUE:%.+]], float [[CONV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR1_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_FLOAT_TY]]* [[COND_LVALUE]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)
// CHECK: call void @__kmpc_end_critical(

// t_var1 = min(t_var1, t_var1_reduction);
// CHECK: load float, float* [[T_VAR1_PRIV]]
// CHECK: [[T_VAR1_REF_INT:%.+]] = bitcast float* [[T_VAR1_REF]] to i32*
// CHECK: [[OLD1:%.+]] = load atomic i32, i32* [[T_VAR1_REF_INT]] monotonic,
// CHECK: br label %[[CONT:.+]]
// CHECK: [[CONT]]
// CHECK: [[ORIG_OLD_INT:%.+]] = phi i32 [ [[OLD1]], %{{.+}} ], [ [[OLD2:%.+]], %{{.+}} ]
// CHECK: [[CMP:%.+]] = fcmp olt float
// CHECK: br i1 [[CMP]]
// CHECK: phi float
// CHECK: [[UP_INT:%.+]] = load i32
// CHECK: [[T_VAR1_REF_INT:%.+]] = bitcast float* [[T_VAR1_REF]] to i32*
// CHECK: [[RES:%.+]] = cmpxchg i32* [[T_VAR1_REF_INT]], i32 [[ORIG_OLD_INT]], i32 [[UP_INT]] monotonic monotonic
// CHECK: [[OLD2:%.+]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[ATOMIC_DONE:.+]], label %[[CONT]]
// CHECK: [[ATOMIC_DONE]]

// __kmpc_end_reduce(<loc>, <gtid>, &<lock>);
// CHECK: call void @__kmpc_end_reduce(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], [8 x i32]* [[REDUCTION_LOCK]])

// break;
// CHECK: br label %[[RED_DONE]]
// CHECK: [[RED_DONE]]
// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
// CHECK: call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]], i{{[0-9]+}} [[GTID]])

// CHECK: ret void

// void reduce_func(void *lhs[<n>], void *rhs[<n>]) {
//  *(Type0*)lhs[0] = ReductionOperation0(*(Type0*)lhs[0], *(Type0*)rhs[0]);
//  ...
//  *(Type<n>-1*)lhs[<n>-1] = ReductionOperation<n>-1(*(Type<n>-1*)lhs[<n>-1],
//  *(Type<n>-1*)rhs[<n>-1]);
// }
// CHECK: define internal void [[REDUCTION_FUNC]](i8*, i8*)
// t_var_lhs = (float*)lhs[0];
// CHECK: [[T_VAR_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS:%.+]], i64 0, i64 0
// CHECK: [[T_VAR_RHS_VOID:%.+]] = load i8*, i8** [[T_VAR_RHS_REF]],
// CHECK: [[T_VAR_RHS:%.+]] = bitcast i8* [[T_VAR_RHS_VOID]] to float*
// t_var_rhs = (float*)rhs[0];
// CHECK: [[T_VAR_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS:%.+]], i64 0, i64 0
// CHECK: [[T_VAR_LHS_VOID:%.+]] = load i8*, i8** [[T_VAR_LHS_REF]],
// CHECK: [[T_VAR_LHS:%.+]] = bitcast i8* [[T_VAR_LHS_VOID]] to float*

// var_lhs = (S<float>*)lhs[1];
// CHECK: [[VAR_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS]], i64 0, i64 1
// CHECK: [[VAR_RHS_VOID:%.+]] = load i8*, i8** [[VAR_RHS_REF]],
// CHECK: [[VAR_RHS:%.+]] = bitcast i8* [[VAR_RHS_VOID]] to [[S_FLOAT_TY]]*
// var_rhs = (S<float>*)rhs[1];
// CHECK: [[VAR_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 1
// CHECK: [[VAR_LHS_VOID:%.+]] = load i8*, i8** [[VAR_LHS_REF]],
// CHECK: [[VAR_LHS:%.+]] = bitcast i8* [[VAR_LHS_VOID]] to [[S_FLOAT_TY]]*

// var1_lhs = (S<float>*)lhs[2];
// CHECK: [[VAR1_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS]], i64 0, i64 2
// CHECK: [[VAR1_RHS_VOID:%.+]] = load i8*, i8** [[VAR1_RHS_REF]],
// CHECK: [[VAR1_RHS:%.+]] = bitcast i8* [[VAR1_RHS_VOID]] to [[S_FLOAT_TY]]*
// var1_rhs = (S<float>*)rhs[2];
// CHECK: [[VAR1_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 2
// CHECK: [[VAR1_LHS_VOID:%.+]] = load i8*, i8** [[VAR1_LHS_REF]],
// CHECK: [[VAR1_LHS:%.+]] = bitcast i8* [[VAR1_LHS_VOID]] to [[S_FLOAT_TY]]*

// t_var1_lhs = (float*)lhs[3];
// CHECK: [[T_VAR1_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS]], i64 0, i64 3
// CHECK: [[T_VAR1_RHS_VOID:%.+]] = load i8*, i8** [[T_VAR1_RHS_REF]],
// CHECK: [[T_VAR1_RHS:%.+]] = bitcast i8* [[T_VAR1_RHS_VOID]] to float*
// t_var1_rhs = (float*)rhs[3];
// CHECK: [[T_VAR1_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 3
// CHECK: [[T_VAR1_LHS_VOID:%.+]] = load i8*, i8** [[T_VAR1_LHS_REF]],
// CHECK: [[T_VAR1_LHS:%.+]] = bitcast i8* [[T_VAR1_LHS_VOID]] to float*

// t_var_lhs += t_var_rhs;
// CHECK: [[T_VAR_LHS_VAL:%.+]] = load float, float* [[T_VAR_LHS]],
// CHECK: [[T_VAR_RHS_VAL:%.+]] = load float, float* [[T_VAR_RHS]],
// CHECK: [[UP:%.+]] = fadd float [[T_VAR_LHS_VAL]], [[T_VAR_RHS_VAL]]
// CHECK: store float [[UP]], float* [[T_VAR_LHS]],

// var_lhs = var_lhs.operator &(var_rhs);
// CHECK: [[UP:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @{{.+}}([[S_FLOAT_TY]]* [[VAR_LHS]], [[S_FLOAT_TY]]* dereferenceable(4) [[VAR_RHS]])
// CHECK: [[BC1:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR_LHS]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_FLOAT_TY]]* [[UP]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// var1_lhs = var1_lhs.operator &&(var1_rhs);
// CHECK: [[TO_FLOAT:%.+]] = call float @{{.+}}([[S_FLOAT_TY]]* [[VAR1_LHS]])
// CHECK: [[VAR1_BOOL:%.+]] = fcmp une float [[TO_FLOAT]], 0.0
// CHECK: br i1 [[VAR1_BOOL]], label %[[TRUE:.+]], label %[[END2:.+]]
// CHECK: [[TRUE]]
// CHECK: [[TO_FLOAT:%.+]] = call float @{{.+}}([[S_FLOAT_TY]]* [[VAR1_RHS]])
// CHECK: [[VAR1_REDUCTION_BOOL:%.+]] = fcmp une float [[TO_FLOAT]], 0.0
// CHECK: br label %[[END2]]
// CHECK: [[END2]]
// CHECK: [[COND_LVALUE:%.+]] = phi i1 [ false, %{{.+}} ], [ [[VAR1_REDUCTION_BOOL]], %[[TRUE]] ]
// CHECK: [[CONV:%.+]] = uitofp i1 [[COND_LVALUE]] to float
// CHECK:  call void @{{.+}}([[S_FLOAT_TY]]* [[COND_LVALUE:%.+]], float [[CONV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_FLOAT_TY]]* [[VAR1_LHS]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_FLOAT_TY]]* [[COND_LVALUE]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// t_var1_lhs = min(t_var1_lhs, t_var1_rhs);
// CHECK: [[T_VAR1_LHS_VAL:%.+]] = load float, float* [[T_VAR1_LHS]],
// CHECK: [[T_VAR1_RHS_VAL:%.+]] = load float, float* [[T_VAR1_RHS]],
// CHECK: [[CMP:%.+]] = fcmp olt float [[T_VAR1_LHS_VAL]], [[T_VAR1_RHS_VAL]]
// CHECK: br i1 [[CMP]]
// CHECK: [[UP:%.+]] = phi float
// CHECK: store float [[UP]], float* [[T_VAR1_LHS]],
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i64 %{{.+}}, i64 %{{.+}}, i32* {{.+}} %{{.+}}, [2 x i32]* dereferenceable(8) %{{.+}}, [10 x [4 x [[S_FLOAT_TY]]]]* dereferenceable(160) %{{.+}})

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [4 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: store i{{[0-9]+}}* %{{.+}}, i{{[0-9]+}}**
// CHECK: store i{{[0-9]+}}* %{{.+}}, i{{[0-9]+}}** [[ARR_ADDR:%.+]],
// CHECK: [[ARR:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[ARR_ADDR]],

// CHECK: [[UB_CAST:%.+]] = ptrtoint i32* [[UB1_UP:%.+]] to i64
// CHECK: [[LB_CAST:%.+]] = ptrtoint i32* [[LB1_0:%.+]] to i64
// CHECK: [[DIFF:%.+]] = sub i64 [[UB_CAST]], [[LB_CAST]]
// CHECK: [[SIZE_1:%.+]] = sdiv exact i64 [[DIFF]], ptrtoint (i32* getelementptr (i32, i32* null, i32 1) to i64)
// CHECK: [[ARR_SIZE:%.+]] = add nuw i64 [[SIZE_1]], 1
// CHECK: call i8* @llvm.stacksave()
// CHECK: [[ARR_PRIV:%.+]] = alloca i32, i64 [[ARR_SIZE]],

// Check initialization of private copy.
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[ARR_PRIV]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[ARR_PRIV]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: store i32 0, i32* %
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// CHECK: [[ARRS_PRIV:%.+]] = alloca [[S_FLOAT_TY]], i64 [[ARRS_SIZE:%.+]],

// Check initialization of private copy.
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[ARRS_PRIV]], i64 [[ARRS_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[ARRS_PRIV]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: call void @_ZN1SIfEC1Ev([[S_FLOAT_TY]]* %
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_for_static_init_4(
// Skip checks for internal operations.
// CHECK: call void @__kmpc_for_static_fini(

// void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};

// CHECK: [[ARR_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 0
// CHECK: [[BITCAST:%.+]] = bitcast i32* [[ARR_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[ARR_PRIV_REF]],
// CHECK: [[ARR_SIZE_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 1
// CHECK: [[BITCAST:%.+]] = inttoptr i64 [[ARR_SIZE]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[ARR_SIZE_REF]],
// CHECK: [[ARRS_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 2
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[ARRS_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[ARRS_PRIV_REF]],
// CHECK: [[ARRS_SIZE_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 3
// CHECK: [[BITCAST:%.+]] = inttoptr i64 [[ARRS_SIZE]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[ARRS_SIZE_REF]],

// res = __kmpc_reduce(<loc>, <gtid>, <n>, sizeof(RedList), RedList, reduce_func, &<lock>);

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast [4 x i8*]* [[RED_LIST]] to i8*
// CHECK: [[RES:%.+]] = call i32 @__kmpc_reduce_nowait(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], i32 2, i64 32, i8* [[BITCAST]], void (i8*, i8*)* [[REDUCTION_FUNC:@.+]], [8 x i32]* [[REDUCTION_LOCK]])

// switch(res)
// CHECK: switch i32 [[RES]], label %[[RED_DONE:.+]] [
// CHECK: i32 1, label %[[CASE1:.+]]
// CHECK: i32 2, label %[[CASE2:.+]]
// CHECK: ]

// case 1:
// CHECK: [[CASE1]]

// arr[:] += arr_reduction[:];
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[LB1_0]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[LB1_0]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: [[ADD:%.+]] = add nsw i32 %
// CHECK: store i32 [[ADD]], i32* %
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// arrs[:] = var.operator &(arrs_reduction[:]);
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[ARRS_LB:%.+]], i64 [[ARRS_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[ARRS_LB]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: [[AND:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @_ZN1SIfEanERKS0_([[S_FLOAT_TY]]* %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[AND]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.+}}, i8* align 4 [[BITCAST]], i64 4, i1 false)
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// __kmpc_end_reduce(<loc>, <gtid>, &<lock>);
// CHECK: call void @__kmpc_end_reduce_nowait(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], [8 x i32]* [[REDUCTION_LOCK]])

// break;
// CHECK: br label %[[RED_DONE]]

// case 2:
// CHECK: [[CASE2]]

// arr[:] += arr_reduction[:];
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[LB1_0]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[LB1_0]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: atomicrmw add i32* %{{.+}}, i32 %{{.+}} monotonic
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// arrs[:] = var.operator &(arrs_reduction[:]);
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[ARRS_LB:%.+]], i64 [[ARRS_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[ARRS_LB]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: call void @__kmpc_critical(
// CHECK: [[AND:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @_ZN1SIfEanERKS0_([[S_FLOAT_TY]]* %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[AND]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.+}}, i8* align 4 [[BITCAST]], i64 4, i1 false)
// CHECK: call void @__kmpc_end_critical(
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// break;
// CHECK: br label %[[RED_DONE]]
// CHECK: [[RED_DONE]]

// Check destruction of private copy.
// CHECK: [[END:%.+]] = getelementptr inbounds [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[ARRS_PRIV]], i64 [[ARRS_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[ARRS_PRIV]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: call void @_ZN1SIfED1Ev([[S_FLOAT_TY]]* %
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[ARRS_PRIV]]
// CHECK: br i1 [[DONE]],
// CHECK: call void @llvm.stackrestore(i8*

// CHECK: ret void

// void reduce_func(void *lhs[<n>], void *rhs[<n>]) {
//  *(Type0*)lhs[0] = ReductionOperation0(*(Type0*)lhs[0], *(Type0*)rhs[0]);
//  ...
//  *(Type<n>-1*)lhs[<n>-1] = ReductionOperation<n>-1(*(Type<n>-1*)lhs[<n>-1],
//  *(Type<n>-1*)rhs[<n>-1]);
// }
// CHECK: define internal void [[REDUCTION_FUNC]](i8*, i8*)
// arr_rhs = (int*)rhs[0];
// CHECK: [[ARR_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS:%.+]], i64 0, i64 0
// CHECK: [[ARR_RHS_VOID:%.+]] = load i8*, i8** [[ARR_RHS_REF]],
// CHECK: [[ARR_RHS:%.+]] = bitcast i8* [[ARR_RHS_VOID]] to i32*
// arr_lhs = (int*)lhs[0];
// CHECK: [[ARR_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS:%.+]], i64 0, i64 0
// CHECK: [[ARR_LHS_VOID:%.+]] = load i8*, i8** [[ARR_LHS_REF]],
// CHECK: [[ARR_LHS:%.+]] = bitcast i8* [[ARR_LHS_VOID]] to i32*

// arr_size = (size_t)lhs[1];
// CHECK: [[ARR_SIZE_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 1
// CHECK: [[ARR_SIZE_VOID:%.+]] = load i8*, i8** [[ARR_SIZE_REF]],
// CHECK: [[ARR_SIZE:%.+]] = ptrtoint i8* [[ARR_SIZE_VOID]] to i64

// arrs_rhs = (S<float>*)rhs[2];
// CHECK: [[ARRS_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS]], i64 0, i64 2
// CHECK: [[ARRS_RHS_VOID:%.+]] = load i8*, i8** [[ARRS_RHS_REF]],
// CHECK: [[ARRS_RHS:%.+]] = bitcast i8* [[ARRS_RHS_VOID]] to [[S_FLOAT_TY]]*
// arrs_lhs = (S<float>*)lhs[2];
// CHECK: [[ARRS_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 2
// CHECK: [[ARRS_LHS_VOID:%.+]] = load i8*, i8** [[ARRS_LHS_REF]],
// CHECK: [[ARRS_LHS:%.+]] = bitcast i8* [[ARRS_LHS_VOID]] to [[S_FLOAT_TY]]*

// arrs_size = (size_t)lhs[3];
// CHECK: [[ARRS_SIZE_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 3
// CHECK: [[ARRS_SIZE_VOID:%.+]] = load i8*, i8** [[ARRS_SIZE_REF]],
// CHECK: [[ARRS_SIZE:%.+]] = ptrtoint i8* [[ARRS_SIZE_VOID]] to i64

// arr_lhs[:] += arr_rhs[:];
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[ARR_LHS]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[ARR_LHS]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: [[ADD:%.+]] = add nsw i32 %
// CHECK: store i32 [[ADD]], i32* %
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// arrs_lhs = arrs_lhs.operator &(arrs_rhs);
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[ARRS_LB:%.+]], i64 [[ARRS_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[ARRS_LB]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: [[AND:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @_ZN1SIfEanERKS0_([[S_FLOAT_TY]]* %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[AND]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.+}}, i8* align 4 [[BITCAST]], i64 4, i1 false)
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK2]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i64 %{{.+}}, i64 %{{.+}}, i32* {{.+}} %{{.+}}, [10 x [4 x [[S_FLOAT_TY]]]]* dereferenceable(160) %{{.+}})

// CHECK: [[ARRS_PRIV:%.+]] = alloca [10 x [4 x [[S_FLOAT_TY]]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [3 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[ARR_SIZE:%.+]] = udiv exact i64
// CHECK: call i8* @llvm.stacksave()
// CHECK: [[ARR_PRIV:%.+]] = alloca i32, i64 [[ARR_SIZE]],

// Check initialization of private copy.
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[ARR_PRIV]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[ARR_PRIV]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: store i32 0, i32* %
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// Check initialization of private copy.
// CHECK: [[BEGIN:%.+]] = getelementptr inbounds [10 x [4 x [[S_FLOAT_TY]]]], [10 x [4 x [[S_FLOAT_TY]]]]* [[ARRS_PRIV]], i32 0, i32 0, i32 0
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[BEGIN]], i64 40
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[BEGIN]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: call void @_ZN1SIfEC1Ev([[S_FLOAT_TY]]* %
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],
// CHECK: [[LHS_BEGIN:%.+]] = bitcast [10 x [4 x [[S_FLOAT_TY]]]]* %{{.+}} to [[S_FLOAT_TY]]*
// CHECK: [[ARRS_PRIV_BEGIN:%.+]] = bitcast [10 x [4 x [[S_FLOAT_TY]]]]* [[ARRS_PRIV]] to [[S_FLOAT_TY]]*

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_for_static_init_4(
// Skip checks for internal operations.
// CHECK: call void @__kmpc_for_static_fini(

// void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};

// CHECK: [[ARR_PRIV_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST]], i64 0, i64 0
// CHECK: [[BITCAST:%.+]] = bitcast i32* [[ARR_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[ARR_PRIV_REF]],
// CHECK: [[ARR_SIZE_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST]], i64 0, i64 1
// CHECK: [[BITCAST:%.+]] = inttoptr i64 [[ARR_SIZE]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[ARR_SIZE_REF]],
// CHECK: [[ARRS_PRIV_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST]], i64 0, i64 2
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[ARRS_PRIV_BEGIN]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[ARRS_PRIV_REF]],

// res = __kmpc_reduce(<loc>, <gtid>, <n>, sizeof(RedList), RedList, reduce_func, &<lock>);

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast [3 x i8*]* [[RED_LIST]] to i8*
// CHECK: [[RES:%.+]] = call i32 @__kmpc_reduce(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], i32 2, i64 24, i8* [[BITCAST]], void (i8*, i8*)* [[REDUCTION_FUNC:@.+]], [8 x i32]* [[REDUCTION_LOCK]])

// switch(res)
// CHECK: switch i32 [[RES]], label %[[RED_DONE:.+]] [
// CHECK: i32 1, label %[[CASE1:.+]]
// CHECK: i32 2, label %[[CASE2:.+]]
// CHECK: ]

// case 1:
// CHECK: [[CASE1]]

// arr[:] += arr_reduction[:];
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[LB1_0:%.+]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[LB1_0]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: [[ADD:%.+]] = add nsw i32 %
// CHECK: store i32 [[ADD]], i32* %
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// arrs[:] = var.operator &(arrs_reduction[:]);
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[LHS_BEGIN]], i64 40
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[LHS_BEGIN]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: [[AND:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @_ZN1SIfEanERKS0_([[S_FLOAT_TY]]* %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[AND]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.+}}, i8* align 4 [[BITCAST]], i64 4, i1 false)
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// __kmpc_end_reduce(<loc>, <gtid>, &<lock>);
// CHECK: call void @__kmpc_end_reduce(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], [8 x i32]* [[REDUCTION_LOCK]])

// break;
// CHECK: br label %[[RED_DONE]]

// case 2:
// CHECK: [[CASE2]]

// arr[:] += arr_reduction[:];
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[LB1_0]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[LB1_0]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: atomicrmw add i32* %{{.+}}, i32 %{{.+}} monotonic
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// arrs[:] = var.operator &(arrs_reduction[:]);
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[LHS_BEGIN]], i64 40
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[LHS_BEGIN]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: call void @__kmpc_critical(
// CHECK: [[AND:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @_ZN1SIfEanERKS0_([[S_FLOAT_TY]]* %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[AND]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.+}}, i8* align 4 [[BITCAST]], i64 4, i1 false)
// CHECK: call void @__kmpc_end_critical(
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// break;
// CHECK: br label %[[RED_DONE]]
// CHECK: [[RED_DONE]]

// Check destruction of private copy.
// CHECK: [[BEGIN:%.+]] = getelementptr inbounds [10 x [4 x [[S_FLOAT_TY]]]], [10 x [4 x [[S_FLOAT_TY]]]]* [[ARRS_PRIV]], i32 0, i32 0, i32 0
// CHECK: [[END:%.+]] = getelementptr inbounds [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[BEGIN]], i64 40
// CHECK: br
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: call void @_ZN1SIfED1Ev([[S_FLOAT_TY]]* %
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[BEGIN]]
// CHECK: br i1 [[DONE]],
// CHECK: call void @llvm.stackrestore(i8*
// CHECK: call void @__kmpc_barrier(

// CHECK: ret void

// void reduce_func(void *lhs[<n>], void *rhs[<n>]) {
//  *(Type0*)lhs[0] = ReductionOperation0(*(Type0*)lhs[0], *(Type0*)rhs[0]);
//  ...
//  *(Type<n>-1*)lhs[<n>-1] = ReductionOperation<n>-1(*(Type<n>-1*)lhs[<n>-1],
//  *(Type<n>-1*)rhs[<n>-1]);
// }
// CHECK: define internal void [[REDUCTION_FUNC]](i8*, i8*)
// arr_rhs = (int*)rhs[0];
// CHECK: [[ARR_RHS_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST_RHS:%.+]], i64 0, i64 0
// CHECK: [[ARR_RHS_VOID:%.+]] = load i8*, i8** [[ARR_RHS_REF]],
// CHECK: [[ARR_RHS:%.+]] = bitcast i8* [[ARR_RHS_VOID]] to i32*
// arr_lhs = (int*)lhs[0];
// CHECK: [[ARR_LHS_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST_LHS:%.+]], i64 0, i64 0
// CHECK: [[ARR_LHS_VOID:%.+]] = load i8*, i8** [[ARR_LHS_REF]],
// CHECK: [[ARR_LHS:%.+]] = bitcast i8* [[ARR_LHS_VOID]] to i32*

// arr_size = (size_t)lhs[1];
// CHECK: [[ARR_SIZE_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST_LHS]], i64 0, i64 1
// CHECK: [[ARR_SIZE_VOID:%.+]] = load i8*, i8** [[ARR_SIZE_REF]],
// CHECK: [[ARR_SIZE:%.+]] = ptrtoint i8* [[ARR_SIZE_VOID]] to i64

// arrs_rhs = (S<float>*)rhs[2];
// CHECK: [[ARRS_RHS_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST_RHS]], i64 0, i64 2
// CHECK: [[ARRS_RHS_VOID:%.+]] = load i8*, i8** [[ARRS_RHS_REF]],
// CHECK: [[ARRS_RHS:%.+]] = bitcast i8* [[ARRS_RHS_VOID]] to [[S_FLOAT_TY]]*
// arrs_lhs = (S<float>*)lhs[2];
// CHECK: [[ARRS_LHS_REF:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[RED_LIST_LHS]], i64 0, i64 2
// CHECK: [[ARRS_LHS_VOID:%.+]] = load i8*, i8** [[ARRS_LHS_REF]],
// CHECK: [[ARRS_LHS:%.+]] = bitcast i8* [[ARRS_LHS_VOID]] to [[S_FLOAT_TY]]*

// arr_lhs[:] += arr_rhs[:];
// CHECK: [[END:%.+]] = getelementptr i32, i32* [[ARR_LHS]], i64 [[ARR_SIZE]]
// CHECK: [[ISEMPTY:%.+]] = icmp eq i32* [[ARR_LHS]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi i32*
// CHECK: [[ADD:%.+]] = add nsw i32 %
// CHECK: store i32 [[ADD]], i32* %
// CHECK: [[DONE:%.+]] = icmp eq i32* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// arrs_lhs = arrs_lhs.operator &(arrs_rhs);
// CHECK: [[END:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[ARRS_LB:%.+]], i64 40
// CHECK: [[ISEMPTY:%.+]] = icmp eq [[S_FLOAT_TY]]* [[ARRS_LB]], [[END]]
// CHECK: br i1 [[ISEMPTY]],
// CHECK: phi [[S_FLOAT_TY]]*
// CHECK: [[AND:%.+]] = call dereferenceable(4) [[S_FLOAT_TY]]* @_ZN1SIfEanERKS0_([[S_FLOAT_TY]]* %{{.+}}, [[S_FLOAT_TY]]* dereferenceable(4) %{{.+}})
// CHECK: [[BITCAST:%.+]] = bitcast [[S_FLOAT_TY]]* [[AND]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %{{.+}}, i8* align 4 [[BITCAST]], i64 4, i1 false)
// CHECK: [[DONE:%.+]] = icmp eq [[S_FLOAT_TY]]* %{{.+}}, [[END]]
// CHECK: br i1 [[DONE]],

// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK3]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i32* {{.*}} %{{.+}})

// CHECK: [[VLA1_ORIG_ADDR:%.+]] = alloca i64
// CHECK: [[VLA2_ORIG_ADDR:%.+]] = alloca i64
// CHECK: [[ARR_ORIG_ADDR:%.+]] = alloca i32*,
// CHECK: [[ARR_PRIV:%.+]] = alloca [1 x [2 x i32]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[VLA1:%.+]] = load i64, i64* [[VLA1_ORIG_ADDR]]
// CHECK: [[VLA2:%.+]] = load i64, i64* [[VLA2_ORIG_ADDR]]
// CHECK: [[ARR_ORIG:%.+]] = load i32*, i32** [[ARR_ORIG_ADDR]],

// CHECK: [[LOW_OFFSET:%.+]] = mul nsw i64 1, [[VLA2]]
// CHECK: [[ARRIDX:%.+]] = getelementptr inbounds i32, i32* [[ARR_ORIG]], i64 [[LOW_OFFSET]]
// CHECK: [[LOW:%.+]] = getelementptr inbounds i32, i32* [[ARRIDX]], i64 0

// CHECK: [[START:%.+]] = ptrtoint i32* [[ARR_ORIG]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint i32* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (i32* getelementptr (i32, i32* null, i32 1) to i64)
// CHECK: [[ARR_PRIV_PTR:%.+]] = bitcast [1 x [2 x i32]]* [[ARR_PRIV]] to i32*
// CHECK: [[ARR_PRIV:%.+]] = getelementptr i32, i32* [[ARR_PRIV_PTR]], i64 [[OFFSET]]

// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK4]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[S_FLOAT_TY]]*** dereferenceable(8) %{{.+}})

// CHECK: [[VAR2_ORIG_ADDR:%.+]] = alloca [[S_FLOAT_TY]]***,

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [2 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[VAR2_ORIG:%.+]] = load [[S_FLOAT_TY]]***, [[S_FLOAT_TY]]**** [[VAR2_ORIG_ADDR]],

// CHECK: [[LAST:%.+]] = ptrtoint [[S_FLOAT_TY]]* %{{.+}} to i64
// CHECK: [[FIRST:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW:%.+]] to i64
// CHECK: [[BYTE_DIF:%.+]] = sub i64 [[LAST]], [[FIRST]]
// CHECK: [[DIF:%.+]] = sdiv exact i64 [[BYTE_DIF]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[SIZE:%.+]] = add nuw i64 [[DIF]], 1
// CHECK: call i8* @llvm.stacksave()
// CHECK: [[VAR2_PRIV:%.+]] = alloca [[S_FLOAT_TY]], i64 [[SIZE]],
// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],
// CHECK: [[ORIG_START:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[LD]],
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[PSEUDO_VAR2_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VAR2_PRIV]], i64 [[OFFSET]]
// CHECK: store [[S_FLOAT_TY]]** [[REF:.+]], [[S_FLOAT_TY]]*** %
// CHECK: store [[S_FLOAT_TY]]* [[PSEUDO_VAR2_PRIV]], [[S_FLOAT_TY]]** [[REF]]
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK5]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[S_FLOAT_TY]]*** dereferenceable(8) %{{.+}})

// CHECK: [[VAR2_ORIG_ADDR:%.+]] = alloca [[S_FLOAT_TY]]***,
// CHECK: [[VAR2_PRIV:%.+]] = alloca [1 x [6 x [[S_FLOAT_TY]]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[VAR2_ORIG:%.+]] = load [[S_FLOAT_TY]]***, [[S_FLOAT_TY]]**** [[VAR2_ORIG_ADDR]],

// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],
// CHECK: [[ARRIDX:%.+]] = getelementptr inbounds [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[LD]], i64 1
// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[ARRIDX]],
// CHECK: [[LOW:%.+]] = getelementptr inbounds [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[LD]], i64 1
// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],

// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],
// CHECK: [[ORIG_START:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[LD]],
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[VAR2_PRIV_PTR:%.+]] = bitcast [1 x [6 x [[S_FLOAT_TY]]]]* [[VAR2_PRIV]] to [[S_FLOAT_TY]]*
// CHECK: [[VAR2_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VAR2_PRIV_PTR]], i64 [[OFFSET]]
// CHECK: store [[S_FLOAT_TY]]** [[REF:.+]], [[S_FLOAT_TY]]*** %
// CHECK: store [[S_FLOAT_TY]]* [[VAR2_PRIV]], [[S_FLOAT_TY]]** [[REF]]
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK6]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[S_FLOAT_TY]]*** dereferenceable(8) %{{.+}})

// CHECK: [[VAR2_ORIG_ADDR:%.+]] = alloca [[S_FLOAT_TY]]***,
// CHECK: [[VAR2_PRIV:%.+]] = alloca [1 x [6 x [[S_FLOAT_TY]]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[VAR2_ORIG:%.+]] = load [[S_FLOAT_TY]]***, [[S_FLOAT_TY]]**** [[VAR2_ORIG_ADDR]],

// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],
// CHECK: [[ARRIDX:%.+]] = getelementptr inbounds [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[LD]], i64 1
// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[ARRIDX]],
// CHECK: [[LOW:%.+]] = getelementptr inbounds [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[LD]], i64 1
// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],

// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],
// CHECK: [[ORIG_START:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[LD]],
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[VAR2_PRIV_PTR:%.+]] = bitcast [1 x [6 x [[S_FLOAT_TY]]]]* [[VAR2_PRIV]] to [[S_FLOAT_TY]]*
// CHECK: [[VAR2_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VAR2_PRIV_PTR]], i64 [[OFFSET]]
// CHECK: store [[S_FLOAT_TY]]** [[REF:.+]], [[S_FLOAT_TY]]*** %
// CHECK: store [[S_FLOAT_TY]]* [[VAR2_PRIV]], [[S_FLOAT_TY]]** [[REF]]
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK7]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[S_FLOAT_TY]]*** dereferenceable(8) %{{.+}})

// CHECK: [[VAR2_ORIG_ADDR:%.+]] = alloca [[S_FLOAT_TY]]***,
// CHECK: [[VAR2_PRIV:%.+]] = alloca [[S_FLOAT_TY]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[VAR2_ORIG:%.+]] = load [[S_FLOAT_TY]]***, [[S_FLOAT_TY]]**** [[VAR2_ORIG_ADDR]],

// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],
// CHECK: [[ARRIDX:%.+]] = getelementptr inbounds [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[LD]], i64 1
// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[ARRIDX]],
// CHECK: [[LOW:%.+]] = getelementptr inbounds [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[LD]], i64 1
// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],

// CHECK: [[LD:%.+]] = load [[S_FLOAT_TY]]**, [[S_FLOAT_TY]]*** [[VAR2_ORIG]],
// CHECK: [[ORIG_START:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** [[LD]],
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[PSEUDO_VAR2_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VAR2_PRIV]], i64 [[OFFSET]]
// CHECK: store [[S_FLOAT_TY]]** [[REF:.+]], [[S_FLOAT_TY]]*** %
// CHECK: store [[S_FLOAT_TY]]* [[PSEUDO_VAR2_PRIV]], [[S_FLOAT_TY]]** [[REF]]
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK8]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [5 x [[S_FLOAT_TY]]]* dereferenceable(20) %{{.+}})

// CHECK: [[VVAR2_ORIG_ADDR:%.+]] = alloca [5 x [[S_FLOAT_TY]]]*,
// CHECK: [[VVAR2_PRIV:%.+]] = alloca [5 x [[S_FLOAT_TY]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],
// CHECK: [[VVAR2_ORIG:%.+]] = load [5 x [[S_FLOAT_TY]]]*, [5 x [[S_FLOAT_TY]]]** [[VVAR2_ORIG_ADDR]],

// CHECK: [[LOW:%.+]] = getelementptr inbounds [5 x [[S_FLOAT_TY]]], [5 x [[S_FLOAT_TY]]]* [[VVAR2_ORIG]], i64 0, i64 0
// CHECK: [[ORIG_START:%.+]] = bitcast [5 x [[S_FLOAT_TY]]]* [[VVAR2_ORIG]] to [[S_FLOAT_TY]]*
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[VVAR2_PRIV_PTR:%.+]] = bitcast [5 x [[S_FLOAT_TY]]]* [[VVAR2_PRIV]] to [[S_FLOAT_TY]]*
// CHECK: [[VVAR2_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VVAR2_PRIV_PTR]], i64 [[OFFSET]]
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK9]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [4 x [[S_FLOAT_TY]]]* dereferenceable(16) %{{.+}})

// CHECK: [[VAR3_ORIG_ADDR:%.+]] = alloca [4 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR3_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]], [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR:%.+]],
// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],

// CHECK: [[LOW:%.+]] = getelementptr inbounds [4 x [[S_FLOAT_TY]]], [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]], i64 0, i64 1
// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],

// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: [[ORIG_START:%.+]] = bitcast [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]] to [[S_FLOAT_TY]]*
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[VAR3_PRIV_PTR:%.+]] = bitcast [2 x [[S_FLOAT_TY]]]* [[VAR3_PRIV]] to [[S_FLOAT_TY]]*
// CHECK: [[PSEUDO_VAR3_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VAR3_PRIV_PTR]], i64 [[OFFSET]]
// CHECK: [[VAR3_PRIV:%.+]] = bitcast [[S_FLOAT_TY]]* [[PSEUDO_VAR3_PRIV]] to [4 x [[S_FLOAT_TY]]]*

// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_PRIV]], [4 x [[S_FLOAT_TY]]]** %

// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK10]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [4 x [[S_FLOAT_TY]]]* dereferenceable(16) %{{.+}})

// CHECK: [[VAR3_ORIG_ADDR:%.+]] = alloca [4 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR3_PRIV:%.+]] = alloca [2 x [[S_FLOAT_TY]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]], [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR:%.+]],
// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],

// CHECK: [[LOW:%.+]] = getelementptr inbounds [4 x [[S_FLOAT_TY]]], [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]], i64 0, i64 0
// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],

// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: [[ORIG_START:%.+]] = bitcast [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]] to [[S_FLOAT_TY]]*
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[VAR3_PRIV_PTR:%.+]] = bitcast [2 x [[S_FLOAT_TY]]]* [[VAR3_PRIV]] to [[S_FLOAT_TY]]*
// CHECK: [[PSEUDO_VAR3_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VAR3_PRIV_PTR]], i64 [[OFFSET]]
// CHECK: [[VAR3_PRIV:%.+]] = bitcast [[S_FLOAT_TY]]* [[PSEUDO_VAR3_PRIV]] to [4 x [[S_FLOAT_TY]]]*

// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_PRIV]], [4 x [[S_FLOAT_TY]]]** %

// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK11]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [4 x [[S_FLOAT_TY]]]* dereferenceable(16) %{{.+}})

// CHECK: [[VAR3_ORIG_ADDR:%.+]] = alloca [4 x [[S_FLOAT_TY]]]*,

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [2 x i8*],

// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]], [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR:%.+]],
// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],

// CHECK: [[LAST:%.+]] = ptrtoint [[S_FLOAT_TY]]* %{{.+}} to i64
// CHECK: [[FIRST:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW:%.+]] to i64
// CHECK: [[BYTE_DIF:%.+]] = sub i64 [[LAST]], [[FIRST]]
// CHECK: [[DIF:%.+]] = sdiv exact i64 [[BYTE_DIF]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[SIZE:%.+]] = add nuw i64 [[DIF]], 1
// CHECK: call i8* @llvm.stacksave()
// CHECK: [[VAR3_PRIV:%.+]] = alloca [[S_FLOAT_TY]], i64 [[SIZE]],
// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: [[ORIG_START:%.+]] = bitcast [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]] to [[S_FLOAT_TY]]*
// CHECK: [[START:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_FLOAT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
// CHECK: [[PSEUDO_VAR3_PRIV:%.+]] = getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* [[VAR3_PRIV]], i64 [[OFFSET]]
// CHECK: [[VAR3_PRIV:%.+]] = bitcast [[S_FLOAT_TY]]* [[PSEUDO_VAR3_PRIV]] to [4 x [[S_FLOAT_TY]]]*

// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_PRIV]], [4 x [[S_FLOAT_TY]]]** %

// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK12]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [4 x [[S_FLOAT_TY]]]* dereferenceable(16) %{{.+}})

// CHECK: [[VAR3_ORIG_ADDR:%.+]] = alloca [4 x [[S_FLOAT_TY]]]*,
// CHECK: [[VAR3_PRIV:%.+]] = alloca [4 x [[S_FLOAT_TY]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]], [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR:%.+]],
// CHECK: [[VAR3_ORIG:%.+]] = load [4 x [[S_FLOAT_TY]]]*, [4 x [[S_FLOAT_TY]]]** [[VAR3_ORIG_ADDR]],
// CHECK: getelementptr inbounds [4 x [[S_FLOAT_TY]]], [4 x [[S_FLOAT_TY]]]* [[VAR3_PRIV]], i32 0, i32 0
// CHECK: getelementptr [[S_FLOAT_TY]], [[S_FLOAT_TY]]* %{{.+}}, i64 4

// CHECK: store [4 x [[S_FLOAT_TY]]]* [[VAR3_PRIV]], [4 x [[S_FLOAT_TY]]]** %
// CHECK: bitcast [4 x [[S_FLOAT_TY]]]* [[VAR3_ORIG]] to [[S_FLOAT_TY]]*

// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT_42]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 6, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, i32*, [[S_INT_TY]]*, [[S_INT_TY]]*, i32*, [2 x i32]*, [2 x [[S_INT_TY]]]*)* [[TMAIN_MICROTASK:@.+]] to void
// Not interested in this one:
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4,
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 5, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [42 x [[S_INT_TY]]]*, [2 x i32]*, i32*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]*)* [[TMAIN_MICROTASK2:@.+]] to void
// CHECK: call {{.*}} [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}}, [[S_INT_TY]]* dereferenceable(4) %{{.+}}, [[S_INT_TY]]* dereferenceable(4) %{{.+}}, i32* dereferenceable(4) %{{.+}}, [2 x i32]* dereferenceable(8) %{{.+}}, [2 x [[S_INT_TY]]]* dereferenceable(8) %{{.+}})
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: alloca i{{[0-9]+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[VAR1_PRIV:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[T_VAR1_PRIV:%.+]] = alloca i{{[0-9]+}},

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [4 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %
// CHECK: [[VAR1_REF:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** %
// CHECK: [[T_VAR1_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %

// For + reduction operation initial value of private variable is 0.
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[T_VAR_PRIV]],

// For & reduction operation initial value of private variable is ones in all bits.
// CHECK: [[VAR_REF:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** %
// CHECK: call {{.*}} [[S_INT_TY_CONSTR:@.+]]([[S_INT_TY]]* [[VAR_PRIV]])

// For && reduction operation initial value of private variable is 1.0.
// CHECK: call {{.*}} [[S_INT_TY_CONSTR:@.+]]([[S_INT_TY]]* [[VAR1_PRIV]])

// For min reduction operation initial value of private variable is largest repesentable value.
// CHECK: store i{{[0-9]+}} 2147483647, i{{[0-9]+}}* [[T_VAR1_PRIV]],

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: call void @__kmpc_for_static_init_4(
// Skip checks for internal operations.
// CHECK: call void @__kmpc_for_static_fini(

// void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};

// CHECK: [[T_VAR_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 0
// CHECK: [[BITCAST:%.+]] = bitcast i{{[0-9]+}}* [[T_VAR_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[T_VAR_PRIV_REF]],
// CHECK: [[VAR_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 1
// CHECK: [[BITCAST:%.+]] = bitcast [[S_INT_TY]]* [[VAR_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[VAR_PRIV_REF]],
// CHECK: [[VAR1_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 2
// CHECK: [[BITCAST:%.+]] = bitcast [[S_INT_TY]]* [[VAR1_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[VAR1_PRIV_REF]],
// CHECK: [[T_VAR1_PRIV_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST]], i64 0, i64 3
// CHECK: [[BITCAST:%.+]] = bitcast i{{[0-9]+}}* [[T_VAR1_PRIV]] to i8*
// CHECK: store i8* [[BITCAST]], i8** [[T_VAR1_PRIV_REF]],

// res = __kmpc_reduce_nowait(<loc>, <gtid>, <n>, sizeof(RedList), RedList, reduce_func, &<lock>);

// CHECK: [[BITCAST:%.+]] = bitcast [4 x i8*]* [[RED_LIST]] to i8*
// CHECK: [[RES:%.+]] = call i32 @__kmpc_reduce_nowait(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], i32 4, i64 32, i8* [[BITCAST]], void (i8*, i8*)* [[REDUCTION_FUNC:@.+]], [8 x i32]* [[REDUCTION_LOCK]])

// switch(res)
// CHECK: switch i32 [[RES]], label %[[RED_DONE:.+]] [
// CHECK: i32 1, label %[[CASE1:.+]]
// CHECK: i32 2, label %[[CASE2:.+]]
// CHECK: ]

// case 1:
// t_var += t_var_reduction;
// CHECK: [[T_VAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_REF]],
// CHECK: [[T_VAR_PRIV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_PRIV]],
// CHECK: [[UP:%.+]] = add nsw i{{[0-9]+}} [[T_VAR_VAL]], [[T_VAR_PRIV_VAL]]
// CHECK: store i{{[0-9]+}} [[UP]], i{{[0-9]+}}* [[T_VAR_REF]],

// var = var.operator &(var_reduction);
// CHECK: [[UP:%.+]] = call dereferenceable(4) [[S_INT_TY]]* @{{.+}}([[S_INT_TY]]* [[VAR_REF]], [[S_INT_TY]]* dereferenceable(4) [[VAR_PRIV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_INT_TY]]* [[VAR_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_INT_TY]]* [[UP]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// var1 = var1.operator &&(var1_reduction);
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_REF]])
// CHECK: [[VAR1_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: br i1 [[VAR1_BOOL]], label %[[TRUE:.+]], label %[[END2:.+]]
// CHECK: [[TRUE]]
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_PRIV]])
// CHECK: [[VAR1_REDUCTION_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: br label %[[END2]]
// CHECK: [[END2]]
// CHECK: [[COND_LVALUE:%.+]] = phi i1 [ false, %{{.+}} ], [ [[VAR1_REDUCTION_BOOL]], %[[TRUE]] ]
// CHECK: [[CONV:%.+]] = zext i1 [[COND_LVALUE]] to i32
// CHECK:  call void @{{.+}}([[S_INT_TY]]* [[COND_LVALUE:%.+]], i32 [[CONV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_INT_TY]]* [[VAR1_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_INT_TY]]* [[COND_LVALUE]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// t_var1 = min(t_var1, t_var1_reduction);
// CHECK: [[T_VAR1_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR1_REF]],
// CHECK: [[T_VAR1_PRIV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR1_PRIV]],
// CHECK: [[CMP:%.+]] = icmp slt i{{[0-9]+}} [[T_VAR1_VAL]], [[T_VAR1_PRIV_VAL]]
// CHECK: br i1 [[CMP]]
// CHECK: [[UP:%.+]] = phi i32
// CHECK: store i{{[0-9]+}} [[UP]], i{{[0-9]+}}* [[T_VAR1_REF]],

// __kmpc_end_reduce_nowait(<loc>, <gtid>, &<lock>);
// CHECK: call void @__kmpc_end_reduce_nowait(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], [8 x i32]* [[REDUCTION_LOCK]])

// break;
// CHECK: br label %[[RED_DONE]]

// case 2:
// t_var += t_var_reduction;
// CHECK: [[T_VAR_PRIV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_PRIV]]
// CHECK: atomicrmw add i32* [[T_VAR_REF]], i32 [[T_VAR_PRIV_VAL]] monotonic

// var = var.operator &(var_reduction);
// CHECK: call void @__kmpc_critical(
// CHECK: [[UP:%.+]] = call dereferenceable(4) [[S_INT_TY]]* @{{.+}}([[S_INT_TY]]* [[VAR_REF]], [[S_INT_TY]]* dereferenceable(4) [[VAR_PRIV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_INT_TY]]* [[VAR_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_INT_TY]]* [[UP]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)
// CHECK: call void @__kmpc_end_critical(

// var1 = var1.operator &&(var1_reduction);
// CHECK: call void @__kmpc_critical(
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_REF]])
// CHECK: [[VAR1_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: br i1 [[VAR1_BOOL]], label %[[TRUE:.+]], label %[[END2:.+]]
// CHECK: [[TRUE]]
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_PRIV]])
// CHECK: [[VAR1_REDUCTION_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: br label %[[END2]]
// CHECK: [[END2]]
// CHECK: [[COND_LVALUE:%.+]] = phi i1 [ false, %{{.+}} ], [ [[VAR1_REDUCTION_BOOL]], %[[TRUE]] ]
// CHECK: [[CONV:%.+]] = zext i1 [[COND_LVALUE]] to i32
// CHECK:  call void @{{.+}}([[S_INT_TY]]* [[COND_LVALUE:%.+]], i32 [[CONV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_INT_TY]]* [[VAR1_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_INT_TY]]* [[COND_LVALUE]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)
// CHECK: call void @__kmpc_end_critical(

// t_var1 = min(t_var1, t_var1_reduction);
// CHECK: [[T_VAR1_PRIV_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR1_PRIV]]
// CHECK: atomicrmw min i32* [[T_VAR1_REF]], i32 [[T_VAR1_PRIV_VAL]] monotonic

// break;
// CHECK: br label %[[RED_DONE]]
// CHECK: [[RED_DONE]]
// CHECK-DAG: call {{.*}} [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call {{.*}} [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: ret void

// void reduce_func(void *lhs[<n>], void *rhs[<n>]) {
//  *(Type0*)lhs[0] = ReductionOperation0(*(Type0*)lhs[0], *(Type0*)rhs[0]);
//  ...
//  *(Type<n>-1*)lhs[<n>-1] = ReductionOperation<n>-1(*(Type<n>-1*)lhs[<n>-1],
//  *(Type<n>-1*)rhs[<n>-1]);
// }
// CHECK: define internal void [[REDUCTION_FUNC]](i8*, i8*)
// t_var_lhs = (i{{[0-9]+}}*)lhs[0];
// CHECK: [[T_VAR_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS:%.+]], i64 0, i64 0
// CHECK: [[T_VAR_RHS_VOID:%.+]] = load i8*, i8** [[T_VAR_RHS_REF]],
// CHECK: [[T_VAR_RHS:%.+]] = bitcast i8* [[T_VAR_RHS_VOID]] to i{{[0-9]+}}*
// t_var_rhs = (i{{[0-9]+}}*)rhs[0];
// CHECK: [[T_VAR_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS:%.+]], i64 0, i64 0
// CHECK: [[T_VAR_LHS_VOID:%.+]] = load i8*, i8** [[T_VAR_LHS_REF]],
// CHECK: [[T_VAR_LHS:%.+]] = bitcast i8* [[T_VAR_LHS_VOID]] to i{{[0-9]+}}*

// var_lhs = (S<i{{[0-9]+}}>*)lhs[1];
// CHECK: [[VAR_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS]], i64 0, i64 1
// CHECK: [[VAR_RHS_VOID:%.+]] = load i8*, i8** [[VAR_RHS_REF]],
// CHECK: [[VAR_RHS:%.+]] = bitcast i8* [[VAR_RHS_VOID]] to [[S_INT_TY]]*
// var_rhs = (S<i{{[0-9]+}}>*)rhs[1];
// CHECK: [[VAR_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 1
// CHECK: [[VAR_LHS_VOID:%.+]] = load i8*, i8** [[VAR_LHS_REF]],
// CHECK: [[VAR_LHS:%.+]] = bitcast i8* [[VAR_LHS_VOID]] to [[S_INT_TY]]*

// var1_lhs = (S<i{{[0-9]+}}>*)lhs[2];
// CHECK: [[VAR1_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS]], i64 0, i64 2
// CHECK: [[VAR1_RHS_VOID:%.+]] = load i8*, i8** [[VAR1_RHS_REF]],
// CHECK: [[VAR1_RHS:%.+]] = bitcast i8* [[VAR1_RHS_VOID]] to [[S_INT_TY]]*
// var1_rhs = (S<i{{[0-9]+}}>*)rhs[2];
// CHECK: [[VAR1_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 2
// CHECK: [[VAR1_LHS_VOID:%.+]] = load i8*, i8** [[VAR1_LHS_REF]],
// CHECK: [[VAR1_LHS:%.+]] = bitcast i8* [[VAR1_LHS_VOID]] to [[S_INT_TY]]*

// t_var1_lhs = (i{{[0-9]+}}*)lhs[3];
// CHECK: [[T_VAR1_RHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_RHS]], i64 0, i64 3
// CHECK: [[T_VAR1_RHS_VOID:%.+]] = load i8*, i8** [[T_VAR1_RHS_REF]],
// CHECK: [[T_VAR1_RHS:%.+]] = bitcast i8* [[T_VAR1_RHS_VOID]] to i{{[0-9]+}}*
// t_var1_rhs = (i{{[0-9]+}}*)rhs[3];
// CHECK: [[T_VAR1_LHS_REF:%.+]] = getelementptr inbounds [4 x i8*], [4 x i8*]* [[RED_LIST_LHS]], i64 0, i64 3
// CHECK: [[T_VAR1_LHS_VOID:%.+]] = load i8*, i8** [[T_VAR1_LHS_REF]],
// CHECK: [[T_VAR1_LHS:%.+]] = bitcast i8* [[T_VAR1_LHS_VOID]] to i{{[0-9]+}}*

// t_var_lhs += t_var_rhs;
// CHECK: [[T_VAR_LHS_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_LHS]],
// CHECK: [[T_VAR_RHS_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR_RHS]],
// CHECK: [[UP:%.+]] = add nsw i{{[0-9]+}} [[T_VAR_LHS_VAL]], [[T_VAR_RHS_VAL]]
// CHECK: store i{{[0-9]+}} [[UP]], i{{[0-9]+}}* [[T_VAR_LHS]],

// var_lhs = var_lhs.operator &(var_rhs);
// CHECK: [[UP:%.+]] = call dereferenceable(4) [[S_INT_TY]]* @{{.+}}([[S_INT_TY]]* [[VAR_LHS]], [[S_INT_TY]]* dereferenceable(4) [[VAR_RHS]])
// CHECK: [[BC1:%.+]] = bitcast [[S_INT_TY]]* [[VAR_LHS]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_INT_TY]]* [[UP]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// var1_lhs = var1_lhs.operator &&(var1_rhs);
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_LHS]])
// CHECK: [[VAR1_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: br i1 [[VAR1_BOOL]], label %[[TRUE:.+]], label %[[END2:.+]]
// CHECK: [[TRUE]]
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_RHS]])
// CHECK: [[VAR1_REDUCTION_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: br label %[[END2]]
// CHECK: [[END2]]
// CHECK: [[COND_LVALUE:%.+]] = phi i1 [ false, %{{.+}} ], [ [[VAR1_REDUCTION_BOOL]], %[[TRUE]] ]
// CHECK: [[CONV:%.+]] = zext i1 [[COND_LVALUE]] to i32
// CHECK:  call void @{{.+}}([[S_INT_TY]]* [[COND_LVALUE:%.+]], i32 [[CONV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_INT_TY]]* [[VAR1_LHS]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_INT_TY]]* [[COND_LVALUE]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[BC1]], i8* align 4 [[BC2]], i64 4, i1 false)

// t_var1_lhs = min(t_var1_lhs, t_var1_rhs);
// CHECK: [[T_VAR1_LHS_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR1_LHS]],
// CHECK: [[T_VAR1_RHS_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR1_RHS]],
// CHECK: [[CMP:%.+]] = icmp slt i{{[0-9]+}} [[T_VAR1_LHS_VAL]], [[T_VAR1_RHS_VAL]]
// CHECK: br i1 [[CMP]]
// CHECK: [[UP:%.+]] = phi i32
// CHECK: store i{{[0-9]+}} [[UP]], i{{[0-9]+}}* [[T_VAR1_LHS]],
// CHECK: ret void

// CHECK: define internal void [[TMAIN_MICROTASK2]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [42 x [[S_INT_TY]]]* dereferenceable(168) %{{.*}}, [2 x i32]* dereferenceable(8) %{{.*}}, i32* dereferenceable(4) %{{.*}}, [2 x [[S_INT_TY]]]* dereferenceable(8) %{{.*}}, [[S_INT_TY]]* dereferenceable(4) %{{.*}})

// CHECK: [[ARR_ORIG_ADDR:%.+]] = alloca [42 x [[S_INT_TY]]]*,
// CHECK: [[ARR_PRIV:%.+]] = alloca [40 x [[S_INT_TY]]],

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [1 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[ARR_ORIG:%.+]] = load [42 x [[S_INT_TY]]]*, [42 x [[S_INT_TY]]]** [[ARR_ORIG_ADDR]],
// CHECK: [[LOW:%.+]] = getelementptr inbounds [42 x [[S_INT_TY]]], [42 x [[S_INT_TY]]]* [[ARR_ORIG]], i64 0, i64 1
// CHECK: [[ORIG_START:%.+]] = bitcast [42 x [[S_INT_TY]]]* [[ARR_ORIG]] to [[S_INT_TY]]*
// CHECK: [[START:%.+]] = ptrtoint [[S_INT_TY]]* [[ORIG_START]] to i64
// CHECK: [[LOW_BOUND:%.+]] = ptrtoint [[S_INT_TY]]* [[LOW]] to i64
// CHECK: [[OFFSET_BYTES:%.+]] = sub i64 [[START]], [[LOW_BOUND]]
// CHECK: [[OFFSET:%.+]] = sdiv exact i64 [[OFFSET_BYTES]], ptrtoint (i32* getelementptr (i32, i32* null, i32 1) to i64)
// CHECK: [[ARR_PRIV_PTR:%.+]] = bitcast [40 x [[S_INT_TY]]]* [[ARR_PRIV]] to [[S_INT_TY]]*
// CHECK: [[PSEUDO_ARR_PRIV:%.+]] = getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[ARR_PRIV_PTR]], i64 [[OFFSET]]
// CHECK: [[ARR_PRIV:%.+]] = bitcast [[S_INT_TY]]* [[PSEUDO_ARR_PRIV]] to [42 x [[S_INT_TY]]]*

// CHECK: ret void

#endif

