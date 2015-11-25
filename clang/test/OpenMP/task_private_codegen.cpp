// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=ARRAY %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
// It doesn't pass on win32. Investigating.
// REQUIRES: shell

#ifndef ARRAY
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

// CHECK-DAG: [[KMP_TASK_T_TY:%.+]] = type { i8*, i32 (i32, i8*)*, i32, i32 (i32, i8*)* }
// CHECK-DAG: [[S_DOUBLE_TY:%.+]] = type { double }
// CHECK-DAG: [[CAP_MAIN_TY:%.+]] = type { i8 }
// CHECK-DAG: [[PRIVATES_MAIN_TY:%.+]] = type {{.?}}{ [2 x [[S_DOUBLE_TY]]], [[S_DOUBLE_TY]], i32, [2 x i32]
// CHECK-DAG: [[KMP_TASK_MAIN_TY:%.+]] = type { [[KMP_TASK_T_TY]], [[PRIVATES_MAIN_TY]] }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i32 }
// CHECK-DAG: [[CAP_TMAIN_TY:%.+]] = type { i8 }
// CHECK-DAG: [[PRIVATES_TMAIN_TY:%.+]] = type { i32, [2 x i32], [2 x [[S_INT_TY]]], [[S_INT_TY]], [104 x i8] }
// CHECK-DAG: [[KMP_TASK_TMAIN_TY:%.+]] = type { [[KMP_TASK_T_TY]], [{{[0-9]+}} x i8], [[PRIVATES_TMAIN_TY]] }
template <typename T>
T tmain() {
  S<T> test;
  T t_var __attribute__((aligned(128))) = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> var(3);
#pragma omp task private(t_var, vec, s_arr, s_arr, var, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
  static int sivar;
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global double
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{( x86_thiscallcc)?}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^ ]+}} @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 48, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// LAMBDA: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// LAMBDA: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// LAMBDA: [[SIVAR_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// LAMBDA: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
// LAMBDA: ret
#pragma omp task private(g, sivar)
  {
    // LAMBDA: define {{.+}} void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG_PTR:%.+]])
    // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
    // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
    // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: [[G_REF:%.+]] = load double*, double** [[G_PTR_REF]]
    // LAMBDA: store double 2.0{{.+}}, double* [[G_REF]]
    // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
    // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
    // LAMBDA: store i{{[0-9]+}} 3, i{{[0-9]+}}* [[SIVAR_REF]]

    // LAMBDA: define internal i32 [[TASK_ENTRY]](i32, %{{.+}}* noalias)
    g = 1;
    sivar = 2;
    // LAMBDA: store double 1.0{{.+}}, double* %{{.+}},
    // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* %{{.+}},
    // LAMBDA: call void [[INNER_LAMBDA]](%
    // LAMBDA: ret
    [&]() {
      g = 2;
      sivar = 3;
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
  // BLOCKS: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^ ]+}} @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 48, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
  // BLOCKS: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 1
  // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
  // BLOCKS: [[SIVAR_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
  // BLOCKS: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
  // BLOCKS: ret
#pragma omp task private(g, sivar)
  {
    // BLOCKS: define {{.+}} void {{@.+}}(i8*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: store double 2.0{{.+}}, double*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: store i{{[0-9]+}} 4, i{{[0-9]+}}*
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: ret

    // BLOCKS: define internal i32 [[TASK_ENTRY]](i32, %{{.+}}* noalias)
    g = 1;
    sivar = 3;
    // BLOCKS: store double 1.0{{.+}}, double* %{{.+}},
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: store i{{[0-9]+}} 3, i{{[0-9]+}}* %{{.+}},
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
    ^{
      g = 2;
      sivar = 4;
    }();
  }
  }();
  return 0;
#else
  S<double> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<double> s_arr[] = {1, 2};
  S<double> var(3);
#pragma omp task private(var, t_var, s_arr, vec, s_arr, var, sivar)
  {
    vec[0] = t_var;
    s_arr[0] = var;
    sivar = 8;
  }
#pragma omp task
  g+=1;
  return tmain<int>();
#endif
}

// CHECK: define i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_DOUBLE_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_DOUBLE_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_DOUBLE_TY]],
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[LOC:%.+]])

// CHECK: call {{.*}} [[S_DOUBLE_TY_DEF_CONSTR:@.+]]([[S_DOUBLE_TY]]* [[TEST]])

// Do not store original variables in capture struct.
// CHECK-NOT: getelementptr inbounds [[CAP_MAIN_TY]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T_TY]] task_data;
//         [[KMP_TASK_MAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc([[LOC]], i32 [[GTID]], i32 1, i64 72, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_MAIN_TY]]*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[RES_KMP_TASK:%.+]] = bitcast i8* [[RES]] to [[KMP_TASK_MAIN_TY]]*

// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// Also copy address of private copy to the corresponding shareds reference.
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 1

// Constructors for s_arr and var.
// a_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: getelementptr inbounds [2 x [[S_DOUBLE_TY]]], [2 x [[S_DOUBLE_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: call void [[S_DOUBLE_TY_DEF_CONSTR]]([[S_DOUBLE_TY]]* [[S_ARR_CUR:%.+]])
// CHECK: getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* [[S_ARR_CUR]], i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK: call void [[S_DOUBLE_TY_DEF_CONSTR]]([[S_DOUBLE_TY]]* [[PRIVATE_VAR_REF:%.+]])

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: store i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_MAIN_TY]]*)* [[DESTRUCTORS:@.+]] to i32 (i32, i8*)*), i32 (i32, i8*)** [[DESTRUCTORS_REF]],

// Start task.
// CHECK: call i32 @__kmpc_omp_task([[LOC]], i32 [[GTID]], i8* [[RES]])
// CHECK: call i32 @__kmpc_omp_task([[LOC]], i32 [[GTID]], i8*

// CHECK: = call i{{.+}} [[TMAIN_INT:@.+]]()

// No destructors must be called for private copies of s_arr and var.
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_DOUBLE_TY_DESTR:@.+]]([[S_DOUBLE_TY]]*
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: ret
//

// CHECK: define internal void [[PRIVATES_MAP_FN:@.+]]([[PRIVATES_MAIN_TY]]* noalias, [[S_DOUBLE_TY]]** noalias, i32** noalias, [2 x [[S_DOUBLE_TY]]]** noalias, [2 x i32]** noalias, i32** noalias)
// CHECK: [[PRIVATES:%.+]] = load [[PRIVATES_MAIN_TY]]*, [[PRIVATES_MAIN_TY]]**
// CHECK: [[PRIV_S_VAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i32 0, i32 0
// CHECK: [[ARG3:%.+]] = load [2 x [[S_DOUBLE_TY]]]**, [2 x [[S_DOUBLE_TY]]]*** %{{.+}},
// CHECK: store [2 x [[S_DOUBLE_TY]]]* [[PRIV_S_VAR]], [2 x [[S_DOUBLE_TY]]]** [[ARG3]],
// CHECK: [[PRIV_VAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i32 0, i32 1
// CHECK: [[ARG1:%.+]] = load [[S_DOUBLE_TY]]**, [[S_DOUBLE_TY]]*** {{.+}},
// CHECK: store [[S_DOUBLE_TY]]* [[PRIV_VAR]], [[S_DOUBLE_TY]]** [[ARG1]],
// CHECK: [[PRIV_T_VAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i32 0, i32 2
// CHECK: [[ARG2:%.+]] = load i32**, i32*** %{{.+}},
// CHECK: store i32* [[PRIV_T_VAR]], i32** [[ARG2]],
// CHECK: [[PRIV_VEC:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i32 0, i32 3
// CHECK: [[ARG4:%.+]] = load [2 x i32]**, [2 x i32]*** %{{.+}},
// CHECK: store [2 x i32]* [[PRIV_VEC]], [2 x i32]** [[ARG4]],
// CHECK: ret void

// CHECK: define internal i32 [[TASK_ENTRY]](i32, [[KMP_TASK_MAIN_TY]]* noalias)

// CHECK: [[PRIV_VAR_ADDR:%.+]] = alloca [[S_DOUBLE_TY]]*,
// CHECK: [[PRIV_T_VAR_ADDR:%.+]] = alloca i32*,
// CHECK: [[PRIV_S_ARR_ADDR:%.+]] = alloca [2 x [[S_DOUBLE_TY]]]*,
// CHECK: [[PRIV_VEC_ADDR:%.+]] = alloca [2 x i32]*,
// CHECK: [[PRIV_SIVAR_ADDR:%.+]] = alloca i32*,
// CHECK: store void (i8*, ...)* bitcast (void ([[PRIVATES_MAIN_TY]]*, [[S_DOUBLE_TY]]**, i32**, [2 x [[S_DOUBLE_TY]]]**, [2 x i32]**, i32**)* [[PRIVATES_MAP_FN]] to void (i8*, ...)*), void (i8*, ...)** [[MAP_FN_ADDR:%.+]],
// CHECK: [[MAP_FN:%.+]] = load void (i8*, ...)*, void (i8*, ...)** [[MAP_FN_ADDR]],
// CHECK: call void (i8*, ...) [[MAP_FN]](i8* %{{.+}}, [[S_DOUBLE_TY]]** [[PRIV_VAR_ADDR]], i32** [[PRIV_T_VAR_ADDR]], [2 x [[S_DOUBLE_TY]]]** [[PRIV_S_ARR_ADDR]], [2 x i32]** [[PRIV_VEC_ADDR]], i32** [[PRIV_SIVAR_ADDR]])
// CHECK: [[PRIV_VAR:%.+]] = load [[S_DOUBLE_TY]]*, [[S_DOUBLE_TY]]** [[PRIV_VAR_ADDR]],
// CHECK: [[PRIV_T_VAR:%.+]] = load i32*, i32** [[PRIV_T_VAR_ADDR]],
// CHECK: [[PRIV_S_ARR:%.+]] = load [2 x [[S_DOUBLE_TY]]]*, [2 x [[S_DOUBLE_TY]]]** [[PRIV_S_ARR_ADDR]],
// CHECK: [[PRIV_VEC:%.+]] = load [2 x i32]*, [2 x i32]** [[PRIV_VEC_ADDR]],
// CHECK: [[PRIV_SIVAR:%.+]] = load i32*, i32** [[PRIV_SIVAR_ADDR]],

// Privates actually are used.
// CHECK-DAG: [[PRIV_VAR]]
// CHECK-DAG: [[PRIV_T_VAR]]
// CHECK-DAG: [[PRIV_S_ARR]]
// CHECK-DAG: [[PRIV_VEC]]
// CHECK_DAG: [[PRIV_SIVAR]]

// CHECK: ret

// CHECK: define internal i32 [[DESTRUCTORS]](i32, [[KMP_TASK_MAIN_TY]]* noalias)
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK: call void [[S_DOUBLE_TY_DESTR]]([[S_DOUBLE_TY]]* [[PRIVATE_VAR_REF]])
// CHECK: getelementptr inbounds [2 x [[S_DOUBLE_TY]]], [2 x [[S_DOUBLE_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} -1
// CHECK: call void [[S_DOUBLE_TY_DESTR]]([[S_DOUBLE_TY]]* [[PRIVATE_S_ARR_ELEM_REF]])
// CHECK: icmp eq
// CHECK: br i1
// CHECK: ret i32

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[LOC:%.+]])

// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])

// Do not store original variables in capture struct.
// CHECK-NOT: getelementptr inbounds [[CAP_TMAIN_TY]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T_TY]] task_data;
//         [[KMP_TASK_TMAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc([[LOC]], i32 [[GTID]], i32 1, i64 256, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_TMAIN_TY]]*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[RES_KMP_TASK:%.+]] = bitcast i8* [[RES]] to [[KMP_TASK_TMAIN_TY]]*

// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0

// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 2

// Constructors for s_arr and var.
// a_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: call void [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[S_ARR_CUR:%.+]])
// CHECK: getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_CUR]], i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[PRIVATE_VAR_REF:%.+]])

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: store i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_TMAIN_TY]]*)* [[DESTRUCTORS:@.+]] to i32 (i32, i8*)*), i32 (i32, i8*)** [[DESTRUCTORS_REF]],

// Start task.
// CHECK: call i32 @__kmpc_omp_task([[LOC]], i32 [[GTID]], i8* [[RES]])

// No destructors must be called for private copies of s_arr and var.
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: ret
//

// CHECK: define internal void [[PRIVATES_MAP_FN:@.+]]([[PRIVATES_TMAIN_TY]]* noalias, i32** noalias, [2 x i32]** noalias, [2 x [[S_INT_TY]]]** noalias, [[S_INT_TY]]** noalias)
// CHECK: [[PRIVATES:%.+]] = load [[PRIVATES_TMAIN_TY]]*, [[PRIVATES_TMAIN_TY]]**
// CHECK: [[PRIV_T_VAR:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i32 0, i32 0
// CHECK: [[ARG1:%.+]] = load i32**, i32*** %{{.+}},
// CHECK: store i32* [[PRIV_T_VAR]], i32** [[ARG1]],
// CHECK: [[PRIV_VEC:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i32 0, i32 1
// CHECK: [[ARG2:%.+]] = load [2 x i32]**, [2 x i32]*** %{{.+}},
// CHECK: store [2 x i32]* [[PRIV_VEC]], [2 x i32]** [[ARG2]],
// CHECK: [[PRIV_S_VAR:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i32 0, i32 2
// CHECK: [[ARG3:%.+]] = load [2 x [[S_INT_TY]]]**, [2 x [[S_INT_TY]]]*** %{{.+}},
// CHECK: store [2 x [[S_INT_TY]]]* [[PRIV_S_VAR]], [2 x [[S_INT_TY]]]** [[ARG3]],
// CHECK: [[PRIV_VAR:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i32 0, i32 3
// CHECK: [[ARG4:%.+]] = load [[S_INT_TY]]**, [[S_INT_TY]]*** {{.+}},
// CHECK: store [[S_INT_TY]]* [[PRIV_VAR]], [[S_INT_TY]]** [[ARG4]],
// CHECK: ret void

// CHECK: define internal i32 [[TASK_ENTRY]](i32, [[KMP_TASK_TMAIN_TY]]* noalias)

// CHECK: [[PRIV_T_VAR_ADDR:%.+]] = alloca i32*,
// CHECK: [[PRIV_VEC_ADDR:%.+]] = alloca [2 x i32]*,
// CHECK: [[PRIV_S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK: [[PRIV_VAR_ADDR:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: store void (i8*, ...)* bitcast (void ([[PRIVATES_TMAIN_TY]]*, i32**, [2 x i32]**, [2 x [[S_INT_TY]]]**, [[S_INT_TY]]**)* [[PRIVATES_MAP_FN]] to void (i8*, ...)*), void (i8*, ...)** [[MAP_FN_ADDR:%.+]],
// CHECK: [[MAP_FN:%.+]] = load void (i8*, ...)*, void (i8*, ...)** [[MAP_FN_ADDR]],
// CHECK: call void (i8*, ...) [[MAP_FN]](i8* %{{.+}}, i32** [[PRIV_T_VAR_ADDR]], [2 x i32]** [[PRIV_VEC_ADDR]], [2 x [[S_INT_TY]]]** [[PRIV_S_ARR_ADDR]], [[S_INT_TY]]** [[PRIV_VAR_ADDR]])
// CHECK: [[PRIV_T_VAR:%.+]] = load i32*, i32** [[PRIV_T_VAR_ADDR]],
// CHECK: [[PRIV_VEC:%.+]] = load [2 x i32]*, [2 x i32]** [[PRIV_VEC_ADDR]],
// CHECK: [[PRIV_S_ARR:%.+]] = load [2 x [[S_INT_TY]]]*, [2 x [[S_INT_TY]]]** [[PRIV_S_ARR_ADDR]],
// CHECK: [[PRIV_VAR:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** [[PRIV_VAR_ADDR]],

// Privates actually are used.
// CHECK-DAG: [[PRIV_VAR]]
// CHECK-DAG: [[PRIV_T_VAR]]
// CHECK-DAG: [[PRIV_S_ARR]]
// CHECK-DAG: [[PRIV_VEC]]

// CHECK: ret

// CHECK: define internal i32 [[DESTRUCTORS]](i32, [[KMP_TASK_TMAIN_TY]]* noalias)
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[PRIVATE_VAR_REF]])
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} -1
// CHECK: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[PRIVATE_S_ARR_ELEM_REF]])
// CHECK: icmp eq
// CHECK: br i1
// CHECK: ret i32

#endif
#else
// ARRAY-LABEL: array_func
struct St {
  int a, b;
  St() : a(0), b(0) {}
  St &operator=(const St &) { return *this; };
  ~St() {}
};

void array_func(int n, float a[n], St s[2]) {
// ARRAY: call i8* @__kmpc_omp_task_alloc(
// ARRAY: call i32 @__kmpc_omp_task(
// ARRAY: store float** %{{.+}}, float*** %{{.+}},
// ARRAY: store %struct.St** %{{.+}}, %struct.St*** %{{.+}},
#pragma omp task private(a, s)
  ;
}
#endif

