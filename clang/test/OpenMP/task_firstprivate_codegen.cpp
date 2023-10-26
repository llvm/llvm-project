// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=ARRAY %s

// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef ARRAY
#ifndef HEADER
#define HEADER

template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  S(const S &s, T t = T()) : f(s.f + t) {}
  operator T() { return T(); }
  ~S() {}
};

volatile double g;

// CHECK-DAG: [[KMP_TASK_T_TY:%.+]] = type { ptr, ptr, i32, %union{{.+}}, %union{{.+}} }
// CHECK-DAG: [[S_DOUBLE_TY:%.+]] = type { double }
// CHECK-DAG: [[PRIVATES_MAIN_TY:%.+]] = type {{.?}}{ [2 x [[S_DOUBLE_TY]]], [[S_DOUBLE_TY]], i32, [2 x i32]
// CHECK-DAG: [[CAP_MAIN_TY:%.+]] = type { i8 }
// CHECK-DAG: [[KMP_TASK_MAIN_TY:%.+]] = type { [[KMP_TASK_T_TY]], [[PRIVATES_MAIN_TY]] }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i32 }
// CHECK-DAG: [[CAP_TMAIN_TY:%.+]] = type { i8 }
// CHECK-DAG: [[PRIVATES_TMAIN_TY:%.+]] = type { i32, [2 x i32], [2 x [[S_INT_TY]]], [[S_INT_TY]], [104 x i8] }
// CHECK-DAG: [[KMP_TASK_TMAIN_TY:%.+]] = type { [[KMP_TASK_T_TY]], [{{[0-9]+}} x i8], [[PRIVATES_TMAIN_TY]] }
template <typename T>
T tmain() {
  S<T> ttt;
  S<T> test(ttt);
  T t_var __attribute__((aligned(128))) = T();
  T vec[] = {1, 2};
  S<T> s_arr[] = {1, 2};
  S<T> var(3);
#pragma omp task firstprivate(t_var, vec, s_arr, s_arr, var, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
  static int sivar;
  float local = 0;
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] ={{.*}} global double
  // LAMBDA: [[SIVAR:@.+]] = internal global i{{[0-9]+}} 0,
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{( x86_thiscallcc)?}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: [[RES:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 56, i64 1, ptr [[TASK_ENTRY:@.+]])
// LAMBDA: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, ptr %{{.+}}, i{{.+}} 0, i{{.+}} 1
// LAMBDA: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// LAMBDA: [[G_VAL:%.+]] = load volatile double, ptr [[G]],
// LAMBDA: store volatile double [[G_VAL]], ptr [[G_PRIVATE_ADDR]]

// LAMBDA: [[SIVAR_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// LAMBDA: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, ptr [[SIVAR]],
// LAMBDA: store i{{[0-9]+}} [[SIVAR_VAL]], ptr [[SIVAR_PRIVATE_ADDR]]

// LAMBDA: [[LOCAL_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// LAMBDA: store float %{{.+}}, ptr [[LOCAL_PRIVATE_ADDR]]

// LAMBDA: call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])
// LAMBDA: ret
#pragma omp task firstprivate(g, sivar, local)
  {
    // LAMBDA: define {{.+}} void [[INNER_LAMBDA:@.+]]({{.+}} [[ARG_PTR:%.+]])
    // LAMBDA: store ptr [[ARG_PTR]], ptr [[ARG_PTR_REF:%.+]],
    // LAMBDA: [[ARG_PTR:%.+]] = load ptr, ptr [[ARG_PTR_REF]]
    // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, ptr [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: [[G_REF:%.+]] = load ptr, ptr [[G_PTR_REF]]
    // LAMBDA: store volatile double 2.0{{.+}}, ptr [[G_REF]]

    // LAMBDA: store ptr %{{.+}}, ptr %{{.+}},
    // LAMBDA: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, ptr noalias noundef %1)
    g = 1;
    sivar = 11;
    // LAMBDA: store double 1.0{{.+}}, ptr %{{.+}},
    // LAMBDA: store i{{[0-9]+}} 11, ptr %{{.+}},
    // LAMBDA: call void [[INNER_LAMBDA]]({{.+}}
    // LAMBDA: ret
    [&]() {
      g = 2;
      sivar = 22;
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] ={{.*}} global double
  // BLOCKS: [[SIVAR:@.+]] = internal global i{{[0-9]+}} 0,
  // BLOCKS-LABEL: @main
  // BLOCKS: call void {{%.+}}(ptr
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(ptr
  // BLOCKS: [[RES:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 56, i64 1, ptr [[TASK_ENTRY:@[^ ]+]])
  // BLOCKS: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, ptr %{{.+}}, i{{.+}} 0, i{{.+}} 1
  // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
  // BLOCKS: [[G_VAL:%.+]] = load volatile double, ptr [[G]],
  // BLOCKS: store volatile double [[G_VAL]], ptr [[G_PRIVATE_ADDR]]

  // BLOCKS: [[SIVAR_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
  // BLOCKS: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, ptr [[SIVAR]],
  // BLOCKS: store i{{[0-9]+}} [[SIVAR_VAL]], ptr [[SIVAR_PRIVATE_ADDR]],

  // BLOCKS: [[LOCAL_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
  // BLOCKS: store float %{{.+}}, ptr [[LOCAL_PRIVATE_ADDR]]

  // BLOCKS: call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])
  // BLOCKS: ret
#pragma omp task firstprivate(g, sivar, local)
  {
    // BLOCKS: define {{.+}} void {{@.+}}(ptr
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: store volatile double 2.0{{.+}}, ptr
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: store i{{[0-9]+}} 22, ptr
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: ret

    // BLOCKS: store ptr %{{.+}}, ptr %{{.+}},
    // BLOCKS: store ptr %{{.+}}, ptr %{{.+}},
    // BLOCKS: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, ptr noalias noundef %1)
    g = 1;
    sivar = 11;
    // BLOCKS: store double 1.0{{.+}}, ptr %{{.+}},
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: store i{{[0-9]+}} 11, ptr %{{.+}},
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(ptr
    ^{
      g = 2;
      sivar = 22;
    }();
  }
  }();
  return 0;
#else
  S<double> ttt;
  S<double> test(ttt);
  int t_var = 0;
  int vec[] = {1, 2};
  S<double> s_arr[] = {1, 2};
  S<double> var(3);
#pragma omp task firstprivate(t_var, s_arr, vec, sivar)
  {
    var.f = 1;
    vec[0] = t_var;
    s_arr[0] = S<double>(3);
    sivar = 33;
  }
  return tmain<int>();
#endif
}

// CHECK: [[SIVAR_ADDR:.+]] = internal global i{{[0-9]+}} 0,
// CHECK: define{{.*}} i{{[0-9]+}} @main()
// CHECK: alloca [[S_DOUBLE_TY]],
// CHECK: [[TEST:%.+]] = alloca [[S_DOUBLE_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_DOUBLE_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_DOUBLE_TY]],
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[LOC:@.+]])

// CHECK: call {{.*}} [[S_DOUBLE_TY_COPY_CONSTR:@.+]](ptr {{[^,]*}} [[TEST]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T]] task_data;
//         [[KMP_TASK_MAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[LOC]], i32 [[GTID]], i32 9, i64 80, i64 1, ptr [[TASK_ENTRY:@[^ ]+]])
// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], ptr [[RES]], i{{.+}} 0, i{{.+}} 0

// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// Also copy address of private copy to the corresponding shareds reference.
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], ptr [[RES]], i{{[0-9]+}} 0, i{{[0-9]+}} 1

// Constructors for s_arr and var.
// s_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: call void [[S_DOUBLE_TY_COPY_CONSTR]](ptr {{[^,]*}} [[S_ARR_CUR:%[^,]+]],
// CHECK: getelementptr [[S_DOUBLE_TY]], ptr [[S_ARR_CUR]], i{{.+}} 1
// CHECK: getelementptr [[S_DOUBLE_TY]], ptr %{{.+}}, i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK-NEXT: call void [[S_DOUBLE_TY_COPY_CONSTR]](ptr {{[^,]*}} [[PRIVATE_VAR_REF]], ptr {{.*}}[[VAR_ADDR]],

// t_var;
// CHECK: [[PRIVATE_T_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NEXT: [[T_VAR:%.+]] = load i32, ptr [[T_VAR_ADDR]],
// CHECK-NEXT: store i32 [[T_VAR]], ptr [[PRIVATE_T_VAR_REF]],

// vec;
// CHECK: [[PRIVATE_VEC_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(

// sivar;
// CHECK: [[PRIVATE_SIVAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 4
// CHECK-NEXT: [[SIVAR:%.+]] = load i32, ptr [[SIVAR_ADDR]],
// CHECK-NEXT: store i32 [[SIVAR]], ptr [[PRIVATE_SIVAR_REF]],

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], ptr [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: store ptr [[DESTRUCTORS:@.+]], ptr [[DESTRUCTORS_REF]],

// Start task.
// CHECK: call i32 @__kmpc_omp_task(ptr [[LOC]], i32 [[GTID]], ptr [[RES]])

// CHECK: = call noundef i{{.+}} [[TMAIN_INT:@.+]]()

// No destructors must be called for private copies of s_arr and var.
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_DOUBLE_TY_DESTR:@.+]](ptr
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: ret
//

// CHECK: define internal void [[PRIVATES_MAP_FN:@.+]](ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, ptr noalias noundef %3, ptr noalias noundef %4, ptr noalias noundef %5)
// CHECK: [[PRIVATES:%.+]] = load ptr, ptr
// CHECK: [[PRIV_S_VAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i32 0, i32 0
// CHECK: [[ARG3:%.+]] = load ptr, ptr %{{.+}},
// CHECK: store ptr [[PRIV_S_VAR]], ptr [[ARG3]],
// CHECK: [[PRIV_VAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i32 0, i32 1
// CHECK: [[ARG1:%.+]] = load ptr, ptr {{.+}},
// CHECK: store ptr [[PRIV_VAR]], ptr [[ARG1]],
// CHECK: [[PRIV_T_VAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i32 0, i32 2
// CHECK: [[ARG2:%.+]] = load ptr, ptr %{{.+}},
// CHECK: store ptr [[PRIV_T_VAR]], ptr [[ARG2]],
// CHECK: [[PRIV_VEC:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i32 0, i32 3
// CHECK: [[ARG4:%.+]] = load ptr, ptr %{{.+}},
// CHECK: store ptr [[PRIV_VEC]], ptr [[ARG4]],
// CHECK: [[PRIV_SIVAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i32 0, i32 4
// CHECK: [[ARG5:%.+]] = load ptr, ptr %{{.+}},
// CHECK: store ptr [[PRIV_SIVAR]], ptr [[ARG5]],
// CHECK: ret void

// CHECK: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, ptr noalias noundef %1)

// CHECK: %__context
// CHECK: [[PRIV_T_VAR_ADDR:%.+]] = alloca ptr,
// CHECK: [[PRIV_S_ARR_ADDR:%.+]] = alloca ptr,
// CHECK: [[PRIV_VEC_ADDR:%.+]] = alloca ptr,
// CHECK: [[PRIV_SIVAR_ADDR:%.+]] = alloca ptr,
// CHECK: [[PRIV_VAR_ADDR:%.+]] = alloca ptr,
// CHECK: store ptr [[PRIVATES_MAP_FN]], ptr [[MAP_FN_ADDR:%.+]],
// CHECK: [[MAP_FN:%.+]] = load ptr, ptr [[MAP_FN_ADDR]],

// CHECK: call void [[MAP_FN]](ptr %{{.+}}, ptr [[PRIV_T_VAR_ADDR]], ptr [[PRIV_S_ARR_ADDR]], ptr [[PRIV_VEC_ADDR]], ptr [[PRIV_SIVAR_ADDR]], ptr [[PRIV_VAR_ADDR]])

// CHECK: [[PRIV_T_VAR:%.+]] = load ptr, ptr [[PRIV_T_VAR_ADDR]],
// CHECK: [[PRIV_S_ARR:%.+]] = load ptr, ptr [[PRIV_S_ARR_ADDR]],
// CHECK: [[PRIV_VEC:%.+]] = load ptr, ptr [[PRIV_VEC_ADDR]],
// CHECK: [[PRIV_SIVAR:%.+]] = load ptr, ptr [[PRIV_SIVAR_ADDR]],
// CHECK: [[PRIV_VAR:%.+]] = load ptr, ptr [[PRIV_VAR_ADDR]],

// Privates actually are used.
// CHECK-DAG: [[PRIV_VAR]]
// CHECK-DAG: [[PRIV_T_VAR]]
// CHECK-DAG: [[PRIV_S_ARR]]
// CHECK-DAG: [[PRIV_VEC]]
// CHECK-DAG: [[PRIV_SIVAR]]

// CHECK: ret

// CHECK: define internal noundef i32 [[DESTRUCTORS]](i32 noundef %0, ptr noalias noundef %1)
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], ptr [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK: call void [[S_DOUBLE_TY_DESTR]](ptr {{[^,]*}} [[PRIVATE_VAR_REF]])
// CHECK: getelementptr inbounds [2 x [[S_DOUBLE_TY]]], ptr [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_DOUBLE_TY]], ptr %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_DOUBLE_TY]], ptr %{{.+}}, i{{.+}} -1
// CHECK: call void [[S_DOUBLE_TY_DESTR]](ptr {{[^,]*}} [[PRIVATE_S_ARR_ELEM_REF]])
// CHECK: icmp eq
// CHECK: br i1
// CHECK: ret i32

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: alloca [[S_INT_TY]],
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32, align 128
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr [[LOC:@.+]])

// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR:@.+]](ptr {{[^,]*}} [[TEST]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T_TY]] task_data;
//         [[KMP_TASK_TMAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr [[LOC]], i32 [[GTID]], i32 9, i64 256, i64 1, ptr [[TASK_ENTRY:@[^ ]+]])
// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], ptr [[RES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0

// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], ptr [[RES]], i{{[0-9]+}} 0, i{{[0-9]+}} 2

// t_var;
// CHECK: [[PRIVATE_T_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK-NEXT: [[T_VAR:%.+]] = load i32, ptr [[T_VAR_ADDR]], align 128
// CHECK-NEXT: store i32 [[T_VAR]], ptr [[PRIVATE_T_VAR_REF]], align 128

// vec;
// CHECK: [[PRIVATE_VEC_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(

// Constructors for s_arr and var.
// a_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], ptr [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_INT_TY]], ptr %{{.+}}, i{{.+}} 2
// CHECK: call void [[S_INT_TY_COPY_CONSTR]](ptr {{[^,]*}} [[S_ARR_CUR:%[^,]+]],
// CHECK: getelementptr [[S_INT_TY]], ptr [[S_ARR_CUR]], i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK-NEXT: call void [[S_INT_TY_COPY_CONSTR]](ptr {{[^,]*}} [[PRIVATE_VAR_REF]],

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], ptr [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: store ptr [[DESTRUCTORS:@.+]], ptr [[DESTRUCTORS_REF]],

// Start task.
// CHECK: call i32 @__kmpc_omp_task(ptr [[LOC]], i32 [[GTID]], ptr [[RES]])

// No destructors must be called for private copies of s_arr and var.
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_DESTR:@.+]](ptr
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NOT: getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: ret
//

// CHECK: define internal void [[PRIVATES_MAP_FN:@.+]](ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, ptr noalias noundef %3, ptr noalias noundef %4)
// CHECK: [[PRIVATES:%.+]] = load ptr, ptr
// CHECK: [[PRIV_T_VAR:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i32 0, i32 0
// CHECK: [[ARG1:%.+]] = load ptr, ptr %{{.+}},
// CHECK: store ptr [[PRIV_T_VAR]], ptr [[ARG1]],
// CHECK: [[PRIV_VEC:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i32 0, i32 1
// CHECK: [[ARG2:%.+]] = load ptr, ptr %{{.+}},
// CHECK: store ptr [[PRIV_VEC]], ptr [[ARG2]],
// CHECK: [[PRIV_S_VAR:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i32 0, i32 2
// CHECK: [[ARG3:%.+]] = load ptr, ptr %{{.+}},
// CHECK: store ptr [[PRIV_S_VAR]], ptr [[ARG3]],
// CHECK: [[PRIV_VAR:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i32 0, i32 3
// CHECK: [[ARG4:%.+]] = load ptr, ptr {{.+}},
// CHECK: store ptr [[PRIV_VAR]], ptr [[ARG4]],
// CHECK: ret void

// CHECK: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, ptr noalias noundef %1)
// CHECK: %__context
// CHECK-DAG: [[PRIV_T_VAR_ADDR:%.+]] = alloca ptr,
// CHECK-DAG: [[PRIV_VEC_ADDR:%.+]] = alloca ptr,
// CHECK-DAG: [[PRIV_S_ARR_ADDR:%.+]] = alloca ptr,
// CHECK-DAG: [[PRIV_VAR_ADDR:%.+]] = alloca ptr,
// CHECK: store ptr [[PRIVATES_MAP_FN]], ptr [[MAP_FN_ADDR:%.+]],
// CHECK: [[MAP_FN:%.+]] = load ptr, ptr [[MAP_FN_ADDR]],
// CHECK: call void [[MAP_FN]](ptr %{{.+}}, ptr [[PRIV_T_VAR_ADDR]], ptr [[PRIV_VEC_ADDR]], ptr [[PRIV_S_ARR_ADDR]], ptr [[PRIV_VAR_ADDR]])
// CHECK: [[PRIV_T_VAR:%.+]] = load ptr, ptr [[PRIV_T_VAR_ADDR]],
// CHECK: [[PRIV_VEC:%.+]] = load ptr, ptr [[PRIV_VEC_ADDR]],
// CHECK: [[PRIV_S_ARR:%.+]] = load ptr, ptr [[PRIV_S_ARR_ADDR]],
// CHECK: [[PRIV_VAR:%.+]] = load ptr, ptr [[PRIV_VAR_ADDR]],

// Privates actually are used.
// CHECK-DAG: [[PRIV_VAR]]
// CHECK-DAG: [[PRIV_T_VAR]]
// CHECK-DAG: [[PRIV_S_ARR]]
// CHECK-DAG: [[PRIV_VEC]]

// CHECK: ret

// CHECK: define internal noundef i32 [[DESTRUCTORS]](i32 noundef %0, ptr noalias noundef %1)
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], ptr [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], ptr [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void [[S_INT_TY_DESTR]](ptr {{[^,]*}} [[PRIVATE_VAR_REF]])
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], ptr [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_INT_TY]], ptr %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_INT_TY]], ptr %{{.+}}, i{{.+}} -1
// CHECK: call void [[S_INT_TY_DESTR]](ptr {{[^,]*}} [[PRIVATE_S_ARR_ELEM_REF]])
// CHECK: icmp eq
// CHECK: br i1
// CHECK: ret i32

#endif
#else
// ARRAY-LABEL: array_func
struct St {
  int a, b;
  St() : a(0), b(0) {}
  St(const St &) {}
  ~St() {}
};

void array_func(int n, float a[n], St s[2], int(& p)[1]) {
// ARRAY: call ptr @__kmpc_omp_task_alloc(
// ARRAY: call i32 @__kmpc_omp_task(
// ARRAY: store ptr %{{.+}}, ptr %{{.+}},
// ARRAY: store ptr %{{.+}}, ptr %{{.+}},
// ARRAY: store ptr %{{.+}}, ptr %{{.+}},
#pragma omp task firstprivate(a, s, p)
  ;
}
#endif

