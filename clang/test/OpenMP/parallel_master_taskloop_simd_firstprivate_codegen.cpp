// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=ARRAY %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp-simd -x c++ -std=c++11 -DARRAY -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
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

// CHECK-DAG: [[KMP_TASK_T_TY:%.+]] = type { i8*, i32 (i32, i8*)*, i32, %union{{.+}}, %union{{.+}}, i64, i64, i64, i32, i8* }
// CHECK-DAG: [[S_DOUBLE_TY:%.+]] = type { double }
// CHECK-DAG: [[PRIVATES_MAIN_TY:%.+]] = type {{.?}}{ [2 x [[S_DOUBLE_TY]]], [[S_DOUBLE_TY]], i32, [2 x i32]
// CHECK-DAG: [[CAP_MAIN_TY:%.+]] = type { [2 x [[S_DOUBLE_TY]]]*, [[S_DOUBLE_TY]]* }
// CHECK-DAG: [[KMP_TASK_MAIN_TY:%.+]] = type { [[KMP_TASK_T_TY]], [[PRIVATES_MAIN_TY]] }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i32 }
// CHECK-DAG: [[CAP_TMAIN_TY:%.+]] = type { [2 x [[S_INT_TY]]]*, [[S_INT_TY]]* }
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
#pragma omp parallel master taskloop simd firstprivate(t_var, vec, s_arr, s_arr, var, var)
  for (int i = 0; i < 10; ++i) {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
  static int sivar;
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] ={{.*}} global double
  // LAMBDA: [[SIVAR:@.+]] = internal global i{{[0-9]+}} 0,
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{( x86_thiscallcc)?}} void [[OUTER_LAMBDA:@.+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
// LAMBDA:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(
// LAMBDA-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// LAMBDA-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// LAMBDA:       [[THEN]]
// LAMBDA: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^ ]+}} @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 96, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// LAMBDA: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 1
// LAMBDA: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// LAMBDA: [[G_VAL:%.+]] = load volatile double, double* @{{.+}},
// LAMBDA: store volatile double [[G_VAL]], double* [[G_PRIVATE_ADDR]]

// LAMBDA: [[SIVAR_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// LAMBDA: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* @{{.+}},
// LAMBDA: store i{{[0-9]+}} [[SIVAR_VAL]], i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]]

// LAMBDA: call void @__kmpc_taskloop(%{{.+}}* @{{.+}}, i32 %{{.+}}, i8* [[RES]], i32 1, i64* %{{.+}}, i64* %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, i8* null)
// LAMBDA:  call {{.*}}void @__kmpc_end_master(
// LAMBDA-NEXT:  br label {{%?}}[[EXIT]]
// LAMBDA:       [[EXIT]]
// LAMBDA: ret
#pragma omp parallel master taskloop simd firstprivate(g, sivar)
  for (int i = 0; i < 10; ++i) {
    // LAMBDA: define {{.+}} void [[INNER_LAMBDA:@.+]](%{{.+}}* {{[^,]*}} [[ARG_PTR:%.+]])
    // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
    // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
    // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: [[G_REF:%.+]] = load double*, double** [[G_PTR_REF]]
    // LAMBDA: store double 2.0{{.+}}, double* [[G_REF]]

    // LAMBDA: store double* %{{.+}}, double** %{{.+}},
    // LAMBDA: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, %{{.+}}* noalias noundef %1)
    g = 1;
    sivar = 11;
    // LAMBDA: store double 1.0{{.+}}, double* %{{.+}},
    // LAMBDA: store i{{[0-9]+}} 11, i{{[0-9]+}}* %{{.+}},
    // LAMBDA: call void [[INNER_LAMBDA]](%
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
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(
  // BLOCKS-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
  // BLOCKS-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
  // BLOCKS:       [[THEN]]
  // BLOCKS: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^ ]+}} @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 96, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
  // BLOCKS: [[PRIVATES:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* %{{.+}}, i{{.+}} 0, i{{.+}} 1
  // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
  // BLOCKS: [[G_VAL:%.+]] = load volatile double, double* @{{.+}},
  // BLOCKS: store volatile double [[G_VAL]], double* [[G_PRIVATE_ADDR]]

  // BLOCKS: [[SIVAR_PRIVATE_ADDR:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
  // BLOCKS: [[SIVAR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* @{{.+}},
  // BLOCKS: store i{{[0-9]+}} [[SIVAR_VAL]], i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]]
  // BLOCKS: call void @__kmpc_taskloop(%{{.+}}* @{{.+}}, i32 %{{.+}}, i8* [[RES]], i32 1, i64* %{{.+}}, i64* %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, i8* null)
  // BLOCKS:  call {{.*}}void @__kmpc_end_master(
  // BLOCKS-NEXT:  br label {{%?}}[[EXIT]]
  // BLOCKS:       [[EXIT]]
  // BLOCKS: ret
#pragma omp parallel master taskloop simd firstprivate(g, sivar)
  for (int i = 0; i < 10; ++i) {
    // BLOCKS: define {{.+}} void {{@.+}}(i8*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: store double 2.0{{.+}}, double*
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: store i{{[0-9]+}} 22, i{{[0-9]+}}*
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: ret

    // BLOCKS: store double* %{{.+}}, double** %{{.+}},
    // BLOCKS: store i{{[0-9]+}}* %{{.+}}, i{{[0-9]+}}** %{{.+}},
    // BLOCKS: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, %{{.+}}* noalias noundef %1)
    g = 1;
    sivar = 11;
    // BLOCKS: store double 1.0{{.+}}, double* %{{.+}},
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: store i{{[0-9]+}} 11, i{{[0-9]+}}* %{{.+}},
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8
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
#pragma omp parallel master taskloop simd firstprivate(var, t_var, s_arr, vec, s_arr, var, sivar)
  for (int i = 0; i < 10; ++i) {
    vec[0] = t_var;
    s_arr[0] = var;
    sivar = 33;
  }
  return tmain<int>();
#endif
}

// CHECK: [[SIVAR:.+]] = internal global i{{[0-9]+}} 0,
// CHECK: define{{.*}} i{{[0-9]+}} @main()
// CHECK: alloca [[S_DOUBLE_TY]],
// CHECK: [[TEST:%.+]] = alloca [[S_DOUBLE_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32,
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_DOUBLE_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_DOUBLE_TY]],

// CHECK: call {{.*}} [[S_DOUBLE_TY_COPY_CONSTR:@.+]]([[S_DOUBLE_TY]]* {{[^,]*}} [[TEST]],

// CHECK:       [[RES:%.+]] = call {{.*}}i32 @__kmpc_master(
// CHECK-NEXT:  [[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       [[THEN]]
// Store original variables in capture struct.
// CHECK: [[S_ARR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: store [2 x [[S_DOUBLE_TY]]]* %{{.+}}, [2 x [[S_DOUBLE_TY]]]** [[S_ARR_REF]],
// CHECK: [[VAR_REF:%.+]] = getelementptr inbounds [[CAP_MAIN_TY]], [[CAP_MAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: store [[S_DOUBLE_TY]]* %{{.+}}, [[S_DOUBLE_TY]]** [[VAR_REF]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T]] task_data;
//         [[KMP_TASK_MAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 9, i64 120, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_MAIN_TY]]*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[RES_KMP_TASK:%.+]] = bitcast i8* [[RES]] to [[KMP_TASK_MAIN_TY]]*

// Fill kmp_task_t->shareds by copying from original capture argument.
// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF_ADDR:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_REF_ADDR]],
// CHECK: [[CAPTURES_ADDR:%.+]] = bitcast [[CAP_MAIN_TY]]* %{{.+}} to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[SHAREDS_REF]], i8* align 8 [[CAPTURES_ADDR]], i64 16, i1 false)

// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// Also copy address of private copy to the corresponding shareds reference.
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 1

// Constructors for s_arr and var.
// s_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: bitcast [2 x [[S_DOUBLE_TY]]]* %{{.+}} to [[S_DOUBLE_TY]]*
// CHECK: call void [[S_DOUBLE_TY_COPY_CONSTR]]([[S_DOUBLE_TY]]* {{[^,]*}} [[S_ARR_CUR:%[^,]+]],
// CHECK: getelementptr [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* [[S_ARR_CUR]], i{{.+}} 1
// CHECK: getelementptr [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK-NEXT: call void [[S_DOUBLE_TY_COPY_CONSTR]]([[S_DOUBLE_TY]]* {{[^,]*}} [[PRIVATE_VAR_REF]], [[S_DOUBLE_TY]]* {{.*}},

// t_var;
// CHECK: [[PRIVATE_T_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK-NEXT: [[T_VAR:%.+]] = load i32, i32* %{{.+}},
// CHECK-NEXT: store i32 [[T_VAR]], i32* [[PRIVATE_T_VAR_REF]],

// vec;
// CHECK: [[PRIVATE_VEC_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK-NEXT: bitcast [2 x i32]* [[PRIVATE_VEC_REF]] to i8*
// CHECK-NEXT: bitcast [2 x i32]* %{{.+}} to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(

// sivar;
// CHECK: [[PRIVATE_SIVAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 4
// CHECK-NEXT: [[SIVAR:%.+]] = load i{{.+}}, i{{.+}}* @{{.+}},
// CHECK-NEXT: store i32 [[SIVAR]], i32* [[PRIVATE_SIVAR_REF]],

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: [[DESTRUCTORS_PTR:%.+]] = bitcast %union{{.+}}* [[DESTRUCTORS_REF]] to i32 (i32, i8*)**
// CHECK: store i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_MAIN_TY]]*)* [[DESTRUCTORS:@.+]] to i32 (i32, i8*)*), i32 (i32, i8*)** [[DESTRUCTORS_PTR]],

// Start task.
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]], i32 1, i64* %{{.+}}, i64* %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, i8* bitcast (void ([[KMP_TASK_MAIN_TY]]*, [[KMP_TASK_MAIN_TY]]*, i32)* [[MAIN_DUP:@.+]] to i8*))
// CHECK:  call {{.*}}void @__kmpc_end_master(
// CHECK-NEXT:  br label {{%?}}[[EXIT]]
// CHECK:       [[EXIT]]

// CHECK: define internal void [[PRIVATES_MAP_FN:@.+]]([[PRIVATES_MAIN_TY]]* noalias noundef %0, [[S_DOUBLE_TY]]** noalias noundef %1, i32** noalias noundef %2, [2 x [[S_DOUBLE_TY]]]** noalias noundef %3, [2 x i32]** noalias noundef %4, i32** noalias noundef %5)
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
// CHECK: [[PRIV_SIVAR:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i32 0, i32 4
// CHECK: [[ARG5:%.+]] = load i{{[0-9]+}}**, i{{[0-9]+}}*** %{{.+}},
// CHECK: store i{{[0-9]+}}* [[PRIV_SIVAR]], i{{[0-9]+}}** [[ARG5]],
// CHECK: ret void

// CHECK: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, [[KMP_TASK_MAIN_TY]]* noalias noundef %1)

// CHECK: [[PRIV_VAR_ADDR:%.+]] = alloca [[S_DOUBLE_TY]]*,
// CHECK: [[PRIV_T_VAR_ADDR:%.+]] = alloca i32*,
// CHECK: [[PRIV_S_ARR_ADDR:%.+]] = alloca [2 x [[S_DOUBLE_TY]]]*,
// CHECK: [[PRIV_VEC_ADDR:%.+]] = alloca [2 x i32]*,
// CHECK: [[PRIV_SIVAR_ADDR:%.+]] = alloca i32*,
// CHECK: store void (i8*, ...)* bitcast (void ([[PRIVATES_MAIN_TY]]*, [[S_DOUBLE_TY]]**, i32**, [2 x [[S_DOUBLE_TY]]]**, [2 x i32]**, i32**)* [[PRIVATES_MAP_FN]] to void (i8*, ...)*), void (i8*, ...)** [[MAP_FN_ADDR:%.+]],
// CHECK: [[MAP_FN:%.+]] = load void (i8*, ...)*, void (i8*, ...)** [[MAP_FN_ADDR]],

// CHECK: [[FN:%.+]] = bitcast void (i8*, ...)* [[MAP_FN]] to void (i8*,
// CHECK: call void [[FN]](i8* %{{.+}}, [[S_DOUBLE_TY]]** [[PRIV_VAR_ADDR]], i32** [[PRIV_T_VAR_ADDR]], [2 x [[S_DOUBLE_TY]]]** [[PRIV_S_ARR_ADDR]], [2 x i32]** [[PRIV_VEC_ADDR]], i32** [[PRIV_SIVAR_ADDR]])

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
// CHECK-DAG: [[PRIV_SIVAR]]

// CHECK: ret

// CHECK: define internal void [[MAIN_DUP]]([[KMP_TASK_MAIN_TY]]* noundef %0, [[KMP_TASK_MAIN_TY]]* noundef %1, i32 noundef %2)
// CHECK: getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* %{{.+}}, i32 0, i32 1
// CHECK: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* %{{.+}}, i32 0, i32 0
// CHECK: getelementptr inbounds [2 x [[S_DOUBLE_TY]]], [2 x [[S_DOUBLE_TY]]]* %{{.+}}, i32 0, i32 0
// CHECK: getelementptr [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i64 2
// CHECK: br i1 %

// CHECK: phi [[S_DOUBLE_TY]]*
// CHECK: call {{.*}} [[S_DOUBLE_TY_COPY_CONSTR]]([[S_DOUBLE_TY]]*
// CHECK: getelementptr [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i32 1
// CHECK: icmp eq [[S_DOUBLE_TY]]* %
// CHECK: br i1 %

// CHECK: getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* %{{.+}}, i32 0, i32 1
// CHECK: call {{.*}} [[S_DOUBLE_TY_COPY_CONSTR]]([[S_DOUBLE_TY]]*
// CHECK: ret void

// CHECK: define internal noundef i32 [[DESTRUCTORS]](i32 noundef %{{.+}}, [[KMP_TASK_MAIN_TY]]* noalias noundef %{{.+}})
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_MAIN_TY]], [[KMP_TASK_MAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_MAIN_TY]], [[PRIVATES_MAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK: call void @_ZN1SIdED1Ev([[S_DOUBLE_TY]]* {{[^,]*}} [[PRIVATE_VAR_REF]])
// CHECK: getelementptr inbounds [2 x [[S_DOUBLE_TY]]], [2 x [[S_DOUBLE_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_DOUBLE_TY]], [[S_DOUBLE_TY]]* %{{.+}}, i{{.+}} -1
// CHECK: call void @_ZN1SIdED1Ev([[S_DOUBLE_TY]]* {{[^,]*}} [[PRIVATE_S_ARR_ELEM_REF]])
// CHECK: icmp eq
// CHECK: br i1
// CHECK: ret i32

// CHECK: alloca [[S_INT_TY]],
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: [[T_VAR_ADDR:%.+]] = alloca i32, align 128
// CHECK: [[VEC_ADDR:%.+]] = alloca [2 x i32],
// CHECK: [[S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]],
// CHECK: [[VAR_ADDR:%.+]] = alloca [[S_INT_TY]],

// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR:@.+]]([[S_INT_TY]]* {{[^,]*}} [[TEST]],

// Store original variables in capture struct.
// CHECK: [[S_ARR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: store [2 x [[S_INT_TY]]]* %{{.+}}, [2 x [[S_INT_TY]]]** [[S_ARR_REF]],
// CHECK: [[VAR_REF:%.+]] = getelementptr inbounds [[CAP_TMAIN_TY]], [[CAP_TMAIN_TY]]* %{{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: store [[S_INT_TY]]* %{{.+}}, [[S_INT_TY]]** [[VAR_REF]],

// Allocate task.
// Returns struct kmp_task_t {
//         [[KMP_TASK_T_TY]] task_data;
//         [[KMP_TASK_TMAIN_TY]] privates;
//       };
// CHECK: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 9, i64 256, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_TMAIN_TY]]*)* [[TASK_ENTRY:@[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[RES_KMP_TASK:%.+]] = bitcast i8* [[RES]] to [[KMP_TASK_TMAIN_TY]]*

// Fill kmp_task_t->shareds by copying from original capture argument.
// CHECK: [[TASK:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF_ADDR:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_REF_ADDR]],
// CHECK: [[CAPTURES_ADDR:%.+]] = bitcast [[CAP_TMAIN_TY]]* %{{.+}} to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[SHAREDS_REF]], i8* align 8 [[CAPTURES_ADDR]], i64 16, i1 false)

// Initialize kmp_task_t->privates with default values (no init for simple types, default constructors for classes).
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK]], i{{[0-9]+}} 0, i{{[0-9]+}} 2

// t_var;
// CHECK: [[PRIVATE_T_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[T_VAR:%.+]] = load i32, i32* %{{.+}}, align 128
// CHECK: store i32 [[T_VAR]], i32* [[PRIVATE_T_VAR_REF]], align 128

// vec;
// CHECK: [[PRIVATE_VEC_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 1
// CHECK-NEXT: bitcast [2 x i32]* [[PRIVATE_VEC_REF]] to i8*
// CHECK-NEXT: bitcast [2 x i32]* %{{.+}} to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(

// Constructors for s_arr and var.
// a_arr;
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: bitcast [2 x [[S_INT_TY]]]* %{{.+}} to [[S_INT_TY]]*
// CHECK: getelementptr [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: call void [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]* {{[^,]*}} [[S_ARR_CUR:%[^,]+]],
// CHECK: getelementptr [[S_INT_TY]], [[S_INT_TY]]* [[S_ARR_CUR]], i{{.+}} 1
// CHECK: icmp eq
// CHECK: br i1

// var;
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK-NEXT: call void [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]* {{[^,]*}} [[PRIVATE_VAR_REF]],

// Provide pointer to destructor function, which will destroy private variables at the end of the task.
// CHECK: [[DESTRUCTORS_REF:%.+]] = getelementptr inbounds [[KMP_TASK_T_TY]], [[KMP_TASK_T_TY]]* [[TASK]], i{{.+}} 0, i{{.+}} 3
// CHECK: [[DESTRUCTORS_PTR:%.+]] = bitcast %union{{.+}}* [[DESTRUCTORS_REF]] to i32 (i32, i8*)**
// CHECK: store i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_TMAIN_TY]]*)* [[DESTRUCTORS:@.+]] to i32 (i32, i8*)*), i32 (i32, i8*)** [[DESTRUCTORS_PTR]],

// Start task.
// CHECK: call void @__kmpc_taskloop(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]], i32 1, i64* %{{.+}}, i64* %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, i8* bitcast (void ([[KMP_TASK_TMAIN_TY]]*, [[KMP_TASK_TMAIN_TY]]*, i32)* [[TMAIN_DUP:@.+]] to i8*))

// CHECK: define internal void [[PRIVATES_MAP_FN:@.+]]([[PRIVATES_TMAIN_TY]]* noalias noundef %{{.+}}, i32** noalias noundef %{{.+}}, [2 x i32]** noalias noundef %{{.+}}, [2 x [[S_INT_TY]]]** noalias noundef %{{.+}}, [[S_INT_TY]]** noalias noundef %{{.+}})
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

// CHECK: define internal noundef i32 [[TASK_ENTRY]](i32 noundef %0, [[KMP_TASK_TMAIN_TY]]* noalias noundef %1)
// CHECK: alloca i32*,
// CHECK-DAG: [[PRIV_T_VAR_ADDR:%.+]] = alloca i32*,
// CHECK-DAG: [[PRIV_VEC_ADDR:%.+]] = alloca [2 x i32]*,
// CHECK-DAG: [[PRIV_S_ARR_ADDR:%.+]] = alloca [2 x [[S_INT_TY]]]*,
// CHECK-DAG: [[PRIV_VAR_ADDR:%.+]] = alloca [[S_INT_TY]]*,
// CHECK: store void (i8*, ...)* bitcast (void ([[PRIVATES_TMAIN_TY]]*, i32**, [2 x i32]**, [2 x [[S_INT_TY]]]**, [[S_INT_TY]]**)* [[PRIVATES_MAP_FN]] to void (i8*, ...)*), void (i8*, ...)** [[MAP_FN_ADDR:%.+]],
// CHECK: [[MAP_FN:%.+]] = load void (i8*, ...)*, void (i8*, ...)** [[MAP_FN_ADDR]],
// CHECK: [[FN:%.+]] = bitcast void (i8*, ...)* [[MAP_FN]] to void (i8*,
// CHECK: call void [[FN]](i8* %{{.+}}, i32** [[PRIV_T_VAR_ADDR]], [2 x i32]** [[PRIV_VEC_ADDR]], [2 x [[S_INT_TY]]]** [[PRIV_S_ARR_ADDR]], [[S_INT_TY]]** [[PRIV_VAR_ADDR]])
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

// CHECK: define internal void [[TMAIN_DUP]]([[KMP_TASK_TMAIN_TY]]* noundef %0, [[KMP_TASK_TMAIN_TY]]* noundef %1, i32 noundef %2)
// CHECK: getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* %{{.+}}, i32 0, i32 2
// CHECK: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* %{{.+}}, i32 0, i32 2
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* %{{.+}}, i32 0, i32 0
// CHECK: getelementptr [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i64 2
// CHECK: br i1 %

// CHECK: phi [[S_INT_TY]]*
// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]*
// CHECK: getelementptr [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i32 1
// CHECK: icmp eq [[S_INT_TY]]* %
// CHECK: br i1 %

// CHECK: getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* %{{.+}}, i32 0, i32 3
// CHECK: call {{.*}} [[S_INT_TY_COPY_CONSTR]]([[S_INT_TY]]*
// CHECK: ret void

// CHECK: define internal noundef i32 [[DESTRUCTORS]](i32 noundef %0, [[KMP_TASK_TMAIN_TY]]* noalias noundef %1)
// CHECK: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_TMAIN_TY]], [[KMP_TASK_TMAIN_TY]]* [[RES_KMP_TASK:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[PRIVATE_S_ARR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[PRIVATE_VAR_REF:%.+]] = getelementptr inbounds [[PRIVATES_TMAIN_TY]], [[PRIVATES_TMAIN_TY]]* [[PRIVATES]], i{{.+}} 0, i{{.+}} 3
// CHECK: call void @_ZN1SIiED1Ev([[S_INT_TY]]* {{[^,]*}} [[PRIVATE_VAR_REF]])
// CHECK: getelementptr inbounds [2 x [[S_INT_TY]]], [2 x [[S_INT_TY]]]* [[PRIVATE_S_ARR_REF]], i{{.+}} 0, i{{.+}} 0
// CHECK: getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} 2
// CHECK: [[PRIVATE_S_ARR_ELEM_REF:%.+]] = getelementptr inbounds [[S_INT_TY]], [[S_INT_TY]]* %{{.+}}, i{{.+}} -1
// CHECK: call void @_ZN1SIiED1Ev([[S_INT_TY]]* {{[^,]*}} [[PRIVATE_S_ARR_ELEM_REF]])
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

void array_func(int n, float a[n], St s[2]) {
// ARRAY: call i8* @__kmpc_omp_task_alloc(
// ARRAY: call void @__kmpc_taskloop(
// ARRAY: store float** %{{.+}}, float*** %{{.+}},
// ARRAY: store %struct.St** %{{.+}}, %struct.St*** %{{.+}},
#pragma omp parallel master taskloop simd firstprivate(a, s)
  for (int i = 0; i < 10; ++i)
    ;
}
#endif

