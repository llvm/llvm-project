// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// REQUIRES: x86-registered-target
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

volatile int g __attribute__((aligned(128))) = 1212;

template <class T>
struct S {
  T f;
  S(T a) : f(a + g) {}
  S() : f(g) {}
  operator T() { return T(); }
  S &operator&(const S &) { return *this; }
  ~S() {}
};

struct SS {
  int a;
  int b : 4;
  int &c;
  SS(int &d) : a(0), b(0), c(d) {
#pragma omp parallel reduction(+: a, b, c)
#ifdef LAMBDA
    [&]() {
      ++this->a, --b, (this)->c /= 1;
#pragma omp parallel reduction(&: a, b, c)
      ++(this)->a, --b, this->c /= 1;
    }();
#elif defined(BLOCKS)
    ^{
      ++a;
      --this->b;
      (this)->c /= 1;
#pragma omp parallel reduction(-: a, b, c)
      ++(this)->a, --b, this->c /= 1;
    }();
#else
    ++this->a, --b, c /= 1;
#endif
  }
};

template<typename T>
struct SST {
  T a;
  SST() : a(T()) {
#pragma omp parallel reduction(*: a)
#ifdef LAMBDA
    [&]() {
      [&]() {
        ++this->a;
#pragma omp parallel reduction(&& :a)
        ++(this)->a;
      }();
    }();
#elif defined(BLOCKS)
    ^{
      ^{
        ++a;
#pragma omp parallel reduction(|: a)
        ++(this)->a;
      }();
    }();
#else
    ++(this)->a;
#endif
  }
};

// CHECK: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// LAMBDA: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// BLOCKS: [[SS_TY:%.+]] = type { i{{[0-9]+}}, i8
// CHECK-DAG: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK-DAG: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK-DAG: [[REDUCTION_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 18, i32 0, i32 0, i8*
// CHECK-DAG: [[REDUCTION_LOCK:@.+]] = common global [8 x i32] zeroinitializer

template <typename T>
T tmain() {
  T t;
  S<T> test;
  SST<T> sst;
  T t_var __attribute__((aligned(128))) = T(), t_var1 __attribute__((aligned(128)));
  T vec[] = {1, 2};
  S<T> s_arr[]  = {1, 2};
  S<T> var __attribute__((aligned(128))) (3), var1 __attribute__((aligned(128)));
#pragma omp parallel reduction(+:t_var) reduction(&:var) reduction(&& : var1) reduction(min: t_var1)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int sivar;
int main() {
  SS ss(sivar);
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: alloca [[SS_TY]],
  // LAMBDA: alloca [[CAP_TY:%.+]],
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@[^(]+]]([[CAP_TY]]*
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* [[G]])
#pragma omp parallel reduction(+:g)
  {
    // LAMBDA: define {{.+}} @{{.+}}([[SS_TY]]*
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
    // LAMBDA: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
    // LAMBDA: store i8
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
    // LAMBDA-NOT: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
    // LAMBDA: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
    // LAMBDA: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*, i32*, i32*, i32*)* [[SS_MICROTASK:@.+]] to void
    // LAMBDA: [[B_REF:%.+]] = getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
    // LAMBDA: store i8 %{{.+}}, i8* [[B_REF]],
    // LAMBDA: ret

    // LAMBDA: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}}, i32* {{.+}}, i32* {{.+}}, i32* {{.+}})
    // LAMBDA-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %
    // LAMBDA: call{{.*}} void
    // LAMBDA: ret void

    // LAMBDA: define internal void @{{.+}}(i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]*
    // LAMBDA: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: store i{{[0-9]+}} -1, i{{[0-9]+}}* [[A_PRIV]],
    // LAMBDA: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
    // LAMBDA: store i{{[0-9]+}} -1, i{{[0-9]+}}* [[B_PRIV]],
    // LAMBDA: store i{{[0-9]+}} -1, i{{[0-9]+}}* [[C_PRIV]],
    // LAMBDA: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
    // LAMBDA: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
    // LAMBDA-NEXT: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_PRIV]],
    // LAMBDA-NEXT: [[INC:%.+]] = add nsw i{{[0-9]+}} [[A_VAL]], 1
    // LAMBDA-NEXT: store i{{[0-9]+}} [[INC]], i{{[0-9]+}}* [[A_PRIV]],
    // LAMBDA-NEXT: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_PRIV]],
    // LAMBDA-NEXT: [[DEC:%.+]] = add nsw i{{[0-9]+}} [[B_VAL]], -1
    // LAMBDA-NEXT: store i{{[0-9]+}} [[DEC]], i{{[0-9]+}}* [[B_PRIV]],
    // LAMBDA-NEXT: [[C_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFC]],
    // LAMBDA-NEXT: [[C_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_PRIV]],
    // LAMBDA-NEXT: [[DIV:%.+]] = sdiv i{{[0-9]+}} [[C_VAL]], 1
    // LAMBDA-NEXT: store i{{[0-9]+}} [[DIV]], i{{[0-9]+}}* [[C_PRIV]],
    // LAMBDA: call i32 @__kmpc_reduce_nowait(
    // LAMBDA: ret void

    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}})
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},

    // Reduction list for runtime.
    // LAMBDA: [[RED_LIST:%.+]] = alloca [1 x i8*],

    // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_REF_ADDR:%.+]]
    // LAMBDA: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[G_PRIVATE_ADDR]], align 128
    g = 1;
    // LAMBDA: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]], align 128
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store i{{[0-9]+}}* [[G_PRIVATE_ADDR]], i{{[0-9]+}}** [[G_PRIVATE_ADDR_REF]]
    // LAMBDA: call void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])

    // LAMBDA: [[G_PRIV_REF:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[RED_LIST]], i64 0, i64 0
    // LAMBDA: [[BITCAST:%.+]] = bitcast i32* [[G_PRIVATE_ADDR]] to i8*
    // LAMBDA: store i8* [[BITCAST]], i8** [[G_PRIV_REF]],
    // LAMBDA: call i32 @__kmpc_reduce_nowait(
    // LAMBDA: switch i32 %{{.+}}, label %[[REDUCTION_DONE:.+]] [
    // LAMBDA: i32 1, label %[[CASE1:.+]]
    // LAMBDA: i32 2, label %[[CASE2:.+]]
    // LAMBDA: [[CASE1]]
    // LAMBDA: [[G_VAL:%.+]] = load i32, i32* [[G_REF]]
    // LAMBDA: [[G_PRIV_VAL:%.+]] = load i32, i32* [[G_PRIVATE_ADDR]]
    // LAMBDA: [[ADD:%.+]] = add nsw i32 [[G_VAL]], [[G_PRIV_VAL]]
    // LAMBDA: store i32 [[ADD]], i32* [[G_REF]]
    // LAMBDA: call void @__kmpc_end_reduce_nowait(
    // LAMBDA: br label %[[REDUCTION_DONE]]
    // LAMBDA: [[CASE2]]
    // LAMBDA: [[G_PRIV_VAL:%.+]] = load i32, i32* [[G_PRIVATE_ADDR]]
    // LAMBDA: atomicrmw add i32* [[G_REF]], i32 [[G_PRIV_VAL]] monotonic
    // LAMBDA: br label %[[REDUCTION_DONE]]
    // LAMBDA: [[REDUCTION_DONE]]
    // LAMBDA: ret void
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_REF]]
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // BLOCKS-LABEL: @main
  // BLOCKS: call
  // BLOCKS: call void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS: call void {{.+}} @__kmpc_fork_call({{.+}}, i32 1, {{.+}}* [[OMP_REGION:@.+]] to {{.+}}, i32* [[G]])
#pragma omp parallel reduction(-:g)
  {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* dereferenceable(4) %{{.+}})
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},

    // Reduction list for runtime.
    // BLOCKS: [[RED_LIST:%.+]] = alloca [1 x i8*],

    // BLOCKS: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_REF_ADDR:%.+]]
    // BLOCKS: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[G_PRIVATE_ADDR]], align 128
    g = 1;
    // BLOCKS: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]], align 128
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: i{{[0-9]+}}* [[G_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: call void {{%.+}}(i8

    // BLOCKS: [[G_PRIV_REF:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[RED_LIST]], i64 0, i64 0
    // BLOCKS: [[BITCAST:%.+]] = bitcast i32* [[G_PRIVATE_ADDR]] to i8*
    // BLOCKS: store i8* [[BITCAST]], i8** [[G_PRIV_REF]],
    // BLOCKS: call i32 @__kmpc_reduce_nowait(
    // BLOCKS: switch i32 %{{.+}}, label %[[REDUCTION_DONE:.+]] [
    // BLOCKS: i32 1, label %[[CASE1:.+]]
    // BLOCKS: i32 2, label %[[CASE2:.+]]
    // BLOCKS: [[CASE1]]
    // BLOCKS: [[G_VAL:%.+]] = load i32, i32* [[G_REF]]
    // BLOCKS: [[G_PRIV_VAL:%.+]] = load i32, i32* [[G_PRIVATE_ADDR]]
    // BLOCKS: [[ADD:%.+]] = add nsw i32 [[G_VAL]], [[G_PRIV_VAL]]
    // BLOCKS: store i32 [[ADD]], i32* [[G_REF]]
    // BLOCKS: call void @__kmpc_end_reduce_nowait(
    // BLOCKS: br label %[[REDUCTION_DONE]]
    // BLOCKS: [[CASE2]]
    // BLOCKS: [[G_PRIV_VAL:%.+]] = load i32, i32* [[G_PRIVATE_ADDR]]
    // BLOCKS: atomicrmw add i32* [[G_REF]], i32 [[G_PRIV_VAL]] monotonic
    // BLOCKS: br label %[[REDUCTION_DONE]]
    // BLOCKS: [[REDUCTION_DONE]]
    // BLOCKS: ret void
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
// BLOCKS: define {{.+}} @{{.+}}([[SS_TY]]*
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// BLOCKS: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// BLOCKS: store i8
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// BLOCKS-NOT: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// BLOCKS: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// BLOCKS: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*, i32*, i32*, i32*)* [[SS_MICROTASK:@.+]] to void
// BLOCKS: [[B_REF:%.+]] = getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
// BLOCKS: store i8 %{{.+}}, i8* [[B_REF]],
// BLOCKS: ret

// BLOCKS: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}}, i32* {{.+}}, i32* {{.+}}, i32* {{.+}})
// BLOCKS-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %
// BLOCKS: call{{.*}} void
// BLOCKS: ret void

// BLOCKS: define internal void @{{.+}}(i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}}, i32* {{.+}}, i32* {{.+}}, i32* {{.+}})
// BLOCKS: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[A_PRIV]],
// BLOCKS: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
// BLOCKS: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[B_PRIV]],
// BLOCKS: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[C_PRIV]],
// BLOCKS: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
// BLOCKS: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
// BLOCKS-NEXT: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_PRIV]],
// BLOCKS-NEXT: [[INC:%.+]] = add nsw i{{[0-9]+}} [[A_VAL]], 1
// BLOCKS-NEXT: store i{{[0-9]+}} [[INC]], i{{[0-9]+}}* [[A_PRIV]],
// BLOCKS-NEXT: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_PRIV]],
// BLOCKS-NEXT: [[DEC:%.+]] = add nsw i{{[0-9]+}} [[B_VAL]], -1
// BLOCKS-NEXT: store i{{[0-9]+}} [[DEC]], i{{[0-9]+}}* [[B_PRIV]],
// BLOCKS-NEXT: [[C_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFC]],
// BLOCKS-NEXT: [[C_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_PRIV]],
// BLOCKS-NEXT: [[DIV:%.+]] = sdiv i{{[0-9]+}} [[C_VAL]], 1
// BLOCKS-NEXT: store i{{[0-9]+}} [[DIV]], i{{[0-9]+}}* [[C_PRIV]],
// BLOCKS: call i32 @__kmpc_reduce_nowait(
// BLOCKS: ret void
#else
  S<float> test;
  float t_var = 0, t_var1;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> var(3), var1;
  float _Complex cf;
#pragma omp parallel reduction(+:t_var) reduction(&:var) reduction(&& : var1) reduction(min: t_var1)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  if (var1)
#pragma omp parallel reduction(+ : t_var) reduction(& : var) reduction(&& : var1) reduction(min : t_var1)
    while (1) {
      vec[0] = t_var;
      s_arr[0] = var;
    }
#pragma omp parallel reduction(+ : cf)
    ;
  return tmain<int>();
#endif
}

// CHECK: define {{.*}}i{{[0-9]+}} @main()
// CHECK: [[TEST:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 6, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i32]*, float*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]*, float*)* [[MAIN_MICROTASK:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 6, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i32]*, float*, [2 x [[S_FLOAT_TY]]]*, [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]*, float*)* [[MAIN_MICROTASK1:@.+]] to void
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, { float, float }*)* [[MAIN_MICROTASK2:@.+]] to void
// CHECK: = call {{.*}}i{{.+}} [[TMAIN_INT:@.+]]()
// CHECK: call {{.*}} [[S_FLOAT_TY_DESTR:@.+]]([[S_FLOAT_TY]]*
// CHECK: ret
//
// CHECK: define internal void [[MAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca float,
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[VAR1_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[T_VAR1_PRIV:%.+]] = alloca float,

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [4 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[T_VAR_REF:%.+]] = load float*, float** %
// CHECK: [[VAR_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %
// CHECK: [[VAR1_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %
// CHECK: [[T_VAR1_REF:%.+]] = load float*, float** %

// For + reduction operation initial value of private variable is 0.
// CHECK: store float 0.0{{.+}}, float* [[T_VAR_PRIV]],

// For & reduction operation initial value of private variable is ones in all bits.
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[VAR_PRIV]])

// For && reduction operation initial value of private variable is 1.0.
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[VAR1_PRIV]])

// For min reduction operation initial value of private variable is largest repesentable value.
// CHECK: store float 0x47EFFFFFE0000000, float* [[T_VAR1_PRIV]],

// Skip checks for internal operations.

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

// res = __kmpc_reduce_nowait(<loc>, <gtid>, <n>, sizeof(RedList), RedList, reduce_func, &<lock>);

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
// CHECK: [[BITCAST:%.+]] = bitcast [4 x i8*]* [[RED_LIST]] to i8*
// CHECK: [[RES:%.+]] = call i32 @__kmpc_reduce_nowait(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], i32 4, i64 32, i8* [[BITCAST]], void (i8*, i8*)* [[REDUCTION_FUNC:@.+]], [8 x i32]* [[REDUCTION_LOCK]])

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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

// t_var1 = min(t_var1, t_var1_reduction);
// CHECK: [[T_VAR1_VAL:%.+]] = load float, float* [[T_VAR1_REF]],
// CHECK: [[T_VAR1_PRIV_VAL:%.+]] = load float, float* [[T_VAR1_PRIV]],
// CHECK: [[CMP:%.+]] = fcmp olt float [[T_VAR1_VAL]], [[T_VAR1_PRIV_VAL]]
// CHECK: br i1 [[CMP]]
// CHECK: [[UP:%.+]] = phi float
// CHECK: store float [[UP]], float* [[T_VAR1_REF]],

// __kmpc_end_reduce_nowait(<loc>, <gtid>, &<lock>);
// CHECK: call void @__kmpc_end_reduce_nowait(%{{.+}}* [[REDUCTION_LOC]], i32 [[GTID]], [8 x i32]* [[REDUCTION_LOCK]])

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
// CHECK: [[UP_INT:%.+]] = load i32
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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)
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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)
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
// CHECK: [[UP:%.+]] = phi float
// CHECK: [[UP_INT:%.+]] = load i32
// CHECK: [[T_VAR1_REF_INT:%.+]] = bitcast float* [[T_VAR1_REF]] to i32*
// CHECK: [[RES:%.+]] = cmpxchg i32* [[T_VAR1_REF_INT]], i32 [[ORIG_OLD_INT]], i32 [[UP_INT]] monotonic monotonic
// CHECK: [[OLD2:%.+]] = extractvalue { i32, i1 } [[RES]], 0
// CHECK: [[SUCCESS_FAIL:%.+]] = extractvalue { i32, i1 } [[RES]], 1
// CHECK: br i1 [[SUCCESS_FAIL]], label %[[ATOMIC_DONE:.+]], label %[[CONT]]
// CHECK: [[ATOMIC_DONE]]

// break;
// CHECK: br label %[[RED_DONE]]
// CHECK: [[RED_DONE]]

// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call {{.*}} [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

// t_var1_lhs = min(t_var1_lhs, t_var1_rhs);
// CHECK: [[T_VAR1_LHS_VAL:%.+]] = load float, float* [[T_VAR1_LHS]],
// CHECK: [[T_VAR1_RHS_VAL:%.+]] = load float, float* [[T_VAR1_RHS]],
// CHECK: [[CMP:%.+]] = fcmp olt float [[T_VAR1_LHS_VAL]], [[T_VAR1_RHS_VAL]]
// CHECK: br i1 [[CMP]]
// CHECK: [[UP:%.+]] = phi float
// CHECK: store float [[UP]], float* [[T_VAR1_LHS]],
// CHECK: ret void

// CHECK: define internal void [[MAIN_MICROTASK1]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca float,
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[VAR1_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[T_VAR1_PRIV:%.+]] = alloca float,

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[T_VAR_REF:%.+]] = load float*, float** %
// CHECK: [[VAR_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %
// CHECK: [[VAR1_REF:%.+]] = load [[S_FLOAT_TY]]*, [[S_FLOAT_TY]]** %
// CHECK: [[T_VAR1_REF:%.+]] = load float*, float** %

// For + reduction operation initial value of private variable is 0.
// CHECK: store float 0.0{{.+}}, float* [[T_VAR_PRIV]],

// For & reduction operation initial value of private variable is ones in all bits.
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[VAR_PRIV]])

// For && reduction operation initial value of private variable is 1.0.
// CHECK: call {{.*}} [[S_FLOAT_TY_CONSTR:@.+]]([[S_FLOAT_TY]]* [[VAR1_PRIV]])

// For min reduction operation initial value of private variable is largest repesentable value.
// CHECK: store float 0x47EFFFFFE0000000, float* [[T_VAR1_PRIV]],

// CHECK-NOT: call i32 @__kmpc_reduce

// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 6, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [2 x i32]*, i32*, [2 x [[S_INT_TY]]]*, [[S_INT_TY]]*, [[S_INT_TY]]*, i32*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call {{.*}} [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define {{.+}} @{{.+}}([[SS_TY]]*
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// CHECK: store i8
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 0
// CHECK-NOT: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 1
// CHECK: getelementptr inbounds [[SS_TY]], [[SS_TY]]* %{{.+}}, i32 0, i32 2
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 4, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*, i{{[0-9]+}}*, i{{[0-9]+}}*, i{{[0-9]+}}*)* [[SS_MICROTASK:@.+]] to void
// CHECK: [[B_REF:%.+]] = getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %{{.*}}, i32 0, i32 1
// CHECK: store i8 %{{.+}}, i8* [[B_REF]],
// CHECK: ret

// CHECK: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]*
// CHECK: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[A_PRIV]],
// CHECK: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[B_PRIV]],
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[C_PRIV]],
// CHECK: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
// CHECK: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
// CHECK-NEXT: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_PRIV]],
// CHECK-NEXT: [[INC:%.+]] = add nsw i{{[0-9]+}} [[A_VAL]], 1
// CHECK-NEXT: store i{{[0-9]+}} [[INC]], i{{[0-9]+}}* [[A_PRIV]],
// CHECK-NEXT: [[B_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[B_PRIV]],
// CHECK-NEXT: [[DEC:%.+]] = add nsw i{{[0-9]+}} [[B_VAL]], -1
// CHECK-NEXT: store i{{[0-9]+}} [[DEC]], i{{[0-9]+}}* [[B_PRIV]],
// CHECK-NEXT: [[C_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFC]],
// CHECK-NEXT: [[C_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C_PRIV]],
// CHECK-NEXT: [[DIV:%.+]] = sdiv i{{[0-9]+}} [[C_VAL]], 1
// CHECK-NEXT: store i{{[0-9]+}} [[DIV]], i{{[0-9]+}}* [[C_PRIV]],
// CHECK: call i32 @__kmpc_reduce_nowait(
// CHECK: ret void

// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}},
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}}, align 128
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]], align 128
// CHECK: [[VAR1_PRIV:%.+]] = alloca [[S_INT_TY]], align 128
// CHECK: [[T_VAR1_PRIV:%.+]] = alloca i{{[0-9]+}}, align 128

// Reduction list for runtime.
// CHECK: [[RED_LIST:%.+]] = alloca [4 x i8*],

// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_ADDR:%.+]],

// CHECK: [[T_VAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %
// CHECK: [[VAR_REF:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** %
// CHECK: [[VAR1_REF:%.+]] = load [[S_INT_TY]]*, [[S_INT_TY]]** %
// CHECK: [[T_VAR1_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** %

// For + reduction operation initial value of private variable is 0.
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* [[T_VAR_PRIV]],

// For & reduction operation initial value of private variable is ones in all bits.
// CHECK: call {{.*}} [[S_INT_TY_CONSTR:@.+]]([[S_INT_TY]]* [[VAR_PRIV]])

// For && reduction operation initial value of private variable is 1.0.
// CHECK: call {{.*}} [[S_INT_TY_CONSTR:@.+]]([[S_INT_TY]]* [[VAR1_PRIV]])

// For min reduction operation initial value of private variable is largest repesentable value.
// CHECK: store i{{[0-9]+}} 2147483647, i{{[0-9]+}}* [[T_VAR1_PRIV]],

// Skip checks for internal operations.

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

// CHECK: [[GTID_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_ADDR]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_REF]]
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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

// var1 = var1.operator &&(var1_reduction);
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_REF]])
// CHECK: [[VAR1_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: br i1 [[VAR1_BOOL]], label %[[TRUE:.+]], label %[[END2:.+]]
// CHECK: [[TRUE]]
// CHECK: [[TO_INT:%.+]] = call i{{[0-9]+}} @{{.+}}([[S_INT_TY]]* [[VAR1_PRIV]])
// CHECK: [[VAR1_REDUCTION_BOOL:%.+]] = icmp ne i{{[0-9]+}} [[TO_INT]], 0
// CHECK: [[END2]]
// CHECK: [[COND_LVALUE:%.+]] = phi i1 [ false, %{{.+}} ], [ [[VAR1_REDUCTION_BOOL]], %[[TRUE]] ]
// CHECK: [[CONV:%.+]] = zext i1 [[COND_LVALUE]] to i32
// CHECK:  call void @{{.+}}([[S_INT_TY]]* [[COND_LVALUE:%.+]], i32 [[CONV]])
// CHECK: [[BC1:%.+]] = bitcast [[S_INT_TY]]* [[VAR1_REF]] to i8*
// CHECK: [[BC2:%.+]] = bitcast [[S_INT_TY]]* [[COND_LVALUE]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)
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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)
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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

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
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[BC1]], i8* [[BC2]], i64 4, i32 4, i1 false)

// t_var1_lhs = min(t_var1_lhs, t_var1_rhs);
// CHECK: [[T_VAR1_LHS_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR1_LHS]],
// CHECK: [[T_VAR1_RHS_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[T_VAR1_RHS]],
// CHECK: [[CMP:%.+]] = icmp slt i{{[0-9]+}} [[T_VAR1_LHS_VAL]], [[T_VAR1_RHS_VAL]]
// CHECK: br i1 [[CMP]]
// CHECK: [[UP:%.+]] = phi i32
// CHECK: store i{{[0-9]+}} [[UP]], i{{[0-9]+}}* [[T_VAR1_LHS]],
// CHECK: ret void

#endif

