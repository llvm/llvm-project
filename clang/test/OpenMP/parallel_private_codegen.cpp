// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++11 -DLAMBDA -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=LAMBDA %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -fblocks -DBLOCKS -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck -check-prefix=BLOCKS %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
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

volatile int g __attribute__((aligned(128))) = 1212;

struct SS {
  int a;
  int b : 4;
  int &c;
  SS(int &d) : a(0), b(0), c(d) {
#pragma omp parallel private(a, b, c)
#ifdef LAMBDA
    [&]() {
      ++this->a, --b, (this)->c /= 1;
#pragma omp parallel private(a, b, c)
      ++(this)->a, --b, this->c /= 1;
    }();
#elif defined(BLOCKS)
    ^{
      ++a;
      --this->b;
      (this)->c /= 1;
#pragma omp parallel private(a, b, c)
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
#pragma omp parallel private(a)
#ifdef LAMBDA
    [&]() {
      [&]() {
        ++this->a;
#pragma omp parallel private(a)
        ++(this)->a;
      }();
    }();
#elif defined(BLOCKS)
    ^{
      ^{
        ++a;
#pragma omp parallel private(a)
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
// CHECK: [[S_FLOAT_TY:%.+]] = type { float }
// CHECK: [[S_INT_TY:%.+]] = type { i{{[0-9]+}} }
// CHECK: [[SST_TY:%.+]] = type { i{{[0-9]+}} }
template <typename T>
T tmain() {
  S<T> test;
  SST<T> sst;
  T t_var __attribute__((aligned(128))) = T();
  T vec[] __attribute__((aligned(128))) = {1, 2};
  S<T> s_arr[] __attribute__((aligned(128))) = {1, 2};
  S<T> var __attribute__((aligned(128))) (3);
#pragma omp parallel private(t_var, vec, s_arr, var)
  {
    vec[0] = t_var;
    s_arr[0] = var;
  }
  return T();
}

int main() {
  static int sivar;
  SS ss(sivar);
#ifdef LAMBDA
  // LAMBDA: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // LAMBDA-LABEL: @main
  // LAMBDA: call{{.*}} void
  // LAMBDA: call{{.*}} void [[OUTER_LAMBDA:@[^(]+]](
  [&]() {
  // LAMBDA: define{{.*}} internal{{.*}} void [[OUTER_LAMBDA]](
  // LAMBDA-NOT: = getelementptr inbounds %{{.+}},
  // LAMBDA: call{{.*}} void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel private(g, sivar)
  {
    // LAMBDA: define {{.+}} @{{.+}}([[SS_TY]]*
    // LAMBDA: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
    // LAMBDA: store i8
    // LAMBDA: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
    // LAMBDA: ret

    // LAMBDA: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
    // LAMBDA-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %
    // LAMBDA: call{{.*}} void
    // LAMBDA: ret void

    // LAMBDA: define internal void @{{.+}}(i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
    // LAMBDA: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
    // LAMBDA: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
    // LAMBDA-NEXT: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
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
    // LAMBDA-NEXT: ret void

    // LAMBDA: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // LAMBDA: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // LAMBDA: [[SIVAR_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    g = 1;
    sivar = 2;
    // LAMBDA: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]],
    // LAMBDA: [[G_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
    // LAMBDA: store i{{[0-9]+}}* [[G_PRIVATE_ADDR]], i{{[0-9]+}}** [[G_PRIVATE_ADDR_REF]]

    // LAMBDA: [[SIVAR_PRIVATE_ADDR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
    // LAMBDA: store i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]], i{{[0-9]+}}** [[SIVAR_PRIVATE_ADDR_REF]]

    // LAMBDA: call{{.*}} void [[INNER_LAMBDA:@.+]](%{{.+}}* [[ARG]])
    [&]() {
      // LAMBDA: define {{.+}} void [[INNER_LAMBDA]](%{{.+}}* [[ARG_PTR:%.+]])
      // LAMBDA: store %{{.+}}* [[ARG_PTR]], %{{.+}}** [[ARG_PTR_REF:%.+]],
      g = 2;
      sivar = 4;
      // LAMBDA: [[ARG_PTR:%.+]] = load %{{.+}}*, %{{.+}}** [[ARG_PTR_REF]]
      // LAMBDA: [[G_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
      // LAMBDA: [[G_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[G_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 2, i{{[0-9]+}}* [[G_REF]]
      // LAMBDA: [[SIVAR_PTR_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[ARG_PTR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
      // LAMBDA: [[SIVAR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[SIVAR_PTR_REF]]
      // LAMBDA: store i{{[0-9]+}} 4, i{{[0-9]+}}* [[SIVAR_REF]]
    }();
  }
  }();
  return 0;
#elif defined(BLOCKS)
  // BLOCKS: [[G:@.+]] = global i{{[0-9]+}} 1212,
  // BLOCKS-LABEL: @main
  // BLOCKS: call
  // BLOCKS: call{{.*}} void {{%.+}}(i8
  ^{
  // BLOCKS: define{{.*}} internal{{.*}} void {{.+}}(i8*
  // BLOCKS-NOT: = getelementptr inbounds %{{.+}},
  // BLOCKS: call{{.*}} void {{.+}} @__kmpc_fork_call({{.+}}, i32 0, {{.+}}* [[OMP_REGION:@.+]] to {{.+}})
#pragma omp parallel private(g, sivar)
  {
    // BLOCKS: define{{.*}} internal{{.*}} void [[OMP_REGION]](i32* noalias %{{.+}}, i32* noalias %{{.+}})
    // BLOCKS: [[G_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    // BLOCKS: [[SIVAR_PRIVATE_ADDR:%.+]] = alloca i{{[0-9]+}},
    g = 1;
    sivar = 20;
    // BLOCKS: store i{{[0-9]+}} 1, i{{[0-9]+}}* [[G_PRIVATE_ADDR]],
    // BLOCKS: store i{{[0-9]+}} 20, i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]],
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS: i{{[0-9]+}}* [[G_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: i{{[0-9]+}}* [[SIVAR_PRIVATE_ADDR]]
    // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
    // BLOCKS: call{{.*}} void {{%.+}}(i8
    ^{
      // BLOCKS: define {{.+}} void {{@.+}}(i8*
      g = 2;
      sivar = 40;
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 2, i{{[0-9]+}}*
      // BLOCKS-NOT: [[G]]{{[[^:word:]]}}
      // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
      // BLOCKS: store i{{[0-9]+}} 40, i{{[0-9]+}}*
      // BLOCKS-NOT: [[SIVAR]]{{[[^:word:]]}}
      // BLOCKS: ret
    }();
  }
  }();
  return 0;
// BLOCKS: define {{.+}} @{{.+}}([[SS_TY]]*
// BLOCKS: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// BLOCKS: store i8
// BLOCKS: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
// BLOCKS: ret

// BLOCKS: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
// BLOCKS-NOT: getelementptr {{.*}}[[SS_TY]], [[SS_TY]]* %
// BLOCKS: call{{.*}} void
// BLOCKS: ret void

// BLOCKS: define internal void @{{.+}}(i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
// BLOCKS: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// BLOCKS: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
// BLOCKS: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
// BLOCKS-NEXT: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
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
// BLOCKS-NEXT: ret void
#else
  S<float> test;
  int t_var = 0;
  int vec[] = {1, 2};
  S<float> s_arr[] = {1, 2};
  S<float> var(3);
#pragma omp parallel private(t_var, vec, s_arr, var, sivar)
  {
    vec[0] = t_var;
    s_arr[0] = var;
    sivar = 3;
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
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_FLOAT_TY]],
// CHECK: [[SIVAR_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_FLOAT_TY]]*
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: call {{.*}} [[S_FLOAT_TY_DEF_CONSTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_FLOAT_TY_DESTR]]([[S_FLOAT_TY]]*
// CHECK: ret void

// CHECK: define {{.*}} i{{[0-9]+}} [[TMAIN_INT]]()
// CHECK: [[TEST:%.+]] = alloca [[S_INT_TY]],
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR:@.+]]([[S_INT_TY]]* [[TEST]])
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 0, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*)* [[TMAIN_MICROTASK:@.+]] to void
// CHECK: call void [[S_INT_TY_DESTR:@.+]]([[S_INT_TY]]*
// CHECK: ret
//
// CHECK: define {{.+}} @{{.+}}([[SS_TY]]* %
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// CHECK: store i8
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SS_TY]]*)* [[SS_MICROTASK:@.+]] to void
// CHECK: ret

// CHECK: define internal void [[SS_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SS_TY]]* %{{.+}})
// CHECK: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[B_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: [[C_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REFA:%.+]],
// CHECK: store i{{[0-9]+}}* [[C_PRIV]], i{{[0-9]+}}** [[REFC:%.+]],
// CHECK-NEXT: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REFA]],
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
// CHECK-NEXT: ret void

// CHECK: define internal void [[TMAIN_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}})
// CHECK: [[T_VAR_PRIV:%.+]] = alloca i{{[0-9]+}}, align 128
// CHECK: [[VEC_PRIV:%.+]] = alloca [2 x i{{[0-9]+}}], align 128
// CHECK: [[S_ARR_PRIV:%.+]] = alloca [2 x [[S_INT_TY]]], align 128
// CHECK: [[VAR_PRIV:%.+]] = alloca [[S_INT_TY]], align 128
// CHECK: store i{{[0-9]+}}* [[GTID_ADDR]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]]
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK-NOT: [[SIVAR_PRIV]]
// CHECK: {{.+}}:
// CHECK: [[S_ARR_PRIV_ITEM:%.+]] = phi [[S_INT_TY]]*
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[S_ARR_PRIV_ITEM]])
// CHECK-NOT: [[T_VAR_PRIV]]
// CHECK-NOT: [[VEC_PRIV]]
// CHECK: call {{.*}} [[S_INT_TY_DEF_CONSTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]* [[VAR_PRIV]])
// CHECK-DAG: call void [[S_INT_TY_DESTR]]([[S_INT_TY]]*
// CHECK: ret void

// CHECK: define {{.+}} @{{.+}}([[SST_TY]]* %
// CHECK: store i{{[0-9]+}} 0, i{{[0-9]+}}* %
// CHECK: call void (%{{.+}}*, i{{[0-9]+}}, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{[0-9]+}} 1, void (i{{[0-9]+}}*, i{{[0-9]+}}*, ...)* bitcast (void (i{{[0-9]+}}*, i{{[0-9]+}}*, [[SST_TY]]*)* [[SST_MICROTASK:@.+]] to void
// CHECK: ret

// CHECK: define internal void [[SST_MICROTASK]](i{{[0-9]+}}* noalias [[GTID_ADDR:%.+]], i{{[0-9]+}}* noalias %{{.+}}, [[SST_TY]]* %{{.+}})
// CHECK: [[A_PRIV:%.+]] = alloca i{{[0-9]+}},
// CHECK: store i{{[0-9]+}}* [[A_PRIV]], i{{[0-9]+}}** [[REF:%.+]],
// CHECK-NEXT: [[A_PRIV:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[REF]],
// CHECK-NEXT: [[A_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A_PRIV]],
// CHECK-NEXT: [[INC:%.+]] = add nsw i{{[0-9]+}} [[A_VAL]], 1
// CHECK-NEXT: store i{{[0-9]+}} [[INC]], i{{[0-9]+}}* [[A_PRIV]],
// CHECK-NEXT: ret void

#endif

